"""Rare-token candidates that also agree on house, street, city, or postal.

Scored with the v7 matcher and unioned with the current holdout lists.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import macro_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from score_v3 import _load_addrs
from v4_features import content_name_tokens_v4, extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
MODEL = BOARD / "lgbm_v7.txt"
WORKERS = 8
IDF_MIN = 0.70
CAP = 8

RARE: set[str] = set()
GATES: dict[str, tuple[set[str], set[str], set[str], set[str]]] = {}


def _worker(args: tuple[str, int]) -> dict[str, list]:
    path, wid = args
    found: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            eid, name, addr, country = line.rstrip("\n").split("\t")
            house, streets, zip5 = parse_addr(addr)
            city = city_token(addr)
            for tok in content_name_tokens_v4(name):
                if len(tok) < 5:
                    continue
                key = country + "|" + tok
                gate = GATES.get(key)
                if gate is None:
                    continue
                houses, sts, cities, zips = gate
                agrees = (
                    (house and house in houses)
                    or bool(streets & sts)
                    or (city and city in cities)
                    or (zip5 and zip5 in zips)
                )
                if not agrees:
                    continue
                bucket = found[key]
                if len(bucket) < CAP:
                    bucket.append((eid, house, tuple(streets), city, zip5))
    return found


def _metrics(truths: dict[str, set[str]], preds: dict[str, set[str]]) -> tuple[float, float]:
    order = list(truths)
    gold = [truths[sid] for sid in order]
    got = [preds.get(sid, set()) for sid in order]
    single_g, single_p = [], []
    for sid, truth in truths.items():
        if not truth:
            single_g.append(set())
            single_p.append(preds.get(sid, set()))
    single = macro_f05(single_g, single_p) if single_g else 0.0
    return macro_f05(gold, got), single


def main() -> None:
    import multiprocessing as mp

    t0 = time.time()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = 1.0
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    order = [row["sid"] for row in val_meta]
    texts: dict[str, tuple[str, str, str]] = {}
    need = set(order)
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            if eid in need:
                texts[eid] = (name, addr, country)
                if len(texts) == len(need):
                    break
    owners: dict[str, list[str]] = defaultdict(list)
    s1_addr: dict[str, tuple[str, frozenset[str], str, str]] = {}
    for sid in order:
        name, addr, country = texts[sid]
        house, streets, zip5 = parse_addr(addr)
        city = city_token(addr)
        s1_addr[sid] = (house, streets, zip5, city)
        for tok in content_name_tokens_v4(name):
            if len(tok) < 5 or idf.get(tok, 0.0) < IDF_MIN:
                continue
            key = country + "|" + tok
            owners[key].append(sid)
            RARE.add(key)
            houses, sts, cities, zips = GATES.setdefault(key, (set(), set(), set(), set()))
            if house:
                houses.add(house)
            sts.update(streets)
            if city:
                cities.add(city)
            if zip5:
                zips.add(zip5)
    print(f"rare keys {len(RARE)} in {time.time() - t0:.0f}s", flush=True)
    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    hits: dict[str, list] = defaultdict(list)
    with ctx.Pool(WORKERS) as pool:
        for part in pool.map(_worker, jobs):
            for key, rows in part.items():
                hits[key].extend(rows)
    for key in list(hits):
        if len(hits[key]) > CAP:
            del hits[key]
    print(f"keys kept {len(hits)}", flush=True)

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    s1_ids = data["s1_id"].astype(str)
    mids = data["match_id"].astype(str)
    val = data["split"].astype(str) == "val"
    s1_ids, mids, X = s1_ids[val], mids[val], X[val]
    have: dict[str, set[str]] = defaultdict(set)
    base_rows: dict[str, list] = defaultdict(list)
    for sid, eid, row in zip(s1_ids, mids, X):
        have[sid].add(eid)
        base_rows[sid].append((eid, row))
    added: dict[str, list[str]] = defaultdict(list)
    tp = fp = 0
    for sid in order:
        house, streets, zip5, city = s1_addr[sid]
        seen = set(have.get(sid, ()))
        got: list[str] = []
        for tok in content_name_tokens_v4(texts[sid][0]):
            key = texts[sid][2] + "|" + tok
            for eid, h, st, c, z in hits.get(key, []):
                if eid in seen or eid in got:
                    continue
                agrees = (
                    house_eq(house, h)
                    or bool(streets and set(st) & streets)
                    or (city and c == city)
                    or (zip5 and z == zip5)
                )
                if not agrees:
                    continue
                got.append(eid)
                if len(got) >= 8:
                    break
            if len(got) >= 8:
                break
        added[sid] = got
        truth = truths[sid]
        for eid in got:
            if eid in truth:
                tp += 1
            else:
                fp += 1
    print(f"new candidates tp {tp} fp {fp}", flush=True)
    if tp < 30:
        print("too few new true links to clear 0.012; skip rescore", flush=True)
        return

    pair_ids = set(order) | set(s1_ids) | set(mids)
    for eids in added.values():
        pair_ids.update(eids)
    more = pair_ids - set(texts)
    addrs = _load_addrs(pair_ids)
    names: dict[str, str] = {eid: rec[0] for eid, rec in texts.items()}
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        if not more:
            break
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, name, _addr, _country = line.rstrip("\n").split("\t")
                if eid in more:
                    names[eid] = name
                    more.discard(eid)
                    if not more:
                        break
    name_of = {eid: name_view(names.get(eid, "")) for eid in pair_ids}
    addr_of = {eid: parse_addr(addrs.get(eid, "")) for eid in pair_ids}
    city_of = {eid: city_token(addrs.get(eid, "")) for eid in pair_ids}
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()

    def vector(sid: str, eid: str, tail, ranks) -> list[float]:
        left_n = name_of.get(sid, empty_name)
        right_n = name_of.get(eid, empty_name)
        if tail is None:
            nr, ar, ns, asc, fn, fa = ranks
            tail = features_prepared(
                prepare_record(names.get(sid, ""), addrs.get(sid, "")),
                prepare_record(names.get(eid, ""), addrs.get(eid, "")),
                country_eq=1.0,
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )[5:]
        return (
            name_five(left_n, right_n)
            + list(tail)
            + extra_nine(addr_of.get(sid, empty_addr), addr_of.get(eid, empty_addr), left_n, right_n)
            + particle_two(left_n, right_n)
            + rare_two(left_n[1], right_n[1], idf, missing)
            + city_two(city_of.get(sid, ""), city_of.get(eid, ""))
        )

    preds: dict[str, set[str]] = {}
    for sid in order:
        items = []
        for eid, row in base_rows.get(sid, []):
            items.append((eid, vector(sid, eid, row[5:21], None)))
        for eid in added.get(sid, []):
            items.append((eid, vector(sid, eid, None, (1.0, 99.0, 1.0, 0.0, 1, 0))))
        if not items:
            preds[sid] = set()
            continue
        mat = np.asarray([vec for _eid, vec in items], dtype=np.float32)
        sc = booster.predict(mat, num_iteration=trees)
        order_i = np.argsort(-sc)
        chosen: set[str] = set()
        for idx in order_i:
            if float(sc[idx]) < 0.70 or len(chosen) >= 12:
                break
            chosen.add(items[idx][0])
        preds[sid] = chosen
    f05, single = _metrics(truths, preds)
    print(
        f"rare+address f05 {f05:.4f} delta {f05 - 0.9126:+.4f} singleton {single:.4f} "
        f"tp {tp} fp {fp} in {time.time() - t0:.0f}s",
        flush=True,
    )


def house_eq(left: str, right: str) -> bool:
    return bool(left and right and left == right)


if __name__ == "__main__":
    main()
