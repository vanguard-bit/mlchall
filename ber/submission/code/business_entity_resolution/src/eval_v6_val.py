"""Validation F0.5 for v6: exact accent-folded and IndicXlit name keys unioned with v5 candidates."""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from f05 import macro_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v6_keys import CACHE, name_key

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
CAP = 12
QUERY: set[str] = set()


def _worker(args: tuple[str, int]) -> dict[str, list[str]]:
    path, wid = args
    found: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            eid, name, _addr, country = line.rstrip("\n").split("\t")
            key = name_key(name, country)
            if key not in QUERY:
                continue
            bucket = found[key]
            if len(bucket) < CAP:
                bucket.append(eid)
    return found


def indic(text: str) -> bool:
    return any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF for ch in text)


def main() -> None:
    global QUERY
    with (BOARD / "indic_xlit_cache.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("roman"):
                CACHE[(row["w"], row["lang"])] = row["roman"]
    print(f"xlit cache {len(CACHE)}", flush=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    order = [row["sid"] for row in val_meta]
    need = set(order)
    texts: dict[str, tuple[str, str, str]] = {}
    for filename in ("train_source1.tsv",):
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, name, addr, country = line.rstrip("\n").split("\t")
                if eid in need:
                    texts[eid] = (name, addr, country)
                    if len(texts) == len(need):
                        break
    s1_key = {}
    for sid in order:
        name, _addr, country = texts[sid]
        key = name_key(name, country)
        if key:
            s1_key[sid] = key
            QUERY.add(key)
    print(f"query keys {len(QUERY)}", flush=True)
    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    hits: dict[str, list[str]] = defaultdict(list)
    with ctx.Pool(WORKERS) as pool:
        for part in pool.map(_worker, jobs):
            for key, eids in part.items():
                hits[key].extend(eids)
    dropped = 0
    for key in list(hits):
        if len(hits[key]) > CAP:
            dropped += 1
            del hits[key]
    print(f"keys hit {len(hits)} dropped common {dropped}", flush=True)

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    feat = [str(x) for x in data["feature_names"]]
    col = {name: feat.index(name) for name in (
        "name_rank", "addr_rank", "name_block_score", "addr_block_score", "from_name_ch", "from_addr_ch"
    )}
    val = data["split"].astype(str) == "val"
    groups: dict[str, list] = defaultdict(list)
    pair_ids: set[str] = set(order)
    xv = data["X"][val]
    for i, (sid, eid) in enumerate(zip(data["s1_id"].astype(str)[val].tolist(), data["match_id"].astype(str)[val].tolist())):
        groups[sid].append((eid, (
            float(xv[i, col["name_rank"]]), float(xv[i, col["addr_rank"]]),
            float(xv[i, col["name_block_score"]]), float(xv[i, col["addr_block_score"]]),
            int(xv[i, col["from_name_ch"]] >= 0.5), int(xv[i, col["from_addr_ch"]] >= 0.5),
        ), xv[i, 5:21].copy()))
        pair_ids.add(eid)
    extra_tp = extra_fp = 0
    for sid in order:
        key = s1_key.get(sid, "")
        have = {eid for eid, _r, _c in groups.get(sid, [])}
        truth = truths[sid]
        for eid in hits.get(key, []):
            if eid in have:
                continue
            groups[sid].append((eid, (1.0, 99.0, 1.0, 0.0, 1, 0), None))
            pair_ids.add(eid)
            if eid in truth:
                extra_tp += 1
            else:
                extra_fp += 1
    print(f"new candidates tp {extra_tp} fp {extra_fp}", flush=True)
    more = pair_ids - set(texts)
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        if not more:
            break
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, name, addr, country = line.rstrip("\n").split("\t")
                if eid in more:
                    texts[eid] = (name, addr, country)
                    more.discard(eid)
                    if not more:
                        break
    from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
    from pair_features import prepare_record
    from v6_keys import romanized_name

    raw_view = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    addr = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v5.txt"))
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    preds, gold = [], []
    for sid in order:
        gold.append(truths[sid])
        s1 = texts[sid]
        items = groups.get(sid, [])
        mids, vecs = [], []
        ln = raw_view.get(sid, empty_name)
        for eid, ranks, tail in items:
            nr, ar, ns, asc, fn, fa = ranks
            if tail is None:
                other = texts.get(eid, ("", "", ""))
                roman = romanized_name(other[0])
                rn = name_view(roman)
                tail = features_prepared(
                    prepare_record(s1[0], s1[1]),
                    prepare_record(roman, other[1]),
                    country_eq=float(s1[2] == other[2]),
                    name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                    from_name_channel=fn, from_addr_channel=fa,
                )[5:]
            else:
                rn = raw_view.get(eid, empty_name)
            vecs.append(
                name_five(ln, rn)
                + list(tail)
                + extra_nine(addr.get(sid, empty_addr), addr.get(eid, empty_addr), ln, rn)
                + particle_two(ln, rn)
            )
            mids.append(eid)
        if not vecs:
            preds.append(set())
            continue
        scores = booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=400)
        preds.append(decode_greedy_f05(mids, [float(s) for s in scores], min_gain=0.70, max_preds=12))
    print(f"v6 val f05 {macro_f05(gold, preds):.4f} v5 baseline 0.9034 extra tp {extra_tp} fp {extra_fp}", flush=True)


def indic_name(name: str) -> bool:
    return any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF for ch in name)


if __name__ == "__main__":
    main()
