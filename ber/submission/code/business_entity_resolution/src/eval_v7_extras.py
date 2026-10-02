"""Holdout score of the v6 extra channels under the v7 matcher.

Base pairs keep the features v7 was trained on. Extra pairs use the ranks the
test scorer fabricates (name rank 1, name score 1, from the name channel) and
a romanized name, which is what score_v7 does. A second pass romanizes the
base pairs too, which is what the shipped test file actually scores.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

import eval_v6_union as u
from decode import decode_greedy_f05
from eval_domain_spell import domain_stems, skeleton, squash_name
from f05 import macro_f05
from normalize import tokens
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from score_v6 import load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v6_keys import name_key
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

BOARD = DATA_DIR / "scoreboard"
MODEL_V7 = BOARD / "lgbm_v7.txt"
MODEL_V7R = BOARD / "lgbm_v7r.txt"
RAW_RATIO_AT = 9


def _channels(order, truths):
    texts = u._load(set(order))
    exact_of: dict[str, list[str]] = defaultdict(list)
    squash_of: dict[str, list[str]] = defaultdict(list)
    stem_of: dict[str, list[str]] = defaultdict(list)
    spell_of: dict[str, list[str]] = defaultdict(list)
    s1_addr: dict[str, tuple[str, frozenset[str]]] = {}
    s1_full: dict[str, str] = {}
    for sid in order:
        name, addr, country = texts[sid]
        house, streets, _zip = parse_addr(addr)
        s1_addr[sid] = (house, streets)
        key = name_key(name, country)
        if key:
            exact_of[key].append(sid)
            u.EXACT_QUERY.add(key)
            s1_full[sid] = key
            if house or streets:
                houses, street_set = u.GATES.setdefault(key, (set(), set()))
                if house:
                    houses.add(house)
                street_set.update(streets)
        squash = squash_name(name)
        if len(squash) >= 8:
            dkey = country + "|" + squash
            squash_of[dkey].append(sid)
            u.SQUASH_QUERY.add(dkey)
            skel = skeleton(squash)
            if len(house) >= 2 and len(skel) >= 8:
                skey = f"{country}|{house}|{skel}"
                spell_of[skey].append(sid)
                u.SPELL_QUERY.add(skey)
        for stem in set(domain_stems(name) + domain_stems(addr)):
            dkey = country + "|" + stem
            stem_of[dkey].append(sid)
            u.STEM_QUERY.add(dkey)
    print(
        f"queries exact {len(u.EXACT_QUERY)} gated {len(u.GATES)} "
        f"squash {len(u.SQUASH_QUERY)} spell {len(u.SPELL_QUERY)}",
        flush=True,
    )
    ctx = mp.get_context("fork")
    jobs = [
        (str(u.TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(u.WORKERS)
    ]
    exact_hits: dict[str, list[str]] = defaultdict(list)
    gated_hits: dict[str, list] = defaultdict(list)
    spell_hits: dict[str, list[str]] = defaultdict(list)
    domain_hits: dict[str, list[str]] = defaultdict(list)
    with ctx.Pool(u.WORKERS) as pool:
        for part_e, part_g, part_s, part_d in pool.map(u._worker, jobs):
            for key, eids in part_e.items():
                exact_hits[key].extend(eids)
            for key, rows in part_g.items():
                gated_hits[key].extend(rows)
            for key, eids in part_s.items():
                spell_hits[key].extend(eids)
            for key, eids in part_d.items():
                domain_hits[key].extend(eids)
    for bucket, cap in (
        (exact_hits, u.EXACT_CAP),
        (spell_hits, u.TIGHT_CAP),
        (domain_hits, u.TIGHT_CAP),
    ):
        for key in list(bucket):
            if len(bucket[key]) > cap:
                del bucket[key]

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    feat = [str(x) for x in data["feature_names"]]
    col = {name: feat.index(name) for name in (
        "name_rank", "addr_rank", "name_block_score", "addr_block_score", "from_name_ch", "from_addr_ch",
    )}
    val = data["split"].astype(str) == "val"
    groups: dict[str, list] = defaultdict(list)
    have: dict[str, set[str]] = defaultdict(set)
    pair_ids: set[str] = set(order)
    xv = data["X"][val]
    for i, (sid, eid) in enumerate(zip(
        data["s1_id"].astype(str)[val].tolist(),
        data["match_id"].astype(str)[val].tolist(),
    )):
        groups[sid].append((eid, (
            float(xv[i, col["name_rank"]]), float(xv[i, col["addr_rank"]]),
            float(xv[i, col["name_block_score"]]), float(xv[i, col["addr_block_score"]]),
            int(xv[i, col["from_name_ch"]] >= 0.5), int(xv[i, col["from_addr_ch"]] >= 0.5),
        ), xv[i, 5:21].copy()))
        have[sid].add(eid)
        pair_ids.add(eid)

    def take_ids(index_of: dict[str, list[str]], hits: dict[str, list[str]], limit: int) -> dict[str, list[str]]:
        added: dict[str, list[str]] = defaultdict(list)
        for key, sids in index_of.items():
            for eid in hits.get(key, []):
                for sid in sids:
                    if eid in have.get(sid, ()) or eid in added[sid]:
                        continue
                    if len(added[sid]) >= limit:
                        continue
                    added[sid].append(eid)
        return added

    exact_added = take_ids(exact_of, exact_hits, 12)
    spell_added = take_ids(spell_of, spell_hits, 12)
    domain_added = take_ids(squash_of, domain_hits, 12)
    for sid, eids in take_ids(stem_of, domain_hits, 12).items():
        for eid in eids:
            if eid not in domain_added[sid] and len(domain_added[sid]) < 12:
                domain_added[sid].append(eid)
    gate_added: dict[str, list[str]] = defaultdict(list)
    for sid in order:
        key = s1_full.get(sid, "")
        house, streets = s1_addr[sid]
        both, only_house, only_street = [], [], []
        seen = set(have.get(sid, ()))
        for eid, row_house, row_streets, _indic in gated_hits.get(key, []):
            if eid in seen:
                continue
            street_hit = bool(set(row_streets) & streets)
            house_hit = bool(house and row_house == house)
            if house_hit and street_hit:
                bucket = both
            elif house_hit:
                bucket = only_house
            elif street_hit:
                bucket = only_street
            else:
                continue
            seen.add(eid)
            bucket.append(eid)
        gate_added[sid] = (both + only_house + only_street)[:25]

    def unite(*parts: dict[str, list[str]]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for sid in order:
            seen: set[str] = set()
            for part in parts:
                for eid in part.get(sid, []):
                    if eid not in seen:
                        seen.add(eid)
                        out[sid].append(eid)
        return out

    channels = {
        "combined": unite(exact_added, gate_added, spell_added, domain_added),
        "combined without domain": unite(exact_added, gate_added, spell_added),
    }
    for label, added in channels.items():
        tp, fp = u._counts(added, truths)
        print(f"{label} candidates tp {tp} fp {fp}", flush=True)
    extra_ids: set[str] = set()
    for added in channels.values():
        for eids in added.values():
            extra_ids.update(eids)
    missing = (pair_ids | extra_ids) - set(texts)
    if missing:
        texts.update(u._load(missing))
    return texts, groups, channels


def _metrics(order, truths, country_of, preds_of) -> str:
    gold = [truths[sid] for sid in order]
    got = [preds_of[sid] for sid in order]
    single_g, single_p = [], []
    india_g, india_p, us_g, us_p = [], [], [], []
    for sid, pred in zip(order, got):
        if not truths[sid]:
            single_g.append(set())
            single_p.append(pred)
        if country_of.get(sid) == "India":
            india_g.append(truths[sid])
            india_p.append(pred)
        elif country_of.get(sid) == "US":
            us_g.append(truths[sid])
            us_p.append(pred)
    single = macro_f05(single_g, single_p) if single_g else 0.0
    india = macro_f05(india_g, india_p) if india_g else 0.0
    us = macro_f05(us_g, us_p) if us_g else 0.0
    return f"f05 {macro_f05(gold, got):.4f} singleton {single:.4f} india {india:.4f} us {us:.4f}"


def main() -> None:
    t0 = time.time()
    load_cache()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = 1.0
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    country_of = {row["sid"]: row.get("country", "") for row in val_meta}
    order = [row["sid"] for row in val_meta]
    texts, groups, channels = _channels(order, truths)
    print(f"texts {len(texts)} in {time.time() - t0:.0f}s", flush=True)

    shown = {eid: shown_name(rec[0]) for eid, rec in texts.items()}
    raw_view = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    roman_view = {eid: name_view(shown[eid]) for eid, rec in texts.items()}
    addr_view = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    city_of = {eid: city_token(rec[1]) for eid, rec in texts.items()}
    raw_join = {eid: " ".join(tokens(rec[0])) for eid, rec in texts.items()}
    roman_join = {eid: " ".join(tokens(shown[eid])) for eid in texts}
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")

    def vector(sid: str, eid: str, ranks, tail, roman: bool) -> list[float]:
        ln = raw_view.get(sid, empty_name)
        rn = roman_view.get(eid, empty_name) if roman else raw_view.get(eid, empty_name)
        if tail is None:
            nr, ar, ns, asc, fn, fa = ranks
            other = texts[eid]
            right = shown[eid] if roman else other[0]
            tail_list = features_prepared(
                prepare_record(texts[sid][0], texts[sid][1]),
                prepare_record(right, other[1]),
                country_eq=float(texts[sid][2] == other[2]),
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )[5:]
        else:
            tail_list = list(tail)
            if roman and shown[eid] != texts[eid][0]:
                tail_list[RAW_RATIO_AT] = fuzz.ratio(raw_join[sid], roman_join[eid]) / 100.0
        return (
            name_five(ln, rn)
            + tail_list
            + extra_nine(addr_view.get(sid, empty_addr), addr_view.get(eid, empty_addr), ln, rn)
            + particle_two(ln, rn)
            + rare_two(ln[1], rn[1], idf, missing)
            + city_two(city_of.get(sid, ""), city_of.get(eid, ""))
        )

    def predict(model_path, roman_base: bool, added) -> dict[str, set[str]]:
        booster = lgb.Booster(model_file=str(model_path))
        trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
        preds = {}
        accepted_tp = accepted_fp = 0
        for sid in order:
            mids = []
            vecs = []
            base_n = 0
            for eid, ranks, tail in groups.get(sid, []):
                mids.append(eid)
                vecs.append(vector(sid, eid, ranks, tail, roman_base))
            base_n = len(mids)
            for eid in added.get(sid, []):
                mids.append(eid)
                vecs.append(vector(sid, eid, (1.0, 99.0, 1.0, 0.0, 1, 0), None, True))
            if not vecs:
                preds[sid] = set()
                continue
            scores = [float(s) for s in booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=trees)]
            pred = decode_greedy_f05(mids, scores, min_gain=0.70, max_preds=12)
            preds[sid] = pred
            if added:
                base_pred = decode_greedy_f05(mids[:base_n], scores[:base_n], min_gain=0.70, max_preds=12)
                for eid in pred - base_pred:
                    if eid in truths[sid]:
                        accepted_tp += 1
                    else:
                        accepted_fp += 1
        return preds, accepted_tp, accepted_fp

    runs = [
        ("v7 raw base", MODEL_V7, False, {}),
        ("v7 roman base", MODEL_V7, True, {}),
        ("v7 roman base + combined", MODEL_V7, True, channels["combined"]),
        ("v7 roman base + no domain", MODEL_V7, True, channels["combined without domain"]),
    ]
    for label, path, roman_base, added in runs:
        preds, tp, fp = predict(path, roman_base, added)
        extra = f" accepted tp {tp} fp {fp}" if added else ""
        print(f"{label} {_metrics(order, truths, country_of, preds)}{extra}", flush=True)
    print(f"v7 pass in {time.time() - t0:.0f}s", flush=True)

    deadline = time.time() + 900
    while not MODEL_V7R.exists() and time.time() < deadline:
        print("waiting for lgbm_v7r.txt", flush=True)
        time.sleep(20)
    if not MODEL_V7R.exists():
        print("lgbm_v7r.txt missing, skipped", flush=True)
        return
    for label, added in (
        ("v7r roman base", {}),
        ("v7r roman base + combined", channels["combined"]),
        ("v7r roman base + no domain", channels["combined without domain"]),
    ):
        preds, tp, fp = predict(MODEL_V7R, True, added)
        extra = f" accepted tp {tp} fp {fp}" if added else ""
        print(f"{label} {_metrics(order, truths, country_of, preds)}{extra}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
