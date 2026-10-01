"""One holdout score for the union of the measured v6 channels.

Channels: exact folded and romanized name (drop keys with more than 12
postings), that same name when the house or street agrees, consonant skeleton
plus house, and an exact website stem. Scored with the v5 matcher.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from eval_domain_spell import domain_stems, skeleton, squash_name
from f05 import macro_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v6_keys import CACHE, indic_token, name_key, romanized_name

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
EXACT_CAP = 12
GATE_CAP = 40
TIGHT_CAP = 8

EXACT_QUERY: set[str] = set()
GATES: dict[str, tuple[set[str], set[str]]] = {}
SQUASH_QUERY: set[str] = set()
STEM_QUERY: set[str] = set()
SPELL_QUERY: set[str] = set()


def _worker(args: tuple[str, int]) -> tuple[dict, dict, dict, dict]:
    path, wid = args
    exact: dict[str, list[str]] = defaultdict(list)
    gated: dict[str, list] = defaultdict(list)
    spell: dict[str, list[str]] = defaultdict(list)
    domain: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            eid, name, addr, country = line.rstrip("\n").split("\t")
            key = name_key(name, country)
            if key in EXACT_QUERY and len(exact[key]) < EXACT_CAP:
                exact[key].append(eid)
            gate = GATES.get(key)
            house, streets, _zip = parse_addr(addr)
            if gate is not None:
                houses, streets_ok = gate
                if ((house and house in houses) or (streets & streets_ok)) and len(gated[key]) < GATE_CAP:
                    gated[key].append((eid, house, tuple(streets), int(indic_token(name))))
            stems = domain_stems(name)
            if not stems and ("." in addr or "www" in addr.lower() or "http" in addr.lower()):
                stems = domain_stems(addr)
            squash = squash_name(name)
            for stem in stems:
                dkey = country + "|" + stem
                if dkey in SQUASH_QUERY and len(domain[dkey]) < TIGHT_CAP:
                    domain[dkey].append(eid)
            if len(squash) >= 8:
                dkey = country + "|" + squash
                if dkey in STEM_QUERY and eid not in domain[dkey] and len(domain[dkey]) < TIGHT_CAP:
                    domain[dkey].append(eid)
                skel = skeleton(squash)
                if len(house) >= 2 and len(skel) >= 8:
                    skey = f"{country}|{house}|{skel}"
                    if skey in SPELL_QUERY and len(spell[skey]) < TIGHT_CAP:
                        spell[skey].append(eid)
    return exact, gated, spell, domain


def _load(need: set[str]) -> dict[str, tuple[str, str, str]]:
    texts: dict[str, tuple[str, str, str]] = {}
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        if not (need - set(texts)):
            break
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, name, addr, country = line.rstrip("\n").split("\t")
                if eid in need:
                    texts[eid] = (name, addr, country)
                    if len(texts) == len(need):
                        break
    return texts


def _counts(added: dict[str, list[str]], truths: dict[str, set[str]]) -> tuple[int, int]:
    tp = fp = 0
    for sid, eids in added.items():
        truth = truths[sid]
        for eid in eids:
            if eid in truth:
                tp += 1
            else:
                fp += 1
    return tp, fp


def main() -> None:
    with (BOARD / "indic_xlit_cache.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("roman"):
                CACHE[(row["w"], row["lang"])] = row["roman"]
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    order = [row["sid"] for row in val_meta]
    texts = _load(set(order))

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
            EXACT_QUERY.add(key)
            s1_full[sid] = key
            if house or streets:
                houses, street_set = GATES.setdefault(key, (set(), set()))
                if house:
                    houses.add(house)
                street_set.update(streets)
        squash = squash_name(name)
        if len(squash) >= 8:
            dkey = country + "|" + squash
            squash_of[dkey].append(sid)
            SQUASH_QUERY.add(dkey)
            skel = skeleton(squash)
            if len(house) >= 2 and len(skel) >= 8:
                skey = f"{country}|{house}|{skel}"
                spell_of[skey].append(sid)
                SPELL_QUERY.add(skey)
        for stem in set(domain_stems(name) + domain_stems(addr)):
            dkey = country + "|" + stem
            stem_of[dkey].append(sid)
            STEM_QUERY.add(dkey)
    print(
        f"queries exact {len(EXACT_QUERY)} gated {len(GATES)} squash {len(SQUASH_QUERY)} spell {len(SPELL_QUERY)}",
        flush=True,
    )

    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    exact_hits: dict[str, list[str]] = defaultdict(list)
    gated_hits: dict[str, list] = defaultdict(list)
    spell_hits: dict[str, list[str]] = defaultdict(list)
    domain_hits: dict[str, list[str]] = defaultdict(list)
    with ctx.Pool(WORKERS) as pool:
        for part_e, part_g, part_s, part_d in pool.map(_worker, jobs):
            for key, eids in part_e.items():
                exact_hits[key].extend(eids)
            for key, rows in part_g.items():
                gated_hits[key].extend(rows)
            for key, eids in part_s.items():
                spell_hits[key].extend(eids)
            for key, eids in part_d.items():
                domain_hits[key].extend(eids)
    for bucket, cap in ((exact_hits, EXACT_CAP), (spell_hits, TIGHT_CAP), (domain_hits, TIGHT_CAP)):
        for key in list(bucket):
            if len(bucket[key]) > cap:
                del bucket[key]
    print(
        f"kept exact {len(exact_hits)} gated {len(gated_hits)} spell {len(spell_hits)} domain {len(domain_hits)}",
        flush=True,
    )

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
    for i, (sid, eid) in enumerate(zip(data["s1_id"].astype(str)[val].tolist(), data["match_id"].astype(str)[val].tolist())):
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
        "exact name": exact_added,
        "name+house|street": gate_added,
        "skeleton+house": spell_added,
        "domain stem": domain_added,
        "combined": unite(exact_added, gate_added, spell_added, domain_added),
        "combined without domain": unite(exact_added, gate_added, spell_added),
    }
    for label, added in channels.items():
        tp, fp = _counts(added, truths)
        print(f"{label} candidates tp {tp} fp {fp}", flush=True)

    extra_ids: set[str] = set()
    for added in channels.values():
        for eids in added.values():
            extra_ids.update(eids)
    missing = (pair_ids | extra_ids) - set(texts)
    if missing:
        texts.update(_load(missing))
    print(f"texts {len(texts)} missing {len(missing - set(texts))}", flush=True)

    raw_view = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    addr_view = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    indic = {eid: int(indic_token(name)) for eid, (name, _a, _c) in texts.items() if eid in extra_ids}
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v5.txt"))
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    gold = [truths[sid] for sid in order]

    def vector(sid: str, eid: str, ranks, tail, use_roman: bool) -> list[float]:
        nr, ar, ns, asc, fn, fa = ranks
        ln = raw_view.get(sid, empty_name)
        if tail is None:
            other = texts[eid]
            shown = romanized_name(other[0]) if use_roman else other[0]
            rn = name_view(shown)
            tail = features_prepared(
                prepare_record(texts[sid][0], texts[sid][1]),
                prepare_record(shown, other[1]),
                country_eq=float(texts[sid][2] == other[2]),
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )[5:]
        else:
            rn = raw_view.get(eid, empty_name)
        return (
            name_five(ln, rn)
            + list(tail)
            + extra_nine(addr_view.get(sid, empty_addr), addr_view.get(eid, empty_addr), ln, rn)
            + particle_two(ln, rn)
        )

    base_mids: dict[str, list[str]] = {}
    base_vecs: dict[str, list] = {}
    for sid in order:
        mids, vecs = [], []
        for eid, ranks, tail in groups.get(sid, []):
            mids.append(eid)
            vecs.append(vector(sid, eid, ranks, tail, False))
        base_mids[sid] = mids
        base_vecs[sid] = vecs
    print("base vectors ready", flush=True)

    def score(added: dict[str, list[str]]) -> tuple[float, float, int, int, float]:
        preds = []
        accepted_tp = accepted_fp = 0
        sizes = []
        for sid in order:
            mids = list(base_mids[sid])
            vecs = list(base_vecs[sid])
            base_n = len(mids)
            for eid in added.get(sid, []):
                mids.append(eid)
                vecs.append(vector(sid, eid, (1.0, 99.0, 1.0, 0.0, 1, 0), None, bool(indic.get(eid, 0))))
            sizes.append(len(mids))
            if not vecs:
                preds.append(set())
                continue
            scores = [float(s) for s in booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=400)]
            pred = decode_greedy_f05(mids, scores, min_gain=0.70, max_preds=12)
            preds.append(pred)
            base_pred = decode_greedy_f05(mids[:base_n], scores[:base_n], min_gain=0.70, max_preds=12)
            for eid in pred - base_pred:
                if eid in truths[sid]:
                    accepted_tp += 1
                else:
                    accepted_fp += 1
        single_gold, single_pred = [], []
        for sid, pred in zip(order, preds):
            if not truths[sid]:
                single_gold.append(set())
                single_pred.append(pred)
        single = macro_f05(single_gold, single_pred) if single_gold else 0.0
        return macro_f05(gold, preds), single, accepted_tp, accepted_fp, sum(sizes) / len(sizes)

    base_f, base_s, _tp, _fp, base_size = score({})
    print(f"baseline f05 {base_f:.4f} singleton {base_s:.4f} mean list {base_size:.1f}", flush=True)
    for label, added in channels.items():
        f05, single, tp, fp, mean_size = score(added)
        print(
            f"{label} f05 {f05:.4f} delta {f05 - base_f:+.4f} singleton {single:.4f} "
            f"accepted tp {tp} fp {fp} mean list {mean_size:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
