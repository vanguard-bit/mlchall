"""Holdout measurement: accent-fold alone, and the v6 name key gated by house or street.

Accent-fold candidates are only pairs whose plain name phrases differ and whose
folded phrases match. The gated channel keeps a folded, romanized name when the
house number matches or a street token overlaps. Both are unioned with the
current validation candidates and scored by the v5 matcher.
"""

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
from v4_features import (
    content_name_tokens_v4,
    extra_nine,
    name_five,
    name_view,
    parse_addr,
    particle_two,
)
from v6_keys import CACHE, fold_text, indic_token, romanized_name

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
ACCENT_CAP = 24
GATE_CAP = 40

FOLD_KEYS: set[str] = set()
ACCENTED_S1_KEYS: set[str] = set()
GATES: dict[str, tuple[set[str], set[str]]] = {}


def _phrase(name: str) -> str:
    return " ".join(content_name_tokens_v4(name))


def _phrases(name: str) -> tuple[str, str, str]:
    """Plain phrase, folded phrase, and folded phrase after IndicXlit."""
    if not any(ord(ch) > 127 for ch in name):
        phrase = _phrase(name)
        return phrase, phrase, phrase
    plain = _phrase(name)
    if indic_token(name):
        full = _phrase(fold_text(romanized_name(name)))
        folded = _phrase(fold_text(name))
    else:
        folded = _phrase(fold_text(name))
        full = folded
    return plain, folded, full


def _key(country: str, phrase: str) -> str:
    if len(phrase) < 8:
        return ""
    return country + "|" + phrase


def _worker(args: tuple[str, int]) -> tuple[dict, dict, dict, int]:
    path, wid = args
    accented: dict[str, list] = defaultdict(list)
    bare: dict[str, list] = defaultdict(list)
    gated: dict[str, list] = defaultdict(list)
    gate_over = 0
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            eid, name, addr, country = line.rstrip("\n").split("\t")
            plain, folded, full = _phrases(name)
            fold_key = _key(country, folded)
            if fold_key in FOLD_KEYS and plain != folded and len(accented[fold_key]) < ACCENT_CAP:
                house, streets, _zip = parse_addr(addr)
                accented[fold_key].append((eid, plain, house, tuple(streets)))
            elif fold_key in ACCENTED_S1_KEYS and plain == folded and len(bare[fold_key]) < ACCENT_CAP:
                house, streets, _zip = parse_addr(addr)
                bare[fold_key].append((eid, plain, house, tuple(streets)))
            full_key = _key(country, full)
            gate = GATES.get(full_key)
            if gate is None:
                continue
            houses, streets_ok = gate
            house, streets, _zip = parse_addr(addr)
            if not ((house and house in houses) or (streets & streets_ok)):
                continue
            bucket = gated[full_key]
            if len(bucket) < GATE_CAP:
                bucket.append((eid, house, tuple(streets), int(indic_token(name))))
            else:
                gate_over += 1
    return accented, bare, gated, gate_over


def _load_texts(need: set[str]) -> dict[str, tuple[str, str, str]]:
    texts: dict[str, tuple[str, str, str]] = {}
    files = ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")
    for filename in files:
        if len(texts) == len(need) and filename != "train_source1.tsv":
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


def main() -> None:
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
    texts = _load_texts(set(order))

    s1_fold: dict[str, str] = {}
    s1_full: dict[str, str] = {}
    s1_plain: dict[str, str] = {}
    s1_addr: dict[str, tuple[str, frozenset[str]]] = {}
    accented_s1 = 0
    for sid in order:
        name, addr, country = texts[sid]
        plain, folded, full = _phrases(name)
        house, streets, _zip = parse_addr(addr)
        s1_plain[sid] = plain
        s1_addr[sid] = (house, streets)
        fold_key = _key(country, folded)
        full_key = _key(country, full)
        if fold_key:
            s1_fold[sid] = fold_key
            FOLD_KEYS.add(fold_key)
            if plain != folded:
                accented_s1 += 1
                ACCENTED_S1_KEYS.add(fold_key)
        if full_key and (house or streets):
            s1_full[sid] = full_key
            houses, street_set = GATES.setdefault(full_key, (set(), set()))
            if house:
                houses.add(house)
            street_set.update(streets)
    print(
        f"val s1 {len(order)} accented s1 {accented_s1} fold keys {len(FOLD_KEYS)} gated keys {len(GATES)}",
        flush=True,
    )

    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    accented: dict[str, list] = defaultdict(list)
    bare: dict[str, list] = defaultdict(list)
    gated: dict[str, list] = defaultdict(list)
    gate_over = 0
    with ctx.Pool(WORKERS) as pool:
        for part_a, part_b, part_g, over in pool.map(_worker, jobs):
            for key, rows in part_a.items():
                accented[key].extend(rows)
            for key, rows in part_b.items():
                bare[key].extend(rows)
            for key, rows in part_g.items():
                gated[key].extend(rows)
            gate_over += over
    print(
        f"accented-row keys {len(accented)} bare keys {len(bare)} gated keys {len(gated)} gate overflows {gate_over}",
        flush=True,
    )

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    feat = [str(x) for x in data["feature_names"]]
    col = {name: feat.index(name) for name in (
        "name_rank", "addr_rank", "name_block_score", "addr_block_score", "from_name_ch", "from_addr_ch",
    )}
    val = data["split"].astype(str) == "val"
    groups: dict[str, list] = defaultdict(list)
    pair_ids: set[str] = set(order)
    xv = data["X"][val]
    s1_ids = data["s1_id"].astype(str)[val].tolist()
    match_ids = data["match_id"].astype(str)[val].tolist()
    for i, (sid, eid) in enumerate(zip(s1_ids, match_ids)):
        groups[sid].append((eid, (
            float(xv[i, col["name_rank"]]), float(xv[i, col["addr_rank"]]),
            float(xv[i, col["name_block_score"]]), float(xv[i, col["addr_block_score"]]),
            int(xv[i, col["from_name_ch"]] >= 0.5), int(xv[i, col["from_addr_ch"]] >= 0.5),
        ), xv[i, 5:21].copy()))
        pair_ids.add(eid)
    have = {sid: {eid for eid, _ranks, _tail in rows} for sid, rows in groups.items()}

    accent_lists: dict[str, list[str]] = {}
    accent_addr_lists: dict[str, list[str]] = {}
    accent_tp = accent_fp = accent_addr_tp = accent_addr_fp = 0
    for sid in order:
        key = s1_fold.get(sid, "")
        house, streets = s1_addr[sid]
        plain, folded, _full = _phrases(texts[sid][0])
        rows = list(accented.get(key, []))
        if plain != folded:
            rows.extend(bare.get(key, []))
        chosen: list[str] = []
        chosen_addr: list[str] = []
        seen = set(have.get(sid, ()))
        for eid, row_plain, row_house, row_streets in rows:
            if eid in seen or row_plain == plain:
                continue
            seen.add(eid)
            if len(chosen) < 12:
                chosen.append(eid)
            street_hit = bool(set(row_streets) & streets)
            house_hit = bool(house and row_house == house)
            if (house_hit or street_hit) and eid not in chosen_addr and len(chosen_addr) < 25:
                chosen_addr.append(eid)
        accent_lists[sid] = chosen
        accent_addr_lists[sid] = chosen_addr
        truth = truths[sid]
        accent_tp += sum(eid in truth for eid in chosen)
        accent_fp += sum(eid not in truth for eid in chosen)
        accent_addr_tp += sum(eid in truth for eid in chosen_addr)
        accent_addr_fp += sum(eid not in truth for eid in chosen_addr)

    gate_lists: dict[str, list[tuple[str, int]]] = {}
    house_lists: dict[str, list[tuple[str, int]]] = {}
    gate_tp = gate_fp = house_tp = house_fp = 0
    agree_both = agree_house = agree_street = 0
    for sid in order:
        key = s1_full.get(sid, "")
        house, streets = s1_addr[sid]
        both: list[tuple[str, int]] = []
        only_house: list[tuple[str, int]] = []
        only_street: list[tuple[str, int]] = []
        seen = set(have.get(sid, ()))
        for eid, row_house, row_streets, is_indic in gated.get(key, []):
            if eid in seen:
                continue
            street_hit = bool(set(row_streets) & streets)
            house_hit = bool(house and row_house == house)
            if house_hit and street_hit:
                agree_both += 1
                bucket = both
            elif house_hit:
                agree_house += 1
                bucket = only_house
            elif street_hit:
                agree_street += 1
                bucket = only_street
            else:
                continue
            seen.add(eid)
            bucket.append((eid, is_indic))
        picked = (both + only_house + only_street)[:25]
        house_picked = (both + only_house)[:25]
        gate_lists[sid] = picked
        house_lists[sid] = house_picked
        truth = truths[sid]
        gate_tp += sum(eid in truth for eid, _flag in picked)
        gate_fp += sum(eid not in truth for eid, _flag in picked)
        house_tp += sum(eid in truth for eid, _flag in house_picked)
        house_fp += sum(eid not in truth for eid, _flag in house_picked)
    print(
        f"accent cap12 tp {accent_tp} fp {accent_fp} | accent+addr tp {accent_addr_tp} fp {accent_addr_fp}",
        flush=True,
    )
    print(
        f"gate candidates tp {gate_tp} fp {gate_fp} house-only-channel tp {house_tp} fp {house_fp} "
        f"agree both {agree_both} house {agree_house} street {agree_street}",
        flush=True,
    )

    extra_ids: set[str] = set()
    for sid in order:
        extra_ids.update(accent_lists[sid])
        extra_ids.update(accent_addr_lists[sid])
        extra_ids.update(eid for eid, _flag in gate_lists[sid])
    missing = (pair_ids | extra_ids) - set(texts)
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        if not missing:
            break
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, name, addr, country = line.rstrip("\n").split("\t")
                if eid in missing:
                    texts[eid] = (name, addr, country)
                    missing.discard(eid)
                    if not missing:
                        break
    print(f"texts {len(texts)} still missing {len(missing)}", flush=True)

    raw_view = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    addr_view = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    indic_flag = {eid: int(indic_token(name)) for eid, (name, _a, _c) in ((e, texts[e]) for e in extra_ids)}
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v5.txt"))
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    gold = [truths[sid] for sid in order]
    base_items = {sid: list(groups.get(sid, [])) for sid in order}

    def vector(sid: str, eid: str, ranks, tail, use_roman: bool) -> list[float]:
        nr, ar, ns, asc, fn, fa = ranks
        ln = raw_view.get(sid, empty_name)
        if tail is None:
            other = texts.get(eid, ("", "", ""))
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
        mids = []
        vecs = []
        for eid, ranks, tail in base_items[sid]:
            mids.append(eid)
            vecs.append(vector(sid, eid, ranks, tail, False))
        base_mids[sid] = mids
        base_vecs[sid] = vecs
    print("base vectors ready", flush=True)

    def score(added: dict[str, list]) -> tuple[float, float, int, int, float]:
        preds = []
        accepted_tp = accepted_fp = 0
        sizes = []
        for sid in order:
            items = base_items[sid]
            mids = list(base_mids[sid])
            vecs = list(base_vecs[sid])
            for eid in added.get(sid, []):
                use_roman = bool(indic_flag.get(eid, 0))
                mids.append(eid)
                vecs.append(vector(sid, eid, (1.0, 99.0, 1.0, 0.0, 1, 0), None, use_roman))
            sizes.append(len(mids))
            if not vecs:
                preds.append(set())
                continue
            scores = booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=400)
            pred = decode_greedy_f05(mids, [float(s) for s in scores], min_gain=0.70, max_preds=12)
            preds.append(pred)
            base_n = len(items)
            base_pred = decode_greedy_f05(
                mids[:base_n], [float(s) for s in scores[:base_n]], min_gain=0.70, max_preds=12,
            )
            truth = truths[sid]
            for eid in pred - base_pred:
                if eid in truth:
                    accepted_tp += 1
                else:
                    accepted_fp += 1
        f05 = macro_f05(gold, preds)
        single_gold = []
        single_pred = []
        for sid, pred in zip(order, preds):
            if not truths[sid]:
                single_gold.append(set())
                single_pred.append(pred)
        single = macro_f05(single_gold, single_pred) if single_gold else 0.0
        mean_size = sum(sizes) / len(sizes)
        return f05, single, accepted_tp, accepted_fp, mean_size

    base_f, base_s, _tp, _fp, base_size = score({})
    print(f"baseline f05 {base_f:.4f} singleton {base_s:.4f} mean list {base_size:.1f}", flush=True)
    variants = (
        ("accent cap12", {sid: accent_lists[sid] for sid in order}),
        ("accent+addr", {sid: accent_addr_lists[sid] for sid in order}),
        ("name+house|street", {sid: [eid for eid, _f in gate_lists[sid]] for sid in order}),
        ("name+house", {sid: [eid for eid, _f in house_lists[sid]] for sid in order}),
    )
    for label, added in variants:
        f05, single, tp, fp, mean_size = score(added)
        print(
            f"{label} f05 {f05:.4f} delta {f05 - base_f:+.4f} singleton {single:.4f} "
            f"accepted tp {tp} fp {fp} mean list {mean_size:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
