"""Expand from matcher-accepted seeds to other noisy rows at the same house and street.

Measures the validation fold only. Does not write a submission.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import macro_f05
from normalize import address_tokens, digit_tokens
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
MALL_CAP = 25
PER_S1 = 12
MIN_GAIN = 0.70

WANTED_IDS: set[str] = set()
WANTED_KEYS: set[str] = set()


def house_street_keys(addr: str) -> set[str]:
    """House number plus a street token. The last token is treated as the city."""
    streets = [t for t in address_tokens(addr) if len(t) >= 4]
    houses: list[str] = []
    for raw in digit_tokens(addr):
        if not 1 <= len(raw) <= 6:
            continue
        house = raw.lstrip("0") or "0"
        if house not in houses:
            houses.append(house)
    if not houses or not streets:
        return set()
    body = streets[:-1] if len(streets) >= 2 else streets
    keys: set[str] = set()
    for house in houses[:2]:
        for street in body[:4]:
            keys.add(f"{house}:{street}")
    return keys


def decode_ids(mids: list[str], scores: list[float], min_gain: float = MIN_GAIN) -> list[str]:
    order = sorted(range(len(mids)), key=lambda i: scores[i], reverse=True)
    chosen: list[str] = []
    for i in order:
        if scores[i] < min_gain:
            break
        chosen.append(mids[i])
        if len(chosen) >= 12:
            break
    return chosen


def _load_worker(args: tuple[str, int]) -> dict[str, tuple[str, str, str]]:
    path, wid = args
    found: dict[str, tuple[str, str, str]] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4 or parts[0] not in WANTED_IDS:
                continue
            found[parts[0]] = (parts[1], parts[2], parts[3])
    return found


def _expand_worker(args: tuple[str, int]) -> tuple[dict[str, int], dict[str, list[tuple[str, str, str]]]]:
    path, wid = args
    counts: dict[str, int] = defaultdict(int)
    kept: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            eid, name, addr, country = parts
            prefix = country + "|"
            for key in house_street_keys(addr):
                full = prefix + key
                if full not in WANTED_KEYS:
                    continue
                counts[full] += 1
                bucket = kept[full]
                if len(bucket) < MALL_CAP:
                    bucket.append((eid, name, addr))
            if i and i % 1_000_000 == 0 and wid == 0:
                print(f"  expand {path.rsplit('/', 1)[-1]} {i}", flush=True)
    return dict(counts), {key: rows for key, rows in kept.items()}


def load_ids(ids: set[str]) -> dict[str, tuple[str, str, str]]:
    global WANTED_IDS
    WANTED_IDS = ids
    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    found: dict[str, tuple[str, str, str]] = {}
    with ctx.Pool(WORKERS) as pool:
        for part in pool.map(_load_worker, jobs):
            found.update(part)
    return found


def baseline_groups() -> tuple[dict[str, dict], dict[str, set[str]], dict[str, set[str]]]:
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v2.txt"))
    val = data["split"].astype(str) == "val"
    scores = booster.predict(data["X"][val])
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    groups: dict[str, dict] = {}
    forward: dict[str, set[str]] = {}
    for row, score in zip(np.where(val)[0], scores):
        sid = str(s1[row])
        slot = groups.get(sid)
        if slot is None:
            slot = {"mids": [], "scores": []}
            groups[sid] = slot
            forward[sid] = set()
        eid = str(mid[row])
        slot["mids"].append(eid)
        slot["scores"].append(float(score))
        forward[sid].add(eid)
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    del data
    gc.collect()
    return groups, forward, truths


def score_added(
    groups: dict[str, dict],
    truths: dict[str, set[str]],
    added: dict[str, list[str]],
    texts: dict[str, tuple[str, str, str]],
    *,
    address_channel: bool,
) -> tuple[float, float, int, int]:
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v2.txt"))
    prepared = {eid: prepare_record(name, addr) for eid, (name, addr, _c) in texts.items()}
    countries = {eid: country for eid, (_n, _a, country) in texts.items()}
    owners: list[str] = []
    mids: list[str] = []
    labels: list[int] = []
    feats: list[list[float]] = []
    for sid, extra in added.items():
        left = prepared.get(sid)
        if left is None:
            continue
        truth = truths[sid]
        for eid in extra:
            right = prepared.get(eid)
            if right is None:
                continue
            if address_channel:
                row = features_prepared(
                    left,
                    right,
                    country_eq=float(countries.get(sid) == countries.get(eid)),
                    name_rank=99.0,
                    addr_rank=30.0,
                    addr_score=1.0,
                    from_addr_channel=1,
                )
            else:
                row = features_prepared(
                    left,
                    right,
                    country_eq=float(countries.get(sid) == countries.get(eid)),
                )
            feats.append(row)
            owners.append(sid)
            mids.append(eid)
            labels.append(int(eid in truth))
    pred_scores = (
        booster.predict(np.asarray(feats, dtype=np.float32)) if feats else np.zeros(0)
    )
    by_sid: dict[str, list[tuple[str, float]]] = defaultdict(list)
    added_tp = added_fp = 0
    for sid, eid, label, score in zip(owners, mids, labels, pred_scores):
        if float(score) < MIN_GAIN:
            continue
        by_sid[sid].append((eid, float(score)))
        if label:
            added_tp += 1
        else:
            added_fp += 1
    order = list(truths)
    preds: list[set[str]] = []
    singleton_preds: list[set[str]] = []
    singleton_gold: list[set[str]] = []
    for sid in order:
        slot = groups.get(sid)
        extra = by_sid.get(sid, [])
        if slot is None:
            chosen = decode_ids([e for e, _s in extra], [s for _e, s in extra])
        else:
            chosen = decode_ids(
                slot["mids"] + [e for e, _s in extra],
                slot["scores"] + [s for _e, s in extra],
            )
        pred = set(chosen)
        preds.append(pred)
        if not truths[sid]:
            singleton_preds.append(pred)
            singleton_gold.append(set())
    score = macro_f05([truths[sid] for sid in order], preds)
    single = macro_f05(singleton_gold, singleton_preds) if singleton_gold else 0.0
    return score, single, added_tp, added_fp


def main() -> None:
    t0 = time.time()
    groups, forward, truths = baseline_groups()
    base_pred = {
        sid: set(decode_ids(groups[sid]["mids"], groups[sid]["scores"])) if sid in groups else set()
        for sid in truths
    }
    order = list(truths)
    base = macro_f05([truths[sid] for sid in order], [base_pred[sid] for sid in order])
    single_ids = [sid for sid in order if not truths[sid]]
    base_single = macro_f05(
        [set() for _ in single_ids],
        [base_pred[sid] for sid in single_ids],
    )
    print(f"baseline f05 {base:.4f} singleton {base_single:.4f} n {len(truths)}", flush=True)

    seed_of: dict[str, list[tuple[int, str]]] = {}
    need = set(truths)
    for sid in order:
        pred = list(base_pred[sid])
        if sid not in groups or not pred:
            seed_of[sid] = []
            continue
        rank = {eid: i for i, eid in enumerate(decode_ids(groups[sid]["mids"], groups[sid]["scores"]))}
        ranked = sorted(pred, key=lambda eid: rank.get(eid, 99))
        seed_of[sid] = [(i, eid) for i, eid in enumerate(ranked)]
        need.update(eid for _i, eid in seed_of[sid])
    print(f"loading texts for {len(need)} ids", flush=True)
    texts = load_ids(need)
    print(f"loaded {len(texts)} texts in {time.time() - t0:.0f}s", flush=True)

    s1_keys: dict[str, dict[str, int]] = {}
    global WANTED_KEYS
    wanted: set[str] = set()
    for sid, seeds in seed_of.items():
        key_rank: dict[str, int] = {}
        for rank, eid in seeds:
            rec = texts.get(eid)
            if rec is None:
                continue
            country = rec[2]
            for key in house_street_keys(rec[1]):
                full = country + "|" + key
                prev = key_rank.get(full)
                if prev is None or rank < prev:
                    key_rank[full] = rank
                wanted.add(full)
        s1_keys[sid] = key_rank
    WANTED_KEYS = wanted
    print(f"seed address keys {len(wanted)}", flush=True)
    gc.collect()

    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    counts: dict[str, int] = defaultdict(int)
    rows_of: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with ctx.Pool(WORKERS) as pool:
        parts = pool.map(_expand_worker, jobs)
    for part_counts, part_rows in parts:
        for key, count in part_counts.items():
            counts[key] += count
        for key, rows in part_rows.items():
            rows_of[key].extend(rows)
    poisoned = sum(1 for key in wanted if counts.get(key, 0) > MALL_CAP)
    print(f"poisoned keys {poisoned} of {len(wanted)}", flush=True)

    def collect(max_seed_rank: int) -> dict[str, list[str]]:
        added: dict[str, list[str]] = {}
        for sid, key_rank in s1_keys.items():
            best: dict[str, int] = {}
            info: dict[str, tuple[str, str, str]] = {}
            fwd = forward.get(sid, set())
            seeds = {eid for _r, eid in seed_of[sid]}
            for key, rank in key_rank.items():
                if rank > max_seed_rank or counts.get(key, 0) > MALL_CAP:
                    continue
                rarity = counts[key]
                for eid, name, addr in rows_of.get(key, []):
                    if eid in fwd or eid in seeds:
                        continue
                    prev = best.get(eid)
                    if prev is not None and prev <= rarity:
                        continue
                    best[eid] = rarity
                    info[eid] = (name, addr, key.split("|", 1)[0])
            chosen = sorted(best, key=lambda eid: best[eid])[:PER_S1]
            if chosen:
                added[sid] = chosen
                for eid in chosen:
                    texts.setdefault(eid, info[eid])
        return added

    for label, max_rank in (("top-seed", 0), ("all-accepted", 11)):
        added = collect(max_rank)
        raw_tp = raw_fp = 0
        shown_tp = shown_fp = 0
        for sid, extra in added.items():
            truth = truths[sid]
            for eid in extra:
                if eid in truth:
                    raw_tp += 1
                    if shown_tp < 8 and label == "all-accepted":
                        ln, la, _lc = texts[sid]
                        rn, ra, _rc = texts[eid]
                        print(f"NEW TP {ln} || {la} <> {rn} || {ra}", flush=True)
                        shown_tp += 1
                else:
                    raw_fp += 1
                    if shown_fp < 6 and label == "all-accepted":
                        ln, la, _lc = texts[sid]
                        rn, ra, _rc = texts[eid]
                        print(f"NEW FP {ln} || {la} <> {rn} || {ra}", flush=True)
                        shown_fp += 1
        print(f"{label} raw new tp {raw_tp} fp {raw_fp}", flush=True)
        for channel, flag in (("hidden", False), ("addr-channel", True)):
            score, single, tp, fp = score_added(groups, truths, added, texts, address_channel=flag)
            print(
                f"{label} {channel} f05 {score:.4f} delta {score - base:+.4f} "
                f"singleton {single:.4f} accepted tp {tp} fp {fp}",
                flush=True,
            )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
