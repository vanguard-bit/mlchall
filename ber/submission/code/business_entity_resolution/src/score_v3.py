"""Train the house-number matcher and rescore the existing test candidates.

Writes output/v3/matching_results.tsv. Candidate ids are unchanged, so
candidate_pairs.tsv is a hardlink of the v2 file. Two score workers, each
holding one batch of texts.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from f05 import macro_f05
from house_features import house_triple, parse_addr
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR, ROOT

TEST = DATA_DIR / "test"
PARTS = DATA_DIR / "export_fast"
BOARD = DATA_DIR / "scoreboard"
TRAIN = DATA_DIR / "train"
OUT = ROOT / "output" / "v3"
MODEL = BOARD / "lgbm_v3.txt"
MIN_GAIN = 0.70
WORKERS = 2
BATCH_S1 = 8_000
FLUSH_PAIRS = 40_000
COUNTRIES = ("France", "US", "India")


def _load_addrs(needed: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, _name, addr, _country = line.rstrip("\n").split("\t")
                if eid in needed:
                    found[eid] = addr
                    if len(found) == len(needed):
                        return found
    return found


def _groups(s1: np.ndarray, mid: np.ndarray, rows: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for row, score in zip(rows, scores):
        sid = str(s1[row])
        slot = groups.get(sid)
        if slot is None:
            slot = {"mids": [], "scores": []}
            groups[sid] = slot
        slot["mids"].append(str(mid[row]))
        slot["scores"].append(float(score))
    for slot in groups.values():
        slot["scores"] = np.asarray(slot["scores"], dtype=np.float32)
    return groups


def _f05(groups: dict[str, dict], truths: dict[str, set[str]]) -> float:
    preds: list[set[str]] = []
    gold: list[set[str]] = []
    for sid, truth in truths.items():
        slot = groups.get(sid)
        chosen: set[str] = set()
        if slot is not None:
            order = np.argsort(-slot["scores"])
            for idx in order:
                if float(slot["scores"][idx]) < MIN_GAIN or len(chosen) >= 12:
                    break
                chosen.add(slot["mids"][idx])
        preds.append(chosen)
        gold.append(truth)
    return macro_f05(gold, preds)


def train_v3() -> None:
    t0 = time.time()
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    X = np.asarray(data["X"], dtype=np.float32)
    y = data["y"].astype(int)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    split = data["split"].astype(str)
    train = split == "train"
    val = split == "val"
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    needed = set(s1.tolist()) | set(mid.tolist())
    print(f"loading {len(needed)} addresses", flush=True)
    addrs = _load_addrs(needed)
    parsed = {eid: parse_addr(addr) for eid, addr in addrs.items()}
    del addrs
    empty = ("", frozenset())
    extra = np.zeros((len(s1), 3), dtype=np.float32)
    for i, (sid, eid) in enumerate(zip(s1, mid)):
        extra[i] = house_triple(parsed.get(str(sid), empty), parsed.get(str(eid), empty))
    del parsed
    gc.collect()
    block = np.concatenate([X, extra], axis=1)
    del extra
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "num_threads": 12,
            "seed": 7,
        },
        lgb.Dataset(block[train], label=y[train]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(block[val], label=y[val])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    scores = model.predict(block[val], num_iteration=model.best_iteration)
    groups = _groups(s1, mid, np.where(val)[0], scores)
    score = _f05(groups, truths)
    print(f"val f05 {score:.4f} trees {model.best_iteration} in {time.time() - t0:.0f}s", flush=True)
    if score < 0.895:
        raise SystemExit(f"val f05 {score:.4f} is below the measured house matcher")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL))
    del block, X, data, groups, scores
    gc.collect()


def load_s1() -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    with (TEST / "test_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            rows[eid] = (name, addr, country)
    return rows


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for _line in handle:
            total += 1
    return total


def load_texts(need: set[str]) -> dict[str, tuple[str, str, str]]:
    texts: dict[str, tuple[str, str, str]] = {}
    if not need:
        return texts
    for name in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid = line.split("\t", 1)[0]
                if eid not in need:
                    continue
                _eid, bname, addr, country = line.rstrip("\n").split("\t")
                texts[eid] = (bname, addr, country)
                if len(texts) == len(need):
                    return texts
    return texts


def score_span(args: tuple[str, int, int, str]) -> int:
    country, start, end, dest = args
    s1 = load_s1()
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    empty_rec = prepare_record("", "")
    empty_house = ("", frozenset())
    pending_x: list[list[float]] = []
    pending_sid: list[str] = []
    pending_mids: list[list[str]] = []
    written = 0
    shard = PARTS / f"{country}.cands.tsv"
    out = open(dest, "w", encoding="utf-8")

    def flush() -> None:
        nonlocal written
        if not pending_sid:
            return
        pred = booster.predict(np.asarray(pending_x, dtype=np.float32), num_iteration=trees)
        cursor = 0
        for sid, mids in zip(pending_sid, pending_mids):
            part = [float(x) for x in pred[cursor : cursor + len(mids)]]
            cursor += len(mids)
            chosen = decode_greedy_f05(mids, part, min_gain=MIN_GAIN, max_preds=12)
            out.write(f"{sid}\t{','.join(mid for mid in mids if mid in chosen)}\n")
            written += 1
        pending_x.clear()
        pending_sid.clear()
        pending_mids.clear()

    def score_batch(lines: list[str]) -> None:
        nonlocal written
        need: set[str] = set()
        parsed: list[tuple[str, str]] = []
        for line in lines:
            sid, _, payload = line.rstrip("\n").partition("\t")
            parsed.append((sid, payload))
            if payload:
                for bit in payload.split(";"):
                    need.add(bit.split("|", 1)[0])
        texts = load_texts(need)
        prepared = {mid: prepare_record(name, addr) for mid, (name, addr, _country) in texts.items()}
        houses = {mid: parse_addr(addr) for mid, (_name, addr, _country) in texts.items()}
        countries = {mid: country_ for mid, (_name, _addr, country_) in texts.items()}
        del texts
        for sid, payload in parsed:
            name, addr, s_country = s1[sid]
            left = prepare_record(name, addr)
            left_house = parse_addr(addr)
            mids: list[str] = []
            if payload:
                for bit in payload.split(";"):
                    mid, nr, ar, ns, asc, fn, fa = bit.split("|")
                    base = features_prepared(
                        left,
                        prepared.get(mid, empty_rec),
                        country_eq=float(s_country == countries.get(mid, "")),
                        name_rank=float(nr),
                        addr_rank=float(ar),
                        name_score=float(ns),
                        addr_score=float(asc),
                        from_name_channel=int(fn),
                        from_addr_channel=int(fa),
                    )
                    pending_x.append(base + house_triple(left_house, houses.get(mid, empty_house)))
                    mids.append(mid)
            if not mids:
                out.write(f"{sid}\t\n")
                written += 1
            else:
                pending_sid.append(sid)
                pending_mids.append(mids)
            if len(pending_x) >= FLUSH_PAIRS:
                flush()
        flush()
        del prepared, houses, countries

    batch: list[str] = []
    with shard.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i < start:
                continue
            if i >= end:
                break
            batch.append(line)
            if len(batch) >= BATCH_S1:
                score_batch(batch)
                batch = []
                print(f"  {country} {start}:{end} {written}", flush=True)
        if batch:
            score_batch(batch)
    out.close()
    print(f"{country} {start}:{end} wrote {written}", flush=True)
    return written


def link_candidates() -> None:
    src = ROOT / "output" / "v2" / "candidate_pairs.tsv"
    dest = OUT / "candidate_pairs.tsv"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.link(src, dest)


def main() -> None:
    t0 = time.time()
    counts = {country: count_lines(PARTS / f"{country}.cands.tsv") for country in COUNTRIES}
    print("shard lines " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    if sum(counts.values()) != 1_732_544:
        raise SystemExit(f"candidate shards sum to {sum(counts.values())}, expected 1732544")
    train_v3()
    OUT.mkdir(parents=True, exist_ok=True)
    part_dir = OUT / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    match_path = OUT / "matching_results.tsv"
    ctx = mp.get_context("spawn")
    with match_path.open("w", encoding="utf-8") as match_out:
        match_out.write("source1_entity_id\tmatched_entity_ids\n")
        for country, n_lines in counts.items():
            mid = n_lines // WORKERS
            spans = [(country, 0, mid, str(part_dir / f"{country}_0.tsv")),
                     (country, mid, n_lines, str(part_dir / f"{country}_1.tsv"))]
            with ctx.Pool(WORKERS) as pool:
                wrote = pool.map(score_span, spans)
            print(f"{country} parts {wrote}", flush=True)
            for _country, _start, _end, part in spans:
                with open(part, encoding="utf-8") as handle:
                    for line in handle:
                        match_out.write(line)
                os.remove(part)
    link_candidates()
    n_rows = count_lines(match_path) - 1
    print(f"wrote {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
    if n_rows != sum(counts.values()):
        raise SystemExit(f"row count {n_rows} != {sum(counts.values())}")


if __name__ == "__main__":
    main()
