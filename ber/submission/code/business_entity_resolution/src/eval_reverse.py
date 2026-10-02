"""Reverse blocking: S2/S3 nominate holdout S1 rows. Count links the forward list missed."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict

import numpy as np

from blocking import addr_keys, name_keys
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
TOP_S1 = 5
KEEP_PER_S1 = 15
POST_CAP = 80

S1_IDS: list[str] = []
S1_COUNTRY: list[str] = []
POSTINGS: dict[str, np.ndarray] = {}
IDF: dict[str, float] = {}


def load_holdout():
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    wanted = {row["sid"]: row for row in meta}
    texts: dict[str, tuple[str, str, str]] = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            if eid in wanted:
                texts[eid] = (name, addr, country)
                if len(texts) == len(wanted):
                    break
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    forward: dict[str, set[str]] = defaultdict(set)
    for sid, mid in zip(data["s1_id"].astype(str), data["match_id"].astype(str)):
        if sid in wanted:
            forward[sid].add(mid)
    return wanted, texts, forward


def build_index(wanted, texts):
    global S1_IDS, S1_COUNTRY, POSTINGS, IDF
    S1_IDS = list(wanted)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(S1_IDS):
        name, addr, country = texts[sid]
        S1_COUNTRY.append(country)
        for key in name_keys(name):
            buckets[country + "|n|" + key].append(i)
        for key in addr_keys(addr):
            buckets[country + "|a|" + key].append(i)
    n = len(S1_IDS)
    kept = {key: ids for key, ids in buckets.items() if 0 < len(ids) <= POST_CAP}
    POSTINGS = {key: np.asarray(ids, dtype=np.int32) for key, ids in kept.items()}
    IDF = {key: float(np.log((n + 1) / (len(ids) + 1))) for key, ids in kept.items()}
    print(f"indexed {n} s1 keys {len(POSTINGS)}", flush=True)


def _worker(args: tuple[str, int]) -> dict[int, list[tuple[float, str]]]:
    path, wid = args
    heaps: dict[int, list[tuple[float, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            eid, name, addr, country = parts
            scores: dict[int, float] = {}
            pn, pa = country + "|n|", country + "|a|"
            for key in name_keys(name):
                ids = POSTINGS.get(pn + key)
                if ids is None:
                    continue
                weight = IDF[pn + key]
                for s1_i in ids:
                    scores[s1_i] = scores.get(s1_i, 0.0) + weight
            for key in addr_keys(addr):
                ids = POSTINGS.get(pa + key)
                if ids is None:
                    continue
                weight = IDF[pa + key]
                for s1_i in ids:
                    scores[s1_i] = scores.get(s1_i, 0.0) + weight
            if not scores:
                continue
            if len(scores) > TOP_S1:
                top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:TOP_S1]
            else:
                top = list(scores.items())
            for s1_i, score in top:
                heap = heaps[s1_i]
                if len(heap) < KEEP_PER_S1:
                    heap.append((score, eid))
                else:
                    worst = min(range(len(heap)), key=lambda j: heap[j][0])
                    if score > heap[worst][0]:
                        heap[worst] = (score, eid)
            if i and i % 1_000_000 == 0 and wid == 0:
                print(f"  reverse {path.rsplit('/', 1)[-1]} {i}", flush=True)
    return dict(heaps)


def main() -> None:
    t0 = time.time()
    wanted, texts, forward = load_holdout()
    build_index(wanted, texts)
    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    with ctx.Pool(WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    merged: dict[int, dict[str, float]] = defaultdict(dict)
    for part in parts:
        for s1_i, hits in part.items():
            slot = merged[s1_i]
            for score, eid in hits:
                prev = slot.get(eid)
                if prev is None or score > prev:
                    slot[eid] = score
    rows: list[tuple[str, str, float, int, int, int]] = []
    val_fwd_tp = val_true = 0
    for s1_i, sid in enumerate(S1_IDS):
        row = wanted[sid]
        if row["split"] != "val":
            continue
        truth = set(row["truth"])
        fwd = forward.get(sid, set())
        val_true += len(truth)
        val_fwd_tp += len(truth & fwd)
        ranked = sorted(merged.get(s1_i, {}).items(), key=lambda item: item[1], reverse=True)[:KEEP_PER_S1]
        for rank, (eid, score) in enumerate(ranked, start=1):
            rows.append((sid, eid, float(score), rank, int(eid in truth), int(eid in fwd)))
    out_s1 = np.asarray([r[0] for r in rows])
    out_mid = np.asarray([r[1] for r in rows])
    out_score = np.asarray([r[2] for r in rows], dtype=np.float32)
    out_rank = np.asarray([r[3] for r in rows], dtype=np.int16)
    out_y = np.asarray([r[4] for r in rows], dtype=np.int8)
    out_fwd = np.asarray([r[5] for r in rows], dtype=np.int8)
    np.savez_compressed(
        BOARD / "reverse_val.npz",
        s1_id=out_s1,
        match_id=out_mid,
        rev_score=out_score,
        rev_rank=out_rank,
        y=out_y,
        in_forward=out_fwd,
    )
    new = out_fwd == 0
    for k in (1, 3, 5, 10, 15):
        keep = new & (out_rank <= k)
        tp = int((out_y[keep] == 1).sum())
        fp = int((out_y[keep] == 0).sum())
        print(f"top {k} new tp {tp} fp {fp}", flush=True)
    print(
        f"val true links {val_true} forward hit {val_fwd_tp} "
        f"recall {val_fwd_tp / val_true if val_true else 0:.3f}",
        flush=True,
    )
    print(f"saved {len(rows)} reverse nominations in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
