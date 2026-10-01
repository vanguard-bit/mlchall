"""Eight-way test export. Writes output/v2 and leaves output/matching_results.tsv alone."""

from __future__ import annotations

import multiprocessing as mp
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np

from blocking import (
    MAX_ADDR_KEYS,
    MAX_NAME_KEYS,
    POSTING_CAP,
    addr_keys,
    name_keys,
    query_key_lists,
)
from decode import decode_greedy_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR, ROOT

TEST = DATA_DIR / "test"
PARTS = DATA_DIR / "export_fast"
OUT = ROOT / "output" / "v2"
MODEL = DATA_DIR / "scoreboard" / "lgbm_v2.txt"
SCAN_WORKERS = 8
QUERY_WORKERS = 2
SCORE_WORKERS = 2
K_NAME = 25
K_ADDR = 25
MIN_GAIN = 0.70

QUERY: set[str] = set()
KEY_ID: dict[str, int] = {}
SIDS: list[str] = []
NAME_KIDS: list[list[int]] = []
ADDR_KIDS: list[list[int]] = []
OFFSETS: np.ndarray
POSTINGS: np.ndarray
IDF: np.ndarray
EIDS: list[str] = []
TEXTS: dict[str, tuple[str, str, str]] = {}
S1: dict[str, tuple[str, str, str]] = {}


def _df_worker(args: tuple[str, int]) -> dict[str, int]:
    path, wid = args
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % SCAN_WORKERS != wid:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            _eid, name, addr, country = parts
            pn, pa = country + "|n|", country + "|a|"
            for key in name_keys(name):
                full = pn + key
                if full in QUERY:
                    counts[full] = counts.get(full, 0) + 1
            for key in addr_keys(addr):
                full = pa + key
                if full in QUERY:
                    counts[full] = counts.get(full, 0) + 1
            if i and i % 2_000_000 == 0 and wid == 0:
                print(f"  df {Path(path).name} {i}", flush=True)
    return counts


def _post_worker(args: tuple[str, int]) -> None:
    path, wid = args
    key_ids = array("I")
    local_docs = array("I")
    eids: list[str] = []
    index: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % SCAN_WORKERS != wid:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            eid, name, addr, country = parts
            hits: list[int] = []
            pn, pa = country + "|n|", country + "|a|"
            for key in name_keys(name):
                kid = KEY_ID.get(pn + key)
                if kid is not None:
                    hits.append(kid)
            for key in addr_keys(addr):
                kid = KEY_ID.get(pa + key)
                if kid is not None:
                    hits.append(kid)
            if not hits:
                continue
            doc = index.get(eid)
            if doc is None:
                doc = len(eids)
                index[eid] = doc
                eids.append(eid)
            for kid in hits:
                key_ids.append(kid)
                local_docs.append(doc)
    stem = PARTS / f"{Path(path).stem}_{wid}"
    np.save(stem.with_suffix(".keys.npy"), np.array(key_ids, dtype=np.uint32))
    np.save(stem.with_suffix(".docs.npy"), np.array(local_docs, dtype=np.uint32))
    stem.with_suffix(".eids.txt").write_text("\n".join(eids), encoding="utf-8")
    print(f"  post {Path(path).name} w{wid} docs {len(eids)}", flush=True)


def _top(score_buf: np.ndarray, chunks: list[np.ndarray], k: int) -> list[tuple[int, float]]:
    if not chunks:
        return []
    seen = np.unique(np.concatenate(chunks))
    scores = score_buf[seen]
    n = int(scores.shape[0])
    if n > k:
        pick = np.argpartition(scores, -k)[-k:]
        pick = pick[np.argsort(scores[pick])[::-1]]
    else:
        pick = np.argsort(scores)[::-1]
    result = [(int(seen[i]), float(scores[i])) for i in pick]
    score_buf[seen] = 0
    return result


def _query_worker(wid: int) -> None:
    score_buf = np.zeros(len(EIDS), dtype=np.float32)
    path = PARTS / f"cands_{wid}.tsv"
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for qi in range(wid, len(SIDS), QUERY_WORKERS):
            chunks = []
            for kid in NAME_KIDS[qi]:
                a, b = int(OFFSETS[kid]), int(OFFSETS[kid + 1])
                if a == b:
                    continue
                idx = POSTINGS[a:b]
                np.add.at(score_buf, idx, IDF[kid])
                chunks.append(idx)
            name_top = _top(score_buf, chunks, K_NAME)
            chunks = []
            for kid in ADDR_KIDS[qi]:
                a, b = int(OFFSETS[kid]), int(OFFSETS[kid + 1])
                if a == b:
                    continue
                idx = POSTINGS[a:b]
                np.add.at(score_buf, idx, IDF[kid])
                chunks.append(idx)
            addr_top = _top(score_buf, chunks, K_ADDR)
            meta: dict[str, dict] = {}
            ordered: list[str] = []
            for rank, (doc, score) in enumerate(name_top):
                mid = EIDS[doc]
                meta[mid] = {"ns": score, "nr": float(rank), "fn": 1}
                ordered.append(mid)
            for rank, (doc, score) in enumerate(addr_top):
                mid = EIDS[doc]
                slot = meta.get(mid)
                if slot is None:
                    meta[mid] = {"as": score, "ar": float(rank), "fa": 1}
                    ordered.append(mid)
                else:
                    slot["as"] = score
                    slot["ar"] = float(rank)
                    slot["fa"] = 1
            bits = []
            for mid in ordered:
                item = meta[mid]
                bits.append(
                    f"{mid}|{item.get('nr', 99):.0f}|{item.get('ar', 99):.0f}|"
                    f"{item.get('ns', 0):.5f}|{item.get('as', 0):.5f}|"
                    f"{item.get('fn', 0)}|{item.get('fa', 0)}"
                )
            handle.write(f"{SIDS[qi]}\t{';'.join(bits)}\n")
            n += 1
            if n % 50_000 == 0:
                print(f"  query w{wid} {n}", flush=True)
    print(f"  query w{wid} done {n}", flush=True)


def _score_worker(wid: int) -> None:
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    src = PARTS / f"cands_{wid}.tsv"
    match_path = PARTS / f"match_{wid}.tsv"
    cand_path = PARTS / f"cand_{wid}.tsv"
    pending_x: list[list[float]] = []
    pending_sid: list[str] = []
    pending_mids: list[list[str]] = []

    def flush(match_out, cand_out) -> None:
        if not pending_sid:
            return
        pred = booster.predict(np.asarray(pending_x, dtype=np.float32), num_iteration=trees)
        cursor = 0
        for sid, mids in zip(pending_sid, pending_mids):
            n = len(mids)
            part = [float(x) for x in pred[cursor : cursor + n]]
            cursor += n
            chosen = decode_greedy_f05(mids, part, min_gain=MIN_GAIN, max_preds=12)
            kept = [mid for mid in mids if mid in chosen]
            match_out.write(f"{sid}\t{','.join(kept)}\n")
            cand_out.write(f"{sid}\t{','.join(mids)}\n")
        pending_x.clear()
        pending_sid.clear()
        pending_mids.clear()

    with src.open(encoding="utf-8") as handle, match_path.open("w", encoding="utf-8") as match_out, cand_path.open("w", encoding="utf-8") as cand_out:
        done = 0
        for line in handle:
            sid, _, payload = line.rstrip("\n").partition("\t")
            name, addr, country = S1[sid]
            left = prepare_record(name, addr)
            mids: list[str] = []
            if payload:
                for bit in payload.split(";"):
                    mid, nr, ar, ns, asc, fn, fa = bit.split("|")
                    pair = TEXTS.get(mid)
                    if pair is None:
                        right = prepare_record("", "")
                        mc = ""
                    else:
                        right = prepare_record(pair[0], pair[1])
                        mc = pair[2]
                    pending_x.append(
                        features_prepared(
                            left, right,
                            country_eq=float(country == mc),
                            name_rank=float(nr),
                            addr_rank=float(ar),
                            name_score=float(ns),
                            addr_score=float(asc),
                            from_name_channel=int(fn),
                            from_addr_channel=int(fa),
                        )
                    )
                    mids.append(mid)
            if not mids:
                match_out.write(f"{sid}\t\n")
                cand_out.write(f"{sid}\t\n")
            else:
                pending_sid.append(sid)
                pending_mids.append(mids)
            if len(pending_x) >= 80_000:
                flush(match_out, cand_out)
            done += 1
            if done % 50_000 == 0:
                print(f"  score w{wid} {done}", flush=True)
        flush(match_out, cand_out)
    print(f"  score w{wid} done", flush=True)


def main() -> None:
    global QUERY, KEY_ID, OFFSETS, POSTINGS, IDF, EIDS, TEXTS, S1
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    S1 = {}
    with (TEST / "test_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            S1[eid] = (name, addr, country)
            SIDS.append(eid)
    print(f"s1 {len(SIDS)}", flush=True)
    for sid in SIDS:
        name, addr, country = S1[sid]
        for key in name_keys(name):
            QUERY.add(country + "|n|" + key)
        for key in addr_keys(addr):
            QUERY.add(country + "|a|" + key)
    print(f"query keys {len(QUERY)} in {time.time() - t0:.0f}s", flush=True)

    ctx = mp.get_context("fork")
    jobs = [(str(TEST / name), wid) for name in ("test_source2.tsv", "test_source3.tsv") for wid in range(SCAN_WORKERS)]
    with ctx.Pool(SCAN_WORKERS) as pool:
        parts = pool.map(_df_worker, jobs)
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for part in parts:
        for key, count in part.items():
            df[key] += count
    for name in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST / name).open(encoding="utf-8") as handle:
            handle.readline()
            for _ in handle:
                n_docs += 1
    print(f"df keys {len(df)} docs {n_docs} in {time.time() - t0:.0f}s", flush=True)

    kept: list[str] = []
    for sid in SIDS:
        name, addr, country = S1[sid]
        kn, ka = query_key_lists(
            name, addr, country, df,
            posting_cap=POSTING_CAP, max_name_keys=MAX_NAME_KEYS, max_addr_keys=MAX_ADDR_KEYS,
        )
        NAME_KIDS.append([])
        ADDR_KIDS.append([])
        for key in kn:
            if key not in KEY_ID:
                KEY_ID[key] = len(kept)
                kept.append(key)
            NAME_KIDS[-1].append(KEY_ID[key])
        for key in ka:
            if key not in KEY_ID:
                KEY_ID[key] = len(kept)
                kept.append(key)
            ADDR_KIDS[-1].append(KEY_ID[key])
    idf = np.array([np.log((n_docs + 1) / (df[key] + 1)) for key in kept], dtype=np.float32)
    del df
    del QUERY
    print(f"kept {len(kept)}", flush=True)

    with ctx.Pool(SCAN_WORKERS) as pool:
        pool.map(_post_worker, jobs)
    eids: list[str] = []
    key_chunks = []
    doc_chunks = []
    for name in ("test_source2.tsv", "test_source3.tsv"):
        for wid in range(SCAN_WORKERS):
            stem = PARTS / f"{Path(name).stem}_{wid}"
            local = stem.with_suffix(".eids.txt").read_text(encoding="utf-8").splitlines()
            if not local:
                continue
            docs = np.load(stem.with_suffix(".docs.npy")).astype(np.uint32) + len(eids)
            key_chunks.append(np.load(stem.with_suffix(".keys.npy")).astype(np.uint32))
            doc_chunks.append(docs)
            eids.extend(local)
    key_np = np.concatenate(key_chunks)
    doc_np = np.concatenate(doc_chunks)
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"indexed {len(eids)} postings {len(postings)} in {time.time() - t0:.0f}s", flush=True)

    global OFFSETS, POSTINGS, IDF, EIDS
    OFFSETS, POSTINGS, IDF, EIDS = offsets, postings, idf, eids
    del KEY_ID
    del kept
    import gc
    gc.collect()
    with ctx.Pool(QUERY_WORKERS) as pool:
        pool.map(_query_worker, range(QUERY_WORKERS))
    del POSTINGS, OFFSETS, key_np, doc_np
    print(f"queries done in {time.time() - t0:.0f}s", flush=True)

    need: set[str] = set()
    for wid in range(QUERY_WORKERS):
        with (PARTS / f"cands_{wid}.tsv").open(encoding="utf-8") as handle:
            for line in handle:
                _sid, _, payload = line.rstrip("\n").partition("\t")
                if not payload:
                    continue
                for bit in payload.split(";"):
                    need.add(bit.split("|", 1)[0])
    print(f"need texts {len(need)}", flush=True)
    texts: dict[str, tuple[str, str, str]] = {}
    for name in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid = line.split("\t", 1)[0]
                if eid not in need:
                    continue
                _eid, name_, addr, country = line.rstrip("\n").split("\t")
                texts[eid] = (name_, addr, country)
                if len(texts) == len(need):
                    break
        if len(texts) == len(need):
            break
    global TEXTS
    TEXTS = texts
    with ctx.Pool(SCORE_WORKERS) as pool:
        pool.map(_score_worker, range(SCORE_WORKERS))

    n = 0
    match_out = OUT / "matching_results.tsv"
    cand_out = OUT / "candidate_pairs.tsv"
    with match_out.open("w", encoding="utf-8") as mout, cand_out.open("w", encoding="utf-8") as cout:
        mout.write("source1_entity_id\tmatched_entity_ids\n")
        cout.write("source1_entity_id\tcandidate_entity_ids\n")
        for wid in range(SCORE_WORKERS):
            mlines = (PARTS / f"match_{wid}.tsv").read_text(encoding="utf-8")
            clines = (PARTS / f"cand_{wid}.tsv").read_text(encoding="utf-8")
            mout.write(mlines)
            cout.write(clines)
            n += mlines.count("\n")
    print(f"wrote {match_out} rows {n} in {time.time() - t0:.0f}s", flush=True)
    if n != len(SIDS):
        raise SystemExit(f"row count {n} != {len(SIDS)}")


if __name__ == "__main__":
    main()
