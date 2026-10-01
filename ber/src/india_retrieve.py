"""Candidate lists for every train India Source 1 row.

Same blocker as the test export. India only, so the posting index stays
smaller than the full test run. Writes ber/data/india_slice/cands_*.tsv.
"""

from __future__ import annotations

import gc
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
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
PARTS = DATA_DIR / "india_slice"
COUNTRY = "India"
SCAN_WORKERS = 8
QUERY_WORKERS = 2
K_NAME = 25
K_ADDR = 25

QUERY: set[str] = set()
KEY_ID: dict[str, int] = {}
SIDS: list[str] = []
NAME_KIDS: list[list[int]] = []
ADDR_KIDS: list[list[int]] = []
OFFSETS: np.ndarray
POSTINGS: np.ndarray
IDF: np.ndarray
EIDS: list[str] = []


def _df_worker(args: tuple[str, int]) -> tuple[dict[str, int], int]:
    path, wid = args
    counts: dict[str, int] = {}
    seen = 0
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % SCAN_WORKERS != wid:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4 or parts[3] != COUNTRY:
                continue
            seen += 1
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
    return counts, seen


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
            if len(parts) != 4 or parts[3] != COUNTRY:
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


def main() -> None:
    global QUERY, KEY_ID, OFFSETS, POSTINGS, IDF, EIDS
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, _name, _addr, country = line.rstrip("\n").split("\t")
            if country == COUNTRY:
                SIDS.append(eid)
    print(f"s1 {len(SIDS)}", flush=True)
    s1: dict[str, tuple[str, str]] = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            if country != COUNTRY:
                continue
            s1[eid] = (name, addr)
            for key in name_keys(name):
                QUERY.add(country + "|n|" + key)
            for key in addr_keys(addr):
                QUERY.add(country + "|a|" + key)
    print(f"query keys {len(QUERY)} in {time.time() - t0:.0f}s", flush=True)

    ctx = mp.get_context("fork")
    files = ("train_source2.tsv", "train_source3.tsv")
    jobs = [(str(TRAIN / name), wid) for name in files for wid in range(SCAN_WORKERS)]
    with ctx.Pool(SCAN_WORKERS) as pool:
        parts = pool.map(_df_worker, jobs)
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for counts, seen in parts:
        n_docs += seen
        for key, count in counts.items():
            df[key] += count
    print(f"df keys {len(df)} india docs {n_docs} in {time.time() - t0:.0f}s", flush=True)

    kept: list[str] = []
    for sid in SIDS:
        name, addr = s1[sid]
        kn, ka = query_key_lists(
            name, addr, COUNTRY, df,
            posting_cap=POSTING_CAP, max_name_keys=MAX_NAME_KEYS, max_addr_keys=MAX_ADDR_KEYS,
        )
        name_row: list[int] = []
        addr_row: list[int] = []
        for key in kn:
            if key not in KEY_ID:
                KEY_ID[key] = len(kept)
                kept.append(key)
            name_row.append(KEY_ID[key])
        for key in ka:
            if key not in KEY_ID:
                KEY_ID[key] = len(kept)
                kept.append(key)
            addr_row.append(KEY_ID[key])
        NAME_KIDS.append(name_row)
        ADDR_KIDS.append(addr_row)
    idf = np.array([np.log((n_docs + 1) / (df[key] + 1)) for key in kept], dtype=np.float32)
    del df, QUERY, s1
    print(f"kept {len(kept)}", flush=True)

    with ctx.Pool(SCAN_WORKERS) as pool:
        pool.map(_post_worker, jobs)
    eids: list[str] = []
    key_chunks = []
    doc_chunks = []
    for name in files:
        for wid in range(SCAN_WORKERS):
            stem = PARTS / f"{Path(name).stem}_{wid}"
            local = stem.with_suffix(".eids.txt").read_text(encoding="utf-8").splitlines()
            if not local:
                continue
            docs = np.load(stem.with_suffix(".docs.npy")).astype(np.uint32) + len(eids)
            key_chunks.append(np.load(stem.with_suffix(".keys.npy")).astype(np.uint32))
            doc_chunks.append(docs)
            eids.extend(local)
    key_np = np.concatenate(key_chunks) if key_chunks else np.zeros(0, np.uint32)
    doc_np = np.concatenate(doc_chunks) if doc_chunks else np.zeros(0, np.uint32)
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"indexed {len(eids)} postings {len(postings)} in {time.time() - t0:.0f}s", flush=True)

    global OFFSETS, POSTINGS, IDF, EIDS
    OFFSETS, POSTINGS, IDF, EIDS = offsets, postings, idf, eids
    del KEY_ID, kept, key_np, doc_np
    gc.collect()
    with ctx.Pool(QUERY_WORKERS) as pool:
        pool.map(_query_worker, range(QUERY_WORKERS))
    print(f"queries done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
