"""How many unique rescue accepts are already in the main top-25 candidate list."""

from __future__ import annotations

import random
import time
import zipfile
from array import array
from collections import defaultdict

import numpy as np

from blocking import (
    MAX_ADDR_KEYS,
    MAX_NAME_KEYS,
    POSTING_CAP,
    addr_keys,
    name_keys,
    query_key_lists,
    rescue_keys,
)
from paths import TRAIN_PREFIX, ZIP_PATH
from rescue import RESCUE_CAP, rescue_accept

SAMPLE_N = 2500
SEED = 2
K_NAME = 25
K_ADDR = 25
RESCUE_TOP = 40


def reservoir(zf: zipfile.ZipFile) -> list[tuple[str, set[str]]]:
    rng = random.Random(SEED)
    sample: list[tuple[str, set[str]]] = []
    seen = 0
    with zf.open(f"{TRAIN_PREFIX}/train_ground_truth.tsv") as handle:
        handle.readline()
        for line in handle:
            seen += 1
            parts = line.decode().rstrip("\n").split("\t")
            mids = [x for x in parts[1].split(",") if x] if len(parts) > 1 else []
            row = (parts[0], set(mids))
            if len(sample) < SAMPLE_N:
                sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < SAMPLE_N:
                    sample[j] = row
    print(f"sampled {len(sample)} from {seen}", flush=True)
    return sample


def load_s1(zf: zipfile.ZipFile, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    with zf.open(f"{TRAIN_PREFIX}/train_source1.tsv") as handle:
        handle.readline()
        for line in handle:
            parts = line.decode().rstrip("\n").split("\t")
            if len(parts) != 4 or parts[0] not in wanted:
                continue
            out[parts[0]] = (parts[1], parts[2], parts[3])
            if len(out) == len(wanted):
                break
    return out


def top_ids(score_buf: np.ndarray, chunks: list[np.ndarray], k: int, id_of: list[str]) -> list[str]:
    if not chunks:
        return []
    seen = np.unique(np.concatenate(chunks))
    scores = score_buf[seen]
    n = int(scores.shape[0])
    if n > k:
        pick = np.argpartition(scores, -k)[-k:]
    else:
        pick = np.arange(n)
    mids = [id_of[int(seen[i])] for i in pick]
    score_buf[seen] = 0
    return mids


def main() -> None:
    t0 = time.time()
    zf = zipfile.ZipFile(ZIP_PATH)
    sample = reservoir(zf)
    s1 = load_s1(zf, {sid for sid, _ in sample})

    main_query: set[str] = set()
    rescue_query: set[str] = set()
    for sid, _mids in sample:
        name, addr, country = s1[sid]
        for key in name_keys(name):
            main_query.add(country + "|n|" + key)
        for key in addr_keys(addr):
            main_query.add(country + "|a|" + key)
        for key in rescue_keys(name, addr):
            rescue_query.add(country + "|r|" + key)
    print(f"main keys {len(main_query)} rescue keys {len(rescue_query)}", flush=True)

    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as handle:
            handle.readline()
            for line in handle:
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                _eid, name, addr, country = parts
                n_docs += 1
                pn, pa, pr = country + "|n|", country + "|a|", country + "|r|"
                for key in name_keys(name):
                    full = pn + key
                    if full in main_query:
                        df[full] += 1
                for key in addr_keys(addr):
                    full = pa + key
                    if full in main_query:
                        df[full] += 1
                for key in rescue_keys(name, addr):
                    full = pr + key
                    if full in rescue_query:
                        df[full] += 1
                if n_docs % 2_000_000 == 0:
                    print(f"  df {n_docs}", flush=True)

    main_kids: list[tuple[list[int], list[int]]] = []
    rescue_kids: list[list[int]] = []
    kept: list[str] = []
    key_id: dict[str, int] = {}

    def intern(key: str) -> int:
        kid = key_id.get(key)
        if kid is None:
            kid = len(kept)
            key_id[key] = kid
            kept.append(key)
        return kid

    for sid, _mids in sample:
        name, addr, country = s1[sid]
        kn, ka = query_key_lists(
            name, addr, country, df,
            posting_cap=POSTING_CAP,
            max_name_keys=MAX_NAME_KEYS,
            max_addr_keys=MAX_ADDR_KEYS,
        )
        main_kids.append(([intern(k) for k in kn], [intern(k) for k in ka]))
        rk = []
        for key in rescue_keys(name, addr):
            full = country + "|r|" + key
            count = df.get(full, 0)
            if 0 < count <= RESCUE_CAP:
                rk.append(intern(full))
        rescue_kids.append(rk)
    del main_query
    del rescue_query
    del df
    print(f"kept keys {len(kept)}", flush=True)

    key_arr = array("I")
    doc_arr = array("I")
    id_of: list[str] = []
    id_index: dict[str, int] = {}
    kept_set = set(kept)
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as handle:
            handle.readline()
            for line_no, line in enumerate(handle, 1):
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                eid, name, addr, country = parts
                hits: list[int] = []
                pn, pa, pr = country + "|n|", country + "|a|", country + "|r|"
                for key in name_keys(name):
                    full = pn + key
                    if full in kept_set:
                        hits.append(key_id[full])
                for key in addr_keys(addr):
                    full = pa + key
                    if full in kept_set:
                        hits.append(key_id[full])
                for key in rescue_keys(name, addr):
                    full = pr + key
                    if full in kept_set:
                        hits.append(key_id[full])
                if not hits:
                    continue
                doc = id_index.get(eid)
                if doc is None:
                    doc = len(id_of)
                    id_index[eid] = doc
                    id_of.append(eid)
                for kid in hits:
                    key_arr.append(kid)
                    doc_arr.append(doc)
                if line_no % 2_000_000 == 0:
                    print(f"  postings {line_no} docs {len(id_of)}", flush=True)
    del id_index
    del kept_set
    print(f"indexed {len(id_of)} postings {len(doc_arr)}", flush=True)

    key_np = np.frombuffer(key_arr, dtype=np.uint32, count=len(key_arr)).copy()
    doc_np = np.frombuffer(doc_arr, dtype=np.uint32, count=len(doc_arr)).copy()
    del key_arr
    del doc_arr
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept))
    del key_np
    del doc_np
    del kept
    del key_id
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del counts

    def channel(kids: list[int], buf: np.ndarray) -> list[np.ndarray]:
        chunks = []
        for kid in kids:
            start, end = int(offsets[kid]), int(offsets[kid + 1])
            if start == end:
                continue
            idx = postings[start:end]
            np.add.at(buf, idx, 1)
            chunks.append(idx)
        return chunks

    score_buf = np.zeros(len(id_of), dtype=np.int16)
    main_sets: list[set[str]] = []
    rescue_lists: list[list[str]] = []
    need: set[str] = set()
    for qi in range(len(sample)):
        name_ch = channel(main_kids[qi][0], score_buf)
        name_top = top_ids(score_buf, name_ch, K_NAME, id_of)
        addr_ch = channel(main_kids[qi][1], score_buf)
        addr_top = top_ids(score_buf, addr_ch, K_ADDR, id_of)
        main_sets.append(set(name_top) | set(addr_top))
        rescue_ch = channel(rescue_kids[qi], score_buf)
        rescue_top = top_ids(score_buf, rescue_ch, RESCUE_TOP, id_of)
        rescue_lists.append(rescue_top)
        need.update(rescue_top)
        if (qi + 1) % 500 == 0:
            print(f"  queried {qi + 1}", flush=True)

    texts: dict[str, tuple[str, str]] = {}
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as handle:
            handle.readline()
            for line in handle:
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4 or parts[0] not in need:
                    continue
                texts[parts[0]] = (parts[1], parts[2])
                if len(texts) == len(need):
                    break
        if len(texts) == len(need):
            break

    tp_in = tp_out = fp_in = fp_out = 0
    shown = 0
    for qi, (sid, truth) in enumerate(sample):
        s_name, s_addr, _country = s1[sid]
        passed = []
        for mid in rescue_lists[qi]:
            pair = texts.get(mid)
            if pair is None:
                continue
            if rescue_accept(s_name, s_addr, pair[0], pair[1]):
                passed.append(mid)
        if len(passed) != 1:
            continue
        mid = passed[0]
        inside = mid in main_sets[qi]
        is_tp = mid in truth
        if is_tp and inside:
            tp_in += 1
        elif is_tp:
            tp_out += 1
            if shown < 12:
                pair = texts[mid]
                print(f"NEW TP {s_name} || {pair[0]}", flush=True)
                shown += 1
        elif inside:
            fp_in += 1
        else:
            fp_out += 1
    total_tp = tp_in + tp_out
    print(
        f"unique rescue TP {total_tp}: already in candidates {tp_in}, "
        f"absent {tp_out}",
        flush=True,
    )
    print(
        f"unique rescue FP {fp_in + fp_out}: already in candidates {fp_in}, "
        f"absent {fp_out}",
        flush=True,
    )
    print(f"done in {time.time() - t0:.0f}s docs {n_docs}", flush=True)


if __name__ == "__main__":
    main()
