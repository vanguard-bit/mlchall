"""Measure how many true links a strict rescue rule adds, and how many false ones."""

from __future__ import annotations

import random
import time
import zipfile
from array import array
from collections import defaultdict

import numpy as np

from blocking import rescue_keys
from paths import TRAIN_PREFIX, ZIP_PATH
from rescue import RESCUE_CAP, rescue_accept

SAMPLE_N = 2500
SEED = 2
TOP_K = 40


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


def main() -> None:
    t0 = time.time()
    zf = zipfile.ZipFile(ZIP_PATH)
    sample = reservoir(zf)
    s1 = load_s1(zf, {sid for sid, _ in sample})
    query_keys: set[str] = set()
    q_keys: list[list[str]] = []
    for sid, _mids in sample:
        name, addr, country = s1[sid]
        keys = [country + "|r|" + key for key in rescue_keys(name, addr)]
        q_keys.append(keys)
        query_keys.update(keys)
    print(f"query rescue keys {len(query_keys)}", flush=True)

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
                prefix = country + "|r|"
                for key in rescue_keys(name, addr):
                    full = prefix + key
                    if full in query_keys:
                        df[full] += 1
                if n_docs % 2_000_000 == 0:
                    print(f"  df {n_docs}", flush=True)
    kept = {key for key, count in df.items() if 0 < count <= RESCUE_CAP}
    key_id = {key: i for i, key in enumerate(kept)}
    print(f"kept {len(kept)} of {len(df)} rescue keys", flush=True)
    del query_keys
    del df

    q_kids: list[list[int]] = []
    for keys in q_keys:
        q_kids.append([key_id[key] for key in keys if key in key_id])
    del q_keys

    key_arr = array("I")
    doc_arr = array("I")
    id_of: list[str] = []
    id_index: dict[str, int] = {}
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as handle:
            handle.readline()
            for line_no, line in enumerate(handle, 1):
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                eid, name, addr, country = parts
                hits = []
                prefix = country + "|r|"
                for key in rescue_keys(name, addr):
                    kid = key_id.get(prefix + key)
                    if kid is not None:
                        hits.append(kid)
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
                    print(f"  postings {fname} {line_no} docs {len(id_of)}", flush=True)
    del id_index
    del key_id
    print(f"indexed {len(id_of)} docs, {len(doc_arr)} postings", flush=True)

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
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del counts

    need: set[str] = set()
    chosen: list[list[str]] = []
    score_buf = np.zeros(len(id_of), dtype=np.int16)
    for qi in range(len(sample)):
        chunks = []
        for kid in q_kids[qi]:
            start, end = int(offsets[kid]), int(offsets[kid + 1])
            if start == end:
                continue
            idx = postings[start:end]
            np.add.at(score_buf, idx, 1)
            chunks.append(idx)
        if not chunks:
            chosen.append([])
            continue
        seen = np.unique(np.concatenate(chunks))
        scores = score_buf[seen]
        n = int(scores.shape[0])
        if n > TOP_K:
            pick = np.argpartition(scores, -TOP_K)[-TOP_K:]
        else:
            pick = np.arange(n)
        mids = [id_of[int(seen[i])] for i in pick]
        chosen.append(mids)
        need.update(mids)
        score_buf[seen] = 0
        if (qi + 1) % 500 == 0:
            print(f"  queried {qi + 1}", flush=True)
    print(f"candidate ids to load {len(need)}", flush=True)

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

    tp = fp = skipped_ambiguous = 0
    shown_tp = shown_fp = 0
    for qi, (sid, truth) in enumerate(sample):
        s_name, s_addr, _country = s1[sid]
        passed: list[str] = []
        for mid in chosen[qi]:
            pair = texts.get(mid)
            if pair is None:
                continue
            if rescue_accept(s_name, s_addr, pair[0], pair[1]):
                passed.append(mid)
        if len(passed) != 1:
            if len(passed) > 1:
                skipped_ambiguous += 1
            continue
        mid = passed[0]
        pair = texts[mid]
        if mid in truth:
            tp += 1
            if shown_tp < 8:
                print(f"TP {s_name} || {pair[0]}", flush=True)
                shown_tp += 1
        else:
            fp += 1
            if shown_fp < 8:
                print(f"FP {s_name} || {pair[0]}", flush=True)
                shown_fp += 1
    n_true = sum(len(truth) for _sid, truth in sample)
    precision = tp / (tp + fp) if tp + fp else 0.0
    print(
        f"unique adds tp={tp} fp={fp} precision={precision:.3f} "
        f"ambiguous_rows={skipped_ambiguous} "
        f"over {n_true} true links in {SAMPLE_N} rows",
        flush=True,
    )
    print(f"done in {time.time() - t0:.0f}s n_docs={n_docs}", flush=True)


if __name__ == "__main__":
    main()
