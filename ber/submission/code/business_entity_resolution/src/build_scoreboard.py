"""Parallel holdout: block 30k train entities and save pair features plus full truth."""

from __future__ import annotations

import json
import multiprocessing as mp
import random
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
from pair_features import FEATURE_NAMES, features_prepared, prepare_record
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
OUT = DATA_DIR / "scoreboard"
PARTS = OUT / "parts"
N_SAMPLE = 30_000
SEED = 7
WORKERS = 8
K_NAME = 25
K_ADDR = 25

QUERY: set[str] = set()
KEY_ID: dict[str, int] = {}


def _bucket(country: str, n_true: int) -> tuple[str, int, int]:
    if n_true <= 1:
        length = n_true
    elif n_true == 2:
        length = 2
    elif n_true <= 4:
        length = 3
    else:
        length = 4
    return country, int(n_true == 0), length


def sample_ids() -> tuple[list[str], list[str]]:
    countries: dict[str, str] = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, _name, _addr, country = line.rstrip("\n").split("\t")
            countries[eid] = country
    buckets: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            n_true = 0 if not rest else len([x for x in rest.split(",") if x])
            buckets[_bucket(countries[sid], n_true)].append(sid)
    total = sum(len(ids) for ids in buckets.values())
    rng = random.Random(SEED)
    chosen: list[str] = []
    for ids in buckets.values():
        take = int(round(N_SAMPLE * len(ids) / total))
        take = min(take, len(ids))
        if take:
            chosen.extend(rng.sample(ids, take))
    rng.shuffle(chosen)
    chosen = chosen[:N_SAMPLE]
    val_n = N_SAMPLE // 5
    val_ids = chosen[:val_n]
    train_ids = chosen[val_n:]
    print(f"sample train {len(train_ids)} val {len(val_ids)}", flush=True)
    return train_ids, val_ids


def load_rows(ids: set[str]) -> tuple[dict[str, tuple[str, str, str]], dict[str, list[str]]]:
    texts: dict[str, tuple[str, str, str]] = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in ids:
                texts[parts[0]] = (parts[1], parts[2], parts[3])
                if len(texts) == len(ids):
                    break
    truths: dict[str, list[str]] = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            if sid in ids:
                truths[sid] = [x for x in rest.split(",") if x] if rest else []
                if len(truths) == len(ids):
                    break
    return texts, truths


def _df_worker(args: tuple[str, int, int]) -> tuple[dict[str, int], int]:
    path, wid, nworkers = args
    counts: dict[str, int] = {}
    seen = 0
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % nworkers != wid:
                continue
            seen += 1
            _eid, name, addr, country = line.rstrip("\n").split("\t")
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


def _post_worker(args: tuple[str, int, int]) -> None:
    path, wid, nworkers = args
    key_ids = array("I")
    local_docs = array("I")
    eids: list[str] = []
    index: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % nworkers != wid:
                continue
            eid, name, addr, country = line.rstrip("\n").split("\t")
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
    if key_ids:
        np.save(stem.with_suffix(".keys.npy"), np.array(key_ids, dtype=np.uint32))
        np.save(stem.with_suffix(".docs.npy"), np.array(local_docs, dtype=np.uint32))
    else:
        np.save(stem.with_suffix(".keys.npy"), np.zeros(0, dtype=np.uint32))
        np.save(stem.with_suffix(".docs.npy"), np.zeros(0, dtype=np.uint32))
    stem.with_suffix(".eids.txt").write_text("\n".join(eids), encoding="utf-8")
    print(f"  post {Path(path).name} w{wid} docs {len(eids)}", flush=True)


def _score_queries(args: tuple[int, int, list[str], list[list[int]], list[list[int]]]) -> list[list[tuple]]:
    start, end, sids, name_kids, addr_kids = args
    out: list[list[tuple]] = []
    score_buf = np.zeros(N_DOCS, dtype=np.float32)
    for qi in range(start, end):
        chunks_n = []
        for kid in name_kids[qi]:
            a, b = int(OFFSETS[kid]), int(OFFSETS[kid + 1])
            if a == b:
                continue
            idx = POSTINGS[a:b]
            np.add.at(score_buf, idx, IDF[kid])
            chunks_n.append(idx)
        name_top = _top(score_buf, chunks_n, K_NAME)
        chunks_a = []
        for kid in addr_kids[qi]:
            a, b = int(OFFSETS[kid]), int(OFFSETS[kid + 1])
            if a == b:
                continue
            idx = POSTINGS[a:b]
            np.add.at(score_buf, idx, IDF[kid])
            chunks_a.append(idx)
        addr_top = _top(score_buf, chunks_a, K_ADDR)
        meta: dict[str, dict] = {}
        ordered: list[str] = []
        for rank, (doc, score) in enumerate(name_top):
            mid = EIDS[doc]
            meta[mid] = {"ns": score, "nr": rank, "fn": 1}
            ordered.append(mid)
        for rank, (doc, score) in enumerate(addr_top):
            mid = EIDS[doc]
            slot = meta.get(mid)
            if slot is None:
                meta[mid] = {"as": score, "ar": rank, "fa": 1}
                ordered.append(mid)
            else:
                slot["as"] = score
                slot["ar"] = rank
                slot["fa"] = 1
        packed = []
        for mid in ordered:
            item = meta[mid]
            packed.append(
                (
                    mid,
                    float(item.get("nr", 99)),
                    float(item.get("ar", 99)),
                    float(item.get("ns", 0.0)),
                    float(item.get("as", 0.0)),
                    int(item.get("fn", 0)),
                    int(item.get("fa", 0)),
                )
            )
        out.append(packed)
    return out


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


def main() -> None:
    global QUERY, KEY_ID, N_DOCS, OFFSETS, POSTINGS, IDF, EIDS
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = sample_ids()
    ids = set(train_ids) | set(val_ids)
    texts, truths = load_rows(ids)
    ordered = train_ids + val_ids
    for sid in ordered:
        name, addr, country = texts[sid]
        for key in name_keys(name):
            QUERY.add(country + "|n|" + key)
        for key in addr_keys(addr):
            QUERY.add(country + "|a|" + key)
    print(f"query keys {len(QUERY)}", flush=True)

    ctx = mp.get_context("fork")
    jobs = []
    for path in (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv"):
        for wid in range(WORKERS):
            jobs.append((str(path), wid, WORKERS))
    with ctx.Pool(WORKERS) as pool:
        parts = pool.map(_df_worker, jobs)
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for part, seen in parts:
        n_docs += seen
        for key, count in part.items():
            df[key] += count
    print(f"df keys {len(df)} docs {n_docs} in {time.time() - t0:.0f}s", flush=True)

    kept: list[str] = []
    for sid in ordered:
        name, addr, country = texts[sid]
        kn, ka = query_key_lists(
            name, addr, country, df,
            posting_cap=POSTING_CAP,
            max_name_keys=MAX_NAME_KEYS,
            max_addr_keys=MAX_ADDR_KEYS,
        )
        for key in kn + ka:
            if key not in KEY_ID:
                KEY_ID[key] = len(kept)
                kept.append(key)
    idf = np.array([np.log((n_docs + 1) / (df[key] + 1)) for key in kept], dtype=np.float32)
    name_kids = []
    addr_kids = []
    for sid in ordered:
        name, addr, country = texts[sid]
        kn, ka = query_key_lists(
            name, addr, country, df,
            posting_cap=POSTING_CAP,
            max_name_keys=MAX_NAME_KEYS,
            max_addr_keys=MAX_ADDR_KEYS,
        )
        name_kids.append([KEY_ID[key] for key in kn])
        addr_kids.append([KEY_ID[key] for key in ka])
    del df
    print(f"kept {len(kept)}", flush=True)

    post_jobs = []
    for path in (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv"):
        for wid in range(WORKERS):
            post_jobs.append((str(path), wid, WORKERS))
    with ctx.Pool(WORKERS) as pool:
        pool.map(_post_worker, post_jobs)

    eids: list[str] = []
    key_chunks = []
    doc_chunks = []
    for path in (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv"):
        for wid in range(WORKERS):
            stem = PARTS / f"{path.stem}_{wid}"
            local_eids = stem.with_suffix(".eids.txt").read_text(encoding="utf-8").splitlines()
            if local_eids == [""]:
                local_eids = []
            keys = np.load(stem.with_suffix(".keys.npy"))
            docs = np.load(stem.with_suffix(".docs.npy"))
            if len(local_eids) == 0:
                continue
            doc_chunks.append(docs.astype(np.uint32) + len(eids))
            key_chunks.append(keys.astype(np.uint32))
            eids.extend(local_eids)
    key_np = np.concatenate(key_chunks) if key_chunks else np.zeros(0, np.uint32)
    doc_np = np.concatenate(doc_chunks) if doc_chunks else np.zeros(0, np.uint32)
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"indexed {len(eids)} postings {len(postings)}", flush=True)

    N_DOCS = len(eids)
    OFFSETS = offsets
    POSTINGS = postings
    IDF = idf
    EIDS = eids
    # Score in the parent. Forking the posting arrays to workers copies too much.
    packed_all = _score_queries((0, len(ordered), ordered, name_kids, addr_kids))
    print(f"queries scored {len(packed_all)} in {time.time() - t0:.0f}s", flush=True)

    need = {mid for row in packed_all for mid, *_rest in row}
    match_text: dict[str, tuple[str, str, str]] = {}
    for path in (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv"):
        with path.open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid = line.split("\t", 1)[0]
                if eid not in need:
                    continue
                _eid, name, addr, country = line.rstrip("\n").split("\t")
                match_text[eid] = (name, addr, country)
                if len(match_text) == len(need):
                    break
        if len(match_text) == len(need):
            break

    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    rows_sid: list[str] = []
    rows_mid: list[str] = []
    rows_split: list[str] = []
    split_of = {sid: "train" for sid in train_ids}
    split_of.update({sid: "val" for sid in val_ids})
    meta = []
    for sid, packed in zip(ordered, packed_all):
        name, addr, country = texts[sid]
        left = prepare_record(name, addr)
        truth = set(truths.get(sid, []))
        meta.append({"sid": sid, "country": country, "truth": truths.get(sid, []), "split": split_of[sid]})
        for mid, nr, ar, ns, asc, fn, fa in packed:
            mn, ma, mc = match_text.get(mid, ("", "", ""))
            rows_x.append(
                features_prepared(
                    left,
                    prepare_record(mn, ma),
                    country_eq=float(country == mc),
                    name_rank=nr,
                    addr_rank=ar,
                    name_score=ns,
                    addr_score=asc,
                    from_name_channel=fn,
                    from_addr_channel=fa,
                )
            )
            rows_y.append(1 if mid in truth else 0)
            rows_sid.append(sid)
            rows_mid.append(mid)
            rows_split.append(split_of[sid])
    np.savez_compressed(
        OUT / "pairs.npz",
        X=np.asarray(rows_x, dtype=np.float32),
        y=np.asarray(rows_y, dtype=np.int8),
        s1_id=np.asarray(rows_sid),
        match_id=np.asarray(rows_mid),
        split=np.asarray(rows_split),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(
        f"pairs {len(rows_y)} pos {sum(rows_y)} val_s1 {len(val_ids)} "
        f"in {time.time() - t0:.0f}s -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
