"""Add 24k more training Source 1 rows. The existing 6k validation ids stay fixed."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict

import numpy as np

import build_scoreboard as bs
from build_scoreboard import _bucket
from pair_features import FEATURE_NAMES, features_prepared, prepare_record
from paths import DATA_DIR

EXTRA = 24_000
SEED = 11
BOARD = DATA_DIR / "scoreboard"
TRAIN = DATA_DIR / "train"


def sample_extra(exclude: set[str]) -> list[str]:
    countries: dict[str, str] = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, _name, _addr, country = line.rstrip("\n").split("\t")
            if eid not in exclude:
                countries[eid] = country
    buckets: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            if sid in exclude:
                continue
            n_true = 0 if not rest else len([x for x in rest.split(",") if x])
            buckets[_bucket(countries[sid], n_true)].append(sid)
    total = sum(len(ids) for ids in buckets.values())
    rng = random.Random(SEED)
    chosen: list[str] = []
    for ids in buckets.values():
        take = min(len(ids), int(round(EXTRA * len(ids) / total)))
        if take:
            chosen.extend(rng.sample(ids, take))
    rng.shuffle(chosen)
    chosen = chosen[:EXTRA]
    print(f"extra train {len(chosen)} excluded {len(exclude)}", flush=True)
    return chosen


def block(train_ids: list[str]) -> None:
    t0 = time.time()
    ids = set(train_ids)
    texts, truths = bs.load_rows(ids)
    bs.PARTS = BOARD / "extra_parts"
    bs.PARTS.mkdir(parents=True, exist_ok=True)
    bs.QUERY = set()
    bs.KEY_ID = {}
    for sid in train_ids:
        name, addr, country = texts[sid]
        for key in bs.name_keys(name):
            bs.QUERY.add(country + "|n|" + key)
        for key in bs.addr_keys(addr):
            bs.QUERY.add(country + "|a|" + key)
    print(f"query keys {len(bs.QUERY)}", flush=True)
    ctx = mp.get_context("fork")
    jobs = [
        (str(path), wid, bs.WORKERS)
        for path in (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv")
        for wid in range(bs.WORKERS)
    ]
    with ctx.Pool(bs.WORKERS) as pool:
        parts = pool.map(bs._df_worker, jobs)
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for part, seen in parts:
        n_docs += seen
        for key, count in part.items():
            df[key] += count
    print(f"df keys {len(df)} docs {n_docs} in {time.time() - t0:.0f}s", flush=True)
    kept: list[str] = []
    name_kids = []
    addr_kids = []
    for sid in train_ids:
        name, addr, country = texts[sid]
        kn, ka = bs.query_key_lists(
            name, addr, country, df,
            posting_cap=bs.POSTING_CAP,
            max_name_keys=bs.MAX_NAME_KEYS,
            max_addr_keys=bs.MAX_ADDR_KEYS,
        )
        for key in kn + ka:
            if key not in bs.KEY_ID:
                bs.KEY_ID[key] = len(kept)
                kept.append(key)
        name_kids.append([bs.KEY_ID[key] for key in kn])
        addr_kids.append([bs.KEY_ID[key] for key in ka])
    idf = np.array([np.log((n_docs + 1) / (df[key] + 1)) for key in kept], dtype=np.float32)
    del df
    gc.collect()
    with ctx.Pool(bs.WORKERS) as pool:
        pool.map(bs._post_worker, jobs)
    eids: list[str] = []
    key_chunks = []
    doc_chunks = []
    for path in (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv"):
        for wid in range(bs.WORKERS):
            stem = bs.PARTS / f"{path.stem}_{wid}"
            local = stem.with_suffix(".eids.txt").read_text(encoding="utf-8").splitlines()
            if local == [""]:
                local = []
            keys = np.load(stem.with_suffix(".keys.npy"))
            docs = np.load(stem.with_suffix(".docs.npy"))
            if not local:
                continue
            doc_chunks.append(docs.astype(np.uint32) + len(eids))
            key_chunks.append(keys.astype(np.uint32))
            eids.extend(local)
    key_np = np.concatenate(key_chunks) if key_chunks else np.zeros(0, np.uint32)
    doc_np = np.concatenate(doc_chunks) if doc_chunks else np.zeros(0, np.uint32)
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"indexed {len(eids)} postings {len(postings)}", flush=True)
    bs.N_DOCS = len(eids)
    bs.OFFSETS = offsets
    bs.POSTINGS = postings
    bs.IDF = idf
    bs.EIDS = eids
    packed_all: list = []
    step = 2000
    for start in range(0, len(train_ids), step):
        packed_all.extend(
            bs._score_queries((start, min(start + step, len(train_ids)), train_ids, name_kids, addr_kids))
        )
        print(f"  queried {len(packed_all)}", flush=True)
    del postings, key_np, doc_np, bs.POSTINGS, bs.OFFSETS, bs.EIDS
    gc.collect()
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
    rows_x = []
    rows_y = []
    rows_sid = []
    rows_mid = []
    meta = []
    for sid, packed in zip(train_ids, packed_all):
        name, addr, country = texts[sid]
        left = prepare_record(name, addr)
        truth = set(truths.get(sid, []))
        meta.append({"sid": sid, "country": country, "truth": truths.get(sid, []), "split": "train"})
        for mid, nr, ar, ns, asc, fn, fa in packed:
            mn, ma, mc = match_text.get(mid, ("", "", ""))
            rows_x.append(
                features_prepared(
                    left, prepare_record(mn, ma),
                    country_eq=float(country == mc),
                    name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                    from_name_channel=fn, from_addr_channel=fa,
                )
            )
            rows_y.append(1 if mid in truth else 0)
            rows_sid.append(sid)
            rows_mid.append(mid)
    np.savez_compressed(
        BOARD / "extra_pairs.npz",
        X=np.asarray(rows_x, dtype=np.float32),
        y=np.asarray(rows_y, dtype=np.int8),
        s1_id=np.asarray(rows_sid),
        match_id=np.asarray(rows_mid),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    (BOARD / "extra_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"extra pairs {len(rows_y)} pos {sum(rows_y)} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    exclude = {row["sid"] for row in meta}
    extra_ids = sample_extra(exclude)
    if set(extra_ids) & exclude:
        raise SystemExit("extra sample overlaps the held-out ids")
    block(extra_ids)


if __name__ == "__main__":
    main()
