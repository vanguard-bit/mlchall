"""Build labeled pair dataset from blocked candidates on a train S1 sample."""

from __future__ import annotations

import random
import sys
import time
import zipfile
from array import array
from collections import defaultdict

import numpy as np

from blocking import BlockerConfig, addr_keys, name_keys, query_key_lists
from pair_features import FEATURE_NAMES, pair_feature_dict
from paths import DATA_DIR, TRAIN_PREFIX, ZIP_PATH

SAMPLE_N = 1200
SEED = 1
OUT_PATH = DATA_DIR / "matcher_sample.npz"


def reservoir_sample(zf: zipfile.ZipFile) -> list[tuple[str, list[str]]]:
    rng = random.Random(SEED)
    sample: list[tuple[str, list[str]]] = []
    seen = 0
    with zf.open(f"{TRAIN_PREFIX}/train_ground_truth.tsv") as f:
        f.readline()
        for line in f:
            seen += 1
            parts = line.decode().rstrip("\n").split("\t")
            sid = parts[0]
            mids = [x for x in parts[1].split(",") if x] if len(parts) > 1 else []
            if len(sample) < SAMPLE_N:
                sample.append((sid, mids))
            else:
                j = rng.randrange(seen)
                if j < SAMPLE_N:
                    sample[j] = (sid, mids)
    return sample


def load_s1(zf: zipfile.ZipFile, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    with zf.open(f"{TRAIN_PREFIX}/train_source1.tsv") as f:
        f.readline()
        for line in f:
            parts = line.decode().rstrip("\n").split("\t")
            if len(parts) != 4 or parts[0] not in wanted:
                continue
            out[parts[0]] = (parts[1], parts[2], parts[3])
            if len(out) == len(wanted):
                break
    return out


def iter_corpus(zf: zipfile.ZipFile):
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as f:
            f.readline()
            for line in f:
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                yield parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    import math

    t0 = time.time()
    cfg = BlockerConfig(k_name=25, k_addr=25)
    zf = zipfile.ZipFile(ZIP_PATH)
    sample = reservoir_sample(zf)
    wanted = {sid for sid, _ in sample}
    s1 = load_s1(zf, wanted)
    print(f"sample {len(sample)} s1 {len(s1)}", flush=True)

    key_to_q: dict[str, list[int]] = defaultdict(list)
    q_meta: list[tuple[str, str, str, set[str]]] = []
    for i, (sid, mids) in enumerate(sample):
        name, addr, country = s1[sid]
        q_meta.append((name, addr, country, set(mids)))
        for k in name_keys(name):
            key_to_q[country + "|n|" + k].append(i)
        for k in addr_keys(addr):
            key_to_q[country + "|a|" + k].append(i)

    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    print("pass1 df", flush=True)
    for _eid, name, addr, country in iter_corpus(zf):
        n_docs += 1
        prefix_n, prefix_a = country + "|n|", country + "|a|"
        for k in name_keys(name):
            full = prefix_n + k
            if full in key_to_q:
                df[full] += 1
        for k in addr_keys(addr):
            full = prefix_a + k
            if full in key_to_q:
                df[full] += 1
        if n_docs % 2_000_000 == 0:
            print(f"  docs {n_docs}", flush=True)

    kept_keys: set[str] = set()
    q_kept_n: list[list[str]] = []
    q_kept_a: list[list[str]] = []
    for name, addr, country, _ in q_meta:
        kn, ka = query_key_lists(name, addr, country, df, posting_cap=cfg.posting_cap)
        q_kept_n.append(kn)
        q_kept_a.append(ka)
        kept_keys.update(kn)
        kept_keys.update(ka)
    print(f"kept keys {len(kept_keys)}", flush=True)

    postings: dict[str, array] = {k: array("I") for k in kept_keys}
    id_of: list[str] = []
    id_index: dict[str, int] = {}

    def doc_id(eid: str) -> int:
        d = id_index.get(eid)
        if d is None:
            d = len(id_of)
            id_index[eid] = d
            id_of.append(eid)
        return d

    print("pass2 postings", flush=True)
    n_docs = 0
    for eid, name, addr, country in iter_corpus(zf):
        n_docs += 1
        hits: list[str] = []
        prefix_n, prefix_a = country + "|n|", country + "|a|"
        for k in name_keys(name):
            full = prefix_n + k
            if full in kept_keys:
                hits.append(full)
        for k in addr_keys(addr):
            full = prefix_a + k
            if full in kept_keys:
                hits.append(full)
        if hits:
            d = doc_id(eid)
            for full in hits:
                postings[full].append(d)
        if n_docs % 2_000_000 == 0:
            print(f"  docs {n_docs} indexed {len(id_of)}", flush=True)

    idf = {
        full: (math.log((n_docs + 1) / (len(arr) + 1)) if arr else 0.0)
        for full, arr in postings.items()
    }

    per_q_candidates: list[list[str]] = []
    q_cand_meta: list[dict[str, dict]] = []
    need_eids: set[str] = set()
    for qi, (name, addr, country, _) in enumerate(q_meta):
        ns: dict[int, float] = defaultdict(float)
        asc: dict[int, float] = defaultdict(float)
        for full in q_kept_n[qi]:
            w = idf.get(full, 0.0)
            for d in postings[full]:
                ns[d] += w
        for full in q_kept_a[qi]:
            w = idf.get(full, 0.0)
            for d in postings[full]:
                asc[d] += w

        def top(score: dict[int, float], k: int) -> list[tuple[str, float, int]]:
            items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
            return [(id_of[d], sc, rank) for rank, (d, sc) in enumerate(items)]

        nt = top(ns, cfg.k_name)
        at = top(asc, cfg.k_addr)
        meta: dict[str, dict] = {}
        for e, sc, rk in nt:
            meta[e] = {"name_score": sc, "name_rank": rk, "from_name": 1}
        for e, sc, rk in at:
            if e in meta:
                meta[e]["addr_score"] = sc
                meta[e]["addr_rank"] = rk
                meta[e]["from_addr"] = 1
            else:
                meta[e] = {"addr_score": sc, "addr_rank": rk, "from_addr": 1}
        cands = list(dict.fromkeys([x[0] for x in nt] + [x[0] for x in at]))
        for e in cands:
            need_eids.add(e)
        per_q_candidates.append(cands)
        q_cand_meta.append(meta)

    print(f"unique candidate ids {len(need_eids)}", flush=True)
    texts: dict[str, tuple[str, str, str]] = {}
    for eid, name, addr, country in iter_corpus(zf):
        if eid in need_eids:
            texts[eid] = (name, addr, country)
            if len(texts) == len(need_eids):
                break

    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    rows_sid: list[str] = []
    rows_mid: list[str] = []
    pos = neg = 0
    for qi, (sid, mids) in enumerate(sample):
        name, addr, country, true = q_meta[qi]
        meta = q_cand_meta[qi]
        for mid in per_q_candidates[qi]:
            mn, ma, mc = texts.get(mid, ("", "", ""))
            m = meta.get(mid, {})
            fd = pair_feature_dict(
                name,
                addr,
                country,
                mn,
                ma,
                mc,
                name_rank=float(m.get("name_rank", 99)),
                addr_rank=float(m.get("addr_rank", 99)),
                name_score=float(m.get("name_score", 0)),
                addr_score=float(m.get("addr_score", 0)),
                from_name_channel=int(m.get("from_name", 0)),
                from_addr_channel=int(m.get("from_addr", 0)),
            )
            y = 1 if mid in true else 0
            rows_x.append([fd[k] for k in FEATURE_NAMES])
            rows_y.append(y)
            rows_sid.append(sid)
            rows_mid.append(mid)
            if y:
                pos += 1
            else:
                neg += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X=np.array(rows_x, dtype=np.float32),
        y=np.array(rows_y, dtype=np.int8),
        s1_id=np.array(rows_sid),
        match_id=np.array(rows_mid),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"pairs {len(rows_y)} pos {pos} neg {neg} -> {OUT_PATH}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
