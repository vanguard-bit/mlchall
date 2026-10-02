"""Name-channel dedupe and backfill on the validation fold.

Keeps 25 distinct normalized names, filling slots that were repeats.
Address-channel candidates stay as they are. Does not write a submission.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

import build_scoreboard as bs
from decode import decode_greedy_f05
from f05 import macro_f05
from normalize import content_name_tokens
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from v4_features import extra_nine, name_five, name_view, parse_addr

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
POOL = 80
KEEP = 25
MODEL = BOARD / "lgbm_v4.txt"


def squash(name: str) -> str:
    return "".join(content_name_tokens(name))


def load_texts(ids: set[str]) -> dict[str, tuple[str, str, str]]:
    found: dict[str, tuple[str, str, str]] = {}
    if not ids:
        return found
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, addr, country = line.rstrip("\n").split("\t")
                if eid in ids:
                    found[eid] = (bname, addr, country)
                    if len(found) == len(ids):
                        return found
    return found


def name_pool(kids: list[int], buf: np.ndarray) -> list[tuple[str, float]]:
    chunks = []
    for kid in kids:
        start, end = int(bs.OFFSETS[kid]), int(bs.OFFSETS[kid + 1])
        if start == end:
            continue
        idx = bs.POSTINGS[start:end]
        np.add.at(buf, idx, bs.IDF[kid])
        chunks.append(idx)
    if not chunks:
        return []
    seen = np.unique(np.concatenate(chunks))
    scores = buf[seen]
    buf[seen] = 0
    n = int(scores.shape[0])
    if n > POOL:
        pick = np.argpartition(scores, -POOL)[-POOL:]
        pick = pick[np.argsort(scores[pick])[::-1]]
    else:
        pick = np.argsort(scores)[::-1]
    return [(bs.EIDS[int(seen[i])], float(scores[i])) for i in pick]


def dedupe(ranked: list[tuple[str, float]], names: dict[str, str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for eid, _score in ranked:
        key = squash(names.get(eid, ("",))[0] if isinstance(names.get(eid), tuple) else names.get(eid, ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        kept.append(eid)
        if len(kept) == KEEP:
            break
    return kept


def main() -> None:
    t0 = time.time()
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_rows = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_rows}
    countries = {row["sid"]: row["country"] for row in val_rows}
    ordered = [row["sid"] for row in val_rows]
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    feat = [str(x) for x in data["feature_names"]]
    fa_col = 0
    split = data["split"].astype(str) == "val"
    addr_of: dict[str, set[str]] = defaultdict(set)
    current: dict[str, set[str]] = defaultdict(set)
    stored: dict[str, dict[str, tuple[float, float, float, float, int, int]]] = defaultdict(dict)
    feat_names = [str(x) for x in data["feature_names"]]
    cols = {name: feat_names.index(name) for name in (
        "from_addr_ch", "from_name_ch", "name_rank", "addr_rank", "name_block_score", "addr_block_score"
    )}
    xv = data["X"][split]
    for i, (sid, eid) in enumerate(zip(
        data["s1_id"].astype(str)[split].tolist(),
        data["match_id"].astype(str)[split].tolist(),
    )):
        current[sid].add(eid)
        if xv[i, cols["from_addr_ch"]] >= 0.5:
            addr_of[sid].add(eid)
        stored[sid][eid] = (
            float(xv[i, cols["name_rank"]]),
            float(xv[i, cols["addr_rank"]]),
            float(xv[i, cols["name_block_score"]]),
            float(xv[i, cols["addr_block_score"]]),
            int(xv[i, cols["from_name_ch"]] >= 0.5),
            int(xv[i, cols["from_addr_ch"]] >= 0.5),
        )
    del data
    s1_text = load_texts(set(ordered))
    print(f"val s1 {len(ordered)} texts {len(s1_text)}", flush=True)

    bs.PARTS = BOARD / "dedupe_parts"
    bs.PARTS.mkdir(parents=True, exist_ok=True)
    bs.QUERY = set()
    bs.KEY_ID = {}
    for sid in ordered:
        name, _addr, country = s1_text[sid]
        for key in bs.name_keys(name):
            bs.QUERY.add(country + "|n|" + key)
    print(f"name query keys {len(bs.QUERY)}", flush=True)
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
    kept_keys: list[str] = []
    name_kids: list[list[int]] = []
    for sid in ordered:
        name, addr, country = s1_text[sid]
        kn, _ka = bs.query_key_lists(name, addr, country, df)
        for key in kn:
            if key not in bs.KEY_ID:
                bs.KEY_ID[key] = len(kept_keys)
                kept_keys.append(key)
        name_kids.append([bs.KEY_ID[key] for key in kn])
    idf = np.array([np.log((n_docs + 1) / (df[key] + 1)) for key in kept_keys], dtype=np.float32)
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
            if not local:
                continue
            docs = np.load(stem.with_suffix(".docs.npy"))
            keys = np.load(stem.with_suffix(".keys.npy"))
            doc_chunks.append(docs.astype(np.uint32) + len(eids))
            key_chunks.append(keys.astype(np.uint32))
            eids.extend(local)
    key_np = np.concatenate(key_chunks) if key_chunks else np.zeros(0, np.uint32)
    doc_np = np.concatenate(doc_chunks) if doc_chunks else np.zeros(0, np.uint32)
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept_keys))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    bs.N_DOCS = len(eids)
    bs.OFFSETS = offsets
    bs.POSTINGS = postings
    bs.IDF = idf
    bs.EIDS = eids
    print(f"indexed {len(eids)} postings {len(postings)}", flush=True)
    buf = np.zeros(len(eids), dtype=np.float32)
    pools: dict[str, list[tuple[str, float]]] = {}
    need: set[str] = set()
    for sid, kids in zip(ordered, name_kids):
        ranked = name_pool(kids, buf)
        pools[sid] = ranked
        for eid, _score in ranked:
            need.add(eid)
    del postings, key_np, doc_np, buf, bs.POSTINGS, bs.EIDS, bs.OFFSETS
    gc.collect()
    print(f"queried {len(pools)} in {time.time() - t0:.0f}s, loading {len(need)} names", flush=True)
    noisy = load_texts(need)
    names = {eid: rec[0] for eid, rec in noisy.items()}

    base_hit = new_hit = total = 0
    recovered = still_out = collapsed = 0
    raw_bins = defaultdict(int)
    sizes = []
    new_cands: dict[str, set[str]] = {}
    for sid in ordered:
        truth = truths[sid]
        total += len(truth)
        old = current.get(sid, set())
        base_hit += len(truth & old)
        ranked = pools.get(sid, [])
        rank_of = {eid: i + 1 for i, (eid, _s) in enumerate(ranked)}
        deduped = dedupe(ranked, names)
        fresh = set(deduped) | addr_of.get(sid, set())
        new_cands[sid] = fresh
        new_hit += len(truth & fresh)
        sizes.append(len(fresh))
        for eid in truth - old:
            rank = rank_of.get(eid)
            if rank is None:
                raw_bins[">80 or absent"] += 1
            elif rank <= 25:
                raw_bins["1-25"] += 1
            elif rank <= 40:
                raw_bins["26-40"] += 1
            elif rank <= 60:
                raw_bins["41-60"] += 1
            else:
                raw_bins["61-80"] += 1
            if eid in fresh:
                recovered += 1
            elif eid in {e for e, _s in ranked}:
                collapsed += 1
            else:
                still_out += 1
    print(f"true {total} current recall {base_hit / total:.4f} deduped recall {new_hit / total:.4f}", flush=True)
    print(f"misses recovered {recovered} collapsed_away {collapsed} still_out {still_out}", flush=True)
    print("raw rank of current misses " + " ".join(f"{k}={raw_bins[k]}" for k in ("1-25", "26-40", "41-60", "61-80", ">80 or absent")), flush=True)
    print(f"mean candidates {sum(sizes) / len(sizes):.1f}", flush=True)

    # Score the union with the saved v4 matcher.
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.num_trees()
    all_ids = set(ordered) | {eid for s in new_cands.values() for eid in s}
    missing = all_ids - set(noisy) - set(s1_text)
    if missing:
        noisy.update(load_texts(missing))
    texts = dict(s1_text)
    texts.update(noisy)
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    empty_rec = prepare_record("", "")
    views = {eid: name_view(texts.get(eid, ("", "", ""))[0]) for eid in all_ids}
    addrs = {eid: parse_addr(texts.get(eid, ("", "", ""))[1]) for eid in all_ids}
    prepared = {eid: prepare_record(texts.get(eid, ("", "", ""))[0], texts.get(eid, ("", "", ""))[1]) for eid in all_ids}
    preds: list[set[str]] = []
    gold: list[set[str]] = []
    for sid in ordered:
        truth = truths[sid]
        gold.append(truth)
        left_name = views[sid]
        left_addr = addrs[sid]
        left = prepared[sid]
        ctry = countries[sid]
        mids = list(new_cands[sid])
        rank_of = {eid: i + 1 for i, eid in enumerate(dedupe(pools.get(sid, []), names))}
        score_of = {eid: sc for eid, sc in pools.get(sid, [])}
        rows = []
        for eid in mids:
            known = stored[sid].get(eid)
            if known is None:
                base = features_prepared(
                    left,
                    prepared.get(eid, empty_rec),
                    country_eq=float(ctry == texts.get(eid, ("", "", ""))[2]),
                    name_rank=float(rank_of.get(eid, 99)),
                    addr_rank=99.0,
                    name_score=float(score_of.get(eid, 0.0)),
                    from_name_channel=1,
                )
            else:
                nr, ar, ns, asc, fn, fa = known
                base = features_prepared(
                    left,
                    prepared.get(eid, empty_rec),
                    country_eq=float(ctry == texts.get(eid, ("", "", ""))[2]),
                    name_rank=nr,
                    addr_rank=ar,
                    name_score=ns,
                    addr_score=asc,
                    from_name_channel=fn,
                    from_addr_channel=fa,
                )
            rows.append(name_five(left_name, views.get(eid, empty_name)) + base[5:] + extra_nine(
                left_addr, addrs.get(eid, empty_addr), left_name, views.get(eid, empty_name)
            ))
        if not rows:
            preds.append(set())
            continue
        scores = booster.predict(np.asarray(rows, dtype=np.float32), num_iteration=trees)
        chosen = decode_greedy_f05(mids, [float(s) for s in scores], min_gain=0.70, max_preds=12)
        preds.append(chosen)
    print(f"deduped f05 {macro_f05(gold, preds):.4f} v4 baseline 0.9021 in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
