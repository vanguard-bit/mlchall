"""Block the full test set, score it with the saved LightGBM model, write submission TSVs.

Retrieval matches the training blocker: same-country name and address indexes,
top 25 per channel. The decision threshold is the one tuned on the training
sample (min score 0.60).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
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
from pair_features import FEATURE_NAMES, features_prepared, prepare_record
from paths import DATA_DIR, ROOT

K_NAME = 25
K_ADDR = 25
MIN_GAIN = float(os.environ.get("MLCHALL_MIN_GAIN", "0.60"))
MAX_PREDS = 12
BATCH_S1 = 20_000

TEST_DIR = DATA_DIR / "test"
EXPORT = Path(os.environ.get("MLCHALL_EXPORT", str(DATA_DIR / "export")))
OUT_DIR = Path(os.environ.get("MLCHALL_OUT", str(ROOT / "output")))
MODEL_PATH = Path(os.environ.get("MLCHALL_MODEL", str(DATA_DIR / "lgbm_matcher.txt")))
VALIDATOR = TEST_DIR / "validate_submission.py"


def rss_gb() -> float:
    rss_kb = 0
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
                break
    return rss_kb / (1024 * 1024)


def parse_line(line: str) -> tuple[str, str, str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def split_corpus() -> dict:
    meta_path = EXPORT / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    EXPORT.mkdir(parents=True, exist_ok=True)
    handles: dict[tuple[str, str], object] = {}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"s1": 0, "s23": 0})

    def out_file(country: str, kind: str):
        key = (country, kind)
        if key not in handles:
            directory = EXPORT / country
            directory.mkdir(parents=True, exist_ok=True)
            handles[key] = open(directory / f"{kind}.tsv", "w", encoding="utf-8")
        return handles[key]

    def pump(path: Path, kind: str) -> int:
        n = 0
        bad = 0
        with path.open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                row = parse_line(line)
                if row is None:
                    bad += 1
                    continue
                country = row[3]
                out_file(country, kind).write(line if line.endswith("\n") else line + "\n")
                counts[country][kind] += 1
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  split {path.name} {n}", flush=True)
        if bad:
            print(f"  {path.name} skipped {bad} malformed rows", flush=True)
        return n

    print("splitting test files by country", flush=True)
    n1 = pump(TEST_DIR / "test_source1.tsv", "s1")
    n23 = pump(TEST_DIR / "test_source2.tsv", "s23")
    n23 += pump(TEST_DIR / "test_source3.tsv", "s23")
    for handle in handles.values():
        handle.close()

    meta = {"n_s1": n1, "n_docs": n23, "countries": dict(counts)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"split done s1={n1} s23={n23} countries={list(counts)}", flush=True)
    return meta


def _pack_row(cands: list[tuple]) -> str:
    return ";".join(
        f"{mid}|{nr:.0f}|{ar:.0f}|{ns:.6f}|{asc:.6f}|{fn}|{fa}"
        for mid, nr, ar, ns, asc, fn, fa in cands
    )


def _parse_row(payload: str) -> list[tuple]:
    if not payload:
        return []
    parsed = []
    for part in payload.split(";"):
        mid, nr, ar, ns, asc, fn, fa = part.split("|")
        parsed.append((mid, float(nr), float(ar), float(ns), float(asc), int(fn), int(fa)))
    return parsed


def _line_count(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as handle:
        for _ in handle:
            n += 1
    return n


def retrieve_country(country: str, n_s1: int, n_docs: int) -> None:
    directory = EXPORT / country
    rows_path = directory / "rows.tsv"
    if (directory / "done").exists():
        print(f"{country}: retrieval cached (already scored)", flush=True)
        return
    if rows_path.exists() and _line_count(rows_path) == n_s1:
        print(f"{country}: retrieval cached ({n_s1} rows)", flush=True)
        return

    t0 = time.time()
    s1_path = directory / "s1.tsv"
    s23_path = directory / "s23.tsv"
    print(f"{country}: query keys  rss={rss_gb():.2f}GB", flush=True)
    query_keys: set[str] = set()
    with s1_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = parse_line(line)
            if row is None:
                continue
            _eid, name, addr, cty = row
            prefix_n = cty + "|n|"
            prefix_a = cty + "|a|"
            for key in name_keys(name):
                query_keys.add(prefix_n + key)
            for key in addr_keys(addr):
                query_keys.add(prefix_a + key)
    print(
        f"{country}: {len(query_keys)} query keys in {time.time() - t0:.0f}s "
        f"rss={rss_gb():.2f}GB",
        flush=True,
    )

    df: dict[str, int] = defaultdict(int)
    seen_docs = 0
    with s23_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = parse_line(line)
            if row is None:
                continue
            _eid, name, addr, cty = row
            seen_docs += 1
            prefix_n = cty + "|n|"
            prefix_a = cty + "|a|"
            for key in name_keys(name):
                full = prefix_n + key
                if full in query_keys:
                    df[full] += 1
            for key in addr_keys(addr):
                full = prefix_a + key
                if full in query_keys:
                    df[full] += 1
            if seen_docs % 500_000 == 0:
                print(f"  {country} df docs {seen_docs} rss={rss_gb():.2f}GB", flush=True)
    del query_keys
    print(
        f"{country}: df keys {len(df)} docs {seen_docs} in {time.time() - t0:.0f}s",
        flush=True,
    )

    kept = [key for key, count in df.items() if 0 < count <= POSTING_CAP]
    key_id = {key: i for i, key in enumerate(kept)}
    idf = np.array(
        [math.log((n_docs + 1) / (df[key] + 1)) for key in kept],
        dtype=np.float32,
    )
    print(f"{country}: kept keys {len(kept)}", flush=True)

    s1_ids: list[str] = []
    name_kids: list[list[int]] = []
    addr_kids: list[list[int]] = []
    with s1_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = parse_line(line)
            if row is None:
                continue
            eid, name, addr, cty = row
            kn, ka = query_key_lists(
                name,
                addr,
                cty,
                df,
                posting_cap=POSTING_CAP,
                max_name_keys=MAX_NAME_KEYS,
                max_addr_keys=MAX_ADDR_KEYS,
            )
            s1_ids.append(eid)
            name_kids.append([key_id[key] for key in kn if key in key_id])
            addr_kids.append([key_id[key] for key in ka if key in key_id])
    if len(s1_ids) != n_s1:
        raise RuntimeError(f"{country}: expected {n_s1} S1 rows, got {len(s1_ids)}")
    del df
    gc.collect()
    print(f"{country}: translated queries rss={rss_gb():.2f}GB", flush=True)

    key_ids_arr = array("I")
    doc_ids_arr = array("I")
    id_of: list[str] = []
    id_index: dict[str, int] = {}
    with s23_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = parse_line(line)
            if row is None:
                continue
            eid, name, addr, cty = row
            hits: list[int] = []
            prefix_n = cty + "|n|"
            prefix_a = cty + "|a|"
            for key in name_keys(name):
                kid = key_id.get(prefix_n + key)
                if kid is not None:
                    hits.append(kid)
            for key in addr_keys(addr):
                kid = key_id.get(prefix_a + key)
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
                key_ids_arr.append(kid)
                doc_ids_arr.append(doc)
            if line_no % 500_000 == 0:
                print(
                    f"  {country} postings lines {line_no} docs {len(id_of)} "
                    f"rss={rss_gb():.2f}GB",
                    flush=True,
                )
    del id_index
    del key_id
    gc.collect()
    print(
        f"{country}: indexed {len(id_of)} docs, {len(doc_ids_arr)} postings "
        f"rss={rss_gb():.2f}GB",
        flush=True,
    )

    key_np = np.frombuffer(key_ids_arr, dtype=np.uint32, count=len(key_ids_arr)).copy()
    doc_np = np.frombuffer(doc_ids_arr, dtype=np.uint32, count=len(doc_ids_arr)).copy()
    del key_ids_arr
    del doc_ids_arr
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    del doc_np
    counts = np.bincount(key_np, minlength=len(kept))
    del key_np
    del kept
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del counts
    gc.collect()

    n_indexed = len(id_of)
    score_buf = np.zeros(n_indexed, dtype=np.float32)
    tmp_path = directory / "rows.tsv.tmp"

    def channel_top(kids: list[int], k: int) -> list[tuple[int, float]]:
        chunks: list[np.ndarray] = []
        for kid in kids:
            start = int(offsets[kid])
            end = int(offsets[kid + 1])
            if start == end:
                continue
            idx = postings[start:end]
            np.add.at(score_buf, idx, idf[kid])
            chunks.append(idx)
        if not chunks:
            return []
        seen = np.unique(np.concatenate(chunks))
        try:
            scores = score_buf[seen]
            n = int(scores.shape[0])
            if n > k:
                pick = np.argpartition(scores, -k)[-k:]
                pick = pick[np.argsort(scores[pick])[::-1]]
            else:
                pick = np.argsort(scores)[::-1]
            return [(int(seen[i]), float(scores[i])) for i in pick]
        finally:
            score_buf[seen] = 0

    with tmp_path.open("w", encoding="utf-8") as out:
        for qi, sid in enumerate(s1_ids):
            name_top = channel_top(name_kids[qi], K_NAME)
            addr_top = channel_top(addr_kids[qi], K_ADDR)
            meta: dict[str, dict] = {}
            ordered: list[str] = []
            for rank, (doc, score) in enumerate(name_top):
                mid = id_of[doc]
                meta[mid] = {"name_score": score, "name_rank": rank, "from_name": 1}
                ordered.append(mid)
            for rank, (doc, score) in enumerate(addr_top):
                mid = id_of[doc]
                if mid in meta:
                    meta[mid]["addr_score"] = score
                    meta[mid]["addr_rank"] = rank
                    meta[mid]["from_addr"] = 1
                else:
                    meta[mid] = {"addr_score": score, "addr_rank": rank, "from_addr": 1}
                    ordered.append(mid)
            packed = []
            for mid in ordered:
                item = meta[mid]
                packed.append(
                    (
                        mid,
                        float(item.get("name_rank", 99)),
                        float(item.get("addr_rank", 99)),
                        float(item.get("name_score", 0.0)),
                        float(item.get("addr_score", 0.0)),
                        int(item.get("from_name", 0)),
                        int(item.get("from_addr", 0)),
                    )
                )
            out.write(f"{sid}\t{_pack_row(packed)}\n")
            if (qi + 1) % 50_000 == 0:
                print(f"  {country} queried {qi + 1} rss={rss_gb():.2f}GB", flush=True)
    tmp_path.replace(rows_path)
    print(f"{country}: retrieval done in {time.time() - t0:.0f}s rss={rss_gb():.2f}GB", flush=True)


def _flush_predictions(
    booster,
    pending: list[tuple[str, list[str]]],
    rows_x: list[list[float]],
    match_out,
    cand_out,
) -> None:
    if not pending:
        return
    scores = booster.predict(np.asarray(rows_x, dtype=np.float32), num_iteration=booster.best_iteration)
    cursor = 0
    for sid, mids in pending:
        n = len(mids)
        part = scores[cursor : cursor + n]
        cursor += n
        chosen = decode_greedy_f05(mids, [float(s) for s in part], min_gain=MIN_GAIN, max_preds=MAX_PREDS)
        matched = [mid for mid in mids if mid in chosen]
        match_out.write(f"{sid}\t{','.join(matched)}\n")
        cand_out.write(f"{sid}\t{','.join(mids)}\n")
    pending.clear()
    rows_x.clear()


def score_country(country: str, n_s1: int) -> None:
    directory = EXPORT / country
    done = directory / "done"
    if done.exists():
        print(f"{country}: score cached", flush=True)
        return

    import lightgbm as lgb

    t0 = time.time()
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    print(f"{country}: model iter {booster.best_iteration} features {len(FEATURE_NAMES)}", flush=True)

    s1_text: dict[str, tuple[str, str, str]] = {}
    with (directory / "s1.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            row = parse_line(line)
            if row is None:
                continue
            s1_text[row[0]] = (row[1], row[2], row[3])

    match_path = directory / "matching.tsv"
    cand_path = directory / "candidates.tsv"
    written = 0
    with (
        (directory / "rows.tsv").open(encoding="utf-8") as rows_in,
        match_path.open("w", encoding="utf-8") as match_out,
        cand_path.open("w", encoding="utf-8") as cand_out,
    ):
        batch: list[str] = []

        def consume(lines: list[str]) -> None:
            nonlocal written
            parsed: list[tuple[str, list[tuple]]] = []
            need: set[str] = set()
            for line in lines:
                sid, _, payload = line.rstrip("\n").partition("\t")
                cands = _parse_row(payload)
                parsed.append((sid, cands))
                for cand in cands:
                    need.add(cand[0])
            texts: dict[str, tuple] = {}
            if need:
                with (directory / "s23.tsv").open(encoding="utf-8") as corpus:
                    for line in corpus:
                        eid = line.split("\t", 1)[0]
                        if eid not in need:
                            continue
                        row = parse_line(line)
                        if row is None:
                            continue
                        texts[eid] = (prepare_record(row[1], row[2]), row[3])
                        if len(texts) == len(need):
                            break
            pending: list[tuple[str, list[str]]] = []
            rows_x: list[list[float]] = []
            for sid, cands in parsed:
                name, addr, cty = s1_text[sid]
                left = prepare_record(name, addr)
                mids: list[str] = []
                if not cands:
                    match_out.write(f"{sid}\t\n")
                    cand_out.write(f"{sid}\t\n")
                    continue
                for mid, nr, ar, ns, asc, fn, fa in cands:
                    prepared = texts.get(mid)
                    if prepared is None:
                        right = prepare_record("", "")
                        m_cty = ""
                    else:
                        right, m_cty = prepared
                    rows_x.append(
                        features_prepared(
                            left,
                            right,
                            country_eq=float(cty == m_cty),
                            name_rank=nr,
                            addr_rank=ar,
                            name_score=ns,
                            addr_score=asc,
                            from_name_channel=fn,
                            from_addr_channel=fa,
                        )
                    )
                    mids.append(mid)
                pending.append((sid, mids))
                if len(rows_x) >= 80_000:
                    _flush_predictions(booster, pending, rows_x, match_out, cand_out)
            _flush_predictions(booster, pending, rows_x, match_out, cand_out)
            written += len(parsed)
            print(f"  {country} scored {written} rss={rss_gb():.2f}GB", flush=True)

        for line in rows_in:
            batch.append(line)
            if len(batch) >= BATCH_S1:
                consume(batch)
                batch = []
        if batch:
            consume(batch)

    if written != n_s1:
        raise RuntimeError(f"{country}: scored {written} rows, expected {n_s1}")
    done.write_text(f"{written}\n", encoding="utf-8")
    for name in ("s1.tsv", "s23.tsv", "rows.tsv"):
        path = directory / name
        if path.exists():
            path.unlink()
    print(f"{country}: score done in {time.time() - t0:.0f}s", flush=True)


def merge_outputs(meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    match_path = OUT_DIR / "matching_results.tsv"
    cand_path = OUT_DIR / "candidate_pairs.tsv"
    n = 0
    with (
        match_path.open("w", encoding="utf-8") as match_out,
        cand_path.open("w", encoding="utf-8") as cand_out,
    ):
        match_out.write("source1_entity_id\tmatched_entity_ids\n")
        cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in meta["countries"]:
            with (EXPORT / country / "matching.tsv").open(encoding="utf-8") as handle:
                for line in handle:
                    match_out.write(line)
                    n += 1
            with (EXPORT / country / "candidates.tsv").open(encoding="utf-8") as handle:
                for line in handle:
                    cand_out.write(line)
    if n != meta["n_s1"]:
        raise RuntimeError(f"merged {n} rows, expected {meta['n_s1']}")
    print(f"wrote {match_path} and {cand_path} ({n} rows)", flush=True)


def validate() -> None:
    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--matching",
        str(OUT_DIR / "matching_results.tsv"),
        "--candidate",
        str(OUT_DIR / "candidate_pairs.tsv"),
        "--test-dir",
        str(TEST_DIR),
    ]
    print("validating format", flush=True)
    subprocess.run(cmd, check=True)
    print("validating ids", flush=True)
    subprocess.run([*cmd, "--check-ids"], check=True)


def foreign_export_pids() -> list[int]:
    me = os.getpid()
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            continue
        if comm.startswith("python") and "export_test.py" in cmd:
            found.append(pid)
    return found


def wait_until_us_released() -> None:
    """US scoring needs about 4GB. Do not start the India score until that process exits."""
    us_done = EXPORT / "US" / "done"
    while True:
        others = foreign_export_pids()
        if us_done.exists() and not others:
            print("US export released memory", flush=True)
            return
        print(
            f"India index ready; waiting for US score to finish "
            f"(us_done={us_done.exists()} other_pids={others})",
            flush=True,
        )
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="", help="Run one country, then merge if every country is done")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"missing model {MODEL_PATH}")
    meta = split_corpus()
    if args.country:
        if args.country not in meta["countries"]:
            raise SystemExit(f"unknown country {args.country}")
        countries = [args.country]
    else:
        countries = sorted(meta["countries"], key=lambda name: meta["countries"][name]["s1"])
    for country in countries:
        info = meta["countries"][country]
        print(f"=== {country} s1={info['s1']} s23={info['s23']} ===", flush=True)
        retrieve_country(country, info["s1"], meta["n_docs"])
        gc.collect()
        if args.country == "India":
            wait_until_us_released()
            gc.collect()
        score_country(country, info["s1"])
        gc.collect()
    if any(not (EXPORT / name / "done").exists() for name in meta["countries"]):
        print("other countries still unfinished; skip merge", flush=True)
        return
    merge_outputs(meta)
    validate()


if __name__ == "__main__":
    main()
