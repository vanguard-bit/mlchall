"""Score the existing test candidates with the v5 matcher. Started after v4 exits."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from pair_features import features_prepared, prepare_record
from paths import ROOT
from score_v3 import (
    BATCH_S1,
    COUNTRIES,
    FLUSH_PAIRS,
    MIN_GAIN,
    PARTS,
    WORKERS,
    count_lines,
    load_s1,
    load_texts,
)
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two

OUT = ROOT / "output" / "v5"
MODEL = ROOT / "ber" / "data" / "scoreboard" / "lgbm_v5.txt"


def score_span(args: tuple[str, int, int, str]) -> int:
    country, start, end, dest = args
    s1 = load_s1()
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    empty_rec = prepare_record("", "")
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    pending_x: list[list[float]] = []
    pending_sid: list[str] = []
    pending_mids: list[list[str]] = []
    written = 0
    shard = PARTS / f"{country}.cands.tsv"
    out = open(dest, "w", encoding="utf-8")

    def flush() -> None:
        nonlocal written
        if not pending_sid:
            return
        pred = booster.predict(np.asarray(pending_x, dtype=np.float32), num_iteration=trees)
        cursor = 0
        for sid, mids in zip(pending_sid, pending_mids):
            part = [float(x) for x in pred[cursor : cursor + len(mids)]]
            cursor += len(mids)
            chosen = decode_greedy_f05(mids, part, min_gain=MIN_GAIN, max_preds=12)
            out.write(f"{sid}\t{','.join(mid for mid in mids if mid in chosen)}\n")
            written += 1
        pending_x.clear()
        pending_sid.clear()
        pending_mids.clear()

    def score_batch(lines: list[str]) -> None:
        nonlocal written
        need: set[str] = set()
        parsed: list[tuple[str, str]] = []
        for line in lines:
            sid, _, payload = line.rstrip("\n").partition("\t")
            parsed.append((sid, payload))
            if payload:
                for bit in payload.split(";"):
                    need.add(bit.split("|", 1)[0])
        texts = load_texts(need)
        prepared = {mid: prepare_record(name, addr) for mid, (name, addr, _c) in texts.items()}
        names = {mid: name_view(name) for mid, (name, _addr, _c) in texts.items()}
        addrs = {mid: parse_addr(addr) for mid, (_name, addr, _c) in texts.items()}
        countries = {mid: country_ for mid, (_n, _a, country_) in texts.items()}
        del texts
        for sid, payload in parsed:
            name, addr, s_country = s1[sid]
            left = prepare_record(name, addr)
            left_name = name_view(name)
            left_addr = parse_addr(addr)
            mids: list[str] = []
            if payload:
                for bit in payload.split(";"):
                    mid, nr, ar, ns, asc, fn, fa = bit.split("|")
                    base = features_prepared(
                        left,
                        prepared.get(mid, empty_rec),
                        country_eq=float(s_country == countries.get(mid, "")),
                        name_rank=float(nr),
                        addr_rank=float(ar),
                        name_score=float(ns),
                        addr_score=float(asc),
                        from_name_channel=int(fn),
                        from_addr_channel=int(fa),
                    )
                    right_name = names.get(mid, empty_name)
                    vec = name_five(left_name, right_name) + base[5:] + extra_nine(
                        left_addr, addrs.get(mid, empty_addr), left_name, right_name
                    ) + particle_two(left_name, right_name)
                    pending_x.append(vec)
                    mids.append(mid)
            if not mids:
                out.write(f"{sid}\t\n")
                written += 1
            else:
                pending_sid.append(sid)
                pending_mids.append(mids)
            if len(pending_x) >= FLUSH_PAIRS:
                flush()
        flush()

    batch: list[str] = []
    with shard.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i < start:
                continue
            if i >= end:
                break
            batch.append(line)
            if len(batch) >= BATCH_S1:
                score_batch(batch)
                batch = []
                print(f"  {country} {start}:{end} {written}", flush=True)
        if batch:
            score_batch(batch)
    out.close()
    print(f"{country} {start}:{end} wrote {written}", flush=True)
    return written


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"missing {MODEL}")
    t0 = time.time()
    counts = {country: count_lines(PARTS / f"{country}.cands.tsv") for country in COUNTRIES}
    if sum(counts.values()) != 1_732_544:
        raise SystemExit(f"candidate shards sum to {sum(counts.values())}, expected 1732544")
    OUT.mkdir(parents=True, exist_ok=True)
    part_dir = OUT / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    match_path = OUT / "matching_results.tsv"
    ctx = mp.get_context("spawn")
    with match_path.open("w", encoding="utf-8") as match_out:
        match_out.write("source1_entity_id\tmatched_entity_ids\n")
        for country, n_lines in counts.items():
            mid = n_lines // WORKERS
            spans = [
                (country, 0, mid, str(part_dir / f"{country}_0.tsv")),
                (country, mid, n_lines, str(part_dir / f"{country}_1.tsv")),
            ]
            with ctx.Pool(WORKERS) as pool:
                wrote = pool.map(score_span, spans)
            print(f"{country} parts {wrote}", flush=True)
            for _country, _start, _end, part in spans:
                with open(part, encoding="utf-8") as handle:
                    for line in handle:
                        match_out.write(line)
                os.remove(part)
    src = ROOT / "output" / "v2" / "candidate_pairs.tsv"
    dest = OUT / "candidate_pairs.tsv"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.link(src, dest)
    n_rows = count_lines(match_path) - 1
    print(f"wrote {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
    if n_rows != 1_732_544:
        raise SystemExit(f"row count {n_rows} != 1732544")


if __name__ == "__main__":
    main()
