"""Score v6's candidate lists with the v7 matcher.

Two workers, one batch of texts at a time. Same memory pattern as v6.
The candidate file is hardlinked from output/v6.
"""

from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from pair_features import features_prepared, prepare_record
from paths import ROOT
from score_v3 import FLUSH_PAIRS, count_lines, load_s1, load_texts
from score_v6 import COUNTRIES, PARTS, _load_extra, load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

OUT = ROOT / "output" / os.environ.get("SCORE_OUT", "v7")
V6 = ROOT / "output" / "v6"
MODEL = ROOT / "ber" / "data" / "scoreboard" / os.environ.get("SCORE_MODEL", "lgbm_v7.txt")
MIN_GAIN = float(os.environ.get("SCORE_MIN_GAIN", "0.70"))
SCORE_WORKERS = 2
BATCH_S1 = 8_000

IDF: dict[str, float] = {}
MISSING_IDF = 1.0


def _prepare_idf() -> None:
    counts, n_docs = document_frequency()
    IDF.clear()
    IDF.update(idf_table(counts, n_docs))
    print(f"idf tokens {len(IDF)}", flush=True)


def score_span(args: tuple[str, int, int, str]) -> int:
    country, start, end, dest = args
    load_cache()
    if not IDF:
        _prepare_idf()
    s1 = load_s1()
    extra = _load_extra(country)
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
        parsed: list[tuple[str, list[tuple[str, str]]]] = []
        for line in lines:
            sid, _, payload = line.rstrip("\n").partition("\t")
            bits: list[tuple[str, str]] = []
            seen: set[str] = set()
            if payload:
                for bit in payload.split(";"):
                    mid = bit.split("|", 1)[0]
                    if mid in seen:
                        continue
                    seen.add(mid)
                    bits.append((mid, bit))
                    need.add(mid)
            for mid in extra.get(sid, []):
                if mid in seen:
                    continue
                seen.add(mid)
                bits.append((mid, f"{mid}|1|99|1|0|1|0"))
                need.add(mid)
            parsed.append((sid, bits))
        texts = load_texts(need)
        prepared = {}
        names = {}
        addrs = {}
        cities = {}
        countries = {}
        for mid, (name, addr, country_) in texts.items():
            text = shown_name(name)
            prepared[mid] = prepare_record(text, addr)
            names[mid] = name_view(text)
            addrs[mid] = parse_addr(addr)
            cities[mid] = city_token(addr)
            countries[mid] = country_
        del texts
        for sid, bits in parsed:
            name, addr, s_country = s1[sid]
            left = prepare_record(name, addr)
            left_name = name_view(name)
            left_addr = parse_addr(addr)
            left_city = city_token(addr)
            mids: list[str] = []
            for mid, bit in bits:
                _mid, nr, ar, ns, asc, fn, fa = bit.split("|")
                base = features_prepared(
                    left,
                    prepared.get(mid, empty_rec),
                    country_eq=float(s_country == countries.get(mid, "")),
                    name_rank=float(nr),
                    addr_rank=float(ar),
                    name_score=float(ns),
                    addr_score=float(asc),
                    from_name_channel=int(float(fn)),
                    from_addr_channel=int(float(fa)),
                )
                right_name = names.get(mid, empty_name)
                pending_x.append(
                    name_five(left_name, right_name)
                    + base[5:]
                    + extra_nine(left_addr, addrs.get(mid, empty_addr), left_name, right_name)
                    + particle_two(left_name, right_name)
                    + rare_two(left_name[1], right_name[1], IDF, MISSING_IDF)
                    + city_two(left_city, cities.get(mid, ""))
                )
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
    import multiprocessing as mp

    t0 = time.time()
    print(f"out {OUT} model {MODEL.name} min_gain {MIN_GAIN}", flush=True)
    if not MODEL.exists():
        raise SystemExit(f"missing {MODEL}")
    for country in COUNTRIES:
        path = V6 / "extra" / f"{country}.tsv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
    counts = {country: count_lines(PARTS / f"{country}.cands.tsv") for country in COUNTRIES}
    if sum(counts.values()) != 1_732_544:
        raise SystemExit(f"candidate shards sum to {sum(counts.values())}")
    OUT.mkdir(parents=True, exist_ok=True)
    part_dir = OUT / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    match_path = OUT / "matching_results.tsv"
    ctx = mp.get_context("spawn")
    with match_path.open("w", encoding="utf-8") as match_out:
        match_out.write("source1_entity_id\tmatched_entity_ids\n")
        for country, n_lines in counts.items():
            mid = n_lines // SCORE_WORKERS
            spans = [
                (country, 0, mid, str(part_dir / f"{country}_0.tsv")),
                (country, mid, n_lines, str(part_dir / f"{country}_1.tsv")),
            ]
            with ctx.Pool(SCORE_WORKERS) as pool:
                wrote = pool.map(score_span, spans)
            print(f"{country} parts {wrote}", flush=True)
            for _country, _start, _end, part in spans:
                with open(part, encoding="utf-8") as handle:
                    for line in handle:
                        match_out.write(line)
                os.remove(part)
    n_rows = count_lines(match_path) - 1
    src = V6 / "candidate_pairs.tsv"
    dest = OUT / "candidate_pairs.tsv"
    if count_lines(src) - 1 == 1_732_544:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.link(src, dest)
        print(f"linked candidates from {src}", flush=True)
    else:
        print(f"v6 candidates not complete ({src})", flush=True)
    print(f"wrote {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
    if n_rows != 1_732_544:
        raise SystemExit(f"row count {n_rows}")


if __name__ == "__main__":
    main()
