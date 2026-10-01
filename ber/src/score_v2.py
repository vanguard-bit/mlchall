"""Score the finished candidate shards without holding every country in memory."""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR, ROOT

TEST = DATA_DIR / "test"
PARTS = DATA_DIR / "export_fast"
OUT = ROOT / "output" / "v2"
MODEL = DATA_DIR / "scoreboard" / "lgbm_v2.txt"
MIN_GAIN = 0.70


def load_s1() -> dict[str, tuple[str, str, str]]:
    rows = {}
    with (TEST / "test_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            rows[eid] = (name, addr, country)
    return rows


def country_cands(s1: dict[str, tuple[str, str, str]], country: str, dest: Path) -> set[str]:
    need: set[str] = set()
    with dest.open("w", encoding="utf-8") as out:
        for wid in (0, 1):
            with (PARTS / f"cands_{wid}.tsv").open(encoding="utf-8") as handle:
                for line in handle:
                    sid = line.split("\t", 1)[0]
                    if s1[sid][2] != country:
                        continue
                    out.write(line)
                    payload = line.rstrip("\n").split("\t", 1)[1] if "\t" in line else ""
                    if payload:
                        for bit in payload.split(";"):
                            need.add(bit.split("|", 1)[0])
    return need


def load_texts(need: set[str]) -> dict[str, tuple[str, str, str]]:
    texts = {}
    for name in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid = line.split("\t", 1)[0]
                if eid not in need:
                    continue
                _eid, name_, addr, country = line.rstrip("\n").split("\t")
                texts[eid] = (name_, addr, country)
                if len(texts) == len(need):
                    return texts
    return texts


def score_country(country: str, s1: dict[str, tuple[str, str, str]], booster, trees: int, match_out, cand_out) -> int:
    t0 = time.time()
    shard = PARTS / f"{country}.cands.tsv"
    if not shard.exists() or shard.stat().st_size == 0:
        country_cands(s1, country, shard)
    print(f"{country}: scoring in batches", flush=True)
    empty = prepare_record("", "")
    pending_x: list[list[float]] = []
    pending_sid: list[str] = []
    pending_mids: list[list[str]] = []
    written = 0
    batch: list[str] = []

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
            match_out.write(f"{sid}\t{','.join(mid for mid in mids if mid in chosen)}\n")
            cand_out.write(f"{sid}\t{','.join(mids)}\n")
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
        prepared = {mid: prepare_record(name, addr) for mid, (name, addr, _country) in texts.items()}
        countries = {mid: cty for mid, (_name, _addr, cty) in texts.items()}
        del texts
        for sid, payload in parsed:
            name, addr, s_country = s1[sid]
            left = prepare_record(name, addr)
            mids: list[str] = []
            if payload:
                for bit in payload.split(";"):
                    mid, nr, ar, ns, asc, fn, fa = bit.split("|")
                    pending_x.append(
                        features_prepared(
                            left, prepared.get(mid, empty),
                            country_eq=float(s_country == countries.get(mid, "")),
                            name_rank=float(nr), addr_rank=float(ar),
                            name_score=float(ns), addr_score=float(asc),
                            from_name_channel=int(fn), from_addr_channel=int(fa),
                        )
                    )
                    mids.append(mid)
            if not mids:
                match_out.write(f"{sid}\t\n")
                cand_out.write(f"{sid}\t\n")
                written += 1
            else:
                pending_sid.append(sid)
                pending_mids.append(mids)
            if len(pending_x) >= 60_000:
                flush()
        flush()
        del prepared, countries

    with shard.open(encoding="utf-8") as handle:
        for line in handle:
            batch.append(line)
            if len(batch) >= 12_000:
                score_batch(batch)
                batch = []
                print(f"  {country} {written}", flush=True)
        if batch:
            score_batch(batch)
    print(f"{country}: scored {written} in {time.time() - t0:.0f}s", flush=True)
    return written


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = load_s1()
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    total = 0
    with (OUT / "matching_results.tsv").open("w", encoding="utf-8") as match_out, (OUT / "candidate_pairs.tsv").open("w", encoding="utf-8") as cand_out:
        match_out.write("source1_entity_id\tmatched_entity_ids\n")
        cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in ("France", "US", "India"):
            total += score_country(country, s1, booster, trees, match_out, cand_out)
    print(f"wrote {total} rows in {time.time() - t0:.0f}s", flush=True)
    if total != len(s1):
        raise SystemExit(f"row count {total} != {len(s1)}")


if __name__ == "__main__":
    main()
