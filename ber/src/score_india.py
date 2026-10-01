"""Score the India slice candidate lists with lgbm_v7r.

Two workers, one candidate file each. Writes every pair score so the
competition model can decode without scoring again.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import lightgbm as lgb
import numpy as np

from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from score_v6 import load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

PARTS = DATA_DIR / "india_slice"
TRAIN = DATA_DIR / "train"
MODEL = DATA_DIR / "scoreboard" / "lgbm_v7r.txt"
BATCH_S1 = 8_000
FLUSH_PAIRS = 40_000
WORKERS = 2

IDF: dict[str, float] = {}
MISSING_IDF = 1.0


def _load_s1() -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            if country == "India":
                rows[eid] = (name, addr, country)
    return rows


def _load_texts(need: set[str]) -> dict[str, tuple[str, str, str]]:
    texts: dict[str, tuple[str, str, str]] = {}
    if not need:
        return texts
    for name in ("train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid = line.split("\t", 1)[0]
                if eid not in need:
                    continue
                _eid, bname, addr, country = line.rstrip("\n").split("\t")
                texts[eid] = (bname, addr, country)
                if len(texts) == len(need):
                    return texts
    return texts


def _prepare_idf() -> None:
    if IDF:
        return
    counts, n_docs = document_frequency()
    IDF.update(idf_table(counts, n_docs))
    print(f"idf tokens {len(IDF)}", flush=True)


def score_file(wid: int) -> int:
    load_cache()
    _prepare_idf()
    s1 = _load_s1()
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    empty_rec = prepare_record("", "")
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    pending_x: list[list[float]] = []
    pending_sid: list[str] = []
    pending_mids: list[list[str]] = []
    written = 0
    src = PARTS / f"cands_{wid}.tsv"
    dest = PARTS / f"scores_{wid}.tsv"
    out = dest.open("w", encoding="utf-8")

    def flush() -> None:
        nonlocal written
        if not pending_sid:
            return
        pred = booster.predict(np.asarray(pending_x, dtype=np.float32), num_iteration=trees)
        cursor = 0
        for sid, mids in zip(pending_sid, pending_mids):
            part = pred[cursor : cursor + len(mids)]
            cursor += len(mids)
            body = ";".join(f"{mid}|{float(score):.6f}" for mid, score in zip(mids, part))
            out.write(f"{sid}\t{body}\n")
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
            parsed.append((sid, bits))
        texts = _load_texts(need)
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
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            batch.append(line)
            if len(batch) >= BATCH_S1:
                score_batch(batch)
                batch = []
                print(f"  w{wid} {written}", flush=True)
        if batch:
            score_batch(batch)
    out.close()
    print(f"w{wid} wrote {written}", flush=True)
    return written


def main() -> None:
    t0 = time.time()
    if not MODEL.exists():
        raise SystemExit(f"missing {MODEL}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        wrote = pool.map(score_file, range(WORKERS))
    print(f"scored {sum(wrote)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
