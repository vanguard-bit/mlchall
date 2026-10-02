"""Build output/v6 from the measured channels, including accent folding.

The folded name key is the accent-fold channel: Café and Cafe share a key.
Uncommon keys are kept up to 12. Common keys are kept only when the house or
a street token also agrees. Consonant skeleton plus house, and an exact
website stem, are added on top. Indic names are romanized before scoring.
The unconstrained accent dump is not included.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from eval_domain_spell import domain_stems, skeleton, squash_name
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR, ROOT
from score_v3 import BATCH_S1, FLUSH_PAIRS, count_lines, load_s1, load_texts
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v6_keys import CACHE, indic_token, name_key, romanized_name

TEST = DATA_DIR / "test"
PARTS = DATA_DIR / "export_fast"
BOARD = DATA_DIR / "scoreboard"
OUT = ROOT / "output" / "v6"
MODEL = BOARD / "lgbm_v5.txt"
MIN_GAIN = 0.70
RETRIEVE_WORKERS = 8
SCORE_WORKERS = 2
EXACT_CAP = 12
GATE_CAP = 40
TIGHT_CAP = 8
COUNTRIES = ("France", "US", "India")

TARGET = ""
EXACT_QUERY: set[str] = set()
GATES: dict[str, tuple[set[str], set[str]]] = {}
SQUASH_QUERY: set[str] = set()
STEM_QUERY: set[str] = set()
SPELL_QUERY: set[str] = set()


def load_cache() -> None:
    if CACHE:
        return
    with (BOARD / "indic_xlit_cache.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("roman"):
                CACHE[(row["w"], row["lang"])] = row["roman"]


def shown_name(name: str) -> str:
    if not indic_token(name):
        return name
    roman = romanized_name(name)
    return roman or name


def _worker(args: tuple[str, int]) -> tuple[dict, dict, dict, dict]:
    path, wid = args
    exact: dict[str, list[str]] = defaultdict(list)
    gated: dict[str, list] = defaultdict(list)
    spell: dict[str, list[str]] = defaultdict(list)
    domain: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % RETRIEVE_WORKERS != wid:
                continue
            if wid == 0 and i and i % 1_000_000 == 0:
                print(f"  scan {os.path.basename(path)} {i}", flush=True)
            eid, name, addr, country = line.rstrip("\n").split("\t")
            if country != TARGET:
                continue
            key = name_key(name, country)
            if key in EXACT_QUERY and len(exact[key]) < EXACT_CAP:
                exact[key].append(eid)
            gate = GATES.get(key)
            house, streets, _zip = parse_addr(addr)
            if gate is not None:
                houses, streets_ok = gate
                if ((house and house in houses) or (streets & streets_ok)) and len(gated[key]) < GATE_CAP:
                    gated[key].append((eid, house, tuple(streets)))
            stems = domain_stems(name)
            if not stems and ("." in addr or "www" in addr.lower() or "http" in addr.lower()):
                stems = domain_stems(addr)
            squash = squash_name(name)
            for stem in stems:
                dkey = country + "|" + stem
                if dkey in SQUASH_QUERY and len(domain[dkey]) < TIGHT_CAP:
                    domain[dkey].append(eid)
            if len(squash) >= 8:
                dkey = country + "|" + squash
                if dkey in STEM_QUERY and eid not in domain[dkey] and len(domain[dkey]) < TIGHT_CAP:
                    domain[dkey].append(eid)
                skel = skeleton(squash)
                if len(house) >= 2 and len(skel) >= 8:
                    skey = f"{country}|{house}|{skel}"
                    if skey in SPELL_QUERY and len(spell[skey]) < TIGHT_CAP:
                        spell[skey].append(eid)
    return exact, gated, spell, domain


def _clear_queries() -> None:
    global TARGET
    TARGET = ""
    EXACT_QUERY.clear()
    GATES.clear()
    SQUASH_QUERY.clear()
    STEM_QUERY.clear()
    SPELL_QUERY.clear()


def retrieve_country(country: str) -> None:
    global TARGET
    dest = OUT / "extra" / f"{country}.tsv"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"{country} extra exists", flush=True)
        return
    _clear_queries()
    TARGET = country
    rows: list[tuple[str, str, frozenset[str], str, str]] = []
    exact_of: dict[str, list[str]] = defaultdict(list)
    squash_of: dict[str, list[str]] = defaultdict(list)
    stem_of: dict[str, list[str]] = defaultdict(list)
    spell_of: dict[str, list[str]] = defaultdict(list)
    with (TEST / "test_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, name, addr, row_country = line.rstrip("\n").split("\t")
            if row_country != country:
                continue
            house, streets, _zip = parse_addr(addr)
            key = name_key(name, country)
            squash = squash_name(name)
            rows.append((sid, house, streets, key, squash))
            if key:
                exact_of[key].append(sid)
                EXACT_QUERY.add(key)
                if house or streets:
                    houses, street_set = GATES.setdefault(key, (set(), set()))
                    if house:
                        houses.add(house)
                    street_set.update(streets)
            if len(squash) >= 8:
                dkey = country + "|" + squash
                squash_of[dkey].append(sid)
                SQUASH_QUERY.add(dkey)
                skel = skeleton(squash)
                if len(house) >= 2 and len(skel) >= 8:
                    skey = f"{country}|{house}|{skel}"
                    spell_of[skey].append(sid)
                    SPELL_QUERY.add(skey)
            for stem in set(domain_stems(name) + domain_stems(addr)):
                dkey = country + "|" + stem
                stem_of[dkey].append(sid)
                STEM_QUERY.add(dkey)
    print(
        f"{country} s1 {len(rows)} exact {len(EXACT_QUERY)} gated {len(GATES)} spell {len(SPELL_QUERY)}",
        flush=True,
    )
    ctx = mp.get_context("fork")
    jobs = [
        (str(TEST / name), wid)
        for name in ("test_source2.tsv", "test_source3.tsv")
        for wid in range(RETRIEVE_WORKERS)
    ]
    exact_hits: dict[str, list[str]] = defaultdict(list)
    gated_hits: dict[str, list] = defaultdict(list)
    spell_hits: dict[str, list[str]] = defaultdict(list)
    domain_hits: dict[str, list[str]] = defaultdict(list)
    with ctx.Pool(RETRIEVE_WORKERS) as pool:
        for part_e, part_g, part_s, part_d in pool.map(_worker, jobs):
            for key, eids in part_e.items():
                exact_hits[key].extend(eids)
            for key, items in part_g.items():
                gated_hits[key].extend(items)
            for key, eids in part_s.items():
                spell_hits[key].extend(eids)
            for key, eids in part_d.items():
                domain_hits[key].extend(eids)
    for bucket, cap in ((exact_hits, EXACT_CAP), (spell_hits, TIGHT_CAP), (domain_hits, TIGHT_CAP)):
        for key in list(bucket):
            if len(bucket[key]) > cap:
                del bucket[key]
    print(
        f"{country} kept exact {len(exact_hits)} gated {len(gated_hits)} spell {len(spell_hits)} domain {len(domain_hits)}",
        flush=True,
    )

    def take(index_of: dict[str, list[str]], hits: dict[str, list[str]], into: dict[str, set[str]], limit: int) -> None:
        for key, sids in index_of.items():
            for eid in hits.get(key, []):
                for sid in sids:
                    bucket = into[sid]
                    if eid in bucket or len(bucket) >= limit:
                        continue
                    bucket.add(eid)

    exact_added: dict[str, set[str]] = defaultdict(set)
    spell_added: dict[str, set[str]] = defaultdict(set)
    domain_added: dict[str, set[str]] = defaultdict(set)
    take(exact_of, exact_hits, exact_added, 12)
    take(spell_of, spell_hits, spell_added, 12)
    take(squash_of, domain_hits, domain_added, 12)
    take(stem_of, domain_hits, domain_added, 12)
    n_ids = 0
    n_rows = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for sid, house, streets, key, _squash in rows:
            chosen = set(exact_added.get(sid, ()))
            chosen.update(spell_added.get(sid, ()))
            chosen.update(domain_added.get(sid, ()))
            both: list[str] = []
            only_house: list[str] = []
            only_street: list[str] = []
            for eid, row_house, row_streets in gated_hits.get(key, []):
                if eid in chosen:
                    continue
                street_hit = bool(set(row_streets) & streets)
                house_hit = bool(house and row_house == house)
                if house_hit and street_hit:
                    both.append(eid)
                elif house_hit:
                    only_house.append(eid)
                elif street_hit:
                    only_street.append(eid)
            for eid in (both + only_house + only_street)[:25]:
                chosen.add(eid)
            if not chosen:
                continue
            handle.write(sid + "\t" + ",".join(chosen) + "\n")
            n_rows += 1
            n_ids += len(chosen)
    print(f"{country} extra rows {n_rows} ids {n_ids}", flush=True)
    _clear_queries()
    del rows, exact_of, squash_of, stem_of, spell_of
    del exact_hits, gated_hits, spell_hits, domain_hits
    del exact_added, spell_added, domain_added
    gc.collect()


def _load_extra(country: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    path = OUT / "extra" / f"{country}.tsv"
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            if rest:
                found[sid] = rest.split(",")
    return found


def score_span(args: tuple[str, int, int, str]) -> int:
    country, start, end, dest = args
    load_cache()
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
    cand = open(dest + ".cand", "w", encoding="utf-8")

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
            cand.write(f"{sid}\t{','.join(mids)}\n")
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
        countries = {}
        for mid, (name, addr, country_) in texts.items():
            text = shown_name(name)
            prepared[mid] = prepare_record(text, addr)
            names[mid] = name_view(text)
            addrs[mid] = parse_addr(addr)
            countries[mid] = country_
        del texts
        for sid, bits in parsed:
            name, addr, s_country = s1[sid]
            left = prepare_record(name, addr)
            left_name = name_view(name)
            left_addr = parse_addr(addr)
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
                )
                mids.append(mid)
            if not mids:
                out.write(f"{sid}\t\n")
                cand.write(f"{sid}\t\n")
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
    cand.close()
    print(f"{country} {start}:{end} wrote {written}", flush=True)
    return written


def _free_gb() -> float:
    return shutil.disk_usage(OUT).free / (1024 ** 3)


def score_all() -> None:
    if _free_gb() < 1.3:
        raise SystemExit(f"only {_free_gb():.2f} GB free, need 1.3 GB for the candidate file")
    counts = {country: count_lines(PARTS / f"{country}.cands.tsv") for country in COUNTRIES}
    if sum(counts.values()) != 1_732_544:
        raise SystemExit(f"candidate shards sum to {sum(counts.values())}, expected 1732544")
    part_dir = OUT / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    match_path = OUT / "matching_results.tsv"
    cand_path = OUT / "candidate_pairs.tsv"
    ctx = mp.get_context("spawn")
    with match_path.open("w", encoding="utf-8") as match_out, cand_path.open("w", encoding="utf-8") as cand_out:
        match_out.write("source1_entity_id\tmatched_entity_ids\n")
        cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
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
                with open(part + ".cand", encoding="utf-8") as handle:
                    for line in handle:
                        cand_out.write(line)
                os.remove(part + ".cand")
    n_rows = count_lines(match_path) - 1
    n_cand = count_lines(cand_path) - 1
    print(f"wrote matching {n_rows} candidates {n_cand}", flush=True)
    if n_rows != 1_732_544 or n_cand != 1_732_544:
        raise SystemExit(f"row count matching {n_rows} candidates {n_cand}")


def main() -> None:
    t0 = time.time()
    load_cache()
    print(f"xlit cache {len(CACHE)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "extra").mkdir(parents=True, exist_ok=True)
    for country in COUNTRIES:
        retrieve_country(country)
        gc.collect()
    score_all()
    print(f"v6 done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
