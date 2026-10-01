"""Give each contested Source 2/3 id to the Source 1 with the higher v7 score.

Training labels never assign one target to two Source 1 rows. The holdout is
too small to see the clashes. This only rescores pairs that the v7 file
already emitted more than once.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from pair_features import features_prepared, prepare_record
from paths import DATA_DIR, ROOT
from score_v3 import load_s1, load_texts
from score_v6 import load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

BOARD = DATA_DIR / "scoreboard"
MODEL = BOARD / os.environ.get("SCORE_MODEL", "lgbm_v7.txt")
SRC = ROOT / "output" / os.environ.get("SCORE_SRC", "v7") / "matching_results.tsv"
OUT = ROOT / "output" / os.environ.get("SCORE_OUT", "v7x")


def main() -> None:
    t0 = time.time()
    print(f"src {SRC} out {OUT} model {MODEL.name}", flush=True)
    claims: dict[str, list[str]] = defaultdict(list)
    rows: list[tuple[str, list[str]]] = []
    with SRC.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            mids = [mid for mid in rest.split(",") if mid] if rest.strip() else []
            rows.append((sid, mids))
            for mid in mids:
                claims[mid].append(sid)
    contested = {mid: sids for mid, sids in claims.items() if len(sids) > 1}
    want = {(sid, mid) for mid, sids in contested.items() for sid in sids}
    print(f"contested targets {len(contested)} pairs {len(want)} in {time.time() - t0:.0f}s", flush=True)
    from score_v6 import COUNTRIES, PARTS

    extra_root = ROOT / "output" / "v6" / "extra"

    ranks: dict[tuple[str, str], tuple[float, float, float, float, int, int]] = {}
    for country in COUNTRIES:
        with (PARTS / f"{country}.cands.tsv").open(encoding="utf-8") as handle:
            for line in handle:
                sid, _, payload = line.rstrip("\n").partition("\t")
                if not payload:
                    continue
                for bit in payload.split(";"):
                    mid = bit.split("|", 1)[0]
                    key = (sid, mid)
                    if key not in want:
                        continue
                    _mid, nr, ar, ns, asc, fn, fa = bit.split("|")
                    ranks[key] = (float(nr), float(ar), float(ns), float(asc), int(float(fn)), int(float(fa)))
        with (extra_root / f"{country}.tsv").open(encoding="utf-8") as handle:
            for line in handle:
                sid, _, rest = line.rstrip("\n").partition("\t")
                if not rest:
                    continue
                for mid in rest.split(","):
                    key = (sid, mid)
                    if key in want and key not in ranks:
                        ranks[key] = (1.0, 99.0, 1.0, 0.0, 1, 0)
    print(f"ranks {len(ranks)}", flush=True)
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = 1.0
    load_cache()
    s1 = load_s1()
    need = set(contested)
    for sids in contested.values():
        need.update(sids)
    texts = load_texts(set(contested))
    print(f"target texts {len(texts)} s1 involved {sum(len(v) for v in contested.values())}", flush=True)
    name_of = {}
    addr_of = {}
    city_of = {}
    countries = {}
    for mid, (name, addr, country) in texts.items():
        shown = shown_name(name)
        name_of[mid] = name_view(shown)
        addr_of[mid] = parse_addr(addr)
        city_of[mid] = city_token(addr)
        countries[mid] = country
    s1_name = {}
    s1_addr = {}
    s1_city = {}
    s1_country = {}
    involved = {sid for sids in contested.values() for sid in sids}
    for sid in involved:
        name, addr, country = s1[sid]
        s1_name[sid] = name_view(name)
        s1_addr[sid] = parse_addr(addr)
        s1_city[sid] = city_token(addr)
        s1_country[sid] = country
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    empty_rec = prepare_record("", "")
    pairs: list[tuple[str, str]] = []
    vecs: list[list[float]] = []
    for mid, sids in contested.items():
        right_n = name_of.get(mid, empty_name)
        right_addr = addr_of.get(mid, empty_addr)
        right_city = city_of.get(mid, "")
        raw = texts.get(mid, ("", "", ""))
        right_rec = prepare_record(shown_name(raw[0]), raw[1])
        for sid in sids:
            left_n = s1_name[sid]
            left_raw = s1[sid]
            nr, ar, ns, asc, fn, fa = ranks.get((sid, mid), (1.0, 99.0, 1.0, 0.0, 1, 0))
            base = features_prepared(
                prepare_record(left_raw[0], left_raw[1]),
                right_rec or empty_rec,
                country_eq=float(s1_country[sid] == countries.get(mid, "")),
                name_rank=nr,
                addr_rank=ar,
                name_score=ns,
                addr_score=asc,
                from_name_channel=fn,
                from_addr_channel=fa,
            )
            vecs.append(
                name_five(left_n, right_n)
                + base[5:]
                + extra_nine(s1_addr[sid], right_addr, left_n, right_n)
                + particle_two(left_n, right_n)
                + rare_two(left_n[1], right_n[1], idf, missing)
                + city_two(s1_city[sid], right_city)
            )
            pairs.append((sid, mid))
            if len(vecs) >= 40_000:
                pass
    print(f"pairs {len(pairs)} features in {time.time() - t0:.0f}s", flush=True)
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    scores = booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=trees)
    best: dict[str, tuple[str, float]] = {}
    for (sid, mid), score in zip(pairs, scores):
        prev = best.get(mid)
        if prev is None or float(score) > prev[1]:
            best[mid] = (sid, float(score))
    OUT.mkdir(parents=True, exist_ok=True)
    match_path = OUT / "matching_results.tsv"
    removed = 0
    with match_path.open("w", encoding="utf-8") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, mids in rows:
            kept = []
            for mid in mids:
                owner = best.get(mid)
                if owner is not None and owner[0] != sid:
                    removed += 1
                    continue
                kept.append(mid)
            handle.write(sid + "\t" + ",".join(kept) + "\n")
    src_cand = SRC.parent / "candidate_pairs.tsv"
    dest = OUT / "candidate_pairs.tsv"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.link(src_cand, dest)
    print(f"removed {removed} extra claims in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
