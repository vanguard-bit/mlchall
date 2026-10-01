"""Second model on India contests. Ship only if it beats highest-score ownership.

Claimants are the Source 1 rows that already kept the target at cutoff 0.75.
Held-out targets (crc32 % 5 == 0) are scored by this model. Every other
contested target stays with the highest v7r score, which is the v7rx rule.
"""

from __future__ import annotations

import os
import time
import zlib
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import macro_f05
from paths import DATA_DIR, ROOT

PARTS = DATA_DIR / "india_slice"
MODEL_PATH = DATA_DIR / "scoreboard" / "lgbm_compete.txt"
FLOOR = 0.75
MIN_DELTA = 0.002


def fold(mid: str) -> int:
    return zlib.crc32(mid.encode()) % 5


def greedy(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    pairs.sort(key=lambda row: row[1], reverse=True)
    kept: list[tuple[str, float]] = []
    for mid, score in pairs:
        if score < FLOOR or len(kept) >= 12:
            break
        kept.append((mid, score))
    return kept


def rows_for(claimants: list[tuple[str, float]]) -> list[list[float]]:
    ordered = sorted(claimants, key=lambda row: row[1], reverse=True)
    n = len(ordered)
    out = []
    for rank, (sid, score) in enumerate(ordered):
        best_other = max(other for other_sid, other in ordered if other_sid != sid)
        out.append([score, float(rank), best_other, score - best_other, float(n), sid])
    return out


def load_slice() -> tuple[dict[str, set[str]], dict[str, list[tuple[str, float]]], dict[str, set[str]]]:
    truths: dict[str, set[str]] = {}
    india: set[str] = set()
    with (DATA_DIR / "train" / "train_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, _name, _addr, country = line.rstrip("\n").split("\t")
            if country == "India":
                india.add(eid)
                truths[eid] = set()
    with (DATA_DIR / "train" / "train_ground_truth.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            if sid not in india:
                continue
            if rest.strip():
                truths[sid] = {mid for mid in rest.split(",") if mid}
    kept: dict[str, list[tuple[str, float]]] = {}
    claims: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for wid in (0, 1):
        with (PARTS / f"scores_{wid}.tsv").open(encoding="utf-8") as handle:
            for line in handle:
                sid, _, body = line.rstrip("\n").partition("\t")
                pairs = []
                if body:
                    for bit in body.split(";"):
                        mid, score = bit.split("|")
                        pairs.append((mid, float(score)))
                chosen = greedy(pairs)
                kept[sid] = chosen
                for mid, score in chosen:
                    claims[mid].append((sid, score))
    contested = {mid: rows for mid, rows in claims.items() if len(rows) > 1}
    print(f"s1 {len(kept)} contested targets {len(contested)}", flush=True)
    return truths, kept, contested


def evaluate(truths, kept, contested, owner_of: dict[str, str]) -> tuple[float, float]:
    order = list(kept)
    gold = []
    preds = []
    single_g = []
    single_p = []
    for sid in order:
        pred = {mid for mid, _score in kept[sid]}
        for mid in list(pred):
            owner = owner_of.get(mid)
            if owner is not None and owner != sid:
                pred.discard(mid)
        gold.append(truths.get(sid, set()))
        preds.append(pred)
        if not truths.get(sid):
            single_g.append(set())
            single_p.append(pred)
    return macro_f05(gold, preds), macro_f05(single_g, single_p)


def owners_by_score(contested) -> dict[str, str]:
    owner = {}
    for mid, rows in contested.items():
        owner[mid] = max(rows, key=lambda row: row[1])[0]
    return owner


def fit_and_owners(truths, contested) -> tuple[dict[str, str], object]:
    train_x = []
    train_y = []
    early_x = []
    early_y = []
    val_pack: dict[str, list[tuple[str, list[float]]]] = {}
    skipped = 0
    for mid, claimants in contested.items():
        built = rows_for(claimants)
        feats = []
        sids = []
        labels = []
        for row in built:
            sid = row[-1]
            feats.append(row[:-1])
            sids.append(sid)
            labels.append(1 if mid in truths.get(sid, ()) else 0)
        if sum(labels) != 1:
            skipped += 1
            continue
        bucket = fold(mid)
        if bucket == 0:
            val_pack[mid] = list(zip(sids, feats))
        elif bucket == 1:
            early_x.extend(feats)
            early_y.extend(labels)
        else:
            train_x.extend(feats)
            train_y.extend(labels)
    print(
        f"train rows {len(train_y)} early {len(early_y)} val targets {len(val_pack)} "
        f"skipped {skipped}",
        flush=True,
    )
    if not train_x or not early_x:
        raise SystemExit("not enough contested targets to train")
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "lambda_l2": 1.0,
            "num_threads": 8,
            "seed": 7,
        },
        lgb.Dataset(np.asarray(train_x, dtype=np.float32), label=np.asarray(train_y)),
        num_boost_round=300,
        valid_sets=[lgb.Dataset(np.asarray(early_x, dtype=np.float32), label=np.asarray(early_y))],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    owner = owners_by_score(contested)
    for mid, pairs in val_pack.items():
        sids = [sid for sid, _feat in pairs]
        feats = np.asarray([feat for _sid, feat in pairs], dtype=np.float32)
        prob = model.predict(feats)
        best_i = max(range(len(sids)), key=lambda i: (float(prob[i]), feats[i][0]))
        owner[mid] = sids[best_i]
    return owner, model


def apply_test(model) -> None:
    """Rewrite test matches. Contested targets in the held-out hash use the model."""
    from pair_features import features_prepared, prepare_record
    from score_v3 import load_s1, load_texts
    from score_v6 import COUNTRIES, PARTS as TEST_PARTS, load_cache, shown_name
    from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
    from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

    src = ROOT / "output" / "v7r" / "matching_results.tsv"
    out_dir = ROOT / "output" / "v7rc"
    t0 = time.time()
    claims: dict[str, list[str]] = defaultdict(list)
    rows: list[tuple[str, list[str]]] = []
    with src.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            mids = [mid for mid in rest.split(",") if mid] if rest.strip() else []
            rows.append((sid, mids))
            for mid in mids:
                claims[mid].append(sid)
    contested = {mid: sids for mid, sids in claims.items() if len(sids) > 1}
    want = {(sid, mid) for mid, sids in contested.items() for sid in sids}
    print(f"test contested {len(contested)} pairs {len(want)}", flush=True)
    extra_root = ROOT / "output" / "v6" / "extra"
    ranks: dict[tuple[str, str], tuple] = {}
    for country in COUNTRIES:
        with (TEST_PARTS / f"{country}.cands.tsv").open(encoding="utf-8") as handle:
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
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    load_cache()
    s1 = load_s1()
    texts = load_texts(set(contested))
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
    involved = {sid for sids in contested.values() for sid in sids}
    s1_name = {}
    s1_addr = {}
    s1_city = {}
    s1_country = {}
    for sid in involved:
        name, addr, country = s1[sid]
        s1_name[sid] = name_view(name)
        s1_addr[sid] = parse_addr(addr)
        s1_city[sid] = city_token(addr)
        s1_country[sid] = country
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    empty_rec = prepare_record("", "")
    score_of: dict[tuple[str, str], float] = {}
    vecs = []
    keys = []
    booster = lgb.Booster(model_file=str(DATA_DIR / "scoreboard" / "lgbm_v7r.txt"))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
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
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )
            vecs.append(
                name_five(left_n, right_n)
                + base[5:]
                + extra_nine(s1_addr[sid], right_addr, left_n, right_n)
                + particle_two(left_n, right_n)
                + rare_two(left_n[1], right_n[1], idf, 1.0)
                + city_two(s1_city[sid], right_city)
            )
            keys.append((sid, mid))
    first = booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=trees)
    for key, score in zip(keys, first):
        score_of[key] = float(score)
    changed = 0
    owner: dict[str, str] = {}
    for mid, sids in contested.items():
        claimants = [(sid, score_of[(sid, mid)]) for sid in sids]
        score_owner = max(claimants, key=lambda row: row[1])[0]
        if fold(mid) != 0:
            owner[mid] = score_owner
            continue
        built = rows_for(claimants)
        feats = np.asarray([row[:-1] for row in built], dtype=np.float32)
        prob = model.predict(feats)
        best_i = max(range(len(built)), key=lambda i: (float(prob[i]), feats[i][0]))
        owner[mid] = built[best_i][-1]
        if owner[mid] != score_owner:
            changed += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    with (out_dir / "matching_results.tsv").open("w", encoding="utf-8") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, mids in rows:
            kept_ids = []
            for mid in mids:
                who = owner.get(mid)
                if who is not None and who != sid:
                    removed += 1
                    continue
                kept_ids.append(mid)
            handle.write(sid + "\t" + ",".join(kept_ids) + "\n")
    src_cand = ROOT / "output" / "v7r" / "candidate_pairs.tsv"
    dest = out_dir / "candidate_pairs.tsv"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.link(src_cand, dest)
    print(
        f"test file {out_dir} removed {removed} model changed {changed} "
        f"targets in {time.time() - t0:.0f}s",
        flush=True,
    )


def main() -> None:
    t0 = time.time()
    truths, kept, contested = load_slice()
    base_owner = owners_by_score(contested)
    base_f, base_s = evaluate(truths, kept, contested, base_owner)
    print(f"argmax f05 {base_f:.4f} singleton {base_s:.4f}", flush=True)
    model_owner, model = fit_and_owners(truths, contested)
    model_f, model_s = evaluate(truths, kept, contested, model_owner)
    print(
        f"model f05 {model_f:.4f} delta {model_f - base_f:+.4f} singleton {model_s:.4f} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    if model_f < base_f + MIN_DELTA or model_s < base_s:
        print("do not ship", flush=True)
        return
    model.save_model(str(MODEL_PATH))
    print(f"saved {MODEL_PATH}", flush=True)
    apply_test(model)


if __name__ == "__main__":
    main()
