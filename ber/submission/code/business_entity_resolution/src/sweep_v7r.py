"""Cutoff sweep for lgbm_v7r on the holdout, romanized names, no extra channels."""

from __future__ import annotations

import json
import math
import time

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

from f05 import macro_f05
from normalize import tokens
from paths import DATA_DIR
from score_v3 import _load_addrs
from score_v6 import load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

BOARD = DATA_DIR / "scoreboard"
RAW_RATIO_COL = 14


def main() -> None:
    t0 = time.time()
    load_cache()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = math.log(n_docs + 1) / math.log(n_docs + 1)
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    val = data["split"].astype(str) == "val"
    s1, mid, X = s1[val], mid[val], X[val]
    needed = set(s1.tolist()) | set(mid.tolist())
    addrs = _load_addrs(needed)
    names: dict[str, str] = {}
    train_dir = DATA_DIR / "train"
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (train_dir / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, _addr, _country = line.rstrip("\n").split("\t")
                if eid in needed and eid not in names:
                    names[eid] = bname
                    if len(names) == len(needed):
                        break
        if len(names) == len(needed):
            break
    shown = {eid: shown_name(names.get(eid, "")) for eid in needed}
    changed = {eid for eid, text in shown.items() if text != names.get(eid, "")}
    raw_right = {eid: " ".join(tokens(shown[eid])) for eid in changed}
    name_of = {eid: name_view(shown[eid]) for eid in needed}
    addr_of = {eid: parse_addr(addrs.get(eid, "")) for eid in needed}
    city_of = {eid: city_token(addrs.get(eid, "")) for eid in needed}
    del names, addrs
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    block = np.empty((len(s1), 36), dtype=np.float32)
    block[:, 5:21] = X[:, 5:21]
    del X
    left_raw: dict[str, str] = {}
    for i, (sid, eid) in enumerate(zip(s1, mid)):
        left_n = name_of.get(sid, empty_name)
        right_n = name_of.get(eid, empty_name)
        block[i, 0:5] = name_five(left_n, right_n)
        block[i, 30:32] = particle_two(left_n, right_n)
        block[i, 21:30] = extra_nine(
            addr_of.get(sid, empty_addr), addr_of.get(eid, empty_addr), left_n, right_n,
        )
        block[i, 32:34] = rare_two(left_n[1], right_n[1], idf, missing)
        block[i, 34:36] = city_two(city_of.get(sid, ""), city_of.get(eid, ""))
        if eid in raw_right:
            raw_l = left_raw.get(sid)
            if raw_l is None:
                raw_l = " ".join(tokens(shown.get(sid, "")))
                left_raw[sid] = raw_l
            block[i, RAW_RATIO_COL] = fuzz.ratio(raw_l, raw_right[eid]) / 100.0
    print(f"features {len(s1)} in {time.time() - t0:.0f}s", flush=True)
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v7r.txt"))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    scores = booster.predict(block, num_iteration=trees)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    country_of = {row["sid"]: row.get("country", "") for row in val_meta}
    grouped: dict[str, list[tuple[str, float]]] = {}
    for sid, eid, score in zip(s1, mid, scores):
        grouped.setdefault(sid, []).append((eid, float(score)))
    order = list(truths)

    def decode(floor: float) -> dict[str, set[str]]:
        out = {}
        for sid in order:
            rows = sorted(grouped.get(sid, []), key=lambda row: row[1], reverse=True)
            chosen: set[str] = set()
            for eid, score in rows:
                if score < floor or len(chosen) >= 12:
                    break
                chosen.add(eid)
            out[sid] = chosen
        return out

    def report(label: str, preds: dict[str, set[str]]) -> None:
        gold = [truths[sid] for sid in order]
        got = [preds[sid] for sid in order]
        single_g, single_p = [], []
        buckets = {"India": ([], []), "US": ([], [])}
        for sid in order:
            if not truths[sid]:
                single_g.append(set())
                single_p.append(preds[sid])
            country = country_of.get(sid, "")
            if country in buckets:
                buckets[country][0].append(truths[sid])
                buckets[country][1].append(preds[sid])
        single = macro_f05(single_g, single_p) if single_g else 0.0
        india = macro_f05(*buckets["India"])
        us = macro_f05(*buckets["US"])
        print(
            f"{label} f05 {macro_f05(gold, got):.4f} singleton {single:.4f} india {india:.4f} us {us:.4f}",
            flush=True,
        )

    for floor in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        report(f"v7r floor {floor:.2f}", decode(floor))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
