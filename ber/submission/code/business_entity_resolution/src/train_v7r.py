"""Retrain v7 with the same name the test scorer shows the model.

score_v7 romanizes Indic Source 2/3 names before every name feature, including
raw_ratio. train_v7 trained on the raw script. This script applies shown_name
to both sides (Source 1 never changes) and rewrites raw_ratio on those pairs.
Saves lgbm_v7r.txt and does not replace lgbm_v7.txt.
"""

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
from score_v3 import _f05, _groups, _load_addrs
from score_v6 import load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

BOARD = DATA_DIR / "scoreboard"
MODEL = BOARD / "lgbm_v7r.txt"
RAW_RATIO_COL = 14


def main() -> None:
    t0 = time.time()
    load_cache()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = math.log(n_docs + 1) / math.log(n_docs + 1)
    print(f"idf docs {n_docs} tokens {len(idf)} in {time.time() - t0:.0f}s", flush=True)

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = data["y"].astype(int)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    split = data["split"].astype(str)
    needed = set(s1.tolist()) | set(mid.tolist())
    print(f"rows {len(y)} ids {len(needed)}", flush=True)
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
    print(
        f"loaded names {len(names)} romanized {len(changed)} in {time.time() - t0:.0f}s",
        flush=True,
    )
    name_of = {eid: name_view(shown[eid]) for eid in needed}
    addr_of = {eid: parse_addr(addrs.get(eid, "")) for eid in needed}
    city_of = {eid: city_token(addrs.get(eid, "")) for eid in needed}
    del names, addrs
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    block = np.empty((len(y), 36), dtype=np.float32)
    block[:, 5:21] = X[:, 5:21]
    del X
    left_raw: dict[str, str] = {}
    rewritten = 0
    for i, (sid, eid) in enumerate(zip(s1, mid)):
        left_n = name_of.get(sid, empty_name)
        right_n = name_of.get(eid, empty_name)
        block[i, 0:5] = name_five(left_n, right_n)
        block[i, 30:32] = particle_two(left_n, right_n)
        block[i, 21:30] = extra_nine(
            addr_of.get(sid, empty_addr),
            addr_of.get(eid, empty_addr),
            left_n,
            right_n,
        )
        block[i, 32:34] = rare_two(left_n[1], right_n[1], idf, missing)
        block[i, 34:36] = city_two(city_of.get(sid, ""), city_of.get(eid, ""))
        if eid in raw_right:
            raw_l = left_raw.get(sid)
            if raw_l is None:
                raw_l = " ".join(tokens(shown.get(sid, "")))
                left_raw[sid] = raw_l
            block[i, RAW_RATIO_COL] = fuzz.ratio(raw_l, raw_right[eid]) / 100.0
            rewritten += 1
        if i and i % 500_000 == 0:
            print(f"  features {i}", flush=True)
    print(f"features in {time.time() - t0:.0f}s raw_ratio rewritten {rewritten}", flush=True)
    train = split == "train"
    val = split == "val"
    for col, label in ((32, "idf_shared"), (33, "idf_conflict"), (34, "city_jw"), (35, "city_eq")):
        pos = float(block[val][y[val] == 1, col].mean())
        neg = float(block[val][y[val] == 0, col].mean())
        print(f"val {label} true {pos:.3f} false {neg:.3f}", flush=True)
    indic_val = sum(1 for eid, is_val in zip(mid, val) if is_val and eid in changed)
    print(f"val pairs with a romanized name {indic_val}", flush=True)

    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "num_threads": 4,
            "seed": 7,
        },
        lgb.Dataset(block[train], label=y[train]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(block[val], label=y[val])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    trees = model.best_iteration if model.best_iteration and model.best_iteration > 0 else model.num_trees()
    scores = model.predict(block[val], num_iteration=trees)
    groups = _groups(s1, mid, np.where(val)[0], scores)
    f05 = _f05(groups, truths)
    single_gold = []
    single_pred = []
    ordered = [sid for sid, truth in truths.items()]
    preds = []
    gold = []
    for sid in ordered:
        truth = truths[sid]
        slot = groups.get(sid)
        chosen: set[str] = set()
        if slot is not None:
            order = np.argsort(-slot["scores"])
            for idx in order:
                if float(slot["scores"][idx]) < 0.70 or len(chosen) >= 12:
                    break
                chosen.add(slot["mids"][idx])
        preds.append(chosen)
        gold.append(truth)
        if not truth:
            single_gold.append(set())
            single_pred.append(chosen)
    single = macro_f05(single_gold, single_pred) if single_gold else 0.0
    print(
        f"val f05 {f05:.4f} delta {f05 - 0.9126:+.4f} singleton {single:.4f} trees {trees} in {time.time() - t0:.0f}s",
        flush=True,
    )
    check = macro_f05(gold, preds)
    if abs(check - f05) > 1e-9:
        print(f"decode check {check:.4f}", flush=True)
    model.save_model(str(MODEL))
    print(f"saved {MODEL}", flush=True)


if __name__ == "__main__":
    main()
