"""Train LightGBM pair classifier and tune decode threshold on held-out S1."""

from __future__ import annotations

import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05, decode_greedy_f05_labeled, tune_min_gain
from f05 import macro_f05
from paths import DATA_DIR

DATA_PATH = DATA_DIR / "matcher_sample.npz"
MODEL_PATH = DATA_DIR / "lgbm_matcher.txt"


def main() -> None:
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    s1_id = data["s1_id"]
    match_id = data["match_id"]
    print(f"loaded {X.shape[0]} pairs dim {X.shape[1]}", flush=True)

    s1_unique = np.unique(s1_id)
    rng = np.random.RandomState(42)
    rng.shuffle(s1_unique)
    n_val = max(1, len(s1_unique) // 5)
    val_s1 = set(s1_unique[:n_val])
    tr_mask = np.array([s not in val_s1 for s in s1_id])
    va_mask = ~tr_mask

    dtrain = lgb.Dataset(X[tr_mask], label=y[tr_mask])
    dval = lgb.Dataset(X[va_mask], label=y[va_mask], reference=dtrain)
    params = {
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
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    booster.save_model(str(MODEL_PATH))
    print(f"saved {MODEL_PATH} best_iter {booster.best_iteration}", flush=True)

    p_val = booster.predict(X[va_mask], num_iteration=booster.best_iteration)
    by_s1: dict[str, list[tuple[str, float]]] = defaultdict(list)
    truths: dict[str, set[str]] = defaultdict(set)
    val_idx = np.where(va_mask)[0]
    for j, i in enumerate(val_idx):
        sid = str(s1_id[i])
        mid = str(match_id[i])
        by_s1[sid].append((mid, float(p_val[j])))
        if y[i]:
            truths[sid].add(mid)
    for sid in by_s1:
        truths.setdefault(sid, set())

    oracle_preds = []
    oracle_truths = []
    for sid, pairs in by_s1.items():
        cids = [p[0] for p in pairs]
        scs = [p[1] for p in pairs]
        oracle_preds.append(decode_greedy_f05_labeled(cids, scs, truths[sid]))
        oracle_truths.append(truths[sid])
    print(f"val macro F0.5 oracle decode {macro_f05(oracle_truths, oracle_preds):.4f}", flush=True)

    val_rows = [( [p[0] for p in pairs], [p[1] for p in pairs], truths[sid]) for sid, pairs in by_s1.items()]
    best_g, best_m = tune_min_gain(val_rows)
    print(f"val macro F0.5 threshold decode min_gain={best_g:.2f} -> {best_m:.4f}", flush=True)

    imp = booster.feature_importance(importance_type="gain")
    names = list(data["feature_names"])
    order = np.argsort(-imp)[:10]
    print("top features:", flush=True)
    for i in order:
        print(f"  {names[i]}: {imp[i]:.1f}", flush=True)


if __name__ == "__main__":
    main()
