"""Train on the scoreboard train split and sweep decoders on full validation truth."""

from __future__ import annotations

import json
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import macro_f05
from paths import DATA_DIR

BOARD = DATA_DIR / "scoreboard"
HOUSE_COL = 10
ADDR_JACCARD_COL = 7


def decode(
    scores: np.ndarray,
    house: np.ndarray,
    addr_j: np.ndarray,
    *,
    min_gain: float,
    house_from: int,
    house_gain: float,
) -> list[int]:
    order = np.argsort(-scores)
    chosen: list[int] = []
    for rank, idx in enumerate(order):
        score = float(scores[idx])
        if score < min_gain:
            break
        if rank >= house_from and house[idx] < 0.5 and addr_j[idx] <= 0 and score < house_gain:
            continue
        chosen.append(int(idx))
        if len(chosen) >= 12:
            break
    return chosen


def evaluate(groups: dict[str, dict], truths: dict[str, set[str]], **params) -> float:
    preds = []
    gold = []
    for sid, group in groups.items():
        chosen = decode(group["scores"], group["house"], group["addr"], **params)
        preds.append({group["mids"][i] for i in chosen})
        gold.append(truths[sid])
    for sid, truth in truths.items():
        if sid not in groups:
            preds.append(set())
            gold.append(truth)
    return macro_f05(gold, preds)


def main() -> None:
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    X = data["X"]
    y = data["y"].astype(int)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    split = data["split"].astype(str)
    train = split == "train"
    val = split == "val"
    booster = lgb.train(
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
            "num_threads": 12,
        },
        lgb.Dataset(X[train], label=y[train]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(X[val], label=y[val])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    scores = booster.predict(X[val], num_iteration=booster.best_iteration)
    groups: dict[str, dict] = {}
    cursor: dict[str, int] = defaultdict(int)
    val_idx = np.where(val)[0]
    for row, score in zip(val_idx, scores):
        sid = s1[row]
        slot = groups.get(sid)
        if slot is None:
            slot = {"mids": [], "scores": [], "house": [], "addr": []}
            groups[sid] = slot
        slot["mids"].append(mid[row])
        slot["scores"].append(float(score))
        slot["house"].append(float(X[row, HOUSE_COL]))
        slot["addr"].append(float(X[row, ADDR_JACCARD_COL]))
        cursor[sid] += 1
    for slot in groups.values():
        slot["scores"] = np.asarray(slot["scores"], dtype=np.float32)
        slot["house"] = np.asarray(slot["house"], dtype=np.float32)
        slot["addr"] = np.asarray(slot["addr"], dtype=np.float32)
    truths = {
        row["sid"]: set(row["truth"])
        for row in meta
        if row["split"] == "val"
    }
    best = (-1.0, {})
    print(f"val entities {len(truths)} scored entities {len(groups)}", flush=True)
    for min_gain in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        for house_from, house_gain in ((99, 1.0), (3, 0.90), (3, 0.95), (2, 0.90)):
            score = evaluate(
                groups,
                truths,
                min_gain=min_gain,
                house_from=house_from,
                house_gain=house_gain,
            )
            params = {"min_gain": min_gain, "house_from": house_from, "house_gain": house_gain}
            print(f"F0.5 {score:.4f} {params}", flush=True)
            if score > best[0]:
                best = (score, params)
    booster.save_model(str(BOARD / "lgbm_v2.txt"))
    (BOARD / "decoder.json").write_text(
        json.dumps({"f05": best[0], **best[1], "trees": booster.best_iteration}),
        encoding="utf-8",
    )
    print(f"best {best[0]:.4f} {best[1]}", flush=True)


if __name__ == "__main__":
    main()
