"""Set-size and address-cluster decoders on the saved holdout. No corpus scan."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from f05 import entity_f05, macro_f05
from paths import DATA_DIR

BOARD = DATA_DIR / "scoreboard"


def prepare_groups(s1, mid, scores, X, split):
    groups: dict[str, dict] = {}
    for row, score in enumerate(scores):
        sid = s1[row]
        slot = groups.get(sid)
        if slot is None:
            slot = {"mids": [], "scores": [], "house": [], "addr": [], "split": split[row]}
            groups[sid] = slot
        slot["mids"].append(mid[row])
        slot["scores"].append(float(score))
        slot["house"].append(float(X[row, 10]))
        slot["addr"].append(float(X[row, 7]))
    for group in groups.values():
        order = np.argsort(-np.asarray(group["scores"]))
        group["mids"] = [group["mids"][i] for i in order]
        group["scores"] = np.asarray([group["scores"][i] for i in order], dtype=np.float32)
        group["house"] = np.asarray([group["house"][i] for i in order], dtype=np.float32)
        group["addr"] = np.asarray([group["addr"][i] for i in order], dtype=np.float32)
    return groups


def score_decoder(groups, truths, choose) -> float:
    return macro_f05(
        [truths[sid] for sid in groups],
        [set(choose(groups[sid])) for sid in groups],
    )


def top_ids(group, k: int, floor: float = 0.0) -> list[str]:
    out = []
    for mid, score in zip(group["mids"], group["scores"]):
        if len(out) >= k or float(score) < floor:
            break
        out.append(str(mid))
    return out


def cluster(group, high: float, low: float) -> list[str]:
    if len(group["scores"]) == 0 or float(group["scores"][0]) < high:
        return []
    out = [str(group["mids"][0])]
    for mid, score, house, addr in zip(
        group["mids"][1:], group["scores"][1:], group["house"][1:], group["addr"][1:]
    ):
        if float(score) < low or len(out) >= 12:
            break
        if house >= 0.5 or addr > 0:
            out.append(str(mid))
    return out


def oracle_k(group, truth: set[str]) -> int:
    best_score, choice = -1.0, 0
    for k in range(0, min(12, len(group["mids"])) + 1):
        score = entity_f05(truth, set(top_ids(group, k)))
        if score > best_score:
            best_score, choice = score, k
    return choice


def row_features(group) -> list[float]:
    scores = group["scores"]
    n = len(scores)
    top = float(scores[0]) if n else 0.0
    second = float(scores[1]) if n > 1 else 0.0
    return [
        float(n),
        top,
        second,
        top - second,
        float(np.sum(scores >= 0.5)) if n else 0.0,
        float(np.sum(scores >= 0.7)) if n else 0.0,
        float(np.sum(scores >= 0.85)) if n else 0.0,
        float(np.mean(scores[:3])) if n else 0.0,
        float(np.mean(group["house"][:5])) if n else 0.0,
    ]


def expected_counts(tp: float, fp: float, fn: float) -> float:
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision == 0 or recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def main() -> None:
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v2.txt"))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    scores = booster.predict(data["X"], num_iteration=trees)
    groups = prepare_groups(
        data["s1_id"].astype(str),
        data["match_id"].astype(str),
        scores,
        data["X"],
        data["split"].astype(str),
    )
    truths = {row["sid"]: set(row["truth"]) for row in meta}
    splits = {row["sid"]: row["split"] for row in meta}
    for sid, split in splits.items():
        groups.setdefault(
            sid,
            {
                "mids": [],
                "scores": np.asarray([], dtype=np.float32),
                "house": np.asarray([], dtype=np.float32),
                "addr": np.asarray([], dtype=np.float32),
                "split": split,
            },
        )
    train = {sid: group for sid, group in groups.items() if splits[sid] == "train"}
    val = {sid: group for sid, group in groups.items() if splits[sid] == "val"}
    val_truth = {sid: truths[sid] for sid in val}
    base = score_decoder(val, val_truth, lambda group: top_ids(group, 12, 0.70))
    print(f"threshold 0.70 {base:.4f}", flush=True)
    oracle_preds = [set(top_ids(val[sid], oracle_k(val[sid], val_truth[sid]))) for sid in val]
    print(f"oracle set size {macro_f05(list(val_truth.values()), oracle_preds):.4f}", flush=True)

    best, best_params = base, {"kind": "threshold", "cutoff": 0.70}
    for high in (0.75, 0.80, 0.85, 0.90):
        for low in (0.40, 0.50, 0.60):
            score = score_decoder(val, val_truth, lambda group, h=high, l=low: cluster(group, h, l))
            print(f"cluster high {high:.2f} low {low:.2f} {score:.4f}", flush=True)
            if score > best:
                best, best_params = score, {"kind": "cluster", "high": high, "low": low}

    count_model = lgb.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 2,
            "learning_rate": 0.05,
            "num_leaves": 31,
        },
        lgb.Dataset(
            np.asarray([row_features(group) for group in train.values()]),
            label=np.asarray([oracle_k(group, truths[sid]) for sid, group in train.items()]),
        ),
        num_boost_round=200,
    )

    def by_count(group, floor: float) -> list[str]:
        guess = float(count_model.predict(np.asarray([row_features(group)]))[0])
        return top_ids(group, int(np.clip(round(guess), 0, 12)), floor)

    for floor in (0.0, 0.30, 0.50, 0.70):
        score = score_decoder(val, val_truth, lambda group, f=floor: by_count(group, f))
        print(f"cardinality floor {floor:.1f} {score:.4f}", flush=True)
        if score > best:
            best, best_params = score, {"kind": "cardinality", "floor": floor}

    def expected(group) -> list[str]:
        probs = np.clip(group["scores"], 1e-4, 1 - 1e-4)
        mass = float(np.sum(probs))
        best_k, best_f, running = 0, expected_counts(0, 0, mass), 0.0
        for k, prob in enumerate(probs[:12], start=1):
            running += float(prob)
            value = expected_counts(running, k - running, max(0.0, mass - running))
            if value > best_f:
                best_f, best_k = value, k
        return top_ids(group, best_k)

    expected_score = score_decoder(val, val_truth, expected)
    print(f"expected F0.5 {expected_score:.4f}", flush=True)
    if expected_score > best:
        best, best_params = expected_score, {"kind": "expected"}
    print(f"best {best:.4f} lift {best - base:+.4f} {best_params}", flush=True)
    (BOARD / "set_decoder.json").write_text(
        json.dumps({"base_070": base, "f05": best, **best_params}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
