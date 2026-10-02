"""Retrain the matcher with a parsed house / street conflict feature. Validation fold only."""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import macro_f05
from normalize import ADDR_STOP, LEGAL, digit_tokens, tokens
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
NEW_NAMES = ("house_same", "house_missing", "house_conflict")

TEXTS: dict[str, str] = {}
NEEDED: set[str] = set()


def parse_addr(addr: str) -> tuple[str, frozenset[str]]:
    """First house-like number, then the street tokens that follow it."""
    house = ""
    streets: list[str] = []
    for tok in tokens(addr):
        digits = [d for d in digit_tokens(tok) if 1 <= len(d) <= 6]
        if not house and digits:
            house = digits[0].lstrip("0") or "0"
            continue
        if house and tok not in ADDR_STOP and tok not in LEGAL and len(tok) >= 4 and not tok.isdigit():
            streets.append(tok)
    if len(streets) >= 2:
        streets = streets[:-1]
    return house, frozenset(streets[:4])


def _parse_worker(ids: list[str]) -> dict[str, tuple[str, frozenset[str]]]:
    return {eid: parse_addr(TEXTS[eid]) for eid in ids}


def address_features(parsed: dict[str, tuple[str, frozenset[str]]], s1: np.ndarray, mid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(s1), 3), dtype=np.float32)
    empty = ("", frozenset())
    for i, (sid, eid) in enumerate(zip(s1, mid)):
        h1, st1 = parsed.get(str(sid), empty)
        h2, st2 = parsed.get(str(eid), empty)
        if not h1 or not h2:
            out[i, 1] = 1.0
            continue
        same = h1 == h2
        out[i, 0] = float(same)
        out[i, 2] = float((not same) and bool(st1 & st2))
    return out


def _load_worker(path: str) -> dict[str, str]:
    found: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, _name, addr, _country = line.rstrip("\n").split("\t")
            if eid in NEEDED:
                found[eid] = addr
    return found


def load_texts(needed: set[str]) -> dict[str, str]:
    global NEEDED
    NEEDED = needed
    ctx = mp.get_context("fork")
    files = [str(TRAIN / name) for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")]
    found: dict[str, str] = {}
    with ctx.Pool(3) as pool:
        for part in pool.map(_load_worker, files):
            found.update(part)
    return found


def train_model(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> lgb.Booster:
    return lgb.train(
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
            "seed": 7,
        },
        lgb.Dataset(X[train], label=y[train]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(X[val], label=y[val])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )


def evaluate(groups: dict[str, dict], truths: dict[str, set[str]], min_gain: float) -> tuple[float, float]:
    preds: list[set[str]] = []
    gold: list[set[str]] = []
    single_p: list[set[str]] = []
    single_g: list[set[str]] = []
    for sid, truth in truths.items():
        slot = groups.get(sid)
        chosen: set[str] = set()
        if slot is not None:
            order = np.argsort(-slot["scores"])
            for rank, idx in enumerate(order):
                if float(slot["scores"][idx]) < min_gain or rank >= 12:
                    if float(slot["scores"][idx]) < min_gain:
                        break
                    if rank >= 12:
                        break
                chosen.add(slot["mids"][idx])
                if len(chosen) >= 12:
                    break
        preds.append(chosen)
        gold.append(truth)
        if not truth:
            single_p.append(chosen)
            single_g.append(set())
    return macro_f05(gold, preds), macro_f05(single_g, single_p)


def group_scores(s1: np.ndarray, mid: np.ndarray, rows: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for row, score in zip(rows, scores):
        sid = str(s1[row])
        slot = groups.get(sid)
        if slot is None:
            slot = {"mids": [], "scores": []}
            groups[sid] = slot
        slot["mids"].append(str(mid[row]))
        slot["scores"].append(float(score))
    for slot in groups.values():
        slot["scores"] = np.asarray(slot["scores"], dtype=np.float32)
    return groups


def main() -> None:
    global TEXTS
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    X = data["X"]
    y = data["y"].astype(int)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    split = data["split"].astype(str)
    train = split == "train"
    val = split == "val"
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    needed = set(s1[val].tolist()) | set(mid[val].tolist()) | set(s1[train].tolist()) | set(mid[train].tolist())
    print(f"pairs {len(s1)} unique ids {len(needed)}", flush=True)
    TEXTS = load_texts(needed)
    print(f"loaded addresses {len(TEXTS)}", flush=True)
    ids = list(TEXTS)
    ctx = mp.get_context("fork")
    chunks = [ids[i::WORKERS] for i in range(WORKERS)]
    parsed: dict[str, tuple[str, frozenset[str]]] = {}
    with ctx.Pool(WORKERS) as pool:
        for part in pool.map(_parse_worker, chunks):
            parsed.update(part)
    extra = address_features(parsed, s1, mid)
    del parsed, TEXTS
    print(
        f"house_same {(extra[:, 0] == 1).mean():.3f} missing {(extra[:, 1] == 1).mean():.3f} "
        f"conflict {(extra[:, 2] == 1).mean():.3f}",
        flush=True,
    )
    for col, name in enumerate(NEW_NAMES):
        on = extra[val, col] == 1
        print(
            f"val {name} rate {on.mean():.3f} positive-rate {y[val][on].mean():.3f} "
            f"off-positive {y[val][~on].mean():.3f}",
            flush=True,
        )
    Xnew = np.concatenate([X, extra], axis=1)
    base_model = lgb.Booster(model_file=str(BOARD / "lgbm_v2.txt"))
    base_scores = base_model.predict(X[val])
    fresh = train_model(X, y, train, val)
    fresh_scores = fresh.predict(X[val], num_iteration=fresh.best_iteration)
    new_model = train_model(Xnew, y, train, val)
    new_scores = new_model.predict(Xnew[val], num_iteration=new_model.best_iteration)
    print(f"trees baseline-refit {fresh.best_iteration} addr {new_model.best_iteration}", flush=True)
    names = [*data["feature_names"].astype(str).tolist(), *NEW_NAMES]
    gain = sorted(zip(names, new_model.feature_importance(), strict=True), key=lambda item: item[1], reverse=True)
    print("importance " + " ".join(f"{name}:{count}" for name, count in gain[:8]), flush=True)
    val_rows = np.where(val)[0]
    base_groups = group_scores(s1, mid, val_rows, base_scores)
    fresh_groups = group_scores(s1, mid, val_rows, fresh_scores)
    new_groups = group_scores(s1, mid, val_rows, new_scores)
    def link_counts(groups: dict[str, dict], min_gain: float) -> tuple[int, int]:
        tp = fp = 0
        for sid, truth in truths.items():
            slot = groups.get(sid)
            chosen: set[str] = set()
            if slot is not None:
                order = np.argsort(-slot["scores"])
                for idx in order:
                    if float(slot["scores"][idx]) < min_gain or len(chosen) >= 12:
                        break
                    chosen.add(slot["mids"][idx])
            tp += len(chosen & truth)
            fp += len(chosen - truth)
        return tp, fp

    btp, bfp = link_counts(base_groups, 0.70)
    ntp, nfp = link_counts(new_groups, 0.70)
    print(f"links at 0.70 shipped tp {btp} fp {bfp} addr tp {ntp} fp {nfp}", flush=True)
    for min_gain in (0.60, 0.70, 0.75, 0.80, 0.85):
        b, bs = evaluate(base_groups, truths, min_gain)
        f, fs = evaluate(fresh_groups, truths, min_gain)
        n, ns = evaluate(new_groups, truths, min_gain)
        print(
            f"cut {min_gain:.2f} shipped {b:.4f} refit {f:.4f} addr {n:.4f} "
            f"delta {n - b:+.4f} singleton shipped {bs:.4f} addr {ns:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
