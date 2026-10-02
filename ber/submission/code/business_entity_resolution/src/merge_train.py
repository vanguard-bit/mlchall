"""Train the house matcher on the original 24k plus 24k extra rows. Same 6k validation ids."""

from __future__ import annotations

import json
import shutil
import time

import lightgbm as lgb
import numpy as np

from house_features import house_triple, parse_addr
from paths import DATA_DIR
from score_v3 import _f05, _groups

BOARD = DATA_DIR / "scoreboard"
TRAIN = DATA_DIR / "train"
MIN_KEEP = 0.895


def load_addrs(needed: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, _name, addr, _country = line.rstrip("\n").split("\t")
                if eid in needed:
                    found[eid] = addr
                    if len(found) == len(needed):
                        return found
    return found


def main() -> None:
    t0 = time.time()
    old = np.load(BOARD / "pairs.npz", allow_pickle=True)
    extra = np.load(BOARD / "extra_pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    extra_meta = json.loads((BOARD / "extra_meta.json").read_text(encoding="utf-8"))
    val_ids = {row["sid"] for row in meta if row["split"] == "val"}
    if val_ids & {row["sid"] for row in extra_meta}:
        raise SystemExit("extra rows overlap validation")
    old_split = old["split"].astype(str)
    train_old = old_split == "train"
    val = old_split == "val"
    X = np.concatenate([old["X"][train_old], extra["X"], old["X"][val]], axis=0)
    y = np.concatenate([old["y"][train_old], extra["y"], old["y"][val]], axis=0)
    s1 = np.concatenate([old["s1_id"][train_old], extra["s1_id"], old["s1_id"][val]])
    mid = np.concatenate([old["match_id"][train_old], extra["match_id"], old["match_id"][val]])
    split = np.concatenate([
        np.full(int(train_old.sum()) + len(extra["y"]), "train"),
        np.full(int(val.sum()), "val"),
    ])
    print(f"rows {len(y)} train {(split == 'train').sum()} val {(split == 'val').sum()}", flush=True)
    needed = set(s1.tolist()) | set(mid.tolist())
    addrs = load_addrs(needed)
    parsed = {eid: parse_addr(addr) for eid, addr in addrs.items()}
    empty = ("", frozenset())
    house = np.zeros((len(s1), 3), dtype=np.float32)
    for i, (sid, eid) in enumerate(zip(s1, mid)):
        house[i] = house_triple(parsed.get(str(sid), empty), parsed.get(str(eid), empty))
    block = np.concatenate([X, house], axis=1)
    train = split == "train"
    val_mask = split == "val"
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
            "num_threads": 12,
            "seed": 7,
        },
        lgb.Dataset(block[train], label=y[train].astype(int)),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(block[val_mask], label=y[val_mask].astype(int))],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    scores = model.predict(block[val_mask], num_iteration=model.best_iteration)
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    groups = _groups(s1, mid, np.where(val_mask)[0], scores)
    f05 = _f05(groups, truths)
    print(f"val f05 {f05:.4f} trees {model.best_iteration} in {time.time() - t0:.0f}s", flush=True)
    if f05 < MIN_KEEP:
        print("kept the 24k matcher", flush=True)
        return
    shutil.copy(BOARD / "pairs.npz", BOARD / "pairs_30k.npz")
    shutil.copy(BOARD / "meta.json", BOARD / "meta_30k.json")
    np.savez_compressed(
        BOARD / "pairs.npz",
        X=X,
        y=y.astype(np.int8),
        s1_id=s1,
        match_id=mid,
        split=split,
        feature_names=old["feature_names"],
    )
    (BOARD / "meta.json").write_text(json.dumps(meta + extra_meta), encoding="utf-8")
    model.save_model(str(BOARD / "lgbm_v3.txt"))
    print("wrote merged pairs and lgbm_v3", flush=True)


if __name__ == "__main__":
    main()
