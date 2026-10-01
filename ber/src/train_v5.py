"""Train v5: v4 features plus a French-particle-stripped alternate name."""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from paths import DATA_DIR
from score_v3 import _f05, _groups, _load_addrs
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two

BOARD = DATA_DIR / "scoreboard"
MODEL = BOARD / "lgbm_v5.txt"


def main() -> None:
    t0 = time.time()
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
    train = DATA_DIR / "train"
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (train / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, _addr, _country = line.rstrip("\n").split("\t")
                if eid in needed and eid not in names:
                    names[eid] = bname
                    if len(names) == len(needed):
                        break
        if len(names) == len(needed):
            break
    print(f"loaded names {len(names)} addrs {len(addrs)} in {time.time() - t0:.0f}s", flush=True)
    name_of = {eid: name_view(names.get(eid, "")) for eid in needed}
    addr_of = {eid: parse_addr(addrs.get(eid, "")) for eid in needed}
    del names, addrs
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    block = np.empty((len(y), 32), dtype=np.float32)
    block[:, 5:21] = X[:, 5:21]
    del X
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
        if i and i % 500_000 == 0:
            print(f"  features {i}", flush=True)
    print(f"features in {time.time() - t0:.0f}s", flush=True)
    train = split == "train"
    val = split == "val"
    import json

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
            "num_threads": 8,
            "seed": 7,
        },
        lgb.Dataset(block[train], label=y[train]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(block[val], label=y[val])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    scores = model.predict(block[val], num_iteration=model.best_iteration)
    groups = _groups(s1, mid, np.where(val)[0], scores)
    f05 = _f05(groups, truths)
    print(f"val f05 {f05:.4f} trees {model.best_iteration} in {time.time() - t0:.0f}s", flush=True)
    if f05 < 0.890:
        raise SystemExit(f"v5 val {f05:.4f} is too far below v4 to queue")
    model.save_model(str(MODEL))
    print(f"saved {MODEL}", flush=True)


if __name__ == "__main__":
    main()
