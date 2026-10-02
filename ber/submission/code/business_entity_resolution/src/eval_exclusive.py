"""Holdout target exclusivity on v7 scores.

Ground truth gives every Source 2/3 record to at most one Source 1.
When two holdout Source 1 rows both clear 0.70 on the same target, only the
higher score keeps it.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import macro_f05
from paths import DATA_DIR
from score_v3 import _load_addrs
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two

BOARD = DATA_DIR / "scoreboard"
MODEL = BOARD / "lgbm_v7.txt"
FLOOR = 0.70


def exclusive(
    groups: dict[str, list[tuple[str, float]]],
    truths: dict[str, set[str]],
    margin: float,
) -> tuple[dict[str, set[str]], int]:
    claims: dict[str, list[tuple[str, float]]] = defaultdict(list)
    base: dict[str, set[str]] = {}
    for sid in truths:
        rows = sorted(groups.get(sid, []), key=lambda item: item[1], reverse=True)
        chosen = greedy_rows(rows)
        base[sid] = chosen
        for mid, score in rows:
            if mid in chosen:
                claims[mid].append((sid, score))
    drop: set[tuple[str, str]] = set()
    contested = 0
    for _mid, owners in claims.items():
        if len(owners) < 2:
            continue
        contested += 1
        owners.sort(key=lambda item: item[1], reverse=True)
        _best_sid, best_score = owners[0]
        for sid, score in owners[1:]:
            if best_score - score >= margin:
                drop.add((sid, _mid))
    preds = {}
    for sid, chosen in base.items():
        preds[sid] = {mid for mid in chosen if (sid, mid) not in drop}
    return preds, contested


def main() -> None:
    t0 = time.time()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = 1.0
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    val = data["split"].astype(str) == "val"
    s1, mid, X = s1[val], mid[val], X[val]
    needed = set(s1.tolist()) | set(mid.tolist())
    print(f"val pairs {len(s1)} ids {len(needed)}", flush=True)
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
    name_of = {eid: name_view(names.get(eid, "")) for eid in needed}
    addr_of = {eid: parse_addr(addrs.get(eid, "")) for eid in needed}
    city_of = {eid: city_token(addrs.get(eid, "")) for eid in needed}
    del names, addrs
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    block = np.empty((len(s1), 36), dtype=np.float32)
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
        block[i, 32:34] = rare_two(left_n[1], right_n[1], idf, missing)
        block[i, 34:36] = city_two(city_of.get(sid, ""), city_of.get(eid, ""))
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    scores = booster.predict(block, num_iteration=trees)
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for sid, eid, score in zip(s1, mid, scores):
        groups[sid].append((eid, float(score)))
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    base = {}
    for sid in truths:
        rows = sorted(groups.get(sid, []), key=lambda item: item[1], reverse=True)
        base[sid] = greedy_rows(rows)
    f05, single = metrics(truths, base)
    print(f"greedy f05 {f05:.4f} singleton {single:.4f} in {time.time() - t0:.0f}s", flush=True)
    for margin in (0.0, 0.02, 0.05, 0.10):
        preds, contested = exclusive(groups, truths, margin)
        f05, single = metrics(truths, preds)
        removed_tp = removed_fp = 0
        for sid, truth in truths.items():
            lost = base[sid] - preds[sid]
            removed_tp += len(lost & truth)
            removed_fp += len(lost - truth)
        print(
            f"margin {margin:.2f} contested {contested} removed tp {removed_tp} fp {removed_fp} "
            f"f05 {f05:.4f} delta {f05 - 0.9126:+.4f} singleton {single:.4f}",
            flush=True,
        )


def greedy_rows(rows: list[tuple[str, float]]) -> set[str]:
    out: set[str] = set()
    for mid, score in rows:
        if score < FLOOR or len(out) >= 12:
            break
        out.add(mid)
    return out


def metrics(truths: dict[str, set[str]], preds: dict[str, set[str]]) -> tuple[float, float]:
    order = list(truths)
    gold = [truths[sid] for sid in order]
    got = [preds.get(sid, set()) for sid in order]
    single_g, single_p = [], []
    for sid, truth in truths.items():
        if not truth:
            single_g.append(set())
            single_p.append(preds.get(sid, set()))
    single = macro_f05(single_g, single_p) if single_g else 0.0
    return macro_f05(gold, got), single


if __name__ == "__main__":
    main()
