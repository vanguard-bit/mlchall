"""Score reverse-only holdout nominations with the shipped matcher and re-decode."""

from __future__ import annotations

import json
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from f05 import entity_f05, macro_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"


def load_texts(needed: set[str]) -> dict[str, tuple[str, str, str]]:
    found: dict[str, tuple[str, str, str]] = {}
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        if len(found) == len(needed):
            break
        with (TRAIN / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, addr, country = line.rstrip("\n").split("\t")
                if eid in needed:
                    found[eid] = (bname, addr, country)
                    if len(found) == len(needed):
                        break
    return found


def decode_ids(mids: list[str], scores: list[float], min_gain: float = 0.70) -> set[str]:
    order = sorted(range(len(mids)), key=lambda i: scores[i], reverse=True)
    chosen: list[str] = []
    for i in order:
        if scores[i] < min_gain:
            break
        chosen.append(mids[i])
        if len(chosen) >= 12:
            break
    return set(chosen)


def main() -> None:
    rev = np.load(BOARD / "reverse_val.npz", allow_pickle=True)
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v2.txt"))
    X = data["X"]
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    split = data["split"].astype(str)
    val = split == "val"
    fwd_scores = booster.predict(X[val])
    groups: dict[str, dict] = {}
    for row, score in zip(np.where(val)[0], fwd_scores):
        sid = s1[row]
        slot = groups.get(sid)
        if slot is None:
            slot = {"mids": [], "scores": []}
            groups[sid] = slot
        slot["mids"].append(mid[row])
        slot["scores"].append(float(score))
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    singleton = {sid for sid, truth in truths.items() if not truth}
    base_pred = {
        sid: decode_ids(groups[sid]["mids"], groups[sid]["scores"]) if sid in groups else set()
        for sid in truths
    }
    base = macro_f05([truths[sid] for sid in truths], [base_pred[sid] for sid in truths])
    base_single = macro_f05(
        [set() for _ in singleton],
        [base_pred[sid] for sid in singleton],
    )
    print(f"baseline f05 {base:.4f} singleton {base_single:.4f} n {len(truths)}", flush=True)

    new = rev["in_forward"] == 0
    need = set(rev["s1_id"][new].astype(str)) | set(rev["match_id"][new].astype(str))
    texts = load_texts(need)
    prepared = {eid: prepare_record(name, addr) for eid, (name, addr, _c) in texts.items()}
    countries = {eid: country for eid, (_n, _a, country) in texts.items()}

    feat_hidden = []
    feat_marked = []
    labels = []
    owners = []
    mids = []
    ranks = []
    for sid, eid, rank, y, fwd in zip(
        rev["s1_id"].astype(str),
        rev["match_id"].astype(str),
        rev["rev_rank"].astype(int),
        rev["y"].astype(int),
        rev["in_forward"].astype(int),
    ):
        if fwd:
            continue
        left = prepared.get(sid)
        right = prepared.get(eid)
        if left is None or right is None:
            continue
        country_eq = float(countries[sid] == countries[eid])
        feat_hidden.append(
            features_prepared(
                left,
                right,
                country_eq=country_eq,
                name_rank=99.0,
                addr_rank=99.0,
            )
        )
        feat_marked.append(
            features_prepared(
                left,
                right,
                country_eq=country_eq,
                name_rank=float(25 + rank),
                addr_rank=float(25 + rank),
                name_score=1.0,
                addr_score=1.0,
                from_name_channel=1,
                from_addr_channel=1,
            )
        )
        labels.append(y)
        owners.append(sid)
        mids.append(eid)
        ranks.append(rank)
    hidden = booster.predict(np.asarray(feat_hidden, dtype=np.float32))
    marked = booster.predict(np.asarray(feat_marked, dtype=np.float32))
    labels_a = np.asarray(labels)
    ranks_a = np.asarray(ranks)
    print(
        f"reverse-only pairs {len(labels_a)} positives {(labels_a == 1).sum()}",
        flush=True,
    )
    for name, scores in (("hidden-rank", hidden), ("marked-channel", marked)):
        for cutoff in (0.50, 0.70, 0.85, 0.90):
            keep = scores >= cutoff
            tp = int(((labels_a == 1) & keep).sum())
            fp = int(((labels_a == 0) & keep).sum())
            print(f"  {name} >= {cutoff:.2f} tp {tp} fp {fp}", flush=True)
        for k in (1, 3, 5):
            keep = (ranks_a <= k) & (scores >= 0.70)
            tp = int(((labels_a == 1) & keep).sum())
            fp = int(((labels_a == 0) & keep).sum())
            print(f"  {name} top{k} and >= 0.70 tp {tp} fp {fp}", flush=True)

    by_sid: dict[str, list[tuple[str, float, float, int, int]]] = defaultdict(list)
    for sid, eid, y, rank, h, m in zip(owners, mids, labels, ranks, hidden, marked):
        by_sid[sid].append((eid, float(h), float(m), int(rank), int(y)))

    order = list(truths)
    gold = [truths[sid] for sid in order]

    def report(tag: str, score_idx: int, cutoff: float, max_rank: int) -> None:
        preds = []
        added_tp = added_fp = 0
        for sid in order:
            base_ids = set(base_pred[sid])
            extra = []
            extra_scores = []
            for eid, h, m, rank, y in by_sid.get(sid, []):
                if rank > max_rank or eid in base_ids:
                    continue
                score = h if score_idx == 0 else m
                if score < cutoff:
                    continue
                extra.append(eid)
                extra_scores.append(score)
                if y:
                    added_tp += 1
                else:
                    added_fp += 1
            slot = groups.get(sid)
            if slot is None:
                pred = decode_ids(extra, extra_scores, cutoff)
            else:
                pred = decode_ids(
                    slot["mids"] + extra,
                    slot["scores"] + extra_scores,
                    cutoff,
                )
            preds.append(pred)
        score = macro_f05(gold, preds)
        single_preds = [preds[i] for i, sid in enumerate(order) if sid in singleton]
        single = macro_f05([set() for _ in singleton], single_preds)
        print(
            f"{tag} f05 {score:.4f} delta {score - base:+.4f} "
            f"singleton {single:.4f} added tp {added_tp} fp {added_fp}",
            flush=True,
        )

    for cutoff in (0.70, 0.85, 0.90):
        for max_rank in (1, 3, 15):
            report(f"hidden {cutoff} rank<={max_rank}", 0, cutoff, max_rank)
            report(f"marked {cutoff} rank<={max_rank}", 1, cutoff, max_rank)

    shown_tp = shown_fp = 0
    order_idx = np.argsort(-hidden)
    for i in order_idx:
        if labels[i] != 1 or shown_tp >= 8:
            continue
        sid, eid = owners[i], mids[i]
        ln, la, _lc = texts[sid]
        rn, ra, _rc = texts[eid]
        print(f"TP {hidden[i]:.3f} {marked[i]:.3f} | {ln} || {la} <> {rn} || {ra}", flush=True)
        shown_tp += 1
    for i in order_idx:
        if labels[i] != 0 or shown_fp >= 8:
            continue
        if hidden[i] < 0.5:
            break
        sid, eid = owners[i], mids[i]
        ln, la, _lc = texts[sid]
        rn, ra, _rc = texts[eid]
        print(f"FP {hidden[i]:.3f} {marked[i]:.3f} | {ln} || {la} <> {rn} || {ra}", flush=True)
        shown_fp += 1


if __name__ == "__main__":
    main()
