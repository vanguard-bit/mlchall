"""Holdout set rule on v7 scores.

A moderate top score with no rare-token or house support becomes no match.
A later ID is kept only when it shares a rare token or the house with the
best ID, or with Source 1. Compared with greedy 0.70.
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


def max_shared_idf(left: frozenset[str], right: frozenset[str], idf: dict[str, float], missing: float) -> float:
    shared = left & right
    if not shared:
        return 0.0
    return max(idf.get(tok, missing) for tok in shared)


def house_eq(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def greedy(scored: list[tuple[str, float]]) -> set[str]:
    out: set[str] = set()
    for mid, score in scored:
        if score < FLOOR or len(out) >= 12:
            break
        out.add(mid)
    return out


def margin_keep(scored: list[tuple[str, float, frozenset[str], str]], gap: float) -> set[str]:
    """Keep later IDs only while they stay within `gap` of the top score."""
    eligible = [(mid, score) for mid, score, _tok, _house in scored if score >= FLOOR]
    if not eligible:
        return set()
    top = eligible[0][1]
    out: set[str] = set()
    for mid, score in eligible:
        if len(out) >= 12 or top - score > gap:
            break
        out.add(mid)
    return out
    out: set[str] = set()
    for mid, score in scored:
        if score < FLOOR or len(out) >= 12:
            break
        out.add(mid)
    return out


def veto(
    scored: list[tuple[str, float, frozenset[str], str]],
    s1_tok: frozenset[str],
    s1_house: str,
    high: float,
    idf_min: float,
    idf: dict[str, float],
    missing: float,
) -> set[str]:
    """Keep a moderate top match only with support. Later IDs need the same support."""
    eligible = [row for row in scored if row[1] >= FLOOR][:12]
    if not eligible:
        return set()
    best_mid, best_score, best_tok, best_house = eligible[0]

    def supported(tok: frozenset[str], house: str, other_tok: frozenset[str], other_house: str) -> bool:
        return max_shared_idf(tok, other_tok, idf, missing) >= idf_min or house_eq(house, other_house)

    best_ok = supported(best_tok, best_house, s1_tok, s1_house) or any(
        supported(tok, house, best_tok, best_house) for _mid, _score, tok, house in eligible[1:]
    )
    if best_score < high and not best_ok:
        return set()
    out = {best_mid}
    for mid, _score, tok, house in eligible[1:]:
        if supported(tok, house, best_tok, best_house) or supported(tok, house, s1_tok, s1_house):
            out.add(mid)
    return out


def singleton_only(
    scored: list[tuple[str, float, frozenset[str], str]],
    s1_tok: frozenset[str],
    s1_house: str,
    high: float,
    idf_min: float,
    idf: dict[str, float],
    missing: float,
) -> set[str]:
    """Blank the row only when the top score is moderate and unsupported."""
    eligible = [(mid, score) for mid, score, _tok, _house in scored if score >= FLOOR][:12]
    if not eligible:
        return set()
    _mid, best_score, best_tok, best_house = scored[0]
    rare = max_shared_idf(best_tok, s1_tok, idf, missing) >= idf_min
    if best_score < high and not (rare or house_eq(best_house, s1_house)):
        return set()
    return {mid for mid, _score in eligible}


def address_bar(
    scored: list[tuple[str, float, frozenset[str], str]],
    sid: str,
    addr_of: dict[str, tuple[str, frozenset[str], str]],
    high: float,
) -> set[str]:
    """0.70 when the address agrees. A higher bar when the name is the only evidence."""
    s_house, s_streets, s_zip = addr_of.get(sid, ("", frozenset(), ""))
    out: set[str] = set()
    for mid, score, _tok, _house in scored:
        if score < FLOOR or len(out) >= 12:
            break
        house, streets, zip5 = addr_of.get(mid, ("", frozenset(), ""))
        agrees = house_eq(house, s_house) or bool(streets and s_streets and (streets & s_streets)) or bool(
            zip5 and s_zip and zip5 == s_zip
        )
        if agrees or score >= high:
            out.add(mid)
    return out


def metrics(truths: dict[str, set[str]], preds: dict[str, set[str]]) -> tuple[float, float]:
    order = list(truths)
    gold = [truths[sid] for sid in order]
    got = [preds.get(sid, set()) for sid in order]
    single_gold = []
    single_pred = []
    for sid, truth in truths.items():
        if not truth:
            single_gold.append(set())
            single_pred.append(preds.get(sid, set()))
    single = macro_f05(single_gold, single_pred) if single_gold else 0.0
    return macro_f05(gold, got), single


def main() -> None:
    t0 = time.time()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = 1.0
    print(f"idf {len(idf)} docs {n_docs} in {time.time() - t0:.0f}s", flush=True)

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    val = data["split"].astype(str) == "val"
    s1 = s1[val]
    mid = mid[val]
    X = X[val]
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
    house_of = {eid: addr_of[eid][0] for eid in needed}
    del names
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
    print(f"features in {time.time() - t0:.0f}s", flush=True)
    booster = lgb.Booster(model_file=str(MODEL))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    scores = booster.predict(block, num_iteration=trees)
    groups: dict[str, list] = defaultdict(list)
    for sid, eid, score in zip(s1, mid, scores):
        groups[sid].append((
            eid,
            float(score),
            name_of.get(eid, empty_name)[1],
            house_of.get(eid, ""),
        ))
    for sid in groups:
        groups[sid].sort(key=lambda row: row[1], reverse=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    base = {
        sid: greedy([(eid, score) for eid, score, _tok, _house in groups.get(sid, [])])
        for sid in truths
    }
    f05, single = metrics(truths, base)
    print(f"greedy 0.70 f05 {f05:.4f} singleton {single:.4f}", flush=True)
    for gap in (0.08, 0.12, 0.18, 0.25):
        preds = {sid: margin_keep(groups.get(sid, []), gap) for sid in truths}
        f05, single = metrics(truths, preds)
        print(
            f"margin {gap:.2f} f05 {f05:.4f} delta {f05 - 0.9126:+.4f} singleton {single:.4f}",
            flush=True,
        )

    for high in (0.80, 0.85, 0.90, 0.95):
        preds = {
            sid: address_bar(groups.get(sid, []), sid, addr_of, high)
            for sid in truths
        }
        f05, single = metrics(truths, preds)
        print(
            f"addr-or-high {high:.2f} f05 {f05:.4f} delta {f05 - 0.9126:+.4f} singleton {single:.4f}",
            flush=True,
        )

    for idf_min in (0.40, 0.50, 0.60):
        for high in (0.85, 0.90, 0.95):
            preds = {}
            for sid in truths:
                s1_tok = name_of.get(sid, empty_name)[1]
                preds[sid] = veto(
                    groups.get(sid, []),
                    s1_tok,
                    house_of.get(sid, ""),
                    high,
                    idf_min,
                    idf,
                    missing,
                )
            f05, single = metrics(truths, preds)
            print(
                f"set idf>={idf_min:.2f} high {high:.2f} f05 {f05:.4f} "
                f"delta {f05 - 0.9126:+.4f} singleton {single:.4f}",
                flush=True,
            )
            preds = {}
            for sid in truths:
                preds[sid] = singleton_only(
                    groups.get(sid, []),
                    name_of.get(sid, empty_name)[1],
                    house_of.get(sid, ""),
                    high,
                    idf_min,
                    idf,
                    missing,
                )
            f05, single = metrics(truths, preds)
            print(
                f"top-only idf>={idf_min:.2f} high {high:.2f} f05 {f05:.4f} "
                f"delta {f05 - 0.9126:+.4f} singleton {single:.4f}",
                flush=True,
            )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
