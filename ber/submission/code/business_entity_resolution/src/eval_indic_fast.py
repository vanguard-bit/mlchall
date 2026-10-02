"""Fast v5 F0.5 if IndicXlit names are added for missed Indic pairs."""

from __future__ import annotations

import json
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from rapidfuzz.distance import JaroWinkler

from decode import decode_greedy_f05
from f05 import macro_f05
from normalize import content_name_tokens
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two

BOARD = DATA_DIR / "scoreboard"
TRAIN = DATA_DIR / "train"


def norm(text: str) -> str:
    return " ".join(content_name_tokens(text))


def jw(a: str, b: str) -> float:
    la, lb = norm(a), norm(b)
    if not la or not lb:
        return 0.0
    return JaroWinkler.normalized_similarity(la, lb)


def indic(text: str) -> bool:
    return any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF for ch in text)


def main() -> None:
    rows = json.loads((BOARD / "indic_misses.json").read_text(encoding="utf-8"))
    keep = []
    for row in rows:
        latin = row["m_name"] if indic(row["s1_name"]) else row["s1_name"]
        score = jw(row.get("xlit") or "", latin)
        exact = norm(row.get("xlit") or "") == norm(latin)
        if score >= 0.90 or exact:
            row["s1_indic"] = indic(row["s1_name"])
            keep.append(row)
    print(f"adding {len(keep)} links", flush=True)
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    feat = [str(x) for x in data["feature_names"]]
    col = {name: feat.index(name) for name in (
        "name_rank", "addr_rank", "name_block_score", "addr_block_score", "from_name_ch", "from_addr_ch"
    )}
    val = data["split"].astype(str) == "val"
    groups: dict[str, list] = defaultdict(list)
    need: set[str] = set()
    xv = data["X"][val]
    s1v = data["s1_id"].astype(str)[val]
    midv = data["match_id"].astype(str)[val]
    for i, (sid, eid) in enumerate(zip(s1v.tolist(), midv.tolist())):
        groups[sid].append((eid, None, (
            float(xv[i, col["name_rank"]]), float(xv[i, col["addr_rank"]]),
            float(xv[i, col["name_block_score"]]), float(xv[i, col["addr_block_score"]]),
            int(xv[i, col["from_name_ch"]] >= 0.5), int(xv[i, col["from_addr_ch"]] >= 0.5),
        ), xv[i, 5:21].copy()))
        need.add(sid)
        need.add(eid)
    for row in keep:
        groups[row["sid"]].append((row["eid"], row["xlit"], (1.0, 99.0, 1.0, 0.0, 1, 0), None))
        need.add(row["sid"])
        need.add(row["eid"])
    texts: dict[str, tuple[str, str, str]] = {}
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, addr, country = line.rstrip("\n").split("\t")
                if eid in need:
                    texts[eid] = (bname, addr, country)
                    if len(texts) == len(need):
                        break
        if len(texts) == len(need):
            break
    print(f"texts {len(texts)}", flush=True)
    view_cache = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    addr_cache = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    order = [row["sid"] for row in meta if row["split"] == "val"]
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v5.txt"))
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    empty_rec = prepare_record("", "")
    base_preds, new_preds, gold = [], [], []
    accepted = 0
    flat_base, flat_new = [], []
    own_base, own_new = [], []
    for sid in order:
        gold.append(truths[sid])
        s1_rec = texts.get(sid, ("", "", ""))
        s1_indic = indic(s1_rec[0])
        items = groups.get(sid, [])
        mids, vecs, is_extra = [], [], []
        for eid, roman, ranks, mid_cols in items:
            nr, ar, ns, asc, fn, fa = ranks
            other = texts.get(eid, ("", "", ""))
            s1_indic = indic(s1_rec[0])
            if roman and s1_indic:
                ln, rn = name_view(roman), view_cache.get(eid, empty_name)
                lprep = prepare_record(roman, s1_rec[1])
                rprep = prepare_record(other[0], other[1])
            elif roman:
                ln, rn = view_cache.get(sid, empty_name), name_view(roman)
                lprep = prepare_record(s1_rec[0], s1_rec[1])
                rprep = prepare_record(roman, other[1])
            else:
                ln = view_cache.get(sid, empty_name)
                rn = view_cache.get(eid, empty_name)
                lprep = rprep = None
            if mid_cols is None:
                feat_tail = features_prepared(
                    lprep, rprep,
                    country_eq=float(s1_rec[2] == other[2]),
                    name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                    from_name_channel=fn, from_addr_channel=fa,
                )[5:]
            else:
                feat_tail = mid_cols
            full = name_five(ln, rn) + list(feat_tail) + extra_nine(
                addr_cache.get(sid, empty_addr), addr_cache.get(eid, empty_addr), ln, rn
            ) + particle_two(ln, rn)
            mids.append(eid)
            vecs.append(full)
            is_extra.append(roman is not None)
        if not vecs:
            base_preds.append(set())
            new_preds.append(set())
            continue
        scores = booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=400)
        base_m = [m for m, extra in zip(mids, is_extra) if not extra]
        base_s = [float(s) for s, extra in zip(scores, is_extra) if not extra]
        base_pred = decode_greedy_f05(base_m, base_s, min_gain=0.70, max_preds=12)
        new_pred = decode_greedy_f05(mids, [float(s) for s in scores], min_gain=0.70, max_preds=12)
        accepted += len((new_pred - base_pred) & truths[sid])
        base_preds.append(base_pred)
        new_preds.append(new_pred)
    print(f"v5 current {macro_f05(gold, base_preds):.4f}", flush=True)
    print(f"v5 plus xlit {macro_f05(gold, new_preds):.4f} accepted new true ids {accepted}", flush=True)


if __name__ == "__main__":
    main()
