"""Unique recall and v5 F0.5 if IndicXlit names are added for missed Indic pairs."""

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


def indic_on_s1(row: dict) -> bool:
    return any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF for ch in row["s1_name"])


def main() -> None:
    rows = json.loads((BOARD / "indic_misses.json").read_text(encoding="utf-8"))
    for row in rows:
        latin = row["m_name"] if indic_on_s1(row) else row["s1_name"]
        row["new_jw"] = jw(row.get("xlit") or "", latin)
        row["exact"] = norm(row.get("xlit") or "") == norm(latin)
    print("band  n  exact  jw>=0.90  jw>=0.80")
    for band in ("hard", "mid", "high"):
        part = [r for r in rows if r["band"] == band]
        print(
            f"{band:5} {len(part):4}  {sum(r['exact'] for r in part):4}  "
            f"{sum(r['new_jw'] >= 0.90 for r in part):4}  {sum(r['new_jw'] >= 0.80 for r in part):4}"
        )
    keep = [r for r in rows if r["new_jw"] >= 0.90 or r["exact"]]
    print(f"would add {len(keep)} true links not in the current candidate list")

    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    feat = [str(x) for x in data["feature_names"]]
    cols = {name: feat.index(name) for name in (
        "name_rank", "addr_rank", "name_block_score", "addr_block_score", "from_name_ch", "from_addr_ch"
    )}
    val = data["split"].astype(str) == "val"
    groups: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    need = set()
    for i, (sid, eid) in enumerate(zip(data["s1_id"].astype(str)[val], data["match_id"].astype(str)[val])):
        xv = data["X"][val][i]
        groups[sid].append((eid, (
            float(xv[cols["name_rank"]]), float(xv[cols["addr_rank"]]),
            float(xv[cols["name_block_score"]]), float(xv[cols["addr_block_score"]]),
            int(xv[cols["from_name_ch"]] >= 0.5), int(xv[cols["from_addr_ch"]] >= 0.5),
            None,
        )))
        need.add(sid)
        need.add(eid)
    for row in keep:
        groups[row["sid"]].append((row["eid"], (1.0, 99.0, 1.0, 0.0, 1, 0, row["xlit"] if indic_on_s1(row) else None)))
        if not indic_on_s1(row):
            # Latin is S1; the match name is Indic and should be replaced.
            groups[row["sid"]][-1] = (row["eid"], (1.0, 99.0, 1.0, 0.0, 1, 0, row["xlit"]))
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
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    order = [row["sid"] for row in meta if row["split"] == "val"]
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v5.txt"))
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    empty_rec = prepare_record("", "")
    base_preds = []
    new_preds = []
    gold = []
    accepted = 0
    for sid in order:
        truth = truths[sid]
        gold.append(truth)
        ctry = texts.get(sid, ("", "", ""))[2]
        left_raw = texts.get(sid, ("", "", ""))
        left = prepare_record(left_raw[0], left_raw[1])
        left_view = name_view(left_raw[0])
        left_addr = parse_addr(left_raw[1])
        base_mids, base_rows = [], []
        new_mids, new_rows = [], []
        for eid, meta_row in groups.get(sid, []):
            nr, ar, ns, asc, fn, fa, xlit = meta_row
            other = texts.get(eid, ("", "", ""))
            other_name = xlit if xlit else other[0]
            # If the Indic text is on S1, swap the view used for S1.
            use_left = left_view
            use_left_name = left_raw[0]
            if xlit and indic_on_s1({"s1_name": left_raw[0]}):
                use_left = name_view(xlit)
                use_left_name = xlit
                other_name = other[0]
            right = prepare_record(other_name, other[1])
            right_view = name_view(other_name)
            base_feat = features_prepared(
                prepare_record(use_left_name, left_raw[1]),
                right,
                country_eq=float(ctry == other[2]),
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )
            vec = name_five(use_left if xlit and indic_on_s1({"s1_name": left_raw[0]}) else left_view, right_view)
            vec = vec + base_feat[5:] + extra_nine(left_addr, parse_addr(other[1]), vec and (use_left if False else left_view), right_view)
            # rebuild cleanly
            ln = name_view(xlit) if xlit and indic_on_s1({"s1_name": left_raw[0]}) else left_view
            rn = name_view(other[0] if xlit and indic_on_s1({"s1_name": left_raw[0]}) else (xlit or other[0]))
            feat = features_prepared(
                prepare_record(xlit if xlit and indic_on_s1({"s1_name": left_raw[0]}) else left_raw[0], left_raw[1]),
                prepare_record(other[0] if xlit and indic_on_s1({"s1_name": left_raw[0]}) else (xlit or other[0]), other[1]),
                country_eq=float(ctry == other[2]),
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )
            full = name_five(ln, rn) + feat[5:] + extra_nine(left_addr, parse_addr(other[1]), ln, rn) + particle_two(ln, rn)
            if xlit is None:
                base_mids.append(eid)
                base_rows.append(full)
            new_mids.append(eid)
            new_rows.append(full)
        def decode(mids, rows):
            if not rows:
                return set()
            scores = booster.predict(np.asarray(rows, dtype=np.float32), num_iteration=400)
            return decode_greedy_f05(mids, [float(s) for s in scores], min_gain=0.70, max_preds=12)
        base_pred = decode(base_mids, base_rows)
        new_pred = decode(new_mids, new_rows)
        accepted += len((new_pred - base_pred) & truth)
        base_preds.append(base_pred)
        new_preds.append(new_pred)
    print(f"v5 on current candidates {macro_f05(gold, base_preds):.4f}")
    print(f"v5 plus xlit hits {macro_f05(gold, new_preds):.4f} new true ids accepted {accepted}")


if __name__ == "__main__":
    main()
