"""Cutoff sweep for lgbm_v7r on romanized names plus the v6 channels, domain dropped."""

from __future__ import annotations

import json
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

from decode import decode_greedy_f05
from eval_v7_extras import BOARD, MODEL_V7R, RAW_RATIO_AT, _channels
from f05 import macro_f05
from normalize import tokens
from pair_features import features_prepared, prepare_record
from score_v6 import load_cache, shown_name
from v4_features import extra_nine, name_five, name_view, parse_addr, particle_two
from v7_features import city_token, city_two, document_frequency, idf_table, rare_two


def main() -> None:
    t0 = time.time()
    load_cache()
    counts, n_docs = document_frequency()
    idf = idf_table(counts, n_docs)
    missing = 1.0
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    country_of = {row["sid"]: row.get("country", "") for row in val_meta}
    order = [row["sid"] for row in val_meta]
    texts, groups, channels = _channels(order, truths)
    added = channels["combined without domain"]
    shown = {eid: shown_name(rec[0]) for eid, rec in texts.items()}
    raw_view = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    roman_view = {eid: name_view(shown[eid]) for eid, rec in texts.items()}
    addr_view = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    city_of = {eid: city_token(rec[1]) for eid, rec in texts.items()}
    raw_join = {eid: " ".join(tokens(rec[0])) for eid, rec in texts.items()}
    roman_join = {eid: " ".join(tokens(shown[eid])) for eid in texts}
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")

    def vector(sid: str, eid: str, ranks, tail, extra: bool) -> list[float]:
        ln = raw_view.get(sid, empty_name)
        rn = roman_view.get(eid, empty_name)
        if extra:
            nr, ar, ns, asc, fn, fa = ranks
            other = texts[eid]
            tail_list = features_prepared(
                prepare_record(texts[sid][0], texts[sid][1]),
                prepare_record(shown[eid], other[1]),
                country_eq=float(texts[sid][2] == other[2]),
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )[5:]
        else:
            tail_list = list(tail)
            if shown[eid] != texts[eid][0]:
                tail_list[RAW_RATIO_AT] = fuzz.ratio(raw_join[sid], roman_join[eid]) / 100.0
        return (
            name_five(ln, rn)
            + tail_list
            + extra_nine(addr_view.get(sid, empty_addr), addr_view.get(eid, empty_addr), ln, rn)
            + particle_two(ln, rn)
            + rare_two(ln[1], rn[1], idf, missing)
            + city_two(city_of.get(sid, ""), city_of.get(eid, ""))
        )

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    booster = lgb.Booster(model_file=str(MODEL_V7R))
    trees = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    for sid in order:
        mids, vecs = [], []
        for eid, ranks, tail in groups.get(sid, []):
            mids.append(eid)
            vecs.append(vector(sid, eid, ranks, tail, False))
        for eid in added.get(sid, []):
            mids.append(eid)
            vecs.append(vector(sid, eid, (1.0, 99.0, 1.0, 0.0, 1, 0), None, True))
        if not vecs:
            continue
        scores = booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=trees)
        grouped[sid] = list(zip(mids, (float(s) for s in scores)))
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    def report(floor: float) -> None:
        preds = []
        gold = []
        single_g, single_p = [], []
        india_g, india_p, us_g, us_p = [], [], [], []
        for sid in order:
            pred = decode_greedy_f05(
                [eid for eid, _score in grouped.get(sid, [])],
                [score for _eid, score in grouped.get(sid, [])],
                min_gain=floor,
                max_preds=12,
            )
            preds.append(pred)
            gold.append(truths[sid])
            if not truths[sid]:
                single_g.append(set())
                single_p.append(pred)
            if country_of.get(sid) == "India":
                india_g.append(truths[sid])
                india_p.append(pred)
            elif country_of.get(sid) == "US":
                us_g.append(truths[sid])
                us_p.append(pred)
        print(
            f"v7r no-domain floor {floor:.2f} f05 {macro_f05(gold, preds):.4f} "
            f"singleton {macro_f05(single_g, single_p):.4f} "
            f"india {macro_f05(india_g, india_p):.4f} us {macro_f05(us_g, us_p):.4f}",
            flush=True,
        )

    for floor in (0.65, 0.70, 0.72, 0.75, 0.78, 0.80):
        report(floor)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
