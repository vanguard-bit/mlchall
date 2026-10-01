"""Cleanup features beyond the house number, then reverse lookup with those cleaned keys.

Validation fold only. Does not write a submission.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import time
import unicodedata
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from f05 import macro_f05
from normalize import ADDR_STOP, LEGAL, content_name_tokens, digit_tokens, tokens
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
MALL_ADDR = 25
MALL_NAME = 8
PER_S1 = 12
MIN_GAIN = 0.70
TLDS = {"com", "org", "net", "info", "biz", "www"}
EXTRA_NAMES = (
    "house_same",
    "house_missing",
    "house_conflict",
    "house_prefix",
    "street_jaccard",
    "postal_same",
    "postal_conflict",
    "name_fold_jw",
    "name_squash_ratio",
)
BUNDLES = {
    "house": (0, 1, 2),
    "house+street": (0, 1, 2, 3, 4),
    "house+postal": (0, 1, 2, 5, 6),
    "house+name": (0, 1, 2, 7, 8),
    "all": tuple(range(9)),
}

NEEDED: set[str] = set()
WANTED_KEYS: set[str] = set()


def fold_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def squash_name(name: str) -> str:
    toks = [t for t in content_name_tokens(fold_text(name)) if t not in TLDS]
    return "".join(toks)


def parse_addr(addr: str) -> tuple[str, frozenset[str], str]:
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
    zip5 = ""
    for raw in digit_tokens(addr):
        if len(raw) >= 5:
            zip5 = raw[:5]
            break
    return house, frozenset(streets[:4]), zip5


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_extra(a: tuple[str, frozenset[str], str, str, str], b: tuple[str, frozenset[str], str, str, str]) -> list[float]:
    ah, ast, az, af, aq = a
    bh, bst, bz, bf, bq = b
    share = bool(ast & bst)
    same = missing = conflict = prefix = 0.0
    if not ah or not bh:
        missing = 1.0
    elif ah == bh:
        same = 1.0
    else:
        shorter, longer = (ah, bh) if len(ah) <= len(bh) else (bh, ah)
        if share and len(shorter) >= 2 and longer.startswith(shorter):
            prefix = 1.0
        elif share:
            conflict = 1.0
    postal_same = postal_conflict = 0.0
    if az and bz:
        postal_same = float(az == bz)
        postal_conflict = float(az != bz)
    fold = JaroWinkler.normalized_similarity(af, bf) if af and bf else 0.0
    squash = fuzz.ratio(aq, bq) / 100.0 if aq and bq else 0.0
    return [same, missing, conflict, prefix, _jaccard(ast, bst), postal_same, postal_conflict, fold, squash]


def _load_worker(path: str) -> dict[str, tuple[str, str, str]]:
    found: dict[str, tuple[str, str, str]] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, name, addr, country = line.rstrip("\n").split("\t")
            if eid in NEEDED:
                found[eid] = (name, addr, country)
    return found


def load_texts(needed: set[str]) -> dict[str, tuple[str, str, str]]:
    global NEEDED
    NEEDED = needed
    ctx = mp.get_context("fork")
    files = [str(TRAIN / name) for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")]
    found: dict[str, tuple[str, str, str]] = {}
    with ctx.Pool(3) as pool:
        for part in pool.map(_load_worker, files):
            found.update(part)
    return found


def build_extra(s1: np.ndarray, mid: np.ndarray, prepared: dict[str, tuple]) -> np.ndarray:
    empty = ("", frozenset(), "", "", "")
    out = np.zeros((len(s1), 9), dtype=np.float32)
    for i, (sid, eid) in enumerate(zip(s1, mid)):
        out[i] = pair_extra(prepared.get(str(sid), empty), prepared.get(str(eid), empty))
    return out


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


def decode_groups(groups: dict[str, dict], truths: dict[str, set[str]], min_gain: float) -> tuple[float, float, int, int]:
    preds: list[set[str]] = []
    gold: list[set[str]] = []
    single_p: list[set[str]] = []
    single_g: list[set[str]] = []
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
        preds.append(chosen)
        gold.append(truth)
        tp += len(chosen & truth)
        fp += len(chosen - truth)
        if not truth:
            single_p.append(chosen)
            single_g.append(set())
    score = macro_f05(gold, preds)
    single = macro_f05(single_g, single_p) if single_g else 0.0
    return score, single, tp, fp


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


def _expand_worker(args: tuple[str, int]) -> tuple[dict[str, int], dict[str, list[tuple[str, str, str, str]]]]:
    path, wid = args
    counts: dict[str, int] = defaultdict(int)
    kept: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            eid, name, addr, country = line.rstrip("\n").split("\t")
            house, streets, _zip = parse_addr(addr)
            keys: list[str] = []
            if house:
                for street in streets:
                    keys.append(f"{country}|h|{house}|{street}")
            squashed = squash_name(name)
            if len(squashed) >= 10:
                keys.append(f"{country}|n|{squashed}")
            for key in keys:
                if key not in WANTED_KEYS:
                    continue
                counts[key] += 1
                cap = MALL_NAME if "|n|" in key else MALL_ADDR
                bucket = kept[key]
                if len(bucket) < cap:
                    bucket.append((eid, name, addr, country))
            if i and i % 1_000_000 == 0 and wid == 0:
                print(f"  clean-reverse {path.rsplit('/', 1)[-1]} {i}", flush=True)
    return dict(counts), dict(kept)


def clean_keys(country: str, prepared: tuple, name: str) -> list[tuple[str, str]]:
    house, streets, _zip, _folded, squashed = prepared
    keys: list[tuple[str, str]] = []
    if house:
        for street in streets:
            keys.append((f"{country}|h|{house}|{street}", "addr"))
    if len(squashed) >= 10:
        keys.append((f"{country}|n|{squashed}", "name"))
    return keys


def score_added(
    model: lgb.Booster,
    cols: tuple[int, ...],
    groups: dict[str, dict],
    truths: dict[str, set[str]],
    added: dict[str, list[str]],
    texts: dict[str, tuple[str, str, str]],
    *,
    address_channel: bool,
) -> tuple[float, float, int, int]:
    prepared_text = {eid: prepare_record(name, addr) for eid, (name, addr, _c) in texts.items()}
    prepared_extra = {}
    for eid, (name, addr, _c) in texts.items():
        house, streets, zip5 = parse_addr(addr)
        folded = " ".join(content_name_tokens(fold_text(name)))
        prepared_extra[eid] = (house, streets, zip5, folded, squash_name(name))
    countries = {eid: country for eid, (_n, _a, country) in texts.items()}
    empty = ("", frozenset(), "", "", "")
    rows = []
    owners: list[str] = []
    mids: list[str] = []
    labels: list[int] = []
    for sid, extra_ids in added.items():
        left = prepared_text.get(sid)
        if left is None:
            continue
        truth = truths[sid]
        for eid in extra_ids:
            right = prepared_text.get(eid)
            if right is None:
                continue
            if address_channel:
                base = features_prepared(
                    left,
                    right,
                    country_eq=float(countries.get(sid) == countries.get(eid)),
                    name_rank=99.0,
                    addr_rank=20.0,
                    addr_score=1.0,
                    from_addr_channel=1,
                )
            else:
                base = features_prepared(
                    left,
                    right,
                    country_eq=float(countries.get(sid) == countries.get(eid)),
                )
            extra = pair_extra(prepared_extra.get(sid, empty), prepared_extra.get(eid, empty))
            rows.append(base + [extra[i] for i in cols])
            owners.append(sid)
            mids.append(eid)
            labels.append(int(eid in truth))
    pred = model.predict(np.asarray(rows, dtype=np.float32)) if rows else np.zeros(0)
    by_sid: dict[str, list[tuple[str, float]]] = defaultdict(list)
    accepted_tp = accepted_fp = 0
    for sid, eid, label, score in zip(owners, mids, labels, pred):
        if float(score) < MIN_GAIN:
            continue
        by_sid[sid].append((eid, float(score)))
        if label:
            accepted_tp += 1
        else:
            accepted_fp += 1
    merged: dict[str, dict] = {}
    for sid, slot in groups.items():
        extra = by_sid.get(sid, [])
        merged[sid] = {
            "mids": slot["mids"] + [eid for eid, _score in extra],
            "scores": np.asarray(list(slot["scores"]) + [score for _eid, score in extra], dtype=np.float32),
        }
    for sid, extra in by_sid.items():
        if sid not in merged:
            merged[sid] = {
                "mids": [eid for eid, _score in extra],
                "scores": np.asarray([score for _eid, score in extra], dtype=np.float32),
            }
    f05, single, _tp, _fp = decode_groups(merged, truths, MIN_GAIN)
    return f05, single, accepted_tp, accepted_fp


def run_reverse(
    model: lgb.Booster,
    cols: tuple[int, ...],
    groups: dict[str, dict],
    truths: dict[str, set[str]],
    forward: dict[str, set[str]],
    val_text: dict[str, tuple[str, str, str]],
    val_prepared: dict[str, tuple],
    base_f05: float,
) -> None:
    global WANTED_KEYS
    key_owner: dict[str, list[str]] = defaultdict(list)
    key_kind: dict[str, str] = {}
    for sid, prep in val_prepared.items():
        country = val_text[sid][2]
        for key, kind in clean_keys(country, prep, val_text[sid][0]):
            key_owner[key].append(sid)
            key_kind[key] = kind
    wanted = {key for key, owners in key_owner.items() if len(owners) <= (5 if key_kind[key] == "name" else 8)}
    WANTED_KEYS = wanted
    print(f"clean reverse keys {len(wanted)}", flush=True)
    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    counts: dict[str, int] = defaultdict(int)
    rows_of: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    with ctx.Pool(WORKERS) as pool:
        parts = pool.map(_expand_worker, jobs)
    for part_counts, part_rows in parts:
        for key, count in part_counts.items():
            counts[key] += count
        for key, rows in part_rows.items():
            rows_of[key].extend(rows)
    poisoned = 0
    for key in list(wanted):
        cap = MALL_NAME if key_kind[key] == "name" else MALL_ADDR
        if counts.get(key, 0) > cap:
            poisoned += 1
            wanted.remove(key)
    print(f"poisoned clean keys {poisoned}", flush=True)

    owners_of: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, sids in key_owner.items():
        if key not in wanted:
            continue
        for sid in sids:
            owners_of[sid].append((key, key_kind[key]))

    def gather_fast(kind: str) -> tuple[dict[str, list[str]], dict[str, tuple[str, str, str]]]:
        added: dict[str, list[str]] = {}
        texts = dict(val_text)
        raw_tp = raw_fp = 0
        shown = 0
        for sid, truth in truths.items():
            best: dict[str, int] = {}
            info: dict[str, tuple[str, str, str]] = {}
            fwd = forward.get(sid, set())
            for key, key_type in owners_of.get(sid, []):
                if key_type != kind:
                    continue
                rarity = counts.get(key, 0)
                for eid, name, addr, country in rows_of.get(key, []):
                    if eid in fwd:
                        continue
                    prev = best.get(eid)
                    if prev is not None and prev <= rarity:
                        continue
                    best[eid] = rarity
                    info[eid] = (name, addr, country)
            chosen = sorted(best, key=lambda eid: best[eid])[:PER_S1]
            if not chosen:
                continue
            added[sid] = chosen
            for eid in chosen:
                texts.setdefault(eid, info[eid])
                if eid in truth:
                    raw_tp += 1
                    if shown < 6:
                        ln, la, _lc = val_text[sid]
                        rn, ra, _rc = info[eid]
                        print(f"{kind} NEW TP {ln} || {la} <> {rn} || {ra}", flush=True)
                        shown += 1
                else:
                    raw_fp += 1
        print(f"{kind} raw new tp {raw_tp} fp {raw_fp}", flush=True)
        return added, texts

    for kind in ("addr", "name"):
        added, texts = gather_fast(kind)
        for channel, flag in (("hidden", False), ("addr-channel", True)):
            f05, single, tp, fp = score_added(
                model, cols, groups, truths, added, texts, address_channel=flag
            )
            print(
                f"{kind} {channel} f05 {f05:.4f} delta {f05 - base_f05:+.4f} "
                f"singleton {single:.4f} accepted tp {tp} fp {fp}",
                flush=True,
            )


def check_examples() -> None:
    left = parse_addr("01926 GREG ST, PMB 7093, CANYON, TX")
    right = parse_addr("1926 Greg Street, Canyon, TX")
    if left[0] != "1926" or right[0] != "1926":
        raise SystemExit(f"house parse failed {left} {right}")
    if squash_name("davisingredients.com") != squash_name("Davis Ingredients"):
        raise SystemExit("domain squash failed")
    if squash_name("Cardiology Médicine PLLC") != squash_name("Cardiology Medicine PLLC"):
        raise SystemExit("accent squash failed")


def main() -> None:
    check_examples()
    t0 = time.time()
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    X = np.asarray(data["X"], dtype=np.float32)
    y = data["y"].astype(int)
    s1 = data["s1_id"].astype(str)
    mid = data["match_id"].astype(str)
    split = data["split"].astype(str)
    train = split == "train"
    val = split == "val"
    truths = {row["sid"]: set(row["truth"]) for row in meta if row["split"] == "val"}
    needed = set(s1.tolist()) | set(mid.tolist())
    print(f"pairs {len(s1)} unique ids {len(needed)}", flush=True)
    texts = load_texts(needed)
    print(f"loaded {len(texts)} in {time.time() - t0:.0f}s", flush=True)
    prepared = {}
    for eid, (name, addr, _country) in texts.items():
        house, streets, zip5 = parse_addr(addr)
        folded_toks = [t for t in content_name_tokens(fold_text(name)) if t not in TLDS]
        prepared[eid] = (house, streets, zip5, " ".join(folded_toks), "".join(folded_toks))
    print(f"parsed {len(prepared)} in {time.time() - t0:.0f}s", flush=True)
    extra = build_extra(s1, mid, prepared)
    print(f"features built in {time.time() - t0:.0f}s", flush=True)
    for col, name in enumerate(EXTRA_NAMES):
        on = extra[val, col] == 1
        if name == "street_jaccard" or name.endswith("jw") or name.endswith("ratio"):
            continue
        if not on.any():
            print(f"val {name} empty", flush=True)
            continue
        print(
            f"val {name} rate {on.mean():.3f} positive-rate {y[val][on].mean():.3f} "
            f"off-positive {y[val][~on].mean():.3f}",
            flush=True,
        )
    val_rows = np.where(val)[0]
    forward: dict[str, set[str]] = defaultdict(set)
    for row in val_rows:
        forward[str(s1[row])].add(str(mid[row]))
    house_groups = None
    house_model = None
    house_f05 = 0.0
    best = ("", -1.0, None, ())
    for label, cols in BUNDLES.items():
        block = np.concatenate([X, extra[:, cols]], axis=1)
        model = train_model(block, y, train, val)
        scores = model.predict(block[val], num_iteration=model.best_iteration)
        groups = group_scores(s1, mid, val_rows, scores)
        f05, single, tp, fp = decode_groups(groups, truths, MIN_GAIN)
        print(
            f"{label} f05 {f05:.4f} singleton {single:.4f} tp {tp} fp {fp} trees {model.best_iteration}",
            flush=True,
        )
        if label == "house":
            house_groups = groups
            house_model = model
            house_f05 = f05
        if f05 > best[1]:
            best = (label, f05, model, cols)
        del block
        gc.collect()
    print(f"best bundle {best[0]} {best[1]:.4f} over house {best[1] - house_f05:+.4f}", flush=True)
    assert house_model is not None and house_groups is not None
    val_text = {sid: texts[sid] for sid in truths if sid in texts}
    val_prepared = {sid: prepared[sid] for sid in val_text}
    del texts, prepared, extra, X, data
    gc.collect()
    print("reverse with house model", flush=True)
    run_reverse(house_model, BUNDLES["house"], house_groups, truths, forward, val_text, val_prepared, house_f05)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
