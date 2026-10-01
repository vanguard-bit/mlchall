"""Holdout check for the two retrieval ideas still open.

Domain stem: one side contains a website and its stem equals the other side's
squashed name. Spelling: consonant skeletons match and the house numbers match.
A third check raises the cutoff when an existing candidate has an empty address.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from decode import decode_greedy_f05
from f05 import macro_f05
from pair_features import features_prepared, prepare_record
from paths import DATA_DIR
from v4_features import (
    content_name_tokens_v4,
    extra_nine,
    name_five,
    name_view,
    parse_addr,
    particle_two,
)
from v6_keys import fold_text

TRAIN = DATA_DIR / "train"
BOARD = DATA_DIR / "scoreboard"
WORKERS = 8
CAP = 8
MIN_LEN = 8
DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][-a-z0-9]*(?:\.[a-z0-9][-a-z0-9]*)+)",
    re.I,
)
TLDS = {"com", "org", "net", "info", "biz", "co", "io", "www", "html"}
GENERIC = {
    "gmail", "yahoo", "hotmail", "outlook", "google", "facebook",
    "company", "business", "online", "website", "email",
    "international", "enterprises", "solutions", "services",
    "technologies", "technology", "industries", "industry",
    "private", "limited", "group", "global",
}

SQUASH_QUERY: set[str] = set()
STEM_QUERY: set[str] = set()
SPELL_QUERY: set[str] = set()


def squash_name(name: str) -> str:
    toks = [tok for tok in content_name_tokens_v4(fold_text(name)) if tok not in TLDS]
    return "".join(toks)


def skeleton(text: str) -> str:
    out: list[str] = []
    prev = ""
    for ch in text:
        if ch in "aeiou" or not ch.isalpha():
            continue
        if ch == prev:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def domain_stems(text: str) -> list[str]:
    found: list[str] = []
    for match in DOMAIN_RE.finditer(text.lower()):
        labels = match.group(1).split("/")[0].split(".")
        if len(labels) < 2:
            continue
        core = labels[-2]
        if len(core) < MIN_LEN or core in GENERIC or not core.isalnum():
            continue
        found.append(core)
    return found


def _edits(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 1:
        return 2
    if left == right:
        return 0
    prev = list(range(len(right) + 1))
    for i, ch in enumerate(left, start=1):
        cur = [i]
        for j, other in enumerate(right, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch != other)))
        prev = cur
    return prev[-1]


def _worker(args: tuple[str, int]) -> tuple[dict[str, list[str]], dict[str, list[str]], int]:
    path, wid = args
    domain_hits: dict[str, list[str]] = defaultdict(list)
    spell_hits: dict[str, list[str]] = defaultdict(list)
    over = 0
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i % WORKERS != wid:
                continue
            eid, name, addr, country = line.rstrip("\n").split("\t")
            stems = domain_stems(name)
            if not stems and ("." in addr or "www" in addr.lower() or "http" in addr.lower()):
                stems = domain_stems(addr)
            squash = squash_name(name)
            for stem in stems:
                key = country + "|" + stem
                if key in SQUASH_QUERY and len(domain_hits[key]) < CAP:
                    domain_hits[key].append(eid)
                elif key in SQUASH_QUERY:
                    over += 1
            if len(squash) >= MIN_LEN:
                key = country + "|" + squash
                if key in STEM_QUERY and len(domain_hits[key]) < CAP:
                    domain_hits[key].append(eid)
                elif key in STEM_QUERY:
                    over += 1
            if SPELL_QUERY:
                house, _streets, _zip = parse_addr(addr)
                skel = skeleton(squash)
                if len(house) >= 2 and len(skel) >= MIN_LEN:
                    key = f"{country}|{house}|{skel}"
                    if key in SPELL_QUERY and len(spell_hits[key]) < CAP:
                        spell_hits[key].append(eid)
                    elif key in SPELL_QUERY:
                        over += 1
    return domain_hits, spell_hits, over


def _load(need: set[str]) -> dict[str, tuple[str, str, str]]:
    texts: dict[str, tuple[str, str, str]] = {}
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        if not (need - set(texts)):
            break
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, name, addr, country = line.rstrip("\n").split("\t")
                if eid in need:
                    texts[eid] = (name, addr, country)
                    if len(texts) == len(need):
                        break
    return texts


def main() -> None:
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    val_meta = [row for row in meta if row["split"] == "val"]
    truths = {row["sid"]: set(row["truth"]) for row in val_meta}
    order = [row["sid"] for row in val_meta]
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    val = data["split"].astype(str) == "val"
    have: dict[str, set[str]] = defaultdict(set)
    feat = [str(x) for x in data["feature_names"]]
    col = {name: feat.index(name) for name in (
        "name_rank", "addr_rank", "name_block_score", "addr_block_score", "from_name_ch", "from_addr_ch",
    )}
    xv = data["X"][val]
    groups: dict[str, list] = defaultdict(list)
    pair_ids: set[str] = set(order)
    for i, (sid, eid) in enumerate(zip(data["s1_id"].astype(str)[val].tolist(), data["match_id"].astype(str)[val].tolist())):
        groups[sid].append((eid, (
            float(xv[i, col["name_rank"]]), float(xv[i, col["addr_rank"]]),
            float(xv[i, col["name_block_score"]]), float(xv[i, col["addr_block_score"]]),
            int(xv[i, col["from_name_ch"]] >= 0.5), int(xv[i, col["from_addr_ch"]] >= 0.5),
        ), xv[i, 5:21].copy()))
        have[sid].add(eid)
        pair_ids.add(eid)

    misses: list[tuple[str, str]] = []
    for sid in order:
        for eid in truths[sid] - have.get(sid, set()):
            misses.append((sid, eid))
    need = set(order) | {eid for _sid, eid in misses}
    texts = _load(need)
    print(f"neither true pairs {len(misses)} texts {len(texts)}", flush=True)

    domain_true = spell_true = spell_edit = missing_addr = missing_exact = 0
    shown_d = shown_s = 0
    for sid, eid in misses:
        if sid not in texts or eid not in texts:
            continue
        ln, la, lc = texts[sid]
        rn, ra, rc = texts[eid]
        if lc != rc:
            continue
        left_stems = set(domain_stems(ln) + domain_stems(la))
        right_stems = set(domain_stems(rn) + domain_stems(ra))
        left_sq, right_sq = squash_name(ln), squash_name(rn)
        domain_hit = bool(left_stems & {right_sq}) or bool(right_stems & {left_sq}) or bool(left_stems & right_stems)
        lh, _ls, _lz = parse_addr(la)
        rh, _rs, _rz = parse_addr(ra)
        left_sk, right_sk = skeleton(left_sq), skeleton(right_sq)
        house_hit = len(lh) >= 2 and lh == rh
        spell_hit = house_hit and len(left_sk) >= MIN_LEN and left_sk == right_sk
        if domain_hit:
            domain_true += 1
            if shown_d < 6:
                print(f"domain {ln} || {la} <> {rn} || {ra}", flush=True)
                shown_d += 1
        if spell_hit:
            spell_true += 1
            if shown_s < 6:
                print(f"spell {ln} || {la} <> {rn} || {ra}", flush=True)
                shown_s += 1
        elif house_hit and _edits(left_sk, right_sk) == 1:
            spell_edit += 1
        if not la.strip() or not ra.strip() or "<null>" in la.lower() or "<null>" in ra.lower():
            missing_addr += 1
            if left_sq == right_sq and len(left_sq) >= MIN_LEN:
                missing_exact += 1
    print(
        f"ceiling domain {domain_true} skeleton+house {spell_true} skeleton edit1+house {spell_edit} "
        f"addr-missing {missing_addr} of which exact name {missing_exact}",
        flush=True,
    )

    squash_of: dict[str, list[str]] = defaultdict(list)
    stem_of: dict[str, list[str]] = defaultdict(list)
    spell_of: dict[str, list[str]] = defaultdict(list)
    for sid in order:
        name, addr, country = texts[sid]
        squash = squash_name(name)
        if len(squash) >= MIN_LEN:
            key = country + "|" + squash
            squash_of[key].append(sid)
            SQUASH_QUERY.add(key)
        for stem in set(domain_stems(name) + domain_stems(addr)):
            key = country + "|" + stem
            stem_of[key].append(sid)
            STEM_QUERY.add(key)
        house, _streets, _zip = parse_addr(addr)
        skel = skeleton(squash)
        if len(house) >= 2 and len(skel) >= MIN_LEN:
            key = f"{country}|{house}|{skel}"
            spell_of[key].append(sid)
            SPELL_QUERY.add(key)
    print(
        f"queries squash {len(SQUASH_QUERY)} stems {len(STEM_QUERY)} spell {len(SPELL_QUERY)}",
        flush=True,
    )

    ctx = mp.get_context("fork")
    jobs = [
        (str(TRAIN / name), wid)
        for name in ("train_source2.tsv", "train_source3.tsv")
        for wid in range(WORKERS)
    ]
    domain_hits: dict[str, list[str]] = defaultdict(list)
    spell_hits: dict[str, list[str]] = defaultdict(list)
    over = 0
    with ctx.Pool(WORKERS) as pool:
        for part_d, part_s, part_over in pool.map(_worker, jobs):
            for key, eids in part_d.items():
                domain_hits[key].extend(eids)
            for key, eids in part_s.items():
                spell_hits[key].extend(eids)
            over += part_over
    dropped_d = dropped_s = 0
    for key in list(domain_hits):
        if len(domain_hits[key]) > CAP:
            dropped_d += 1
            del domain_hits[key]
    for key in list(spell_hits):
        if len(spell_hits[key]) > CAP:
            dropped_s += 1
            del spell_hits[key]
    print(f"domain keys {len(domain_hits)} dropped {dropped_d} spell keys {len(spell_hits)} dropped {dropped_s} overflows {over}", flush=True)

    def collect(index_of: dict[str, list[str]], hits: dict[str, list[str]]) -> dict[str, list[str]]:
        added: dict[str, list[str]] = defaultdict(list)
        tp = fp = 0
        for key, sids in index_of.items():
            for eid in hits.get(key, []):
                for sid in sids:
                    if eid in have.get(sid, ()):
                        continue
                    if eid in added[sid]:
                        continue
                    if len(added[sid]) >= 12:
                        continue
                    added[sid].append(eid)
                    if eid in truths[sid]:
                        tp += 1
                    else:
                        fp += 1
        print(f"new candidates tp {tp} fp {fp}", flush=True)
        return added

    print("domain", end=" ", flush=True)
    domain_added = collect(squash_of, domain_hits)
    # Source 1 website against a Source 2/3 squashed name uses the stem key.
    print("domain-from-s1", end=" ", flush=True)
    domain_from_s1 = collect(stem_of, domain_hits)
    for sid, eids in domain_from_s1.items():
        for eid in eids:
            if eid not in domain_added[sid] and len(domain_added[sid]) < 12:
                domain_added[sid].append(eid)
    print("spell", end=" ", flush=True)
    spell_added = collect(spell_of, spell_hits)

    extra_ids: set[str] = set()
    for added in (domain_added, spell_added):
        for eids in added.values():
            extra_ids.update(eids)
    missing = (pair_ids | extra_ids) - set(texts)
    if missing:
        texts.update(_load(missing))
    print(f"texts {len(texts)} missing {len(missing - set(texts))}", flush=True)

    raw_view = {eid: name_view(rec[0]) for eid, rec in texts.items()}
    addr_view = {eid: parse_addr(rec[1]) for eid, rec in texts.items()}
    booster = lgb.Booster(model_file=str(BOARD / "lgbm_v5.txt"))
    empty_name = ("", frozenset(), "", "")
    empty_addr = ("", frozenset(), "")
    gold = [truths[sid] for sid in order]

    def vector(sid: str, eid: str, ranks, tail) -> list[float]:
        nr, ar, ns, asc, fn, fa = ranks
        ln = raw_view.get(sid, empty_name)
        if tail is None:
            other = texts[eid]
            rn = raw_view.get(eid, empty_name)
            tail = features_prepared(
                prepare_record(texts[sid][0], texts[sid][1]),
                prepare_record(other[0], other[1]),
                country_eq=float(texts[sid][2] == other[2]),
                name_rank=nr, addr_rank=ar, name_score=ns, addr_score=asc,
                from_name_channel=fn, from_addr_channel=fa,
            )[5:]
        else:
            rn = raw_view.get(eid, empty_name)
        return (
            name_five(ln, rn)
            + list(tail)
            + extra_nine(addr_view.get(sid, empty_addr), addr_view.get(eid, empty_addr), ln, rn)
            + particle_two(ln, rn)
        )

    base_mids: dict[str, list[str]] = {}
    base_vecs: dict[str, list] = {}
    base_empty: dict[str, list[bool]] = {}
    for sid in order:
        mids, vecs, empty = [], [], []
        for eid, ranks, tail in groups.get(sid, []):
            mids.append(eid)
            vecs.append(vector(sid, eid, ranks, tail))
            empty.append(bool(tail[7] >= 0.5 or tail[8] >= 0.5))
        base_mids[sid] = mids
        base_vecs[sid] = vecs
        base_empty[sid] = empty
    print("base vectors ready", flush=True)

    def score(added: dict[str, list[str]] | None, empty_cutoff: float) -> tuple[float, float, int, int]:
        preds = []
        accepted_tp = accepted_fp = 0
        for sid in order:
            mids = list(base_mids[sid])
            vecs = list(base_vecs[sid])
            base_n = len(mids)
            for eid in (added or {}).get(sid, []):
                mids.append(eid)
                vecs.append(vector(sid, eid, (1.0, 99.0, 1.0, 0.0, 1, 0), None))
            if not vecs:
                preds.append(set())
                continue
            scores = [float(s) for s in booster.predict(np.asarray(vecs, dtype=np.float32), num_iteration=400)]
            base_pred = decode_greedy_f05(mids[:base_n], scores[:base_n], min_gain=0.70, max_preds=12)
            if empty_cutoff > 0.70:
                for i, blank in enumerate(base_empty[sid]):
                    if blank and scores[i] < empty_cutoff:
                        scores[i] = 0.0
            pred = decode_greedy_f05(mids, scores, min_gain=0.70, max_preds=12)
            preds.append(pred)
            for eid in pred - base_pred:
                if eid in truths[sid]:
                    accepted_tp += 1
                else:
                    accepted_fp += 1
        single_gold, single_pred = [], []
        for sid, pred in zip(order, preds):
            if not truths[sid]:
                single_gold.append(set())
                single_pred.append(pred)
        single = macro_f05(single_gold, single_pred) if single_gold else 0.0
        return macro_f05(gold, preds), single, accepted_tp, accepted_fp

    base_f, base_s, _tp, _fp = score(None, 0.70)
    print(f"baseline f05 {base_f:.4f} singleton {base_s:.4f}", flush=True)
    union: dict[str, list[str]] = defaultdict(list)
    for sid in order:
        seen: set[str] = set()
        for eid in domain_added.get(sid, []) + spell_added.get(sid, []):
            if eid not in seen:
                seen.add(eid)
                union[sid].append(eid)
    runs = (
        ("domain stem", domain_added, 0.70),
        ("skeleton+house", spell_added, 0.70),
        ("domain+skeleton", union, 0.70),
        ("empty-addr cutoff 0.85", None, 0.85),
        ("empty-addr cutoff 0.90", None, 0.90),
    )
    for label, added, cutoff in runs:
        f05, single, tp, fp = score(added, cutoff)
        print(
            f"{label} f05 {f05:.4f} delta {f05 - base_f:+.4f} singleton {single:.4f} accepted tp {tp} fp {fp}",
            flush=True,
        )


if __name__ == "__main__":
    main()
