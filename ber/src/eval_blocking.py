"""Measure same-country name + address blocking recall on a train sample."""

from __future__ import annotations

import math
import random
import sys
import time
import zipfile
from array import array
from collections import defaultdict

from blocking import POSTING_CAP, MAX_ADDR_KEYS, MAX_NAME_KEYS, addr_keys, name_keys
from normalize import content_name_tokens, has_non_latin, latin_tokens
from paths import TRAIN_PREFIX, ZIP_PATH

SAMPLE_N = 4000
KS = (5, 10, 15, 20, 30)
SEED = 0


def reservoir_sample(zf: zipfile.ZipFile) -> list[tuple[str, list[str]]]:
    rng = random.Random(SEED)
    sample: list[tuple[str, list[str]]] = []
    seen = 0
    with zf.open(f"{TRAIN_PREFIX}/train_ground_truth.tsv") as f:
        f.readline()
        for line in f:
            seen += 1
            parts = line.decode().rstrip("\n").split("\t")
            sid = parts[0]
            mids = [x for x in parts[1].split(",") if x] if len(parts) > 1 else []
            row = (sid, mids)
            if len(sample) < SAMPLE_N:
                sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < SAMPLE_N:
                    sample[j] = row
            if seen % 500000 == 0:
                print(f"  gt rows {seen}", flush=True)
    print(f"sampled {len(sample)} from {seen}", flush=True)
    return sample


def load_s1(zf: zipfile.ZipFile, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    with zf.open(f"{TRAIN_PREFIX}/train_source1.tsv") as f:
        f.readline()
        for line in f:
            parts = line.decode().rstrip("\n").split("\t")
            if len(parts) != 4 or parts[0] not in wanted:
                continue
            out[parts[0]] = (parts[1], parts[2], parts[3])
            if len(out) == len(wanted):
                break
    return out


def main() -> None:
    t0 = time.time()
    zf = zipfile.ZipFile(ZIP_PATH)
    print("sampling ground truth", flush=True)
    sample = reservoir_sample(zf)
    wanted = {sid for sid, _ in sample}
    print("loading S1 sample", flush=True)
    s1 = load_s1(zf, wanted)
    print(f"loaded {len(s1)} S1", flush=True)

    key_to_q: dict[str, list[int]] = defaultdict(list)
    q_country: list[str] = []
    q_true: list[set[str]] = []
    q_name_toks: list[set[str]] = []
    non_latin_s1 = 0
    for i, (sid, mids) in enumerate(sample):
        name, addr, country = s1[sid]
        if has_non_latin(name):
            non_latin_s1 += 1
        q_country.append(country)
        q_true.append(set(mids))
        q_name_toks.append(set(latin_tokens(content_name_tokens(name))))
        for k in name_keys(name):
            key_to_q[country + "|n|" + k].append(i)
        for k in addr_keys(addr):
            key_to_q[country + "|a|" + k].append(i)
    print(f"query keys {len(key_to_q)} non_latin_s1 {non_latin_s1}", flush=True)

    true_ids: set[str] = set()
    for mids in q_true:
        true_ids.update(mids)
    match_text: dict[str, tuple[str, str, str]] = {}

    df: dict[str, int] = defaultdict(int)
    print("pass 1: document frequency", flush=True)
    n_docs = 0
    for src, fname in (("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as f:
            f.readline()
            for line_no, line in enumerate(f, 1):
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                eid, name, addr, country = parts
                n_docs += 1
                if eid in true_ids and eid not in match_text:
                    match_text[eid] = (name, addr, country)
                prefix_n = country + "|n|"
                prefix_a = country + "|a|"
                for k in name_keys(name):
                    full = prefix_n + k
                    if full in key_to_q:
                        df[full] += 1
                for k in addr_keys(addr):
                    full = prefix_a + k
                    if full in key_to_q:
                        df[full] += 1
                if line_no % 1000000 == 0:
                    print(f"  {src} {line_no}", flush=True)
    print(f"docs {n_docs} keys with df {len(df)} in {time.time()-t0:.0f}s", flush=True)

    kept_keys: set[str] = set()
    q_kept_n: list[list[str]] = [[] for _ in sample]
    q_kept_a: list[list[str]] = [[] for _ in sample]
    for full, qis in key_to_q.items():
        d = df.get(full, 0)
        if d == 0 or d > POSTING_CAP:
            continue
        channel = full.split("|", 2)[1]
        for qi in qis:
            bucket = q_kept_n[qi] if channel == "n" else q_kept_a[qi]
            bucket.append(full)
    for qi in range(len(sample)):
        def prefer(keys: list[str], must: tuple[str, ...], limit: int) -> list[str]:
            pinned = [k for k in keys if any(tag in k for tag in must)]
            rest = [k for k in keys if k not in pinned]
            rest.sort(key=lambda k: df[k])
            return (pinned + rest)[:limit]

        q_kept_n[qi] = prefer(q_kept_n[qi], ("|n|ne:", "|n|ns:", "|n|nd:"), MAX_NAME_KEYS)
        q_kept_a[qi] = prefer(q_kept_a[qi], ("|a|ah:", "|a|ap:", "|a|az:"), MAX_ADDR_KEYS)
        kept_keys.update(q_kept_n[qi])
        kept_keys.update(q_kept_a[qi])
    print(f"kept keys {len(kept_keys)}", flush=True)

    key_to_q_kept = {k: key_to_q[k] for k in kept_keys}
    postings: dict[str, array] = {k: array("I") for k in kept_keys}
    id_of: list[str] = []
    id_index: dict[str, int] = {}

    def doc_id(eid: str) -> int:
        d = id_index.get(eid)
        if d is None:
            d = len(id_of)
            id_index[eid] = d
            id_of.append(eid)
        return d

    print("pass 2: postings", flush=True)
    for src, fname in (("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")):
        with zf.open(f"{TRAIN_PREFIX}/{fname}") as f:
            f.readline()
            for line_no, line in enumerate(f, 1):
                parts = line.decode().rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                eid, name, addr, country = parts
                hits: list[str] = []
                prefix_n = country + "|n|"
                prefix_a = country + "|a|"
                for k in name_keys(name):
                    full = prefix_n + k
                    if full in key_to_q_kept:
                        hits.append(full)
                for k in addr_keys(addr):
                    full = prefix_a + k
                    if full in key_to_q_kept:
                        hits.append(full)
                if not hits:
                    continue
                d = doc_id(eid)
                for full in hits:
                    postings[full].append(d)
                if line_no % 1000000 == 0:
                    print(f"  {src} {line_no} indexed_docs {len(id_of)}", flush=True)
    print(f"indexed docs {len(id_of)} in {time.time()-t0:.0f}s", flush=True)

    idf_of = {
        full: (math.log((n_docs + 1) / (len(arr) + 1)) if arr else 0.0)
        for full, arr in postings.items()
    }
    name_scores = [defaultdict(float) for _ in sample]
    addr_scores = [defaultdict(float) for _ in sample]
    for qi in range(len(sample)):
        for full in q_kept_n[qi]:
            idf = idf_of[full]
            for d in postings[full]:
                name_scores[qi][d] += idf
        for full in q_kept_a[qi]:
            idf = idf_of[full]
            for d in postings[full]:
                addr_scores[qi][d] += idf

    def top_ids(score: dict[int, float], k: int) -> list[str]:
        if not score:
            return []
        items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [id_of[d] for d, _s in items]

    zero_pairs: list[tuple[int, str]] = []
    all_pairs: list[tuple[int, str]] = []
    non_latin_match = 0
    n_match_seen = 0
    for qi, (_sid, mids) in enumerate(sample):
        nt = q_name_toks[qi]
        for mid in mids:
            all_pairs.append((qi, mid))
            mt = match_text.get(mid)
            if mt is None:
                continue
            n_match_seen += 1
            if has_non_latin(mt[0]):
                non_latin_match += 1
            mt_toks = set(latin_tokens(content_name_tokens(mt[0])))
            if not (nt & mt_toks):
                zero_pairs.append((qi, mid))

    n_all = len(all_pairs)
    n_zero = len(zero_pairs)
    print("\nK  recall  mean_cands  zero_name_recall  n_zero", flush=True)
    for k in KS:
        unions = [
            set(top_ids(name_scores[qi], k)) | set(top_ids(addr_scores[qi], k))
            for qi in range(len(sample))
        ]
        hit = sum(1 for qi, mid in all_pairs if mid in unions[qi])
        zhit = sum(1 for qi, mid in zero_pairs if mid in unions[qi])
        mean_c = sum(len(u) for u in unions) / len(unions)
        rec = hit / n_all if n_all else 0
        zrec = zhit / n_zero if n_zero else 0
        print(f"{k:2d}  {rec:.3f}  {mean_c:.1f}  {zrec:.3f}  {n_zero}", flush=True)

    print(f"true links {n_all} match text found {n_match_seen}", flush=True)
    print(
        f"non_latin among found matches {non_latin_match}/{n_match_seen} "
        f"({non_latin_match/n_match_seen if n_match_seen else 0:.3f})",
        flush=True,
    )
    print(f"zero latin-token overlap links {n_zero}/{n_all}", flush=True)

    k = 15
    unions = [
        set(top_ids(name_scores[qi], k)) | set(top_ids(addr_scores[qi], k))
        for qi in range(len(sample))
    ]
    shown = 0
    print("\nmiss examples at K=15", flush=True)
    for qi, mid in all_pairs:
        if mid in unions[qi]:
            continue
        sid = sample[qi][0]
        sn, sa, sc = s1[sid]
        mt = match_text.get(mid, ("?", "?", "?"))
        print(f"S1 {sn} | {sa} | {sc}", flush=True)
        print(f"   {mid} {mt[0]} | {mt[1]} | {mt[2]}", flush=True)
        shown += 1
        if shown >= 12:
            break
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
