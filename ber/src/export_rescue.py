"""Add strict rescue matches on top of the finished country files.

Waits until France, the US, and India have all been scored and memory is free,
then writes output/matching_results_v2.tsv.
"""

from __future__ import annotations

import gc
import time
from array import array
from pathlib import Path

import numpy as np

from blocking import rescue_keys
from paths import DATA_DIR, ROOT
from rescue import RESCUE_CAP, rescue_accept

EXPORT = DATA_DIR / "export"
TEST_DIR = DATA_DIR / "test"
OUT_PATH = ROOT / "output" / "matching_results_v2.tsv"
TOP_K = 30


def mem_available_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def wait_until_ready() -> None:
    needed = [EXPORT / name / "done" for name in ("France", "US", "India")]
    while True:
        missing = [path.parent.name for path in needed if not path.exists()]
        avail = mem_available_gb()
        if not missing and avail >= 6.0:
            print(f"ready, mem_available={avail:.1f}GB", flush=True)
            return
        print(f"waiting missing={missing or '-'} mem_available={avail:.1f}GB", flush=True)
        time.sleep(20)


def load_base(country: str) -> dict[str, list[str]]:
    base: dict[str, list[str]] = {}
    with (EXPORT / country / "matching.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            sid, _, rest = line.rstrip("\n").partition("\t")
            base[sid] = [x for x in rest.split(",") if x] if rest else []
    return base


def load_s1(country: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with (TEST_DIR / "test_source1.tsv").open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4 and parts[3] == country:
                out[parts[0]] = (parts[1], parts[2])
    return out


def rescue_country(country: str, s1: dict[str, tuple[str, str]]) -> dict[str, list[str]]:
    t0 = time.time()
    sids = list(s1)
    query_keys: set[str] = set()
    q_keys: list[list[str]] = []
    for sid in sids:
        name, addr = s1[sid]
        keys = [country + "|r|" + key for key in rescue_keys(name, addr)]
        q_keys.append(keys)
        query_keys.update(keys)
    print(f"{country}: {len(sids)} queries, {len(query_keys)} keys", flush=True)

    df: dict[str, int] = {}
    n_docs = 0
    for fname in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST_DIR / fname).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4 or parts[3] != country:
                    continue
                n_docs += 1
                prefix = country + "|r|"
                for key in rescue_keys(parts[1], parts[2]):
                    full = prefix + key
                    if full in query_keys:
                        df[full] = df.get(full, 0) + 1
                if n_docs % 1_000_000 == 0:
                    print(f"  {country} df {n_docs}", flush=True)
    kept = [key for key, count in df.items() if 0 < count <= RESCUE_CAP]
    key_id = {key: i for i, key in enumerate(kept)}
    del query_keys
    del df
    q_kids = [[key_id[key] for key in keys if key in key_id] for keys in q_keys]
    del q_keys
    print(f"{country}: kept {len(kept)} keys", flush=True)

    key_arr = array("I")
    doc_arr = array("I")
    id_of: list[str] = []
    id_index: dict[str, int] = {}
    for fname in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST_DIR / fname).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4 or parts[3] != country:
                    continue
                hits = []
                prefix = country + "|r|"
                for key in rescue_keys(parts[1], parts[2]):
                    kid = key_id.get(prefix + key)
                    if kid is not None:
                        hits.append(kid)
                if not hits:
                    continue
                doc = id_index.get(parts[0])
                if doc is None:
                    doc = len(id_of)
                    id_index[parts[0]] = doc
                    id_of.append(parts[0])
                for kid in hits:
                    key_arr.append(kid)
                    doc_arr.append(doc)
    del id_index
    del key_id
    print(f"{country}: indexed {len(id_of)} in {time.time() - t0:.0f}s", flush=True)
    if not id_of:
        return {sid: [] for sid in sids}

    key_np = np.frombuffer(key_arr, dtype=np.uint32, count=len(key_arr)).copy()
    doc_np = np.frombuffer(doc_arr, dtype=np.uint32, count=len(doc_arr)).copy()
    del key_arr
    del doc_arr
    order = np.argsort(key_np, kind="stable")
    postings = doc_np[order]
    counts = np.bincount(key_np, minlength=len(kept))
    del key_np
    del doc_np
    del kept
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    del counts

    picked: list[list[str]] = []
    need: set[str] = set()
    score_buf = np.zeros(len(id_of), dtype=np.int16)
    for kids in q_kids:
        chunks = []
        for kid in kids:
            start, end = int(offsets[kid]), int(offsets[kid + 1])
            if start == end:
                continue
            idx = postings[start:end]
            np.add.at(score_buf, idx, 1)
            chunks.append(idx)
        if not chunks:
            picked.append([])
            continue
        seen = np.unique(np.concatenate(chunks))
        scores = score_buf[seen]
        n = int(scores.shape[0])
        if n > TOP_K:
            choose = np.argpartition(scores, -TOP_K)[-TOP_K:]
        else:
            choose = np.arange(n)
        mids = [id_of[int(seen[i])] for i in choose]
        picked.append(mids)
        need.update(mids)
        score_buf[seen] = 0
    del postings
    del score_buf
    del id_of
    gc.collect()

    texts: dict[str, tuple[str, str]] = {}
    if need:
        for fname in ("test_source2.tsv", "test_source3.tsv"):
            with (TEST_DIR / fname).open(encoding="utf-8") as handle:
                handle.readline()
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 4 or parts[0] not in need:
                        continue
                    texts[parts[0]] = (parts[1], parts[2])
                    if len(texts) == len(need):
                        break
            if len(texts) == len(need):
                break

    added: dict[str, list[str]] = {}
    n_add = 0
    for sid, mids in zip(sids, picked):
        name, addr = s1[sid]
        extra = []
        for mid in mids:
            pair = texts.get(mid)
            if pair is None:
                continue
            if rescue_accept(name, addr, pair[0], pair[1]):
                extra.append(mid)
        added[sid] = extra if len(extra) == 1 else []
        n_add += len(extra)
    print(f"{country}: rescue accepts {n_add} in {time.time() - t0:.0f}s", flush=True)
    return added


def main() -> None:
    wait_until_ready()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tsv.tmp")
    n = 0
    n_add = 0
    with tmp.open("w", encoding="utf-8") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for country in ("France", "US", "India"):
            base = load_base(country)
            s1 = load_s1(country)
            extra = rescue_country(country, s1)
            for sid, mids in base.items():
                have = set(mids)
                for mid in extra.get(sid, []):
                    if mid not in have:
                        mids.append(mid)
                        have.add(mid)
                        n_add += 1
                out.write(f"{sid}\t{','.join(mids)}\n")
                n += 1
            del base
            del s1
            del extra
            gc.collect()
    tmp.replace(OUT_PATH)
    print(f"wrote {OUT_PATH} rows={n} added={n_add}", flush=True)


if __name__ == "__main__":
    main()
