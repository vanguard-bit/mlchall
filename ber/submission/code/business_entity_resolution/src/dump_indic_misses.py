"""Dump Indic-script validation misses for the romanized-name check."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from rapidfuzz.distance import JaroWinkler

from normalize import content_name_tokens
from paths import DATA_DIR

BOARD = DATA_DIR / "scoreboard"
TRAIN = DATA_DIR / "train"
OUT = BOARD / "indic_misses.json"


def script_kind(text: str) -> str:
    indic = False
    for ch in text:
        if ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF:
            indic = True
            break
    return "indic" if indic else "latin"


def lang_of(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "hi"
        if 0x0980 <= o <= 0x09FF:
            return "bn"
        if 0x0A00 <= o <= 0x0A7F:
            return "pa"
        if 0x0A80 <= o <= 0x0AFF:
            return "gu"
        if 0x0B00 <= o <= 0x0B7F:
            return "or"
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        if 0x0C80 <= o <= 0x0CFF:
            return "kn"
        if 0x0D00 <= o <= 0x0D7F:
            return "ml"
    return "hi"


def name_jw(a: str, b: str) -> float:
    la = " ".join(content_name_tokens(a))
    lb = " ".join(content_name_tokens(b))
    if not la or not lb:
        return 0.0
    return JaroWinkler.normalized_similarity(la, lb)


def main() -> None:
    data = np.load(BOARD / "pairs.npz", allow_pickle=True)
    names = [str(x) for x in data["feature_names"]]
    val = (data["split"].astype(str) == "val") & (data["y"].astype(int) == 1)
    fn = data["X"][:, names.index("from_name_ch")]
    fa = data["X"][:, names.index("from_addr_ch")]
    found: dict[str, set[str]] = defaultdict(set)
    for sid, eid, nflag, aflag in zip(
        data["s1_id"].astype(str)[val],
        data["match_id"].astype(str)[val],
        fn[val],
        fa[val],
    ):
        if float(nflag) >= 0.5 or float(aflag) >= 0.5:
            found[str(sid)].add(str(eid))
    meta = json.loads((BOARD / "meta.json").read_text(encoding="utf-8"))
    misses = []
    need: set[str] = set()
    for row in meta:
        if row["split"] != "val":
            continue
        for eid in row["truth"]:
            if eid not in found.get(row["sid"], ()):
                misses.append((row["sid"], eid))
                need.add(row["sid"])
                need.add(eid)
    texts: dict[str, tuple[str, str]] = {}
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, addr, _country = line.rstrip("\n").split("\t")
                if eid in need:
                    texts[eid] = (bname, addr)
                    if len(texts) == len(need):
                        break
        if len(texts) == len(need):
            break
    rows = []
    for sid, eid in misses:
        n1, a1 = texts[sid]
        n2, a2 = texts.get(eid, ("", ""))
        k1, k2 = script_kind(n1), script_kind(n2)
        if k1 == k2:
            continue
        jw = name_jw(n1, n2)
        indic_name = n1 if k1 == "indic" else n2
        rows.append({
            "sid": sid,
            "eid": eid,
            "s1_name": n1,
            "s1_addr": a1,
            "m_name": n2,
            "m_addr": a2,
            "jw": round(jw, 4),
            "lang": lang_of(indic_name),
            "band": "hard" if jw < 0.60 else ("mid" if jw < 0.80 else "high"),
        })
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    from collections import Counter
    print(f"indic misses {len(rows)}", dict(Counter(r["band"] for r in rows)), dict(Counter(r["lang"] for r in rows)))


if __name__ == "__main__":
    main()
