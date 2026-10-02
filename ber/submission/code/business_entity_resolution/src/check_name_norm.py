"""On the fixed validation pairs, see whether each name-normalization rule helps true pairs more than false ones."""

from __future__ import annotations

from collections import Counter

import numpy as np
from rapidfuzz.distance import JaroWinkler

from normalize import LEGAL, _edits_within, is_fuzzy_legal, tokens, transliterate
from paths import DATA_DIR

BOARD = DATA_DIR / "scoreboard"
TRAIN = DATA_DIR / "train"


def content(
    text: str,
    *,
    translit: bool = True,
    honorific: bool = True,
    legal: bool = True,
    com: bool = True,
    short: bool = True,
) -> list[str]:
    raw_text = transliterate(text) if translit else text
    raw = tokens(raw_text)
    if honorific and raw and raw[0] in {"m", "ms"}:
        raw = raw[1:]
    out = []
    for i, tok in enumerate(raw):
        if com and tok.endswith("com") and len(tok) > 6:
            tok = tok[:-3]
        tail = i >= len(raw) - 2
        if legal and (tok in LEGAL or (tail and is_fuzzy_legal(tok))):
            continue
        if short and len(tok) < 2:
            continue
        out.append(tok)
    return out


def jw(a: list[str], b: list[str]) -> float:
    left, right = " ".join(a), " ".join(b)
    if not left or not right:
        return 0.0
    return JaroWinkler.normalized_similarity(left, right)


def load_names(needed: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / name).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                eid, bname, _addr, _country = line.rstrip("\n").split("\t")
                if eid in needed:
                    found[eid] = bname
                    if len(found) == len(needed):
                        return found
    return found


def fuzzy_hits(text: str) -> list[str]:
    raw = tokens(transliterate(text))
    hits = []
    for i, tok in enumerate(raw):
        if tok in LEGAL or len(tok) < 5:
            continue
        if i < len(raw) - 2:
            continue
        for legal in ("private", "limited", "incorporated", "corporation", "company"):
            if _edits_within(tok, legal, 2):
                hits.append(tok)
                break
    return hits


def com_hits(text: str) -> list[str]:
    hits = []
    for tok in tokens(transliterate(text)):
        if tok.endswith("com") and len(tok) > 6:
            hits.append(tok)
    return hits


def main() -> None:
    data = np.load(BOARD / "pairs_30k.npz", allow_pickle=True)
    split = data["split"].astype(str)
    val = split == "val"
    if val.sum() == 0:
        raise SystemExit("no validation rows in the pair file")
    s1 = data["s1_id"].astype(str)[val]
    mid = data["match_id"].astype(str)[val]
    y = data["y"].astype(int)[val]
    names = load_names(set(s1.tolist()) | set(mid.tolist()))
    print(f"val pairs {len(y)} names {len(names)}", flush=True)
    rules = {
        "transliteration": dict(translit=False),
        "drop leading m/ms": dict(honorific=False),
        "strip legal endings": dict(legal=False),
        "strip glued com": dict(com=False),
        "drop 1-character tokens": dict(short=False),
    }
    base = [content(names.get(sid, "")) for sid in s1]
    other = [content(names.get(eid, "")) for eid in mid]
    base_jw = np.array([jw(a, b) for a, b in zip(base, other)], dtype=np.float32)
    print(f"baseline jw true {base_jw[y == 1].mean():.3f} false {base_jw[y == 0].mean():.3f}", flush=True)
    for label, flags in rules.items():
        left = [content(names.get(sid, ""), **flags) for sid in s1]
        right = [content(names.get(eid, ""), **flags) for eid in mid]
        alt = np.array([jw(a, b) for a, b in zip(left, right)], dtype=np.float32)
        delta = base_jw - alt
        true = y == 1
        false = ~true
        print(
            f"{label}: true delta {delta[true].mean():+.4f} "
            f"helped {(delta[true] > 0.05).sum()} hurt {(delta[true] < -0.05).sum()} | "
            f"false delta {delta[false].mean():+.4f} "
            f"made-closer {(delta[false] > 0.05).sum()} made-farther {(delta[false] < -0.05).sum()}",
            flush=True,
        )
        shown = 0
        order = np.argsort(delta)
        for i in order:
            if not true[i] or delta[i] >= -0.15 or shown >= 3:
                continue
            print(f"  true hurt {names.get(s1[i], '')} <> {names.get(mid[i], '')} {delta[i]:+.2f}", flush=True)
            shown += 1
        shown = 0
        for i in order[::-1]:
            if not false[i] or delta[i] <= 0.15 or shown >= 3:
                continue
            print(f"  false closer {names.get(s1[i], '')} <> {names.get(mid[i], '')} {delta[i]:+.2f}", flush=True)
            shown += 1
    fuzzy = Counter()
    com = Counter()
    for sid, eid, label in zip(s1, mid, y):
        for tok in fuzzy_hits(names.get(sid, "")) + fuzzy_hits(names.get(eid, "")):
            fuzzy[tok] += 1
        for tok in com_hits(names.get(sid, "")) + com_hits(names.get(eid, "")):
            com[(tok, int(label))] += 1
    print("fuzzy legal removals", fuzzy.most_common(12), flush=True)
    print("glued com", com.most_common(12), flush=True)


if __name__ == "__main__":
    main()
