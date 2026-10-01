"""Rare-token IDF and the city token the address parser currently drops."""

from __future__ import annotations

import math
from collections import Counter

from rapidfuzz.distance import JaroWinkler

from normalize import ADDR_STOP, LEGAL, digit_tokens, tokens
from paths import DATA_DIR
from v4_features import content_name_tokens_v4


def document_frequency() -> tuple[dict[str, int], int]:
    """Token document frequency over train Source 1 names. One name is one document."""
    counts: Counter[str] = Counter()
    n_docs = 0
    path = DATA_DIR / "train" / "train_source1.tsv"
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            name = line.split("\t", 2)[1]
            seen = set(content_name_tokens_v4(name))
            if not seen:
                n_docs += 1
                continue
            counts.update(seen)
            n_docs += 1
    return dict(counts), n_docs


def idf_table(counts: dict[str, int], n_docs: int) -> dict[str, float]:
    scale = math.log(n_docs + 1)
    return {token: math.log((n_docs + 1) / (df + 1)) / scale for token, df in counts.items()}


def rare_two(left: frozenset[str], right: frozenset[str], idf: dict[str, float], missing: float) -> list[float]:
    """Max IDF of a shared token, and max IDF of a token on only one side."""
    shared = left & right
    conflict = (left | right) - shared
    shared_max = max((idf.get(token, missing) for token in shared), default=0.0)
    conflict_max = max((idf.get(token, missing) for token in conflict), default=0.0)
    return [shared_max, conflict_max]


def city_token(addr: str) -> str:
    """Last street-like token, which parse_addr drops when a street remains before it."""
    house = ""
    streets: list[str] = []
    for tok in tokens(addr):
        digits = [d for d in digit_tokens(tok) if 1 <= len(d) <= 6]
        if not house and digits:
            house = digits[0]
            continue
        if house and tok not in ADDR_STOP and tok not in LEGAL and len(tok) >= 4 and not tok.isdigit():
            streets.append(tok)
    if len(streets) >= 2:
        return streets[-1]
    return ""


def city_two(left: str, right: str) -> list[float]:
    if not left or not right:
        return [0.0, 0.0]
    return [JaroWinkler.normalized_similarity(left, right), float(left == right)]


_SKIP = {"and", "the", "of", "for"}


def soft_align(left: frozenset[str], right: frozenset[str], idf: dict[str, float], missing: float) -> float:
    """IDF-weighted token alignment. A token counts only if Jaro-Winkler is at least 0.9."""
    if not left or not right:
        return 0.0
    num = 0.0
    den = 0.0
    right_list = list(right)
    for tok in left:
        weight = idf.get(tok, missing)
        den += weight
        best = 1.0 if tok in right else 0.0
        if best < 0.9:
            for other in right_list:
                if abs(len(tok) - len(other)) > 2:
                    continue
                sim = JaroWinkler.normalized_similarity(tok, other)
                if sim > best:
                    best = sim
                    if best >= 0.999:
                        break
        if best >= 0.9:
            num += weight * best
    return num / den if den else 0.0


def initialism(joined_left: str, joined_right: str) -> float:
    """1 when a 2–5 letter token equals the other name's initials."""
    left = [tok for tok in joined_left.split() if tok not in _SKIP]
    right = [tok for tok in joined_right.split() if tok not in _SKIP]
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    left_init = "".join(tok[0] for tok in left)
    right_init = "".join(tok[0] for tok in right)
    if 2 <= len(left_init) <= 5 and left_init in right_set:
        return 1.0
    if 2 <= len(right_init) <= 5 and right_init in left_set:
        return 1.0
    return 0.0
