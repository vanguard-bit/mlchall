"""Numeric features for a Source1 ↔ Source2/3 candidate pair."""

from __future__ import annotations

from typing import NamedTuple

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize import (
    address_tokens,
    content_name_tokens,
    digit_tokens,
    latin_tokens,
    tokens,
)


class Prepared(NamedTuple):
    n: str
    a: str
    name_toks: frozenset[str]
    addr_toks: frozenset[str]
    postals: frozenset[str]
    houses: frozenset[str]
    raw_name: str
    addr_empty: float
    ng_name: frozenset[str]
    ng_addr: frozenset[str]


def _ngram_set(text: str, n: int = 3) -> frozenset[str]:
    text = "".join(ch for ch in text.lower() if ch.isalnum())
    if len(text) < n:
        return frozenset((text,)) if text else frozenset()
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cos(sa: frozenset[str], sb: frozenset[str]) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / ((len(sa) * len(sb)) ** 0.5)


def prepare_record(name: str, addr: str) -> Prepared:
    content = content_name_tokens(name)
    joined_name = " ".join(content)
    addr_toks = address_tokens(addr)
    joined_addr = " ".join(addr_toks)
    digits = set(digit_tokens(addr))
    return Prepared(
        n=joined_name,
        a=joined_addr,
        name_toks=frozenset(latin_tokens(content)),
        addr_toks=frozenset(addr_toks),
        postals=frozenset(d for d in digits if len(d) >= 5),
        houses=frozenset(d for d in digits if 1 <= len(d) <= 6),
        raw_name=" ".join(tokens(name)),
        addr_empty=float(not addr.strip()),
        ng_name=_ngram_set(joined_name, 3),
        ng_addr=_ngram_set(joined_addr, 3),
    )


def features_prepared(
    left: Prepared,
    right: Prepared,
    *,
    country_eq: float,
    name_rank: float = 99.0,
    addr_rank: float = 99.0,
    name_score: float = 0.0,
    addr_score: float = 0.0,
    from_name_channel: int = 0,
    from_addr_channel: int = 0,
) -> list[float]:
    return [
        JaroWinkler.normalized_similarity(left.n, right.n),
        fuzz.ratio(left.n, right.n) / 100.0,
        _jaccard(left.name_toks, right.name_toks),
        float(left.name_toks == right.name_toks and bool(left.name_toks)),
        _cos(left.ng_name, right.ng_name),
        JaroWinkler.normalized_similarity(left.a, right.a),
        fuzz.ratio(left.a, right.a) / 100.0,
        _jaccard(left.addr_toks, right.addr_toks),
        _cos(left.ng_addr, right.ng_addr),
        float(bool(left.postals & right.postals)),
        float(bool(left.houses & right.houses)),
        country_eq,
        left.addr_empty,
        right.addr_empty,
        fuzz.ratio(left.raw_name, right.raw_name) / 100.0,
        name_rank,
        addr_rank,
        name_score,
        addr_score,
        float(from_name_channel),
        float(from_addr_channel),
    ]


def pair_feature_dict(
    s1_name: str,
    s1_addr: str,
    s1_cty: str,
    m_name: str,
    m_addr: str,
    m_cty: str,
    *,
    name_rank: float = 0.0,
    addr_rank: float = 0.0,
    name_score: float = 0.0,
    addr_score: float = 0.0,
    from_name_channel: int = 0,
    from_addr_channel: int = 0,
) -> dict[str, float]:
    vals = features_prepared(
        prepare_record(s1_name, s1_addr),
        prepare_record(m_name, m_addr),
        country_eq=float(s1_cty == m_cty),
        name_rank=name_rank,
        addr_rank=addr_rank,
        name_score=name_score,
        addr_score=addr_score,
        from_name_channel=from_name_channel,
        from_addr_channel=from_addr_channel,
    )
    return dict(zip(FEATURE_NAMES, vals, strict=True))


FEATURE_NAMES = [
    "jw_name",
    "ratio_name",
    "token_jaccard_name",
    "sorted_token_eq_name",
    "ngram_cos_name",
    "jw_addr",
    "ratio_addr",
    "token_jaccard_addr",
    "ngram_cos_addr",
    "postal_eq",
    "house_eq",
    "country_eq",
    "s1_addr_empty",
    "m_addr_empty",
    "raw_ratio_name",
    "name_rank",
    "addr_rank",
    "name_block_score",
    "addr_block_score",
    "from_name_ch",
    "from_addr_ch",
]
