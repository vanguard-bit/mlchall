"""Same-country inverted-index blocking (name + address channels)."""

from __future__ import annotations

import math
import re
from array import array
from collections import defaultdict
from dataclasses import dataclass

from normalize import address_tokens, content_name_tokens, digit_tokens

POSTING_CAP = 2500
MAX_NAME_KEYS = 14
MAX_ADDR_KEYS = 12
DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][-a-z0-9]*(?:\.[a-z0-9][-a-z0-9]*)+)",
    re.I,
)


def name_keys(name: str) -> set[str]:
    toks = content_name_tokens(name)
    keys: set[str] = set()
    if not toks:
        return keys
    phrase = " ".join(toks)
    if len(phrase) >= 4:
        keys.add("ne:" + phrase)
    for t in toks:
        if len(t) >= 3:
            keys.add("nt:" + t)
    if len(toks) >= 2:
        keys.add("ns:" + " ".join(sorted(toks)))
    squashed = "".join(t for t in toks if t.isascii())
    if len(squashed) >= 6:
        step = max(1, (len(squashed) - 3) // 10)
        for i in range(0, len(squashed) - 3, step):
            keys.add("ng:" + squashed[i : i + 4])
    raw = name.lower()
    for m in DOMAIN_RE.finditer(raw):
        dom = m.group(1).split("/")[0]
        base = dom.replace(".com", "").replace(".org", "").replace(".net", "")
        if len(base) >= 6:
            keys.add("nd:" + base)
        if len(dom) >= 6:
            keys.add("nd:" + dom.replace(".", ""))
    return keys


def addr_keys(addr: str) -> set[str]:
    keys: set[str] = set()
    if not addr or not addr.strip():
        return keys
    words = address_tokens(addr)
    for t in words[:16]:
        keys.add("at:" + t)
    digits = digit_tokens(addr)
    postals = [d for d in digits if len(d) >= 5]
    houses = [d for d in digits if 1 <= len(d) <= 6]
    for p in postals[:3]:
        keys.add("az:" + p)
    streets = [t for t in words if not t.isdigit() and len(t) >= 4]
    picked = streets[:3] + streets[-2:]
    seen: set[str] = set()
    street_keys: list[str] = []
    for s in picked:
        if s not in seen:
            seen.add(s)
            street_keys.append(s)
    for h in houses[:3]:
        for s in street_keys[:5]:
            keys.add(f"ah:{h}:{s}")
    for p in postals[:2]:
        for s in street_keys[:3]:
            keys.add(f"ap:{p}:{s}")
    if len(streets) >= 2:
        keys.add("as:" + " ".join(sorted(streets[:3])))
    if len(streets) >= 1:
        city = streets[-1]
        for h in houses[:2]:
            keys.add(f"ah:{h}:{city}")
    return keys


def _skeleton(token: str) -> str:
    out: list[str] = []
    prev = ""
    for ch in token:
        if ch in "aeiou" or not ch.isalpha():
            continue
        if ch == prev:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def rescue_keys(name: str, addr: str) -> set[str]:
    """Keys for typos and domain-names the main index ranks too low.

    Full consonant skeletons plus each single-character deletion, and an
    8-consonant prefix of the whole name and of any domain.
    """
    keys: set[str] = set()
    toks = [t for t in content_name_tokens(name) if t.isascii() and len(t) >= 5]
    skeletons = [sk for t in toks if len(sk := _skeleton(t)) >= 5]
    for sk in skeletons[:6]:
        keys.add("rk:" + sk)
        if len(sk) <= 12:
            for i in range(len(sk)):
                keys.add("rd:" + sk[:i] + sk[i + 1 :])
    squashed = _skeleton("".join(toks))
    if len(squashed) >= 8:
        keys.add("rp:" + squashed[:8])
    raw = name.lower()
    for match in DOMAIN_RE.finditer(raw):
        base = match.group(1).split("/")[0]
        base = base.replace(".com", "").replace(".org", "").replace(".net", "")
        sk = _skeleton(base)
        if len(sk) >= 8:
            keys.add("rp:" + sk[:8])
        if len(sk) >= 5:
            keys.add("rk:" + sk[:12])
    houses = [d for d in digit_tokens(addr) if 1 <= len(d) <= 6]
    cities = [t for t in address_tokens(addr) if len(t) >= 4]
    if houses and cities:
        keys.add(f"rh:{houses[0]}:{cities[-1]}")
    return keys


def query_key_lists(
    name: str,
    addr: str,
    country: str,
    df: dict[str, int],
    *,
    posting_cap: int = POSTING_CAP,
    max_name_keys: int = MAX_NAME_KEYS,
    max_addr_keys: int = MAX_ADDR_KEYS,
) -> tuple[list[str], list[str]]:
    prefix_n = country + "|n|"
    prefix_a = country + "|a|"
    cand_n: list[str] = []
    cand_a: list[str] = []
    for k in name_keys(name):
        full = prefix_n + k
        d = df.get(full, 0)
        if 0 < d <= posting_cap:
            cand_n.append(full)
    for k in addr_keys(addr):
        full = prefix_a + k
        d = df.get(full, 0)
        if 0 < d <= posting_cap:
            cand_a.append(full)

    def prefer(keys: list[str], must: tuple[str, ...], limit: int) -> list[str]:
        pinned = [k for k in keys if any(tag in k for tag in must)]
        rest = [k for k in keys if k not in pinned]
        rest.sort(key=lambda k: df[k])
        return (pinned + rest)[:limit]

    kept_n = prefer(cand_n, ("|n|ne:", "|n|ns:", "|n|nd:"), max_name_keys)
    kept_a = prefer(cand_a, ("|a|ah:", "|a|ap:", "|a|az:"), max_addr_keys)
    return kept_n, kept_a


@dataclass
class BlockerConfig:
    posting_cap: int = POSTING_CAP
    max_name_keys: int = MAX_NAME_KEYS
    max_addr_keys: int = MAX_ADDR_KEYS
    k_name: int = 20
    k_addr: int = 20


class CountryBlocker:
    """Two-pass inverted index over Source 2/3 rows in one country partition."""

    def __init__(self, config: BlockerConfig | None = None) -> None:
        self.cfg = config or BlockerConfig()
        self.df: dict[str, int] = {}
        self.n_docs = 0
        self.postings: dict[str, array] = {}
        self.id_of: list[str] = []
        self.id_index: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def _doc_id(self, eid: str) -> int:
        d = self.id_index.get(eid)
        if d is None:
            d = len(self.id_of)
            self.id_index[eid] = d
            self.id_of.append(eid)
        return d

    def pass1_df(self, rows, query_keys: set[str]) -> None:
        df: dict[str, int] = defaultdict(int)
        n_docs = 0
        for _eid, name, addr, country in rows:
            n_docs += 1
            prefix_n = country + "|n|"
            prefix_a = country + "|a|"
            for k in name_keys(name):
                full = prefix_n + k
                if full in query_keys:
                    df[full] += 1
            for k in addr_keys(addr):
                full = prefix_a + k
                if full in query_keys:
                    df[full] += 1
        self.df = dict(df)
        self.n_docs = n_docs

    def pass2_postings(self, rows, kept_keys: set[str]) -> None:
        self.postings = {k: array("I") for k in kept_keys}
        for eid, name, addr, country in rows:
            hits: list[str] = []
            prefix_n = country + "|n|"
            prefix_a = country + "|a|"
            for k in name_keys(name):
                full = prefix_n + k
                if full in kept_keys:
                    hits.append(full)
            for k in addr_keys(addr):
                full = prefix_a + k
                if full in kept_keys:
                    hits.append(full)
            if not hits:
                continue
            d = self._doc_id(eid)
            for full in hits:
                self.postings[full].append(d)
        self.idf = {
            full: (math.log((self.n_docs + 1) / (len(arr) + 1)) if arr else 0.0)
            for full, arr in self.postings.items()
        }

    def score_query(
        self, name: str, addr: str, country: str
    ) -> tuple[dict[int, float], dict[int, float], list[str], list[str]]:
        kept_n, kept_a = query_key_lists(
            name,
            addr,
            country,
            self.df,
            posting_cap=self.cfg.posting_cap,
            max_name_keys=self.cfg.max_name_keys,
            max_addr_keys=self.cfg.max_addr_keys,
        )
        ns: dict[int, float] = defaultdict(float)
        asc: dict[int, float] = defaultdict(float)
        for full in kept_n:
            idf = self.idf.get(full, 0.0)
            arr = self.postings.get(full)
            if arr is None:
                continue
            for d in arr:
                ns[d] += idf
        for full in kept_a:
            idf = self.idf.get(full, 0.0)
            arr = self.postings.get(full)
            if arr is None:
                continue
            for d in arr:
                asc[d] += idf
        return ns, asc, kept_n, kept_a

    def candidates(
        self, name: str, addr: str, country: str
    ) -> tuple[list[str], list[str], list[str]]:
        ns, asc, _, _ = self.score_query(name, addr, country)
        kn, ka = self.cfg.k_name, self.cfg.k_addr

        def top(score: dict[int, float], k: int) -> list[str]:
            if not score:
                return []
            items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
            return [self.id_of[d] for d, _ in items]

        name_top = top(ns, kn)
        addr_top = top(asc, ka)
        union = list(dict.fromkeys(name_top + addr_top))
        return union, name_top, addr_top
