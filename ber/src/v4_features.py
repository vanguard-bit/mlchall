"""v4 pair features. Does not change the tokenizer the running v3 scorer imported."""

from __future__ import annotations

import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize import LEGAL, digit_tokens, latin_tokens, tokens, transliterate
from pair_features import _cos, _jaccard, _ngram_set

TLDS = {"com", "org", "net", "info", "biz", "www"}


def content_name_tokens_v4(text: str) -> list[str]:
    """v3 tokenizer without fuzzy legal deletion, glued-com chopping, or 1-character drops."""
    raw = tokens(transliterate(text))
    if raw and raw[0] in {"m", "ms"}:
        raw = raw[1:]
    return [tok for tok in raw if tok not in LEGAL]


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def parse_addr(addr: str) -> tuple[str, frozenset[str], str]:
    from normalize import ADDR_STOP

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


def name_view(name: str) -> tuple[str, frozenset[str], str, str]:
    toks = content_name_tokens_v4(name)
    folded = [tok for tok in content_name_tokens_v4(_fold(name)) if tok not in TLDS]
    return " ".join(toks), frozenset(latin_tokens(toks)), " ".join(folded), "".join(folded)


def name_five(left: tuple[str, frozenset[str], str, str], right: tuple[str, frozenset[str], str, str]) -> list[float]:
    lj, lt, _lf, _lq = left
    rj, rt, _rf, _rq = right
    if lj and rj:
        jw = JaroWinkler.normalized_similarity(lj, rj)
        ratio = fuzz.ratio(lj, rj) / 100.0
        cos = _cos(_ngram_set(lj, 3), _ngram_set(rj, 3))
    else:
        jw = ratio = cos = 0.0
    return [jw, ratio, _jaccard(lt, rt), float(lt == rt and bool(lt)), cos]


def extra_nine(
    left_addr: tuple[str, frozenset[str], str],
    right_addr: tuple[str, frozenset[str], str],
    left_name: tuple[str, frozenset[str], str, str],
    right_name: tuple[str, frozenset[str], str, str],
) -> list[float]:
    ah, ast, az = left_addr
    bh, bst, bz = right_addr
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
    fold = JaroWinkler.normalized_similarity(left_name[2], right_name[2]) if left_name[2] and right_name[2] else 0.0
    squash = fuzz.ratio(left_name[3], right_name[3]) / 100.0 if left_name[3] and right_name[3] else 0.0
    street = 0.0 if not ast or not bst else len(ast & bst) / len(ast | bst)
    return [same, missing, conflict, prefix, street, postal_same, postal_conflict, fold, squash]


PARTICLES = {"le", "la", "les", "de", "du", "des"}


def particle_two(
    left_name: tuple[str, frozenset[str], str, str],
    right_name: tuple[str, frozenset[str], str, str],
) -> list[float]:
    """Alternate name with French particles dropped. The main name features stay intact."""
    left = [tok for tok in left_name[0].split() if tok not in PARTICLES]
    right = [tok for tok in right_name[0].split() if tok not in PARTICLES]
    lj, rj = " ".join(left), " ".join(right)
    jw = JaroWinkler.normalized_similarity(lj, rj) if lj and rj else 0.0
    return [jw, _jaccard(frozenset(left), frozenset(right))]
