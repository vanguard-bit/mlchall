"""Field normalization for business name and address.

Keeps several representations. Legal suffixes are stripped only from the
content form; the raw string is preserved by the caller.
"""

from __future__ import annotations

import re
import string

LEGAL = {
    "pvt", "private", "ltd", "limited", "inc", "incorporated", "corp",
    "corporation", "llc", "llp", "lp", "co", "company", "sarl", "sas",
    "sa", "eurl", "gmbh", "plc", "pllc", "pc", "pa", "pte", "bv", "nv",
    "ag", "kg", "oy", "ab", "srl", "spa", "sl", "ltda", "opc", "lllp",
}

ADDR_STOP = {
    "road", "street", "lane", "avenue", "ave", "blvd", "drive", "dr",
    "nagar", "marg", "colony", "coloney", "floor", "flr", "near", "opp",
    "opposite", "behind", "beside", "sector", "plot", "flat", "house",
    "no", "number", "ward", "block", "phase", "stage", "cross", "main",
    "city", "dist", "district", "tal", "taluk", "tq", "po", "ps", "via",
    "india", "indian", "state", "pin", "zip", "code", "apartment", "apt",
    "suite", "unit", "building", "bldg", "tower", "complex", "market",
    "bazaar", "chowk", "circle", "highway", "hwy", "rd", "st", "ln",
    "ct", "court", "place", "pl", "park", "society", "soc", "hsg",
    "cooperative", "coop", "limited", "private", "the", "and", "of",
}

_PUNCT = str.maketrans({c: " " for c in string.punctuation})
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.I)
_SPACE = re.compile(r"\s+")
_LATIN = re.compile(r"[a-z0-9]+")
_DIGIT = re.compile(r"\d+")


def clean(text: str) -> str:
    text = text.lower().replace("&", " and ")
    if "http" in text or "www." in text or "@" in text:
        text = _URL.sub(" ", text)
    text = text.translate(_PUNCT)
    return _SPACE.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    if not text:
        return []
    return clean(text).split()


_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क्ष": "ksh",
    "ज्ञ": "gy", "ड़": "d", "ढ़": "dh",
}
_MATRA = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
    "ॅ": "e", "ॉ": "o",
}
_VOWEL = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
}
_HALANT = "्"
_ANUSVARA = "ं"
_CHANDRA = "ँ"

_GURMUKHI = {
    "ਅ": "a", "ਆ": "a", "ਇ": "i", "ਈ": "i", "ਉ": "u", "ਊ": "u",
    "ਏ": "e", "ਐ": "ai", "ਓ": "o", "ਔ": "au",
    "ਕ": "k", "ਖ": "kh", "ਗ": "g", "ਘ": "gh", "ਙ": "n",
    "ਚ": "ch", "ਛ": "chh", "ਜ": "j", "ਝ": "jh", "ਞ": "n",
    "ਟ": "t", "ਠ": "th", "ਡ": "d", "ਢ": "dh", "ਣ": "n",
    "ਤ": "t", "ਥ": "th", "ਦ": "d", "ਧ": "dh", "ਨ": "n",
    "ਪ": "p", "ਫ": "ph", "ਬ": "b", "ਭ": "bh", "ਮ": "m",
    "ਯ": "y", "ਰ": "r", "ਲ": "l", "ਵ": "v", "ਸ": "s", "ਹ": "h",
    "ਾ": "a", "ਿ": "i", "ੀ": "i", "ੁ": "u", "ੂ": "u", "ੇ": "e", "ੈ": "ai", "ੋ": "o", "ੌ": "au",
    "੍": "", "ਂ": "n", "ੰ": "n", "ੱ": "",
}


def _gurmukhi_to_latin(text: str) -> str:
    if not any("\u0a00" <= ch <= "\u0a7f" for ch in text):
        return text
    return "".join(_GURMUKHI.get(ch, ch) for ch in text)


def transliterate(text: str) -> str:
    text = _gurmukhi_to_latin(text)
    if not any(ord(ch) > 127 for ch in text):
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _CONS:
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == _HALANT:
                out.append(_CONS[ch])
                i += 2
                continue
            if nxt in _MATRA:
                out.append(_CONS[ch] + _MATRA[nxt])
                i += 2
                continue
            out.append(_CONS[ch] + "a")
            i += 1
            continue
        if ch in _VOWEL:
            out.append(_VOWEL[ch])
        elif ch in _MATRA:
            out.append(_MATRA[ch])
        elif ch in (_ANUSVARA, _CHANDRA):
            out.append("n")
        elif ch == "ः":
            out.append("h")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _edits_within(a: str, b: str, limit: int) -> bool:
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            best = ins if ins < delete else delete
            if sub < best:
                best = sub
            cur.append(best)
            if best < row_min:
                row_min = best
        if row_min > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def is_fuzzy_legal(token: str) -> bool:
    if token in LEGAL:
        return True
    if len(token) < 5:
        return False
    for legal in ("private", "limited", "incorporated", "corporation", "company"):
        if _edits_within(token, legal, 2):
            return True
    return False


def content_name_tokens(text: str) -> list[str]:
    text = transliterate(text)
    raw = tokens(text)
    if raw and raw[0] in {"m", "ms"}:
        raw = raw[1:]
    out = []
    for i, t in enumerate(raw):
        if t.endswith("com") and len(t) > 6:
            t = t[:-3]
        tail = i >= len(raw) - 2
        if t in LEGAL or (tail and is_fuzzy_legal(t)) or len(t) < 2:
            continue
        out.append(t)
    return out


def latin_tokens(ts: list[str]) -> list[str]:
    return [t for t in ts if _LATIN.fullmatch(t)]


def has_non_latin(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and ord(ch) > 127:
            return True
    return False


def digit_tokens(text: str) -> list[str]:
    return _DIGIT.findall(text)


def address_tokens(text: str) -> list[str]:
    out = []
    for t in tokens(text):
        if t in ADDR_STOP or t in LEGAL or len(t) < 4:
            continue
        if t.isdigit():
            continue
        out.append(t)
    return out
