"""Shared accent-fold and IndicXlit name key."""

from __future__ import annotations

import unicodedata

from normalize import tokens
from v4_features import content_name_tokens_v4

CACHE: dict[tuple[str, str], str] = {}


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
    return ""


def indic_token(text: str) -> bool:
    return any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF for ch in text)


def fold_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def romanized_name(name: str) -> str:
    parts: list[str] = []
    for tok in tokens(name):
        if indic_token(tok):
            lang = lang_of(tok)
            parts.append(CACHE.get((tok, lang), ""))
        else:
            parts.append(tok)
    return " ".join(part for part in parts if part)


def name_key(name: str, country: str) -> str:
    phrase = " ".join(content_name_tokens_v4(fold_text(romanized_name(name))))
    if len(phrase) < 8:
        return ""
    return country + "|" + phrase
