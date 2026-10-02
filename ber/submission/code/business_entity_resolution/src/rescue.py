"""High-precision rescue for typo and domain names the main index drops."""

from __future__ import annotations

from normalize import address_tokens, content_name_tokens, digit_tokens
from rapidfuzz.distance import JaroWinkler

RESCUE_CAP = 400


_DROP = {"and", "the", "of", "for"}


def _squashed(name: str) -> str:
    parts: list[str] = []
    for token in content_name_tokens(name):
        if not token.isascii() or token in _DROP:
            continue
        if token.endswith("com") and len(token) > 6:
            token = token[:-3]
        parts.append(token)
    text = "".join(parts)
    if text.endswith("com") and len(text) > 8:
        text = text[:-3]
    return text


def _houses(addr: str) -> set[str]:
    return {d for d in digit_tokens(addr) if 1 <= len(d) <= 6}


def _body_streets(addr: str) -> set[str]:
    tokens = [token for token in address_tokens(addr) if len(token) >= 5]
    if len(tokens) >= 2:
        tokens = tokens[:-1]
    return set(tokens)


def rescue_accept(s1_name: str, s1_addr: str, m_name: str, m_addr: str) -> bool:
    """Almost the same name, and the house number or the street agrees.

    Near-names such as Infrastructure and Infratech are rejected. When both
    sides have house numbers and they differ, a match still requires the
    squashed names to be equal and a street token to agree, which is the
    domain-name case with a truncated number.
    """
    left, right = _squashed(s1_name), _squashed(m_name)
    if len(left) < 8 or len(right) < 8:
        return False
    similarity = JaroWinkler.normalized_similarity(left, right)
    if similarity < 0.95:
        return False
    houses_left, houses_right = _houses(s1_addr), _houses(m_addr)
    if houses_left and houses_right:
        if houses_left & houses_right:
            return True
        return similarity >= 0.99 and bool(_body_streets(s1_addr) & _body_streets(m_addr))
    if not s1_addr.strip() or not m_addr.strip():
        return similarity >= 0.99
    return similarity >= 0.97 and bool(_body_streets(s1_addr) & _body_streets(m_addr))
