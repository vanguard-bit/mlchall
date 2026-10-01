"""Parsed house number and street, matching the measured v3 features."""

from __future__ import annotations

from normalize import ADDR_STOP, LEGAL, digit_tokens, tokens


def parse_addr(addr: str) -> tuple[str, frozenset[str]]:
    """First house-like number, then the street tokens that follow it."""
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
    return house, frozenset(streets[:4])


def house_triple(left: tuple[str, frozenset[str]], right: tuple[str, frozenset[str]]) -> list[float]:
    house_l, streets_l = left
    house_r, streets_r = right
    if not house_l or not house_r:
        return [0.0, 1.0, 0.0]
    same = house_l == house_r
    conflict = (not same) and bool(streets_l & streets_r)
    return [float(same), 0.0, float(conflict)]
