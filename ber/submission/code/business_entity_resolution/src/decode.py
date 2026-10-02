"""Greedy set decoder that maximizes per-entity F0.5."""

from __future__ import annotations

from f05 import entity_f05


def decode_greedy_f05(
    candidate_ids: list[str],
    scores: list[float],
    *,
    min_gain: float = 0.0,
    max_preds: int = 12,
) -> set[str]:
    """Score threshold decode (min_gain tuned on validation macro F0.5)."""
    if not candidate_ids:
        return set()
    pairs = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    out: list[str] = []
    for eid, sc in pairs:
        if sc < min_gain:
            break
        out.append(eid)
        if len(out) >= max_preds:
            break
    return set(out)


def decode_greedy_f05_labeled(
    candidate_ids: list[str],
    scores: list[float],
    true_ids: set[str],
) -> set[str]:
    """Validation-only: pick set maximizing entity F0.5 vs known truth."""
    if not candidate_ids:
        return set()
    order = sorted(range(len(candidate_ids)), key=lambda i: scores[i], reverse=True)
    chosen: set[str] = set()
    best_f = entity_f05(true_ids, chosen)
    for i in order:
        trial = chosen | {candidate_ids[i]}
        f = entity_f05(true_ids, trial)
        if f >= best_f:
            best_f = f
            chosen = trial
    return chosen


def tune_min_gain(
    val_rows: list[tuple[list[str], list[float], set[str]]],
    grid: list[float] | None = None,
) -> tuple[float, float]:
    """Return (best_min_gain, macro_f05) on validation rows."""
    from f05 import macro_f05

    if grid is None:
        grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    best_g, best_m = 0.0, -1.0
    for g in grid:
        preds = [decode_greedy_f05(cids, scs, min_gain=g) for cids, scs, _ in val_rows]
        truths = [t for _, _, t in val_rows]
        m = macro_f05(truths, preds)
        if m > best_m:
            best_m, best_g = m, g
    return best_g, best_m
