"""Entity-level F0.5 (beta=0.5) for set predictions."""

from __future__ import annotations

BETA = 0.5
B2 = BETA * BETA


def entity_f05(true_ids: set[str], pred_ids: set[str]) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0
    inter = len(true_ids & pred_ids)
    if inter == 0:
        return 0.0
    p = inter / len(pred_ids)
    r = inter / len(true_ids)
    return (1 + B2) * p * r / (B2 * p + r)


def macro_f05(truths: list[set[str]], preds: list[set[str]]) -> float:
    if len(truths) != len(preds):
        raise ValueError("length mismatch")
    if not truths:
        return 0.0
    return sum(entity_f05(t, p) for t, p in zip(truths, preds)) / len(truths)
