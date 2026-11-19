"""Precision / recall / F1 from (tp, fp, fn) with the blueprint 7.3 edge-case contract."""

from __future__ import annotations


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` as floats.

    Edge cases: nothing to find and nothing claimed (0, 0, 0) is a perfect audit -> (1, 1, 1);
    any zero denominator otherwise yields 0.0 (never NaN). F1 = 2tp / (2tp + fp + fn).
    """
    if min(tp, fp, fn) < 0:
        raise ValueError("tp, fp and fn must be non-negative")
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return precision, recall, f1


__all__ = ["compute_metrics"]
