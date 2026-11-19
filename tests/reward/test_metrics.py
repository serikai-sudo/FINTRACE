"""Precision / recall / F1 edge-case contract (blueprint 7.3)."""

from __future__ import annotations

import math

import pytest

from fintrace.evaluator.metrics import compute_metrics


def test_metrics_empty_empty_is_one() -> None:
    assert compute_metrics(0, 0, 0) == (1.0, 1.0, 1.0)


def test_metrics_false_positives_only() -> None:
    precision, recall, f1 = compute_metrics(0, 5, 0)
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)
    assert not any(math.isnan(v) for v in (precision, recall, f1))
    assert compute_metrics(0, 0, 7) == (0.0, 0.0, 0.0)  # silence on erroneous books


def test_metrics_mixed() -> None:
    precision, recall, f1 = compute_metrics(3, 1, 2)
    assert precision == 0.75 and recall == 0.6
    assert f1 == pytest.approx(2 / 3)
    assert all(isinstance(v, float) for v in (precision, recall, f1))
    with pytest.raises(ValueError, match="non-negative"):
        compute_metrics(1, -1, 0)
