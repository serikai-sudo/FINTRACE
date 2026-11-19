"""The 12 reward scenarios of blueprint 7.5 against the official formula."""

from __future__ import annotations

import pytest

from fintrace.constants import MAX_STEPS
from fintrace.evaluator.matcher import match
from fintrace.evaluator.metrics import compute_metrics
from fintrace.models import Issue
from fintrace.reward.official import official_reward


def test_perfect_findings_with_20_steps() -> None:
    assert official_reward(1.0, 20) == pytest.approx(0.96, abs=1e-12)


def test_silence_on_erroneous_books() -> None:
    assert official_reward(0.0, 30) == -1.0  # -1.06 clipped


def test_silence_on_clean_books() -> None:
    _, _, f1 = compute_metrics(0, 0, 0)  # empty truth, empty submission
    assert official_reward(f1, 30) == pytest.approx(0.94, abs=1e-12)


def test_false_accusation_on_clean_books() -> None:
    _, _, f1 = compute_metrics(0, 1, 0)
    assert official_reward(f1, 10) == -1.0  # -1.02 clipped


def test_zero_steps_perfect() -> None:
    assert official_reward(1.0, 0) == 1.0


def test_zero_steps_zero_f1() -> None:
    assert official_reward(0.0, 0) == -1.0


def test_f1_half_no_cost() -> None:
    assert official_reward(0.5, 0) == 0.0


def test_clipping_upper() -> None:
    assert official_reward(1.0, 0) == 1.0
    assert official_reward(1.5, 0) == 1.0  # out-of-range f1 cannot exceed the ceiling


def test_clipping_lower() -> None:
    assert official_reward(0.0, 500) == -1.0
    assert official_reward(1.0, 10_000) == -1.0


def test_nan_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        official_reward(float("nan"), 10)
    with pytest.raises(ValueError, match="finite"):
        official_reward(float("inf"), 10)


def test_negative_steps_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        official_reward(0.5, -1)


def test_truncation_empty_findings_full_budget() -> None:
    issues = tuple(
        Issue(
            issue_id=f"ISS-0{k}",
            type="fx_error",
            affected_ids=(f"L-00000{k}",),
            expected_amount_minor=1000 * k,
            currency="USD",
            acceptable_evidence_sets=((f"L-00000{k}",),),
        )
        for k in range(1, 6)
    )
    tp, fp, fn = match([], issues)
    assert (tp, fp, fn) == (0, 0, 5)
    _, _, f1 = compute_metrics(tp, fp, fn)
    assert f1 == 0.0
    assert official_reward(f1, MAX_STEPS) == -1.0  # -1.2 clipped
