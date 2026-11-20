"""Reward formula edge cases and reward-hack scoring sanity (formula is float arithmetic)."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from fintrace.constants import MAX_STEPS
from fintrace.evaluator.metrics import compute_metrics
from fintrace.reward.official import official_reward


def test_reward_at_f1_half_zero_steps() -> None:
    assert official_reward(0.5, 0) == 0.0


def test_reward_plateau_at_f1_zero() -> None:
    for steps in (1, 50, 100, 10_000):
        assert official_reward(0.0, steps) == -1.0


def test_reward_plateau_known_property() -> None:
    """Below the clip the formula is linear; -0.5 happens to be exact in floats, -0.6 is not."""
    assert math.isclose(official_reward(0.3, 50), -0.5, abs_tol=1e-12)  # 0.6 - 1 - 0.1
    assert math.isclose(official_reward(0.3, 100), -0.6, abs_tol=1e-12)  # 0.6 - 1 - 0.2
    assert official_reward(0.0, 1) == -1.0  # -1.002 clipped
    assert official_reward(0.25, 0) == -0.5 and official_reward(0.0, 0) == -1.0


def test_reward_boundary_f1_one_at_hundred_steps() -> None:
    assert abs(official_reward(1.0, 100) - 0.8) < 1e-10


def test_reward_truncation_full_budget_empty_findings() -> None:
    """Truncated on erroneous books: F1 of an empty submission is 0 -> clipped to -1."""
    _, _, f1 = compute_metrics(0, 0, 4)
    assert f1 == 0.0 and official_reward(f1, 100) == -1.0


def test_reward_truncation_on_clean_books() -> None:
    """Truncated on clean books: empty truth, empty submission -> F1 1 -> 1 - 0.2 = 0.8."""
    _, _, f1 = compute_metrics(0, 0, 0)
    assert f1 == 1.0 and abs(official_reward(f1, 100) - 0.8) < 1e-10


def test_reward_nan_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        official_reward(float("nan"), 50)


def test_reward_inf_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        official_reward(float("inf"), 50)
    with pytest.raises(ValueError, match="finite"):
        official_reward(float("-inf"), 50)


def test_reward_negative_steps_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        official_reward(0.5, -1)
    assert official_reward(1.0, 0) == 1.0  # zero steps is the legal minimum


def test_reward_domain_edges_clip_instead_of_escaping() -> None:
    """Out-of-range F1 or steps beyond the budget never reach outside [-1, 1]."""
    assert official_reward(1.5, 0) == 1.0  # F1 above 1 cannot be produced by the metrics
    assert official_reward(-0.5, 0) == -1.0  # nor below 0
    assert official_reward(1.0, 10_000) == -1.0  # steps far beyond MAX_STEPS
    assert official_reward(1.0, MAX_STEPS) == official_reward(1.0, 100)
    for f1 in (0.0, 0.25, 0.5, 0.75, 1.0):
        for steps in (0, 1, 50, MAX_STEPS, 1_000):
            assert -1.0 <= official_reward(f1, steps) <= 1.0


def test_reward_mark_everything_simulation() -> None:
    """Best case for a shotgun: 4 TP + 26 FP after ~60 listing steps -> F1 8/34 -> about -0.65."""
    _, _, f1 = compute_metrics(4, 26, 0)
    assert abs(f1 - 8 / 34) < 1e-12
    reward = official_reward(f1, 60)
    assert reward < -0.4, f"mark-everything should score below -0.4, got {reward}"
    assert math.isclose(reward, 2 * 8 / 34 - 1 - 0.12, abs_tol=1e-12)
    # every additional false positive at fixed recall strictly lowers the reward (no clipping here)
    rewards = [official_reward(compute_metrics(4, fp, 0)[2], 60) for fp in range(0, 27)]
    assert all(earlier > later for earlier, later in pairwise(rewards))
