"""OFFICIAL REWARD FORMULA — NEVER MODIFY.

    reward = clip(2 * F1 - 1 - 0.002 * inspection_steps, -1, 1)

``inspection_steps`` = every step before the terminal one (invalid actions and failed submits
included); for a truncated episode it is MAX_STEPS and F1 is that of an empty submission.
This module depends on constants only: no matcher, no metrics, no experimental rewards.
"""

from __future__ import annotations

import math

from fintrace.constants import (
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    REWARD_F1_BIAS,
    REWARD_F1_COEF,
    REWARD_STEP_COST,
)


def official_reward(f1: float, inspection_steps: int) -> float:
    """clip(2 * f1 - 1 - 0.002 * inspection_steps, -1, 1); NaN/inf f1 or negative steps raise."""
    if math.isnan(f1) or math.isinf(f1):
        raise ValueError("f1 must be finite")
    if inspection_steps < 0:
        raise ValueError("inspection_steps must be non-negative")
    raw = (
        float(REWARD_F1_COEF) * f1
        - float(REWARD_F1_BIAS)
        - float(REWARD_STEP_COST) * inspection_steps
    )
    return max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, raw))


__all__ = ["official_reward"]
