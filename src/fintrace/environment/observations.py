"""Observation builders: ok / error ToolResults with the step budget, the reset observation, info.

Errors carry a code and nothing else (CONTRIBUTING.md §6.4); the developer-facing message passed to
``build_error_result`` is dropped before anything reaches the agent.
"""

from __future__ import annotations

from typing import Literal

from fintrace.constants import ENV_VERSION, MAX_STEPS
from fintrace.models import (
    EpisodeState,
    Error,
    ErrorCode,
    FrozenModel,
    JsonObject,
    StepCount,
    ToolName,
    ToolResult,
)


class ResetObservation(FrozenModel):
    """Observation returned by reset(): episode header, policy, tool names and the step budget."""

    ok: Literal[True] = True
    tool: Literal["reset"] = "reset"
    result: JsonObject
    steps_used: StepCount = 0
    steps_remaining: StepCount = MAX_STEPS


def step_info() -> JsonObject:
    """The only ``info`` the environment ever emits: exactly max_steps and version, never a seed."""
    return {"max_steps": MAX_STEPS, "version": ENV_VERSION}


def _remaining(step_count: int) -> int:
    return max(0, MAX_STEPS - step_count)


def build_ok_result(tool: ToolName, result: JsonObject, state: EpisodeState) -> ToolResult:
    """Successful tool observation stamped with the current step budget."""
    return ToolResult(
        ok=True,
        tool=tool,
        result=result,
        error=None,
        steps_used=state.step_count,
        steps_remaining=_remaining(state.step_count),
    )


def build_error_result(
    tool: ToolName,
    code: ErrorCode,
    _message_ignored: str,
    state: EpisodeState,
    steps_delta: int = 0,
) -> ToolResult:
    """Error observation: code only. ``_message_ignored`` documents intent for developers and is
    never written anywhere the agent can read."""
    used = min(MAX_STEPS, max(0, state.step_count + steps_delta))
    return ToolResult(
        ok=False,
        tool=tool,
        result=None,
        error=Error(code=code),
        steps_used=used,
        steps_remaining=_remaining(used),
    )


__all__ = ["ResetObservation", "build_error_result", "build_ok_result", "step_info"]
