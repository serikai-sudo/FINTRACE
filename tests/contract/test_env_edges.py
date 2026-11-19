"""Edge cases of the step/reset contract that the main env tests do not pin."""

from __future__ import annotations

import pytest

from fintrace.constants import ERROR_CODES
from fintrace.environment.env import FintraceEnv
from fintrace.models import JsonObject, ToolResult

FOREIGN_EPISODE_ID = "ep_0123456789ab"  # well-formed, never this episode's content hash
Step = tuple[ToolResult, float, bool, bool, JsonObject]


def _env(seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed)
    return env


def _submit(env: FintraceEnv, episode_id: str) -> Step:
    args: JsonObject = {"episode_id": episode_id, "findings": []}
    return env.step({"tool": "submit", "args": args})


def _list_one(env: FintraceEnv, limit: int = 1) -> Step:
    args: JsonObject = {"table": "ledger", "cursor": None, "limit": limit}
    return env.step({"tool": "list_records", "args": args})


def test_submit_with_wrong_episode_id_rejected() -> None:
    env = _env()
    observation, reward, terminated, truncated, _ = _submit(env, FOREIGN_EPISODE_ID)
    assert observation.ok is False and observation.error is not None
    assert observation.error.code == "INVALID_SUBMISSION"
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert env.state.step_count == 1  # the failed submit consumed a step, the episode goes on
    follow_up, *_ = _list_one(env)
    assert follow_up.ok and follow_up.steps_used == 2


def test_action_after_terminated_returns_episode_over() -> None:
    env = _env()
    _, _, terminated, _, _ = _submit(env, env.episode.manifest.episode_id)
    assert terminated
    observation, reward, terminated, truncated, info = _list_one(env)
    assert observation.error is not None and observation.error.code == "EPISODE_OVER"
    assert reward == 0.0 and terminated and not truncated
    assert observation.steps_used == 1  # nothing is consumed after the end
    assert set(info) == {"max_steps", "version"}


def test_resubmit_after_submit_returns_episode_over() -> None:
    env = _env()
    episode_id = env.episode.manifest.episode_id
    first, _, terminated, _, _ = _submit(env, episode_id)
    assert first.ok and terminated
    second, reward_again, still_terminated, _, _ = _submit(env, episode_id)
    assert second.error is not None and second.error.code == "EPISODE_OVER"
    assert reward_again == 0.0 and still_terminated
    assert second.steps_used == first.steps_used == 1


def test_calculator_empty_expression_rejected() -> None:
    observation, *_ = _env(1).step({"tool": "calculator", "args": {"expression": ""}})
    assert observation.error is not None and observation.error.code == "CALC_ERROR"


def test_calculator_only_whitespace_rejected() -> None:
    observation, *_ = _env(1).step({"tool": "calculator", "args": {"expression": "   "}})
    assert observation.error is not None and observation.error.code == "CALC_ERROR"


@pytest.mark.parametrize("limit", [0, -1, 51, 1_000_000])
def test_limit_out_of_range_rejected(limit: int) -> None:
    observation, reward, terminated, _, _ = _list_one(_env(1), limit)
    assert observation.error is not None and observation.error.code == "INVALID_ACTION"
    assert reward == 0.0 and not terminated


@pytest.mark.parametrize("action", [None, [], 42, "string_action", True, 3.14, b"bytes"])
def test_non_dict_actions_are_invalid(action: object) -> None:
    env = _env(1)
    observation, reward, terminated, truncated, _ = env.step(action)
    assert observation.ok is False and observation.error is not None
    assert observation.error.code == "INVALID_ACTION" and observation.error.code in ERROR_CODES
    assert observation.tool == "list_records"  # the neutral tool stamps undispatchable actions
    assert env.state.step_count == 1 and (reward, terminated, truncated) == (0.0, False, False)
