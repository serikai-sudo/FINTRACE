"""FINTRACE jury adapter: the connection protocol the Genesis jury harness calls.

Three entry points, a thin translation layer over the existing environment and evaluator:

    generate_episode(seed, config, output_dir)       build one episode into the jury's folder
    make_env()                                       an object with reset(...) and step(...)
    score_episode(episode_dir, findings, steps)      the closed scoring call (never a policy action)

The environment object speaks the jury envelope: actions are ``{"name": ..., "args": {...}}``
(``list_records``, ``get_record``, ``get_document``, ``calculator``, ``submit``), the reset
observation is ``{"task", "manifest", "policy", "remaining_steps"}`` and every step observation
is ``{"result", "error", "remaining_steps"}`` with ``error = {"code", "message"}`` or ``null``.
In closed mode (``config["mode"] == "closed"``) the episode is loaded exactly as written from
``episode_dir`` and the reward handed to the policy is 0 on every step; the true return is only
available through ``score_episode``. Nothing here re-implements generation, matching or reward.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from fintrace.constants import MAX_STEPS
from fintrace.environment.env import FintraceEnv
from fintrace.environment.observations import ResetObservation
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.exporters.episode_files import read_episode, write_episode
from fintrace.generator.build import build_episode
from fintrace.models import EpisodeConfig, JsonObject, PrivateEpisode, Submission, ToolResult
from fintrace.reward.official import official_reward

TASK_DESCRIPTION: Final[str] = (
    "Audit the ledger against its source documents (invoices, payments, FX rates) using the five "
    "tools, then submit findings with evidence for duplicate_invoice, payment_allocation, fx_error "
    "and period_misclassification issues."
)
# generic, static texts: no paths, no seeds, no echoed input, nothing about the hidden episode
ERROR_MESSAGES: Final[dict[str, str]] = {
    "INVALID_ACTION": "the action was malformed or is not one of the supported actions",
    "INVALID_TABLE": "unknown table; use ledger, invoices, payments or fx",
    "NOT_FOUND": "no record or document with that id",
    "INVALID_CURSOR": "the cursor was not issued for this table",
    "CALC_ERROR": "the expression could not be evaluated",
    "EPISODE_OVER": "the episode has ended; no further actions are accepted",
    "INVALID_SUBMISSION": "the submission was not accepted for this episode",
}
MODE_CLOSED: Final[str] = "closed"
MODE_TRAIN: Final[str] = "train"
SCORE_KEYS: Final[tuple[str, ...]] = ("tp", "fp", "fn", "precision", "recall", "f1")


# A directory the harness hands us can fail to read in many ways: a missing file or a permission
# problem (OSError), broken JSON, a value the models reject (both ValueError), a CSV whose header
# or rows do not line up (KeyError, TypeError), a field past the csv limit (csv.Error) or JSON
# nested past the interpreter's limit (RecursionError).
_LOAD_ERRORS: Final[tuple[type[Exception], ...]] = (
    OSError, ValueError, KeyError, TypeError, csv.Error, RecursionError,
)  # fmt: skip


def _load_episode(episode_dir: str) -> PrivateEpisode:
    """Read an episode directory, or fail with one documented, path-free error.

    The jury owns these paths, so a directory that cannot be read is never scored: both entry
    points that take an ``episode_dir`` raise the same ``ValueError`` with a fixed text, which
    carries no path, no seed and no parser detail.
    """
    try:
        return read_episode(Path(episode_dir))
    except _LOAD_ERRORS:
        raise ValueError("episode could not be loaded") from None


def _team_config(config: object) -> EpisodeConfig:
    """``config["team_config"]`` (or a bare config mapping) as an ``EpisodeConfig``."""
    raw = config if isinstance(config, dict) else {}
    team = raw.get("team_config", raw) if "team_config" in raw or "mode" not in raw else {}
    return EpisodeConfig.model_validate(team if isinstance(team, dict) else {})


def _findings_list(value: object) -> object:
    """The findings list inside a findings.json payload (``{"findings": [...]}``) or a bare list."""
    if isinstance(value, dict) and "findings" in value:
        return value["findings"]
    return value


def generate_episode(seed: int, config: dict[str, Any], output_dir: str) -> None:
    """Build the episode for ``seed`` under the team config and write it to ``output_dir``."""
    private = build_episode(int(seed), _team_config(config))
    write_episode(private, Path(output_dir), int(seed))


def score_episode(
    episode_dir: str, findings: dict[str, Any], inspection_steps: int
) -> dict[str, Any]:
    """Closed scoring call: match ``findings`` against the hidden truth of ``episode_dir``.

    An episode that cannot be read raises ``ValueError("episode could not be loaded")``, the same
    documented failure ``reset`` gives for the same directory: no path, no parser detail. It is
    never scored as -1, because the paths are the harness's own and ``return`` is what ranks a
    team; a submission the schema refuses is the case that returns -1 with null metrics.
    """
    private = _load_episode(episode_dir)
    try:
        submission = Submission.model_validate(
            {"episode_id": private.truth.episode_id, "findings": _findings_list(findings)}
        )
        steps = int(inspection_steps)
        if steps < 0:
            raise ValueError("inspection_steps must be non-negative")
    except (ValidationError, TypeError, ValueError):
        return {**dict.fromkeys(SCORE_KEYS), "return": -1.0}
    tp, fp, fn = match(
        submission.findings, private.truth.issues, public_ids=public_ids_of(private.public)
    )
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "return": official_reward(f1, steps),
    }


def _project_result(tool: str, result: JsonObject) -> object:
    """The jury's result shape per action (internal extras such as ``records_total`` dropped)."""
    if tool == "list_records":
        return {"records": result.get("records", []), "next_cursor": result.get("next_cursor")}
    if tool == "get_record":
        return result.get("record")
    if tool == "get_document":
        return result.get("document")
    if tool == "calculator":
        return result.get("value")
    if tool == "submit":
        return {"accepted": True}
    return result


def _to_jury_reset_obs(observation: ResetObservation) -> dict[str, Any]:
    result = observation.result
    return {
        "task": TASK_DESCRIPTION,
        "manifest": result.get("manifest"),
        "policy": result.get("policy"),
        "remaining_steps": MAX_STEPS,
    }


def _to_jury_step_obs(observation: ToolResult) -> dict[str, Any]:
    if observation.ok and observation.result is not None:
        return {
            "result": _project_result(observation.tool, observation.result),
            "error": None,
            "remaining_steps": observation.steps_remaining,
        }
    code = observation.error.code if observation.error is not None else "INVALID_ACTION"
    return {
        "result": None,
        "error": {"code": code, "message": ERROR_MESSAGES.get(code, "the action failed")},
        "remaining_steps": observation.steps_remaining,
    }


def _to_internal_action(action: object, episode_id: str) -> object:
    """Jury envelope ``{"name", "args"}`` -> FINTRACE ``{"tool", "args"}``; junk passes through."""
    if not isinstance(action, dict):
        return action
    name = action.get("name")
    args = action.get("args", {})
    if name == "submit" and isinstance(args, dict) and "findings" in args:
        args = {"episode_id": episode_id, "findings": _findings_list(args["findings"])}
    internal: dict[str, Any] = {"tool": name, "args": args}
    if isinstance(action.get("thought"), str):
        internal["thought"] = action["thought"]
    return internal


class _JudgeEnv:
    """The environment object returned by ``make_env``: jury envelope in, jury envelope out."""

    def __init__(self) -> None:
        self._env = FintraceEnv()
        self._mode = MODE_TRAIN

    def reset(
        self, seed: int, config: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``closed`` loads ``episode_dir`` as written; ``train`` generates from the seed."""
        settings = config or {}
        mode = str(settings.get("mode", MODE_TRAIN))
        if mode == MODE_CLOSED:
            episode_dir = settings.get("episode_dir")
            if not isinstance(episode_dir, str) or not episode_dir:
                raise ValueError("closed mode requires config['episode_dir']")
            observation = self._env.reset_loaded(_load_episode(episode_dir))
        elif mode == MODE_TRAIN:
            observation, _ = self._env.reset(seed=int(seed), config=_team_config(settings))
        else:
            raise ValueError("unknown mode; expected 'closed' or 'train'")
        self._mode = mode
        return _to_jury_reset_obs(observation), {}

    def step(self, action: object) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """One jury action; in closed mode the policy-facing reward is always 0."""
        internal = _to_internal_action(action, self._env.episode.manifest.episode_id)
        observation, reward, terminated, truncated, _ = self._env.step(internal)
        policy_reward = 0.0 if self._mode == MODE_CLOSED else float(reward)
        return _to_jury_step_obs(observation), policy_reward, terminated, truncated, {}


def make_env() -> _JudgeEnv:
    """The environment object with ``reset(seed, config)`` and ``step(action)``."""
    return _JudgeEnv()


__all__ = [
    "ERROR_MESSAGES",
    "MODE_CLOSED",
    "MODE_TRAIN",
    "SCORE_KEYS",
    "TASK_DESCRIPTION",
    "generate_episode",
    "make_env",
    "score_episode",
]
