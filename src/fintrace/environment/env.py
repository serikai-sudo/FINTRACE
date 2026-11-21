"""FintraceEnv: reset/step loop over a PublicEpisode with the fixed five-tool API and a step budget.

Contract (blueprint 6.1-6.5, 7.4): ``reset(seed, config) -> (observation, info)``;
``step(action) -> (observation, reward, terminated, truncated, info)``; at most MAX_STEPS steps
including submit; intermediate reward is always 0.0; submit terminates with the official reward
of the matched submission, the budget truncates with the official reward of an empty submission
at MAX_STEPS inspection steps; ``info`` is always exactly ``{"max_steps", "version"}``.
Tools read only the PublicEpisode. The environment is the single place that touches the
evaluator: truth is registered in a TruthStore at reset and read only on the terminal step.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import ClassVar, Final

from pydantic import ValidationError

from fintrace.constants import (
    ENV_VERSION,
    MAX_PAGE_SIZE,
    MAX_STEPS,
    REWARD_CLIP_MIN,
    REWARD_INTERMEDIATE,
    TIME_BUDGET_SECONDS,
    TOOL_NAMES,
)
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.evaluator.truth_store import EvaluatorJournal, TruthStore
from fintrace.generator.build import build_episode
from fintrace.logging.trajectory import TrajectoryLogger
from fintrace.models import (
    Action,
    EpisodeConfig,
    EpisodeState,
    ErrorCode,
    Finding,
    JournalEntry,
    JsonObject,
    PrivateEpisode,
    PublicEpisode,
    StepResult,
    Submission,
    ToolName,
    ToolResult,
)
from fintrace.reward.official import official_reward
from fintrace.tools.list_records import clear_cache
from fintrace.tools.registry import dispatch, parse_action

from .observations import ResetObservation, build_error_result, step_info
from .state import advance, finish, is_over, new_state

_TOOL_BY_NAME: Final[dict[str, ToolName]] = {name: name for name in TOOL_NAMES}
_NEUTRAL_TOOL: Final[ToolName] = "list_records"


def _tool_of(action: object) -> ToolName:
    """Tool name to stamp on an observation for an action that could not be dispatched.

    The observation contract requires a real ToolName, so an unknown or missing name is
    reported under the neutral read-only tool ``list_records``.
    """
    if isinstance(action, dict):
        tool = action.get("tool")
        if isinstance(tool, str):
            return _TOOL_BY_NAME.get(tool, _NEUTRAL_TOOL)
    return _NEUTRAL_TOOL


def _is_refused_submission(action: object, code: ErrorCode) -> bool:
    """True when a well-formed ``submit`` envelope carries a payload the schema refused.

    Jury Block 3 (tests 18-20) and case section 05: a structurally invalid submission ends the
    episode with -1. Two neighbours stay outside that rule: a submission that only names another
    episode is a binding error (``INVALID_SUBMISSION``, see ``tools/submit.py``), and an envelope
    the action contract itself rejects (no ``args``, a non-object ``args``, an over-long
    ``thought``) is an ordinary malformed action (jury C13). Both cost a step and leave the
    episode open.
    """
    if code == "INVALID_SUBMISSION":
        return False
    try:
        envelope = Action.model_validate(action)
    except ValidationError:
        return False
    return envelope.tool == "submit"


def _submission_error_types(action: object) -> tuple[str, ...]:
    """Why the submission was refused, as pydantic error types of the agent's own payload.

    Evaluator-side detail only: it repeats the validation the parser already failed and never
    looks at the truth. ``("envelope",)`` means the action envelope itself was malformed.
    """
    args = action.get("args") if isinstance(action, dict) else None
    try:
        Submission.model_validate(args)
    except ValidationError as exc:
        return tuple(sorted({str(error["type"]) for error in exc.errors()}))
    return ("envelope",)


class FintraceEnv:
    """Single-episode environment; call ``reset`` before ``step``. Not a gym.Env on purpose."""

    VERSION: ClassVar[str] = ENV_VERSION
    MAX_STEPS: ClassVar[int] = MAX_STEPS
    MAX_PAGE_SIZE: ClassVar[int] = MAX_PAGE_SIZE

    def __init__(self) -> None:
        self._episode: PublicEpisode | None = None
        self._private: PrivateEpisode | None = None  # evaluator-side; never read by tools
        self._public_ids: frozenset[str] = frozenset()  # ids the matcher may accept as evidence
        self._state: EpisodeState | None = None
        self._truth_store = TruthStore()  # the only path from truth to the scorer
        self._journal = EvaluatorJournal()  # evaluator-side record of refused submissions
        self._logger: TrajectoryLogger | None = None  # optional per-episode JSONL transcript
        # wall-clock safety deadline (time.monotonic); never part of the state or any output
        self._deadline: float | None = None

    @property
    def episode(self) -> PublicEpisode:
        """The agent-visible episode (raises if reset was never called)."""
        return self._require_reset()[0]

    @property
    def state(self) -> EpisodeState:
        """Step counters and end flags (raises if reset was never called)."""
        return self._require_reset()[1]

    @property
    def evaluator_journal(self) -> tuple[JournalEntry, ...]:
        """Submissions this episode refused, for the harness; never part of any observation."""
        return self._journal.entries()

    def reset(
        self, seed: int, config: EpisodeConfig | None = None
    ) -> tuple[ResetObservation, JsonObject]:
        """Build a fresh episode and return the initial observation and ``info``.

        The observation carries the public manifest and policy, the tool names and the step
        budget; ``info`` never contains the seed; nothing from truth/provenance is included.
        """
        # build first: if generation fails, the previous episode and its truth stay intact;
        # injectors run per config.errors (all zero -> clean episode, backward compatible)
        private = build_episode(seed, config)
        return self._load(private), step_info()

    def reset_loaded(self, private: PrivateEpisode) -> ResetObservation:
        """Start an episode from a prebuilt ``PrivateEpisode`` (the jury's closed mode).

        Same setup as ``reset`` without generation: the episode keeps its id, the truth goes to
        the evaluator-side store only, the step and time budgets start afresh.
        """
        return self._load(private)

    def _load(self, private: PrivateEpisode) -> ResetObservation:
        self._detach_logger()  # a logger belongs to one episode: never record the next one
        if self._episode is not None:
            clear_cache(self._episode)  # the view of the object we leave, never by episode_id
        if self._private is not None:
            self._truth_store.clear(self._private.public.manifest.episode_id)
        public = private.public
        self._private = private
        self._episode = public
        self._public_ids = public_ids_of(public)
        self._state = new_state(public.manifest.episode_id)
        self._truth_store.register(private.truth)
        self._journal.clear()  # a journal belongs to one episode
        self._deadline = time.monotonic() + TIME_BUDGET_SECONDS
        return ResetObservation(
            result={
                "manifest": public.manifest.model_dump(mode="json"),
                "policy": public.policy.model_dump(mode="json"),
                "tool_schema": list(TOOL_NAMES),
                "budget": {"steps_remaining": MAX_STEPS},
            }
        )

    def step(self, action: object) -> tuple[ToolResult, float, bool, bool, JsonObject]:
        """Apply one action. Every call after the episode ended is an EPISODE_OVER no-op."""
        episode, state = self._require_reset()
        if is_over(state):
            observation = build_error_result(
                _tool_of(action), "EPISODE_OVER", "episode already ended", state
            )
            return self._emit(action, observation, REWARD_INTERMEDIATE, state)
        advance(state)  # invalid actions consume budget too
        if self._deadline is not None and time.monotonic() >= self._deadline:
            # wall-clock safety timeout (6.6 / jury C18): a hung run is truncated exactly like
            # step exhaustion, with the same fixed empty-submission reward, so a well-behaved
            # run never observes it and seed determinism holds
            finish(state, submitted=False, truncated=True)
            observation = build_error_result(_tool_of(action), "EPISODE_OVER", "time budget", state)
            return self._emit(action, observation, self._truncation_reward(), state)
        parsed = parse_action(action)
        if isinstance(parsed, str):
            observation = build_error_result(_tool_of(action), parsed, "rejected action", state)
            if _is_refused_submission(action, parsed):
                # a submission the schema refuses ends the episode at -1 whatever the books hold
                # (never the empty-submission reward, which is positive on clean books), and the
                # evaluator keeps the reason; the agent still sees a code-only error
                finish(state, submitted=False, truncated=False, terminated=True)
                self._journal.record(
                    JournalEntry(
                        event="invalid_submission",
                        step=state.step_count,
                        code=parsed,
                        reason=_submission_error_types(action),
                    )
                )
                return self._emit(action, observation, REWARD_CLIP_MIN, state)
            truncated = self._truncate_if_exhausted(state)
            reward = self._truncation_reward() if truncated else REWARD_INTERMEDIATE
            return self._emit(action, observation, reward, state)
        tool, args = parsed
        observation = dispatch(tool, args, state, episode)
        reward = REWARD_INTERMEDIATE
        if tool == "submit" and observation.ok and isinstance(args, Submission):
            finish(state, submitted=True, truncated=False)
            # every earlier step was inspection, invalid actions and failed submits included
            reward = self._terminal_reward(args.findings, inspection_steps=state.step_count - 1)
        elif self._truncate_if_exhausted(state):
            reward = self._truncation_reward()
        return self._emit(action, observation, reward, state)

    def attach_logger(self, logger: TrajectoryLogger) -> None:
        """Record every following step of the current episode; closed on the terminal step."""
        episode, _ = self._require_reset()
        logger.open(episode.manifest.episode_id)  # validate the new logger first
        self._detach_logger()
        self._logger = logger

    def _detach_logger(self) -> None:
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    def _emit(
        self, action: object, observation: ToolResult, reward: float, state: EpisodeState
    ) -> tuple[ToolResult, float, bool, bool, JsonObject]:
        """Pack the step and, with a logger attached, append it (and close on the terminal)."""
        packed = self._pack(observation, reward, state)
        if self._logger is not None:
            observation, reward, terminated, truncated, _ = packed
            try:
                self._logger.log_step(action, observation, reward, terminated, truncated)
            except Exception as exc:  # recording must never break a step
                self._logger.abort(exc)  # the step contract (6.4/6.6) outranks the transcript
                self._logger = None
            if terminated or truncated:
                self._detach_logger()
        return packed

    def _terminal_reward(self, findings: Sequence[Finding], inspection_steps: int) -> float:
        """Official reward of ``findings`` against this episode's truth (evaluator side)."""
        episode, _ = self._require_reset()
        truth = self._truth_store.get(episode.manifest.episode_id)
        if truth is None:
            raise RuntimeError("no truth registered for the current episode")
        tp, fp, fn = match(findings, truth.issues, public_ids=self._public_ids)
        _, _, f1 = compute_metrics(tp, fp, fn)
        return official_reward(f1, inspection_steps)

    def _truncation_reward(self) -> float:
        """Blueprint 7.4: a truncated episode scores as an empty submission after MAX_STEPS."""
        return self._terminal_reward([], inspection_steps=MAX_STEPS)

    def _require_reset(self) -> tuple[PublicEpisode, EpisodeState]:
        if self._episode is None or self._state is None:
            raise RuntimeError("call reset() before step()")
        return self._episode, self._state

    @staticmethod
    def _truncate_if_exhausted(state: EpisodeState) -> bool:
        if state.step_count >= MAX_STEPS and not state.terminated:
            finish(state, submitted=False, truncated=True)
            return True
        return False

    @staticmethod
    def _pack(
        observation: ToolResult, reward: float, state: EpisodeState
    ) -> tuple[ToolResult, float, bool, bool, JsonObject]:
        """Validate the step against the StepResult contract and unpack it into the tuple."""
        result = StepResult(
            observation=observation,
            reward=reward,
            terminated=state.terminated,
            truncated=state.truncated,
            info=step_info(),
        )
        return result.observation, result.reward, result.terminated, result.truncated, result.info


__all__ = ["FintraceEnv"]
