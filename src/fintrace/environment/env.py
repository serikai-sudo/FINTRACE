"""FintraceEnv: reset/step loop over a PublicEpisode with the fixed five-tool API and a step budget.

Contract (blueprint 6.1-6.5): ``reset(seed, config) -> (observation, info)``;
``step(action) -> (observation, reward, terminated, truncated, info)``; at most MAX_STEPS steps
including submit; intermediate reward is always 0.0; submit terminates, the budget truncates;
``info`` is always exactly ``{"max_steps", "version"}``. Tools read only the PublicEpisode; the
PrivateEpisode is held for the evaluator-side matcher (step 6) and never reaches an observation.
"""

from __future__ import annotations

from typing import ClassVar, Final

from fintrace.constants import (
    ENV_VERSION,
    MAX_PAGE_SIZE,
    MAX_STEPS,
    REWARD_INTERMEDIATE,
    TOOL_NAMES,
)
from fintrace.generator.build import build_clean_episode
from fintrace.models import (
    EpisodeConfig,
    EpisodeState,
    JsonObject,
    PrivateEpisode,
    PublicEpisode,
    StepResult,
    ToolName,
    ToolResult,
)
from fintrace.tools.list_records import clear_cache
from fintrace.tools.registry import dispatch, parse_action
from fintrace.tools.submit import SUBMIT_PLACEHOLDER_REWARD

from .observations import ResetObservation, build_error_result, step_info
from .state import advance, finish, is_over, new_state

# TODO(step-6): reward for a truncated episode = official reward of an empty submission.
TRUNCATION_PLACEHOLDER_REWARD: Final[float] = 0.0
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


class FintraceEnv:
    """Single-episode environment; call ``reset`` before ``step``. Not a gym.Env on purpose."""

    VERSION: ClassVar[str] = ENV_VERSION
    MAX_STEPS: ClassVar[int] = MAX_STEPS
    MAX_PAGE_SIZE: ClassVar[int] = MAX_PAGE_SIZE

    def __init__(self) -> None:
        self._episode: PublicEpisode | None = None
        self._private: PrivateEpisode | None = None  # evaluator-side; never read by tools
        self._state: EpisodeState | None = None

    @property
    def episode(self) -> PublicEpisode:
        """The agent-visible episode (raises if reset was never called)."""
        return self._require_reset()[0]

    @property
    def state(self) -> EpisodeState:
        """Step counters and end flags (raises if reset was never called)."""
        return self._require_reset()[1]

    def reset(
        self, seed: int, config: EpisodeConfig | None = None
    ) -> tuple[ResetObservation, JsonObject]:
        """Build a fresh episode and return the initial observation and ``info``.

        The observation carries the public manifest and policy, the tool names and the step
        budget; ``info`` never contains the seed; nothing from truth/provenance is included.
        """
        if self._episode is not None:
            clear_cache(self._episode.manifest.episode_id)
        private = build_clean_episode(seed, config)
        public = private.public
        self._private = private
        self._episode = public
        self._state = new_state(public.manifest.episode_id)
        observation = ResetObservation(
            result={
                "manifest": public.manifest.model_dump(mode="json"),
                "policy": public.policy.model_dump(mode="json"),
                "tool_schema": list(TOOL_NAMES),
                "budget": {"steps_remaining": MAX_STEPS},
            }
        )
        return observation, step_info()

    def step(self, action: object) -> tuple[ToolResult, float, bool, bool, JsonObject]:
        """Apply one action. Every call after the episode ended is an EPISODE_OVER no-op."""
        episode, state = self._require_reset()
        if is_over(state):
            observation = build_error_result(
                _tool_of(action), "EPISODE_OVER", "episode already ended", state
            )
            return self._pack(observation, REWARD_INTERMEDIATE, state)
        advance(state)  # invalid actions consume budget too
        parsed = parse_action(action)
        if isinstance(parsed, str):
            observation = build_error_result(_tool_of(action), parsed, "rejected action", state)
            self._truncate_if_exhausted(state)
            return self._pack(observation, REWARD_INTERMEDIATE, state)
        tool, args = parsed
        observation = dispatch(tool, args, state, episode)
        reward = REWARD_INTERMEDIATE
        if tool == "submit" and observation.ok:
            finish(state, submitted=True, truncated=False)
            reward = SUBMIT_PLACEHOLDER_REWARD
        elif self._truncate_if_exhausted(state):
            reward = TRUNCATION_PLACEHOLDER_REWARD
        return self._pack(observation, reward, state)

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


__all__ = ["TRUNCATION_PLACEHOLDER_REWARD", "FintraceEnv"]
