# TODO(step-6): integrate evaluator.matcher + reward.official.
# Current behavior: accepts submission, returns 0.0 reward, terminates episode.
"""submit: the terminal tool. Step-5 placeholder: validates the episode id and accepts.

The matcher and the official reward are injected by the environment in step 6 (dependency
injection; this module never imports fintrace.evaluator). Until then every accepted submission
earns ``SUBMIT_PLACEHOLDER_REWARD`` and the agent sees only ``{status, findings_count}``.
"""

from __future__ import annotations

from typing import Final

from fintrace.environment.observations import build_error_result, build_ok_result
from fintrace.models import EpisodeState, JsonObject, PublicEpisode, Submission, ToolResult

SUBMIT_PLACEHOLDER_REWARD: Final[float] = 0.0  # matcher pending (step 6)


def handle_submit(args: Submission, state: EpisodeState, episode: PublicEpisode) -> ToolResult:
    """Accept a submission for this episode; a foreign episode_id is INVALID_SUBMISSION."""
    if args.episode_id != episode.manifest.episode_id:
        return build_error_result("submit", "INVALID_SUBMISSION", "episode_id mismatch", state)
    result: JsonObject = {"status": "accepted", "findings_count": len(args.findings)}
    return build_ok_result("submit", result, state)


__all__ = ["SUBMIT_PLACEHOLDER_REWARD", "handle_submit"]
