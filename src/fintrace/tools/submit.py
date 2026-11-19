"""submit: the terminal tool. Validates the episode id and accepts the submission.

Scoring happens in the environment, not here: the env hands an accepted ``Submission`` to the
evaluator and the official reward (dependency injection), so this module never imports
``fintrace.evaluator``. The agent sees only ``{status, findings_count}``.
"""

from __future__ import annotations

from fintrace.environment.observations import build_error_result, build_ok_result
from fintrace.models import EpisodeState, JsonObject, PublicEpisode, Submission, ToolResult


def handle_submit(args: Submission, state: EpisodeState, episode: PublicEpisode) -> ToolResult:
    """Accept a submission for this episode; a foreign episode_id is INVALID_SUBMISSION."""
    if args.episode_id != episode.manifest.episode_id:
        return build_error_result("submit", "INVALID_SUBMISSION", "episode_id mismatch", state)
    result: JsonObject = {"status": "accepted", "findings_count": len(args.findings)}
    return build_ok_result("submit", result, state)


__all__ = ["handle_submit"]
