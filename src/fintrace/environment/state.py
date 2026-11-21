"""Helpers around the mutable ``EpisodeState`` (the model itself lives in fintrace.models)."""

from __future__ import annotations

from fintrace.constants import MAX_STEPS
from fintrace.models import EpisodeState


def new_state(episode_id: str) -> EpisodeState:
    """Fresh counters for one episode."""
    return EpisodeState(episode_id=episode_id)


def is_over(state: EpisodeState) -> bool:
    """True once the episode ended by submit (terminated) or by the step budget (truncated)."""
    return state.terminated or state.truncated


def advance(state: EpisodeState) -> None:
    """Consume one step of the budget; the caller must have checked ``is_over`` first."""
    if state.step_count >= MAX_STEPS:
        raise RuntimeError("step budget exhausted: an ended episode must not advance")
    state.step_count += 1


def finish(
    state: EpisodeState, *, submitted: bool, truncated: bool, terminated: bool | None = None
) -> None:
    """Close the episode: a successful submit terminates it, the step budget truncates it.

    ``terminated`` defaults to ``submitted``. Pass it explicitly for a submission the environment
    refused: that ends the episode (jury Block 3, tests 18-20) without recording a submission.
    """
    state.submitted = submitted
    state.terminated = submitted if terminated is None else terminated
    state.truncated = truncated


__all__ = ["advance", "finish", "is_over", "new_state"]
