"""TruthStore: evaluator-side registry of Truth by episode_id (single process, not thread-safe)."""

from __future__ import annotations

from fintrace.models import Truth


class TruthStore:
    """Holds ground truth per episode; the environment registers at reset and clears the old one.

    This is the only path from a PrivateEpisode to the scorer, so a future out-of-process
    evaluator replaces this class without touching the environment loop.
    """

    def __init__(self) -> None:
        self._store: dict[str, Truth] = {}

    def register(self, truth: Truth) -> None:
        """Associate ``truth.episode_id`` with ``truth``; an existing entry is overwritten."""
        self._store[truth.episode_id] = truth

    def get(self, episode_id: str) -> Truth | None:
        return self._store.get(episode_id)

    def clear(self, episode_id: str) -> None:
        self._store.pop(episode_id, None)

    def __len__(self) -> int:
        return len(self._store)


__all__ = ["TruthStore"]
