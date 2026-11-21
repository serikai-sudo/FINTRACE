"""Evaluator-side state the agent never sees: the Truth registry and the rejection journal.

Both live on the evaluator side of the environment (single process, not thread-safe). A future
out-of-process evaluator replaces these classes without touching the environment loop.
"""

from __future__ import annotations

from fintrace.models import JournalEntry, Truth


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


class EvaluatorJournal:
    """Append-only record of submissions the environment refused (jury Block 3, tests 18-20).

    A structurally invalid submit ends the episode with reward -1 and "the error is written in the
    evaluator's journal": this is that journal. It holds only what the agent itself sent (the step,
    the error code and the pydantic error types of its own payload), never truth or provenance, and
    it is reachable from the environment object alone: no observation, ``info`` or trajectory line
    ever carries it.
    """

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    def record(self, entry: JournalEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> tuple[JournalEntry, ...]:
        """Everything recorded for the current episode, oldest first."""
        return tuple(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["EvaluatorJournal", "TruthStore"]
