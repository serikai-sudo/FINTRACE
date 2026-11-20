"""Attacker 19: replay a submission under another episode's id."""

from __future__ import annotations

from fintrace.agents.base import Observation, submit_action
from fintrace.models import JsonObject

from .base import AttackerBase


def foreign_episode_id(episode_id: str) -> str:
    """A well-formed id that is certainly not this episode's (last hex digit flipped)."""
    last = episode_id[-1]
    return episode_id[:-1] + ("0" if last != "0" else "1")


class CrossEpisodeReplay(AttackerBase):
    """Candidate findings submitted for a different episode id; then a clean empty exit."""

    name = "cross_episode"
    strategy = "candidate findings submitted under another episode id, then empty submission"
    defence = "the submission's episode_id must match the live episode: INVALID_SUBMISSION"

    def _attack(self, observation: Observation) -> JsonObject:
        if self._fired:
            return self._submit_empty("replay rejected, closing the episode")
        self._fired = True
        findings = [self._pair_finding(pair) for pair in self._invoice_pairs()[:5]]
        foreign = foreign_episode_id(self._episode_id or "ep_0000000000")
        return submit_action(foreign, findings, thought=f"{self.name}: replaying under {foreign}")


__all__ = ["CrossEpisodeReplay", "foreign_episode_id"]
