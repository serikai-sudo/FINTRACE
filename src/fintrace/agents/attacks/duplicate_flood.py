"""Attacker 8: the single best guess, repeated thirty times."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


class DuplicateFlood(AttackerBase):
    """One candidate finding copied to the cap: at most one true positive, the rest false."""

    name = "duplicate_flood"
    strategy = "the best candidate finding repeated 30 times"
    defence = "one-to-one matching: copies of a matched finding are false positives"

    def _attack(self, observation: Observation) -> JsonObject:
        pairs = self._invoice_pairs()
        best = self._pair_finding(pairs[0]) if pairs else self._random_pair_finding()
        if best is None:
            return self._submit_empty("nothing to flood with")
        return self._submit_with([dict(best) for _ in range(MAX_FINDINGS)], "flooding one guess")


__all__ = ["DuplicateFlood"]
