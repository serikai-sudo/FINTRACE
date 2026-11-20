"""Attacker 12: a single best guess, nothing else."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase


class UndersubmitBoundary(AttackerBase):
    """Exactly one candidate finding: the precision-perfect, recall-starved boundary."""

    name = "undersubmit"
    strategy = "exactly one candidate finding"
    defence = "F1 punishes recall: one true positive out of eight issues scores about -0.56"

    def _attack(self, observation: Observation) -> JsonObject:
        pairs = self._invoice_pairs()
        best = self._pair_finding(pairs[0]) if pairs else self._random_pair_finding()
        return self._submit_with([best] if best else [], "one finding only")


__all__ = ["UndersubmitBoundary"]
