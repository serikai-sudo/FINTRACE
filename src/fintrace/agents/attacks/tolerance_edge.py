"""Attacker 3: claim amounts just beside the exact expected amount (the tolerance boundary)."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import AMOUNT_TOLERANCE_MINOR, MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


def tolerance_of(issue_type: str) -> int:
    """The published per-type tolerance in minor units (0 for every type: an exact match)."""
    return AMOUNT_TOLERANCE_MINOR.get(issue_type, 0)


class ToleranceEdgeProbe(AttackerBase):
    """Per candidate pair: amount + tol + 1, amount - tol - 1 and amount + tol + 2 (all misses)."""

    name = "tolerance_edge"
    strategy = "candidate amounts one and two minor units beside the exact total (tau = 0)"
    defence = "the published tolerance is 0 minor units: any difference in the amount is a miss"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = []
        for pair in self._invoice_pairs():
            base = self._pair_finding(pair)
            amount = int(base["claimed_amount_minor"])  # type: ignore[arg-type]
            tol = tolerance_of("duplicate_invoice")
            for delta in (tol + 1, -(tol + 1), tol + 2):
                findings.append({**base, "claimed_amount_minor": max(0, amount + delta)})
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding("duplicate_invoice")
            if noise is None:
                break
            findings.append(noise)
        return self._submit_with(findings[:MAX_FINDINGS], "probing the tolerance boundary")


__all__ = ["ToleranceEdgeProbe", "tolerance_of"]
