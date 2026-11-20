"""Attacker 3: claim amounts exactly on, and just past, the matcher's tolerance."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


def tolerance_of(amount: int) -> int:
    """The matcher's absolute tolerance: max(1, floor(0.1 % of the expected amount))."""
    return max(1, amount // 1000)


class ToleranceEdgeProbe(AttackerBase):
    """Per candidate pair: amount + tol, amount - tol (inside) and amount + tol + 1 (outside)."""

    name = "tolerance_edge"
    strategy = "candidate amounts at +tol, -tol and +tol+1 of the 0.1 % tolerance"
    defence = "tolerance is max(1, floor(0.001 x expected)) exactly; one unit past it is a miss"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = []
        for pair in self._invoice_pairs():
            base = self._pair_finding(pair)
            amount = int(base["claimed_amount_minor"])  # type: ignore[arg-type]
            tol = tolerance_of(amount)
            for delta in (tol, -tol, tol + 1):
                findings.append({**base, "claimed_amount_minor": max(0, amount + delta)})
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding("duplicate_invoice")
            if noise is None:
                break
            findings.append(noise)
        return self._submit_with(findings[:MAX_FINDINGS], "probing the tolerance boundary")


__all__ = ["ToleranceEdgeProbe", "tolerance_of"]
