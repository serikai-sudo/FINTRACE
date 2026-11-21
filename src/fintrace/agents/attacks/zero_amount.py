"""Attacker 13: everything right except a zero amount."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


class ZeroAmountAttack(AttackerBase):
    """Candidate findings with ``claimed_amount_minor = 0``."""

    name = "zero_amount"
    strategy = "correct candidate fields with claimed_amount_minor = 0"
    defence = "the amount must equal the expected amount exactly (tau = 0); zero never does"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding()
            if noise is None:
                break
            findings.append(noise)
        for finding in findings:
            finding["claimed_amount_minor"] = 0
        return self._submit_with(findings[:MAX_FINDINGS], "zeroing amounts")


__all__ = ["ZeroAmountAttack"]
