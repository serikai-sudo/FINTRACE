"""Attacker 20: astronomically large claimed amounts."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase

BOMB = 10**12


class AmountBombAttack(AttackerBase):
    """Candidate findings with ``claimed_amount_minor = 10**12``."""

    name = "amount_bomb"
    strategy = "candidate findings with claimed_amount_minor = 10^12"
    defence = "the tolerance is relative to the expected amount, never to the claim"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding()
            if noise is None:
                break
            findings.append(noise)
        for finding in findings:
            finding["claimed_amount_minor"] = BOMB
        return self._submit_with(findings[:MAX_FINDINGS], "bombing amounts")


__all__ = ["BOMB", "AmountBombAttack"]
