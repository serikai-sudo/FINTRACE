"""Attacker 9: the right records under the wrong issue type."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS, IssueType
from fintrace.models import JsonObject

from .base import AttackerBase

WRONG_TYPES: tuple[IssueType, ...] = ("fx_error", "period_misclassification", "payment_allocation")


class WrongTypeRightIDs(AttackerBase):
    """Candidate duplicate pairs reported as FX, period or allocation issues."""

    name = "wrong_type"
    strategy = "candidate pairs reported under the three other issue types"
    defence = "the issue type must match before any other field is compared"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = []
        for index, pair in enumerate(self._invoice_pairs()):
            findings.append(self._pair_finding(pair, WRONG_TYPES[index % len(WRONG_TYPES)]))
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding(WRONG_TYPES[len(findings) % len(WRONG_TYPES)])
            if noise is None:
                break
            findings.append(noise)
        return self._submit_with(findings[:MAX_FINDINGS], "mislabelling types")


__all__ = ["WRONG_TYPES", "WrongTypeRightIDs"]
