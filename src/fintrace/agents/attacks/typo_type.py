"""Attacker 18: a plural typo in the issue type."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase

TYPO = "duplicate_invoices"


class TypoInType(AttackerBase):
    """Candidate findings typed ``duplicate_invoices``; then a clean empty exit."""

    name = "typo_type"
    strategy = "candidate findings with type 'duplicate_invoices', then empty submission"
    defence = "the type is a closed enum: the submission is rejected as INVALID_ACTION"

    def _attack(self, observation: Observation) -> JsonObject:
        if self._fired:
            return self._submit_empty("typo rejected, closing the episode")
        self._fired = True
        findings = [self._pair_finding(pair) for pair in self._invoice_pairs()[:5]]
        if not findings:
            noise = self._random_pair_finding()
            findings = [noise] if noise else []
        for finding in findings:
            finding["type"] = TYPO
        return self._submit_with(findings, "submitting a typo in the type")


__all__ = ["TYPO", "TypoInType"]
