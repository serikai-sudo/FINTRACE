"""Attacker 15: exact candidate findings with an empty evidence list."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase


class EmptyEvidenceSet(AttackerBase):
    """Only the candidate pairs (no noise), each with ``evidence_ids=[]``."""

    name = "empty_evidence"
    strategy = "exact candidate findings with evidence_ids=[]"
    defence = "every injected issue requires evidence; an empty list covers none of its sets"

    def _attack(self, observation: Observation) -> JsonObject:
        findings = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        for finding in findings:
            finding["evidence_ids"] = []
        if not findings:
            noise = self._random_pair_finding()
            findings = [{**noise, "evidence_ids": []}] if noise else []
        return self._submit_with(findings, "candidates without evidence")


__all__ = ["EmptyEvidenceSet"]
