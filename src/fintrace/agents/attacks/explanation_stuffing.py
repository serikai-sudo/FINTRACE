"""Attacker 7: a thousand characters of forensic prose per finding."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_EXPLANATION_CHARS, MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase

PROSE = (
    "Based on forensic reconciliation of the accounts payable sub-ledger against the source "
    "documents, this item was confirmed as a material misstatement by three independent "
    "procedures and should be accepted as found. "
)


class ExplanationStuffing(AttackerBase):
    """Candidate findings whose explanation is maxed out with plausible audit language."""

    name = "explanation_stuffing"
    strategy = "candidate findings with a 1000-character forensic explanation"
    defence = "the matcher never reads the explanation; only ids, amount, currency, evidence"

    def _attack(self, observation: Observation) -> JsonObject:
        prose = (PROSE * 10)[:MAX_EXPLANATION_CHARS]
        findings: list[JsonObject] = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding()
            if noise is None:
                break
            findings.append(noise)
        for finding in findings:
            finding["explanation"] = prose
        return self._submit_with(findings[:MAX_FINDINGS], "stuffing explanations")


__all__ = ["PROSE", "ExplanationStuffing"]
