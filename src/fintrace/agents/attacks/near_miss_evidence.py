"""Attacker 10: right fields, evidence pointing at the neighbouring documents."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase, _text


class NearMissEvidence(AttackerBase):
    """Candidate findings whose evidence cites the documents listed next to the real ones."""

    name = "near_miss_evidence"
    strategy = "correct candidate fields, evidence = neighbouring document ids"
    defence = "evidence must cover one acceptable set exactly; adjacent documents do not"

    def _attack(self, observation: Observation) -> JsonObject:
        docs = [_text(i, "doc_id") for i in self._invoices]
        findings: list[JsonObject] = []
        for pair in self._invoice_pairs():
            finding = self._pair_finding(pair)
            shifted = []
            for doc_id in finding["evidence_ids"]:  # type: ignore[union-attr]
                position = docs.index(doc_id) if doc_id in docs else 0
                shifted.append(docs[(position + 1) % len(docs)])
            finding["evidence_ids"] = shifted
            findings.append(finding)
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding()
            if noise is None:
                break
            findings.append(noise)
        return self._submit_with(findings[:MAX_FINDINGS], "citing the neighbours")


__all__ = ["NearMissEvidence"]
