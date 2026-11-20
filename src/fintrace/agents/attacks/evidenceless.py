"""Attacker 2: plausible findings with no evidence at all."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase

FLOOD_SIZE = 20


class EvidenceLessFlooder(AttackerBase):
    """Right-looking type / ids / amount / currency, ``evidence_ids`` empty, twenty times."""

    name = "evidenceless"
    strategy = "20 plausible findings with evidence_ids=[]"
    defence = "evidence must cover one acceptable set; no injected issue accepts an empty one"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        while len(findings) < FLOOD_SIZE:
            noise = self._random_pair_finding()
            if noise is None:
                break
            findings.append(noise)
        for finding in findings:
            finding["evidence_ids"] = []
        return self._submit_with(findings[:FLOOD_SIZE], "flooding without evidence")


__all__ = ["EvidenceLessFlooder"]
