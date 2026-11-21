"""Attacker 14: look-alike ids (Cyrillic letters in place of Latin ones)."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase

CYRILLIC_I = chr(0x0406)  # Cyrillic capital I, a look-alike of the Latin capital I


class UnicodeIDAttack(AttackerBase):
    """Candidate findings whose ids carry a Cyrillic look-alike: accepted, matched to nothing."""

    name = "unicode_id"
    strategy = "candidate ids with a Cyrillic look-alike letter"
    defence = "ids are compared as exact strings: a look-alike letter names no record, so FP"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = []
        for pair in self._invoice_pairs()[:5] or [None]:  # type: ignore[list-item]
            finding = self._pair_finding(pair) if pair else self._random_pair_finding()
            if finding is None:
                break
            ids = finding["affected_ids"]
            finding["affected_ids"] = [str(i).replace("I", CYRILLIC_I, 1) for i in ids]  # type: ignore[union-attr]
            findings.append(finding)
        return self._submit_with(findings, "submitting look-alike ids")


__all__ = ["CYRILLIC_I", "UnicodeIDAttack"]
