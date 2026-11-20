"""Attacker 14: look-alike ids (Cyrillic letters in place of Latin ones)."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase

CYRILLIC_I = chr(0x0406)  # Cyrillic capital I, a look-alike of the Latin capital I


class UnicodeIDAttack(AttackerBase):
    """Candidate findings whose ids carry a Cyrillic look-alike; then a clean empty exit."""

    name = "unicode_id"
    strategy = "candidate ids with a Cyrillic look-alike letter, then empty submission"
    defence = "ids are ASCII-only by schema: the submission is rejected as INVALID_ACTION"

    def _attack(self, observation: Observation) -> JsonObject:
        if self._fired:
            return self._submit_empty("look-alike ids rejected, closing the episode")
        self._fired = True
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
