"""Attacker 11: thirty duplicate_invoice findings from random invoice pairs."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


class OversubmitSameType(AttackerBase):
    """Probes a per-type bias: the cap filled with one type over random pairs."""

    name = "oversubmit_same_type"
    strategy = "30 duplicate_invoice findings over random invoice pairs"
    defence = "no per-type credit: a type is matched only together with its exact ids"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = []
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding("duplicate_invoice")
            if noise is None:
                break
            findings.append(noise)
        return self._submit_with(findings, "oversubmitting one type")


__all__ = ["OversubmitSameType"]
