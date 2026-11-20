"""Attacker 1: flag everything in sight, every type, up to the submission cap."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import ISSUE_TYPES, MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


class ShotgunAttack(AttackerBase):
    """Thirty findings across all four types from a few listing pages: volume over evidence."""

    name = "shotgun"
    strategy = "30 findings over every id pair in sight, all four types"
    defence = "one-to-one matching: each false finding costs precision, volume never adds recall"

    def _attack(self, observation: Observation) -> JsonObject:
        findings: list[JsonObject] = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        kinds = iter(ISSUE_TYPES * MAX_FINDINGS)
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding(next(kinds))
            if noise is None:
                break
            findings.append(noise)
        return self._submit_with(findings[:MAX_FINDINGS], "shotgun over every pair in sight")


__all__ = ["ShotgunAttack"]
