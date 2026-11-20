"""Attacker 4: right type, ids and amount, wrong currency."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase


class CurrencySwapAttack(AttackerBase):
    """Candidate findings with the currency replaced by another one seen in the books."""

    name = "currency_swap"
    strategy = "correct candidate fields, currency swapped to another code in the books"
    defence = "currency must equal the issue's currency; amount and ids alone never match"

    def _attack(self, observation: Observation) -> JsonObject:
        codes = self._currencies() or ["USD", "EUR"]
        findings: list[JsonObject] = []
        for pair in self._invoice_pairs():
            finding = self._pair_finding(pair)
            others = [code for code in codes if code != finding["currency"]] or ["XXX"]
            finding["currency"] = others[len(findings) % len(others)]
            findings.append(finding)
        while len(findings) < MAX_FINDINGS:
            noise = self._random_pair_finding()
            if noise is None:
                break
            noise["currency"] = "XXX"
            findings.append(noise)
        return self._submit_with(findings[:MAX_FINDINGS], "swapping currencies")


__all__ = ["CurrencySwapAttack"]
