"""Attacker 5: right amounts and types, ids drawn as random permutations of the id pool."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.constants import ISSUE_TYPES, MAX_FINDINGS
from fintrace.models import JsonObject

from .base import AttackerBase, _amount, _text


class IDPermutationAttack(AttackerBase):
    """Keeps a candidate's amount and currency but names two random ids from the pool."""

    name = "id_permutation"
    strategy = "candidate amount/currency with random INV/PAY id permutations"
    defence = "affected ids are canonicalised and must match exactly; order is irrelevant"

    def _attack(self, observation: Observation) -> JsonObject:
        pool = [_text(i, "invoice_id") for i in self._invoices] + [
            _text(p, "payment_id") for p in self._payments
        ]
        docs = [_text(i, "doc_id") for i in self._invoices]
        docs += [_text(p, "doc_id") for p in self._payments]
        pairs = self._invoice_pairs() or [(i, i) for i in self._invoices[:1]]
        findings: list[JsonObject] = []
        while pairs and len(pool) >= 2 and len(findings) < MAX_FINDINGS:
            first, _ = pairs[len(findings) % len(pairs)]
            ids = self._rng.sample(pool, 2)
            self._rng.shuffle(ids)  # the permutation is the point: the matcher must not care
            findings.append(
                self._finding(
                    ISSUE_TYPES[len(findings) % len(ISSUE_TYPES)],
                    ids,
                    _amount(first, "total_minor"),
                    _text(first, "currency"),
                    self._rng.sample(docs, min(2, len(docs))),
                )
            )
        return self._submit_with(findings, "permuting ids")


__all__ = ["IDPermutationAttack"]
