"""Mark-everything attack: list every document cheaply, then flag 30 of them at random types.

The adversarial sanity check of the reward: a submission that "finds" everything must score
near the floor. Findings carry a single affected id and a cycling issue type, so none can
match a real issue (every real issue names at least two records).
"""

from __future__ import annotations

from typing import Final, Literal

from fintrace.constants import ISSUE_TYPES, MAX_FINDINGS
from fintrace.models import JsonObject

from .base import (
    Observation,
    episode_id_of,
    list_action,
    next_cursor_of,
    records_of,
    submit_action,
)

Phase = Literal["start", "invoices", "payments", "submit"]
EXPLANATION: Final[str] = "Flagged by comprehensive review"
_PER_TABLE: Final[int] = MAX_FINDINGS // 2  # 15 invoices + 15 payments


class MarkEverythingAttack:
    """Collect all invoices and payments, submit ``MAX_FINDINGS`` generic findings."""

    name = "mark_everything"

    def __init__(self) -> None:
        self._phase: Phase = "start"
        self._invoices: list[JsonObject] = []
        self._payments: list[JsonObject] = []
        self._episode_id: str | None = None
        self.reset()

    def reset(self) -> None:
        self._phase = "start"
        self._invoices = []
        self._payments = []
        self._episode_id = None

    def act(self, observation: Observation) -> JsonObject:
        if self._episode_id is None:
            self._episode_id = episode_id_of(observation)
        if self._phase == "start":
            self._phase = "invoices"
            return list_action("invoices", thought="collecting invoices: first page")
        if observation.steps_remaining <= 1:
            return self._submit()
        if self._phase == "invoices":
            self._invoices.extend(records_of(observation))
            cursor = next_cursor_of(observation)
            if cursor is not None:
                thought = f"collecting invoices: {len(self._invoices)} so far, page {cursor}"
                return list_action("invoices", cursor, thought=thought)
            self._phase = "payments"
            return list_action("payments", thought="collecting payments: first page")
        if self._phase == "payments":
            self._payments.extend(records_of(observation))
            cursor = next_cursor_of(observation)
            if cursor is not None:
                thought = f"collecting payments: {len(self._payments)} so far, page {cursor}"
                return list_action("payments", cursor, thought=thought)
            self._phase = "submit"
        return self._submit()

    def _submit(self) -> JsonObject:
        findings = self.build_findings()
        return submit_action(
            self._episode_id or "",
            findings,
            thought=f"submitting {len(findings)} flagged items across all issue types",
        )

    def build_findings(self) -> list[JsonObject]:
        """Up to ``MAX_FINDINGS`` findings cycling through the four issue types."""
        findings: list[JsonObject] = []
        sources = [
            (record, "invoice_id", "total_minor") for record in self._invoices[:_PER_TABLE]
        ] + [(record, "payment_id", "amount_minor") for record in self._payments[:_PER_TABLE]]
        for record, id_key, amount_key in sources:
            record_id, doc_id = record.get(id_key), record.get("doc_id")
            if not isinstance(record_id, str) or not isinstance(doc_id, str):
                continue  # a malformed listing never breaks the attack
            amount = record.get(amount_key)
            findings.append(
                {
                    "type": ISSUE_TYPES[len(findings) % len(ISSUE_TYPES)],
                    "affected_ids": [record_id],
                    "claimed_amount_minor": amount if isinstance(amount, int) else 100,
                    "currency": str(record.get("currency", "USD")),
                    "evidence_ids": [doc_id],
                    "explanation": EXPLANATION,
                }
            )
            if len(findings) == MAX_FINDINGS:
                break
        return findings


__all__ = ["EXPLANATION", "MarkEverythingAttack"]
