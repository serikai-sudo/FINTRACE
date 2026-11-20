"""Shared scaffolding for the reward-hack attackers (differentiator D1, blueprint 16).

An attacker is a Policy that tries to score without auditing: flooding, boundary probing, field
tampering, malformed input, budget abuse. ``AttackerBase`` gives every attack the same cheap
reconnaissance (a few ``list_records`` pages of invoices and payments), the same budget guard
(submit before the last step, never truncate by accident) and the same finding builders, so
each attack file holds only its strategy. Attackers see exactly what any agent sees: this
package depends on ``fintrace.agents.base``, ``fintrace.models`` and ``fintrace.constants`` only
(the import firewall tests in tests/contract cover it too).
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import pairwise
from typing import ClassVar, Final

from fintrace.agents.base import (
    Observation,
    episode_id_of,
    list_action,
    next_cursor_of,
    records_of,
    submit_action,
)
from fintrace.constants import ISSUE_TYPES, IssueType, TableName
from fintrace.models import JsonObject

SCAN_TABLES: Final[tuple[TableName, ...]] = ("invoices", "payments")
MAX_SCAN_PAGES: Final[int] = 3  # per table: plenty of ids for an attack, a handful of steps
Pair = tuple[JsonObject, JsonObject]


def _text(record: JsonObject, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _amount(record: JsonObject, key: str) -> int:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class AttackerBase:
    """Reconnaissance, budget guard and finding builders shared by the 20 attackers."""

    name: ClassVar[str] = "attacker"
    strategy: ClassVar[str] = ""  # one line for the report table
    defence: ClassVar[str] = ""  # which rule of the reward defeats it (report narrative)
    scan: ClassVar[bool] = True  # list invoices and payments before attacking

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._episode_id: str | None = None
        self._invoices: list[JsonObject] = []
        self._payments: list[JsonObject] = []
        self._scan_index = 0
        self._pages = 0
        self._started = False
        self._fired = False  # the one-shot attacks: first the probe, then a clean exit

    # ------------------------------------------------------------------ policy
    def act(self, observation: Observation) -> JsonObject:
        self._extract_episode_id(observation)
        if observation.steps_remaining <= 1:
            return self._submit_empty("budget nearly exhausted, closing the episode")
        if self.scan:
            action = self._scan_step(observation)
            if action is not None:
                return action
        return self._attack(observation)

    def _attack(self, observation: Observation) -> JsonObject:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def _extract_episode_id(self, observation: Observation) -> str | None:
        if self._episode_id is None:
            self._episode_id = episode_id_of(observation)
        return self._episode_id

    @staticmethod
    def _collect_records(observation: Observation, store: list[JsonObject]) -> int:
        records = records_of(observation)
        store.extend(records)
        return len(records)

    @staticmethod
    def _make_action(tool: str, args: JsonObject, thought: str) -> JsonObject:
        return {"tool": tool, "args": args, "thought": thought}

    def _submit_empty(self, thought: str = "attack: empty submission") -> JsonObject:
        return submit_action(self._episode_id or "", [], thought=f"{self.name}: {thought}")

    def _submit_with(self, findings: list[JsonObject], thought: str = "") -> JsonObject:
        note = thought or f"submitting {len(findings)} findings"
        return submit_action(self._episode_id or "", findings, thought=f"{self.name}: {note}")

    def _scan_step(self, observation: Observation) -> JsonObject | None:
        """Page through invoices then payments (a few pages each); None once done."""
        if not self._started:
            self._started = True
            return list_action("invoices", thought=f"{self.name}: listing invoices")
        if self._scan_index >= len(SCAN_TABLES):
            return None
        table = SCAN_TABLES[self._scan_index]
        store = self._invoices if table == "invoices" else self._payments
        self._collect_records(observation, store)
        self._pages += 1
        cursor = next_cursor_of(observation)
        if cursor is not None and self._pages < MAX_SCAN_PAGES:
            return list_action(table, cursor, thought=f"{self.name}: listing {table} {cursor}")
        self._scan_index += 1
        self._pages = 0
        if self._scan_index < len(SCAN_TABLES):
            table = SCAN_TABLES[self._scan_index]
            return list_action(table, thought=f"{self.name}: listing {table}")
        return None

    def _invoice_pairs(self) -> list[Pair]:
        """Invoices sharing counterparty, total and currency: the cheapest duplicate guess."""
        groups: dict[tuple[str, int, str], list[JsonObject]] = defaultdict(list)
        for invoice in self._invoices:
            entity, currency = _text(invoice, "entity_id"), _text(invoice, "currency")
            groups[(entity, _amount(invoice, "total_minor"), currency)].append(invoice)
        pairs: list[Pair] = []
        for _, members in sorted(groups.items()):
            members.sort(key=lambda i: (_text(i, "issued_at"), _text(i, "invoice_id")))
            pairs.extend(pairwise(members))
        return pairs

    def _currencies(self) -> list[str]:
        return sorted({_text(i, "currency") for i in self._invoices if _text(i, "currency")})

    @staticmethod
    def _finding(
        issue_type: str,
        affected: list[str],
        amount: int,
        currency: str,
        evidence: list[str],
        explanation: str = "probe",
    ) -> JsonObject:
        return {
            "type": issue_type,
            "affected_ids": affected,
            "claimed_amount_minor": amount,
            "currency": currency,
            "evidence_ids": evidence,
            "explanation": explanation,
        }

    def _pair_finding(self, pair: Pair, issue_type: IssueType = "duplicate_invoice") -> JsonObject:
        first, second = pair
        return self._finding(
            issue_type,
            sorted((_text(first, "invoice_id"), _text(second, "invoice_id"))),
            _amount(first, "total_minor"),
            _text(first, "currency"),
            [_text(first, "doc_id"), _text(second, "doc_id")],
        )

    def _random_pair_finding(self, issue_type: IssueType | None = None) -> JsonObject | None:
        """A finding over two random invoice ids (or an invoice and a payment): pure noise."""
        if len(self._invoices) < 2:
            return None
        first, second = self._rng.sample(self._invoices, 2)
        kind = issue_type or self._rng.choice(ISSUE_TYPES)
        partner = second
        if self._payments and kind != "duplicate_invoice":
            partner = self._rng.choice(self._payments)
        partner_id = _text(partner, "payment_id") or _text(partner, "invoice_id")
        return self._finding(
            kind,
            sorted((_text(first, "invoice_id"), partner_id)),
            _amount(first, "total_minor"),
            _text(first, "currency"),
            [_text(first, "doc_id"), _text(partner, "doc_id")],
        )


__all__ = ["MAX_SCAN_PAGES", "SCAN_TABLES", "AttackerBase", "Pair"]
