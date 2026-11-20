"""Random investigator: the lower-bound baseline.

A seeded ``random.Random`` picks 10..60 valid tool calls over ids it has already seen, then
submits either nothing or a handful of random findings. Every action is well-formed, so the
score measures guessing, not malformed input.
"""

from __future__ import annotations

import random
from typing import Final

from fintrace.constants import ISSUE_TYPES, MAX_PAGE_SIZE, TABLE_NAMES, TableName
from fintrace.models import JsonObject

from .base import (
    Observation,
    calculator_action,
    document_action,
    episode_id_of,
    list_action,
    next_cursor_of,
    record_action,
    records_of,
    submit_action,
)

MIN_RANDOM_STEPS: Final[int] = 10
MAX_RANDOM_STEPS: Final[int] = 60
MAX_RANDOM_FINDINGS: Final[int] = 5
_ID_KEY: Final[dict[TableName, str]] = {
    "ledger": "row_id",
    "invoices": "invoice_id",
    "payments": "payment_id",
    "fx": "fx_id",
}
_PROBES: Final[tuple[str, ...]] = ("list", "list", "record", "document", "calc")


class RandomInvestigator:
    """Lower bound: random valid tool calls, then an empty or random submission."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._step = 0
        self._max_steps = MIN_RANDOM_STEPS
        self._episode_id: str | None = None
        self._seen: dict[TableName, list[JsonObject]] = {}
        self._cursors: dict[TableName, list[str]] = {}
        self._last_table: TableName = "ledger"
        self.reset()

    def reset(self) -> None:
        """Re-seed: the same seed and the same episode replay the same trajectory."""
        self._rng = random.Random(self._seed)
        self._step = 0
        self._max_steps = self._rng.randint(MIN_RANDOM_STEPS, MAX_RANDOM_STEPS)
        self._episode_id = None
        self._seen = {table: [] for table in TABLE_NAMES}
        self._cursors = {table: [] for table in TABLE_NAMES}
        self._last_table = "ledger"

    def act(self, observation: Observation) -> JsonObject:
        self._step += 1
        if self._episode_id is None:
            self._episode_id = episode_id_of(observation)
        self._remember(observation)
        if self._episode_id is not None and (
            self._step >= self._max_steps or observation.steps_remaining <= 1
        ):
            return self._submit()
        probe = self._rng.choice(_PROBES)
        if probe == "record" and any(self._seen.values()):
            table = self._rng.choice([t for t in TABLE_NAMES if self._seen[t]])
            record_id = self._rng.choice(self._seen[table]).get(_ID_KEY[table])
            if isinstance(record_id, str):
                thought = f"random probe: open {table} record {record_id}"
                return record_action(table, record_id, thought=thought)
        documents = [
            record["doc_id"]
            for table in ("invoices", "payments")
            for record in self._seen[table]
            if isinstance(record.get("doc_id"), str)
        ]
        if probe == "document" and documents:
            doc_id = self._rng.choice(documents)
            return document_action(doc_id, thought=f"random probe: read document {doc_id}")
        if probe == "calc":
            left, right = self._rng.randint(1, 99_999), self._rng.randint(1, 999)
            expression = f"{left} * {right} / 100"
            return calculator_action(expression, thought=f"random probe: compute {expression}")
        table = self._rng.choice(TABLE_NAMES)
        cursors = self._cursors[table]
        cursor = self._rng.choice([None, *cursors]) if cursors else None
        self._last_table = table
        limit = self._rng.randint(1, MAX_PAGE_SIZE)
        where = cursor or "start"
        return list_action(
            table, cursor, limit, thought=f"random probe: list {table} from {where}, limit {limit}"
        )

    def _remember(self, observation: Observation) -> None:
        records = records_of(observation)
        if records:
            self._seen[self._last_table].extend(records)
        cursor = next_cursor_of(observation)
        if cursor is not None and cursor not in self._cursors[self._last_table]:
            self._cursors[self._last_table].append(cursor)

    def _submit(self) -> JsonObject:
        if self._episode_id is None:  # never happens after a reset observation; stay valid
            return list_action("ledger", thought="random probe: no episode id yet, listing ledger")
        findings: list[JsonObject] = []
        if self._rng.random() < 0.5:
            findings = self._random_findings()
        what = f"{len(findings)} random findings" if findings else "an empty submission"
        return submit_action(
            self._episode_id,
            findings,
            thought=f"random probe budget of {self._max_steps} steps reached: {what}",
        )

    def _random_findings(self) -> list[JsonObject]:
        invoices = [(r, "invoice_id", "total_minor") for r in self._seen["invoices"]]
        payments = [(r, "payment_id", "amount_minor") for r in self._seen["payments"]]
        candidates = [
            (record, id_key, amount_key)
            for record, id_key, amount_key in invoices + payments
            if isinstance(record.get(id_key), str) and isinstance(record.get("doc_id"), str)
        ]
        if not candidates:
            return []
        count = self._rng.randint(0, min(MAX_RANDOM_FINDINGS, len(candidates)))
        findings: list[JsonObject] = []
        for record, id_key, amount_key in self._rng.sample(candidates, count):
            amount = record.get(amount_key)
            findings.append(
                {
                    "type": self._rng.choice(ISSUE_TYPES),
                    "affected_ids": [record[id_key]],
                    "claimed_amount_minor": amount if isinstance(amount, int) else 1,
                    "currency": str(record.get("currency", "USD")),
                    "evidence_ids": [record["doc_id"]],
                    "explanation": "random guess",
                }
            )
        return findings


__all__ = ["MAX_RANDOM_FINDINGS", "MAX_RANDOM_STEPS", "MIN_RANDOM_STEPS", "RandomInvestigator"]
