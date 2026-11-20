"""Rule-based reconciliation: the honest reference baseline, narrating every step.

The agent works like a junior auditor with the five public tools and a 100-step budget:

    start      read the episode header (base currency, period, FX and rounding policy)
    invoices   page through the invoices; group them by (counterparty, total, currency)
    payments   page through the payments; note the foreign-currency ones
    fx         load the daily rate table
    ledger     page through the ledger; map every journal to its invoice / payment
    probe      targeted checks, cheapest and most decisive first:
                 duplicate   pairs of same-counterparty, same-total invoices issued within days
                             of each other -> read both documents, compare the line items
                 period      journals posted outside the reporting period (or documents dated
                             outside it while their journal is inside) -> read one row for the
                             journal id
                 fx          every foreign-currency payment -> read one journal row, re-rate the
                             payment with the policy rule and compare the booked base amount
                 allocation  every payment document -> allocations vs amount vs the invoice the
                             ledger books (missing / partial / transposed / wrong reference)
    submit     all findings, strongest evidence first, within the 30-finding cap

Every action carries a ``thought``: what was just concluded and what is checked next, naming
the record ids involved. Thoughts only ever quote public data. The agent watches
``steps_remaining`` and submits whatever it has before the budget runs out, so it never
truncates; a failed tool call skips the check instead of stopping the investigation.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Final, Literal

from fintrace.constants import MAX_FINDINGS, IssueType, TableName
from fintrace.models import JsonObject

from .base import (
    Observation,
    document_action,
    episode_id_of,
    list_action,
    next_cursor_of,
    record_action,
    records_of,
    records_total_of,
    submit_action,
)

Phase = Literal["start", "invoices", "payments", "fx", "ledger", "probe", "done"]
ProbeKind = Literal["duplicate", "period", "fx", "allocation"]

# Duplicates are keyed within days of the original; monthly recurring invoices and the
# "similar invoice" decoys (weeks apart) must stay outside the window.
DUPLICATE_WINDOW_DAYS: Final[int] = 12
FINDING_PRIORITY: Final[tuple[IssueType, ...]] = (
    "duplicate_invoice",
    "fx_error",
    "period_misclassification",
    "payment_allocation",
)
# policy.rounding.mode (the closed RoundingMode literal) -> decimal's rounding constant, which
# is itself a str; an unknown mode cannot pass the contract models, HALF_EVEN is the fallback
_ROUNDING: Final[dict[str, str]] = {
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
    "ROUND_HALF_UP": ROUND_HALF_UP,
    "ROUND_DOWN": ROUND_DOWN,
}
_ONE: Final[Decimal] = Decimal(1)


@dataclass(frozen=True)
class Probe:
    """One targeted check: what it is about and which ledger rows it may need."""

    kind: ProbeKind
    subject: str  # invoice id (duplicate / period on an invoice) or payment id
    partner: str = ""  # duplicate: the other invoice; period: "invoice" | "payment"
    row_ids: tuple[str, ...] = ()


def _text(record: JsonObject, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _amount(record: JsonObject, key: str) -> int:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parse_timestamp(iso: str) -> datetime | None:
    """Aware datetime from an ISO-8601 string, ``None`` when the text is not one."""
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _parse_decimal(text: object) -> Decimal | None:
    """Exact Decimal from the rate's JSON string; ``None`` for anything else (never a float)."""
    if not isinstance(text, str):
        return None
    try:
        value = Decimal(text)
    except (ArithmeticError, ValueError):
        return None
    return value if value.is_finite() and value > 0 else None


def _days_between(first: str, second: str) -> int | None:
    start, end = _parse_timestamp(first), _parse_timestamp(second)
    if start is None or end is None:
        return None
    return abs(end - start).days


class RuleBasedReconciliation:
    """Honest rule-based auditor: scans, targeted checks, narrated thoughts, one submission."""

    name = "rule_based"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase: Phase = "start"
        self._episode_id: str | None = None
        self._base = "USD"
        self._period_start: datetime | None = None
        self._period_end: datetime | None = None
        self._start_inclusive = True
        self._end_exclusive = True
        self._fx_rules: JsonObject = {}
        self._rounding_mode = "ROUND_HALF_EVEN"
        self._totals: dict[str, int] = {}
        self._invoices: dict[str, JsonObject] = {}
        self._payments: dict[str, JsonObject] = {}
        self._fx_records: list[JsonObject] = []
        self._fx_series: dict[tuple[str, str], tuple[list[str], list[Decimal]]] = {}
        self._ledger_rows = 0
        self._rows_by_payment: dict[str, list[JsonObject]] = defaultdict(list)
        self._rows_by_invoice: dict[str, list[JsonObject]] = defaultdict(list)
        self._docs: dict[str, JsonObject] = {}
        self._rows: dict[str, JsonObject] = {}
        self._failed: set[str] = set()
        self._probes: deque[Probe] = deque()
        self._awaiting: str | None = None
        self._last_list: tuple[TableName, str | None] | None = None
        self._retried = False
        self._note = ""
        self._findings: dict[tuple[str, tuple[str, ...]], JsonObject] = {}

    # ------------------------------------------------------------------ dispatch
    def act(self, observation: Observation) -> JsonObject:
        if self._phase == "start":
            return self._start(observation)
        if self._phase == "done":
            return self._submit(observation, "already submitted")
        if self._phase == "probe":
            # file the answer that just arrived before looking at the budget: the finding it
            # completes must make it into the submission even on the last allowed step
            self._absorb(observation)
            self._settle()
            if observation.steps_remaining <= 1:
                return self._submit(observation, "step budget nearly exhausted")
            return self._next_probe_action(observation)
        if observation.steps_remaining <= 1:
            return self._submit(observation, "step budget nearly exhausted")
        if not observation.ok and self._last_list is not None and not self._retried:
            table, cursor = self._last_list  # one retry per page, then move on
            self._retried = True
            where = cursor or "the first page"
            return self._list(table, cursor, f"Listing {table} from {where} failed once, retrying.")
        self._retried = False
        if self._phase == "invoices":
            return self._scan_invoices(observation)
        if self._phase == "payments":
            return self._scan_payments(observation)
        if self._phase == "fx":
            return self._scan_fx(observation)
        return self._scan_ledger(observation)

    def _list(self, table: TableName, cursor: str | None, thought: str) -> JsonObject:
        self._last_list = (table, cursor)
        return list_action(table, cursor, thought=thought)

    # ------------------------------------------------------------------ phases
    def _start(self, observation: Observation) -> JsonObject:
        self._episode_id = episode_id_of(observation)
        result = observation.result if isinstance(observation.result, dict) else {}
        manifest = result.get("manifest")
        policy = result.get("policy")
        period_text = "unknown period"
        if isinstance(manifest, dict):
            self._base = _text(manifest, "base_currency") or self._base
            start, end = _text(manifest, "period_start"), _text(manifest, "period_end")
            self._period_start = _parse_timestamp(start)
            self._period_end = _parse_timestamp(end)
            if self._period_start is not None and self._period_end is not None:
                period_text = f"period {start[:10]} to {end[:10]} ({_text(manifest, 'timezone')})"
        if isinstance(policy, dict):
            fx_rules = policy.get("fx")
            if isinstance(fx_rules, dict):
                self._fx_rules = fx_rules
            period_rules = policy.get("period")
            if isinstance(period_rules, dict):
                self._start_inclusive = bool(period_rules.get("start_inclusive", True))
                self._end_exclusive = bool(period_rules.get("end_exclusive", True))
            rounding = policy.get("rounding")
            if isinstance(rounding, dict):
                self._rounding_mode = _text(rounding, "mode") or self._rounding_mode
        self._phase = "invoices"
        rule = _text(self._fx_rules, "rate_date_rule") or "unknown rate-date rule"
        return self._list(
            "invoices",
            None,
            (
                f"Starting investigation of {self._episode_id}: base currency {self._base}, "
                f"{period_text}, FX rate taken on the {rule.replace('_', ' ')}. Scanning invoices "
                "for duplicate patterns first."
            ),
        )

    def _scan_invoices(self, observation: Observation) -> JsonObject:
        for record in records_of(observation):
            self._invoices[_text(record, "invoice_id")] = record
        self._remember_total("invoices", observation)
        cursor = next_cursor_of(observation)
        if cursor is not None:
            seen, total = len(self._invoices), self._totals.get("invoices", 0)
            return self._list(
                "invoices",
                cursor,
                f"Scanning invoices: {seen} of {total} seen, fetching page {cursor}.",
            )
        self._phase = "payments"
        counterparties = len({_text(i, "entity_id") for i in self._invoices.values()})
        foreign = sum(1 for i in self._invoices.values() if _text(i, "currency") != self._base)
        return self._list(
            "payments",
            None,
            (
                f"Invoice scan complete: {len(self._invoices)} invoices from {counterparties} "
                f"counterparties, {foreign} in foreign currency. Scanning payments for "
                "allocation issues."
            ),
        )

    def _scan_payments(self, observation: Observation) -> JsonObject:
        for record in records_of(observation):
            self._payments[_text(record, "payment_id")] = record
        self._remember_total("payments", observation)
        cursor = next_cursor_of(observation)
        if cursor is not None:
            seen, total = len(self._payments), self._totals.get("payments", 0)
            return self._list(
                "payments",
                cursor,
                f"Scanning payments: {seen} of {total} seen, fetching page {cursor}.",
            )
        self._phase = "fx"
        currencies = sorted(
            {_text(p, "currency") for p in self._payments.values()} - {self._base}
        )
        foreign = sum(1 for p in self._payments.values() if _text(p, "currency") != self._base)
        return self._list(
            "fx",
            None,
            (
                f"Payment scan complete: {len(self._payments)} payments, {foreign} in foreign "
                f"currency ({', '.join(currencies) or 'none'}). Loading the FX rate table to "
                "re-rate them later."
            ),
        )

    def _scan_fx(self, observation: Observation) -> JsonObject:
        self._fx_records.extend(records_of(observation))
        self._remember_total("fx", observation)
        cursor = next_cursor_of(observation)
        if cursor is not None:
            seen, total = len(self._fx_records), self._totals.get("fx", 0)
            thought = f"Loading FX rates: {seen} of {total}, fetching page {cursor}."
            return self._list("fx", cursor, thought)
        self._build_fx_series()
        self._phase = "ledger"
        pairs = ", ".join(f"{s}->{t}" for s, t in sorted(self._fx_series))
        return self._list(
            "ledger",
            None,
            (
                f"FX table loaded: {len(self._fx_records)} daily rates ({pairs or 'no series'}). "
                "Scanning the ledger to map every journal to its invoice or payment."
            ),
        )

    def _scan_ledger(self, observation: Observation) -> JsonObject:
        for row in records_of(observation):
            self._ledger_rows += 1
            payment_id, invoice_id = _text(row, "payment_id"), _text(row, "invoice_id")
            if payment_id:
                self._rows_by_payment[payment_id].append(row)
            elif invoice_id:
                self._rows_by_invoice[invoice_id].append(row)
        self._remember_total("ledger", observation)
        cursor = next_cursor_of(observation)
        if cursor is not None:
            total = self._totals.get("ledger", 0)
            return self._list(
                "ledger",
                cursor,
                f"Scanning ledger: {self._ledger_rows} of {total} rows, page {cursor}.",
            )
        self._plan()
        self._phase = "probe"
        counts = Counter(probe.kind for probe in self._probes)
        self._note = (
            f"Ledger scan complete ({self._ledger_rows} rows). Plan: {counts['duplicate']} "
            f"duplicate candidate pairs, {counts['period']} journals or documents outside the "
            f"period, {counts['fx']} foreign-currency payments to re-rate, {counts['allocation']} "
            f"payment documents to reconcile; {observation.steps_remaining} steps left."
        )
        return self._next_probe_action(observation)

    # ------------------------------------------------------------------ planning
    def _remember_total(self, table: str, observation: Observation) -> None:
        total = records_total_of(observation)
        if total is not None:
            self._totals[table] = total

    def _build_fx_series(self) -> None:
        grouped: dict[tuple[str, str], list[tuple[str, Decimal]]] = defaultdict(list)
        for record in self._fx_records:
            rate = _parse_decimal(record.get("rate"))
            day = _text(record, "date")
            if rate is None or not day:
                continue
            key = (_text(record, "source_currency"), _text(record, "target_currency"))
            grouped[key].append((day, rate))
        for key, points in grouped.items():
            points.sort()
            self._fx_series[key] = ([day for day, _ in points], [rate for _, rate in points])

    def _plan(self) -> None:
        probes: list[Probe] = []
        probes.extend(self._duplicate_candidates())
        probes.extend(self._period_candidates())
        for payment_id, payment in self._payments.items():
            rows = self._rows_by_payment.get(payment_id)
            if rows and _text(payment, "currency") != self._base:
                probes.append(Probe("fx", payment_id, row_ids=(_text(rows[0], "row_id"),)))
        probes.extend(Probe("allocation", payment_id) for payment_id in self._payments)
        self._probes = deque(probes)

    def _duplicate_candidates(self) -> list[Probe]:
        groups: dict[tuple[str, int, str], list[JsonObject]] = defaultdict(list)
        for invoice in self._invoices.values():
            entity, currency = _text(invoice, "entity_id"), _text(invoice, "currency")
            groups[(entity, _amount(invoice, "total_minor"), currency)].append(invoice)
        probes: list[Probe] = []
        for _, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            members.sort(key=lambda i: (_text(i, "issued_at"), _text(i, "invoice_id")))
            for index, first in enumerate(members):
                for second in members[index + 1 :]:
                    gap = _days_between(_text(first, "issued_at"), _text(second, "issued_at"))
                    if gap is not None and gap <= DUPLICATE_WINDOW_DAYS:
                        pair = (_text(first, "invoice_id"), _text(second, "invoice_id"))
                        probes.append(Probe("duplicate", *pair))
        return probes

    def _period_candidates(self) -> list[Probe]:
        probes: list[Probe] = []
        seen: set[str] = set()
        journals = (("payment", self._rows_by_payment), ("invoice", self._rows_by_invoice))
        for kind, rows_by in journals:
            for owner, rows in rows_by.items():
                if owner in seen or all(self._in_period(_text(r, "timestamp")) for r in rows):
                    continue
                seen.add(owner)
                probes.append(Probe("period", owner, kind, tuple(_text(r, "row_id") for r in rows)))
        for invoice_id, invoice in self._invoices.items():
            rows = self._rows_by_invoice.get(invoice_id)
            if invoice_id in seen or not rows or self._in_period(_text(invoice, "issued_at")):
                continue
            seen.add(invoice_id)
            probes.append(
                Probe("period", invoice_id, "invoice", tuple(_text(r, "row_id") for r in rows))
            )
        return probes

    def _in_period(self, iso: str) -> bool:
        moment = _parse_timestamp(iso)
        if self._period_start is None or self._period_end is None or moment is None:
            return True  # nothing to judge against: never a finding on unparsable data
        start, end = self._period_start, self._period_end
        after_start = start <= moment if self._start_inclusive else start < moment
        before_end = moment < end if self._end_exclusive else moment <= end
        return after_start and before_end

    # ------------------------------------------------------------------ probing
    def _absorb(self, observation: Observation) -> None:
        """File the answer to the last probe request (document or ledger row) or mark it failed."""
        awaiting, self._awaiting = self._awaiting, None
        if awaiting is None:
            return
        result = observation.result if observation.ok else None
        if not isinstance(result, dict):
            self._failed.add(awaiting)
            return
        document = result.get("document")
        record = result.get("record")
        if observation.tool == "get_document" and isinstance(document, dict):
            self._docs[_text(document, "doc_id") or awaiting] = document
        elif observation.tool == "get_record" and isinstance(record, dict):
            self._rows[_text(record, "row_id") or awaiting] = record
        else:
            self._failed.add(awaiting)

    def _settle(self) -> None:
        """Resolve every probe at the head of the queue that needs no further fetch."""
        while self._probes and self._needs(self._probes[0]) is None:
            self._resolve(self._probes.popleft())

    def _next_probe_action(self, observation: Observation) -> JsonObject:
        self._settle()
        need = self._needs(self._probes[0]) if self._probes else None
        if need is None:
            return self._submit(observation, "all planned checks done")
        return self._action_for(self._probes[0], need)

    def _missing_document(self, doc_id: str) -> bool:
        return bool(doc_id) and doc_id not in self._docs and doc_id not in self._failed

    def _needs(self, probe: Probe) -> str | None:
        """The document or ledger row id this probe still has to read, ``None`` when resolvable."""
        if probe.kind == "duplicate":
            for invoice_id in (probe.subject, probe.partner):
                doc_id = _text(self._invoices.get(invoice_id, {}), "doc_id")
                if self._missing_document(doc_id):
                    return doc_id
            return None
        if probe.kind == "allocation":
            doc_id = _text(self._payments.get(probe.subject, {}), "doc_id")
            return doc_id if self._missing_document(doc_id) else None
        row_id = probe.row_ids[0] if probe.row_ids else ""
        if row_id and row_id not in self._rows and row_id not in self._failed:
            return row_id
        return None

    def _action_for(self, probe: Probe, need: str) -> JsonObject:
        """The fetch for ``need`` with the narrated reason; remembers what is awaited."""
        self._awaiting = need
        if probe.kind == "duplicate":
            thought = self._duplicate_thought(probe, need)
            return document_action(need, thought=self._with_note(thought))
        if probe.kind == "allocation":
            payment = self._payments.get(probe.subject, {})
            thought = (
                f"Reconciling payment {probe.subject} ({_amount(payment, 'amount_minor')} "
                f"{_text(payment, 'currency')}) against its allocations: reading {need}."
            )
            return document_action(need, thought=self._with_note(thought))
        if probe.kind == "period":
            thought = (
                f"Journal of {probe.subject} is posted outside the reporting period: reading "
                f"ledger row {need} for its journal id and amount."
            )
        else:
            payment = self._payments.get(probe.subject, {})
            amount, currency = _amount(payment, "amount_minor"), _text(payment, "currency")
            thought = (
                f"Re-rating foreign payment {probe.subject} ({amount} {currency} on "
                f"{_text(payment, 'timestamp')[:10]}): reading ledger row {need} for the "
                f"booked {self._base} amount."
            )
        return record_action("ledger", need, thought=self._with_note(thought))

    def _duplicate_thought(self, probe: Probe, doc_id: str) -> str:
        first = self._invoices.get(probe.subject, {})
        second = self._invoices.get(probe.partner, {})
        gap = _days_between(_text(first, "issued_at"), _text(second, "issued_at"))
        apart = f"{gap} days apart" if gap is not None else "on unreadable dates"
        return (
            f"Checking {probe.subject} against {probe.partner}: same counterparty "
            f"{_text(first, 'entity_id')}, same total {_amount(first, 'total_minor')} "
            f"{_text(first, 'currency')}, issued {apart}. Reading {doc_id} to compare line items."
        )

    def _with_note(self, thought: str) -> str:
        note, self._note = self._note, ""
        return f"{note} {thought}".strip()

    # ------------------------------------------------------------------ resolution
    def _resolve(self, probe: Probe) -> None:
        if probe.kind == "duplicate":
            self._resolve_duplicate(probe)
        elif probe.kind == "period":
            self._resolve_period(probe)
        elif probe.kind == "fx":
            self._resolve_fx(probe)
        else:
            self._resolve_allocation(probe)

    def _resolve_duplicate(self, probe: Probe) -> None:
        first = self._invoices.get(probe.subject, {})
        second = self._invoices.get(probe.partner, {})
        doc_a = self._docs.get(_text(first, "doc_id"))
        doc_b = self._docs.get(_text(second, "doc_id"))
        if doc_a is None or doc_b is None:
            self._note = f"{probe.subject} vs {probe.partner}: a document could not be read."
            return
        lines_a, lines_b = doc_a.get("line_items"), doc_b.get("line_items")
        if lines_a != lines_b or not isinstance(lines_a, list):
            self._note = (
                f"{probe.subject} and {probe.partner}: line items differ, so this is a similar "
                "invoice rather than a duplicate."
            )
            return
        total, currency = _amount(first, "total_minor"), _text(first, "currency")
        gap = _days_between(_text(first, "issued_at"), _text(second, "issued_at")) or 0
        self._add_finding(
            "duplicate_invoice",
            (probe.subject, probe.partner),
            total,
            currency,
            [_text(first, "doc_id"), _text(second, "doc_id")],
            f"Invoices {probe.subject} and {probe.partner}: same counterparty "
            f"{_text(first, 'entity_id')}, same total {total} {currency}, identical "
            f"{len(lines_a)} line items, issued {gap} days apart.",
        )
        self._note = (
            f"{probe.subject} and {probe.partner}: identical {len(lines_a)} line items, duplicate "
            "invoice recorded."
        )

    def _resolve_period(self, probe: Probe) -> None:
        row = self._rows.get(probe.row_ids[0]) if probe.row_ids else None
        if row is None:
            self._note = f"Journal of {probe.subject}: ledger row unavailable, skipped."
            return
        journal_id = _text(row, "journal_id")
        if probe.partner == "invoice":
            invoice = self._invoices.get(probe.subject, {})
            amount, currency = _amount(invoice, "total_minor"), _text(invoice, "currency")
            evidence = [_text(invoice, "doc_id"), *probe.row_ids]
            what = f"invoice {probe.subject} dated {_text(invoice, 'issued_at')[:10]}"
        else:
            payment = self._payments.get(probe.subject, {})
            invoice = self._invoices.get(_text(row, "invoice_id"), {})
            amount, currency = _amount(payment, "amount_minor"), _text(payment, "currency")
            evidence = [_text(payment, "doc_id"), _text(invoice, "doc_id"), *probe.row_ids]
            what = f"payment {probe.subject} dated {_text(payment, 'timestamp')[:10]}"
        posted = _text(row, "timestamp")
        self._add_finding(
            "period_misclassification",
            (probe.subject, journal_id),
            amount,
            currency,
            [e for e in evidence if e],
            f"Journal {journal_id} of {what} is posted at {posted}, on the other side of the "
            "reporting period boundary from its document.",
        )
        self._note = (
            f"{what}: journal {journal_id} posted {posted[:16]}, outside the period, "
            "period misclassification recorded."
        )

    def _resolve_fx(self, probe: Probe) -> None:
        payment = self._payments.get(probe.subject, {})
        row = self._rows.get(probe.row_ids[0]) if probe.row_ids else None
        invoice = self._invoices.get(_text(row, "invoice_id"), {}) if row else {}
        if row is None or not invoice:
            self._note = f"{probe.subject}: journal row or invoice unavailable, FX check skipped."
            return
        booked = _amount(row, "debit_minor") + _amount(row, "credit_minor")
        transaction_date = _text(self._fx_rules, "rate_date_rule") == "transaction_date"
        anchor = _text(invoice, "issued_at") if transaction_date else _text(payment, "timestamp")
        day = anchor[:10]
        currency = _text(payment, "currency")
        rate = self._policy_rate(currency, day)
        if rate is None:
            self._note = f"{probe.subject}: no {currency} rate for {day}, FX check skipped."
            return
        amount = _amount(payment, "amount_minor")
        correct = self._convert(amount, rate)
        if booked == correct:
            self._note = (
                f"{probe.subject}: booked {booked} {self._base} matches {amount} "
                f"{_text(payment, 'currency')} at {rate} ({day}), clean."
            )
            return
        gap = abs(booked - correct)
        self._add_finding(
            "fx_error",
            (_text(invoice, "invoice_id"), probe.subject),
            gap,
            self._base,
            [_text(payment, "doc_id"), _text(invoice, "doc_id")],
            f"Payment {probe.subject} of {amount} {_text(payment, 'currency')} is booked at "
            f"{booked} {self._base}; the policy rate {rate} of {day} gives {correct} "
            f"{self._base} (difference {gap}).",
        )
        self._note = (
            f"{probe.subject}: booked {booked} {self._base} but {amount} "
            f"{_text(payment, 'currency')} at the policy rate {rate} ({day}) is {correct}, "
            f"FX error of {gap} recorded."
        )

    def _resolve_allocation(self, probe: Probe) -> None:
        payment = self._payments.get(probe.subject, {})
        document = self._docs.get(_text(payment, "doc_id"))
        if document is None:
            self._note = f"{probe.subject}: payment document unavailable, skipped."
            return
        amount = _amount(document, "amount_minor")
        refs = document.get("invoice_refs")
        refs = [ref for ref in refs if isinstance(ref, dict)] if isinstance(refs, list) else []
        rows = self._rows_by_payment.get(probe.subject, [])
        booked_invoice = _text(rows[0], "invoice_id") if rows else ""
        pay_doc = _text(payment, "doc_id")
        currency = _text(payment, "currency")
        if not refs:
            if not booked_invoice:
                self._note = f"{probe.subject}: no allocation and no ledger booking, skipped."
                return
            self._add_finding(
                "payment_allocation",
                (probe.subject, booked_invoice),
                amount,
                currency,
                [pay_doc, self._doc_of(booked_invoice)],
                f"Payment {probe.subject} ({amount} {currency}) carries no invoice allocation "
                f"although the ledger books it against {booked_invoice}.",
            )
            self._note = f"{probe.subject}: no allocation at all, missing allocation recorded."
            return
        allocated = sum(_amount(ref, "allocated_minor") for ref in refs)
        if len(refs) == 1:
            ref_invoice = _text(refs[0], "invoice_id")
            if allocated != amount:
                gap = abs(amount - allocated)
                self._add_finding(
                    "payment_allocation",
                    (probe.subject, ref_invoice),
                    gap,
                    currency,
                    [pay_doc, self._doc_of(ref_invoice)],
                    f"Payment {probe.subject} of {amount} {currency} allocates {allocated} to "
                    f"{ref_invoice} (difference {gap}).",
                )
                self._note = (
                    f"{probe.subject}: allocates {allocated} of {amount} {currency} to "
                    f"{ref_invoice}, allocation gap of {gap} recorded."
                )
                return
            if booked_invoice and ref_invoice != booked_invoice:
                self._add_finding(
                    "payment_allocation",
                    (probe.subject, booked_invoice, ref_invoice),
                    allocated,
                    currency,
                    [pay_doc, self._doc_of(booked_invoice), self._doc_of(ref_invoice)],
                    f"Payment {probe.subject} allocates {allocated} {currency} to {ref_invoice} "
                    f"while the ledger books it against {booked_invoice}.",
                )
                self._note = (
                    f"{probe.subject}: document allocates {ref_invoice} but the ledger books "
                    f"{booked_invoice}, wrong invoice reference recorded."
                )
                return
            self._note = f"{probe.subject}: fully allocated to {ref_invoice}, clean."
            return
        if allocated != amount:
            affected = (probe.subject, *(_text(ref, "invoice_id") for ref in refs))
            self._add_finding(
                "payment_allocation",
                affected,
                abs(amount - allocated),
                currency,
                [pay_doc, *(self._doc_of(_text(ref, "invoice_id")) for ref in refs)],
                f"Payment {probe.subject} of {amount} {currency} allocates {allocated} across "
                f"{len(refs)} invoices.",
            )
            self._note = f"{probe.subject}: {len(refs)} allocations do not sum to the amount."
            return
        self._note = f"{probe.subject}: {len(refs)} allocations sum to the amount, clean."

    # ------------------------------------------------------------------ helpers
    def _doc_of(self, invoice_id: str) -> str:
        return _text(self._invoices.get(invoice_id, {}), "doc_id")

    def _policy_rate(self, source: str, day: str) -> Decimal | None:
        series = self._fx_series.get((source, self._base))
        if series is None or not day:
            return None
        dates, rates = series
        index = bisect_left(dates, day)
        if index < len(dates) and dates[index] == day:
            return rates[index]
        fallback = _text(self._fx_rules, "fallback")
        if fallback == "previous_available_date" and index > 0:
            return rates[index - 1]
        if fallback == "next_available_date" and index < len(dates):
            return rates[index]
        return None

    def _convert(self, amount: int, rate: Decimal) -> int:
        value = Decimal(amount)
        direction = _text(self._fx_rules, "rate_direction") or "source_to_target"
        converted = value * rate if direction == "source_to_target" else value / rate
        mode = _ROUNDING.get(self._rounding_mode, ROUND_HALF_EVEN)
        return int(converted.quantize(_ONE, rounding=mode))

    def _add_finding(
        self,
        issue_type: IssueType,
        affected: tuple[str, ...],
        amount: int,
        currency: str,
        evidence: list[str],
        explanation: str,
    ) -> None:
        canonical = tuple(sorted({a for a in affected if a}))
        key = (issue_type, canonical)
        if key in self._findings or not canonical or amount <= 0:
            return
        unique_evidence = list(dict.fromkeys(e for e in evidence if e))
        self._findings[key] = {
            "type": issue_type,
            "affected_ids": list(canonical),
            "claimed_amount_minor": amount,
            "currency": currency,
            "evidence_ids": unique_evidence[:10],
            "explanation": explanation[:1000],
        }

    def findings(self) -> list[JsonObject]:
        """Findings in submission order: strongest evidence type first, then by affected ids."""
        order = {issue_type: rank for rank, issue_type in enumerate(FINDING_PRIORITY)}
        keys = sorted(self._findings, key=lambda k: (order.get(k[0], len(order)), k[1]))
        return [self._findings[key] for key in keys]

    def _submit(self, observation: Observation, reason: str) -> JsonObject:
        self._phase = "done"
        findings = self.findings()[:MAX_FINDINGS]
        counts = Counter(str(finding["type"]) for finding in findings)
        summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items())) or "none"
        thought = (
            f"Investigation complete after {observation.steps_used} steps ({reason}): "
            f"submitting {len(findings)} findings: {summary}."
        )
        return submit_action(self._episode_id or "", findings, thought=self._with_note(thought))


__all__ = ["DUPLICATE_WINDOW_DAYS", "FINDING_PRIORITY", "Probe", "RuleBasedReconciliation"]
