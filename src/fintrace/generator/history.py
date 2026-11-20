"""Clean financial history for one episode: invoices, payments and a double-entry ledger.

Posting rules (clean case):
- invoice journal, in the document currency: supplier -> EXPENSE Dr / AP Cr; customer -> AR Dr /
  REVENUE Cr; timestamp = issued_at.
- payment journal: in the document currency when it equals the base currency (AP Dr / CASH Cr,
  or CASH Dr / AR Cr); otherwise both rows in the base currency at the FX rate of the policy
  date (``rate_date_rule``), rounded with the policy rounding mode.
- document-less "misc" journals (payroll/fees: EXPENSE Dr / CASH Cr; cash sales: CASH Dr /
  REVENUE Cr) fill the ledger to its target size, in the base currency, optionally tagged with a
  counterparty that also has invoices; they never touch AP/AR.
Clean does not mean fully settled: an invoice whose payment lag crosses period_end stays open,
and a split whose later part crosses period_end stays partially paid. Injectors must stay
distinguishable from these (an injected partial payment misallocates; a clean one is just late).
All amounts derive from Benford-sampled totals by scale-preserving splits.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final, NamedTuple

from fintrace.models import (
    AccountName,
    CurrencyCode,
    EpisodeConfig,
    FXRate,
    Invoice,
    InvoiceAllocation,
    LedgerRow,
    LineItem,
    Payment,
    Policy,
)

from .benford import benford_amount_minor, benford_amounts_minor, derive_sub_rng
from .company import (
    CREDIT_NOTE_PREFIX,
    LINE_ITEM_DESCRIPTIONS,
    RECURRING_SERVICES,
    CompanyProfile,
    Entity,
    EntityKind,
)
from .fx import build_fx_index, convert_minor, policy_date, rate_on

_SECONDS_PER_DAY: Final[int] = 86_400
_MIN_SPLIT_PART_MINOR: Final[int] = 100  # >= 1 major unit, so an FX conversion never rounds to 0
_RECURRING_SERIES_LENGTH: Final[int] = 3
_RECURRING_SPACING_DAYS: Final[int] = 30
_PAYMENT_LAG_DAYS: Final[tuple[int, int]] = (1, 45)
_SPLIT_GAP_DAYS: Final[tuple[int, int]] = (3, 20)
_DUE_TERMS_DAYS: Final[tuple[int, ...]] = (14, 30, 45)
_MISC_EXPENSE_SHARE: Final[Decimal] = Decimal("0.6")
_MISC_TAGGED_SHARE: Final[Decimal] = Decimal("0.5")  # misc journals carrying a counterparty
_DOCS_PER_ROWS_DIVISOR: Final[int] = 3  # documents <= target rows / 3 leaves room for misc rows
_DOCS_PER_ROWS_FLOOR_DIVISOR: Final[int] = 10  # documents >= target rows / 10 (less misc)
_RECURRING_SHARE_DIVISOR: Final[int] = 6  # recurring invoices <= half of all invoices
_DOCS_PER_INVOICE_TENTHS: Final[int] = 19  # expected documents per invoice ~ 1.9
# decoy knobs: payments of a split decoy, credit-note share, spacing of "similar" invoices
_DECOY_SPLIT_FIRST_LAG_DAYS: Final[tuple[int, int]] = (1, 10)
_DECOY_REFUND_LAG_DAYS: Final[tuple[int, int]] = (1, 30)
_DECOY_REFUND_PCT: Final[tuple[int, ...]] = (25, 50, 100)
_DECOY_SIMILAR_GAP_DAYS: Final[tuple[int, int]] = (14, 45)
_DECOY_TAIL_DAYS: Final[int] = 31  # a decoy with follow-up documents starts this early at least
_DECOY_LINE_ATTEMPTS: Final[int] = 12  # draws to get line items that differ from the twin's

FxLookup = Callable[[str, CurrencyCode, CurrencyCode], Decimal]


class DecoyGroup(NamedTuple):
    """One legitimate-but-suspicious pattern instance: its name and the documents it owns."""

    op: str  # repeated_amounts | split_payment | refund | similar_invoices
    invoice_ids: tuple[str, ...]
    payment_ids: tuple[str, ...]


class CleanHistory(NamedTuple):
    """Canonical documents and ledger of a clean episode plus the decoy markers."""

    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]
    ledger: tuple[LedgerRow, ...]
    recurring_series: tuple[tuple[str, ...], ...]  # invoice ids per legitimate recurring series
    decoys: tuple[DecoyGroup, ...]  # legitimate patterns that look like errors


def _require(condition: bool, message: str) -> None:
    """Internal invariant guard; a failure is a generator bug, not a data condition."""
    if not condition:
        raise ValueError(f"history invariant violated: {message}")


class _IdCounter:
    """Sequential labels per prefix (INV-0001, ...); build.py permutes them afterwards."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, prefix: str, width: int = 4) -> str:
        n = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = n
        return f"{prefix}-{n:0{width}d}"


def _timestamp_between(rng: random.Random, start: datetime, end: datetime) -> datetime:
    """Uniform second-resolution instant in [start, end)."""
    span = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randrange(span))


def _add_months(moment: datetime, months: int) -> tuple[int, int]:
    """(year, month) of ``moment`` shifted by ``months``."""
    index = moment.year * 12 + (moment.month - 1) + months
    return index // 12, index % 12 + 1


def _split_amount(rng: random.Random, total: int, parts: int, min_part: int = 1) -> list[int]:
    """Split ``total`` into ``parts`` integers >= ``min_part`` summing exactly to ``total``."""
    if parts <= 1 or total < parts * min_part:
        return [total]
    body = total - parts * min_part
    cuts = sorted(rng.randint(0, body) for _ in range(parts - 1))
    bounds = [0, *cuts, body]
    return [min_part + bounds[i + 1] - bounds[i] for i in range(parts)]


def _line_items(
    rng: random.Random, total: int, recurring_label: str | None
) -> tuple[LineItem, ...]:
    """1-5 lines with amount == qty * unit summing exactly to total (last line absorbs rounding)."""
    if recurring_label is not None:
        return (LineItem(desc=recurring_label, qty=1, unit_minor=total, amount_minor=total),)
    parts = _split_amount(rng, total, min(rng.randint(1, 5), total))
    descs = rng.sample(LINE_ITEM_DESCRIPTIONS, k=len(parts))
    items: list[LineItem] = []
    carry = 0
    for desc, part in zip(descs[:-1], parts[:-1], strict=True):
        qty = rng.randint(1, 5)
        if part < qty:
            qty = 1
        unit = part // qty
        carry += part - qty * unit
        items.append(LineItem(desc=desc, qty=qty, unit_minor=unit, amount_minor=qty * unit))
    last = parts[-1] + carry
    items.append(LineItem(desc=descs[-1], qty=1, unit_minor=last, amount_minor=last))
    _require(sum(item.amount_minor for item in items) == total, "line items must sum to total")
    return tuple(items)


def _generate_single_invoice(
    rng: random.Random,
    entity: Entity,
    issued_at: datetime,
    total: int,
    recurring_label: str | None,
    ids: _IdCounter,
) -> Invoice:
    """One invoice for ``entity`` in its default currency with sequential ids."""
    due_at = issued_at + timedelta(days=rng.choice(_DUE_TERMS_DAYS))
    return Invoice(
        doc_id=ids.next("DOC"),
        invoice_id=ids.next("INV"),
        entity_id=entity.entity_id,
        issued_at=issued_at.isoformat(),
        due_at=due_at.isoformat(),
        currency=entity.default_currency,
        total_minor=total,
        line_items=_line_items(rng, total, recurring_label),
    )


def _recurring_series_length(start: datetime, end: datetime) -> int:
    """How many monthly invoices (30-day spacing) fit in the period: 3, 2, or 0 (no series)."""
    period_days = (end - start).days
    fits = 1 + (period_days - 1) // _RECURRING_SPACING_DAYS
    return min(_RECURRING_SERIES_LENGTH, fits) if fits >= 2 else 0


def _generate_recurring_series(
    rng: random.Random,
    entity: Entity,
    service: str,
    total: int,
    start: datetime,
    end: datetime,
    series_length: int,
    ids: _IdCounter,
) -> list[Invoice]:
    """Invoices with an identical total, exactly 30 days apart, desc '<service> - YYYY-MM'."""
    period_days = (end - start).days
    latest_first = period_days - _RECURRING_SPACING_DAYS * (series_length - 1) - 1
    _require(latest_first >= 0, "recurring series must fit inside the period")
    first = start + timedelta(
        days=rng.randint(0, latest_first), seconds=rng.randrange(_SECONDS_PER_DAY)
    )
    series: list[Invoice] = []
    for k in range(series_length):
        issued = first + timedelta(days=_RECURRING_SPACING_DAYS * k)
        _require(issued < end, "recurring invoice inside the period")
        year, month = _add_months(first, k)
        label = f"{service} - {year:04d}-{month:02d}"
        series.append(_generate_single_invoice(rng, entity, issued, total, label, ids))
    return series


def _generate_payments_for_invoice(
    rng: random.Random,
    invoice: Invoice,
    issued_at: datetime,
    period_end: datetime,
    config: EpisodeConfig,
    ids: _IdCounter,
    allow_split: bool,
    force_split: bool,
    force_paid: bool,
) -> list[Payment]:
    """0-3 payments: none if the lag crosses period_end (unless forced); 2-3 parts when split."""
    first = issued_at + timedelta(
        days=rng.randint(*_PAYMENT_LAG_DAYS), seconds=rng.randrange(_SECONDS_PER_DAY)
    )
    wants_split = rng.random() < float(config.split_payment_rate) or force_split
    split = allow_split and wants_split and invoice.total_minor >= 2 * _MIN_SPLIT_PART_MINOR
    if force_split or force_paid:
        # pull the first payment back so the whole (split) payment fits before period_end
        reserve = timedelta(days=_SPLIT_GAP_DAYS[1] + 1 if split else 1)
        first = max(issued_at + timedelta(seconds=1), min(first, period_end - reserve))
    if first >= period_end:
        return []
    parts = (
        _split_amount(rng, invoice.total_minor, rng.randint(2, 3), _MIN_SPLIT_PART_MINOR)
        if split
        else [invoice.total_minor]
    )
    payments: list[Payment] = []
    when = first
    for part in parts:
        if when >= period_end:
            break
        payments.append(
            Payment(
                doc_id=ids.next("DOC"),
                payment_id=ids.next("PAY"),
                timestamp=when.isoformat(),
                currency=invoice.currency,
                amount_minor=part,
                invoice_refs=(
                    InvoiceAllocation(invoice_id=invoice.invoice_id, allocated_minor=part),
                ),
            )
        )
        when = when + timedelta(days=rng.randint(*_SPLIT_GAP_DAYS))
    return payments


def _row(
    ids: _IdCounter,
    journal_id: str,
    timestamp: str,
    currency: CurrencyCode,
    account: AccountName,
    debit: int,
    credit: int,
    entity_id: str | None,
    invoice_id: str | None,
    payment_id: str | None,
) -> LedgerRow:
    return LedgerRow(
        row_id=ids.next("L", 6),
        timestamp=timestamp,
        account=account,
        debit_minor=debit,
        credit_minor=credit,
        currency=currency,
        entity_id=entity_id,
        invoice_id=invoice_id,
        payment_id=payment_id,
        journal_id=journal_id,
    )


def _ledger_pair(
    ids: _IdCounter,
    timestamp: str,
    currency: CurrencyCode,
    amount: int,
    debit_account: AccountName,
    credit_account: AccountName,
    entity_id: str | None,
    invoice_id: str | None,
    payment_id: str | None,
) -> list[LedgerRow]:
    """One balanced two-row journal."""
    journal_id = ids.next("J", 6)
    refs = (entity_id, invoice_id, payment_id)
    return [
        _row(ids, journal_id, timestamp, currency, debit_account, amount, 0, *refs),
        _row(ids, journal_id, timestamp, currency, credit_account, 0, amount, *refs),
    ]


def _generate_ledger_rows_for_invoice(
    invoice: Invoice, side: EntityKind, ids: _IdCounter, credit_note: bool = False
) -> list[LedgerRow]:
    """Invoice journal in the document currency; a credit note reverses the accounts."""
    debit, credit = ("EXPENSE", "AP") if side == "supplier" else ("AR", "REVENUE")
    if credit_note:
        debit, credit = credit, debit
    return _ledger_pair(
        ids, invoice.issued_at, invoice.currency, invoice.total_minor, debit, credit,
        invoice.entity_id, invoice.invoice_id, None,
    )  # fmt: skip


def _generate_ledger_rows_for_payment(
    payment: Payment,
    invoice: Invoice,
    side: EntityKind,
    policy: Policy,
    base_currency: CurrencyCode,
    fx_lookup: FxLookup,
    ids: _IdCounter,
) -> list[LedgerRow]:
    """Payment journal: document currency, or base currency at the policy-date rate."""
    if payment.currency == base_currency:
        amount, currency = payment.amount_minor, payment.currency
    else:
        day = policy_date(policy, invoice.issued_at, payment.timestamp)
        rate = fx_lookup(day, payment.currency, base_currency)
        amount, currency = convert_minor(payment.amount_minor, rate, policy), base_currency
    _require(amount >= 1, "converted payment amount must be positive")
    debit, credit = ("AP", "CASH") if side == "supplier" else ("CASH", "AR")
    return _ledger_pair(
        ids, payment.timestamp, currency, amount, debit, credit,
        invoice.entity_id, invoice.invoice_id, payment.payment_id,
    )  # fmt: skip


def _generate_misc_journals(
    rng: random.Random,
    base_currency: CurrencyCode,
    suppliers: list[str],
    customers: list[str],
    n_rows: int,
    start: datetime,
    end: datetime,
    ids: _IdCounter,
) -> list[LedgerRow]:
    """Document-less journals filling exactly ``n_rows`` (an odd count uses one 3-row journal).

    Expense journals may name a supplier, cash sales a customer (only entities that also have
    invoices, so the validator can check the reference).
    """
    if n_rows <= 0:
        return []
    _require(n_rows != 1, "a single misc row cannot balance")
    n_journals = n_rows // 2
    amounts = benford_amounts_minor(rng, n_journals)
    rows: list[LedgerRow] = []
    for i, amount in enumerate(amounts):
        when = _timestamp_between(rng, start, end).isoformat()
        expense = rng.random() < float(_MISC_EXPENSE_SHARE)
        debit, credit = ("EXPENSE", "CASH") if expense else ("CASH", "REVENUE")
        pool = suppliers if expense else customers
        tagged = bool(pool) and rng.random() < float(_MISC_TAGGED_SHARE)
        entity_id = rng.choice(pool) if tagged else None
        if i == 0 and n_rows % 2 == 1:
            journal_id = ids.next("J", 6)
            part_a, part_b = _split_amount(rng, amount, 2)
            refs = (entity_id, None, None)
            rows.append(_row(ids, journal_id, when, base_currency, debit, amount, 0, *refs))
            rows.append(_row(ids, journal_id, when, base_currency, credit, 0, part_a, *refs))
            rows.append(_row(ids, journal_id, when, base_currency, credit, 0, part_b, *refs))
        else:
            rows.extend(
                _ledger_pair(ids, when, base_currency, amount, debit, credit, entity_id, None, None)
            )
    _require(len(rows) == n_rows, "misc journals must fill the requested row count")
    return rows


# ----------------------------------------------------------------------------- decoys


class _DecoyContext(NamedTuple):
    """Everything a decoy maker needs; the dicts are the history's own registries (mutated)."""

    rng: random.Random
    company: CompanyProfile
    config: EpisodeConfig
    start: datetime
    end: datetime
    ids: _IdCounter
    issued_at: dict[str, datetime]
    side_of: dict[str, EntityKind]
    payments_of: dict[str, list[Payment]]
    credit_notes: set[str]  # invoice ids posted as reversals (membership only)


DecoyMaker = Callable[[_DecoyContext], tuple[list[Invoice], DecoyGroup]]


def _register_invoice(ctx: _DecoyContext, invoice: Invoice, entity: Entity, when: datetime) -> None:
    ctx.issued_at[invoice.invoice_id] = when
    ctx.side_of[invoice.invoice_id] = entity.kind
    ctx.payments_of[invoice.invoice_id] = []


def _decoy_window(ctx: _DecoyContext, tail_days: int) -> datetime:
    """A random issue instant leaving ``tail_days`` before period_end for follow-up documents."""
    latest = ctx.end - timedelta(days=tail_days)
    _require(latest > ctx.start, "period long enough for decoys with follow-up documents")
    return _timestamp_between(ctx.rng, ctx.start, latest)


def _line_items_key(items: tuple[LineItem, ...]) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(sorted((li.desc, li.qty, li.unit_minor, li.amount_minor) for li in items))


def _decoy_repeated_amounts(ctx: _DecoyContext) -> tuple[list[Invoice], DecoyGroup]:
    """Two payments of one amount on different dates, each settling a different invoice in full.

    Different counterparties of one currency when the company has them (then the duplicate
    checker never groups the invoices); otherwise the same counterparty, which also makes the
    pair a "similar invoices" decoy.
    """
    rng = ctx.rng
    by_currency: dict[str, list[Entity]] = {}
    for entity in ctx.company.entities:
        by_currency.setdefault(entity.default_currency, []).append(entity)
    rich = sorted(code for code, members in by_currency.items() if len(members) >= 2)
    currency = rng.choice(rich) if rich else rng.choice(sorted(by_currency))
    pool = by_currency[currency]
    entities = rng.sample(pool, k=2) if len(pool) >= 2 else [pool[0], pool[0]]
    total = max(benford_amount_minor(rng), 2 * _MIN_SPLIT_PART_MINOR)
    invoices: list[Invoice] = []
    payments: list[Payment] = []
    for entity in entities:
        when = _decoy_window(ctx, 2)
        invoice = _generate_single_invoice(rng, entity, when, total, None, ctx.ids)
        if invoices:  # never an exact twin of the first invoice, whoever the counterparty is
            invoice = _with_distinct_lines(rng, invoice, invoices[0], total)
        _register_invoice(ctx, invoice, entity, when)
        batch = _generate_payments_for_invoice(
            rng, invoice, when, ctx.end, ctx.config, ctx.ids,
            allow_split=False, force_split=False, force_paid=True,
        )  # fmt: skip
        _require(len(batch) == 1 and batch[0].amount_minor == total, "decoy paid in full once")
        if payments and batch[0].timestamp[:10] == payments[0].timestamp[:10]:
            batch = [_on_another_day(batch[0], when, ctx.end)]  # "different dates" means dates
        ctx.payments_of[invoice.invoice_id] = batch
        invoices.append(invoice)
        payments.append(batch[0])
    group = DecoyGroup(
        "repeated_amounts",
        tuple(i.invoice_id for i in invoices),
        tuple(p.payment_id for p in payments),
    )
    return invoices, group


def _with_distinct_lines(
    rng: random.Random, invoice: Invoice, twin: Invoice, total: int
) -> Invoice:
    """``invoice`` with line items that differ from ``twin``'s (re-drawn, bounded)."""
    for _ in range(_DECOY_LINE_ATTEMPTS):
        if _line_items_key(invoice.line_items) != _line_items_key(twin.line_items):
            return invoice
        invoice = invoice.model_copy(update={"line_items": _line_items(rng, total, None)})
    _require(
        _line_items_key(invoice.line_items) != _line_items_key(twin.line_items),
        "decoy twins differ in their line items",
    )
    return invoice


def _on_another_day(payment: Payment, issued: datetime, end: datetime) -> Payment:
    """The same payment one day later (or earlier, if later would leave the period)."""
    moment = datetime.fromisoformat(payment.timestamp)
    later = moment + timedelta(days=1)
    moved = later if later < end else moment - timedelta(days=1)
    _require(issued < moved < end, "repeated-amount payment stays inside the period")
    return payment.model_copy(update={"timestamp": moved.isoformat()})


def _decoy_split_payments(ctx: _DecoyContext) -> tuple[list[Invoice], DecoyGroup]:
    """One invoice settled by exactly two payments whose allocations sum to its total."""
    rng = ctx.rng
    entity = rng.choice(ctx.company.entities)
    total = max(benford_amount_minor(rng), 2 * _MIN_SPLIT_PART_MINOR)
    when = _decoy_window(ctx, _DECOY_TAIL_DAYS)
    invoice = _generate_single_invoice(rng, entity, when, total, None, ctx.ids)
    _register_invoice(ctx, invoice, entity, when)
    parts = _split_amount(rng, total, 2, _MIN_SPLIT_PART_MINOR)
    first = when + timedelta(
        days=rng.randint(*_DECOY_SPLIT_FIRST_LAG_DAYS), seconds=rng.randrange(_SECONDS_PER_DAY)
    )
    second = first + timedelta(days=rng.randint(*_SPLIT_GAP_DAYS))
    _require(second < ctx.end, "split decoy settles inside the period")
    payments = [
        Payment(
            doc_id=ctx.ids.next("DOC"),
            payment_id=ctx.ids.next("PAY"),
            timestamp=moment.isoformat(),
            currency=invoice.currency,
            amount_minor=part,
            invoice_refs=(InvoiceAllocation(invoice_id=invoice.invoice_id, allocated_minor=part),),
        )
        for moment, part in zip((first, second), parts, strict=True)
    ]
    _require(sum(p.amount_minor for p in payments) == total, "split decoy sums to the total")
    ctx.payments_of[invoice.invoice_id] = payments
    group = DecoyGroup(
        "split_payment", (invoice.invoice_id,), tuple(p.payment_id for p in payments)
    )
    return [invoice], group


def _decoy_refunds(ctx: _DecoyContext) -> tuple[list[Invoice], DecoyGroup]:
    """An invoice followed by a credit note of the same counterparty reversing part of it."""
    rng = ctx.rng
    entity = rng.choice(ctx.company.entities)
    total = max(benford_amount_minor(rng), 2 * _MIN_SPLIT_PART_MINOR)
    when = _decoy_window(ctx, _DECOY_TAIL_DAYS)
    original = _generate_single_invoice(rng, entity, when, total, None, ctx.ids)
    _register_invoice(ctx, original, entity, when)
    refunded = max(1, total * rng.choice(_DECOY_REFUND_PCT) // 100)
    note_when = when + timedelta(days=rng.randint(*_DECOY_REFUND_LAG_DAYS))
    _require(note_when < ctx.end, "credit note inside the period")
    desc = f"{CREDIT_NOTE_PREFIX}{original.line_items[0].desc}"
    note = Invoice(
        doc_id=ctx.ids.next("DOC"),
        invoice_id=ctx.ids.next("INV"),
        entity_id=entity.entity_id,
        issued_at=note_when.isoformat(),
        due_at=(note_when + timedelta(days=rng.choice(_DUE_TERMS_DAYS))).isoformat(),
        currency=entity.default_currency,
        total_minor=refunded,
        line_items=(LineItem(desc=desc, qty=1, unit_minor=refunded, amount_minor=refunded),),
    )
    _register_invoice(ctx, note, entity, note_when)
    ctx.credit_notes.add(note.invoice_id)
    group = DecoyGroup("refund", (original.invoice_id, note.invoice_id), ())
    return [original, note], group


def _decoy_similar_invoices(ctx: _DecoyContext) -> tuple[list[Invoice], DecoyGroup]:
    """Two invoices of one counterparty with one total but different lines and service periods."""
    rng = ctx.rng
    entity = rng.choice(ctx.company.entities)
    total = max(benford_amount_minor(rng), 2 * _MIN_SPLIT_PART_MINOR)
    first_when = _decoy_window(ctx, _DECOY_SIMILAR_GAP_DAYS[0] + 1)
    first = _generate_single_invoice(rng, entity, first_when, total, None, ctx.ids)
    _register_invoice(ctx, first, entity, first_when)
    latest_gap = min(_DECOY_SIMILAR_GAP_DAYS[1], (ctx.end - first_when).days - 1)
    gap = rng.randint(_DECOY_SIMILAR_GAP_DAYS[0], max(_DECOY_SIMILAR_GAP_DAYS[0], latest_gap))
    second_when = first_when + timedelta(days=gap)
    _require(second_when < ctx.end, "similar invoice inside the period")
    second = _generate_single_invoice(rng, entity, second_when, total, None, ctx.ids)
    second = _with_distinct_lines(rng, second, first, total)
    _register_invoice(ctx, second, entity, second_when)
    group = DecoyGroup("similar_invoices", (first.invoice_id, second.invoice_id), ())
    return [first, second], group


def _generate_decoys(ctx: _DecoyContext) -> tuple[list[Invoice], list[DecoyGroup]]:
    """All configured decoy groups, in a fixed pattern order."""
    counts = ctx.config.decoys
    plan: tuple[tuple[DecoyMaker, int], ...] = (
        (_decoy_repeated_amounts, counts.repeated_payment_amounts),
        (_decoy_split_payments, counts.split_payments),
        (_decoy_refunds, counts.refunds),
        (_decoy_similar_invoices, counts.similar_invoices),
    )
    invoices: list[Invoice] = []
    groups: list[DecoyGroup] = []
    for maker, count in plan:
        for _ in range(count):
            made, group = maker(ctx)
            invoices.extend(made)
            groups.append(group)
    return invoices, groups


def _fx_lookup(fx_rates: tuple[FXRate, ...], policy: Policy) -> FxLookup:
    """Closure over an FX index: (date, source, target) -> rate under the policy fallback."""
    index = build_fx_index(fx_rates)

    def lookup(day: str, source: CurrencyCode, target: CurrencyCode) -> Decimal:
        return rate_on(index, day, source, target, policy)

    return lookup


def _check_journals_balanced(ledger: list[LedgerRow]) -> None:
    sums: dict[tuple[str, str], list[int]] = {}
    for row in ledger:
        debit_credit = sums.setdefault((row.journal_id, row.currency), [0, 0])
        debit_credit[0] += row.debit_minor
        debit_credit[1] += row.credit_minor
    _require(all(d == c for d, c in sums.values()), "every journal balances per currency")


def _document_bounds(
    rng: random.Random, config: EpisodeConfig, n_rows_target: int
) -> tuple[int, int, int]:
    """(lo, hi, target) document counts: config range clipped to the row budget, biased upward."""
    docs_lo, docs_hi = config.n_documents_range
    hi = min(docs_hi, max(2, n_rows_target // _DOCS_PER_ROWS_DIVISOR))
    lo = min(docs_lo, hi)
    drawn = rng.randint(lo, hi)
    target = max(drawn, min(hi, n_rows_target // _DOCS_PER_ROWS_FLOOR_DIVISOR))
    return lo, hi, target


def generate_clean_history(
    company: CompanyProfile,
    fx_rates: tuple[FXRate, ...],
    rng: random.Random,
    config: EpisodeConfig,
) -> CleanHistory:
    """Phase 1 invoices (recurring + regular), phase 2 payments, phase 3 ledger + misc journals.

    Document and ledger-row counts land inside the configured ranges (documents are trimmed or
    topped up after phase 2; misc journals fill the ledger exactly). Deterministic given
    (company, fx_rates, rng, config); returned in canonical order (timestamp, id) with
    sequential ids that build.py permutes afterwards. ``recurring_series`` lists the invoice
    ids of each legitimate recurring series so injectors never pick a decoy as their source.
    """
    start = datetime.fromisoformat(company.period_start)
    end = datetime.fromisoformat(company.period_end)
    policy, base = company.policy, company.base_currency
    fx_lookup = _fx_lookup(fx_rates, policy)
    ids = _IdCounter()

    n_rows_target = rng.randint(*config.n_ledger_rows_range)
    docs_lo, docs_hi, n_docs_target = _document_bounds(rng, config, n_rows_target)
    n_invoices = max(1, (n_docs_target * 10) // _DOCS_PER_INVOICE_TENTHS)

    series_length = _recurring_series_length(start, end)
    suppliers = [e for e in company.entities if e.kind == "supplier"]
    n_recurring = 0
    if series_length and config.recurring_supplier_count and suppliers:
        at_least_one = 1 if n_invoices >= series_length else 0
        cap_share = max(at_least_one, n_invoices // _RECURRING_SHARE_DIVISOR)
        cap_docs = docs_hi // (2 * series_length)
        n_recurring = min(config.recurring_supplier_count, len(suppliers), cap_share, cap_docs)
    recurring_entities = rng.sample(suppliers, k=n_recurring)
    services = rng.sample(RECURRING_SERVICES, k=n_recurring)
    n_regular = max(0, n_invoices - n_recurring * series_length)
    totals = benford_amounts_minor(rng, n_regular + n_recurring)
    foreign_entities = [e for e in company.entities if e.default_currency != base]

    # Phase 1 - invoices
    recurring: list[Invoice] = []
    regular: list[Invoice] = []
    issued_at: dict[str, datetime] = {}
    side_of: dict[str, EntityKind] = {}
    recurring_series: list[tuple[str, ...]] = []
    recurring_plan = zip(recurring_entities, services, totals[:n_recurring], strict=True)
    for entity, service, total in recurring_plan:
        series = _generate_recurring_series(
            rng, entity, service, total, start, end, series_length, ids
        )
        recurring_series.append(tuple(invoice.invoice_id for invoice in series))
        for invoice in series:
            recurring.append(invoice)
            issued_at[invoice.invoice_id] = datetime.fromisoformat(invoice.issued_at)
            side_of[invoice.invoice_id] = entity.kind

    def new_regular(total: int) -> Invoice:
        # the first regular invoice always goes to a foreign-currency entity when one exists:
        # tests and the FX injectors need >= 1 foreign document per episode
        first = not regular and bool(foreign_entities)
        entity = rng.choice(foreign_entities if first else company.entities)
        when = _timestamp_between(rng, start, end)
        invoice = _generate_single_invoice(rng, entity, when, total, None, ids)
        regular.append(invoice)
        issued_at[invoice.invoice_id] = when
        side_of[invoice.invoice_id] = entity.kind
        return invoice

    for total in totals[n_recurring:]:
        new_regular(total)

    # Phase 2 - payments (one allocation each; split = several payments for one invoice)
    payments_of: dict[str, list[Payment]] = {}
    need_split = config.split_payment_rate > 0
    for invoice in recurring:
        payments_of[invoice.invoice_id] = _generate_payments_for_invoice(
            rng, invoice, issued_at[invoice.invoice_id], end, config, ids,
            allow_split=False, force_split=False, force_paid=False,
        )  # fmt: skip
    for invoice in regular:
        batch = _generate_payments_for_invoice(
            rng, invoice, issued_at[invoice.invoice_id], end, config, ids,
            allow_split=True, force_split=need_split, force_paid=False,
        )  # fmt: skip
        if len(batch) >= 2:
            need_split = False
        payments_of[invoice.invoice_id] = batch

    # Phase 2b - decoys (legitimate patterns that look like errors); their own sub-RNG is
    # derived only when asked for, so a config without decoys reproduces the history exactly
    decoy_invoices: list[Invoice] = []
    decoy_groups: list[DecoyGroup] = []
    credit_notes: set[str] = set()
    if config.decoys.total():
        context = _DecoyContext(
            derive_sub_rng(rng, "decoys"), company, config, start, end, ids,
            issued_at, side_of, payments_of, credit_notes,
        )  # fmt: skip
        decoy_invoices, decoy_groups = _generate_decoys(context)

    def n_docs() -> int:
        documents = len(recurring) + len(regular) + len(decoy_invoices)
        return documents + sum(len(p) for p in payments_of.values())

    # keep the document count inside [docs_lo, docs_hi]: trim regular invoices from the end,
    # then top up with exactly-sized additions (invoice + one payment = 2, invoice alone = 1)
    while n_docs() > docs_hi and regular:
        dropped = regular.pop()
        del payments_of[dropped.invoice_id]
    if n_docs() > docs_hi:
        raise ValueError("decoys exceed the document budget of this configuration")
    while n_docs() < docs_lo:
        with_payment = docs_lo - n_docs() >= 2  # one missing document -> invoice alone
        invoice = new_regular(benford_amount_minor(rng))
        batch: list[Payment] = []
        if with_payment:
            batch = _generate_payments_for_invoice(
                rng, invoice, issued_at[invoice.invoice_id], end, config, ids,
                allow_split=False, force_split=False, force_paid=True,
            )  # fmt: skip
        payments_of[invoice.invoice_id] = batch
    _require(docs_lo <= n_docs() <= docs_hi, "document count inside the configured range")
    invoices = [*recurring, *regular, *decoy_invoices]
    payments = [p for invoice in invoices for p in payments_of[invoice.invoice_id]]
    eligible_for_split = [
        i for i in regular
        if i.total_minor >= 2 * _MIN_SPLIT_PART_MINOR
        and issued_at[i.invoice_id] + timedelta(days=_SPLIT_GAP_DAYS[1] + 2) < end
    ]  # fmt: skip
    _require(not (need_split and eligible_for_split), "a split payment exists when possible")

    # Phase 3 - ledger
    ledger: list[LedgerRow] = []
    for invoice in invoices:
        ledger.extend(
            _generate_ledger_rows_for_invoice(
                invoice, side_of[invoice.invoice_id], ids, invoice.invoice_id in credit_notes
            )
        )
    invoice_by_id = {invoice.invoice_id: invoice for invoice in invoices}
    for payment in payments:
        invoice = invoice_by_id[payment.invoice_refs[0].invoice_id]
        ledger.extend(
            _generate_ledger_rows_for_payment(
                payment, invoice, side_of[invoice.invoice_id], policy, base, fx_lookup, ids
            )
        )
    n_misc = n_rows_target - len(ledger)
    _require(n_misc >= 0 and n_misc != 1, "document journals fit the ledger budget")
    invoiced_suppliers = sorted(
        {i.entity_id for i in invoices if side_of[i.invoice_id] == "supplier"}
    )
    invoiced_customers = sorted(
        {i.entity_id for i in invoices if side_of[i.invoice_id] == "customer"}
    )
    ledger.extend(
        _generate_misc_journals(
            rng, base, invoiced_suppliers, invoiced_customers, n_misc, start, end, ids
        )
    )

    _check_journals_balanced(ledger)
    _require(
        all(start <= datetime.fromisoformat(row.timestamp) < end for row in ledger),
        "all ledger timestamps inside [period_start, period_end)",
    )
    invoices.sort(key=lambda i: (i.issued_at, i.invoice_id))
    payments.sort(key=lambda p: (p.timestamp, p.payment_id))
    ledger.sort(key=lambda r: (r.timestamp, r.row_id))
    return CleanHistory(
        tuple(invoices),
        tuple(payments),
        tuple(ledger),
        tuple(recurring_series),
        tuple(decoy_groups),
    )


__all__ = ["CleanHistory", "DecoyGroup", "FxLookup", "generate_clean_history"]
