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

from .benford import benford_amount_minor, benford_amounts_minor
from .company import LINE_ITEM_DESCRIPTIONS, RECURRING_SERVICES, CompanyProfile, Entity, EntityKind
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

FxLookup = Callable[[str, CurrencyCode, CurrencyCode], Decimal]


class CleanHistory(NamedTuple):
    """Canonical documents and ledger of a clean episode plus the recurring-decoy series ids."""

    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]
    ledger: tuple[LedgerRow, ...]
    recurring_series: tuple[tuple[str, ...], ...]  # invoice ids per legitimate recurring series


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
    invoice: Invoice, side: EntityKind, ids: _IdCounter
) -> list[LedgerRow]:
    """Invoice journal in the document currency."""
    debit, credit = ("EXPENSE", "AP") if side == "supplier" else ("AR", "REVENUE")
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

    def n_docs() -> int:
        return len(recurring) + len(regular) + sum(len(p) for p in payments_of.values())

    # keep the document count inside [docs_lo, docs_hi]: trim regular invoices from the end,
    # then top up with exactly-sized additions (invoice + one payment = 2, invoice alone = 1)
    while n_docs() > docs_hi and regular:
        dropped = regular.pop()
        del payments_of[dropped.invoice_id]
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
    invoices = [*recurring, *regular]
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
        ledger.extend(_generate_ledger_rows_for_invoice(invoice, side_of[invoice.invoice_id], ids))
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
    return CleanHistory(tuple(invoices), tuple(payments), tuple(ledger), tuple(recurring_series))


__all__ = ["CleanHistory", "FxLookup", "generate_clean_history"]
