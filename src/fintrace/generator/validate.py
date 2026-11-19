"""Consistency checks for a clean episode: five required checks, a Benford signal, an orchestrator.

Every check raises ``ConsistencyError(check_name, detail, items)`` with sorted, bounded item
lists; nothing here mutates its inputs. Benford is a signal (returned), not a contract.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Literal

from fintrace.models import (
    CurrencyCode,
    FXRate,
    Invoice,
    LedgerRow,
    Manifest,
    Payment,
    Policy,
    PrivateEpisode,
    PublicEpisode,
)

from .benford import chi_squared_benford
from .fx import FxIndex, build_fx_index, convert_minor, policy_date, rate_on

Side = Literal["supplier", "customer"]
_MAX_ITEMS = 20


class ConsistencyError(Exception):
    """A clean-history invariant does not hold; ``check_name`` identifies the failing check."""

    def __init__(self, check_name: str, detail: str, items: list[str] | None = None) -> None:
        self.check_name = check_name
        self.detail = detail
        self.items = sorted(items or [])
        shown = f" items={self.items[:_MAX_ITEMS]}" if self.items else ""
        super().__init__(f"[{check_name}] {detail}{shown}")


def check_balanced(ledger: Sequence[LedgerRow]) -> None:
    """Per journal_id and currency: sum(debit) == sum(credit), >= 2 rows, one timestamp."""
    sums: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    rows_per_journal: dict[str, int] = defaultdict(int)
    timestamp_of: dict[str, str] = {}
    mixed: list[str] = []
    for row in ledger:
        debit_credit = sums[(row.journal_id, row.currency)]
        debit_credit[0] += row.debit_minor
        debit_credit[1] += row.credit_minor
        rows_per_journal[row.journal_id] += 1
        if timestamp_of.setdefault(row.journal_id, row.timestamp) != row.timestamp:
            mixed.append(row.journal_id)
    unbalanced = [f"{j}/{c}" for (j, c), (d, cr) in sums.items() if d != cr]
    if unbalanced:
        raise ConsistencyError("balanced", "debit != credit within a journal/currency", unbalanced)
    singles = [j for j, n in rows_per_journal.items() if n < 2]
    if singles:
        raise ConsistencyError("balanced", "journal with fewer than two rows", singles)
    if mixed:
        raise ConsistencyError("balanced", "journal rows with different timestamps", mixed)


def check_invoice_totals(invoices: Sequence[Invoice]) -> None:
    """total_minor == sum(line amounts), each line amount == qty * unit, at least one line."""
    bad: list[str] = []
    for invoice in invoices:
        lines = invoice.line_items
        if not lines or invoice.total_minor != sum(line.amount_minor for line in lines):
            bad.append(invoice.invoice_id)
        elif any(line.amount_minor != line.qty * line.unit_minor for line in lines):
            bad.append(invoice.invoice_id)
    if bad:
        raise ConsistencyError("invoice_totals", "invoice total or line arithmetic mismatch", bad)


def check_payment_allocation(payments: Sequence[Payment], invoices: Sequence[Invoice]) -> None:
    """Allocations hit existing same-currency invoices, sum to the payment, never overpay."""
    invoice_by_id = {invoice.invoice_id: invoice for invoice in invoices}
    allocated: dict[str, int] = defaultdict(int)
    bad: list[str] = []
    for payment in payments:
        refs = payment.invoice_refs
        if not refs or sum(ref.allocated_minor for ref in refs) != payment.amount_minor:
            bad.append(payment.payment_id)
            continue
        for ref in refs:
            invoice = invoice_by_id.get(ref.invoice_id)
            if invoice is None or invoice.currency != payment.currency or ref.allocated_minor < 1:
                bad.append(payment.payment_id)
                break
            allocated[ref.invoice_id] += ref.allocated_minor
    if bad:
        raise ConsistencyError(
            "payment_allocation", "allocation does not match payment/invoice", bad
        )
    overpaid = [i for i, total in allocated.items() if total > invoice_by_id[i].total_minor]
    if overpaid:
        raise ConsistencyError("payment_allocation", "allocations exceed invoice total", overpaid)


def _two_rows(rows: list[LedgerRow]) -> tuple[LedgerRow, LedgerRow] | None:
    """Return (debit_row, credit_row) for a well-formed two-row journal, else None."""
    if len(rows) != 2 or rows[0].journal_id != rows[1].journal_id:
        return None
    debit = [r for r in rows if r.debit_minor > 0]
    credit = [r for r in rows if r.credit_minor > 0]
    if len(debit) != 1 or len(credit) != 1:
        return None
    return debit[0], credit[0]


def _invoice_side(invoice: Invoice, rows: list[LedgerRow]) -> Side | None:
    """Side implied by a correct invoice journal, or None if the journal is wrong."""
    pair = _two_rows(rows)
    if pair is None:
        return None
    debit, credit = pair
    for row in pair:
        if (
            row.currency != invoice.currency
            or row.timestamp != invoice.issued_at
            or row.entity_id != invoice.entity_id
            or row.payment_id is not None
        ):
            return None
    if debit.debit_minor != invoice.total_minor or credit.credit_minor != invoice.total_minor:
        return None
    accounts = (debit.account, credit.account)
    if accounts == ("EXPENSE", "AP"):
        return "supplier"
    if accounts == ("AR", "REVENUE"):
        return "customer"
    return None


def _payment_rows_ok(
    payment: Payment,
    rows: list[LedgerRow],
    invoice: Invoice,
    side: Side,
    policy: Policy,
    base_currency: CurrencyCode,
    fx_index: FxIndex,
) -> bool:
    """True if the payment journal matches the posting rule for ``side``."""
    if len(payment.invoice_refs) != 1:
        return False
    pair = _two_rows(rows)
    if pair is None:
        return False
    if payment.currency == base_currency:
        amount, currency = payment.amount_minor, payment.currency
    else:
        day = policy_date(policy, invoice.issued_at, payment.timestamp)
        try:
            rate = rate_on(fx_index, day, payment.currency, base_currency, policy)
        except LookupError:
            return False
        amount, currency = convert_minor(payment.amount_minor, rate, policy), base_currency
    debit, credit = pair
    expected = ("AP", "CASH") if side == "supplier" else ("CASH", "AR")
    return all(
        row.currency == currency
        and row.timestamp == payment.timestamp
        and row.invoice_id == invoice.invoice_id
        and row.entity_id == invoice.entity_id
        for row in pair
    ) and (
        debit.debit_minor == amount
        and credit.credit_minor == amount
        and (debit.account, credit.account) == expected
    )


def check_ledger_matches_documents(
    ledger: Sequence[LedgerRow],
    invoices: Sequence[Invoice],
    payments: Sequence[Payment],
    policy: Policy,
    fx_rates: Sequence[FXRate],
    base_currency: CurrencyCode,
) -> None:
    """Every document has exactly its journal; every referenced document exists; AP/AR need docs."""
    invoice_by_id = {invoice.invoice_id: invoice for invoice in invoices}
    payment_by_id = {payment.payment_id: payment for payment in payments}
    rows_by_invoice: dict[str, list[LedgerRow]] = defaultdict(list)
    rows_by_payment: dict[str, list[LedgerRow]] = defaultdict(list)
    orphans: list[str] = []
    undocumented: list[str] = []
    for row in ledger:
        if row.payment_id is not None:
            if row.payment_id in payment_by_id and row.invoice_id in invoice_by_id:
                rows_by_payment[row.payment_id].append(row)
            else:
                orphans.append(row.row_id)
        elif row.invoice_id is not None:
            if row.invoice_id in invoice_by_id:
                rows_by_invoice[row.invoice_id].append(row)
            else:
                orphans.append(row.row_id)
        elif row.account in ("AP", "AR"):
            undocumented.append(row.row_id)
    name = "ledger_matches_documents"
    if orphans:
        raise ConsistencyError(name, "ledger rows reference unknown documents", orphans)
    if undocumented:
        raise ConsistencyError(name, "AP/AR movement without a document", undocumented)
    side_of: dict[str, Side] = {}
    bad_invoices: list[str] = []
    for invoice in invoices:
        side = _invoice_side(invoice, rows_by_invoice.get(invoice.invoice_id, []))
        if side is None:
            bad_invoices.append(invoice.invoice_id)
        else:
            side_of[invoice.invoice_id] = side
    if bad_invoices:
        raise ConsistencyError(name, "invoice journal missing or malformed", bad_invoices)
    fx_index = build_fx_index(fx_rates)
    bad_payments: list[str] = []
    for payment in payments:
        first_ref = payment.invoice_refs[0].invoice_id if payment.invoice_refs else None
        invoice = invoice_by_id.get(first_ref) if first_ref is not None else None
        rows = rows_by_payment.get(payment.payment_id, [])
        if invoice is None or not _payment_rows_ok(
            payment, rows, invoice, side_of[invoice.invoice_id], policy, base_currency, fx_index
        ):
            bad_payments.append(payment.payment_id)
    if bad_payments:
        raise ConsistencyError(name, "payment journal missing or malformed", bad_payments)


def check_period(ledger: Sequence[LedgerRow], manifest: Manifest) -> None:
    """Every ledger timestamp lies in [period_start, period_end)."""
    start = datetime.fromisoformat(manifest.period_start)
    end = datetime.fromisoformat(manifest.period_end)
    outside = [r.row_id for r in ledger if not start <= datetime.fromisoformat(r.timestamp) < end]
    if outside:
        raise ConsistencyError("period", "ledger rows outside the reporting period", outside)


def check_benford(
    ledger: Sequence[LedgerRow],
    invoices: Sequence[Invoice],
    threshold: Decimal | None = None,
) -> Decimal:
    """Chi-squared of first digits, one amount per transaction; raises only if a threshold is given.

    A transaction is a journal (its debit total: debit and credit rows carry the same amount, so
    counting rows would double every observation) plus any invoice that has no journal.
    """
    debit_by_journal: dict[str, int] = defaultdict(int)
    journaled_invoices: set[str] = set()  # membership only
    for row in ledger:
        debit_by_journal[row.journal_id] += row.debit_minor
        if row.invoice_id is not None and row.payment_id is None:
            journaled_invoices.add(row.invoice_id)
    amounts = list(debit_by_journal.values())
    amounts.extend(i.total_minor for i in invoices if i.invoice_id not in journaled_invoices)
    chi = chi_squared_benford(amounts)
    if threshold is not None and chi > threshold:
        raise ConsistencyError("benford", f"chi-squared {chi} exceeds threshold {threshold}")
    return chi


def validate_clean_history(ep: PublicEpisode | PrivateEpisode) -> None:
    """Run the five required checks (loud) plus the Benford signal; raise ConsistencyError."""
    public = ep.public if isinstance(ep, PrivateEpisode) else ep
    manifest = public.manifest
    n_documents = len(public.invoices) + len(public.payments)
    if manifest.n_ledger_rows != len(public.ledger) or manifest.n_documents != n_documents:
        raise ConsistencyError("manifest", "manifest counts do not match the episode content")
    check_balanced(public.ledger)
    check_invoice_totals(public.invoices)
    check_payment_allocation(public.payments, public.invoices)
    check_ledger_matches_documents(
        public.ledger,
        public.invoices,
        public.payments,
        public.policy,
        public.fx,
        manifest.base_currency,
    )
    check_period(public.ledger, manifest)
    check_benford(public.ledger, public.invoices)


__all__ = [
    "ConsistencyError",
    "check_balanced",
    "check_benford",
    "check_invoice_totals",
    "check_ledger_matches_documents",
    "check_payment_allocation",
    "check_period",
    "validate_clean_history",
]
