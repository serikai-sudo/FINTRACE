"""Consistency checks for a clean episode: required checks, a Benford signal, an orchestrator.

Every check raises ``ConsistencyError(check_name, detail, items)`` with sorted, bounded item
lists; nothing here mutates its inputs. Benford is a signal (returned), not a contract.
Clean-history contract enforced here (injectors relax specific points on Day 2): one allocation
per payment, payments never dated before their invoice, misc journals in base currency on
EXPENSE/CASH/REVENUE only, FX curves complete over the period for every document currency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Final, Literal

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
_MAX_ITEMS: Final[int] = 20
_MISC_ACCOUNTS: Final[frozenset[str]] = frozenset({"EXPENSE", "CASH", "REVENUE"})  # membership


class ConsistencyError(Exception):
    """A clean-history invariant does not hold; ``check_name`` identifies the failing check."""

    def __init__(self, check_name: str, detail: str, items: list[str] | None = None) -> None:
        self.check_name = check_name
        self.detail = detail
        self.items = sorted(items or [])
        shown = f" items={self.items[:_MAX_ITEMS]}" if self.items else ""
        super().__init__(f"[{check_name}] {detail}{shown}")


def _duplicates(values: Iterable[str]) -> list[str]:
    return [value for value, n in Counter(values).items() if n > 1]


def check_unique_ids(
    ledger: Sequence[LedgerRow], invoices: Sequence[Invoice], payments: Sequence[Payment]
) -> None:
    """row_id, invoice_id, payment_id and doc_id (across both document types) are unique."""
    families = (
        ("row_id", _duplicates(r.row_id for r in ledger)),
        ("invoice_id", _duplicates(i.invoice_id for i in invoices)),
        ("payment_id", _duplicates(p.payment_id for p in payments)),
        ("doc_id", _duplicates([*(i.doc_id for i in invoices), *(p.doc_id for p in payments)])),
    )
    for label, duplicates in families:
        if duplicates:
            raise ConsistencyError("unique_ids", f"duplicate {label}", duplicates)


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
    """total == sum(line amounts), each line amount == qty * unit, >= 1 line, due after issue."""
    bad: list[str] = []
    late_due: list[str] = []
    for invoice in invoices:
        lines = invoice.line_items
        if not lines or invoice.total_minor != sum(line.amount_minor for line in lines):
            bad.append(invoice.invoice_id)
        elif any(line.amount_minor != line.qty * line.unit_minor for line in lines):
            bad.append(invoice.invoice_id)
        if datetime.fromisoformat(invoice.due_at) <= datetime.fromisoformat(invoice.issued_at):
            late_due.append(invoice.invoice_id)
    if bad:
        raise ConsistencyError("invoice_totals", "invoice total or line arithmetic mismatch", bad)
    if late_due:
        raise ConsistencyError("invoice_totals", "due_at must be after issued_at", late_due)


def check_payment_allocation(
    payments: Sequence[Payment],
    invoices: Sequence[Invoice],
    ignore_ids: frozenset[str] = frozenset(),
) -> None:
    """One allocation per payment hitting an existing same-currency invoice, dated after it;
    allocations equal the payment amount and never overpay the invoice.

    Payments in ``ignore_ids`` carry a deliberately injected allocation error that the injector
    verifies on its own (``independent_check``). For them only the two rules such an error
    violates are relaxed: the single allocation equal to the amount, and the overpayment
    accounting. Every allocation they still carry must hit an existing same-currency invoice
    issued no later than the payment, with a positive amount.
    """
    invoice_by_id = {invoice.invoice_id: invoice for invoice in invoices}
    allocated: dict[str, int] = defaultdict(int)
    bad: list[str] = []
    for payment in payments:
        refs = payment.invoice_refs
        relaxed = payment.payment_id in ignore_ids
        if not relaxed and (len(refs) != 1 or refs[0].allocated_minor != payment.amount_minor):
            bad.append(payment.payment_id)
            continue
        paid_at = datetime.fromisoformat(payment.timestamp)
        for ref in refs:
            invoice = invoice_by_id.get(ref.invoice_id)
            if (
                invoice is None
                or invoice.currency != payment.currency
                or ref.allocated_minor < 1
                or paid_at < datetime.fromisoformat(invoice.issued_at)
            ):
                bad.append(payment.payment_id)
                break
        else:
            if not relaxed:
                for ref in refs:
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


def _invoice_side(
    invoice: Invoice, rows: list[LedgerRow], relax_timestamp: bool = False
) -> Side | None:
    """Side implied by a correct invoice journal, or None if the journal is wrong.

    ``relax_timestamp`` (an injected period error on this journal) drops only the rule that the
    posting carries the invoice date; currency, entity, amounts and accounts stay enforced.
    """
    pair = _two_rows(rows)
    if pair is None:
        return None
    debit, credit = pair
    for row in pair:
        if (
            row.currency != invoice.currency
            or (not relax_timestamp and row.timestamp != invoice.issued_at)
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
    check_amount: bool = True,
) -> bool:
    """True if the payment journal matches the posting rule for ``side``.

    With ``check_amount`` False (an injected FX error on this journal) the structure is still
    required: two rows, the right accounts, base currency, timestamp, invoice and entity; only the
    converted amount is left unchecked.
    """
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
    structure_ok = all(
        row.currency == currency
        and row.timestamp == payment.timestamp
        and row.invoice_id == invoice.invoice_id
        and row.entity_id == invoice.entity_id
        for row in pair
    ) and (debit.account, credit.account) == expected
    if not check_amount:
        return structure_ok
    return structure_ok and debit.debit_minor == amount and credit.credit_minor == amount


def check_ledger_matches_documents(
    ledger: Sequence[LedgerRow],
    invoices: Sequence[Invoice],
    payments: Sequence[Payment],
    policy: Policy,
    fx_rates: Sequence[FXRate],
    base_currency: CurrencyCode,
    ignore_ids: frozenset[str] = frozenset(),
) -> None:
    """Every document has exactly its journal; referenced documents exist; misc rows are tame.

    Misc (document-less) rows: never AP/AR, base currency only, EXPENSE/CASH/REVENUE only, and
    an entity_id, if present, belongs to a counterparty that has an invoice in the episode.
    The journal check is skipped for payments in ``ignore_ids`` (injected allocation errors);
    their ledger rows still have to reference existing documents. A payment journal whose rows
    are in ``ignore_ids`` (injected FX error) keeps its structural check but not the amount one.
    """
    invoice_by_id = {invoice.invoice_id: invoice for invoice in invoices}
    payment_by_id = {payment.payment_id: payment for payment in payments}
    known_entities = {invoice.entity_id for invoice in invoices}  # membership only
    rows_by_invoice: dict[str, list[LedgerRow]] = defaultdict(list)
    rows_by_payment: dict[str, list[LedgerRow]] = defaultdict(list)
    orphans: list[str] = []
    undocumented: list[str] = []
    bad_misc: list[str] = []
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
        elif (
            row.currency != base_currency
            or row.account not in _MISC_ACCOUNTS
            or (row.entity_id is not None and row.entity_id not in known_entities)
        ):
            bad_misc.append(row.row_id)
    name = "ledger_matches_documents"
    if orphans:
        raise ConsistencyError(name, "ledger rows reference unknown documents", orphans)
    if undocumented:
        raise ConsistencyError(name, "AP/AR movement without a document", undocumented)
    if bad_misc:
        raise ConsistencyError(name, "misc row: wrong currency, account or entity", bad_misc)
    side_of: dict[str, Side] = {}
    bad_invoices: list[str] = []
    for invoice in invoices:
        rows = rows_by_invoice.get(invoice.invoice_id, [])
        relax_timestamp = bool(rows) and all(row.row_id in ignore_ids for row in rows)
        side = _invoice_side(invoice, rows, relax_timestamp)
        if side is None:
            bad_invoices.append(invoice.invoice_id)
        else:
            side_of[invoice.invoice_id] = side
    if bad_invoices:
        raise ConsistencyError(name, "invoice journal missing or malformed", bad_invoices)
    fx_index = build_fx_index(fx_rates)
    bad_payments: list[str] = []
    for payment in payments:
        if payment.payment_id in ignore_ids:
            continue
        first_ref = payment.invoice_refs[0].invoice_id if payment.invoice_refs else None
        invoice = invoice_by_id.get(first_ref) if first_ref is not None else None
        rows = rows_by_payment.get(payment.payment_id, [])
        # relaxed only when the whole journal was rewritten (an FX error touches both rows)
        check_amount = not (rows and all(row.row_id in ignore_ids for row in rows))
        if invoice is None or not _payment_rows_ok(
            payment,
            rows,
            invoice,
            side_of[invoice.invoice_id],
            policy,
            base_currency,
            fx_index,
            check_amount,
        ):
            bad_payments.append(payment.payment_id)
    if bad_payments:
        raise ConsistencyError(name, "payment journal missing or malformed", bad_payments)


def check_period(
    ledger: Sequence[LedgerRow], manifest: Manifest, ignore_ids: frozenset[str] = frozenset()
) -> None:
    """Every ledger timestamp lies in [period_start, period_end); relaxed rows are exempt."""
    start = datetime.fromisoformat(manifest.period_start)
    end = datetime.fromisoformat(manifest.period_end)
    outside = [
        r.row_id
        for r in ledger
        if r.row_id not in ignore_ids and not start <= datetime.fromisoformat(r.timestamp) < end
    ]
    if outside:
        raise ConsistencyError("period", "ledger rows outside the reporting period", outside)


def check_fx_coverage(
    fx_rates: Sequence[FXRate],
    manifest: Manifest,
    invoices: Sequence[Invoice],
    payments: Sequence[Payment],
) -> None:
    """Every document currency other than base has a rate to base on every day of the period."""
    base = manifest.base_currency
    needed = sorted({d.currency for d in (*invoices, *payments)} - {base})
    if not needed:
        return
    start = date.fromisoformat(manifest.period_start[:10])
    end = date.fromisoformat(manifest.period_end[:10])
    days = [(start + timedelta(days=k)).isoformat() for k in range((end - start).days + 1)]
    have: dict[tuple[str, str], set[str]] = defaultdict(set)  # membership only
    for fx in fx_rates:
        have[(fx.source_currency, fx.target_currency)].add(fx.date)
    missing = [
        f"{currency}->{base}@{day}"
        for currency in needed
        for day in days
        if day not in have.get((currency, base), ())
    ]
    if missing:
        raise ConsistencyError("fx_coverage", "FX curve incomplete over the period", missing)


def check_benford(
    ledger: Sequence[LedgerRow],
    invoices: Sequence[Invoice],
    threshold: Decimal | None = None,
) -> Decimal:
    """Chi-squared of first digits, one amount per transaction; raises only if a threshold is given.

    A transaction is a journal (its debit total: debit and credit rows carry the same amount, so
    counting rows would double every observation) plus any invoice that has no journal. Over
    default-config seeds this statistic sits far below the 5 % critical value for the large
    majority of episodes; small episodes with many repeated amounts are the noisiest.
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


def validate_clean_history(
    ep: PublicEpisode | PrivateEpisode, *, ignore_ids: frozenset[str] = frozenset()
) -> None:
    """Run every required check (loud) plus the Benford signal; raise ConsistencyError.

    Order: manifest counts, unique_ids, balanced, invoice_totals, payment_allocation,
    ledger_matches_documents, period, fx_coverage, benford (signal only). ``ignore_ids`` names
    the records an injector deliberately made inconsistent: for a document only the allocation
    and journal checks are skipped, for ledger rows only the amount, posting-date and period
    rules; everything else is still enforced.
    """
    public = ep.public if isinstance(ep, PrivateEpisode) else ep
    manifest = public.manifest
    n_documents = len(public.invoices) + len(public.payments)
    if manifest.n_ledger_rows != len(public.ledger) or manifest.n_documents != n_documents:
        raise ConsistencyError("manifest", "manifest counts do not match the episode content")
    check_unique_ids(public.ledger, public.invoices, public.payments)
    check_balanced(public.ledger)
    check_invoice_totals(public.invoices)
    check_payment_allocation(public.payments, public.invoices, ignore_ids)
    check_ledger_matches_documents(
        public.ledger,
        public.invoices,
        public.payments,
        public.policy,
        public.fx,
        manifest.base_currency,
        ignore_ids,
    )
    check_period(public.ledger, manifest, ignore_ids)
    check_fx_coverage(public.fx, manifest, public.invoices, public.payments)
    check_benford(public.ledger, public.invoices)


__all__ = [
    "ConsistencyError",
    "check_balanced",
    "check_benford",
    "check_fx_coverage",
    "check_invoice_totals",
    "check_ledger_matches_documents",
    "check_payment_allocation",
    "check_period",
    "check_unique_ids",
    "validate_clean_history",
]
