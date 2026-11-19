"""Duplicate invoice injector (blueprint 5.1).

Picks a clean, non-recurring invoice INV-A and adds INV-B with new invoice/doc ids, the same
entity, total, currency and line items, issued 0..10 days later (inside the period). Posts the
second journal (EXPENSE/AP or AR/REVENUE) and, for 30 % of paid originals, a payment of INV-B with
journal (double-payment variant). The overstatement equals INV-B's total.
Legitimate recurring series (rent-like decoys) are never chosen: they are excluded both through
the provenance markers written by the generator and through a public-data heuristic.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import Final

from fintrace.constants import INJECTOR_DUPLICATE_DATE_SHIFT_MAX_DAYS
from fintrace.generator.company import EntityKind
from fintrace.generator.fx import build_fx_index, convert_minor, policy_date, rate_on
from fintrace.generator.validate import ConsistencyError
from fintrace.models import (
    AccountName,
    InjectorParams,
    Invoice,
    InvoiceAllocation,
    Issue,
    IssueType,
    LedgerRow,
    LineItem,
    Payment,
    PrivateEpisode,
    Provenance,
    ProvenanceEntry,
    PublicEpisode,
)

from .base import (
    InjectorError,
    derive_issue_id,
    fresh_label,
    invoice_side,
    shift_timestamp,
    touched_ids,
    used_labels,
)

issue_type: Final[IssueType] = "duplicate_invoice"

_DOUBLE_PAYMENT_RATE: Final[Decimal] = Decimal("0.3")
_PAYMENT_LAG_DAYS: Final[tuple[int, int]] = (1, 20)
_SECONDS_PER_DAY: Final[int] = 86_400
_RECURRING_GAP_DAYS: Final[tuple[int, int]] = (25, 35)
_RECURRING_DESC_RE: Final[re.Pattern[str]] = re.compile(r" - [0-9]{4}-[0-9]{2}$")

GroupKey = tuple[str, int, str]  # (entity_id, total_minor, currency)
LineItemsKey = tuple[tuple[str, int, int, int], ...]


def _line_items_key(items: tuple[LineItem, ...]) -> LineItemsKey:
    """Canonical, order-independent key of an invoice's line items."""
    return tuple(sorted((li.desc, li.qty, li.unit_minor, li.amount_minor) for li in items))


def _group_key(invoice: Invoice) -> GroupKey:
    return (invoice.entity_id, invoice.total_minor, invoice.currency)


def _groups(invoices: tuple[Invoice, ...]) -> dict[GroupKey, list[Invoice]]:
    """Invoices sharing entity, total and currency, in canonical invoice order."""
    groups: dict[GroupKey, list[Invoice]] = defaultdict(list)
    for invoice in invoices:
        groups[_group_key(invoice)].append(invoice)
    return groups


def _is_duplicate_pattern(group: list[Invoice]) -> bool:
    """Two or more invoices with identical economics and line items, not a monthly series."""
    if len(group) < 2:
        return False
    same_lines = len({_line_items_key(invoice.line_items) for invoice in group}) == 1
    return same_lines and not _looks_recurring(group)


def _looks_recurring(invoices: list[Invoice]) -> bool:
    """Rent-like series: '<service> - YYYY-MM' descriptions, distinct months, ~30-day spacing."""
    if len(invoices) < 2:
        return False
    descs = [invoice.line_items[0].desc for invoice in invoices]
    if not all(_RECURRING_DESC_RE.search(desc) for desc in descs) or len(set(descs)) != len(descs):
        return False
    issued = sorted(datetime.fromisoformat(invoice.issued_at) for invoice in invoices)
    lo, hi = _RECURRING_GAP_DAYS
    return all(lo <= (later - earlier).days <= hi for earlier, later in pairwise(issued))


def _recurring_invoice_ids(provenance: Provenance) -> set[str]:
    """Invoice ids the generator marked as legitimate recurring series (membership only)."""
    ids: set[str] = set()
    for entry in provenance.history:
        if entry.get("op") == "generate_recurring_series":
            invoice_ids = entry.get("invoice_ids")
            if isinstance(invoice_ids, list):
                ids.update(str(invoice_id) for invoice_id in invoice_ids)
    return ids


def _eligible_invoices(episode: PrivateEpisode) -> list[Invoice]:
    """Clean candidates: not recurring, not part of an earlier issue, unique in their group.

    Recurring decoys are excluded twice: through the generator's provenance markers and through
    the public-data rule that a source must be the only invoice of its (entity, total, currency)
    group. The rule also keeps coincidental twins and earlier injections out, so the injected
    pair is always the sole member of its group and ``independent_check`` is unambiguous.
    """
    excluded = _recurring_invoice_ids(episode.provenance) | touched_ids(episode)
    for group in _groups(episode.public.invoices).values():
        if len(group) > 1:
            excluded.update(invoice.invoice_id for invoice in group)
    return [i for i in episode.public.invoices if i.invoice_id not in excluded]


def _build_duplicate(
    original: Invoice, public: PublicEpisode, rng: random.Random, taken: set[str]
) -> Invoice:
    """Same economics, new ids, issued 0..10 whole days later, always inside the period.

    The shift is capped by the days left before ``period_end`` instead of clamping the result,
    so the duplicate keeps the original's time of day and its issued->due gap; a source issued
    on the last day of the period is rejected (the retry draws another one).
    """
    room = datetime.fromisoformat(public.manifest.period_end) - timedelta(seconds=1)
    days_left = (room - datetime.fromisoformat(original.issued_at)).days
    if days_left < 0:
        raise InjectorError("source invoice is issued too late for a same-period duplicate")
    shift = rng.randint(0, min(INJECTOR_DUPLICATE_DATE_SHIFT_MAX_DAYS, days_left))
    issued_at = shift_timestamp(original.issued_at, shift)
    invoice_id = fresh_label("INV", taken, rng)
    taken.add(invoice_id)
    doc_id = fresh_label("DOC", taken, rng)
    taken.add(doc_id)
    return Invoice.model_validate(
        {
            **original.model_dump(),
            "invoice_id": invoice_id,
            "doc_id": doc_id,
            "issued_at": issued_at,
            "due_at": shift_timestamp(original.due_at, shift),
        }
    )


def _journal(
    accounts: tuple[AccountName, AccountName],
    amount: int,
    currency: str,
    timestamp: str,
    entity_id: str,
    invoice_id: str,
    payment_id: str | None,
    rng: random.Random,
    taken: set[str],
) -> list[LedgerRow]:
    """One balanced two-row journal with fresh in-range row and journal labels."""
    journal_id = fresh_label("J", taken, rng)
    taken.add(journal_id)
    rows: list[LedgerRow] = []
    for account, debit, credit in ((accounts[0], amount, 0), (accounts[1], 0, amount)):
        row_id = fresh_label("L", taken, rng)
        taken.add(row_id)
        rows.append(
            LedgerRow(
                row_id=row_id,
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
        )
    return rows


def _ledger_rows_for_invoice(
    invoice: Invoice, side: EntityKind, rng: random.Random, taken: set[str]
) -> list[LedgerRow]:
    accounts: tuple[AccountName, AccountName] = (
        ("EXPENSE", "AP") if side == "supplier" else ("AR", "REVENUE")
    )
    return _journal(
        accounts, invoice.total_minor, invoice.currency, invoice.issued_at,
        invoice.entity_id, invoice.invoice_id, None, rng, taken,
    )  # fmt: skip


def _is_paid(public: PublicEpisode, invoice_id: str) -> bool:
    return any(ref.invoice_id == invoice_id for p in public.payments for ref in p.invoice_refs)


def _maybe_double_payment(
    duplicate: Invoice,
    original: Invoice,
    side: EntityKind,
    public: PublicEpisode,
    rng: random.Random,
    taken: set[str],
) -> tuple[Payment | None, list[LedgerRow]]:
    """Double-payment variant: 30 % of already-paid originals get the duplicate paid in full too.

    The payment and its journal follow the same rules as history.py (document currency if base,
    otherwise both rows in base at the policy-date rate). Unpaid originals never take this
    branch, so ``with_double_payment`` in provenance always means two payments exist.
    """
    if not _is_paid(public, original.invoice_id) or rng.random() >= float(_DOUBLE_PAYMENT_RATE):
        return None, []
    issued = datetime.fromisoformat(duplicate.issued_at)
    end = datetime.fromisoformat(public.manifest.period_end)
    when = issued + timedelta(
        days=rng.randint(*_PAYMENT_LAG_DAYS), seconds=rng.randrange(_SECONDS_PER_DAY)
    )
    if when >= end:
        return None, []
    doc_id = fresh_label("DOC", taken, rng)
    taken.add(doc_id)
    payment_id = fresh_label("PAY", taken, rng)
    taken.add(payment_id)
    payment = Payment(
        doc_id=doc_id,
        payment_id=payment_id,
        timestamp=when.isoformat(),
        currency=duplicate.currency,
        amount_minor=duplicate.total_minor,
        invoice_refs=(
            InvoiceAllocation(
                invoice_id=duplicate.invoice_id, allocated_minor=duplicate.total_minor
            ),
        ),
    )
    base = public.manifest.base_currency
    if payment.currency == base:
        amount, currency = payment.amount_minor, base
    else:
        day = policy_date(public.policy, duplicate.issued_at, payment.timestamp)
        rate = rate_on(build_fx_index(public.fx), day, payment.currency, base, public.policy)
        amount, currency = convert_minor(payment.amount_minor, rate, public.policy), base
    if amount < 1:  # same guard as history: a journal row needs a positive amount
        return None, []
    accounts: tuple[AccountName, AccountName] = (
        ("AP", "CASH") if side == "supplier" else ("CASH", "AR")
    )
    rows = _journal(
        accounts, amount, currency, payment.timestamp,
        duplicate.entity_id, duplicate.invoice_id, payment.payment_id, rng, taken,
    )  # fmt: skip
    return payment, rows


def _with_additions(
    public: PublicEpisode, invoice: Invoice, payment: Payment | None, rows: list[LedgerRow]
) -> PublicEpisode:
    """New PublicEpisode in canonical order with the manifest counts updated."""
    invoices = tuple(sorted((*public.invoices, invoice), key=lambda i: (i.issued_at, i.invoice_id)))
    payments = public.payments if payment is None else tuple(
        sorted((*public.payments, payment), key=lambda p: (p.timestamp, p.payment_id))
    )
    ledger = tuple(sorted((*public.ledger, *rows), key=lambda r: (r.timestamp, r.row_id)))
    manifest = public.manifest.model_copy(
        update={"n_ledger_rows": len(ledger), "n_documents": len(invoices) + len(payments)}
    )
    return public.model_copy(
        update={"manifest": manifest, "invoices": invoices, "payments": payments, "ledger": ledger}
    )


def inject(
    episode: PrivateEpisode, rng: random.Random, params: InjectorParams
) -> tuple[PrivateEpisode, Issue, ProvenanceEntry]:
    """Inject one duplicate invoice; ``params`` carries no duplicate-specific knobs."""
    del params  # count is handled by build.py; the module constants hold the knobs
    public = episode.public
    candidates = _eligible_invoices(episode)
    if not candidates:
        raise InjectorError("no eligible invoice for a duplicate injection")
    original = rng.choice(candidates)
    side = invoice_side(public, original)
    taken = used_labels(public)

    duplicate = _build_duplicate(original, public, rng, taken)
    rows = _ledger_rows_for_invoice(duplicate, side, rng, taken)
    payment, payment_rows = _maybe_double_payment(duplicate, original, side, public, rng, taken)
    new_public = _with_additions(public, duplicate, payment, [*rows, *payment_rows])

    issue_id = derive_issue_id(
        public.manifest.episode_id, original.invoice_id, {i.issue_id for i in episode.truth.issues}
    )
    issue = Issue(
        issue_id=issue_id,
        type=issue_type,
        affected_ids=tuple(sorted((original.invoice_id, duplicate.invoice_id))),
        expected_amount_minor=duplicate.total_minor,
        currency=duplicate.currency,
        acceptable_evidence_sets=(
            (original.doc_id, duplicate.doc_id),
            (original.invoice_id, duplicate.invoice_id),
            tuple(sorted(row.row_id for row in rows)),
        ),
    )
    created = [duplicate.invoice_id, duplicate.doc_id, *(row.row_id for row in rows)]
    if payment is not None:
        created.extend((payment.payment_id, payment.doc_id, *(row.row_id for row in payment_rows)))
    entry = ProvenanceEntry(
        issue_id=issue_id,
        op=issue_type,
        source_id=original.invoice_id,
        created_ids=tuple(created),
        original={"invoice_count": 1, "issued_at": original.issued_at},
        modified={
            "invoice_count": 2,
            "issued_at": duplicate.issued_at,
            "with_double_payment": payment is not None,
        },
        caused_by=(),
    )
    return episode.model_copy(update={"public": new_public}), issue, entry


def independent_check(public: PublicEpisode, issue: Issue) -> None:
    """Rediscover the duplicate from public files alone and compare with the truth issue.

    Invoices are grouped by (entity, total, currency); a group of two or more with identical
    line items that is not a legitimate monthly series is a duplicate pattern. One such group
    must match ``issue.affected_ids`` exactly, and its total and currency must equal the issue's.
    """
    suspicious = [g for g in _groups(public.invoices).values() if _is_duplicate_pattern(g)]
    if not suspicious:
        raise ConsistencyError("duplicate_check", "no duplicate pattern in public data")
    affected = set(issue.affected_ids)
    group = next((g for g in suspicious if {i.invoice_id for i in g} == affected), None)
    if group is None:
        raise ConsistencyError(
            "duplicate_check", "no duplicate group matches affected_ids", sorted(affected)
        )
    if group[0].total_minor != issue.expected_amount_minor:
        raise ConsistencyError(
            "duplicate_check",
            f"public overstatement {group[0].total_minor} != truth {issue.expected_amount_minor}",
        )
    if group[0].currency != issue.currency:
        raise ConsistencyError("duplicate_check", "currency mismatch between public and truth")


__all__ = ["independent_check", "inject", "issue_type"]
