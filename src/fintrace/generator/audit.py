"""Truth-blind integrity scan of a public episode, independent of the injectors.

Why this module exists. Each injector ships an ``independent_check``, but that check is driven by
the very record it is meant to confirm: it receives ``issue.affected_ids`` and
``issue.expected_amount_minor``, looks only where truth points, and the FX one re-uses the exact
conversion helpers the generator injected with. Two failures slip through it: a discrepancy an
injection created but never recorded in truth, and a bug shared by the injector and its own check.

``scan`` reads a ``PublicEpisode`` and nothing else -- no truth, no provenance, no
``fintrace.generator`` import (``tests/contract/test_runner.py`` pins that) -- and re-derives every
discrepancy with its own arithmetic. The conversion and period rules themselves are the episode's
published ``Policy``, read from its fields here rather than imported, so a mistake in the
generator's plumbing (wrong rate date, wrong direction, wrong quantum) shows up as a mismatch
instead of cancelling out.

``new_discrepancies(before, after)`` is the comparison the jury describes: the clean books and the
injected books are scanned separately and whatever the clean books already looked like cancels
out, so coincidental twins and the legitimate decoys never count as injections.
``verify_declared`` then compares what the injections added with what truth claims, in both
directions, which is what makes the check a real gate rather than a confirmation.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from decimal import ROUND_HALF_EVEN, Decimal
from itertools import pairwise
from typing import Final, NamedTuple
from zoneinfo import ZoneInfo

from fintrace.models import (
    FXRate,
    Invoice,
    IssueType,
    LedgerRow,
    Policy,
    PolicyPeriod,
    PublicEpisode,
    Truth,
)

# A rent-like series is "<service> - YYYY-MM" descriptions, distinct months, ~30 days apart; it
# is a legitimate pattern, not a duplicate. Re-derived here from the published data shape.
_RECURRING_DESC: Final[re.Pattern[str]] = re.compile(r" - [0-9]{4}-[0-9]{2}$")
_RECURRING_GAP_DAYS: Final[tuple[int, int]] = (25, 35)
# Converted amounts carry legitimate rounding noise; only a gap above this floor is an error.
_NOISE_FRAC: Final[Decimal] = Decimal("0.01")  # 1 % of the correct base amount
_NOISE_MIN_MINOR: Final[int] = 100
_ONE: Final[Decimal] = Decimal(1)
_OFFSET: Final[re.Pattern[str]] = re.compile(r"(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})")
_JOURNAL_ROWS: Final[int] = 2  # a document posting is one balanced two-row journal

LineKey = tuple[tuple[str, int, int, int], ...]


class Detected(NamedTuple):
    """One discrepancy found in the public books, in the same vocabulary as ``Issue``."""

    type: IssueType
    affected_ids: tuple[str, ...]  # sorted and deduplicated, as ``Issue.affected_ids`` is
    amount_minor: int
    currency: str

    @property
    def key(self) -> tuple[IssueType, tuple[str, ...]]:
        """Identity of the discrepancy: what kind it is and which records it spans."""
        return self.type, self.affected_ids


class AuditError(Exception):
    """The public books and the truth record disagree; the episode must not be published."""


# ---------------------------------------------------------------- duplicate_invoice


def _line_key(invoice: Invoice) -> LineKey:
    return tuple((li.desc, li.qty, li.unit_minor, li.amount_minor) for li in invoice.line_items)


def _looks_recurring(group: Sequence[Invoice]) -> bool:
    """A legitimate monthly series: dated descriptions, each month once, 25..35 days apart."""
    descs = [invoice.line_items[0].desc for invoice in group if invoice.line_items]
    if len(descs) != len(group) or len(set(descs)) != len(descs):
        return False
    if not all(_RECURRING_DESC.search(desc) for desc in descs):
        return False
    issued = sorted(datetime.fromisoformat(invoice.issued_at) for invoice in group)
    low, high = _RECURRING_GAP_DAYS
    gaps = [(later - earlier).days for earlier, later in pairwise(issued)]
    return all(low <= gap <= high for gap in gaps)


def _duplicate_invoices(public: PublicEpisode) -> list[Detected]:
    """Invoices sharing entity, total, currency and line items, outside a recurring series."""
    groups: dict[tuple[str, int, str], list[Invoice]] = defaultdict(list)
    for invoice in public.invoices:
        groups[(invoice.entity_id, invoice.total_minor, invoice.currency)].append(invoice)
    found: list[Detected] = []
    for (_, total, currency), group in groups.items():
        if len(group) < 2 or len({_line_key(invoice) for invoice in group}) != 1:
            continue
        if _looks_recurring(group):
            continue
        ids = tuple(sorted(invoice.invoice_id for invoice in group))
        found.append(Detected("duplicate_invoice", ids, total, currency))
    return found


# ---------------------------------------------------------------- payment_allocation


def _payment_allocations(public: PublicEpisode) -> list[Detected]:
    """A payment document that disagrees with itself or with the ledger about its allocation.

    Two shapes: the document and the ledger name different invoices (the misallocated sum is the
    misstatement), or the allocated amounts do not add up to the amount paid (the gap is). A
    split payment allocates several invoices that do add up, so it is not reported.
    """
    booked: dict[str, set[str]] = defaultdict(set)
    for row in public.ledger:
        if row.payment_id is not None and row.invoice_id is not None:
            booked[row.payment_id].add(row.invoice_id)
    found: list[Detected] = []
    for payment in public.payments:
        referenced = {ref.invoice_id for ref in payment.invoice_refs}
        on_ledger = booked.get(payment.payment_id, set())
        allocated = sum(ref.allocated_minor for ref in payment.invoice_refs)
        ids = tuple(sorted({payment.payment_id, *referenced, *on_ledger}))
        if len(referenced) == 1 and len(on_ledger) == 1 and referenced != on_ledger:
            found.append(Detected("payment_allocation", ids, allocated, payment.currency))
        elif payment.amount_minor != allocated:
            gap = abs(payment.amount_minor - allocated)
            found.append(Detected("payment_allocation", ids, gap, payment.currency))
    return found


# ---------------------------------------------------------------- fx_error


def _fx_curves(rates: Sequence[FXRate]) -> dict[tuple[str, str], list[tuple[str, Decimal]]]:
    curves: dict[tuple[str, str], list[tuple[str, Decimal]]] = defaultdict(list)
    for rate in rates:
        curves[(rate.source_currency, rate.target_currency)].append((rate.date, rate.rate))
    for curve in curves.values():
        curve.sort()
    return curves


def _rate_for(
    curves: dict[tuple[str, str], list[tuple[str, Decimal]]],
    day: str,
    source: str,
    target: str,
    policy: Policy,
) -> Decimal | None:
    """The rate the policy points at, or None when the policy cannot name one."""
    curve = curves.get((source, target))
    if not curve:
        return None
    exact = [rate for date_, rate in curve if date_ == day]
    if exact:
        return exact[0]
    if policy.fx.fallback == "previous_available_date":
        earlier = [rate for date_, rate in curve if date_ < day]
        return earlier[-1] if earlier else None
    if policy.fx.fallback == "next_available_date":
        later = [rate for date_, rate in curve if date_ > day]
        return later[0] if later else None
    return None


def _to_base(amount_minor: int, rate: Decimal, policy: Policy) -> int:
    """Minor units converted under the policy's direction and rounding mode."""
    amount = Decimal(amount_minor)
    converted = amount * rate if policy.fx.rate_direction == "source_to_target" else amount / rate
    return int(converted.quantize(_ONE, rounding=policy.rounding.mode))


def _noise_floor(expected_base: int) -> int:
    """max(100 minor units, 1 % of the correct base amount): below it, a gap is rounding."""
    scaled = (Decimal(expected_base) * _NOISE_FRAC).quantize(_ONE, rounding=ROUND_HALF_EVEN)
    return max(_NOISE_MIN_MINOR, int(scaled))


def _fx_errors(public: PublicEpisode) -> list[Detected]:
    """A foreign-currency payment whose base-currency journal is not the policy conversion."""
    base = public.manifest.base_currency
    curves = _fx_curves(public.fx)
    invoices = {invoice.invoice_id: invoice for invoice in public.invoices}
    rows_of: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in public.ledger:
        if row.payment_id is not None:
            rows_of[row.payment_id].append(row)
    found: list[Detected] = []
    for payment in public.payments:
        if payment.currency == base or len(payment.invoice_refs) != 1:
            continue
        invoice = invoices.get(payment.invoice_refs[0].invoice_id)
        rows = rows_of.get(payment.payment_id, [])
        if invoice is None or len(rows) != _JOURNAL_ROWS:
            continue
        if any(row.currency != base for row in rows):
            continue
        amounts = {row.debit_minor + row.credit_minor for row in rows}
        if len(amounts) != 1:
            continue
        transaction, posting = invoice.issued_at, payment.timestamp
        rule = public.policy.fx.rate_date_rule
        day = transaction[:10] if rule == "transaction_date" else posting[:10]
        rate = _rate_for(curves, day, payment.currency, base, public.policy)
        if rate is None:
            continue
        expected = _to_base(payment.amount_minor, rate, public.policy)
        gap = abs(amounts.pop() - expected)
        if gap and gap >= _noise_floor(expected):
            ids = tuple(sorted((payment.payment_id, invoice.invoice_id)))
            found.append(Detected("fx_error", ids, gap, base))
    return found


# ---------------------------------------------------------------- period_misclassification


def _zone(name: str) -> tzinfo:
    """``UTC``, a fixed ``+HH:MM`` offset (the generator's own format) or an IANA zone name."""
    if name == "UTC":
        return UTC
    match = _OFFSET.fullmatch(name)
    if match is None:
        return ZoneInfo(name)
    sign = -1 if match["sign"] == "-" else 1
    offset = timedelta(hours=int(match["hours"]), minutes=int(match["minutes"]))
    return timezone(sign * offset)


def _inside(iso: str, start: datetime, end: datetime, tz: tzinfo, rules: PolicyPeriod) -> bool:
    """Period membership in the policy timezone, under its inclusive/exclusive boundaries."""
    moment = datetime.fromisoformat(iso).astimezone(tz)
    after_start = start <= moment if rules.start_inclusive else start < moment
    before_end = moment < end if rules.end_exclusive else moment <= end
    return after_start and before_end


def _documents(public: PublicEpisode) -> Iterator[tuple[str, str, int, str, list[LedgerRow]]]:
    """(document id, its own date, amount, currency, the rows it owns) for every document."""
    owned: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in public.ledger:
        if row.payment_id is not None:
            owned[row.payment_id].append(row)
        elif row.invoice_id is not None:
            owned[row.invoice_id].append(row)
    for invoice in public.invoices:
        rows = owned.get(invoice.invoice_id, [])
        yield invoice.invoice_id, invoice.issued_at, invoice.total_minor, invoice.currency, rows
    for payment in public.payments:
        rows = owned.get(payment.payment_id, [])
        yield payment.payment_id, payment.timestamp, payment.amount_minor, payment.currency, rows


def _period_misclassifications(public: PublicEpisode) -> list[Detected]:
    """A document whose date and whose posting fall in different reporting periods."""
    rules = public.policy.period
    tz = _zone(rules.timezone)
    start = datetime.fromisoformat(public.manifest.period_start).astimezone(tz)
    end = datetime.fromisoformat(public.manifest.period_end).astimezone(tz)
    journal_size = Counter(row.journal_id for row in public.ledger)
    found: list[Detected] = []
    for doc_id, doc_date, amount, currency, rows in _documents(public):
        journals = {row.journal_id for row in rows}
        if len(rows) != _JOURNAL_ROWS or len(journals) != 1:
            continue
        journal_id = journals.pop()
        if journal_size[journal_id] != _JOURNAL_ROWS:
            continue
        posted = {row.timestamp for row in rows}
        if len(posted) != 1:
            continue
        posting = posted.pop()
        document_ts = doc_date if rules.assignment_field == "issued_at" else posting
        if _inside(document_ts, start, end, tz, rules) == _inside(posting, start, end, tz, rules):
            continue
        ids = tuple(sorted((doc_id, journal_id)))
        found.append(Detected("period_misclassification", ids, amount, currency))
    return found


# ---------------------------------------------------------------- the scan and its two gates


def scan(public: PublicEpisode) -> tuple[Detected, ...]:
    """Every discrepancy the public files alone reveal, in canonical order.

    Truth-blind by construction: the only argument is the public episode. On clean books the
    result is not guaranteed to be empty -- two unrelated invoices can coincide, and that is why
    the gates below compare the clean books with the injected ones instead of trusting one scan.
    """
    found = [
        *_duplicate_invoices(public),
        *_payment_allocations(public),
        *_fx_errors(public),
        *_period_misclassifications(public),
    ]
    return tuple(sorted(found))


def new_discrepancies(before: PublicEpisode, after: PublicEpisode) -> tuple[Detected, ...]:
    """What ``after`` shows and ``before`` did not: the discrepancies the injections added."""
    baseline = Counter(scan(before))
    return tuple(sorted((Counter(scan(after)) - baseline).elements()))


def verify_declared(before: PublicEpisode, after: PublicEpisode, truth: Truth) -> None:
    """Require the added discrepancies and the declared issues to be the same set.

    Both directions are errors: an issue truth declares that the books do not show (an injection
    that did not land, or that a later one masked) and a discrepancy the books show that truth
    does not declare (an injection that changed more than it recorded). Amount and currency are
    compared for every pair that does match, which is where an arithmetic bug surfaces.
    """
    found = new_discrepancies(before, after)
    declared = tuple(
        Detected(issue.type, issue.affected_ids, issue.expected_amount_minor, issue.currency)
        for issue in truth.issues
    )
    found_by_key = {item.key: item for item in found}
    declared_by_key = {item.key: item for item in declared}
    if len(found_by_key) != len(found) or len(declared_by_key) != len(declared):
        raise AuditError("two discrepancies share one (type, affected_ids) key")
    missing = [item for item in declared if item.key not in found_by_key]
    if missing:
        raise AuditError(f"declared but not present in the books: {_render(missing)}")
    extra = [item for item in found if item.key not in declared_by_key]
    if extra:
        raise AuditError(f"present in the books but not declared: {_render(extra)}")
    for key, expected in declared_by_key.items():
        actual = found_by_key[key]
        if (actual.amount_minor, actual.currency) != (expected.amount_minor, expected.currency):
            raise AuditError(
                f"{key[0]} on {list(key[1])}: books say "
                f"{actual.amount_minor} {actual.currency}, truth says "
                f"{expected.amount_minor} {expected.currency}"
            )


def _render(items: Sequence[Detected]) -> str:
    return "; ".join(
        f"{item.type} on {list(item.affected_ids)} ({item.amount_minor} {item.currency})"
        for item in items
    )


__all__ = ["AuditError", "Detected", "new_discrepancies", "scan", "verify_declared"]
