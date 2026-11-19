"""Period misclassification injector: a document and its posting land in different periods.

Period membership is ``[period_start, period_end)`` (per ``policy.period`` flags) evaluated in
``policy.period.timezone`` on ``policy.period.assignment_field`` (``issued_at``: the document date
decides). In clean books an invoice journal carries the invoice's issued_at exactly, so document
and posting always share a period; this injector breaks that for one invoice.

Variants (uniform):
  (a) ``ledger_row_shifted_in``: the invoice really belongs to the neighbouring period (issued_at
      and due_at move past the boundary) while its journal stays posted inside the period, which
      is therefore overstated by the invoice total;
  (b) ``ledger_row_shifted_out``: the invoice stays inside, both rows of its journal are posted
      past the boundary, so the period is understated by the invoice total.
Source: an invoice issued within ``BOUNDARY_PROXIMITY_HOURS`` of either boundary (cut-off errors
happen around period end), untouched by earlier injections and not a recurring decoy. The moved
timestamp lands 1..6 h past the boundary in the policy timezone, keeping the original minutes,
seconds and UTC offset, so nothing but the period membership changes; the shift is a whole number
of hours. Expected amount = invoice total in the invoice currency; evidence = invoice document and
the two journal rows. ``assignment_field == "posted_at"`` classifies by the posting itself, so no
mismatch can exist and the injector refuses such a policy.

Feasibility: a very small episode may have no untouched, non-recurring invoice within a week of
either boundary; build.py then raises InjectorError deterministically for that (seed, config)
and dataset builders skip the seed (2 of the first 200 default seeds).
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Final, Literal
from zoneinfo import ZoneInfo

from fintrace.generator.company import fixed_offset
from fintrace.generator.validate import ConsistencyError
from fintrace.models import (
    InjectorParams,
    Invoice,
    Issue,
    IssueType,
    LedgerRow,
    Payment,
    PolicyPeriod,
    PrivateEpisode,
    ProvenanceEntry,
    PublicEpisode,
)

from .base import (
    RELAXED_IDS_KEY,
    InjectorError,
    derive_issue_id,
    recurring_invoice_ids,
    touched_ids,
)

issue_type: Final[IssueType] = "period_misclassification"

PeriodVariant = Literal["ledger_row_shifted_in", "ledger_row_shifted_out"]
VARIANTS: Final[tuple[PeriodVariant, ...]] = ("ledger_row_shifted_in", "ledger_row_shifted_out")
MIN_SHIFT_HOURS: Final[int] = 1  # the moved timestamp lands at least this far past the boundary
MAX_SHIFT_HOURS: Final[int] = 6  # ... and less than this far
BOUNDARY_PROXIMITY_HOURS: Final[int] = 168  # source invoices issued within 7 days of a boundary
_CHECK: Final[str] = "period_check"

Injected = tuple[PrivateEpisode, Issue, ProvenanceEntry]
Journal = tuple[LedgerRow, LedgerRow]


def _tzinfo(tz: str) -> tzinfo:
    """``±HH:MM`` (fixed offset, the generator's own format), ``UTC`` or an IANA name (zoneinfo)."""
    if tz == "UTC":
        return UTC
    try:
        return fixed_offset(tz)
    except ValueError:
        return ZoneInfo(tz)


def _local(iso: str, tz: tzinfo) -> datetime:
    return datetime.fromisoformat(iso).astimezone(tz)


def _period_bounds(public: PublicEpisode) -> tuple[datetime, datetime, tzinfo]:
    """(start, end, tz): the manifest period expressed in the policy timezone."""
    tz = _tzinfo(public.policy.period.timezone)
    return _local(public.manifest.period_start, tz), _local(public.manifest.period_end, tz), tz


def _inside(iso: str, start: datetime, end: datetime, tz: tzinfo, rules: PolicyPeriod) -> bool:
    """Period membership of ``iso`` under the policy's inclusive/exclusive boundary flags."""
    moment = _local(iso, tz)
    after_start = start <= moment if rules.start_inclusive else start < moment
    before_end = moment < end if rules.end_exclusive else moment <= end
    return after_start and before_end


def _nearest_boundary(moment: datetime, start: datetime, end: datetime) -> tuple[datetime, int]:
    """The closer boundary and the outward direction: -1 (before start) or +1 (past end)."""
    return (start, -1) if moment - start <= end - moment else (end, 1)


def _landing(issued: datetime, boundary: datetime, outward: int, rng: random.Random) -> datetime:
    """A UTC instant 1..6 h past ``boundary`` on the outward side, carrying ``issued``'s mm:ss.

    Past the end: ``end + h:mm:ss`` with h in [1, 5] (distance in [1 h, 6 h)); before the start:
    ``start - h + mm:ss`` with h in [2, 6] (distance in (1 h, 6 h]). The arithmetic runs on UTC
    instants (aware wall-clock arithmetic would drift across a DST change in an IANA zone); with
    a whole-hour offset the local minutes and seconds of the original are kept and the shift is a
    whole number of hours. The caller renders the result in the data's own offset.
    """
    tail = timedelta(minutes=issued.minute, seconds=issued.second)
    boundary_utc = boundary.astimezone(UTC)
    if outward > 0:
        hours = rng.randint(MIN_SHIFT_HOURS, MAX_SHIFT_HOURS - 1)
        return boundary_utc + timedelta(hours=hours) + tail
    hours = rng.randint(MIN_SHIFT_HOURS + 1, MAX_SHIFT_HOURS)
    return boundary_utc - timedelta(hours=hours) + tail


def _candidates(episode: PrivateEpisode) -> list[tuple[Invoice, Journal]]:
    """Untouched, non-recurring invoices near a boundary with a clean two-row journal inside."""
    public = episode.public
    start, end, tz = _period_bounds(public)
    excluded = touched_ids(episode) | recurring_invoice_ids(episode.provenance)
    proximity = timedelta(hours=BOUNDARY_PROXIMITY_HOURS)
    rows_by_invoice: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in public.ledger:
        if row.invoice_id is not None and row.payment_id is None:
            rows_by_invoice[row.invoice_id].append(row)
    candidates: list[tuple[Invoice, Journal]] = []
    for invoice in public.invoices:
        if invoice.invoice_id in excluded or invoice.doc_id in excluded:
            continue
        issued = _local(invoice.issued_at, tz)
        if not (issued - start < proximity or end - issued < proximity):
            continue
        rows = sorted(rows_by_invoice.get(invoice.invoice_id, []), key=lambda r: r.row_id)
        if len(rows) != 2 or len({r.journal_id for r in rows}) != 1:
            continue
        if any(r.row_id in excluded or r.timestamp != invoice.issued_at for r in rows):
            continue
        candidates.append((invoice, (rows[0], rows[1])))
    return candidates


def _payments_of(public: PublicEpisode, invoice_id: str) -> list[Payment]:
    return [p for p in public.payments if any(r.invoice_id == invoice_id for r in p.invoice_refs)]


def _redated_invoice(public: PublicEpisode, invoice: Invoice, landing: datetime) -> Invoice:
    """Variant (a): the invoice re-dated to ``landing`` (due_at moves by the same delta).

    Rejected when a payment would predate the new date, or when the invoice is foreign and the
    FX policy keys the rate on the transaction date: the payment journals' conversions would then
    silently stop matching the books, which is a different error (fx_error), not this one.
    """
    payments = _payments_of(public, invoice.invoice_id)
    if any(datetime.fromisoformat(p.timestamp) < landing for p in payments):
        raise InjectorError("a payment would predate the re-dated invoice")
    rate_on_transaction = public.policy.fx.rate_date_rule == "transaction_date"
    if payments and rate_on_transaction and invoice.currency != public.manifest.base_currency:
        raise InjectorError("re-dating a paid foreign invoice would move its FX policy date")
    delta = landing - datetime.fromisoformat(invoice.issued_at)
    due_at = (datetime.fromisoformat(invoice.due_at) + delta).isoformat()
    dumped = {**invoice.model_dump(), "issued_at": landing.isoformat(), "due_at": due_at}
    return Invoice.model_validate(dumped)


def _reposted_journal(rows: Journal, landing: datetime) -> Journal:
    """Variant (b): both journal rows re-posted at ``landing``."""
    first, second = (
        LedgerRow.model_validate({**row.model_dump(), "timestamp": landing.isoformat()})
        for row in rows
    )
    return first, second


def _with_changes(
    episode: PrivateEpisode, invoice: Invoice | None, rows: Journal | None
) -> PrivateEpisode:
    """Replace the invoice and/or journal rows and restore canonical (timestamp, id) order."""
    public = episode.public
    update: dict[str, object] = {}
    if invoice is not None:
        invoices = (invoice if i.invoice_id == invoice.invoice_id else i for i in public.invoices)
        update["invoices"] = tuple(sorted(invoices, key=lambda i: (i.issued_at, i.invoice_id)))
    if rows is not None:
        by_id = {row.row_id: row for row in rows}
        ledger = (by_id.get(row.row_id, row) for row in public.ledger)
        update["ledger"] = tuple(sorted(ledger, key=lambda r: (r.timestamp, r.row_id)))
    return episode.model_copy(update={"public": public.model_copy(update=update)})


def _inject_variant(
    episode: PrivateEpisode, rng: random.Random, variant: PeriodVariant
) -> Injected:
    """Apply one variant to a clean invoice near a period boundary."""
    public = episode.public
    rules = public.policy.period
    if rules.assignment_field != "issued_at":
        raise InjectorError("period injector needs policy.period.assignment_field == 'issued_at'")
    start, end, tz = _period_bounds(public)
    candidates = _candidates(episode)
    if not candidates:
        raise InjectorError("no untouched invoice near a period boundary")
    invoice, rows = rng.choice(candidates)
    issued = _local(invoice.issued_at, tz)
    boundary, outward = _nearest_boundary(issued, start, end)
    # rendered in the data's own offset (never a lone row in another offset)
    data_tz = datetime.fromisoformat(invoice.issued_at).tzinfo
    landing = _landing(issued, boundary, outward, rng).astimezone(data_tz)
    if _inside(landing.isoformat(), start, end, tz, rules):
        raise InjectorError("landing did not leave the period")

    if variant == "ledger_row_shifted_in":
        new_private = _with_changes(episode, _redated_invoice(public, invoice, landing), None)
        issued_after, posted_after = landing.isoformat(), rows[0].timestamp
    else:
        new_private = _with_changes(episode, None, _reposted_journal(rows, landing))
        issued_after, posted_after = invoice.issued_at, landing.isoformat()

    journal_id = rows[0].journal_id
    row_ids = (rows[0].row_id, rows[1].row_id)
    taken = {issue.issue_id for issue in episode.truth.issues}
    issue_id = derive_issue_id(public.manifest.episode_id, invoice.invoice_id, taken)
    issue = Issue(
        issue_id=issue_id,
        type=issue_type,
        affected_ids=tuple(sorted((invoice.invoice_id, journal_id))),
        expected_amount_minor=invoice.total_minor,
        currency=invoice.currency,  # the misstatement is the document amount, in its currency
        acceptable_evidence_sets=(
            (invoice.doc_id, *row_ids),
            (invoice.invoice_id, journal_id),
            row_ids,
        ),
    )
    # instant differences (different tzinfo objects -> UTC arithmetic), whole hours for the
    # generator's whole-hour offsets
    shift_seconds = int((landing - issued).total_seconds())
    side = "start" if outward < 0 else "end"
    mover = "document" if variant == "ledger_row_shifted_in" else "posting"
    entry = ProvenanceEntry(
        issue_id=issue_id,
        op=issue_type,
        source_id=invoice.invoice_id,
        created_ids=(),
        original={
            "variant": variant,
            "issued_at": invoice.issued_at,
            "journal_timestamp": rows[0].timestamp,
            "policy_tz": rules.timezone,
        },
        modified={
            "variant": variant,
            "issued_at": issued_after,
            "journal_timestamp": posted_after,
            "boundary": boundary.isoformat(),
            "outward": outward,
            "shift_seconds_utc": shift_seconds,
            "shift_hours_utc": shift_seconds // 3600,
            "seconds_past_boundary": abs(int((landing - boundary).total_seconds())),
            RELAXED_IDS_KEY: list(row_ids),
        },
        caused_by=(),
        note=f"Cut-off error: {mover} lands past the period {side} in {rules.timezone}",
    )
    return new_private, issue, entry


def inject(episode: PrivateEpisode, rng: random.Random, params: InjectorParams) -> Injected:
    """Pick a variant uniformly, apply it and return (episode, issue, entry)."""
    del params  # count is handled by build.py; the module constants hold the knobs
    variant: PeriodVariant = rng.choice(VARIANTS)
    return _inject_variant(episode, rng, variant)


def independent_check(public: PublicEpisode, issue: Issue) -> None:
    """Rediscover the cut-off error from public files alone.

    ``affected_ids`` names one invoice and its journal; the journal must be the invoice's two-row
    posting; the document (per ``assignment_field``) and the posting must fall in different
    periods when evaluated in the policy timezone; expected amount and currency are the invoice's.
    """
    affected = set(issue.affected_ids)
    invoices = [i for i in public.invoices if i.invoice_id in affected]
    if len(invoices) != 1:
        raise ConsistencyError(_CHECK, "affected_ids must name one invoice", sorted(affected))
    invoice = invoices[0]
    others = sorted(affected - {invoice.invoice_id})
    if len(others) != 1:
        raise ConsistencyError(_CHECK, "affected_ids must name the invoice and its journal", others)
    journal_id = others[0]
    rows = sorted((r for r in public.ledger if r.journal_id == journal_id), key=lambda r: r.row_id)
    row_ids = [row.row_id for row in rows]
    if len(rows) != 2 or any(
        row.invoice_id != invoice.invoice_id or row.payment_id is not None for row in rows
    ):
        raise ConsistencyError(_CHECK, "journal is not the invoice's two-row posting", row_ids)
    if len({row.timestamp for row in rows}) != 1:
        raise ConsistencyError(_CHECK, "journal rows carry different timestamps", row_ids)
    start, end, tz = _period_bounds(public)
    rules = public.policy.period
    document_ts = invoice.issued_at if rules.assignment_field == "issued_at" else rows[0].timestamp
    document_inside = _inside(document_ts, start, end, tz, rules)
    posting_inside = _inside(rows[0].timestamp, start, end, tz, rules)
    if document_inside == posting_inside:
        raise ConsistencyError(_CHECK, "document and posting fall in the same period", row_ids)
    if invoice.currency != issue.currency:
        raise ConsistencyError(_CHECK, "finding currency must be the invoice currency")
    if invoice.total_minor != issue.expected_amount_minor:
        raise ConsistencyError(
            _CHECK, f"invoice total {invoice.total_minor} != truth {issue.expected_amount_minor}"
        )


__all__ = [
    "BOUNDARY_PROXIMITY_HOURS",
    "MAX_SHIFT_HOURS",
    "MIN_SHIFT_HOURS",
    "VARIANTS",
    "PeriodVariant",
    "independent_check",
    "inject",
    "issue_type",
]
