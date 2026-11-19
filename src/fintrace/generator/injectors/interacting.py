"""Interacting scenarios: two base errors composed into one causally linked story (blueprint 5.1).

Scenario A - partial payment of a foreign-currency invoice near the period boundary. A clean
foreign payment within SCENARIO_A_PROXIMITY_HOURS of a period boundary has its allocation cut by
30..50 % (payment_allocation) and its journal re-posted 1..6 h past the nearest boundary
(period_misclassification). Two issues in the payment currency: the unallocated remainder, and
the payment amount booked in the wrong period. They describe different discrepancies and never add
up to one figure.

Scenario B - duplicate foreign invoice, paid and booked at the wrong rate. A clean foreign invoice
is duplicated (duplicate_invoice), the duplicate is paid in full, and the payment journal is
converted at a perturbed rate (fx_error). Two issues: the duplicate total in the invoice currency,
and the conversion gap on the duplicate's payment rows in the base currency. The duplicate amount
is the document total; the FX error is the *difference* on the booked rows; no overlap.

Each issue carries the other's id in ``interacts_with``; provenance links them through
``caused_by`` (root entry first). Every issue is verified by the base injector's
``independent_check`` of its type, so a finding of either issue alone is a true positive on its
own. The scenarios reuse the base injectors' package-internal building blocks rather than copying
them; they never import the evaluator or the environment.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timedelta
from typing import Final

from fintrace.generator.company import EntityKind
from fintrace.generator.fx import build_fx_index, convert_minor, policy_date, rate_on
from fintrace.models import (
    InjectorParams,
    Invoice,
    InvoiceAllocation,
    Issue,
    JsonObject,
    LedgerRow,
    Payment,
    PrivateEpisode,
    ProvenanceEntry,
    PublicEpisode,
)

from . import duplicate, fx_error, period
from .base import (
    RELAXED_IDS_KEY,
    InjectorError,
    InjectorInfeasible,
    derive_issue_id,
    fresh_label,
    invoice_side,
    payment_journal_rows,
    touched_ids,
    used_labels,
)

SCENARIOS: Final[tuple[str, str]] = ("scenario_a", "scenario_b")
SCENARIO_A_PROXIMITY_HOURS: Final[int] = 720  # payments within 30 days of a boundary
SCENARIO_A_REDUCTION_PCT: Final[tuple[int, ...]] = (30, 37, 42, 50)
OP_A_PARTIAL: Final[str] = "interacting_scenario_a_partial"  # in SOURCE_MODIFYING_OPS
OP_A_PERIOD: Final[str] = "interacting_scenario_a_period"
OP_B_DUPLICATE: Final[str] = "interacting_scenario_b_duplicate"
OP_B_FX: Final[str] = "interacting_scenario_b_fx"
_PAYMENT_LAG_DAYS: Final[tuple[int, int]] = (1, 20)
_SECONDS_PER_DAY: Final[int] = 86_400

Journal = tuple[LedgerRow, LedgerRow]
ScenarioResult = tuple[PrivateEpisode, tuple[Issue, ...], tuple[ProvenanceEntry, ...]]


def _pair_ids(episode: PrivateEpisode, source_id: str) -> tuple[str, str]:
    """Two distinct deterministic issue ids for one source (the second re-hashes on collision)."""
    taken = {issue.issue_id for issue in episode.truth.issues}
    first = derive_issue_id(episode.public.manifest.episode_id, source_id, taken)
    second = derive_issue_id(episode.public.manifest.episode_id, source_id, taken | {first})
    return first, second


def _sorted_ledger(rows: tuple[LedgerRow, ...]) -> tuple[LedgerRow, ...]:
    return tuple(sorted(rows, key=lambda r: (r.timestamp, r.row_id)))


# ------------------------------------------------------------------ scenario A


def _scenario_a_candidates(episode: PrivateEpisode) -> list[tuple[Payment, Invoice, Journal]]:
    """Clean foreign payments near a boundary: single full allocation, untouched, two base rows."""
    public = episode.public
    base = public.manifest.base_currency
    start, end, tz = period._period_bounds(public)
    touched = touched_ids(episode)
    payments_per_invoice = Counter(r.invoice_id for p in public.payments for r in p.invoice_refs)
    invoice_by_id = {invoice.invoice_id: invoice for invoice in public.invoices}
    proximity = timedelta(hours=SCENARIO_A_PROXIMITY_HOURS)
    candidates: list[tuple[Payment, Invoice, Journal]] = []
    for payment in public.payments:
        refs = payment.invoice_refs
        if payment.currency == base or len(refs) != 1:
            continue
        if refs[0].allocated_minor != payment.amount_minor:
            continue
        invoice = invoice_by_id.get(refs[0].invoice_id)
        if invoice is None or payments_per_invoice[invoice.invoice_id] != 1:
            continue
        if {payment.payment_id, payment.doc_id, invoice.invoice_id} & touched:
            continue
        rows = payment_journal_rows(public, payment.payment_id)
        if len(rows) != 2 or any(r.currency != base or r.row_id in touched for r in rows):
            continue
        if len({r.journal_id for r in rows}) != 1 or len({r.timestamp for r in rows}) != 1:
            continue
        posted = period._local(rows[0].timestamp, tz)
        if not (posted - start < proximity or end - posted < proximity):
            continue
        candidates.append((payment, invoice, (rows[0], rows[1])))
    return candidates


def _partial_refs(payment: Payment, pct: int) -> tuple[InvoiceAllocation, ...]:
    """The single allocation cut by ``pct`` percent (integer arithmetic, stays positive)."""
    ref = payment.invoice_refs[0]
    new_allocated = ref.allocated_minor - ref.allocated_minor * pct // 100
    if not 1 <= new_allocated < ref.allocated_minor:
        raise InjectorError("partial reduction left the allocation unchanged or non-positive")
    return (InvoiceAllocation(invoice_id=ref.invoice_id, allocated_minor=new_allocated),)


def _repost_journal(
    public: PublicEpisode, rows: Journal, rng: random.Random
) -> tuple[Journal, datetime, datetime, int]:
    """Re-post both rows 1..6 h past the nearest boundary: (rows, landing, boundary, outward)."""
    start, end, tz = period._period_bounds(public)
    rules = public.policy.period
    posted = period._local(rows[0].timestamp, tz)
    boundary, outward = period._nearest_boundary(posted, start, end)
    data_tz = datetime.fromisoformat(rows[0].timestamp).tzinfo
    landing = period._landing(posted, boundary, outward, rng).astimezone(data_tz)
    if period._inside(landing.isoformat(), start, end, tz, rules):
        raise InjectorError("landing did not leave the period")
    return period._reposted_journal(rows, landing), landing, boundary, outward


def inject_scenario_a(
    episode: PrivateEpisode, rng: random.Random, params: InjectorParams
) -> ScenarioResult:
    """Scenario A: a foreign payment is under-allocated and posted in the wrong period."""
    del params  # the count lives in config.errors.interacting; build.py drives it
    public = episode.public
    if public.policy.period.assignment_field != "issued_at":
        raise InjectorInfeasible("scenario A needs policy.period.assignment_field 'issued_at'")
    candidates = _scenario_a_candidates(episode)
    if not candidates:
        raise InjectorInfeasible("scenario A: no foreign payment near a period boundary")
    payment, invoice, rows = rng.choice(candidates)
    pct = rng.choice(SCENARIO_A_REDUCTION_PCT)
    refs = _partial_refs(payment, pct)
    partial = Payment.model_validate(
        {**payment.model_dump(), "invoice_refs": [ref.model_dump() for ref in refs]}
    )
    reposted, landing, boundary, outward = _repost_journal(public, rows, rng)

    payments = tuple(
        partial if p.payment_id == payment.payment_id else p for p in public.payments
    )
    by_id = {row.row_id: row for row in reposted}
    ledger = _sorted_ledger(tuple(by_id.get(row.row_id, row) for row in public.ledger))
    new_public = public.model_copy(update={"payments": payments, "ledger": ledger})
    new_private = episode.model_copy(update={"public": new_public})

    partial_id, period_id = _pair_ids(episode, payment.payment_id)
    row_ids = (rows[0].row_id, rows[1].row_id)
    journal_id = rows[0].journal_id
    remainder = payment.amount_minor - sum(ref.allocated_minor for ref in refs)
    issue_partial = Issue(
        issue_id=partial_id,
        type="payment_allocation",
        affected_ids=tuple(sorted((payment.payment_id, invoice.invoice_id))),
        expected_amount_minor=remainder,
        currency=payment.currency,
        acceptable_evidence_sets=(
            (payment.doc_id, invoice.doc_id),
            (payment.payment_id, invoice.invoice_id),
            row_ids,
        ),
        interacts_with=(period_id,),
    )
    issue_period = Issue(
        issue_id=period_id,
        type="period_misclassification",
        affected_ids=tuple(sorted((payment.payment_id, journal_id))),
        expected_amount_minor=payment.amount_minor,
        currency=payment.currency,
        acceptable_evidence_sets=(
            (payment.doc_id, invoice.doc_id),
            (payment.payment_id, journal_id),
            row_ids,
        ),
        interacts_with=(partial_id,),
    )
    shift_seconds = int((landing - datetime.fromisoformat(rows[0].timestamp)).total_seconds())
    side = "start" if outward < 0 else "end"
    entry_partial = ProvenanceEntry(
        issue_id=partial_id,
        op=OP_A_PARTIAL,
        source_id=payment.payment_id,
        created_ids=(),
        original={
            "scenario": "A",
            "variant": "partial_allocation",
            "allocated_minor": payment.invoice_refs[0].allocated_minor,
        },
        modified={
            "scenario": "A",
            "variant": "partial_allocation",
            "allocated_minor": refs[0].allocated_minor,
            "reduction_pct": pct,
        },
        caused_by=(),
        note=f"Scenario A root: foreign payment under-allocated by {pct} %",
    )
    entry_period = ProvenanceEntry(
        issue_id=period_id,
        op=OP_A_PERIOD,
        source_id=payment.payment_id,
        created_ids=(),
        original={
            "scenario": "A",
            "variant": "payment_journal_shifted_out",
            "journal_timestamp": rows[0].timestamp,
            "policy_tz": public.policy.period.timezone,
        },
        modified={
            "scenario": "A",
            "variant": "payment_journal_shifted_out",
            "journal_timestamp": landing.isoformat(),
            "boundary": boundary.isoformat(),
            "outward": outward,
            "shift_seconds_utc": shift_seconds,
            "seconds_past_boundary": abs(int((landing - boundary).total_seconds())),
            RELAXED_IDS_KEY: list(row_ids),
        },
        caused_by=(partial_id,),
        note=f"Scenario A: the under-allocated payment is posted past the period {side}",
    )
    return new_private, (issue_partial, issue_period), (entry_partial, entry_period)


# ------------------------------------------------------------------ scenario B


def _scenario_b_candidates(episode: PrivateEpisode) -> list[Invoice]:
    """Duplicate-eligible invoices (untouched, non-recurring, unique group) in foreign currency."""
    base = episode.public.manifest.base_currency
    return [i for i in duplicate._eligible_invoices(episode) if i.currency != base]


def _pay_duplicate(
    public: PublicEpisode,
    dup: Invoice,
    side: EntityKind,
    rng: random.Random,
    taken: set[str],
) -> tuple[Payment, Journal, int]:
    """Pay the duplicate in full 1..20 days later, inside the period: (payment, rows, policy base).

    The journal is first built at the policy rate (the clean posting rule); the caller then
    re-books it at the wrong rate.
    """
    issued = datetime.fromisoformat(dup.issued_at)
    end = datetime.fromisoformat(public.manifest.period_end)
    when = issued + timedelta(
        days=rng.randint(*_PAYMENT_LAG_DAYS), seconds=rng.randrange(_SECONDS_PER_DAY)
    )
    if when >= end:
        raise InjectorError("the duplicate's payment would fall outside the period")
    doc_id = fresh_label("DOC", taken, rng)
    taken.add(doc_id)
    payment_id = fresh_label("PAY", taken, rng)
    taken.add(payment_id)
    payment = Payment(
        doc_id=doc_id,
        payment_id=payment_id,
        timestamp=when.isoformat(),
        currency=dup.currency,
        amount_minor=dup.total_minor,
        invoice_refs=(
            InvoiceAllocation(invoice_id=dup.invoice_id, allocated_minor=dup.total_minor),
        ),
    )
    base = public.manifest.base_currency
    day = policy_date(public.policy, dup.issued_at, payment.timestamp)
    rate = rate_on(build_fx_index(public.fx), day, payment.currency, base, public.policy)
    policy_base = convert_minor(payment.amount_minor, rate, public.policy)
    if policy_base < 1:
        raise InjectorError("converted payment amount is not positive")
    accounts = ("AP", "CASH") if side == "supplier" else ("CASH", "AR")
    rows = duplicate._journal(
        accounts, policy_base, base, payment.timestamp,
        dup.entity_id, dup.invoice_id, payment.payment_id, rng, taken,
    )  # fmt: skip
    return payment, (rows[0], rows[1]), policy_base


def _misprice_journal(
    public: PublicEpisode,
    payment: Payment,
    dup: Invoice,
    rows: Journal,
    policy_base: int,
    rng: random.Random,
) -> tuple[Journal, JsonObject, JsonObject]:
    """Re-book the payment rows at a perturbed rate: (rows, original, modified) details."""
    policy_rate = fx_error._policy_rate(public, payment, dup)
    wrong_rate, factor = fx_error._perturbed_rate(policy_rate, rng, fx_error.PERTURB_FACTORS)
    wrong_base = convert_minor(payment.amount_minor, wrong_rate, public.policy)
    discrepancy = abs(wrong_base - policy_base)
    if not fx_error._passes_noise_floor(discrepancy, policy_base):
        raise InjectorError("scenario B conversion gap is below the noise floor")
    mispriced = fx_error._rewrite_rows(rows, wrong_base)
    original: JsonObject = {
        "scenario": "B",
        "variant": "perturbed_rate",
        "policy_date": policy_date(public.policy, dup.issued_at, payment.timestamp),
        "policy_rate": str(policy_rate),
        "policy_base_minor": policy_base,
    }
    modified: JsonObject = {
        "scenario": "B",
        "variant": "perturbed_rate",
        "wrong_rate": str(wrong_rate),
        "wrong_base_minor": wrong_base,
        "factor": str(factor),
        RELAXED_IDS_KEY: [row.row_id for row in rows],
    }
    return (mispriced[0], mispriced[1]), original, modified


def inject_scenario_b(
    episode: PrivateEpisode, rng: random.Random, params: InjectorParams
) -> ScenarioResult:
    """Scenario B: a foreign invoice is duplicated; the duplicate is paid at a wrong rate."""
    del params  # the count lives in config.errors.interacting; build.py drives it
    public = episode.public
    candidates = _scenario_b_candidates(episode)
    if not candidates:
        raise InjectorInfeasible("no eligible foreign-currency invoice for scenario B")
    original = rng.choice(candidates)
    side = invoice_side(public, original)
    taken = used_labels(public)
    dup = duplicate._build_duplicate(original, public, rng, taken)
    dup_rows = duplicate._ledger_rows_for_invoice(dup, side, rng, taken)
    payment, clean_rows, policy_base = _pay_duplicate(public, dup, side, rng, taken)
    mispriced, fx_original, fx_modified = _misprice_journal(
        public, payment, dup, clean_rows, policy_base, rng
    )
    new_public = duplicate._with_additions(public, dup, payment, [*dup_rows, *mispriced])
    new_private = episode.model_copy(update={"public": new_public})

    dup_id, fx_id = _pair_ids(episode, original.invoice_id)
    dup_row_ids = tuple(sorted(row.row_id for row in dup_rows))
    payment_row_ids = (clean_rows[0].row_id, clean_rows[1].row_id)
    issue_dup = Issue(
        issue_id=dup_id,
        type="duplicate_invoice",
        affected_ids=tuple(sorted((original.invoice_id, dup.invoice_id))),
        expected_amount_minor=dup.total_minor,
        currency=dup.currency,  # the overstatement is the document total, in its currency
        acceptable_evidence_sets=(
            (original.doc_id, dup.doc_id),
            (original.invoice_id, dup.invoice_id),
            dup_row_ids,
        ),
        interacts_with=(fx_id,),
    )
    discrepancy = abs(int(str(fx_modified["wrong_base_minor"])) - policy_base)
    issue_fx = Issue(
        issue_id=fx_id,
        type="fx_error",
        affected_ids=tuple(sorted((dup.invoice_id, payment.payment_id))),
        expected_amount_minor=discrepancy,
        currency=public.manifest.base_currency,  # the conversion gap lives in the base books
        acceptable_evidence_sets=(
            (payment.doc_id, dup.doc_id),
            (payment.payment_id, dup.invoice_id),
            payment_row_ids,
        ),
        interacts_with=(dup_id,),
    )
    entry_dup = ProvenanceEntry(
        issue_id=dup_id,
        op=OP_B_DUPLICATE,
        source_id=original.invoice_id,
        created_ids=(
            dup.invoice_id,
            dup.doc_id,
            *dup_row_ids,
            payment.payment_id,
            payment.doc_id,
            *payment_row_ids,
        ),
        original={
            "scenario": "B",
            "variant": "duplicate_invoice",
            "invoice_count": 1,
            "issued_at": original.issued_at,
        },
        modified={
            "scenario": "B",
            "variant": "duplicate_invoice",
            "invoice_count": 2,
            "issued_at": dup.issued_at,
            "duplicate_paid": True,
        },
        caused_by=(),
        note="Scenario B root: foreign invoice duplicated and the duplicate paid in full",
    )
    entry_fx = ProvenanceEntry(
        issue_id=fx_id,
        op=OP_B_FX,
        source_id=payment.payment_id,
        created_ids=(),
        original=fx_original,
        modified=fx_modified,
        caused_by=(dup_id,),
        note="Scenario B: the duplicate's payment is booked at a perturbed rate",
    )
    return new_private, (issue_dup, issue_fx), (entry_dup, entry_fx)


__all__ = [
    "OP_A_PARTIAL",
    "OP_A_PERIOD",
    "OP_B_DUPLICATE",
    "OP_B_FX",
    "SCENARIOS",
    "SCENARIO_A_PROXIMITY_HOURS",
    "SCENARIO_A_REDUCTION_PCT",
    "ScenarioResult",
    "inject_scenario_a",
    "inject_scenario_b",
]
