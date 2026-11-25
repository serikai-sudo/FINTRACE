"""FX error injector: a foreign-currency payment is booked in base currency at the wrong rate.

Posting model (history.py): a foreign-currency invoice is journaled in its own currency; the
conversion to base happens in the payment journal (AP/CASH or CASH/AR, both rows in base at the
rate of the policy date, ``policy.fx.rate_date_rule``). This injector rewrites both rows of one
such journal with the amount converted at a wrong rate; the journal stays balanced, timestamps,
ids, accounts and every other record are untouched.

Variants (uniform):
  (a) ``wrong_date``: the rate of a day 3..14 days before or after the policy date, inside the
      reporting period;
  (b) ``perturbed_rate``: the policy rate times a factor from a subtle set (0.95..0.98,
      1.02..1.05).
Both are capped at +-6 % of the policy rate: a wild rate would be a fingerprint, not an error.

Noise threshold: the discrepancy must be at least 1 % of the correct base amount AND at least
100 minor units; below that it dissolves into rounding noise and the draw is rejected (build.py
retries on a fresh RNG). The finding currency is the BASE currency.

Feasibility: an episode whose foreign payments are all tiny (a few currency units) or already
claimed by earlier injectors cannot host an FX error inside the cap and the floor; build.py then
raises InjectorError deterministically for that (seed, config). Dataset builders skip such seeds.
The perturbed rate is a keying error, so its implied rate is usually not a table rate; finding
it still means computing the implied rate of every foreign payment and comparing with the table.

Independent check: the policy rate from fx + policy alone, the correct amount via the shared
conversion rule, and ``|booked - correct|`` must equal ``expected_amount_minor``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final, Literal

from fintrace.generator.fx import build_fx_index, convert_minor, policy_date, rate_on
from fintrace.generator.validate import ConsistencyError
from fintrace.models import (
    InjectorParams,
    Invoice,
    Issue,
    IssueType,
    JsonObject,
    LedgerRow,
    Manifest,
    Payment,
    PrivateEpisode,
    ProvenanceEntry,
    PublicEpisode,
)

from .base import (
    RELAXED_IDS_KEY,
    InjectorError,
    derive_issue_id,
    payment_journal_rows,
    pick_foreign_currency_payment,
)

issue_type: Final[IssueType] = "fx_error"

FXVariant = Literal["wrong_date", "perturbed_rate"]
VARIANTS: Final[tuple[FXVariant, ...]] = ("wrong_date", "perturbed_rate")
# subtle, realistic factors only; anything outside [0.94, 1.06] would stand out as injected
PERTURB_FACTORS: Final[tuple[Decimal, ...]] = (
    Decimal("0.95"),
    Decimal("0.96"),
    Decimal("0.97"),
    Decimal("0.98"),
    Decimal("1.02"),
    Decimal("1.03"),
    Decimal("1.04"),
    Decimal("1.05"),
)
NOISE_THRESHOLD_FRAC: Final[Decimal] = Decimal("0.01")  # 1 % of the correct base amount
NOISE_THRESHOLD_MIN_MINOR: Final[int] = 100
WRONG_DATE_SHIFT_DAYS: Final[tuple[int, int]] = (3, 14)
MAX_RATE_RATIO_DEVIATION: Final[Decimal] = Decimal("0.06")
_RATE_QUANTUM: Final[Decimal] = Decimal("0.0001")
_ONE: Final[Decimal] = Decimal(1)
_CHECK: Final[str] = "fx_error_check"

Injected = tuple[PrivateEpisode, Issue, ProvenanceEntry]


def _find_invoice(public: PublicEpisode, invoice_id: str) -> Invoice | None:
    return next((i for i in public.invoices if i.invoice_id == invoice_id), None)


def _policy_rate(public: PublicEpisode, payment: Payment, invoice: Invoice) -> Decimal:
    """The authoritative rate: policy date rule over (invoice, payment), fx table, fallback."""
    day = policy_date(public.policy, invoice.issued_at, payment.timestamp)
    base = public.manifest.base_currency
    return rate_on(build_fx_index(public.fx), day, payment.currency, base, public.policy)


def noise_floor(policy_base: int) -> int:
    """max(100 minor units, 1 % of the correct base amount, rounded half-even).

    The published bar for calling a conversion gap an error rather than rounding. It is shared:
    ``payment.py`` refuses a misallocation only when it would push a journal across this same
    floor, so the generator never rejects a draw the audit would not have flagged either.
    """
    scaled = (Decimal(policy_base) * NOISE_THRESHOLD_FRAC).quantize(_ONE, rounding=ROUND_HALF_EVEN)
    return max(NOISE_THRESHOLD_MIN_MINOR, int(scaled))


def _passes_noise_floor(discrepancy: int, policy_base: int) -> bool:
    return discrepancy >= noise_floor(policy_base)


def _rate_ratio_ok(wrong_rate: Decimal, policy_rate: Decimal) -> bool:
    """Distinct from the policy rate and within the +-6 % cap."""
    if wrong_rate == policy_rate or policy_rate <= 0:
        return False
    return abs(wrong_rate / policy_rate - _ONE) <= MAX_RATE_RATIO_DEVIATION


def _wrong_date_candidates(policy_day: str, manifest: Manifest) -> list[str]:
    """Days 3..14 before or after the policy day that fall inside the reporting period."""
    anchor = date.fromisoformat(policy_day)
    start = date.fromisoformat(manifest.period_start[:10])
    end = date.fromisoformat(manifest.period_end[:10])
    lo, hi = WRONG_DATE_SHIFT_DAYS
    shifts = (*range(-hi, -lo + 1), *range(lo, hi + 1))
    days = (anchor + timedelta(days=k) for k in shifts)
    return [day.isoformat() for day in days if start <= day <= end]


def _pick_wrong_date_rate(
    public: PublicEpisode,
    payment: Payment,
    invoice: Invoice,
    policy_rate: Decimal,
    policy_base: int,
    rng: random.Random,
) -> tuple[str, Decimal, int]:
    """(wrong_day, wrong_rate, wrong_base): first shuffled candidate that clears every guard."""
    policy_day = policy_date(public.policy, invoice.issued_at, payment.timestamp)
    candidates = _wrong_date_candidates(policy_day, public.manifest)
    index = build_fx_index(public.fx)
    base = public.manifest.base_currency
    for day in rng.sample(candidates, len(candidates)):
        rate = rate_on(index, day, payment.currency, base, public.policy)
        if not _rate_ratio_ok(rate, policy_rate):
            continue
        wrong_base = convert_minor(payment.amount_minor, rate, public.policy)
        if _passes_noise_floor(abs(wrong_base - policy_base), policy_base):
            return day, rate, wrong_base
    raise InjectorError("no wrong date clears the noise floor; retry with another payment")


def _perturbed_rate(
    policy_rate: Decimal, rng: random.Random, factors: Sequence[Decimal]
) -> tuple[Decimal, Decimal]:
    """(wrong_rate, factor): the policy rate scaled by one factor, quantized like the fx table."""
    factor = rng.choice(list(factors))
    rate = (policy_rate * factor).quantize(_RATE_QUANTUM)
    if not _rate_ratio_ok(rate, policy_rate):
        raise InjectorError("perturbation collapsed onto the policy rate or left the cap")
    return rate, factor


def _rewrite_rows(rows: Sequence[LedgerRow], amount: int) -> tuple[LedgerRow, ...]:
    """The same rows with the booked amount replaced on their debit or credit side."""
    rewritten: list[LedgerRow] = []
    for row in rows:
        dumped = row.model_dump()
        dumped["debit_minor"] = amount if row.debit_minor > 0 else 0
        dumped["credit_minor"] = amount if row.credit_minor > 0 else 0
        rewritten.append(LedgerRow.model_validate(dumped))
    return tuple(rewritten)


def _with_rows(episode: PrivateEpisode, rows: Sequence[LedgerRow]) -> PrivateEpisode:
    """Replace rows in place; timestamps and ids are unchanged, so canonical order holds."""
    by_id = {row.row_id: row for row in rows}
    ledger = tuple(by_id.get(row.row_id, row) for row in episode.public.ledger)
    public = episode.public.model_copy(update={"ledger": ledger})
    return episode.model_copy(update={"public": public})


def _inject_variant(
    episode: PrivateEpisode,
    rng: random.Random,
    variant: FXVariant,
    *,
    factors: Sequence[Decimal] = PERTURB_FACTORS,
) -> Injected:
    """Apply one variant to a clean foreign-currency payment journal."""
    public = episode.public
    base = public.manifest.base_currency
    payment = pick_foreign_currency_payment(episode, rng)
    invoice = _find_invoice(public, payment.invoice_refs[0].invoice_id)
    if invoice is None:
        raise InjectorError(f"invoice of payment {payment.payment_id} not found")
    rows = payment_journal_rows(public, payment.payment_id)
    policy_day = policy_date(public.policy, invoice.issued_at, payment.timestamp)
    policy_rate = _policy_rate(public, payment, invoice)
    policy_base = convert_minor(payment.amount_minor, policy_rate, public.policy)
    if any(row.debit_minor + row.credit_minor != policy_base for row in rows):
        raise InjectorError("clean journal does not match the policy conversion")
    detail: JsonObject
    if variant == "wrong_date":
        wrong_day, wrong_rate, wrong_base = _pick_wrong_date_rate(
            public, payment, invoice, policy_rate, policy_base, rng
        )
        detail = {"wrong_date": wrong_day}
    else:
        wrong_rate, factor = _perturbed_rate(policy_rate, rng, factors)
        wrong_base = convert_minor(payment.amount_minor, wrong_rate, public.policy)
        detail = {"factor": str(factor)}
    discrepancy = abs(wrong_base - policy_base)
    if not _passes_noise_floor(discrepancy, policy_base):
        floor = noise_floor(policy_base)
        raise InjectorError(f"discrepancy {discrepancy} below the noise floor {floor}")

    new_private = _with_rows(episode, _rewrite_rows(rows, wrong_base))
    taken = {issue.issue_id for issue in episode.truth.issues}
    issue_id = derive_issue_id(public.manifest.episode_id, payment.payment_id, taken)
    row_ids = tuple(row.row_id for row in rows)
    issue = Issue(
        issue_id=issue_id,
        type=issue_type,
        affected_ids=tuple(sorted((invoice.invoice_id, payment.payment_id))),
        expected_amount_minor=discrepancy,
        currency=base,  # the misstatement lives in the base-currency books
        acceptable_evidence_sets=(
            (payment.doc_id, invoice.doc_id),
            (payment.payment_id, invoice.invoice_id),
            row_ids,
        ),
    )
    entry = ProvenanceEntry(
        issue_id=issue_id,
        op=issue_type,
        source_id=payment.payment_id,
        created_ids=(),
        original={
            "variant": variant,
            "policy_date": policy_day,
            "policy_rate": str(policy_rate),
            "policy_base_minor": policy_base,
        },
        modified={
            "variant": variant,
            "wrong_rate": str(wrong_rate),
            "wrong_base_minor": wrong_base,
            RELAXED_IDS_KEY: list(row_ids),
            **detail,
        },
        caused_by=(),
        note=f"FX {variant}: discrepancy {discrepancy} {base}",
    )
    return new_private, issue, entry


def inject(episode: PrivateEpisode, rng: random.Random, params: InjectorParams) -> Injected:
    """Pick a variant uniformly, apply it and return (episode, issue, entry)."""
    del params  # count is handled by build.py; the module constants hold the knobs
    variant: FXVariant = rng.choice(VARIANTS)
    return _inject_variant(episode, rng, variant)


def independent_check(public: PublicEpisode, issue: Issue) -> None:
    """Recompute the FX discrepancy from fx + policy + documents + ledger alone.

    The payment is located through ``affected_ids`` and the other affected id must be its
    invoice. Its journal must be two base-currency rows carrying one amount; that amount minus
    the policy conversion of the payment amount, in absolute value, must be positive and equal
    ``expected_amount_minor``; the issue currency must be the base currency.
    """
    affected = set(issue.affected_ids)
    payments = [p for p in public.payments if p.payment_id in affected]
    if len(payments) != 1:
        raise ConsistencyError(_CHECK, "affected_ids must name one payment", sorted(affected))
    payment = payments[0]
    others = sorted(affected - {payment.payment_id})
    refs = payment.invoice_refs
    if len(others) != 1 or len(refs) != 1 or refs[0].invoice_id != others[0]:
        raise ConsistencyError(_CHECK, "affected invoice must be the payment's invoice", others)
    invoice = _find_invoice(public, others[0])
    if invoice is None:
        raise ConsistencyError(_CHECK, "affected invoice not found", others)
    base = public.manifest.base_currency
    if issue.currency != base:
        raise ConsistencyError(_CHECK, "finding currency must be the base currency")
    if payment.currency == base:
        raise ConsistencyError(_CHECK, "payment is not a foreign-currency payment")
    rows = payment_journal_rows(public, payment.payment_id)
    row_ids = [row.row_id for row in rows]
    if len(rows) != 2 or any(row.currency != base for row in rows):
        raise ConsistencyError(_CHECK, "payment journal must be two base-currency rows", row_ids)
    booked = sorted({row.debit_minor + row.credit_minor for row in rows})
    if len(booked) != 1:
        raise ConsistencyError(_CHECK, "payment journal rows carry different amounts", row_ids)
    try:
        policy_rate = _policy_rate(public, payment, invoice)
    except LookupError as exc:
        raise ConsistencyError(_CHECK, "no policy rate for the payment") from exc
    policy_base = convert_minor(payment.amount_minor, policy_rate, public.policy)
    discrepancy = abs(booked[0] - policy_base)
    if discrepancy == 0 or discrepancy != issue.expected_amount_minor:
        raise ConsistencyError(
            _CHECK, f"public discrepancy {discrepancy} != truth {issue.expected_amount_minor}"
        )


__all__ = [
    "MAX_RATE_RATIO_DEVIATION",
    "NOISE_THRESHOLD_FRAC",
    "NOISE_THRESHOLD_MIN_MINOR",
    "PERTURB_FACTORS",
    "VARIANTS",
    "WRONG_DATE_SHIFT_DAYS",
    "FXVariant",
    "independent_check",
    "inject",
    "issue_type",
    "noise_floor",
]
