"""Payment allocation injector with four variants (blueprint 5.2 plus a transposition extension).

(a) ``missing_allocation``: the payment's invoice_refs are removed; cash left, the invoice
    looks unpaid.
(b) ``partial_allocation``: allocated_minor is cut by 10..50 %; the payment is under-allocated.
(c) ``wrong_invoice_reference``: the allocation points at another same-currency invoice (issued
    no later than the payment, total at least the allocation, never a split-paid decoy) while
    the ledger still books the cash against the right one.
(d) ``transposition_typo``: two adjacent digits of allocated_minor are swapped (1450 -> 1540),
    the classical bookkeeper data-entry error that audit standards list among the misstatement
    patterns to consider (PCAOB AS 2110). The result may exceed or fall short of the amount.

Only the payment document changes; the ledger is never modified. Split payments (several
payments on one invoice) are legitimate decoys and are never selected as a source. The variant
is recorded in provenance only.

Independent check: the payment is located through ``affected_ids``. With one affected invoice
(a, b, d) the payment's own figures must disagree by exactly ``expected_amount_minor``; with two
affected invoices (c) the payment document and the ledger must name different invoices, the two
affected ones, and the misallocated sum must equal ``expected_amount_minor``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Final, Literal

from fintrace.generator.validate import ConsistencyError
from fintrace.models import (
    InjectorParams,
    Invoice,
    InvoiceAllocation,
    Issue,
    IssueType,
    JsonObject,
    Payment,
    PrivateEpisode,
    ProvenanceEntry,
    PublicEpisode,
)

from .base import (
    InjectorError,
    derive_issue_id,
    find_other_invoice_same_currency,
    pick_eligible_payment,
)

issue_type: Final[IssueType] = "payment_allocation"

PaymentVariant = Literal[
    "missing_allocation", "partial_allocation", "wrong_invoice_reference", "transposition_typo"
]
VARIANTS: Final[tuple[PaymentVariant, ...]] = (
    "missing_allocation",
    "partial_allocation",
    "wrong_invoice_reference",
    "transposition_typo",
)
_PARTIAL_REDUCTION_PCT: Final[tuple[int, int]] = (10, 50)
_CHECK: Final[str] = "payment_allocation_check"

Injected = tuple[PrivateEpisode, Issue, ProvenanceEntry]
Refs = tuple[InvoiceAllocation, ...]


def _lookup_invoice(public: PublicEpisode, invoice_id: str) -> Invoice:
    for invoice in public.invoices:
        if invoice.invoice_id == invoice_id:
            return invoice
    raise InjectorError(f"invoice {invoice_id} not found")


def _replace_payment(
    payments: tuple[Payment, ...], payment_id: str, new_payment: Payment
) -> tuple[Payment, ...]:
    """Swap one payment in place; timestamp and id are unchanged, so canonical order holds."""
    return tuple(new_payment if p.payment_id == payment_id else p for p in payments)


def _modified_payment(payment: Payment, refs: Refs) -> Payment:
    """Copy of ``payment`` with new allocations, re-validated by the contract model."""
    dumped = {**payment.model_dump(), "invoice_refs": [ref.model_dump() for ref in refs]}
    return Payment.model_validate(dumped)


def _with_payment(episode: PrivateEpisode, payment: Payment, refs: Refs) -> PrivateEpisode:
    payments = _replace_payment(
        episode.public.payments, payment.payment_id, _modified_payment(payment, refs)
    )
    public = episode.public.model_copy(update={"payments": payments})
    return episode.model_copy(update={"public": public})


def _payment_row_ids(public: PublicEpisode, payment_id: str) -> tuple[str, ...]:
    return tuple(sorted(r.row_id for r in public.ledger if r.payment_id == payment_id))


def _issue(
    episode: PrivateEpisode,
    payment: Payment,
    affected: tuple[str, ...],
    expected: int,
    documents: tuple[Invoice, ...],
) -> Issue:
    """Issue with canonical affected ids and three evidence routes: docs, ids, ledger rows."""
    taken = {issue.issue_id for issue in episode.truth.issues}
    issue_id = derive_issue_id(episode.public.manifest.episode_id, payment.payment_id, taken)
    return Issue(
        issue_id=issue_id,
        type=issue_type,
        affected_ids=tuple(sorted(set(affected))),
        expected_amount_minor=expected,
        currency=payment.currency,
        acceptable_evidence_sets=(
            (payment.doc_id, *(invoice.doc_id for invoice in documents)),
            (payment.payment_id, *(invoice.invoice_id for invoice in documents)),
            _payment_row_ids(episode.public, payment.payment_id),
        ),
    )


def _entry(
    issue: Issue,
    payment: Payment,
    variant: PaymentVariant,
    original: JsonObject,
    modified: JsonObject,
    note: str,
) -> ProvenanceEntry:
    return ProvenanceEntry(
        issue_id=issue.issue_id,
        op=issue_type,
        source_id=payment.payment_id,
        created_ids=(),
        original={"variant": variant, **original},
        modified={"variant": variant, **modified},
        caused_by=(),
        note=note,
    )


def _inject_missing(episode: PrivateEpisode, rng: random.Random) -> Injected:
    """Variant (a): remove the allocations; the discrepancy is the whole payment amount."""
    payment = pick_eligible_payment(episode, rng)
    invoice = _lookup_invoice(episode.public, payment.invoice_refs[0].invoice_id)
    new_private = _with_payment(episode, payment, ())
    issue = _issue(
        episode, payment, (payment.payment_id, invoice.invoice_id), payment.amount_minor, (invoice,)
    )
    entry = _entry(
        issue,
        payment,
        "missing_allocation",
        {"invoice_refs_count": len(payment.invoice_refs), "invoice_id": invoice.invoice_id},
        {"invoice_refs_count": 0},
        "Allocation removed; cash out but the invoice looks unpaid",
    )
    return new_private, issue, entry


def _inject_partial(episode: PrivateEpisode, rng: random.Random) -> Injected:
    """Variant (b): cut the allocation by 10..50 % (integer percent, no float)."""
    payment = pick_eligible_payment(episode, rng)
    ref = payment.invoice_refs[0]
    invoice = _lookup_invoice(episode.public, ref.invoice_id)
    pct = rng.randint(*_PARTIAL_REDUCTION_PCT)
    new_allocated = ref.allocated_minor - ref.allocated_minor * pct // 100
    if not 1 <= new_allocated < ref.allocated_minor:
        raise InjectorError("partial reduction left the allocation unchanged or non-positive")
    refs: Refs = (
        InvoiceAllocation(invoice_id=ref.invoice_id, allocated_minor=new_allocated),
        *payment.invoice_refs[1:],
    )
    new_private = _with_payment(episode, payment, refs)
    expected = payment.amount_minor - sum(r.allocated_minor for r in refs)
    issue = _issue(episode, payment, (payment.payment_id, invoice.invoice_id), expected, (invoice,))
    entry = _entry(
        issue,
        payment,
        "partial_allocation",
        {"allocated_minor": ref.allocated_minor},
        {"allocated_minor": new_allocated, "reduction_pct": pct},
        f"Allocation reduced by {pct} %",
    )
    return new_private, issue, entry


def _inject_wrong_ref(episode: PrivateEpisode, rng: random.Random) -> Injected:
    """Variant (c): redirect the allocation to another same-currency invoice."""
    payment = pick_eligible_payment(episode, rng, currency_match_count=2)
    ref = payment.invoice_refs[0]
    correct = _lookup_invoice(episode.public, ref.invoice_id)
    wrong = find_other_invoice_same_currency(
        episode, {correct.invoice_id}, payment.currency, rng, payment=payment
    )
    refs: Refs = (
        InvoiceAllocation(invoice_id=wrong.invoice_id, allocated_minor=ref.allocated_minor),
        *payment.invoice_refs[1:],
    )
    new_private = _with_payment(episode, payment, refs)
    issue = _issue(
        episode,
        payment,
        (payment.payment_id, correct.invoice_id, wrong.invoice_id),
        ref.allocated_minor,
        (correct, wrong),
    )
    entry = _entry(
        issue,
        payment,
        "wrong_invoice_reference",
        {"invoice_id": correct.invoice_id},
        {"invoice_id": wrong.invoice_id},
        "Allocation redirected to an unrelated same-currency invoice",
    )
    return new_private, issue, entry


def _apply_transposition(amount: int, rng: random.Random) -> int | None:
    """Swap one adjacent digit pair of ``amount``; None when no swap gives a distinct valid number.

    A position qualifies when its two digits differ (otherwise the swap is the identity) and
    the result keeps a non-zero leading digit. Single-digit amounts have no pair at all.
    """
    digits = str(amount)
    positions = [
        i
        for i in range(len(digits) - 1)
        if digits[i] != digits[i + 1] and not (i == 0 and digits[1] == "0")
    ]
    if not positions:
        return None
    i = rng.choice(positions)
    return int(f"{digits[:i]}{digits[i + 1]}{digits[i]}{digits[i + 2:]}")


def _inject_transposition(episode: PrivateEpisode, rng: random.Random) -> Injected:
    """Variant (d): adjacent-digit transposition in allocated_minor (bookkeeper typo).

    The typo may land above or below the amount; ``expected_amount_minor`` is the absolute gap,
    so the direction is not recoverable from truth (provenance keeps both figures). An
    allocation above the invoice total is part of the error, not a tell: it is what the typo did.
    """
    payment = pick_eligible_payment(episode, rng)
    ref = payment.invoice_refs[0]
    invoice = _lookup_invoice(episode.public, ref.invoice_id)
    typo = _apply_transposition(ref.allocated_minor, rng)
    if typo is None:
        raise InjectorError("no adjacent digit pair to transpose; retry with another payment")
    refs: Refs = (
        InvoiceAllocation(invoice_id=ref.invoice_id, allocated_minor=typo),
        *payment.invoice_refs[1:],
    )
    new_private = _with_payment(episode, payment, refs)
    expected = abs(payment.amount_minor - sum(r.allocated_minor for r in refs))
    issue = _issue(episode, payment, (payment.payment_id, invoice.invoice_id), expected, (invoice,))
    entry = _entry(
        issue,
        payment,
        "transposition_typo",
        {"allocated_minor": ref.allocated_minor},
        {"allocated_minor": typo, "mechanism": "adjacent_digit_swap"},
        "PCAOB AS 2110 transposition pattern (digit swap)",
    )
    return new_private, issue, entry


_INJECTORS: Final[dict[PaymentVariant, Callable[[PrivateEpisode, random.Random], Injected]]] = {
    "missing_allocation": _inject_missing,
    "partial_allocation": _inject_partial,
    "wrong_invoice_reference": _inject_wrong_ref,
    "transposition_typo": _inject_transposition,
}


def inject(episode: PrivateEpisode, rng: random.Random, params: InjectorParams) -> Injected:
    """Pick one of the four variants uniformly, apply it and return (episode, issue, entry)."""
    del params  # count is handled by build.py; the module constants hold the knobs
    variant: PaymentVariant = rng.choice(VARIANTS)
    return _INJECTORS[variant](episode, rng)


def independent_check(public: PublicEpisode, issue: Issue) -> None:
    """Rediscover the allocation discrepancy from public files alone and compare with the truth.

    One affected invoice (a, b, d): ``|amount_minor - sum(allocated)|`` must be positive and
    equal ``expected_amount_minor``, and the invoice must be named by the payment document or by
    the ledger rows of the payment. Two affected invoices (c): the document references one of
    them, the ledger books the other, and the referenced sum equals ``expected_amount_minor``.
    """
    affected = set(issue.affected_ids)
    payments = [p for p in public.payments if p.payment_id in affected]
    if len(payments) != 1:
        raise ConsistencyError(_CHECK, "affected_ids must name one payment", sorted(affected))
    payment = payments[0]
    invoice_ids = sorted(affected - {payment.payment_id})
    known = {i.invoice_id for i in public.invoices}  # membership only
    if not 1 <= len(invoice_ids) <= 2 or any(i not in known for i in invoice_ids):
        raise ConsistencyError(_CHECK, "affected invoices missing or malformed", invoice_ids)
    if payment.currency != issue.currency:
        raise ConsistencyError(_CHECK, "currency mismatch between payment and truth")
    allocated_sum = sum(r.allocated_minor for r in payment.invoice_refs)
    referenced = sorted({r.invoice_id for r in payment.invoice_refs})
    booked = sorted(
        {
            r.invoice_id
            for r in public.ledger
            if r.payment_id == payment.payment_id and r.invoice_id is not None
        }
    )
    if len(invoice_ids) == 2:  # wrong reference: document and ledger disagree on the invoice
        if (
            len(referenced) != 1
            or len(booked) != 1
            or referenced == booked
            or sorted({referenced[0], booked[0]}) != invoice_ids
        ):
            raise ConsistencyError(_CHECK, "document and ledger must split the two invoices")
        if allocated_sum != issue.expected_amount_minor:
            raise ConsistencyError(
                _CHECK, f"misallocated {allocated_sum} != truth {issue.expected_amount_minor}"
            )
        return
    (invoice_id,) = invoice_ids
    if invoice_id not in referenced and invoice_id not in booked:
        raise ConsistencyError(_CHECK, "invoice neither referenced nor booked", invoice_ids)
    discrepancy = abs(payment.amount_minor - allocated_sum)
    if discrepancy == 0 or discrepancy != issue.expected_amount_minor:
        raise ConsistencyError(
            _CHECK, f"public discrepancy {discrepancy} != truth {issue.expected_amount_minor}"
        )


__all__ = ["VARIANTS", "PaymentVariant", "independent_check", "inject", "issue_type"]
