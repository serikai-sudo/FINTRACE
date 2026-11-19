"""payment_allocation injector: four variants, decoy protection, independent check, determinism."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from datetime import datetime

import pytest

from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.injectors import payment as payment_injector
from fintrace.generator.injectors.base import append_truth_and_provenance, inconsistent_ids
from fintrace.generator.validate import ConsistencyError, validate_clean_history
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    InjectorParams,
    Issue,
    Payment,
    PrivateEpisode,
    ProvenanceEntry,
)

Injected = tuple[PrivateEpisode, Issue, ProvenanceEntry]
VariantFn = Callable[[PrivateEpisode, random.Random], Injected]
VARIANT_FNS: dict[str, VariantFn] = {
    "missing_allocation": payment_injector._inject_missing,
    "partial_allocation": payment_injector._inject_partial,
    "wrong_invoice_reference": payment_injector._inject_wrong_ref,
    "transposition_typo": payment_injector._inject_transposition,
}
VARIANTS = tuple(VARIANT_FNS)
CHI_SQUARED_3DF_P01 = 11.345


def _config(count: int = 1) -> EpisodeConfig:
    params = InjectorParams(count=count)
    return EpisodeConfig(errors=ErrorInjectionConfig(payment_allocation=params))


@pytest.fixture(scope="module")
def clean() -> PrivateEpisode:
    return build_clean_episode(42)


def _force(clean: PrivateEpisode, variant: str, start: int = 0) -> Injected:
    """Apply one variant with the first rng seed >= start that the variant accepts."""
    for rng_seed in range(start, start + 20):
        try:
            return VARIANT_FNS[variant](clean, random.Random(rng_seed))
        except payment_injector.InjectorError:
            continue
    raise AssertionError(f"{variant} rejected 20 rng seeds in a row")


def _payment(episode: PrivateEpisode, payment_id: str | None) -> Payment:
    return next(p for p in episode.public.payments if p.payment_id == payment_id)


@pytest.mark.parametrize("variant", VARIANTS)
def test_injector_adds_one_truth_issue(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    assert injected.truth.issues == ()  # inject() returns; build.py appends
    appended = append_truth_and_provenance(injected, issue, entry)
    assert len(appended.truth.issues) == len(clean.truth.issues) + 1
    assert issue.type == "payment_allocation" and entry.op == "payment_allocation"
    assert entry.modified["variant"] == variant == entry.original["variant"]
    assert entry.created_ids == () and entry.source_id in issue.affected_ids


@pytest.mark.parametrize("variant", VARIANTS)
def test_injector_does_not_add_new_documents(clean: PrivateEpisode, variant: str) -> None:
    injected, _, _ = _force(clean, variant)
    for attr in ("invoice_id", "doc_id"):
        before = {getattr(i, attr) for i in clean.public.invoices}
        assert before == {getattr(i, attr) for i in injected.public.invoices}
    assert {p.payment_id for p in clean.public.payments} == {
        p.payment_id for p in injected.public.payments
    }
    assert {p.doc_id for p in clean.public.payments} == {p.doc_id for p in injected.public.payments}
    assert injected.public.manifest == clean.public.manifest
    assert injected.public.invoices == clean.public.invoices


@pytest.mark.parametrize("variant", VARIANTS)
def test_independent_check_passes_on_injected(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    payment_injector.independent_check(injected.public, issue)  # no raise
    with pytest.raises(ConsistencyError):  # the clean books carry no such discrepancy
        payment_injector.independent_check(clean.public, issue)
    off_by_one = issue.model_copy(update={"expected_amount_minor": issue.expected_amount_minor + 1})
    with pytest.raises(ConsistencyError, match="payment_allocation_check"):
        payment_injector.independent_check(injected.public, off_by_one)
    other_payment = next(p for p in clean.public.payments if p.payment_id != entry.source_id)
    two_payments = issue.model_copy(
        update={"affected_ids": tuple(sorted((*issue.affected_ids, other_payment.payment_id)))}
    )
    with pytest.raises(ConsistencyError, match="one payment"):
        payment_injector.independent_check(injected.public, two_payments)
    assert entry.source_id is not None
    # the structural contract holds everywhere except on the one document the variant changed
    validate_clean_history(injected.public, ignore_ids=frozenset({entry.source_id}))
    with pytest.raises(ConsistencyError):
        validate_clean_history(injected.public)


@pytest.mark.parametrize("variant", VARIANTS)
def test_injector_preserves_double_entry_ledger(clean: PrivateEpisode, variant: str) -> None:
    injected, _, _ = _force(clean, variant)
    assert injected.public.ledger == clean.public.ledger
    assert injected.public.fx == clean.public.fx and injected.public.policy == clean.public.policy


@pytest.mark.slow
def test_200_seeds_build_and_validate() -> None:
    variants: Counter[str] = Counter()
    for seed in range(200):
        episode = build_episode(seed, _config(1))
        assert len(episode.truth.issues) == 1
        validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
        payment_injector.independent_check(episode.public, episode.truth.issues[0])
        assert episode.truth.episode_id == episode.public.manifest.episode_id
        variants[str(episode.provenance.injections[0].modified["variant"])] += 1
    assert set(variants) == set(VARIANTS)
    chi_squared = sum((n - 50) ** 2 / 50 for n in variants.values())
    assert chi_squared < CHI_SQUARED_3DF_P01, dict(variants)


@pytest.mark.parametrize("variant", VARIANTS)
def test_affected_ids_canonical_sorted(clean: PrivateEpisode, variant: str) -> None:
    _, issue, _ = _force(clean, variant)
    assert issue.affected_ids == tuple(sorted(set(issue.affected_ids)))
    assert issue.expected_amount_minor > 0 and len(issue.acceptable_evidence_sets) == 3


def test_missing_variant_empty_refs(clean: PrivateEpisode) -> None:
    injected, issue, entry = _force(clean, "missing_allocation")
    modified = _payment(injected, entry.source_id)
    assert modified.invoice_refs == ()
    original = _payment(clean, entry.source_id)
    assert len(original.invoice_refs) == 1
    assert modified.amount_minor == original.amount_minor  # only the allocation changed
    assert set(issue.affected_ids) == {original.payment_id, original.invoice_refs[0].invoice_id}


def test_missing_variant_expected_equals_amount(clean: PrivateEpisode) -> None:
    _, issue, entry = _force(clean, "missing_allocation")
    assert issue.expected_amount_minor == _payment(clean, entry.source_id).amount_minor


def test_partial_variant_allocated_reduced(clean: PrivateEpisode) -> None:
    injected, _, entry = _force(clean, "partial_allocation")
    before = _payment(clean, entry.source_id).invoice_refs[0].allocated_minor
    after = _payment(injected, entry.source_id).invoice_refs[0].allocated_minor
    assert 0 < after < before
    assert before // 2 <= after <= before - before // 10  # 10..50 % reduction
    assert 10 <= int(str(entry.modified["reduction_pct"])) <= 50


def test_partial_variant_expected_matches_diff(clean: PrivateEpisode) -> None:
    injected, issue, entry = _force(clean, "partial_allocation")
    modified = _payment(injected, entry.source_id)
    new_sum = sum(r.allocated_minor for r in modified.invoice_refs)
    assert issue.expected_amount_minor == modified.amount_minor - new_sum > 0


def test_wrong_ref_variant_three_affected_ids(clean: PrivateEpisode) -> None:
    injected, issue, entry = _force(clean, "wrong_invoice_reference")
    assert len(issue.affected_ids) == 3
    modified = _payment(injected, entry.source_id)
    original = _payment(clean, entry.source_id)
    wrong = modified.invoice_refs[0].invoice_id
    correct = original.invoice_refs[0].invoice_id
    assert wrong != correct and set(issue.affected_ids) == {modified.payment_id, wrong, correct}
    booked = {r.invoice_id for r in injected.public.ledger if r.payment_id == modified.payment_id}
    assert booked == {correct}  # the ledger still books the cash against the right invoice
    assert issue.expected_amount_minor == original.invoice_refs[0].allocated_minor


def test_wrong_ref_variant_currencies_match(clean: PrivateEpisode) -> None:
    injected, _, entry = _force(clean, "wrong_invoice_reference")
    modified = _payment(injected, entry.source_id)
    by_id = {i.invoice_id: i for i in injected.public.invoices}
    wrong = by_id[modified.invoice_refs[0].invoice_id]
    assert wrong.currency == modified.currency
    assert by_id[str(entry.original["invoice_id"])].currency == modified.currency
    # no single-document tell: the wrong invoice predates the payment, can absorb the
    # allocation, and is not a split-paid decoy
    assert datetime.fromisoformat(wrong.issued_at) <= datetime.fromisoformat(modified.timestamp)
    assert wrong.total_minor >= modified.invoice_refs[0].allocated_minor
    per_invoice = Counter(r.invoice_id for p in clean.public.payments for r in p.invoice_refs)
    assert per_invoice[wrong.invoice_id] <= 1


def test_transposition_variant_amount_changed(clean: PrivateEpisode) -> None:
    injected, issue, entry = _force(clean, "transposition_typo")
    before = _payment(clean, entry.source_id).invoice_refs[0].allocated_minor
    after = _payment(injected, entry.source_id).invoice_refs[0].allocated_minor
    assert after != before and issue.expected_amount_minor == abs(before - after) > 0
    a, b = str(before), str(after)
    assert len(a) == len(b)
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    assert len(diff) == 2 and diff[1] == diff[0] + 1  # exactly one adjacent pair swapped
    assert (a[diff[0]], a[diff[1]]) == (b[diff[1]], b[diff[0]])


def test_transposition_variant_no_leading_zero() -> None:
    seen = 0
    for seed in range(30):
        clean = build_clean_episode(seed)
        try:
            injected, _, entry = _force(clean, "transposition_typo")
        except AssertionError:
            continue
        after = _payment(injected, entry.source_id).invoice_refs[0].allocated_minor
        assert after > 0 and str(after)[0] != "0" and len(str(after)) >= 2
        seen += 1
    assert seen >= 25


def test_transposition_rejects_identity_pairs() -> None:
    apply = payment_injector._apply_transposition
    for rng_seed in range(20):
        rng = random.Random(rng_seed)
        assert apply(1100, rng) == 1010  # the only pair with distinct digits
        assert apply(11, rng) is None and apply(7, rng) is None and apply(1000, rng) is None
        assert apply(1450, rng) in {4150, 1540, 1405}
        result = apply(1023, rng)
        assert result in {1203, 1032} and str(result)[0] != "0"  # "0123" is rejected


def test_split_payments_never_selected() -> None:
    splits_seen = 0
    for seed in range(30):
        clean = build_clean_episode(seed)
        per_invoice = Counter(r.invoice_id for p in clean.public.payments for r in p.invoice_refs)
        split_invoices = {i for i, n in per_invoice.items() if n >= 2}
        splits_seen += bool(split_invoices)
        split_payments = {
            p.payment_id
            for p in clean.public.payments
            if p.invoice_refs[0].invoice_id in split_invoices
        }
        for variant in VARIANTS:
            try:
                _, issue, entry = _force(clean, variant)
            except AssertionError:
                continue
            assert entry.source_id not in split_payments
            assert not set(issue.affected_ids) & split_invoices
    assert splits_seen == 30  # the clean generator forces at least one split per episode


def test_count_2_produces_2_independent_issues() -> None:
    episode = build_episode(7, _config(2))
    first, second = episode.truth.issues
    assert first.issue_id != second.issue_id
    assert not set(first.affected_ids) & set(second.affected_ids)
    for issue in episode.truth.issues:
        payment_injector.independent_check(episode.public, issue)
    validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
    assert len(inconsistent_ids(episode)) == 2
    assert build_episode(7, _config(2)).model_dump_json() == episode.model_dump_json()
    assert build_episode(7).model_dump_json() == build_clean_episode(7).model_dump_json()
