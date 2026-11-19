"""duplicate_invoice injector: structure, determinism, decoy protection, independent check."""

from __future__ import annotations

import random
import re
from datetime import datetime

import pytest

from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.injectors import duplicate
from fintrace.generator.validate import ConsistencyError, validate_clean_history
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    InjectorParams,
    Issue,
    PrivateEpisode,
    ProvenanceEntry,
)

PARAMS = InjectorParams(count=1)


def _config(count: int = 1, **over: object) -> EpisodeConfig:
    errors = ErrorInjectionConfig(duplicate_invoice=InjectorParams(count=count))
    return EpisodeConfig(errors=errors, **over)  # type: ignore[arg-type]


Injected = tuple[PrivateEpisode, PrivateEpisode, Issue, ProvenanceEntry]


def _inject(seed: int = 42, rng_seed: int = 1) -> Injected:
    clean = build_clean_episode(seed)
    injected, issue, entry = duplicate.inject(clean, random.Random(rng_seed), PARAMS)
    return clean, injected, issue, entry


def _pair(injected: PrivateEpisode, issue: Issue, entry: ProvenanceEntry) -> tuple:
    by_id = {i.invoice_id: i for i in injected.public.invoices}
    original = by_id[entry.source_id or ""]
    dup_id = next(i for i in issue.affected_ids if i != original.invoice_id)
    return original, by_id[dup_id]


@pytest.mark.parametrize(("rng_seed", "extra_rows", "extra_payments"), [(1, 2, 0), (4, 4, 1)])
def test_inject_adds_one_invoice_and_two_ledger_rows(
    rng_seed: int, extra_rows: int, extra_payments: int
) -> None:
    # rng seed 1 takes the plain branch, seed 4 the double-payment branch (both pinned)
    clean, injected, issue, entry = _inject(rng_seed=rng_seed)
    assert len(injected.public.invoices) == len(clean.public.invoices) + 1
    assert len(injected.public.ledger) - len(clean.public.ledger) == extra_rows
    assert len(injected.public.payments) - len(clean.public.payments) == extra_payments
    assert entry.modified["with_double_payment"] is bool(extra_payments)
    if extra_payments:  # a double payment means the original was paid as well
        paid = {ref.invoice_id for p in clean.public.payments for ref in p.invoice_refs}
        assert entry.source_id in paid
    manifest = injected.public.manifest
    assert manifest.n_ledger_rows == len(injected.public.ledger)
    assert manifest.n_documents == len(injected.public.invoices) + len(injected.public.payments)
    # the clean records are untouched: the injection only adds
    assert set(clean.public.ledger) <= set(injected.public.ledger)
    assert set(clean.public.payments) <= set(injected.public.payments)
    assert set(clean.public.invoices) <= set(injected.public.invoices)
    assert injected.truth.issues == () and issue.type == "duplicate_invoice"
    assert len(build_episode(42, _config(1)).truth.issues) == 1  # build.py appends the issue


def test_injected_duplicate_has_same_entity_total_currency_line_items_as_original() -> None:
    _, injected, issue, entry = _inject()
    original, dup = _pair(injected, issue, entry)
    assert (dup.entity_id, dup.total_minor, dup.currency) == (
        original.entity_id, original.total_minor, original.currency,
    )  # fmt: skip
    assert dup.line_items == original.line_items


def test_injected_duplicate_has_different_invoice_id_and_doc_id() -> None:
    clean, injected, issue, entry = _inject()
    original, dup = _pair(injected, issue, entry)
    clean_ids = {i.invoice_id for i in clean.public.invoices}
    clean_docs = {i.doc_id for i in clean.public.invoices}
    clean_docs |= {p.doc_id for p in clean.public.payments}
    assert dup.invoice_id != original.invoice_id and dup.invoice_id not in clean_ids
    assert dup.doc_id != original.doc_id and dup.doc_id not in clean_docs
    assert re.fullmatch(r"INV-[0-9]{4}", dup.invoice_id)
    assert re.fullmatch(r"DOC-[0-9]{4}", dup.doc_id)
    max_clean = max(int(i.invoice_id.split("-")[1]) for i in clean.public.invoices)
    assert int(dup.invoice_id.split("-")[1]) < max_clean  # in-range gap, not a trailing label
    new_rows = [r for r in injected.public.ledger if r.invoice_id == dup.invoice_id]
    assert len({r.row_id for r in new_rows}) == len(new_rows)
    assert all(re.fullmatch(r"L-[0-9]{6}", r.row_id) for r in new_rows)


def test_injected_duplicate_issued_at_shifted_within_0_to_10_days() -> None:
    for rng_seed in range(8):
        _, injected, issue, entry = _inject(rng_seed=rng_seed)
        original, dup = _pair(injected, issue, entry)
        delta = datetime.fromisoformat(dup.issued_at) - datetime.fromisoformat(original.issued_at)
        assert 0 <= delta.total_seconds() <= 10 * 86_400 and delta.seconds == 0  # whole days
        assert dup.issued_at[10:] == original.issued_at[10:]  # time of day + offset preserved
        due_gap = datetime.fromisoformat(dup.due_at) - datetime.fromisoformat(dup.issued_at)
        original_gap = datetime.fromisoformat(original.due_at) - datetime.fromisoformat(
            original.issued_at
        )
        assert due_gap == original_gap  # no clamping: the payment terms look like the original's


def test_injected_duplicate_issued_at_within_period() -> None:
    for seed in range(12):
        clean = build_clean_episode(seed)
        injected, issue, entry = duplicate.inject(clean, random.Random(seed), PARAMS)
        _, dup = _pair(injected, issue, entry)
        start = datetime.fromisoformat(clean.public.manifest.period_start)
        end = datetime.fromisoformat(clean.public.manifest.period_end)
        assert start <= datetime.fromisoformat(dup.issued_at) < end
        rows = injected.public.ledger
        assert all(start <= datetime.fromisoformat(r.timestamp) < end for r in rows)


def test_independent_check_passes_on_injected() -> None:
    _, injected, issue, _ = _inject()
    duplicate.independent_check(injected.public, issue)  # no raise
    validate_clean_history(injected.public)  # the books stay structurally consistent


def test_independent_check_fails_if_duplicate_removed() -> None:
    clean, injected, issue, entry = _inject()
    with pytest.raises(ConsistencyError) as caught:
        duplicate.independent_check(clean.public, issue)  # the duplicate never existed here
    assert caught.value.check_name == "duplicate_check"
    _, dup = _pair(injected, issue, entry)
    kept = tuple(i for i in injected.public.invoices if i.invoice_id != dup.invoice_id)
    without_dup = injected.public.model_copy(update={"invoices": kept})
    with pytest.raises(ConsistencyError, match="duplicate_check"):
        duplicate.independent_check(without_dup, issue)
    wrong_amount = issue.model_copy(
        update={"expected_amount_minor": issue.expected_amount_minor + 1}
    )
    with pytest.raises(ConsistencyError, match="overstatement"):
        duplicate.independent_check(injected.public, wrong_amount)


def test_recurring_invoices_never_selected_as_source() -> None:
    for seed in range(20):
        clean = build_clean_episode(seed, EpisodeConfig(recurring_supplier_count=3))
        recurring = {
            invoice_id
            for event in clean.provenance.history
            if event.get("op") == "generate_recurring_series"
            for invoice_id in event["invoice_ids"]  # type: ignore[union-attr]
        }
        assert len(recurring) >= 3
        _, issue, entry = duplicate.inject(clean, random.Random(seed), PARAMS)
        assert entry.source_id not in recurring
        assert not set(issue.affected_ids) & recurring


def test_issue_structure_correct() -> None:
    _, injected, issue, entry = _inject()
    original, dup = _pair(injected, issue, entry)
    assert issue.type == "duplicate_invoice"
    assert issue.affected_ids == tuple(sorted((original.invoice_id, dup.invoice_id)))
    assert len(issue.affected_ids) == 2
    assert issue.expected_amount_minor == original.total_minor
    assert issue.currency == original.currency
    assert len(issue.acceptable_evidence_sets) >= 1
    assert (original.doc_id, dup.doc_id) in issue.acceptable_evidence_sets
    assert re.fullmatch(r"ISS-[0-9]{4}", issue.issue_id)
    again_issue = _inject()[2]
    assert again_issue == issue  # same seed, same rng -> identical issue (and issue_id)


def test_provenance_recorded() -> None:
    _, injected, issue, entry = _inject()
    original, dup = _pair(injected, issue, entry)
    assert entry.op == "duplicate_invoice" and entry.issue_id == issue.issue_id
    assert entry.source_id == original.invoice_id
    created = set(entry.created_ids)
    assert {dup.invoice_id, dup.doc_id} <= created
    new_rows = {r.row_id for r in injected.public.ledger if r.invoice_id == dup.invoice_id}
    assert new_rows <= created
    assert entry.modified["invoice_count"] == 2 and entry.original["invoice_count"] == 1


@pytest.mark.slow
def test_200_seeds_build_and_validate() -> None:
    clean_ids = set()
    for seed in range(200):
        episode = build_episode(seed, _config(1))
        assert len(episode.truth.issues) == 1
        validate_clean_history(episode)
        duplicate.independent_check(episode.public, episode.truth.issues[0])
        assert episode.truth.episode_id == episode.public.manifest.episode_id
        clean_ids.add(episode.public.manifest.episode_id)
    assert len(clean_ids) == 200


def test_two_duplicates_are_disjoint_checkable_and_deterministic() -> None:
    episode = build_episode(7, _config(2))
    first, second = episode.truth.issues
    assert first.issue_id != second.issue_id
    assert not set(first.affected_ids) & set(second.affected_ids)
    injections = episode.provenance.injections
    assert len({e.source_id for e in injections}) == 2  # two different originals
    for entry, issue in zip(injections, episode.truth.issues, strict=True):
        assert entry.issue_id == issue.issue_id and entry.source_id in issue.affected_ids
    for issue in (first, second):
        duplicate.independent_check(episode.public, issue)
    validate_clean_history(episode)
    assert build_episode(7, _config(2)).model_dump_json() == episode.model_dump_json()
    clean = build_clean_episode(7)
    assert len(episode.public.invoices) == len(clean.public.invoices) + 2


def test_build_episode_with_zero_count_is_backward_compat() -> None:
    clean_json = build_clean_episode(42).model_dump_json()
    assert build_episode(42).model_dump_json() == clean_json
    explicit = EpisodeConfig(errors=ErrorInjectionConfig())
    assert build_episode(42, explicit).model_dump_json() == clean_json
    # the clean base does not depend on the error plan: same books, different id and truth only
    clean, injected = build_clean_episode(42), build_episode(42, _config(1))
    assert clean.provenance.seed_hash == injected.provenance.seed_hash
    assert clean.public.manifest.episode_id != injected.public.manifest.episode_id
    clean_ids = {i.invoice_id for i in clean.public.invoices}
    assert clean_ids < {i.invoice_id for i in injected.public.invoices}
