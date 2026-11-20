"""Interacting scenarios A and B: two linked issues each, verifiable alone, no double count."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.injectors import duplicate, fx_error, payment, period
from fintrace.generator.injectors.base import InjectorError, inconsistent_ids
from fintrace.generator.validate import ConsistencyError, validate_clean_history
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    InjectorParams,
    InteractingScenarioParams,
    LedgerRow,
    PrivateEpisode,
)

CHECKS = {
    "duplicate_invoice": duplicate.independent_check,
    "payment_allocation": payment.independent_check,
    "fx_error": fx_error.independent_check,
    "period_misclassification": period.independent_check,
}
SEED_A = 42
SEED_B = 42


def _config(a: int = 0, b: int = 0) -> EpisodeConfig:
    interacting = InteractingScenarioParams(scenario_a=a, scenario_b=b)
    return EpisodeConfig(errors=ErrorInjectionConfig(interacting=interacting))


@pytest.fixture(scope="module")
def scenario_a() -> PrivateEpisode:
    return build_episode(SEED_A, _config(a=1))


@pytest.fixture(scope="module")
def scenario_b() -> PrivateEpisode:
    return build_episode(SEED_B, _config(b=1))


def _inside(iso: str, episode: PrivateEpisode) -> bool:
    manifest = episode.public.manifest
    start = datetime.fromisoformat(manifest.period_start)
    end = datetime.fromisoformat(manifest.period_end)
    return start <= datetime.fromisoformat(iso) < end


# ------------------------------------------------------------------ scenario A


def test_scenario_a_produces_two_issues(scenario_a: PrivateEpisode) -> None:
    clean = build_clean_episode(SEED_A)
    assert len(scenario_a.truth.issues) == len(clean.truth.issues) + 2
    assert len(scenario_a.provenance.injections) == 2
    assert _config(a=1).errors.total() == 2
    # scenario A modifies in place: no new documents or rows
    assert len(scenario_a.public.payments) == len(clean.public.payments)
    assert len(scenario_a.public.ledger) == len(clean.public.ledger)
    assert scenario_a.public.manifest.n_documents == clean.public.manifest.n_documents


def test_scenario_a_issue_types_correct(scenario_a: PrivateEpisode) -> None:
    partial, cutoff = scenario_a.truth.issues
    assert partial.type == "payment_allocation" and cutoff.type == "period_misclassification"
    entries = scenario_a.provenance.injections
    expected_ops = ["interacting_scenario_a_partial", "interacting_scenario_a_period"]
    assert [e.op for e in entries] == expected_ops
    assert entries[0].source_id == entries[1].source_id  # one payment carries both errors
    assert all(e.modified["scenario"] == "A" for e in entries)


def test_scenario_a_interacts_with_populated(scenario_a: PrivateEpisode) -> None:
    partial, cutoff = scenario_a.truth.issues
    assert partial.interacts_with == (cutoff.issue_id,)
    assert cutoff.interacts_with == (partial.issue_id,)
    assert partial.issue_id != cutoff.issue_id
    root, follower = scenario_a.provenance.injections
    assert root.caused_by == () and follower.caused_by == (partial.issue_id,)


def test_scenario_a_independent_checks_both_pass(scenario_a: PrivateEpisode) -> None:
    clean = build_clean_episode(SEED_A)
    for issue in scenario_a.truth.issues:
        CHECKS[issue.type](scenario_a.public, issue)  # no raise
        with pytest.raises(ConsistencyError):
            CHECKS[issue.type](clean.public, issue)  # the clean books carry neither error
    validate_clean_history(scenario_a, ignore_ids=inconsistent_ids(scenario_a))
    with pytest.raises(ConsistencyError):
        validate_clean_history(scenario_a.public)
    assert build_episode(SEED_A, _config(a=1)).model_dump_json() == scenario_a.model_dump_json()


def test_scenario_a_amounts_do_not_double_count(scenario_a: PrivateEpisode) -> None:
    partial, cutoff = scenario_a.truth.issues
    payment_id = scenario_a.provenance.injections[0].source_id
    paid = next(p for p in scenario_a.public.payments if p.payment_id == payment_id)
    base = scenario_a.public.manifest.base_currency
    assert partial.currency == cutoff.currency == paid.currency != base  # a foreign payment
    allocated = sum(r.allocated_minor for r in paid.invoice_refs)
    assert partial.expected_amount_minor == paid.amount_minor - allocated
    assert cutoff.expected_amount_minor == paid.amount_minor
    assert partial.expected_amount_minor != cutoff.expected_amount_minor
    # the remainder is 30..50 % of the payment: the two amounts never sum to a third figure
    lo, hi = paid.amount_minor * 30 // 100, paid.amount_minor * 50 // 100
    assert lo <= partial.expected_amount_minor <= hi
    # document still inside the period, posting outside; payment document untouched otherwise
    journal_id = next(x for x in cutoff.affected_ids if x != payment_id)
    rows = [r for r in scenario_a.public.ledger if r.journal_id == journal_id]
    assert len(rows) == 2 and {r.payment_id for r in rows} == {payment_id}
    assert _inside(paid.timestamp, scenario_a) and not _inside(rows[0].timestamp, scenario_a)
    assert all(r.timestamp.endswith("+05:00") for r in rows)


@pytest.mark.slow
def test_scenario_a_100_seeds() -> None:
    infeasible: list[int] = []
    for seed in range(100):
        try:
            episode = build_episode(seed, _config(a=1))
        except InjectorError:
            infeasible.append(seed)
            continue
        issues = episode.truth.issues
        assert [i.type for i in issues] == ["payment_allocation", "period_misclassification"]
        assert issues[0].interacts_with == (issues[1].issue_id,)
        assert issues[1].interacts_with == (issues[0].issue_id,)
        for issue in issues:
            CHECKS[issue.type](episode.public, issue)
        validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
    # tiny episodes without a foreign payment near a boundary; the same seeds every time
    assert infeasible == INFEASIBLE_A


INFEASIBLE_A = [14, 40, 45, 58, 78]  # of the first 100 seeds; 111..190 add seven more


# ------------------------------------------------------------------ scenario B


def test_scenario_b_produces_two_issues(scenario_b: PrivateEpisode) -> None:
    clean = build_clean_episode(SEED_B)
    assert len(scenario_b.truth.issues) == 2 and len(scenario_b.provenance.injections) == 2
    assert len(scenario_b.public.invoices) == len(clean.public.invoices) + 1
    assert len(scenario_b.public.payments) == len(clean.public.payments) + 1
    assert len(scenario_b.public.ledger) == len(clean.public.ledger) + 4
    manifest = scenario_b.public.manifest
    assert manifest.n_documents == clean.public.manifest.n_documents + 2
    assert manifest.n_ledger_rows == clean.public.manifest.n_ledger_rows + 4


def test_scenario_b_issue_types_correct(scenario_b: PrivateEpisode) -> None:
    dup, fx = scenario_b.truth.issues
    assert dup.type == "duplicate_invoice" and fx.type == "fx_error"
    entries = scenario_b.provenance.injections
    expected_ops = ["interacting_scenario_b_duplicate", "interacting_scenario_b_fx"]
    assert [e.op for e in entries] == expected_ops
    assert entries[0].modified["duplicate_paid"] is True
    created = set(entries[0].created_ids)
    assert set(fx.affected_ids) <= created | set(dup.affected_ids)  # fx sits on created records


def test_scenario_b_currencies_differ(scenario_b: PrivateEpisode) -> None:
    dup, fx = scenario_b.truth.issues
    base = scenario_b.public.manifest.base_currency
    assert dup.currency != base and fx.currency == base
    by_id = {i.invoice_id: i for i in scenario_b.public.invoices}
    original_id, duplicate_id = dup.affected_ids
    assert by_id[original_id].currency == by_id[duplicate_id].currency == dup.currency
    assert dup.expected_amount_minor == by_id[duplicate_id].total_minor
    # the FX gap is the difference on the booked payment rows, not the document total
    payment_id = next(x for x in fx.affected_ids if x.startswith("PAY-"))
    rows = [r for r in scenario_b.public.ledger if r.payment_id == payment_id]
    booked = {r.debit_minor + r.credit_minor for r in rows}
    assert len(rows) == 2 and len(booked) == 1 and {r.currency for r in rows} == {base}
    policy_base = int(str(scenario_b.provenance.injections[1].original["policy_base_minor"]))
    assert fx.expected_amount_minor == abs(booked.pop() - policy_base) > 0


def test_scenario_b_interacts_with_populated(scenario_b: PrivateEpisode) -> None:
    dup, fx = scenario_b.truth.issues
    assert dup.interacts_with == (fx.issue_id,) and fx.interacts_with == (dup.issue_id,)
    root, follower = scenario_b.provenance.injections
    assert root.caused_by == () and follower.caused_by == (dup.issue_id,)
    assert follower.source_id in fx.affected_ids  # the duplicate's payment


def test_scenario_b_independent_checks_both_pass(scenario_b: PrivateEpisode) -> None:
    clean = build_clean_episode(SEED_B)
    for issue in scenario_b.truth.issues:
        CHECKS[issue.type](scenario_b.public, issue)
        with pytest.raises(ConsistencyError):
            CHECKS[issue.type](clean.public, issue)
    validate_clean_history(scenario_b, ignore_ids=inconsistent_ids(scenario_b))
    with pytest.raises(ConsistencyError):
        validate_clean_history(scenario_b.public)
    assert build_episode(SEED_B, _config(b=1)).model_dump_json() == scenario_b.model_dump_json()


@pytest.mark.slow
def test_scenario_b_100_seeds() -> None:
    infeasible: list[int] = []
    for seed in range(100):
        try:
            episode = build_episode(seed, _config(b=1))
        except InjectorError:
            infeasible.append(seed)
            continue
        issues = episode.truth.issues
        assert [i.type for i in issues] == ["duplicate_invoice", "fx_error"]
        assert issues[0].interacts_with == (issues[1].issue_id,)
        assert issues[1].interacts_with == (issues[0].issue_id,)
        for issue in issues:
            CHECKS[issue.type](episode.public, issue)
        validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
    assert infeasible == INFEASIBLE_B


INFEASIBLE_B: list[int] = []


# ------------------------------------------------------------------ composition


def _all_checks(episode: PrivateEpisode) -> None:
    for issue in episode.truth.issues:
        CHECKS[issue.type](episode.public, issue)
    validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))


def _linked(episode: PrivateEpisode) -> set[frozenset[str]]:
    return {
        frozenset((issue.issue_id, other))
        for issue in episode.truth.issues
        for other in issue.interacts_with
    }


def test_scenarios_combine_with_single_injectors() -> None:
    config = EpisodeConfig(
        errors=ErrorInjectionConfig(
            duplicate_invoice=InjectorParams(count=1),
            payment_allocation=InjectorParams(count=1),
            fx_error=InjectorParams(count=1),
            period_misclassification=InjectorParams(count=1),
            interacting=InteractingScenarioParams(scenario_a=1, scenario_b=1),
        )
    )
    episode = build_episode(42, config)
    issues = episode.truth.issues
    assert len(issues) == 8 == config.errors.total()
    assert [i.type for i in issues[:4]] == [
        "duplicate_invoice", "payment_allocation", "fx_error", "period_misclassification",
    ]  # fmt: skip
    _all_checks(episode)
    # records overlap only inside a linked pair (the scenario's shared payment or duplicate)
    linked = _linked(episode)
    for k, first in enumerate(issues):
        for second in issues[k + 1 :]:
            shared = set(first.affected_ids) & set(second.affected_ids)
            if frozenset((first.issue_id, second.issue_id)) in linked:
                assert shared  # the pair shares its root record on purpose
            else:
                assert not shared, (first.issue_id, second.issue_id)
    entries = episode.provenance.injections
    created = [set(e.created_ids) for e in entries]
    assert all(not a & b for k, a in enumerate(created) for b in created[k + 1 :])
    relaxed = [set(map(str, e.modified.get("relaxed_ids", []))) for e in entries]  # type: ignore[arg-type]
    assert all(not a & b for k, a in enumerate(relaxed) for b in relaxed[k + 1 :])
    assert len({i.issue_id for i in issues}) == 8


def test_scenario_counts_of_two_use_distinct_records() -> None:
    episode = build_episode(42, _config(a=2, b=2))
    issues = episode.truth.issues
    assert [i.type for i in issues] == [
        "payment_allocation", "period_misclassification", "payment_allocation",
        "period_misclassification", "duplicate_invoice", "fx_error",
        "duplicate_invoice", "fx_error",
    ]  # fmt: skip
    _all_checks(episode)
    sources = [e.source_id for e in episode.provenance.injections]
    assert sources[0] == sources[1] and sources[2] == sources[3]  # one payment per scenario A
    assert sources[0] != sources[2]  # the second instance picks another payment
    assert len({sources[4], sources[6]}) == 2  # two different original invoices for B
    assert build_episode(42, _config(a=2, b=2)).model_dump_json() == episode.model_dump_json()


def _with_rows(episode: PrivateEpisode, rows: list[LedgerRow]) -> PrivateEpisode:
    by_id = {r.row_id: r for r in rows}
    ledger = tuple(by_id.get(r.row_id, r) for r in episode.public.ledger)
    public = episode.public.model_copy(update={"ledger": ledger})
    return episode.model_copy(update={"public": public})


def test_relaxed_payment_journals_keep_structural_checks(
    scenario_a: PrivateEpisode, scenario_b: PrivateEpisode
) -> None:
    """ignore_ids relaxes amount and (for scenario A) posting date only; everything else bites."""
    for episode in (scenario_a, scenario_b):
        follower = episode.provenance.injections[1]
        relaxed_ids = [str(x) for x in follower.modified["relaxed_ids"]]  # type: ignore[union-attr]
        rows = [r for r in episode.public.ledger if r.row_id in relaxed_ids]
        relaxed = inconsistent_ids(episode)
        assert len(rows) == 2
        first, second = rows
        swapped = [
            first.model_copy(update={"account": second.account}),
            second.model_copy(update={"account": first.account}),
        ]
        with pytest.raises(ConsistencyError, match="ledger_matches_documents"):
            validate_clean_history(_with_rows(episode, swapped), ignore_ids=relaxed)
        recoloured = [r.model_copy(update={"currency": "CHF"}) for r in rows]
        with pytest.raises(ConsistencyError):
            validate_clean_history(_with_rows(episode, recoloured), ignore_ids=relaxed)
    # scenario B's payment document is clean, so its (mispriced) journal must still carry the
    # payment date: a shifted posting is not covered by the FX relaxation
    follower = scenario_b.provenance.injections[1]
    relaxed_ids = [str(x) for x in follower.modified["relaxed_ids"]]  # type: ignore[union-attr]
    rows = [r for r in scenario_b.public.ledger if r.row_id in relaxed_ids]
    relaxed = inconsistent_ids(scenario_b)
    later = datetime.fromisoformat(rows[0].timestamp) + timedelta(hours=1)
    shifted = [r.model_copy(update={"timestamp": later.isoformat()}) for r in rows]
    with pytest.raises(ConsistencyError, match="ledger_matches_documents"):
        validate_clean_history(_with_rows(scenario_b, shifted), ignore_ids=relaxed)
