"""fx_error injector: wrong_date / perturbed_rate, noise threshold, balance, no wild rates."""

from __future__ import annotations

import random
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.injectors import fx_error
from fintrace.generator.injectors.base import (
    InjectorError,
    append_truth_and_provenance,
    inconsistent_ids,
)
from fintrace.generator.validate import ConsistencyError, check_balanced, validate_clean_history
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
VARIANTS = fx_error.VARIANTS
CHI_SQUARED_1DF_P01 = 6.63
RATIO_LO, RATIO_HI = Decimal("0.94"), Decimal("1.06")


def _config(count: int = 1) -> EpisodeConfig:
    return EpisodeConfig(errors=ErrorInjectionConfig(fx_error=InjectorParams(count=count)))


@pytest.fixture(scope="module")
def clean() -> PrivateEpisode:
    return build_clean_episode(42)


def _force(clean: PrivateEpisode, variant: str, start: int = 0) -> Injected:
    """Apply one variant with the first rng seed >= start that clears the guards."""
    for rng_seed in range(start, start + 30):
        try:
            return fx_error._inject_variant(clean, random.Random(rng_seed), variant)  # type: ignore[arg-type]
        except InjectorError:
            continue
    raise AssertionError(f"{variant} rejected 30 rng seeds in a row")


def _payment(episode: PrivateEpisode, payment_id: str | None) -> Payment:
    return next(p for p in episode.public.payments if p.payment_id == payment_id)


def _rows(episode: PrivateEpisode, payment_id: str | None) -> list:
    return [r for r in episode.public.ledger if r.payment_id == payment_id]


def _ratio(entry: ProvenanceEntry) -> Decimal:
    return Decimal(str(entry.modified["wrong_rate"])) / Decimal(str(entry.original["policy_rate"]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_fx_error_adds_one_truth_issue(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    assert injected.truth.issues == ()  # inject() returns; build.py appends
    appended = append_truth_and_provenance(injected, issue, entry)
    assert len(appended.truth.issues) == len(clean.truth.issues) + 1
    assert issue.type == "fx_error" == entry.op
    assert entry.modified["variant"] == variant == entry.original["variant"]
    assert entry.created_ids == () and entry.source_id in issue.affected_ids


@pytest.mark.parametrize("variant", VARIANTS)
def test_fx_error_currency_is_base(clean: PrivateEpisode, variant: str) -> None:
    _, issue, entry = _force(clean, variant)
    base = clean.public.manifest.base_currency
    assert issue.currency == base == "USD"
    assert _payment(clean, entry.source_id).currency != base  # a foreign-currency payment


@pytest.mark.parametrize("variant", VARIANTS)
def test_fx_error_affected_ids_invoice_and_row(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    payment = _payment(clean, entry.source_id)
    assert len(issue.affected_ids) == 2
    assert set(issue.affected_ids) == {payment.payment_id, payment.invoice_refs[0].invoice_id}
    assert issue.affected_ids == tuple(sorted(issue.affected_ids))
    row_ids = tuple(sorted(r.row_id for r in _rows(injected, payment.payment_id)))
    assert len(row_ids) == 2 and row_ids in issue.acceptable_evidence_sets
    assert entry.modified["relaxed_ids"] == list(row_ids)


@pytest.mark.parametrize("variant", VARIANTS)
def test_independent_check_passes_on_injected(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    fx_error.independent_check(injected.public, issue)  # no raise
    with pytest.raises(ConsistencyError, match="fx_error_check"):
        fx_error.independent_check(clean.public, issue)  # clean books: no discrepancy
    expected = issue.expected_amount_minor
    off_by_one = issue.model_copy(update={"expected_amount_minor": expected + 1})
    with pytest.raises(ConsistencyError, match="fx_error_check"):
        fx_error.independent_check(injected.public, off_by_one)
    foreign = _payment(clean, entry.source_id).currency
    wrong_currency = issue.model_copy(update={"currency": foreign})
    with pytest.raises(ConsistencyError, match="base currency"):
        fx_error.independent_check(injected.public, wrong_currency)
    # the clean contract holds everywhere except on the two rewritten rows
    relaxed = frozenset(str(x) for x in entry.modified["relaxed_ids"])  # type: ignore[union-attr]
    validate_clean_history(injected.public, ignore_ids=relaxed)
    with pytest.raises(ConsistencyError, match="ledger_matches_documents"):
        validate_clean_history(injected.public)


def test_wrong_date_variant_date_within_shift_bounds(clean: PrivateEpisode) -> None:
    lo, hi = fx_error.WRONG_DATE_SHIFT_DAYS
    for start in (0, 10, 20):
        _, _, entry = _force(clean, "wrong_date", start)
        policy_day = date.fromisoformat(str(entry.original["policy_date"]))
        wrong_day = date.fromisoformat(str(entry.modified["wrong_date"]))
        assert lo <= abs((wrong_day - policy_day).days) <= hi
        assert entry.modified["wrong_rate"] != entry.original["policy_rate"]


def test_wrong_date_variant_within_period(clean: PrivateEpisode) -> None:
    start = date.fromisoformat(clean.public.manifest.period_start[:10])
    end = date.fromisoformat(clean.public.manifest.period_end[:10])
    for offset in (0, 10, 20):
        _, _, entry = _force(clean, "wrong_date", offset)
        wrong_day = date.fromisoformat(str(entry.modified["wrong_date"]))
        assert start <= wrong_day <= end
        rates = {fx.date for fx in clean.public.fx}  # the wrong rate really is a table rate
        assert wrong_day.isoformat() in rates


def test_perturbed_variant_factor_in_realistic_range(clean: PrivateEpisode) -> None:
    for start in (0, 10, 20):
        _, _, entry = _force(clean, "perturbed_rate", start)
        factor = Decimal(str(entry.modified["factor"]))
        assert factor in fx_error.PERTURB_FACTORS  # strict membership
        policy_rate = Decimal(str(entry.original["policy_rate"]))
        expected_rate = (policy_rate * factor).quantize(Decimal("0.0001"))
        assert Decimal(str(entry.modified["wrong_rate"])) == expected_rate


def test_noise_threshold_rejects_small_discrepancy(clean: PrivateEpisode) -> None:
    # public, because payment.py's misallocation guard stands on this same floor
    assert fx_error.noise_floor(1_000) == 100  # the 100-minor floor dominates
    assert fx_error.noise_floor(50_000) == 500  # 1 % dominates
    assert fx_error.noise_floor(12_345) == 123
    assert fx_error._passes_noise_floor(123, 12_345)
    assert not fx_error._passes_noise_floor(122, 12_345)
    tiny = (Decimal("1.001"),)  # 0.1 %: always inside rounding noise
    for rng_seed in range(10):
        with pytest.raises(InjectorError):
            fx_error._inject_variant(clean, random.Random(rng_seed), "perturbed_rate", factors=tiny)


@pytest.mark.parametrize("variant", VARIANTS)
def test_double_entry_still_balanced_after_injection(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    check_balanced(injected.public.ledger)  # no raise
    rows = _rows(injected, entry.source_id)
    wrong_base = int(str(entry.modified["wrong_base_minor"]))
    assert {r.debit_minor + r.credit_minor for r in rows} == {wrong_base}
    policy_base = int(str(entry.original["policy_base_minor"]))
    assert issue.expected_amount_minor == abs(wrong_base - policy_base)
    assert {r.debit_minor + r.credit_minor for r in _rows(clean, entry.source_id)} == {policy_base}


@pytest.mark.slow
def test_200_seeds_build_and_validate() -> None:
    variants: Counter[str] = Counter()
    infeasible: list[int] = []
    for seed in range(200):
        try:
            episode = build_episode(seed, _config(1))
        except InjectorError:
            # seed 145 has three foreign payments of 1.10..11.60 GBP: no rate inside the +-6 %
            # cap can move them by 100 minor units, so the floor rejects every draw, loudly
            infeasible.append(seed)
            continue
        assert len(episode.truth.issues) == 1
        validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
        fx_error.independent_check(episode.public, episode.truth.issues[0])
        assert episode.truth.episode_id == episode.public.manifest.episode_id
        assert episode.truth.issues[0].currency == episode.public.manifest.base_currency
        variants[str(episode.provenance.injections[0].modified["variant"])] += 1
    assert set(variants) == set(VARIANTS)
    assert infeasible == [145]  # deterministic: the same seed, the same clean InjectorError
    assert sum(variants.values()) == 199


@pytest.mark.slow
def test_wild_rate_never_used() -> None:
    irregular = 0
    for seed in range(50):
        entry = build_episode(seed, _config(1)).provenance.injections[0]
        assert RATIO_LO <= _ratio(entry) <= RATIO_HI, entry.modified
        irregular += int(str(entry.modified["wrong_base_minor"])) % 100 != 0
    assert irregular >= 45  # booked amounts are products of rate and amount, never round figures


@pytest.mark.parametrize("variant", VARIANTS)
def test_counterpart_row_also_mutated(clean: PrivateEpisode, variant: str) -> None:
    injected, _, entry = _force(clean, variant)
    changed = [
        (before, after)
        for before, after in zip(clean.public.ledger, injected.public.ledger, strict=True)
        if before != after
    ]
    assert len(changed) == 2  # both rows of the one journal, nothing else
    assert {after.payment_id for _, after in changed} == {entry.source_id}
    assert len({after.journal_id for _, after in changed}) == 1
    for before, after in changed:
        assert (before.row_id, before.timestamp, before.account, before.currency) == (
            after.row_id, after.timestamp, after.account, after.currency,
        )  # fmt: skip
        assert (before.debit_minor > 0) == (after.debit_minor > 0)  # side preserved
    assert injected.public.manifest == clean.public.manifest
    assert injected.public.payments == clean.public.payments
    assert injected.public.invoices == clean.public.invoices


def test_fx_error_requires_foreign_currency_invoice() -> None:
    config = EpisodeConfig(foreign_currencies=())
    clean = build_clean_episode(3, config)
    assert all(p.currency == clean.public.manifest.base_currency for p in clean.public.payments)
    with pytest.raises(InjectorError):
        fx_error.inject(clean, random.Random(0), InjectorParams())
    with_fx = config.model_copy(update={"errors": _config(1).errors})
    with pytest.raises(InjectorError):
        build_episode(3, with_fx)


@pytest.mark.parametrize("variant", VARIANTS)
def test_relaxed_rows_keep_structural_checks(clean: PrivateEpisode, variant: str) -> None:
    """check_amount=False relaxes only the amount: accounts and timestamps are still enforced."""
    injected, _, entry = _force(clean, variant)
    relaxed = frozenset(str(x) for x in entry.modified["relaxed_ids"])  # type: ignore[union-attr]
    rows = _rows(injected, entry.source_id)
    first, second = rows
    swapped = [
        first.model_copy(update={"account": second.account}),
        second.model_copy(update={"account": first.account}),
    ]
    by_id = {r.row_id: r for r in swapped}
    ledger = tuple(by_id.get(r.row_id, r) for r in injected.public.ledger)
    broken = injected.public.model_copy(update={"ledger": ledger})
    with pytest.raises(ConsistencyError, match="ledger_matches_documents"):
        validate_clean_history(broken, ignore_ids=relaxed)
    later = (datetime.fromisoformat(first.timestamp) + timedelta(seconds=1)).isoformat()
    shifted = first.model_copy(update={"timestamp": later})
    ledger = tuple(shifted if r.row_id == first.row_id else r for r in injected.public.ledger)
    broken = injected.public.model_copy(update={"ledger": ledger})
    with pytest.raises(ConsistencyError):  # balanced (one timestamp per journal) or journal check
        validate_clean_history(broken, ignore_ids=relaxed)


@pytest.mark.slow
def test_combined_config_infeasible_seeds_are_deterministic() -> None:
    """duplicate+payment+fx on seeds 0..199: the few seeds without a usable foreign payment fail
    loudly with InjectorError, always the same ones; every other seed builds and checks."""
    config = EpisodeConfig(
        errors=ErrorInjectionConfig(
            duplicate_invoice=InjectorParams(count=1),
            payment_allocation=InjectorParams(count=1),
            fx_error=InjectorParams(count=1),
        )
    )
    infeasible: list[int] = []
    for seed in range(200):
        try:
            episode = build_episode(seed, config)
        except InjectorError:
            infeasible.append(seed)
            continue
        assert [i.type for i in episode.truth.issues] == [
            "duplicate_invoice", "payment_allocation", "fx_error",
        ]  # fmt: skip
        fx_error.independent_check(episode.public, episode.truth.issues[2])
    assert infeasible == [40, 46, 145]  # two foreign payments consumed, or all below the floor


@pytest.mark.slow
def test_variant_distribution_uniform() -> None:
    variants: Counter[str] = Counter(
        str(build_episode(seed, _config(1)).provenance.injections[0].modified["variant"])
        for seed in range(100)
    )
    assert set(variants) == set(VARIANTS)
    chi_squared = sum((n - 50) ** 2 / 50 for n in variants.values())
    assert chi_squared < CHI_SQUARED_1DF_P01, dict(variants)
