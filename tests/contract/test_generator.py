"""Contract tests for build_clean_episode: validity, determinism, structure, Benford, timing."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from itertools import pairwise

import pytest

from fintrace.constants import BENFORD_CHI_SQUARED_CRITICAL, MAX_LEDGER_ROWS, MIN_LEDGER_ROWS
from fintrace.generator.build import build_clean_episode
from fintrace.generator.fx import build_fx_index, convert_minor, policy_date, rate_on
from fintrace.generator.validate import check_benford, validate_clean_history
from fintrace.models import EpisodeConfig, PrivateEpisode

SWEEP_SEEDS = 1000
BENFORD_SEEDS = 100


@pytest.fixture(scope="module")
def default_episode() -> PrivateEpisode:
    return build_clean_episode(seed=42)


def test_build_clean_passes_validation_1000_seeds() -> None:
    config = EpisodeConfig()
    started = time.monotonic()
    for seed in range(SWEEP_SEEDS):
        episode = build_clean_episode(seed=seed, config=config)
        validate_clean_history(episode)
        n_rows = len(episode.public.ledger)
        assert MIN_LEDGER_ROWS <= n_rows <= MAX_LEDGER_ROWS + 2, seed  # +2: odd misc fill-up
        assert episode.truth.issues == () and episode.provenance.injections == ()
    elapsed = time.monotonic() - started
    assert elapsed < 60.0, f"{SWEEP_SEEDS} seeds took {elapsed:.1f}s"


def test_deterministic_regeneration(default_episode: PrivateEpisode) -> None:
    first = build_clean_episode(seed=42, config=EpisodeConfig()).model_dump_json()
    second = build_clean_episode(seed=42).model_dump_json()
    assert first == second == default_episode.model_dump_json()
    other_config = build_clean_episode(seed=42, config=EpisodeConfig(period_days=91))
    assert other_config.model_dump_json() != first  # config is part of the determinism key


def test_different_seeds_different_episodes() -> None:
    one, two = build_clean_episode(seed=1), build_clean_episode(seed=2)
    assert one.public.manifest.episode_id != two.public.manifest.episode_id
    assert one.public.manifest.episode_id.startswith("ep_")

    def shape(ep: PrivateEpisode) -> tuple[int, int, str]:
        return len(ep.public.ledger), len(ep.public.invoices), ep.public.manifest.period_start

    assert shape(one) != shape(two)
    assert one.public.model_dump_json() != two.public.model_dump_json()


def test_opaque_ids_not_sequential(default_episode: PrivateEpisode) -> None:
    public = default_episode.public
    n = len(public.invoices)
    in_time_order = [i.invoice_id for i in sorted(public.invoices, key=lambda i: i.issued_at)]
    assert sorted(in_time_order) == [f"INV-{k:04d}" for k in range(1, n + 1)]  # same label set
    assert in_time_order != sorted(in_time_order)  # ...but no longer in generation order
    row_ids = [r.row_id for r in public.ledger]  # ledger is time-ordered
    assert row_ids != sorted(row_ids)
    journal_first_seen = list(dict.fromkeys(r.journal_id for r in public.ledger))
    assert journal_first_seen != sorted(journal_first_seen)
    assert len(set(row_ids)) == len(row_ids)  # bijection: no label lost or duplicated


def test_all_timestamps_in_period(default_episode: PrivateEpisode) -> None:
    public = default_episode.public
    start = datetime.fromisoformat(public.manifest.period_start)
    end = datetime.fromisoformat(public.manifest.period_end)
    assert all(start <= datetime.fromisoformat(r.timestamp) < end for r in public.ledger)
    assert all(start <= datetime.fromisoformat(i.issued_at) < end for i in public.invoices)
    assert all(start <= datetime.fromisoformat(p.timestamp) < end for p in public.payments)
    assert public.ledger == tuple(sorted(public.ledger, key=lambda r: (r.timestamp, r.row_id)))


def test_double_entry_balanced_per_journal(default_episode: PrivateEpisode) -> None:
    sums: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    currencies_per_journal: dict[str, set[str]] = defaultdict(set)
    for row in default_episode.public.ledger:
        sums[(row.journal_id, row.currency)][0] += row.debit_minor
        sums[(row.journal_id, row.currency)][1] += row.credit_minor
        currencies_per_journal[row.journal_id].add(row.currency)
    assert sums and all(debit == credit for debit, credit in sums.values())
    assert all(len(c) == 1 for c in currencies_per_journal.values())  # clean journals: one currency
    assert all(r.account != "FX_GAIN_LOSS" for r in default_episode.public.ledger)


def test_fx_rates_cover_period(default_episode: PrivateEpisode) -> None:
    public = default_episode.public
    base = public.manifest.base_currency
    start = date.fromisoformat(public.manifest.period_start[:10])
    end = date.fromisoformat(public.manifest.period_end[:10])
    days = {(start + timedelta(days=k)).isoformat() for k in range((end - start).days + 1)}
    by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fx in public.fx:
        by_pair[(fx.source_currency, fx.target_currency)].add(fx.date)
        assert fx.rate > 0 and fx.rate == fx.rate.quantize(fx.rate) and fx.target_currency == base
    assert set(by_pair) == {(c, base) for c in EpisodeConfig().foreign_currencies}
    assert all(days <= dates for dates in by_pair.values())
    assert public.fx == tuple(
        sorted(public.fx, key=lambda r: (r.date, r.source_currency, r.target_currency))
    )


def test_recurring_supplier_present(default_episode: PrivateEpisode) -> None:
    public = default_episode.public
    groups: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for invoice in public.invoices:
        groups[(invoice.entity_id, invoice.total_minor, invoice.currency)].append(invoice.issued_at)
    series = [sorted(v) for v in groups.values() if len(v) >= 3]
    recurring_invoices = sum(len(s) for s in series)
    assert recurring_invoices >= EpisodeConfig().recurring_supplier_count
    for issued in series:
        gaps = [
            (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days
            for a, b in pairwise(issued)
        ]
        assert all(25 <= g <= 35 for g in gaps), gaps  # ~monthly


def test_split_payments_present(default_episode: PrivateEpisode) -> None:
    public = default_episode.public
    per_invoice = Counter(p.invoice_refs[0].invoice_id for p in public.payments)
    assert any(n >= 2 for n in per_invoice.values())
    split_id = next(i for i, n in per_invoice.items() if n >= 2)
    invoice = next(i for i in public.invoices if i.invoice_id == split_id)
    parts = [p.amount_minor for p in public.payments if p.invoice_refs[0].invoice_id == split_id]
    assert sum(parts) <= invoice.total_minor and min(parts) >= 100
    assert all(len(p.invoice_refs) == 1 for p in public.payments)  # clean: one allocation each


def test_foreign_currency_handled(default_episode: PrivateEpisode) -> None:
    public = default_episode.public
    base = public.manifest.base_currency
    foreign = [i for i in public.invoices if i.currency != base]
    assert foreign, "config has foreign currencies, so at least one foreign invoice is expected"
    index = build_fx_index(public.fx)
    invoice_by_id = {i.invoice_id: i for i in public.invoices}
    checked = 0
    for payment in public.payments:
        invoice = invoice_by_id[payment.invoice_refs[0].invoice_id]
        if invoice.currency == base:
            continue
        rows = [r for r in public.ledger if r.payment_id == payment.payment_id]
        cash = next(r for r in rows if r.account == "CASH")
        day = policy_date(public.policy, invoice.issued_at, payment.timestamp)
        rate = rate_on(index, day, payment.currency, base, public.policy)
        expected = convert_minor(payment.amount_minor, rate, public.policy)
        assert cash.currency == base and (cash.debit_minor or cash.credit_minor) == expected
        invoice_rows = [
            r for r in public.ledger if r.invoice_id == invoice.invoice_id and not r.payment_id
        ]
        assert {r.currency for r in invoice_rows} == {invoice.currency}  # AP/AR in doc currency
        checked += 1
    assert checked >= 1


def test_benford_passes_95_of_100_seeds() -> None:
    passing = 0
    for seed in range(BENFORD_SEEDS):
        public = build_clean_episode(seed=seed).public
        if check_benford(public.ledger, public.invoices) < BENFORD_CHI_SQUARED_CRITICAL:
            passing += 1
    assert passing >= 95, f"only {passing}/{BENFORD_SEEDS} seeds pass the Benford chi-squared test"


def test_generation_time_under_5_seconds() -> None:
    config = EpisodeConfig(n_ledger_rows_range=(1000, 1000))
    started = time.monotonic()
    episode = build_clean_episode(seed=3, config=config)
    elapsed = time.monotonic() - started
    assert len(episode.public.ledger) == 1000
    assert elapsed < 5.0, f"build took {elapsed:.2f}s"
