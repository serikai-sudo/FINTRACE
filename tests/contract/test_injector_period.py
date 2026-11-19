"""period_misclassification injector: timezone-aware boundary crossing, both variants."""

from __future__ import annotations

import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.injectors import period
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
    Invoice,
    Issue,
    LedgerRow,
    PolicyPeriod,
    PrivateEpisode,
    ProvenanceEntry,
)

Injected = tuple[PrivateEpisode, Issue, ProvenanceEntry]
VARIANTS = period.VARIANTS
CHI_SQUARED_1DF_P01 = 6.63
HOUR = timedelta(hours=1)


def _config(count: int = 1) -> EpisodeConfig:
    params = InjectorParams(count=count)
    return EpisodeConfig(errors=ErrorInjectionConfig(period_misclassification=params))


@pytest.fixture(scope="module")
def clean() -> PrivateEpisode:
    return build_clean_episode(42)


def _force(clean: PrivateEpisode, variant: str, start: int = 0) -> Injected:
    for rng_seed in range(start, start + 30):
        try:
            return period._inject_variant(clean, random.Random(rng_seed), variant)  # type: ignore[arg-type]
        except InjectorError:
            continue
    raise AssertionError(f"{variant} rejected 30 rng seeds in a row")


def _invoice(episode: PrivateEpisode, invoice_id: str | None) -> Invoice:
    return next(i for i in episode.public.invoices if i.invoice_id == invoice_id)


def _journal(episode: PrivateEpisode, issue: Issue, invoice_id: str | None) -> list[LedgerRow]:
    journal_id = next(x for x in issue.affected_ids if x != invoice_id)
    return [r for r in episode.public.ledger if r.journal_id == journal_id]


def _bounds(episode: PrivateEpisode) -> tuple[datetime, datetime]:
    manifest = episode.public.manifest
    start, end = manifest.period_start, manifest.period_end
    return datetime.fromisoformat(start), datetime.fromisoformat(end)


def _inside(iso: str, episode: PrivateEpisode) -> bool:
    start, end = _bounds(episode)
    return start <= datetime.fromisoformat(iso) < end


@pytest.mark.parametrize("variant", VARIANTS)
def test_period_adds_one_issue(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    assert injected.truth.issues == ()  # inject() returns; build.py appends
    appended = append_truth_and_provenance(injected, issue, entry)
    assert len(appended.truth.issues) == len(clean.truth.issues) + 1
    assert issue.type == "period_misclassification" == entry.op
    assert entry.modified["variant"] == variant == entry.original["variant"]
    assert entry.created_ids == () and entry.source_id in issue.affected_ids


@pytest.mark.parametrize("variant", VARIANTS)
def test_period_currency_is_invoice_currency(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    invoice = _invoice(injected, entry.source_id)
    assert issue.currency == invoice.currency
    # a foreign-currency source keeps its own currency (not the base), both variants
    foreign = next(
        (
            iss
            for seed in range(12)
            for ep in (build_episode(seed, _config(1)),)
            for iss in ep.truth.issues
            if iss.currency != ep.public.manifest.base_currency
        ),
        None,
    )
    assert foreign is not None and foreign.currency in {"EUR", "GBP"}


@pytest.mark.parametrize("variant", VARIANTS)
def test_period_affected_ids_invoice_and_row(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    assert len(issue.affected_ids) == 2 and issue.affected_ids == tuple(sorted(issue.affected_ids))
    rows = _journal(injected, issue, entry.source_id)
    assert len(rows) == 2 and {r.invoice_id for r in rows} == {entry.source_id}
    row_ids = tuple(sorted(r.row_id for r in rows))
    assert row_ids in issue.acceptable_evidence_sets
    assert entry.modified["relaxed_ids"] == list(row_ids)
    invoice = _invoice(injected, entry.source_id)
    assert (invoice.doc_id, *row_ids) in issue.acceptable_evidence_sets


@pytest.mark.parametrize("variant", VARIANTS)
def test_period_expected_equals_invoice_total(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    invoice = _invoice(injected, entry.source_id)
    assert issue.expected_amount_minor == invoice.total_minor
    assert invoice.total_minor == _invoice(clean, entry.source_id).total_minor  # amount untouched


@pytest.mark.parametrize("variant", VARIANTS)
def test_independent_check_passes_on_injected(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    period.independent_check(injected.public, issue)  # no raise
    with pytest.raises(ConsistencyError, match="same period"):
        period.independent_check(clean.public, issue)  # clean books: document and posting agree
    off_by_one = issue.model_copy(update={"expected_amount_minor": issue.expected_amount_minor + 1})
    with pytest.raises(ConsistencyError, match="period_check"):
        period.independent_check(injected.public, off_by_one)
    posted_at = injected.public.policy.period.model_copy(update={"assignment_field": "posted_at"})
    policy = injected.public.policy.model_copy(update={"period": posted_at})
    with pytest.raises(ConsistencyError, match="same period"):  # posting classifies itself
        period.independent_check(injected.public.model_copy(update={"policy": policy}), issue)
    relaxed = frozenset(str(x) for x in entry.modified["relaxed_ids"])  # type: ignore[union-attr]
    validate_clean_history(injected.public, ignore_ids=relaxed)
    with pytest.raises(ConsistencyError):
        validate_clean_history(injected.public)


@pytest.mark.parametrize("variant", VARIANTS)
def test_shift_within_bounds_1_to_6_hours(clean: PrivateEpisode, variant: str) -> None:
    moved_key = "issued_at" if variant == "ledger_row_shifted_in" else "journal_timestamp"
    for start in (0, 10, 20):
        _, _, entry = _force(clean, variant, start)
        boundary = datetime.fromisoformat(str(entry.modified["boundary"]))
        moved = datetime.fromisoformat(str(entry.modified[moved_key]))
        distance = abs(moved - boundary)
        assert HOUR <= distance <= 6 * HOUR
        assert distance.total_seconds() == entry.modified["seconds_past_boundary"]
        shift = int(str(entry.modified["shift_hours_utc"]))
        original = datetime.fromisoformat(str(entry.original["issued_at"]))
        assert shift != 0 and shift * 3600 == int((moved - original).total_seconds())  # whole hours
        assert entry.modified["shift_seconds_utc"] == shift * 3600


@pytest.mark.parametrize("variant", VARIANTS)
def test_timestamp_still_valid_iso8601(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    invoice = _invoice(injected, entry.source_id)
    rows = _journal(injected, issue, entry.source_id)
    for iso in (invoice.issued_at, invoice.due_at, *(r.timestamp for r in rows)):
        parsed = datetime.fromisoformat(iso)
        assert parsed.tzinfo is not None and iso.endswith("+05:00")  # policy offset preserved
    original = _invoice(clean, entry.source_id)
    assert invoice.issued_at[14:19] == original.issued_at[14:19]  # mm:ss kept (document)
    assert rows[0].timestamp[14:19] == original.issued_at[14:19]  # mm:ss kept (posting)
    due_gap = datetime.fromisoformat(invoice.due_at) - datetime.fromisoformat(invoice.issued_at)
    original_due = datetime.fromisoformat(original.due_at)
    assert due_gap == original_due - datetime.fromisoformat(original.issued_at)


@pytest.mark.parametrize("variant", VARIANTS)
def test_boundary_crossing_in_policy_tz(clean: PrivateEpisode, variant: str) -> None:
    injected, issue, entry = _force(clean, variant)
    invoice = _invoice(injected, entry.source_id)
    rows = _journal(injected, issue, entry.source_id)
    assert _inside(invoice.issued_at, injected) != _inside(rows[0].timestamp, injected)
    if variant == "ledger_row_shifted_in":
        assert not _inside(invoice.issued_at, injected) and _inside(rows[0].timestamp, injected)
    else:
        assert _inside(invoice.issued_at, injected) and not _inside(rows[0].timestamp, injected)
    # the same verdict when every instant is expressed in UTC: the trap is the local boundary
    start, end = _bounds(injected)
    for iso in (invoice.issued_at, rows[0].timestamp):
        expected = _inside(iso, injected)
        utc = datetime.fromisoformat(iso).astimezone(UTC)
        assert (start.astimezone(UTC) <= utc < end.astimezone(UTC)) == expected
        local_day = datetime.fromisoformat(iso).date()  # the local calendar day decides, not UTC
        assert (start.date() <= local_day < end.date()) == expected


@pytest.mark.parametrize("variant", VARIANTS)
def test_double_entry_still_balanced(clean: PrivateEpisode, variant: str) -> None:
    injected, _, _ = _force(clean, variant)
    check_balanced(injected.public.ledger)  # no raise, both rows share the new timestamp
    ledger = injected.public.ledger
    assert ledger == tuple(sorted(ledger, key=lambda r: (r.timestamp, r.row_id)))
    invoices = injected.public.invoices
    assert invoices == tuple(sorted(invoices, key=lambda i: (i.issued_at, i.invoice_id)))

    def amounts(rows: tuple[LedgerRow, ...]) -> list[tuple[str, int, int, str]]:
        return sorted((r.row_id, r.debit_minor, r.credit_minor, r.account) for r in rows)

    assert amounts(clean.public.ledger) == amounts(ledger)  # only timestamps moved
    assert injected.public.manifest == clean.public.manifest
    assert injected.public.payments == clean.public.payments


@pytest.mark.slow
def test_200_seeds_build_and_validate() -> None:
    variants: Counter[str] = Counter()
    infeasible: list[int] = []
    for seed in range(200):
        try:
            episode = build_episode(seed, _config(1))
        except InjectorError:
            infeasible.append(seed)
            continue
        assert len(episode.truth.issues) == 1
        validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
        period.independent_check(episode.public, episode.truth.issues[0])
        assert episode.truth.episode_id == episode.public.manifest.episode_id
        variants[str(episode.provenance.injections[0].modified["variant"])] += 1
    assert set(variants) == set(VARIANTS)
    # two tiny episodes (17 and 21 invoices) have no invoice within a week of a boundary: the
    # same seeds, the same clean InjectorError every time
    assert infeasible == INFEASIBLE_SEEDS


INFEASIBLE_SEEDS = [11, 58]


def _built(seeds: range) -> list[PrivateEpisode]:
    episodes: list[PrivateEpisode] = []
    for seed in seeds:
        if seed in INFEASIBLE_SEEDS:
            continue
        episodes.append(build_episode(seed, _config(1)))
    return episodes


def test_variant_distribution_uniform() -> None:
    episodes = _built(range(50))
    variants: Counter[str] = Counter(
        str(episode.provenance.injections[0].modified["variant"]) for episode in episodes
    )
    assert set(variants) == set(VARIANTS) and sum(variants.values()) == 49
    expected = len(episodes) / 2
    chi_squared = sum((n - expected) ** 2 / expected for n in variants.values())
    assert chi_squared < CHI_SQUARED_1DF_P01, dict(variants)


def test_timezone_helpers_offsets_utc_and_dst() -> None:
    """Boundary flags, offset/UTC/IANA zones, and DST-safe landing bounds in UTC terms."""
    five = period._tzinfo("+05:00")
    assert period._tzinfo("+05:30").utcoffset(None) == timedelta(hours=5, minutes=30)
    assert period._tzinfo("UTC") is UTC
    new_york = period._tzinfo("America/New_York")
    assert isinstance(new_york, ZoneInfo)
    rules = PolicyPeriod(
        start_inclusive=True, end_exclusive=True, timezone="+05:00", assignment_field="issued_at"
    )
    start, end = datetime(2025, 1, 1, tzinfo=five), datetime(2025, 4, 1, tzinfo=five)
    assert period._inside("2025-01-01T00:00:00+05:00", start, end, five, rules)
    assert period._inside("2024-12-31T19:00:00+00:00", start, end, five, rules)  # same instant
    assert not period._inside("2024-12-31T18:59:59+00:00", start, end, five, rules)
    assert not period._inside("2025-04-01T00:00:00+05:00", start, end, five, rules)
    open_start = rules.model_copy(update={"start_inclusive": False})
    assert not period._inside("2025-01-01T00:00:00+05:00", start, end, five, open_start)
    closed_end = rules.model_copy(update={"end_exclusive": False})
    assert period._inside("2025-04-01T00:00:00+05:00", start, end, five, closed_end)
    # DST transitions in New York: spring forward 2026-03-08, fall back 2026-11-01; the landing
    # stays 1..6 real hours past the boundary on either side
    for boundary in (datetime(2026, 3, 8, tzinfo=new_york), datetime(2026, 11, 1, tzinfo=new_york)):
        issued = boundary - timedelta(hours=13, minutes=42, seconds=17)
        for rng_seed in range(12):
            after = period._landing(issued, boundary, 1, random.Random(rng_seed))
            before = period._landing(issued, boundary, -1, random.Random(rng_seed))
            assert HOUR <= after - boundary.astimezone(UTC) <= 6 * HOUR
            assert HOUR <= boundary.astimezone(UTC) - before <= 6 * HOUR
            assert after.tzinfo is UTC and before.tzinfo is UTC


def test_relaxed_journal_keeps_amount_checks(clean: PrivateEpisode) -> None:
    """ignore_ids relaxes the posting date only: a wrong amount on a relaxed journal still fails."""
    injected, issue, entry = _force(clean, "ledger_row_shifted_out")
    relaxed = frozenset(str(x) for x in entry.modified["relaxed_ids"])  # type: ignore[union-attr]
    rows = _journal(injected, issue, entry.source_id)
    bumped = [
        r.model_copy(update={"debit_minor": r.debit_minor + 1}) if r.debit_minor else
        r.model_copy(update={"credit_minor": r.credit_minor + 1})
        for r in rows
    ]  # fmt: skip
    by_id = {r.row_id: r for r in bumped}
    ledger = tuple(by_id.get(r.row_id, r) for r in injected.public.ledger)
    broken = injected.public.model_copy(update={"ledger": ledger})
    with pytest.raises(ConsistencyError, match="ledger_matches_documents"):
        validate_clean_history(broken, ignore_ids=relaxed)
    # and a second injection never reuses the first one's journal or invoice
    episode = build_episode(42, _config(2))
    first, second = episode.truth.issues
    assert not set(first.affected_ids) & set(second.affected_ids)
    relaxed_sets = [set(e.modified["relaxed_ids"]) for e in episode.provenance.injections]  # type: ignore[arg-type]
    assert not relaxed_sets[0] & relaxed_sets[1]
    for iss in episode.truth.issues:
        period.independent_check(episode.public, iss)


def test_no_day_level_shift_from_boundary() -> None:
    """The moved timestamp always lands within 6 h of the boundary and keeps the local time-of-day
    minutes and seconds; nothing in the public records jumps by whole days away from the cut-off."""
    for episode in _built(range(30)):
        entry = episode.provenance.injections[0]
        past = int(str(entry.modified["seconds_past_boundary"]))
        assert 3600 <= past <= 6 * 3600
        boundary = datetime.fromisoformat(str(entry.modified["boundary"]))
        assert boundary.utcoffset() == timedelta(hours=5)  # evaluated in the policy timezone
        assert (boundary.hour, boundary.minute, boundary.second) == (0, 0, 0)  # local midnight
