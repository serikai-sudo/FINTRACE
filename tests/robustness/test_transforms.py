"""Paired robustness tests: oracle findings and rule-based scores survive the transforms."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date, timedelta
from functools import cache

import pytest

import fintrace.environment.env as env_module
from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.evaluator.runner import load_split_config, run_baseline
from fintrace.generator.build import build_episode
from fintrace.generator.injectors.base import inconsistent_ids
from fintrace.generator.validate import check_fx_coverage, validate_clean_history
from fintrace.models import EpisodeConfig, Finding, JsonObject, PrivateEpisode
from fintrace.transforms.robustness import (
    SHIFT_DAYS_RANGE,
    TRANSFORMS,
    TransformMapping,
    apply_transforms,
    canonical_findings,
    oracle_findings,
    remap_ids,
    shift_time,
    shuffle_records,
    translate_findings,
)

DEV_SEEDS = (10000, 10001, 10002, 10003, 10004)
F1_TOLERANCE = 0.1
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
PUBLIC_FORBIDDEN = ("truth", "provenance", "expected_amount", "issue_id", "acceptable_evidence")


def _transform_seed(seed: int) -> int:
    return seed * 2 + 1


@cache
def _config() -> EpisodeConfig:
    return load_split_config("dev")


@cache
def _episode(seed: int) -> PrivateEpisode:
    return build_episode(seed, _config())


def _env_for(private: PrivateEpisode, monkeypatch: pytest.MonkeyPatch) -> FintraceEnv:
    """An environment loaded with a prebuilt episode (the env itself only builds from a seed)."""
    monkeypatch.setattr(env_module, "build_episode", lambda _seed, _config=None: private)
    return FintraceEnv()


def _rule_based(private: PrivateEpisode, monkeypatch: pytest.MonkeyPatch) -> tuple[float, str]:
    """(F1, canonical findings) of the rule-based auditor on ``private``."""
    env = _env_for(private, monkeypatch)
    trace: list[JsonObject] = []
    run_baseline(RuleBasedReconciliation(), env, 0, None, trace=trace)
    submitted = trace[-1]["args"]["findings"]
    assert isinstance(submitted, list)
    findings = [Finding.model_validate(f) for f in submitted]
    tp, fp, fn = match(findings, private.truth.issues, public_ids=public_ids_of(private.public))
    return compute_metrics(tp, fp, fn)[2], canonical_findings(submitted)


def _public_ids(private: PrivateEpisode) -> set[str]:
    public = private.public
    ids: set[str] = set()
    for row in public.ledger:
        ids.update(x for x in (row.row_id, row.entity_id, row.invoice_id, row.payment_id) if x)
        ids.add(row.journal_id)
    for invoice in public.invoices:
        ids.update((invoice.doc_id, invoice.invoice_id, invoice.entity_id))
    for payment in public.payments:
        ids.update((payment.doc_id, payment.payment_id))
        ids.update(ref.invoice_id for ref in payment.invoice_refs)
    return ids


def _oracle_equivalent(
    original: PrivateEpisode, transformed: PrivateEpisode, mapping: TransformMapping
) -> None:
    ported = translate_findings(oracle_findings(original.truth), mapping)
    found = oracle_findings(transformed.truth)
    assert canonical_findings(ported) == canonical_findings(found)
    assert len(transformed.truth.issues) == len(original.truth.issues)
    # not a tautology: every id the transformed truth names exists in the transformed public books
    public_ids = _public_ids(transformed)
    for finding in found:
        assert set(finding["affected_ids"]) <= public_ids  # type: ignore[arg-type]
        assert set(finding["evidence_ids"]) <= public_ids  # type: ignore[arg-type]


def _valid(private: PrivateEpisode) -> None:
    validate_clean_history(private, ignore_ids=inconsistent_ids(private))


def _list_all(env: FintraceEnv, table: str) -> tuple[list[JsonObject], int]:
    records: list[JsonObject] = []
    cursor: str | None = None
    total = -1
    while True:
        args: JsonObject = {"table": table, "cursor": cursor, "limit": 50}
        observation, *_ = env.step({"tool": "list_records", "args": args})
        assert observation.ok and observation.result is not None
        records.extend(observation.result["records"])  # type: ignore[arg-type]
        total = int(observation.result["records_total"])  # type: ignore[arg-type]
        cursor = observation.result["next_cursor"]  # type: ignore[assignment]
        if cursor is None:
            return records, total


# ---------------------------------------------------------------- remap_ids


def test_remap_ids_oracle_findings_equivalent() -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        remapped, mapping = remap_ids(original, _transform_seed(seed))
        _oracle_equivalent(original, remapped, mapping)
        assert remapped.public.manifest.episode_id != original.public.manifest.episode_id
        assert remapped.truth.episode_id == remapped.public.manifest.episode_id


def test_remap_ids_rule_based_f1_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        remapped, _ = remap_ids(original, _transform_seed(seed))
        f1_original, _ = _rule_based(original, monkeypatch)
        f1_remapped, _ = _rule_based(remapped, monkeypatch)
        assert abs(f1_original - f1_remapped) < F1_TOLERANCE, (seed, f1_original, f1_remapped)
        assert f1_remapped > 0.5


def test_remap_ids_double_entry_balanced() -> None:
    """The remapped books pass every clean-history check with the translated relaxations."""
    for seed in DEV_SEEDS:
        remapped, mapping = remap_ids(_episode(seed), _transform_seed(seed))
        _valid(remapped)
        original = _episode(seed)
        relaxed = frozenset(mapping.translate_ids(inconsistent_ids(original)))
        assert inconsistent_ids(remapped) == relaxed
        assert remapped.public.manifest.n_ledger_rows == original.public.manifest.n_ledger_rows


def test_remap_ids_byte_stable() -> None:
    original = _episode(10000)
    first, mapping_a = remap_ids(original, 42)
    second, mapping_b = remap_ids(original, 42)
    assert mapping_a == mapping_b and first.model_dump_json() == second.model_dump_json()
    other, mapping_c = remap_ids(original, 43)
    assert mapping_c.id_map != mapping_a.id_map
    assert other.model_dump_json() != first.model_dump_json()
    old_ids, new_ids = set(mapping_a.id_map), set(mapping_a.id_map.values())
    assert len(mapping_a.id_map) >= 100 and len(new_ids) == len(mapping_a.id_map)
    assert not old_ids & new_ids  # every id is fresh
    for old, new in mapping_a.id_map.items():
        family, digits = old.split("-", 1)
        assert new.startswith(f"{family}-") and len(new.split("-", 1)[1]) == len(digits)
    # no stale id survives anywhere: public books, truth, provenance (old and new ids are disjoint)
    dumps = (first.public, first.truth, first.provenance)
    for text in (model.model_dump_json() for model in dumps):
        assert not any(f'"{old}"' in text for old in old_ids)
    public_ids = _public_ids(first)
    for entry in first.provenance.injections:
        assert entry.source_id is None or entry.source_id in public_ids
        assert set(entry.created_ids) <= public_ids


# ---------------------------------------------------------------- shuffle_records


def test_shuffle_records_env_list_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _episode(10000)
    shuffled, _ = shuffle_records(original, _transform_seed(10000))
    assert shuffled.public.ledger != original.public.ledger  # storage order really changed
    row_ids = sorted(r.row_id for r in original.public.ledger)
    assert sorted(r.row_id for r in shuffled.public.ledger) == row_ids
    pages: list[tuple[list[JsonObject], int]] = []
    for private in (original, shuffled):
        env = _env_for(private, monkeypatch)
        env.reset(seed=0)
        pages.append([_list_all(env, table) for table in ("ledger", "invoices", "payments", "fx")])
    assert json.dumps(pages[0], sort_keys=True) == json.dumps(pages[1], sort_keys=True)


def test_shuffle_records_rule_based_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    for seed in DEV_SEEDS[:3]:
        original = _episode(seed)
        shuffled, _ = shuffle_records(original, _transform_seed(seed))
        f1_original, findings_original = _rule_based(original, monkeypatch)
        f1_shuffled, findings_shuffled = _rule_based(shuffled, monkeypatch)
        assert f1_original == f1_shuffled and findings_original == findings_shuffled


def test_shuffle_records_truth_unchanged() -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        shuffled, mapping = shuffle_records(original, _transform_seed(seed))
        assert mapping.id_map == {} and mapping.time_shift_seconds == 0
        stripped = [i.model_dump() for i in shuffled.truth.issues]
        assert stripped == [i.model_dump() for i in original.truth.issues]
        assert shuffled.provenance == original.provenance
        _valid(shuffled)


# ---------------------------------------------------------------- shift_time


def test_shift_time_within_period_rules() -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        shifted, mapping = shift_time(original, _transform_seed(seed))
        days = mapping.time_shift_seconds / 86_400
        assert days == int(days) and SHIFT_DAYS_RANGE[0] <= days <= SHIFT_DAYS_RANGE[1]
        _valid(shifted)
        start, end = shifted.public.manifest.period_start, shifted.public.manifest.period_end
        before = original.public.manifest
        old_start, old_end = before.period_start, before.period_end
        rows = zip(original.public.ledger, shifted.public.ledger, strict=True)
        for row_before, row_after in rows:
            assert row_after.row_id == row_before.row_id
            inside_before = old_start <= row_before.timestamp < old_end
            assert inside_before == (start <= row_after.timestamp < end)
        invoices = zip(original.public.invoices, shifted.public.invoices, strict=True)
        for inv_before, inv_after in invoices:
            inside_before = old_start <= inv_before.issued_at < old_end
            assert inside_before == (start <= inv_after.issued_at < end)


def test_shift_time_oracle_findings_equivalent() -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        shifted, mapping = shift_time(original, _transform_seed(seed))
        _oracle_equivalent(original, shifted, mapping)
        kinds = [i.type for i in shifted.truth.issues]
        assert kinds.count("period_misclassification") == 2  # cut-off errors survive the shift


def test_shift_time_rule_based_f1_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        shifted, _ = shift_time(original, _transform_seed(seed))
        f1_original, _ = _rule_based(original, monkeypatch)
        f1_shifted, _ = _rule_based(shifted, monkeypatch)
        assert abs(f1_original - f1_shifted) < F1_TOLERANCE, (seed, f1_original, f1_shifted)


def test_shift_time_fx_coverage_preserved() -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        shifted, mapping = shift_time(original, _transform_seed(seed))
        public = shifted.public
        check_fx_coverage(public.fx, public.manifest, public.invoices, public.payments)
        days = mapping.time_shift_seconds // 86_400
        # provenance dates follow the calendar too (evaluator-side consistency)
        before_dates = sorted(_date_strings(original.provenance.model_dump()))
        after_dates = sorted(_date_strings(shifted.provenance.model_dump()))
        assert after_dates == sorted(_plus_days(d, days) for d in before_dates) and before_dates
        expected = {
            (f.source_currency, f.target_currency, _plus_days(f.date, days))
            for f in original.public.fx
        }
        assert {(f.source_currency, f.target_currency, f.date) for f in public.fx} == expected
        assert len(public.fx) == len(original.public.fx)


def _plus_days(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def _date_strings(value: object) -> list[str]:
    """Every plain ``YYYY-MM-DD`` string in a JSON tree (timestamps are covered via their date)."""
    if isinstance(value, str):
        return [value[:10]] if DATE_RE.match(value[:10]) else []
    if isinstance(value, dict):
        return [d for item in value.values() for d in _date_strings(item)]
    if isinstance(value, list | tuple):
        return [d for item in value for d in _date_strings(item)]
    return []


# ---------------------------------------------------------------- all transforms


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_all_transforms_preserve_issue_count_and_public_surface(name: str) -> None:
    for seed in DEV_SEEDS:
        original = _episode(seed)
        transformed, _ = TRANSFORMS[name](original, _transform_seed(seed))
        assert len(transformed.truth.issues) == len(original.truth.issues) == 8
        manifest, before = transformed.public.manifest, original.public.manifest
        assert manifest.n_ledger_rows == before.n_ledger_rows
        assert manifest.n_documents == before.n_documents
        assert len(transformed.public.ledger) == manifest.n_ledger_rows
        lowered = transformed.public.model_dump_json().lower()
        assert not any(token in lowered for token in PUBLIC_FORBIDDEN)


def test_combined_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    for seed in DEV_SEEDS[:3]:
        original = _episode(seed)
        combined, mapping = apply_transforms(
            original, _transform_seed(seed), ("remap_ids", "shuffle_records", "shift_time")
        )
        _oracle_equivalent(original, combined, mapping)
        _valid(combined)
        assert mapping.time_shift_seconds % 86_400 == 0 and mapping.time_shift_seconds > 0
        assert len(mapping.id_map) >= 100
        f1_original, _ = _rule_based(original, monkeypatch)
        f1_combined, _ = _rule_based(combined, monkeypatch)
        assert abs(f1_original - f1_combined) < F1_TOLERANCE, (seed, f1_original, f1_combined)


def test_transforms_deterministic() -> None:
    original = _episode(10001)
    transform: Callable[[PrivateEpisode, int], tuple[PrivateEpisode, TransformMapping]]
    for name, transform in sorted(TRANSFORMS.items()):
        first, mapping_a = transform(original, 7)
        second, mapping_b = transform(original, 7)
        assert first.model_dump_json() == second.model_dump_json(), name
        assert mapping_a == mapping_b, name
        assert first.public.manifest.episode_id == second.public.manifest.episode_id
        assert first.public.manifest.episode_id != original.public.manifest.episode_id
