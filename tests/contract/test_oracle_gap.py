"""Oracle-gap diagnostic (D3): matrix structure, reference row, differential signal, attacker
floor, determinism and byte stability; the full script against its committed artefacts (slow)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

import fintrace.environment.env as env_module
from fintrace.agents.attacks import ALL_ATTACKERS
from fintrace.constants import ISSUE_TYPES
from fintrace.diagnostics import oracle_gap
from fintrace.diagnostics.oracle_gap import (
    ATTACKER_FLOOR_GAP,
    BASELINE_ORDER,
    CLEAN_TYPE,
    DIAGNOSTIC_SEED_BASE,
    DIAGNOSTIC_TYPES,
    attacker_floor,
    build_diagnostic_set,
    check_results,
    compute_oracle_gap,
    differential_spread,
    policy_factories,
)
from fintrace.exporters.episode_files import canonical_json
from fintrace.generator.build import build_episode
from fintrace.models import JsonObject

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
N_PER_TYPE = 3
POLICY_ORDER = [*BASELINE_ORDER, *(cls.name for cls in ALL_ATTACKERS)]

_STORE: dict[tuple[int, str], object] = {}


def _cached_build(seed: int, config: object = None) -> object:
    """Build each (seed, config) once per session: the columns differ by config, not by seed."""
    key = (seed, "" if config is None else config.model_dump_json())  # type: ignore[attr-defined]
    if key not in _STORE:
        _STORE[key] = build_episode(seed, config)  # type: ignore[arg-type]
    return _STORE[key]


@pytest.fixture(autouse=True)
def _cached_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module, "build_episode", _cached_build)
    monkeypatch.setattr(oracle_gap, "build_episode", _cached_build)


@cache
def _full() -> JsonObject:
    """The committed configuration: 24 policies x 5 types x 3 paired episodes."""
    episodes, skipped = build_diagnostic_set(N_PER_TYPE)
    return compute_oracle_gap(episodes, policy_factories(), skipped=skipped)


@cache
def _small(run: int) -> JsonObject:
    """4 baselines x 5 types x 1 episode; ``run`` only forces an independent computation."""
    del run
    episodes, skipped = build_diagnostic_set(1)
    return compute_oracle_gap(episodes, policy_factories(BASELINE_ORDER), skipped=skipped)


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_oracle_gap.py"
    spec = importlib.util.spec_from_file_location("run_oracle_gap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_gap_structure() -> None:
    results = _full()
    assert results["policies"] == POLICY_ORDER and len(POLICY_ORDER) == 24
    assert results["types"] == list(DIAGNOSTIC_TYPES) == [CLEAN_TYPE, *ISSUE_TYPES]
    assert results["n_per_type"] == N_PER_TYPE and results["skipped"] == []
    assert results["seeds"] == [DIAGNOSTIC_SEED_BASE + i for i in range(N_PER_TYPE)]
    for policy in POLICY_ORDER:
        for error_type in DIAGNOSTIC_TYPES:
            assert isinstance(results["mean_reward"][policy][error_type], float)
            assert isinstance(results["gap"][policy][error_type], float)
            assert len(results["rewards"][policy][error_type]) == N_PER_TYPE
            assert all(isinstance(s, int) for s in results["steps"][policy][error_type])
    ids = [e["episode_id"] for t in DIAGNOSTIC_TYPES for e in results["episodes"][t]]
    assert len(ids) == len(set(ids)) == 5 * N_PER_TYPE
    for error_type in DIAGNOSTIC_TYPES:
        expected = 0 if error_type == CLEAN_TYPE else 1
        assert [e["n_issues"] for e in results["episodes"][error_type]] == [expected] * N_PER_TYPE
    # paired design: the five columns of one seed are the same base books plus one injection
    episodes, _ = build_diagnostic_set(N_PER_TYPE)
    for seed in results["seeds"]:
        column = [item for item in episodes if item.seed == seed]
        assert [item.error_type for item in column] == list(DIAGNOSTIC_TYPES)
        assert len({item.episode.provenance.seed_hash for item in column}) == 1
        clean = next(item for item in column if item.error_type == CLEAN_TYPE)
        assert all(len(item.episode.public.ledger) >= len(clean.episode.public.ledger)
                   for item in column)  # fmt: skip


def test_oracle_gap_zero_for_oracle() -> None:
    results = _full()
    assert all(results["gap"]["oracle"][t] == 0.0 for t in DIAGNOSTIC_TYPES)
    assert check_results(results) == []


def test_rule_based_differential() -> None:
    """The auditing baseline is further from the oracle on some types than on others."""
    gaps = {t: _full()["gap"]["rule_based"][t] for t in ISSUE_TYPES}
    assert len(set(gaps.values())) > 1, gaps
    assert differential_spread(_full(), "rule_based") > 0.1, gaps
    hardest = max(gaps, key=gaps.__getitem__)
    easiest = min(gaps, key=gaps.__getitem__)
    assert hardest != easiest and gaps[hardest] > 2 * gaps[easiest], gaps
    assert all(0.0 < gap < 1.0 for gap in gaps.values()), gaps  # it audits: never at the floor


def test_attackers_uniform_floor() -> None:
    results = _full()
    attackers = [name for name in results["policies"] if name not in BASELINE_ORDER]
    assert len(attackers) == 20
    for name in attackers:
        for error_type in ISSUE_TYPES:
            assert results["gap"][name][error_type] >= ATTACKER_FLOOR_GAP, (name, error_type)
            assert results["mean_reward"][name][error_type] <= 0.0, (name, error_type)
        assert results["gap"][name][CLEAN_TYPE] >= 0.0, name  # silence on clean books is legal
    floor = attacker_floor(results)
    assert floor["n_attackers"] == 20
    assert ATTACKER_FLOOR_GAP <= floor["min_gap"] <= floor["max_gap"] <= 2.0


def test_clean_type_oracle_near_one() -> None:
    results = _full()
    assert results["mean_reward"]["oracle"][CLEAN_TYPE] > 0.9
    assert results["rewards"]["oracle"][CLEAN_TYPE] == [1.0] * N_PER_TYPE  # silence, 0 steps
    assert all(results["mean_reward"]["oracle"][t] > 0.9 for t in DIAGNOSTIC_TYPES)


def test_gap_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _small(1), _small(2)
    assert first is not second and first == second
    assert first["policies"] == list(BASELINE_ORDER) and first["n_per_type"] == 1
    assert first["gap"]["oracle"] == dict.fromkeys(DIAGNOSTIC_TYPES, 0.0)
    # the same matrix with the real generator: every run_baseline rebuilds the prebuilt episode
    # from (seed, config), so this also proves the rebuild equals the primed episode
    monkeypatch.undo()
    assert env_module.build_episode is build_episode and oracle_gap.build_episode is build_episode
    episodes, skipped = build_diagnostic_set(1)
    rebuilt = compute_oracle_gap(episodes, policy_factories(BASELINE_ORDER), skipped=skipped)
    assert rebuilt == first


def test_oracle_gap_byte_stable_json() -> None:
    first = canonical_json(_small(1)) + "\n"
    second = canonical_json(_small(2)) + "\n"
    assert first == second
    digests = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in (first, second)}
    assert len(digests) == 1
    assert "elapsed" not in first and "elapsed" not in canonical_json(_full())


@pytest.mark.slow
def test_run_oracle_gap_full(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    assert script.main(["--policies", "rule_based", "--docs-dir", str(tmp_path)]) == 2
    assert script.main(["--policies", "oracle,nope", "--docs-dir", str(tmp_path)]) == 2
    assert script.main(["--docs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "rule_based gaps:" in out and "attacker floor" in out
    results = json.loads((tmp_path / "oracle_gap_results.json").read_text(encoding="utf-8"))
    assert results["policies"] == POLICY_ORDER and results["n_per_type"] == N_PER_TYPE
    assert check_results(results) == []
    report = (tmp_path / "oracle_gap_report.md").read_text(encoding="utf-8")
    assert report.startswith("# FINTRACE Oracle Gap Diagnostic")
    for heading in ("## Methodology", "## Key insight", "## Reproducibility"):
        assert heading in report
    assert "| Policy | clean | duplicate_invoice |" in report
    assert report.count("| oracle |") == 2  # the reward table and the gap table
    png = (tmp_path / "figures" / "oracle_gap_matrix.png").read_bytes()
    assert png[:8] == PNG_MAGIC and len(png) > 10_000
    assert struct.unpack(">II", png[16:24]) == (1400, 1000)
    # the committed artefacts are exactly what this run produced (json + report byte for byte);
    # the committed png is checked structurally (magic + dimensions): matplotlib PNG bytes differ
    # across freetype / font-hinting stacks, so its size is not comparable between machines
    for name in ("oracle_gap_results.json", "oracle_gap_report.md"):
        committed = DOCS / name
        assert committed.is_file(), f"run `make oracle-gap` and commit docs/{name}"
        assert committed.read_bytes() == (tmp_path / name).read_bytes(), f"{name} drifted"
    committed_png = DOCS / "figures" / "oracle_gap_matrix.png"
    assert committed_png.is_file(), "run `make oracle-gap` and commit docs/figures/*.png"
    data = committed_png.read_bytes()
    assert data[:8] == PNG_MAGIC and len(data) > 10_000
    assert struct.unpack(">II", data[16:24]) == (1400, 1000)
