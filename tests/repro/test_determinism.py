"""Reproducibility: the same seed and config give the same bytes, from the generator to the
trajectory file (CLAUDE.md §6.5)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import load_split_config, run_baseline
from fintrace.generator.build import build_clean_episode
from fintrace.logging.trajectory import TrajectoryLogger
from fintrace.models import EpisodeConfig


def test_deterministic_regeneration() -> None:
    first = build_clean_episode(seed=42, config=EpisodeConfig()).model_dump_json()
    second = build_clean_episode(seed=42).model_dump_json()
    assert first == second == build_clean_episode(seed=42).model_dump_json()
    other_config = build_clean_episode(seed=42, config=EpisodeConfig(period_days=91))
    assert other_config.model_dump_json() != first  # config is part of the determinism key


def _record_baseline(path: Path, seed: int, config: EpisodeConfig) -> None:
    logger = TrajectoryLogger(path)
    run_baseline(RuleBasedReconciliation(), FintraceEnv(), seed, config, logger=logger)
    assert not logger.is_open


def test_logger_byte_stable_rerun(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _record_baseline(first, 10001, load_split_config("dev"))
    _record_baseline(second, 10001, load_split_config("dev"))
    assert first.read_bytes() == second.read_bytes()
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_sha == hashlib.sha256(second.read_bytes()).hexdigest()
