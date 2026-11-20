"""Slow end-to-end: all four baselines over the dev split and the committed results document."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.evaluator.runner import BASELINE_FACTORIES, run_split, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DEV = REPO_ROOT / "docs" / "results_dev.md"


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_all_four_baselines_run_on_dev_split() -> None:
    stats = {
        name: summarize(run_split(factory, "dev")) for name, factory in BASELINE_FACTORIES.items()
    }
    assert all(s["n"] == 10 for s in stats.values())
    assert stats["oracle"]["min_reward"] > 0.9
    assert stats["oracle"]["mean_reward"] > stats["rule_based"]["mean_reward"] > 0.0
    assert stats["rule_based"]["mean_reward"] > stats["random"]["mean_reward"]
    assert stats["random"]["max_reward"] <= 0.0
    assert stats["mark_everything"]["max_reward"] < -0.9
    assert all(s["mean_steps"] <= 100 for s in stats.values())


@pytest.mark.slow
def test_results_dev_md_generated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    out = tmp_path / "results_dev.md"
    assert script.main(["--split", "dev", "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "rule_based" in stdout and "wrote" in stdout
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Baseline results: dev split")
    for section in ("## Summary", "## Reward per episode", "## Steps per episode"):
        assert section in text
    assert "| Policy | Mean reward | Std | Min | Max | Mean steps |" in text
    header = "| Seed | Episode | random | rule_based | mark_everything | oracle |"
    assert text.count(header) == 2
    assert text.count("| 10000 | ep_") == 2  # first dev seed in both per-episode tables
    assert RESULTS_DEV.is_file(), "run `make baselines` and commit docs/results_dev.md"
    assert RESULTS_DEV.read_bytes() == out.read_bytes(), "results_dev.md drifted: make baselines"
