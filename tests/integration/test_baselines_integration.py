"""Slow end-to-end: all four baselines over the dev split and the committed results document."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.evaluator.runner import BASELINE_FACTORIES, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DEV = REPO_ROOT / "docs" / "results_dev.md"
NAMES = ("random", "rule_based", "mark_everything", "oracle")


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_baselines_dev_split_and_results_md(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One full dev run of the four baselines: ordering sanity, evaluator-scored P/R/F1 and the
    committed results_dev.md (its byte-pinned part; the timing below the marker is free)."""
    script = _load_script()
    assert set(NAMES) == set(BASELINE_FACTORIES)
    results, scores, timing = script.run_all("dev", list(NAMES))
    stats = {name: summarize(rows) for name, rows in results.items()}
    assert all(s["n"] == 10 for s in stats.values())
    assert stats["oracle"]["min_reward"] > 0.9
    assert stats["oracle"]["mean_reward"] > stats["rule_based"]["mean_reward"] > 0.0
    assert stats["rule_based"]["mean_reward"] > stats["random"]["mean_reward"]
    assert stats["random"]["max_reward"] <= 0.0
    assert stats["mark_everything"]["max_reward"] < -0.9
    assert all(s["mean_steps"] <= 100 for s in stats.values())
    # evaluator-scored quality: the oracle is perfect, the rule-based auditor never lies, the
    # attackers find nothing; every recomputed return equalled the environment's reward (checked
    # inside score_episode, which raises otherwise)
    pooled = {name: script.pooled(scores[name]) for name in NAMES}
    assert pooled["oracle"]["f1"] == 1.0 and pooled["oracle"]["fp"] == 0
    assert pooled["rule_based"]["precision"] == 1.0 and 0.5 < pooled["rule_based"]["recall"] < 1.0
    assert pooled["mark_everything"]["tp"] == 0 and pooled["mark_everything"]["fp"] > 0
    assert pooled["random"]["tp"] == 0
    assert set(timing) == {*NAMES, "total"} and all(v >= 0 for v in timing.values())
    # the committed report is exactly what this run renders above the timing marker
    text = script.render_markdown("dev", results, scores, script.config_hash(REPO_ROOT / "config"))
    assert text.startswith("# Baseline results: dev split")
    sections = ("Summary", "Reward per episode", "F1 per episode", "Steps per episode")
    assert all(f"## {section}" in text for section in (*sections, "Where each policy fails"))
    assert "| Policy | Precision | Recall | F1 | Return | Std | Min | Max | Steps |" in text
    assert "| oracle | 1.000 | 1.000 | 1.000 | +0.9" in text
    header = "| Seed | Episode | random | rule_based | mark_everything | oracle |"
    assert text.count(header) == 3
    assert text.count("| 10000 | ep_") == 3  # first dev seed in all three per-episode tables
    assert all(f"- **{name}**: finds" in text for name in NAMES)
    assert script.TIMING_MARKER not in text  # the deterministic part carries no wall clock
    assert RESULTS_DEV.is_file(), "run `make baselines` and commit docs/results_dev.md"
    committed = RESULTS_DEV.read_text(encoding="utf-8")
    assert committed.count(script.TIMING_MARKER) == 1
    pinned, timing_text = script.split_document(committed)
    assert pinned == text, "results_dev.md drifted: make baselines"
    assert script.compose_document(pinned, timing_text) == committed  # exact inverse
    assert "## Timing" in timing_text and all(f"| {name} |" in timing_text for name in NAMES)
    # the CLI itself, on one episode: writes the file and reports on stdout
    out = tmp_path / "results_dev.md"
    assert script.main(["--split", "dev", "--seeds", "1", "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "rule_based" in stdout and "F1=" in stdout and "wrote" in stdout
    written = out.read_text(encoding="utf-8")
    assert written.count("| 10000 | ep_") == 3 and script.TIMING_MARKER in written
