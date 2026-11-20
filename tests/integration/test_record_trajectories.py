"""Slow end-to-end: the recording script over dev; structure, leak scan, sample, rerun identity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.agents.base import BASELINE_NAMES
from fintrace.evaluator.runner import split_seeds
from fintrace.logging.trajectory import FORBIDDEN_TOKENS, scan_trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "trajectories" / "rule_based_dev_10000.jsonl"
NAME_RE = re.compile(r"^(random|rule_based|mark_everything|oracle)_dev_(\d+)\.jsonl$")
RERUN_POLICIES = "random,rule_based,mark_everything,oracle"  # every policy, first seeds only
RERUN_SEEDS = 2


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "record_trajectories.py"
    spec = importlib.util.spec_from_file_location("record_trajectories", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hashes(root: Path) -> dict[str, str]:
    files = sorted(root.glob("*.jsonl"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.mark.slow
def test_record_trajectories_dev_with_rerun_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One full run (40 valid, clean files equal to the sample), then an identical partial rerun."""
    script = _load_script()
    first, second = tmp_path / "first", tmp_path / "second"
    assert script.main(["--split", "dev", "--out-dir", str(first)]) == 0
    assert "leak scan: 40 files clean" in capsys.readouterr().out
    files = sorted(first.glob("*.jsonl"))
    assert len(files) == 40
    seeds = {str(seed) for seed in split_seeds("dev")}
    for path in files:
        match = NAME_RE.match(path.name)
        assert match and match.group(2) in seeds, path.name
        lines = path.read_text(encoding="utf-8").split("\n")
        assert lines[-1] == "" and len(lines) >= 2
        entries = [json.loads(line) for line in lines[:-1]]
        assert [e["step"] for e in entries] == list(range(1, len(entries) + 1))
        assert entries[-1]["terminated"] is True and entries[-1]["action"]["tool"] == "submit"
        assert all(isinstance(e["action"].get("thought"), str) for e in entries)
    for name in BASELINE_NAMES:
        assert sum(1 for p in files if p.name.startswith(f"{name}_dev_")) == 10, name
    # leak scan of every produced file (the script scans too; this is the independent check)
    for path in files:
        assert scan_trajectory(path) == [], path.name
        lowered = path.read_text(encoding="utf-8").lower()
        assert not any(token in lowered for token in FORBIDDEN_TOKENS), path.name
    assert SAMPLE.is_file(), "run `make trajectories` and commit the demo sample"
    assert (first / SAMPLE.name).read_bytes() == SAMPLE.read_bytes(), "demo sample drifted"
    # rerun identity: the same policies on the same episodes give the same bytes
    argv = ["--split", "dev", "--out-dir", str(second), "--policies", RERUN_POLICIES]
    assert script.main([*argv, "--seeds", str(RERUN_SEEDS)]) == 0
    second_hashes = _hashes(second)
    assert len(second_hashes) == len(BASELINE_NAMES) * RERUN_SEEDS
    first_hashes = _hashes(first)
    assert all(first_hashes[name] == digest for name, digest in second_hashes.items())
