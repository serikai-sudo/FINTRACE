"""Slow end-to-end: the recording script over dev, run twice, byte-identical, sample pinned."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "trajectories" / "rule_based_dev_10000.jsonl"
NAME_RE = re.compile(r"^(random|rule_based|mark_everything|oracle)_dev_(\d+)\.jsonl$")


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
def test_record_trajectories_dev(tmp_path: Path) -> None:
    script = _load_script()
    assert script.main(["--split", "dev", "--out-dir", str(tmp_path)]) == 0
    files = sorted(tmp_path.glob("*.jsonl"))
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


@pytest.mark.slow
def test_trajectory_rerun_identical(tmp_path: Path) -> None:
    script = _load_script()
    first, second = tmp_path / "first", tmp_path / "second"
    assert script.main(["--split", "dev", "--out-dir", str(first)]) == 0
    assert script.main(["--split", "dev", "--out-dir", str(second)]) == 0
    hashes_first, hashes_second = _hashes(first), _hashes(second)
    assert len(hashes_first) == 40 and hashes_first == hashes_second
    assert SAMPLE.is_file(), "run `make trajectories` and commit the demo sample"
    assert (first / SAMPLE.name).read_bytes() == SAMPLE.read_bytes(), "demo sample drifted"
