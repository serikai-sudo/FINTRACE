"""Trajectory files carry the agent-visible transcript only: no truth, no seed, oracle included."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import load_split_config, run_baseline
from fintrace.evaluator.testing import oracle_for
from fintrace.logging.trajectory import FORBIDDEN_TOKENS, TrajectoryLogger, scan_trajectory
from fintrace.models import JsonObject

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_SEED = 10000
DISTINCTIVE_SEED = 918273645  # a substring hit on these digits cannot be a coincidence


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "record_trajectories.py"
    spec = importlib.util.spec_from_file_location("record_trajectories", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        found = set(value)
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found |= _keys(item)
        return found
    return set()


def test_no_truth_tokens_in_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "rule_based_dev_10000.jsonl"
    env = FintraceEnv()
    result = run_baseline(
        RuleBasedReconciliation(), env, DEV_SEED, load_split_config("dev"),
        logger=TrajectoryLogger(path),
    )  # fmt: skip
    assert result["reward"] > 0 and result["terminated"]
    assert scan_trajectory(path) == []
    lowered = path.read_text(encoding="utf-8").lower()
    assert not any(token in lowered for token in FORBIDDEN_TOKENS)
    assert "iss-" not in lowered  # issue ids never surface
    # positive controls: the scanner fires on a planted token, a planted key and a broken line
    planted = tmp_path / "planted.jsonl"
    planted.write_text(
        '{"action":{"thought":"I saw the Truth file"}}\n'
        '{"observation":{"result":{"episode_seed":1}},"metrics":{"f1":1.0}}\n'
        "not json\n",
        encoding="utf-8",
    )
    assert scan_trajectory(planted) == [
        "planted.jsonl:1: truth",
        "planted.jsonl:2: key episode_seed",
        "planted.jsonl:2: key f1",
        "planted.jsonl:2: key metrics",
        "planted.jsonl:3: not a JSON object",
    ]


def test_no_seed_in_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "distinctive.jsonl"
    env = FintraceEnv()
    run_baseline(
        RuleBasedReconciliation(), env, DISTINCTIVE_SEED, load_split_config("dev"),
        logger=TrajectoryLogger(path),
    )  # fmt: skip
    text = path.read_text(encoding="utf-8")
    assert str(DISTINCTIVE_SEED) not in text
    assert json.dumps(DISTINCTIVE_SEED) not in text
    dev_path = tmp_path / "dev.jsonl"
    run_baseline(
        RuleBasedReconciliation(), env, DEV_SEED, load_split_config("dev"),
        logger=TrajectoryLogger(dev_path),
    )  # fmt: skip
    for line in dev_path.read_text(encoding="utf-8").splitlines():
        keys = _keys(json.loads(line))
        assert not any("seed" in key.lower() for key in keys), keys
        assert not any("config" in key.lower() or "metric" in key.lower() for key in keys)


def test_oracle_trajectory_no_leak(tmp_path: Path) -> None:
    path = tmp_path / "oracle_dev_10000.jsonl"
    env = FintraceEnv()
    env.reset(seed=DEV_SEED, config=load_split_config("dev"))
    oracle = oracle_for(env)
    logger = TrajectoryLogger(path)
    result = run_baseline(oracle, env, DEV_SEED, load_split_config("dev"), logger=logger)
    assert result["reward"] > 0.9
    assert scan_trajectory(path) == []
    text = path.read_text(encoding="utf-8")
    assert "ISS-" not in text and "truth" not in text.lower()
    entries: list[JsonObject] = [json.loads(line) for line in text.splitlines()]
    tools = [e["action"]["tool"] for e in entries]
    assert set(tools[:-1]) == {"get_document"} and tools[-1] == "submit"
    assert all("oracle" in e["action"]["thought"] for e in entries)
    # the submit carries findings (public ids and amounts) but never the truth's field names
    submitted = json.dumps(entries[-1]["action"]["args"]["findings"]).lower()
    assert not any(token in submitted for token in FORBIDDEN_TOKENS)


def test_all_baselines_leak_scan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    assert script.main(["--split", "dev", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "leak scan: 40 files clean" in out
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 40
    for path in files:
        assert scan_trajectory(path) == [], path.name
        lowered = path.read_text(encoding="utf-8").lower()
        assert not any(token in lowered for token in FORBIDDEN_TOKENS), path.name
