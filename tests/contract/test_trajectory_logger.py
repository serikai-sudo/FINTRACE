"""TrajectoryLogger contract: canonical JSONL per step, thoughts kept, byte-stable, closed."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import load_split_config, run_baseline
from fintrace.logging.trajectory import (
    ENTRY_FIELDS,
    TrajectoryLogger,
    canonical_line,
    sanitize_action,
)
from fintrace.models import EpisodeConfig, JsonObject, ToolResult

LIST_LEDGER: JsonObject = {"tool": "list_records", "args": {"table": "ledger", "limit": 5}}


def _record(path: Path, seed: int = 42, config: EpisodeConfig | None = None) -> list[JsonObject]:
    env = FintraceEnv()
    trace: list[JsonObject] = []
    logger = TrajectoryLogger(path)
    run_baseline(RuleBasedReconciliation(), env, seed, config, trace=trace, logger=logger)
    assert not logger.is_open
    return trace


def _lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    assert text.endswith("\n")
    return text.split("\n")[:-1]


def test_logger_writes_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rule_based_dev_42.jsonl"
    trace = _record(path)
    lines = _lines(path)
    assert len(lines) == len(trace) >= 20
    entries = [json.loads(line) for line in lines]
    assert [e["step"] for e in entries] == list(range(1, len(lines) + 1))
    assert len({e["episode_id"] for e in entries}) == 1
    assert entries[-1]["terminated"] is True and entries[-1]["action"]["tool"] == "submit"
    assert all(e["terminated"] is False and e["reward"] == 0.0 for e in entries[:-1])


def test_logger_preserves_thought_field(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    trace = _record(path)
    entries = [json.loads(line) for line in _lines(path)]
    assert [e["action"]["thought"] for e in entries] == [a["thought"] for a in trace]
    assert all(len(e["action"]["thought"]) >= 10 for e in entries)
    assert [e["action"] for e in entries] == trace  # the action is recorded exactly as sent


def test_logger_byte_stable_rerun(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _record(first, 10001, load_split_config("dev"))
    _record(second, 10001, load_split_config("dev"))
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()


def test_logger_canonical_json(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _record(path)
    for line in _lines(path):
        entry = json.loads(line)
        assert tuple(entry) == ENTRY_FIELDS  # sorted keys, exactly the seven fields
        compact = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert line == compact == canonical_line(entry)
        assert not line.startswith('{"action": ')  # no spaces after separators
    assert canonical_line({"b": 1, "a": "é"}) == '{"a":"é","b":1}'  # ensure_ascii=False, compact
    with pytest.raises(ValueError):
        canonical_line({"x": float("nan")})


def test_logger_no_wall_clock(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _record(path)
    entries = [json.loads(line) for line in _lines(path)]
    for entry in entries:
        assert set(entry) == set(ENTRY_FIELDS)
        assert not any("time" in key or "elapsed" in key for key in entry["action"])
        observation = entry["observation"]
        assert set(observation) == {
            "ok", "tool", "result", "error", "steps_used", "steps_remaining",
        }  # fmt: skip
        ToolResult.model_validate(observation)  # exactly the public observation contract
    again = tmp_path / "again.jsonl"
    _record(again)
    assert again.read_bytes() == path.read_bytes()  # any wall-clock field would break this


def test_logger_step_count_matches_file_lines(tmp_path: Path) -> None:
    env = FintraceEnv()
    env.reset(seed=42)
    logger = TrajectoryLogger(tmp_path / "manual.jsonl")
    env.attach_logger(logger)
    for _ in range(50):
        env.step(LIST_LEDGER)
    assert logger.step_count == 50 and logger.is_open
    logger.close()
    assert len(_lines(tmp_path / "manual.jsonl")) == 50
    assert json.loads(_lines(tmp_path / "manual.jsonl")[-1])["observation"]["steps_used"] == 50


def test_logger_closed_after_terminal(tmp_path: Path) -> None:
    env = FintraceEnv()
    env.reset(seed=42)
    logger = TrajectoryLogger(tmp_path / "t.jsonl")
    env.attach_logger(logger)
    env.step(LIST_LEDGER)
    submit: JsonObject = {
        "tool": "submit",
        "args": {"episode_id": env.episode.manifest.episode_id, "findings": []},
    }
    _, _, terminated, _, _ = env.step(submit)
    assert terminated and not logger.is_open and env._logger is None
    env.step(LIST_LEDGER)  # EPISODE_OVER after the terminal step: not recorded
    lines = _lines(tmp_path / "t.jsonl")
    assert len(lines) == 2 and json.loads(lines[-1])["terminated"] is True
    observation = ToolResult.model_validate(json.loads(lines[0])["observation"])
    with pytest.raises(RuntimeError):
        logger.log_step(LIST_LEDGER, observation, 0.0, False, False)
    # a fresh reset never writes into a logger of the previous episode
    env.attach_logger(TrajectoryLogger(tmp_path / "second.jsonl"))
    env.reset(seed=43)
    assert env._logger is None
    env.step(LIST_LEDGER)
    assert _lines(tmp_path / "second.jsonl") == []


def test_logger_handles_invalid_action(tmp_path: Path) -> None:
    env = FintraceEnv()
    env.reset(seed=42)
    logger = TrajectoryLogger(tmp_path / "t.jsonl")
    env.attach_logger(logger)
    env.step({"tool": "nope", "args": {}})
    env.step("garbage")
    env.step({"tool": "list_records", "args": {"table": "x"}})
    env.step({"tool": "calculator", "args": {"expression": "1/0"}, "thought": "divide"})
    logger.close()
    entries = [json.loads(line) for line in _lines(tmp_path / "t.jsonl")]
    assert [e["observation"]["error"]["code"] for e in entries] == [
        "INVALID_ACTION", "INVALID_ACTION", "INVALID_TABLE", "CALC_ERROR",
    ]  # fmt: skip
    assert entries[0]["action"] == {"args": {}, "tool": "nope"}
    assert entries[1]["action"] == {"invalid_action": "str"}
    assert entries[3]["action"]["thought"] == "divide"
    assert all(e["observation"]["ok"] is False and e["reward"] == 0.0 for e in entries)
    assert sanitize_action({"tool": "x", "args": {"set": {1, 2}}})["args"]["set"] == "<set>"
    assert sanitize_action({"amount": Decimal("10.10")}) == {"amount": "10.10"}
    assert sanitize_action(None) == {"invalid_action": "NoneType"}
    assert sanitize_action({1: "non-string key"}) == {"1": "non-string key"}


class _ExplodingLogger(TrajectoryLogger):
    def log_step(self, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")


def test_logger_hostile_actions_never_break_the_step(tmp_path: Path) -> None:
    """Lone surrogates, absurd nesting, non-JSON objects, a failing writer: the step returns."""
    env = FintraceEnv()
    env.reset(seed=42)
    logger = TrajectoryLogger(tmp_path / "hostile.jsonl")
    env.attach_logger(logger)
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(5000):
        cursor["n"] = {}
        cursor = cursor["n"]  # type: ignore[assignment]
    hostile: list[object] = [
        {"tool": "list_records", "args": {"table": "ledger"}, "thought": "\ud800 lone surrogate"},
        {"tool": "list_records", "args": nested},
        {"tool": "list_records", "args": {"table": "ledger"}, "thought": object()},
        {"tool": "list_records", "args": {"table": "ledger"}, "thought": float("nan")},
        {"tool": "list_records", "args": {"table": "ledger"}, "thought": "ünïcödé ✓"},
    ]
    for action in hostile:
        observation, reward, terminated, truncated, _ = env.step(action)
        assert reward == 0.0 and not terminated and not truncated
    assert logger.is_open and logger.error is None and logger.step_count == 5
    logger.close()
    raw = (tmp_path / "hostile.jsonl").read_bytes()
    text = raw.decode("utf-8")  # strictly encodable
    assert not re.search(r"0x[0-9a-f]{6,}", text)  # no memory addresses
    entries = [json.loads(line) for line in text.split("\n")[:-1]]
    assert entries[0]["action"] == {"invalid_action": "unserialisable"}
    assert entries[1]["action"] == {"invalid_action": "unserialisable"}
    assert entries[2]["action"]["thought"] == "<object>"
    assert entries[3]["action"] == {"invalid_action": "unserialisable"}
    assert entries[4]["action"]["thought"] == "ünïcödé ✓" and "ünïcödé ✓" in text
    again = TrajectoryLogger(tmp_path / "again.jsonl")
    env.reset(seed=42)
    env.attach_logger(again)
    for action in hostile:
        env.step(action)
    again.close()
    assert (tmp_path / "again.jsonl").read_bytes() == raw  # byte-stable despite junk input
    # a writer that fails mid-episode is aborted and detached; the step itself still succeeds
    env.reset(seed=42)
    exploding = _ExplodingLogger(tmp_path / "exploding.jsonl")
    env.attach_logger(exploding)
    observation, *_ = env.step(LIST_LEDGER)
    assert observation.ok and env._logger is None
    assert exploding.error == "OSError" and not exploding.is_open
