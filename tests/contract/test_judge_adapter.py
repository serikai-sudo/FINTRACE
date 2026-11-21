"""The jury connection protocol (judge_adapter.py) against the published checks and rewards."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import judge_adapter
import pytest
from judge_adapter import _JudgeEnv

import fintrace.environment.env as env_module
from fintrace.constants import MAX_STEPS
from fintrace.evaluator.runner import load_split_config
from fintrace.exporters.episode_files import (
    EPISODE_FILES,
    read_episode,
    verify_episode,
    write_episode,
)
from fintrace.generator.build import build_episode
from fintrace.models import EpisodeState, JsonObject, PrivateEpisode, Truth
from fintrace.reward.official import official_reward
from fintrace.transforms.robustness import oracle_findings, remap_ids

DEV_SEED = 10000
FORBIDDEN = (
    "truth", "provenance", "seed", "issue_id", "expected_amount", "acceptable_evidence", "iss-",
)  # fmt: skip
Step = tuple[dict[str, object], float, bool, bool]


@pytest.fixture(scope="module")
def team_config() -> JsonObject:
    return json.loads(load_split_config("dev").model_dump_json())


@pytest.fixture(scope="module")
def episode_dir(tmp_path_factory: pytest.TempPathFactory, team_config: JsonObject) -> Path:
    out = tmp_path_factory.mktemp("jury") / "ep"
    judge_adapter.generate_episode(DEV_SEED, {"team_config": team_config}, str(out))
    return out


def _closed(episode_dir: Path) -> tuple[_JudgeEnv, dict[str, object]]:
    env = judge_adapter.make_env()
    observation, _ = env.reset(0, {"mode": "closed", "episode_dir": str(episode_dir)})
    return env, observation


def _list(env: _JudgeEnv, table: str, cursor: str | None = None, limit: int = 50) -> Step:
    args = {"table": table, "cursor": cursor, "limit": limit}
    observation, reward, terminated, truncated, _ = env.step({"name": "list_records", "args": args})
    return observation, reward, terminated, truncated


def _submit(env: _JudgeEnv, findings: list[JsonObject]) -> Step:
    args = {"findings": {"findings": findings}}
    observation, reward, terminated, truncated, _ = env.step({"name": "submit", "args": args})
    return observation, reward, terminated, truncated


def _records(observation: dict[str, object]) -> list[dict[str, object]]:
    result = observation["result"]
    assert isinstance(result, dict)
    records = result["records"]
    assert isinstance(records, list)
    return records


def _error(observation: dict[str, object]) -> dict[str, object]:
    error = observation["error"]
    assert isinstance(error, dict)
    return error


def test_adapter_entry_points_exist() -> None:
    sig = inspect.signature
    assert list(sig(judge_adapter.generate_episode).parameters) == ["seed", "config", "output_dir"]
    assert list(sig(judge_adapter.make_env).parameters) == []
    scoring = ["episode_dir", "findings", "inspection_steps"]
    assert list(sig(judge_adapter.score_episode).parameters) == scoring
    env = judge_adapter.make_env()
    assert list(sig(env.reset).parameters) == ["seed", "config"]
    assert list(sig(env.step).parameters) == ["action"]


def test_generate_episode_writes_all_files(
    episode_dir: Path, team_config: JsonObject, tmp_path: Path
) -> None:
    assert {p.name for p in episode_dir.iterdir()} == set(EPISODE_FILES)
    assert len(EPISODE_FILES) == 10 and verify_episode(episode_dir)
    assert read_episode(episode_dir) == build_episode(DEV_SEED, load_split_config("dev"))
    judge_adapter.generate_episode(DEV_SEED, {"team_config": team_config}, str(tmp_path / "b"))
    for name in sorted(EPISODE_FILES):  # byte-identical regeneration (self-check 12)
        assert (tmp_path / "b" / name).read_bytes() == (episode_dir / name).read_bytes(), name


def test_closed_mode_loads_exact_episode(
    episode_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = env_module.build_episode

    def counting(seed: int, config: object = None) -> object:
        calls.append(seed)
        return real(seed, config)  # type: ignore[arg-type]

    monkeypatch.setattr(env_module, "build_episode", counting)
    env, observation = _closed(episode_dir)
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    assert observation["manifest"] == manifest
    _list(env, "ledger")
    again, _ = env.reset(12345, {"mode": "closed", "episode_dir": str(episode_dir)})
    assert again["manifest"] == manifest and again["remaining_steps"] == MAX_STEPS
    assert calls == []  # nothing regenerated in closed mode
    missing = episode_dir.parent / "no_such_episode"
    with pytest.raises(ValueError, match=r"^episode could not be loaded$") as excinfo:
        env.reset(0, {"mode": "closed", "episode_dir": str(missing)})
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
    with pytest.raises(ValueError, match=r"^unknown mode; expected 'closed' or 'train'$"):
        env.reset(0, {"mode": "sekret-mode-42"})
    survivor, *_ = _list(env, "ledger")
    assert survivor["error"] is None  # the loaded episode survived both failed resets


def test_train_mode_generates_from_seed(team_config: JsonObject) -> None:
    env = judge_adapter.make_env()
    observation, info = env.reset(DEV_SEED, {"mode": "train", "team_config": team_config})
    expected = build_episode(DEV_SEED, load_split_config("dev"))
    manifest = observation["manifest"]
    assert isinstance(manifest, dict) and info == {}
    assert manifest["episode_id"] == expected.public.manifest.episode_id
    _, reward, terminated, _ = _submit(env, oracle_findings(expected.truth))
    assert terminated and reward > 0.9  # train mode: the real reward reaches the policy


def test_reset_obs_jury_format(episode_dir: Path) -> None:
    env = judge_adapter.make_env()
    observation, info = env.reset(0, {"mode": "closed", "episode_dir": str(episode_dir)})
    assert set(observation) == {"task", "manifest", "policy", "remaining_steps"}
    assert observation["remaining_steps"] == 100 and isinstance(observation["task"], str)
    assert info == {}
    lowered = json.dumps([observation, info]).lower()
    assert not any(token in lowered for token in FORBIDDEN)


def test_step_envelope_name_key(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    observation, *_ = _list(env, "invoices", limit=5)
    assert observation["error"] is None and len(_records(observation)) == 5
    assert observation["remaining_steps"] == 99
    junk_actions: tuple[object, ...] = (
        {"tool": "list_records", "args": {"table": "ledger"}}, {"args": {}}, "x", None, 42,
    )  # fmt: skip
    for junk in junk_actions:
        observation, reward, terminated, truncated, _ = env.step(junk)
        assert observation["result"] is None and _error(observation)["code"] == "INVALID_ACTION"
        assert reward == 0.0 and not terminated and not truncated
    assert observation["remaining_steps"] == 99 - len(junk_actions)  # junk still consumes budget


def test_step_obs_jury_format(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    ok, *_ = _list(env, "payments", limit=3)
    assert set(ok) == {"result", "error", "remaining_steps"} and ok["error"] is None
    assert isinstance(ok["result"], dict) and set(ok["result"]) == {"records", "next_cursor"}
    bad, *_ = env.step({"name": "get_record", "args": {"table": "invoices", "id": "INV-9999"}})
    assert set(bad) == {"result", "error", "remaining_steps"} and bad["result"] is None
    error = _error(bad)
    assert set(error) == {"code", "message"} and error["code"] == "NOT_FOUND"
    assert error["message"] == judge_adapter.ERROR_MESSAGES["NOT_FOUND"]
    assert "/" not in str(error["message"]) and "INV-9999" not in str(error["message"])
    first, *_ = _list(env, "invoices", limit=1)
    invoice_id = _records(first)[0]["invoice_id"]
    record, *_ = env.step({"name": "get_record", "args": {"table": "invoices", "id": invoice_id}})
    assert record["error"] is None and isinstance(record["result"], dict)
    assert {"invoice_id", "line_items", "total_minor"} <= set(record["result"])
    doc_id = _records(first)[0]["doc_id"]
    document, *_ = env.step({"name": "get_document", "args": {"doc_id": doc_id}})
    assert isinstance(document["result"], dict) and document["result"]["doc_id"] == doc_id


def test_closed_mode_reward_zero_to_policy(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    rewards = [_list(env, "ledger", limit=10)[1] for _ in range(3)]
    private = read_episode(episode_dir)
    observation, reward, terminated, truncated = _submit(env, oracle_findings(private.truth))
    assert observation["result"] == {"accepted": True} and terminated and not truncated
    assert rewards == [0.0, 0.0, 0.0] and reward == 0.0  # C19: the perfect submission still sees 0
    exact = {"findings": oracle_findings(private.truth)}
    assert judge_adapter.score_episode(str(episode_dir), exact, 3)["f1"] == 1.0


def test_score_episode_valid(episode_dir: Path) -> None:
    private = read_episode(episode_dir)
    exact = {"findings": oracle_findings(private.truth)}
    report = judge_adapter.score_episode(str(episode_dir), exact, 12)
    assert set(report) == {"tp", "fp", "fn", "precision", "recall", "f1", "return"}
    assert (report["tp"], report["fp"], report["fn"]) == (8, 0, 0)
    assert report["precision"] == report["recall"] == report["f1"] == 1.0
    assert report["return"] == official_reward(1.0, 12) > 0
    bare = judge_adapter.score_episode(str(episode_dir), exact["findings"], 12)  # type: ignore[arg-type]
    assert bare == report  # a bare list is accepted as well


def test_score_episode_invalid_structure(episode_dir: Path) -> None:
    private = read_episode(episode_dir)
    one = oracle_findings(private.truth)[0]
    nulls = {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None}
    too_many = judge_adapter.score_episode(str(episode_dir), {"findings": [one] * 31}, 5)
    assert too_many == {**nulls, "return": -1.0}
    bad_field = {"findings": [{**one, "currency": "usd"}]}
    assert judge_adapter.score_episode(str(episode_dir), bad_field, 5) == {**nulls, "return": -1.0}
    assert judge_adapter.score_episode(str(episode_dir), {"findings": "nope"}, 5)["return"] == -1.0
    assert judge_adapter.score_episode(str(episode_dir), {"findings": []}, -1)["return"] == -1.0


def test_calculator_decimal_string(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    observation, *_ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
    assert observation["result"] == "0.3" and isinstance(observation["result"], str)
    observation, *_ = env.step({"name": "calculator", "args": {"expression": "1/3"}})
    assert isinstance(observation["result"], str) and observation["result"].startswith("0.3333")
    observation, *_ = env.step({"name": "calculator", "args": {"expression": "1/0"}})
    assert observation["result"] is None and _error(observation)["code"] == "CALC_ERROR"


def test_pagination_stable_C05_C06(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    seen: list[str] = []
    cursor: str | None = None
    pages: dict[str | None, list[str]] = {}
    while True:
        observation, *_ = _list(env, "ledger", cursor, 50)
        assert isinstance(observation["result"], dict)
        assert set(observation["result"]) == {"records", "next_cursor"}  # no records_total
        ids = [str(r["row_id"]) for r in _records(observation)]
        pages[cursor] = ids
        seen.extend(ids)
        cursor = observation["result"]["next_cursor"]  # type: ignore[assignment]
        if cursor is None:
            break
    private = read_episode(episode_dir)
    assert len(seen) == len(set(seen)) == len(private.public.ledger)  # no gaps, no repeats
    second = next(c for c in pages if c is not None)
    again, *_ = _list(env, "ledger", second, 50)
    assert [str(r["row_id"]) for r in _records(again)] == pages[second]


def test_submit_as_action_100_C16(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    for _ in range(99):
        observation, _, terminated, truncated = _list(env, "fx", limit=1)
        assert not terminated and not truncated
    assert observation["remaining_steps"] == 1
    observation, reward, terminated, truncated = _submit(env, [])
    assert observation["result"] == {"accepted": True} and terminated and not truncated
    assert observation["remaining_steps"] == 0 and reward == 0.0


def test_truncate_at_100_no_submit_C17(episode_dir: Path) -> None:
    env, _ = _closed(episode_dir)
    for index in range(100):
        observation, reward, terminated, truncated = _list(env, "fx", limit=1)
        assert reward == 0.0 and not terminated
        assert truncated == (index == 99)
    assert observation["remaining_steps"] == 0
    after, *_ = env.step({"name": "list_records", "args": {"table": "fx"}})
    assert _error(after)["code"] == "EPISODE_OVER"
    assert judge_adapter.score_episode(str(episode_dir), {"findings": []}, 100)["return"] == -1.0


def test_60s_timeout_C18(
    episode_dir: Path, monkeypatch: pytest.MonkeyPatch, team_config: JsonObject
) -> None:
    monkeypatch.setattr(env_module, "TIME_BUDGET_SECONDS", 0.0)
    env, observation = _closed(episode_dir)
    assert "deadline" not in json.dumps(observation).lower()
    args = {"table": "ledger", "cursor": None, "limit": 50}
    action = {"name": "list_records", "args": args}
    observation, reward, terminated, truncated, info = env.step(action)
    assert truncated and not terminated and reward == 0.0 and info == {}
    assert _error(observation)["code"] == "EPISODE_OVER" and observation["result"] is None
    assert "deadline" not in json.dumps(observation).lower()
    assert "deadline" not in EpisodeState.model_fields
    assert judge_adapter.score_episode(str(episode_dir), {"findings": []}, 1)["return"] == -1.0
    # train mode: the truncation reward is the fixed empty-submission-at-MAX_STEPS value
    train = judge_adapter.make_env()
    train.reset(DEV_SEED, {"mode": "train", "team_config": team_config})
    probe = {"name": "list_records", "args": {"table": "fx"}}
    _, reward, terminated, truncated, _ = train.step(probe)
    assert truncated and reward == official_reward(0.0, MAX_STEPS) == -1.0


def test_isolation_through_adapter_C20(episode_dir: Path) -> None:
    env, observation = _closed(episode_dir)
    surfaces: list[str] = [json.dumps(observation)]
    docs: list[str] = []
    for table in ("ledger", "invoices", "payments", "fx"):
        cursor: str | None = None
        while True:
            page, *_ = _list(env, table, cursor, 50)
            surfaces.append(json.dumps(page))
            docs.extend(str(r["doc_id"]) for r in _records(page) if "doc_id" in r)
            assert isinstance(page["result"], dict)
            cursor = page["result"]["next_cursor"]  # type: ignore[assignment]
            if cursor is None or len(surfaces) > 40:
                break
    for doc_id in docs[:5]:
        page, *_ = env.step({"name": "get_document", "args": {"doc_id": doc_id}})
        surfaces.append(json.dumps(page))
    junk_actions = (
        {"name": "get_document", "args": {"doc_id": "DOC-0000"}},
        {"name": "nope"},
        {"name": "submit", "args": {"findings": 7}},
    )
    for junk in junk_actions:
        page, *_ = env.step(junk)
        surfaces.append(json.dumps(page))
    blob = "\n".join(surfaces).lower()
    for token in FORBIDDEN:
        assert token not in blob, token
    assert '"seed"' not in blob and '"deadline"' not in blob


def test_score_matches_reward_block(episode_dir: Path, tmp_path: Path) -> None:
    private = read_episode(episode_dir)
    exact = {"findings": oracle_findings(private.truth)}
    assert judge_adapter.score_episode(str(episode_dir), exact, 10)["return"] == pytest.approx(0.98)
    assert judge_adapter.score_episode(str(episode_dir), {"findings": []}, 10)["return"] == -1.0
    # one-issue truth for the T5 / T12 cases
    single = private.truth.issues[0]
    truth = Truth(episode_id=private.truth.episode_id, issues=(single,))
    write_episode(private.model_copy(update={"truth": truth}), tmp_path / "one", DEV_SEED)
    correct = oracle_findings(truth)[0]
    twice = judge_adapter.score_episode(str(tmp_path / "one"), {"findings": [correct, correct]}, 10)
    assert (twice["tp"], twice["fp"], twice["fn"]) == (1, 1, 0)
    assert math.isclose(twice["return"], 47 / 150, abs_tol=1e-12)  # T5: two identical findings
    wrong = {**correct, "affected_ids": ["INV-0000", "INV-0001"]}
    flood = {"findings": [correct, *([wrong] * 19)]}
    scored = judge_adapter.score_episode(str(tmp_path / "one"), flood, 10)
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 19, 0)
    assert math.isclose(scored["return"], -871 / 1050, abs_tol=1e-12)  # T12: mark-everything


def _same_id_remapped_copy(
    private: PrivateEpisode, seed: int
) -> tuple[PrivateEpisode, dict[str, str]]:
    """A copy with every id remapped but the ORIGINAL episode_id kept (the jury's paired copy)."""
    remapped, mapping = remap_ids(private, seed)
    episode_id = private.public.manifest.episode_id
    manifest = remapped.public.manifest.model_copy(update={"episode_id": episode_id})
    public = remapped.public.model_copy(update={"manifest": manifest})
    truth = remapped.truth.model_copy(update={"episode_id": episode_id})
    return remapped.model_copy(update={"public": public, "truth": truth}), mapping.id_map


def _get(env: _JudgeEnv, table: str, record_id: str) -> dict[str, object]:
    return env.step({"name": "get_record", "args": {"table": table, "id": record_id}})[0]


def test_paired_copy_same_episode_id_served_by_new_env(tmp_path: Path, episode_dir: Path) -> None:
    """Paired check as the jury runs it: the original episode first, then a NEW env object loads
    a transformed copy that keeps the same episode_id but carries remapped ids. The copy's env
    must serve the copy's records (the old ids are unknown to it) and the first env must keep
    serving the original: no view may be shared between the two episode objects."""
    original = read_episode(episode_dir)
    copy, id_map = _same_id_remapped_copy(original, seed=7)
    copy_dir = tmp_path / "copy"
    write_episode(copy, copy_dir, DEV_SEED)
    old_id = original.public.invoices[0].invoice_id
    new_id = id_map[old_id]
    assert new_id != old_id
    assert copy.public.manifest.episode_id == original.public.manifest.episode_id
    env_x, _ = _closed(episode_dir)
    assert _get(env_x, "invoices", old_id)["result"]["invoice_id"] == old_id  # primes the cache
    env_y, _ = _closed(copy_dir)  # second env object, same episode_id, different records
    hit = _get(env_y, "invoices", new_id)
    assert hit["error"] is None and hit["result"]["invoice_id"] == new_id
    assert _error(_get(env_y, "invoices", old_id))["code"] == "NOT_FOUND"
    evidence = copy.truth.issues[0].acceptable_evidence_sets[0]
    doc_id = next(x for x in evidence if x.startswith("DOC"))
    document = env_y.step({"name": "get_document", "args": {"doc_id": doc_id}})[0]
    assert document["error"] is None and document["result"]["doc_id"] == doc_id
    page = _records(_list(env_y, "invoices", None, 5)[0])
    expected = sorted(copy.public.invoices, key=lambda i: (i.issued_at, i.invoice_id))[:5]
    assert [r["invoice_id"] for r in page] == [i.invoice_id for i in expected]
    # the first env still serves its own episode, untouched by the second one
    assert _get(env_x, "invoices", old_id)["result"]["invoice_id"] == old_id
    assert _error(_get(env_x, "invoices", new_id))["code"] == "NOT_FOUND"
