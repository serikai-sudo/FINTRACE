"""Contract tests for FintraceEnv: reset/step shape, budget, pagination, calculator, submit."""

from __future__ import annotations

import json
import re

import pytest

from fintrace.constants import ENV_VERSION, EPISODE_ID_PATTERN, MAX_FINDINGS, MAX_STEPS
from fintrace.environment.env import FintraceEnv
from fintrace.generator.build import build_episode
from fintrace.models import EpisodeConfig, JsonObject, PrivateEpisode, ToolResult
from fintrace.reward.official import official_reward
from fintrace.transforms.robustness import remap_ids

LIST_LEDGER_1: JsonObject = {
    "tool": "list_records",
    "args": {"table": "ledger", "cursor": None, "limit": 1},
}


def _env(seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed)
    return env


def _list(env: FintraceEnv, table: str, cursor: str | None = None, limit: int = 50) -> ToolResult:
    action = {"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": limit}}
    return env.step(action)[0]


def _calc(env: FintraceEnv, expression: str) -> ToolResult:
    return env.step({"tool": "calculator", "args": {"expression": expression}})[0]


def _submit(env: FintraceEnv, episode_id: str | None = None) -> tuple:
    episode_id = episode_id or env.episode.manifest.episode_id
    return env.step({"tool": "submit", "args": {"episode_id": episode_id, "findings": []}})


def test_reset_returns_observation_and_info() -> None:
    env = FintraceEnv()
    observation, info = env.reset(seed=42)
    assert info == {"max_steps": MAX_STEPS, "version": ENV_VERSION}
    assert info == {"max_steps": 100, "version": "1.0.0"}
    assert observation.ok is True and observation.tool == "reset"
    assert observation.steps_used == 0 and observation.steps_remaining == MAX_STEPS
    assert set(observation.result) == {"manifest", "policy", "tool_schema", "budget"}
    manifest = observation.result["manifest"]
    assert isinstance(manifest, dict)
    assert re.fullmatch(EPISODE_ID_PATTERN, str(manifest["episode_id"]))
    assert observation.result["tool_schema"] == [
        "list_records", "get_record", "get_document", "calculator", "submit",
    ]  # fmt: skip
    assert observation.result["budget"] == {"steps_remaining": MAX_STEPS}
    assert "seed" not in observation.model_dump_json()


def test_info_never_contains_seed() -> None:
    env = FintraceEnv()
    _, info = env.reset(seed=42)
    assert set(info) == {"max_steps", "version"}
    *_, step_info = env.step(LIST_LEDGER_1)
    assert set(step_info) == {"max_steps", "version"}
    assert "42" not in str(info.values()) or info["max_steps"] == 100  # no seed echo


def test_reset_isolates_state() -> None:
    env = FintraceEnv()
    first, _ = env.reset(seed=1)
    env.step(LIST_LEDGER_1)
    assert env.state.step_count == 1
    second, _ = env.reset(seed=2)
    assert first.result["manifest"]["episode_id"] != second.result["manifest"]["episode_id"]
    assert env.state.step_count == 0 and not env.state.terminated and not env.state.truncated
    again, _ = env.reset(seed=1)
    assert again.model_dump() == first.model_dump()  # same seed -> identical reset observation


def test_step_counter_increments_on_valid_action() -> None:
    env = _env()
    observation, reward, terminated, truncated, _ = env.step(LIST_LEDGER_1)
    assert observation.ok and observation.tool == "list_records"
    assert observation.steps_used == 1 and observation.steps_remaining == MAX_STEPS - 1
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert env.state.step_count == 1


def test_step_counter_increments_on_invalid_action() -> None:
    env = _env()
    observation, reward, terminated, truncated, _ = env.step({"tool": "nonsense", "args": {}})
    assert not observation.ok and observation.error is not None
    assert observation.error.code == "INVALID_ACTION" and observation.result is None
    assert observation.steps_used == 1 and env.state.step_count == 1
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert env.step("not even a dict")[0].error.code == "INVALID_ACTION"
    bad_table = env.step({"tool": "list_records", "args": {"table": "truth"}})[0]
    assert bad_table.error.code == "INVALID_TABLE"
    bad_limit = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 51}})[0]
    assert bad_limit.error.code == "INVALID_ACTION"
    zero_limit = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 0}})[0]
    assert zero_limit.error.code == "INVALID_ACTION"
    # structural failures are protocol errors even when the field name is a mapped one
    extra_field = env.step({"tool": "calculator", "args": {"expression": "1", "table": "x"}})[0]
    assert extra_field.error.code == "INVALID_ACTION"
    missing_field = env.step({"tool": "calculator", "args": {}})[0]
    assert missing_field.error.code == "INVALID_ACTION"
    assert env.state.step_count == 7


def test_pagination_stable_across_reset() -> None:
    env = _env(seed=42)
    first = [r["invoice_id"] for r in _list(env, "invoices", limit=5).result["records"]]
    env.reset(seed=42)
    second = [r["invoice_id"] for r in _list(env, "invoices", limit=5).result["records"]]
    assert first == second and len(first) == 5
    other = FintraceEnv()
    other.reset(seed=42)
    assert [r["invoice_id"] for r in _list(other, "invoices", limit=5).result["records"]] == first


def test_pagination_cursor_out_of_range() -> None:
    env = _env()
    for cursor in ("c_99999", "c_-1", "c_01", "offset:5", "zzz"):
        observation = _list(env, "ledger", cursor=cursor, limit=5)
        assert not observation.ok and observation.error is not None, cursor
        assert observation.error.code == "INVALID_CURSOR", cursor
    total = _list(env, "ledger", limit=1).result["records_total"]
    assert _list(env, "ledger", cursor=f"c_{total}").error.code == "INVALID_CURSOR"
    assert _list(env, "ledger", cursor=f"c_{total - 1}").ok  # last row is reachable
    assert _list(env, "ledger", cursor="c 1").error.code == "INVALID_CURSOR"  # pattern failure
    first_page = _list(env, "ledger", cursor=None, limit=3).result
    assert _list(env, "ledger", cursor="c_0", limit=3).result == first_page  # c_0 == start


def test_pagination_next_cursor_consistent() -> None:
    env = _env()
    cursor: str | None = None
    ids: list[str] = []
    totals: set[int] = set()
    for page in range(5):
        observation = _list(env, "ledger", cursor=cursor, limit=10)
        assert observation.ok
        assert set(observation.result) == {"records", "next_cursor", "records_total"}
        records = observation.result["records"]
        assert len(records) == 10
        assert set(records[0]) == {
            "row_id", "timestamp", "account", "currency", "entity_id", "invoice_id", "payment_id",
        }  # fmt: skip
        ids.extend(r["row_id"] for r in records)
        totals.add(observation.result["records_total"])
        cursor = observation.result["next_cursor"]
        assert cursor == f"c_{(page + 1) * 10}"
    assert len(ids) == 50 and len(set(ids)) == 50
    rows = {r.row_id: r for r in env.episode.ledger}
    keys = [(rows[i].timestamp, i) for i in ids]
    assert keys == sorted(keys) and len(totals) == 1


def test_calculator_safe_simple() -> None:
    env = _env()
    observation = _calc(env, "125000 * 1.0842")
    assert observation.ok and observation.result == {"value": "135525.0000"}
    assert _calc(env, "(1 + 2) * 3 - 4 / 8").result["value"] == "8.5"
    assert _calc(env, "-7 // 2").result["value"] == "-3"
    assert _calc(env, "10 % 3").result["value"] == "1"
    # literals are exact Decimals taken from the source text, never float round-trips
    assert _calc(env, "0.1234567890123456789 * 1").result["value"] == "0.1234567890123456789"
    assert _calc(env, "123456789.123456789 + 0").result["value"] == "123456789.123456789"
    assert _calc(env, "1_000 + 1e3").result["value"] == "2000"  # no exponent in the output


def test_calculator_rejects_power() -> None:
    env = _env()
    observation = _calc(env, "2**10")
    assert not observation.ok and observation.error is not None
    assert observation.error.code == "CALC_ERROR"
    assert _calc(env, "2 ** 10000000").error.code == "CALC_ERROR"


def test_calculator_rejects_import() -> None:
    env = _env()
    hostile = (
        "__import__('os').system('ls')", "open('/etc/passwd')", "x", "1 if 1 else 2", "1 < 2",
    )  # fmt: skip
    for expression in hostile:
        observation = _calc(env, expression)
        assert observation.error is not None and observation.error.code == "CALC_ERROR", expression
    assert _calc(env, "1/0").error.code == "CALC_ERROR"
    assert _calc(env, "1e400 * 1").error.code == "CALC_ERROR"
    assert _calc(env, "1e308 * 1e308").error.code == "CALC_ERROR"  # magnitude cap
    assert _calc(env, "0x10 + 1").error.code == "CALC_ERROR"  # hex spelling is not money


def test_calculator_rejects_too_long() -> None:
    env = _env()
    observation = _calc(env, "1+" * 100 + "1")  # 201 chars
    assert observation.error is not None and observation.error.code == "CALC_ERROR"
    serialised = observation.model_dump_json()
    assert "TOO_LONG" not in serialised and "message" not in serialised
    assert observation.model_dump()["error"] == {"code": "CALC_ERROR"}


def test_submit_terminates_episode() -> None:
    env = _env()
    wrong = _submit(env, episode_id="ep_ffffffffffff")
    assert wrong[0].error is not None and wrong[0].error.code == "INVALID_SUBMISSION"
    assert wrong[2] is False and not env.state.terminated
    observation, reward, terminated, truncated, info = _submit(env)
    assert observation.ok and observation.result == {"status": "accepted", "findings_count": 0}
    assert terminated is True and truncated is False
    # clean books, empty submission -> F1 = 1; the failed submit counts as one inspection step
    assert reward == official_reward(1.0, 1) == pytest.approx(0.998)
    assert env.state.submitted and env.state.terminated and env.state.step_count == 2
    assert info == {"max_steps": MAX_STEPS, "version": ENV_VERSION}


def test_truncation_at_100_steps() -> None:
    env = _env()
    for step in range(1, MAX_STEPS):
        observation, _, terminated, truncated, _ = env.step(LIST_LEDGER_1)
        assert observation.ok and not terminated and not truncated, step
        assert observation.steps_remaining == MAX_STEPS - step
    observation, reward, terminated, truncated, _ = env.step(LIST_LEDGER_1)
    assert observation.ok and observation.steps_used == MAX_STEPS
    assert observation.steps_remaining == 0
    assert truncated is True and terminated is False
    # truncation scores as an empty submission after MAX_STEPS: clean books -> F1 = 1 -> 0.8
    assert reward == official_reward(1.0, MAX_STEPS) == pytest.approx(0.8)
    assert env.state.truncated and not env.state.submitted
    # an invalid action as the 100th step still truncates, with a code-only observation
    invalid_last = _env(seed=4)
    for _ in range(MAX_STEPS - 1):
        invalid_last.step(LIST_LEDGER_1)
    nonsense: JsonObject = {"tool": "nonsense", "args": {}}
    observation, reward, terminated, truncated, _ = invalid_last.step(nonsense)
    assert observation.error is not None and observation.error.code == "INVALID_ACTION"
    assert (observation.steps_used, observation.steps_remaining) == (MAX_STEPS, 0)
    assert (terminated, truncated) == (False, True) and reward == official_reward(1.0, MAX_STEPS)
    # a successful submit as the 100th step terminates instead of truncating
    submit_last = _env(seed=4)
    for _ in range(MAX_STEPS - 1):
        submit_last.step(LIST_LEDGER_1)
    observation, _, terminated, truncated, _ = _submit(submit_last)
    assert observation.ok and (terminated, truncated) == (True, False)


def test_episode_over_is_no_op() -> None:
    env = _env()
    _submit(env)
    before = env.state.model_dump()
    observation, reward, terminated, truncated, _ = env.step(LIST_LEDGER_1)
    assert observation.error is not None and observation.error.code == "EPISODE_OVER"
    assert observation.tool == "list_records" and observation.steps_used == before["step_count"]
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert env.state.model_dump() == before
    truncated_env = _env(seed=3)
    for _ in range(MAX_STEPS):
        truncated_env.step(LIST_LEDGER_1)
    observation, _, terminated, truncated, _ = truncated_env.step({"tool": "nonsense", "args": {}})
    assert observation.error is not None and observation.error.code == "EPISODE_OVER"
    assert (terminated, truncated) == (False, True) and truncated_env.state.step_count == MAX_STEPS
    with pytest.raises(RuntimeError, match="reset"):
        FintraceEnv().step(LIST_LEDGER_1)


def _get(env: FintraceEnv, table: str, record_id: str) -> ToolResult:
    return env.step({"tool": "get_record", "args": {"table": table, "id": record_id}})[0]


def test_two_envs_same_episode_id_keep_their_own_views() -> None:
    """Two environment objects holding different episodes under the same episode_id (a jury
    paired copy with remapped ids) never share a tool view: each serves exactly its own records."""
    original = build_episode(42, EpisodeConfig())
    remapped, mapping = remap_ids(original, 7)
    episode_id = original.public.manifest.episode_id
    manifest = remapped.public.manifest.model_copy(update={"episode_id": episode_id})
    public = remapped.public.model_copy(update={"manifest": manifest})
    truth = remapped.truth.model_copy(update={"episode_id": episode_id})
    copy: PrivateEpisode = remapped.model_copy(update={"public": public, "truth": truth})
    old_id = original.public.ledger[0].row_id
    new_id = mapping.id_map[old_id]
    env_x = FintraceEnv()
    env_x.reset_loaded(original)
    assert _get(env_x, "ledger", old_id).ok  # the view of the original is built and cached
    env_y = FintraceEnv()
    env_y.reset_loaded(copy)
    hit = _get(env_y, "ledger", new_id)
    assert hit.ok and hit.result is not None and hit.result["record"]["row_id"] == new_id
    miss = _get(env_y, "ledger", old_id)
    assert miss.error is not None and miss.error.code == "NOT_FOUND"
    page = _list(env_y, "ledger", None, 3)
    assert page.result is not None
    copy_rows = {row.row_id for row in copy.public.ledger}
    assert all(record["row_id"] in copy_rows for record in page.result["records"])
    assert _get(env_x, "ledger", old_id).ok  # the first env is unaffected by the second
    stale = _get(env_x, "ledger", new_id)
    assert stale.error is not None and stale.error.code == "NOT_FOUND"


def _submit_findings(env: FintraceEnv, findings: list[JsonObject]) -> tuple:
    action = {
        "tool": "submit",
        "args": {"episode_id": env.episode.manifest.episode_id, "findings": findings},
    }
    return env.step(action)


def _finding(**over: object) -> JsonObject:
    data: JsonObject = {
        "type": "duplicate_invoice",
        "affected_ids": ["INV-0001", "INV-0002"],
        "claimed_amount_minor": 1000,
        "currency": "USD",
        "evidence_ids": ["DOC-0001", "DOC-0002"],
        "explanation": "probe",
    }
    data.update(over)
    return data


@pytest.mark.parametrize(
    ("case", "findings"),
    [
        ("31 findings", [_finding()] * (MAX_FINDINGS + 1)),
        ("11 evidence ids", [_finding(evidence_ids=[f"DOC-{i:04d}" for i in range(11)])]),
        ("1001 characters", [_finding(explanation="a" * 1001)]),
    ],
)
def test_structurally_invalid_submit_terminates_with_minus_one(
    case: str, findings: list[JsonObject]
) -> None:
    """Jury Block 3, tests 18-20: a submission past a structural limit ends the episode at -1.

    The agent sees a code-only error, the step is charged, nothing is recorded as submitted, and
    the evaluator keeps the reason on its own side.
    """
    env = _env()  # clean books: an empty submission would score +1, a refused one must not
    observation, reward, terminated, truncated, info = _submit_findings(env, findings)
    assert observation.ok is False and observation.error is not None, case
    assert observation.model_dump()["error"] == {"code": "INVALID_ACTION"}, case
    assert (reward, terminated, truncated) == (-1.0, True, False), case
    assert env.state.step_count == 1 and not env.state.submitted and env.state.terminated
    assert info == {"max_steps": MAX_STEPS, "version": ENV_VERSION}
    entry = env.evaluator_journal[-1]
    assert (entry.event, entry.step, entry.code) == ("invalid_submission", 1, "INVALID_ACTION")
    assert entry.reason and all(isinstance(kind, str) for kind in entry.reason)
    # the journal is evaluator-side: no observation or info carries it
    blob = observation.model_dump_json() + json.dumps(info)
    assert "invalid_submission" not in blob and entry.reason[0] not in blob
    after, _, terminated_after, _, _ = env.step(LIST_LEDGER_1)
    assert after.error is not None and after.error.code == "EPISODE_OVER" and terminated_after


def test_refused_submission_boundaries() -> None:
    """What does NOT end the episode: a wrong episode_id (binding) and a junk envelope (C13).

    A structurally valid finding with wrong values is scored as a false positive, so the episode
    ends the ordinary way, through the submit itself.
    """
    env = _env()
    wrong_episode, reward, terminated, _, _ = env.step(
        {"tool": "submit", "args": {"episode_id": "ep_ffffffffffff", "findings": []}}
    )
    assert wrong_episode.error is not None and wrong_episode.error.code == "INVALID_SUBMISSION"
    assert (reward, terminated) == (0.0, False) and not env.state.terminated
    envelopes: list[object] = [
        {"tool": "shell", "args": {}},  # unknown action name (jury C13)
        {"tool": "submit"},  # no args: the action contract rejects the envelope, not the payload
        {"tool": "submit", "args": None},
        {"tool": "submit", "args": {"episode_id": env.episode.manifest.episode_id, "findings": []},
         "thought": "x" * 600},  # our own scratchpad field, not part of findings.json
    ]  # fmt: skip
    for envelope in envelopes:
        junk, reward, terminated, _, _ = env.step(envelope)
        assert junk.error is not None and junk.error.code == "INVALID_ACTION", envelope
        assert (reward, terminated) == (0.0, False), envelope
    assert env.evaluator_journal == () and not env.state.terminated
    accepted, reward, terminated, _, _ = _submit_findings(env, [_finding(type="made_up")])
    assert accepted.ok and terminated is True  # a wrong value is a false positive, not a crash
    assert reward == official_reward(0.0, env.state.step_count - 1) == -1.0 and env.state.submitted


def test_structurally_invalid_submit_as_action_100_terminates() -> None:
    """At the step limit the refusal still terminates (it never becomes a truncation)."""
    env = _env(seed=7)
    for _ in range(MAX_STEPS - 1):
        env.step(LIST_LEDGER_1)
    observation, reward, terminated, truncated, _ = _submit_findings(env, [_finding()] * 31)
    assert observation.error is not None and (reward, terminated, truncated) == (-1.0, True, False)
    assert env.state.step_count == MAX_STEPS and not env.state.truncated
    assert len(env.evaluator_journal) == 1 and env.evaluator_journal[0].step == MAX_STEPS
