"""Isolation: nothing from the private episode reaches the agent surface; thought is inert."""

from __future__ import annotations

import json
import random

from fintrace.environment.env import FintraceEnv
from fintrace.models import Action, JsonObject

FORBIDDEN_MARKERS = (
    "truth",
    "provenance",
    "issue_id",
    "expected_amount",
    "acceptable_evidence",
    "interacts_with",
    "injections",
    "seed_hash",
    "generator_version",
    "PrivateEpisode",
    "Traceback",
)


ID_KEYS = (
    ("ledger", "row_id"), ("invoices", "invoice_id"), ("payments", "payment_id"), ("fx", "fx_id"),
)  # fmt: skip


def _harvest_ids(env: FintraceEnv, surfaced: list[str]) -> dict[str, list[str]]:
    """First page of ids per table and the document ids, taken through the public tools."""
    ids: dict[str, list[str]] = {}
    for table, key in ID_KEYS:
        action: JsonObject = {"tool": "list_records", "args": {"table": table, "limit": 20}}
        observation, *_, info = env.step(action)
        assert observation.ok and observation.result is not None
        surfaced.extend((observation.model_dump_json(), json.dumps(info)))
        ids[table] = [str(r[key]) for r in observation.result["records"]]
    docs, *_, info = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 5}})
    surfaced.extend((docs.model_dump_json(), json.dumps(info)))
    ids["documents"] = [str(r["doc_id"]) for r in docs.result["records"]]
    return ids


def _random_action(rng: random.Random, ids: dict[str, list[str]]) -> JsonObject:
    kind = rng.choice(["list", "record", "document", "calc"])
    table = rng.choice(["ledger", "invoices", "payments", "fx"])
    if kind == "list":
        cursor = f"c_{rng.randrange(5)}"
        args: JsonObject = {"table": table, "cursor": cursor, "limit": rng.randint(1, 50)}
        return {"tool": "list_records", "args": args}
    if kind == "record":
        return {"tool": "get_record", "args": {"table": table, "id": rng.choice(ids[table])}}
    if kind == "document":
        return {"tool": "get_document", "args": {"doc_id": rng.choice(ids["documents"])}}
    expression = f"{rng.randint(1, 99999)} * {rng.randint(1, 999)} / 100"
    return {"tool": "calculator", "args": {"expression": expression}}


SEED = 918273645  # distinctive digits: the blob is scanned for the seed itself


def test_no_truth_in_observations_via_100_random_steps() -> None:
    env = FintraceEnv()
    observation, info = env.reset(seed=SEED)
    surfaced: list[str] = [observation.model_dump_json(), json.dumps(info)]
    ids = _harvest_ids(env, surfaced)  # 5 steps, recorded too
    rng = random.Random(0)
    while env.state.step_count < 99:
        observation, reward, terminated, truncated, info = env.step(_random_action(rng, ids))
        assert observation.ok, observation
        assert (reward, terminated, truncated) == (0.0, False, False)
        surfaced.extend((observation.model_dump_json(), json.dumps(info)))
    observation, _, terminated, _, info = env.step(
        {"tool": "submit", "args": {"episode_id": env.episode.manifest.episode_id, "findings": []}}
    )
    assert observation.ok and terminated
    surfaced.extend((observation.model_dump_json(), json.dumps(info)))
    blob = "\n".join(surfaced)
    for marker in FORBIDDEN_MARKERS:
        assert marker not in blob, marker
    private = env._private  # the test deliberately inspects the evaluator side
    assert private is not None and private.truth.issues == ()
    assert str(SEED) not in blob and private.provenance.seed_hash not in blob
    assert private.provenance.history[0]["company"] not in blob  # company name is private too
    assert len(surfaced) == 2 * (1 + 100)  # reset + 100 steps, each with observation and info


def test_action_thought_preserved_in_logged_action_but_ignored_by_env() -> None:
    plain: JsonObject = {"tool": "list_records", "args": {"table": "invoices", "limit": 3}}
    with_thought: JsonObject = {**plain, "thought": "checking INV-0042 against its payments"}
    parsed = Action.model_validate(with_thought)
    assert parsed.thought == "checking INV-0042 against its payments"
    assert Action.model_validate(plain).thought is None
    env_a, env_b = FintraceEnv(), FintraceEnv()
    env_a.reset(seed=5)
    env_b.reset(seed=5)
    result_a, result_b = env_a.step(plain), env_b.step(with_thought)
    assert result_a[0].model_dump() == result_b[0].model_dump()
    assert result_a[1:] == result_b[1:]
    assert "checking INV-0042" not in result_b[0].model_dump_json()  # env does not echo it
    too_long = env_b.step({**plain, "thought": "x" * 501})[0]
    assert too_long.error is not None and too_long.error.code == "INVALID_ACTION"
