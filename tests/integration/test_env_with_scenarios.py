"""End-to-end: interacting scenario A through the env; each linked issue earns its own credit."""

from __future__ import annotations

from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.metrics import compute_metrics
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    InteractingScenarioParams,
    Issue,
    JsonObject,
)
from fintrace.reward.official import official_reward

SCENARIO_A = EpisodeConfig(
    errors=ErrorInjectionConfig(interacting=InteractingScenarioParams(scenario_a=1))
)
SCENARIO_B = EpisodeConfig(
    errors=ErrorInjectionConfig(interacting=InteractingScenarioParams(scenario_b=1))
)
SECRETS = ("ISS-", "scenario", "interacts", "caused_by", "variant", "truth", "provenance")


def _reset(seed: int = 42, config: EpisodeConfig = SCENARIO_A) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=config)
    return env


def _finding(issue: Issue) -> JsonObject:
    return {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "foreign payment under-allocated and posted past the period boundary",
    }


def _submit(env: FintraceEnv, findings: list[JsonObject]) -> tuple:
    args: JsonObject = {"episode_id": env.episode.manifest.episode_id, "findings": findings}
    return env.step({"tool": "submit", "args": args})


def test_env_reset_with_scenario_a_produces_two_truth_issues() -> None:
    env = _reset()
    assert env._private is not None
    issues = env._private.truth.issues
    assert [i.type for i in issues] == ["payment_allocation", "period_misclassification"]
    assert issues[0].interacts_with == (issues[1].issue_id,)
    assert env._private.truth.episode_id == env.episode.manifest.episode_id
    observation, _ = env.reset(seed=42, config=SCENARIO_A)
    surfaces = [observation.model_dump_json()]
    for table in ("payments", "ledger", "invoices"):
        result, *_ = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})
        surfaces.append(result.model_dump_json())
    assert all(token not in surface for token in SECRETS for surface in surfaces)


def test_env_submit_one_finding_gets_partial_reward() -> None:
    env = _reset()
    assert env._private is not None
    partial, _ = env._private.truth.issues
    for _ in range(4):
        env.step({"tool": "list_records", "args": {"table": "payments", "limit": 50}})
    observation, reward, terminated, truncated, _ = _submit(env, [_finding(partial)])
    assert observation.ok and terminated and not truncated
    _, _, f1 = compute_metrics(1, 0, 1)  # TP 1, FP 0, FN 1 -> F1 = 2/3
    assert reward == official_reward(f1, 4) > 0.0


def test_env_submit_both_findings_perfect_reward() -> None:
    env = _reset()
    assert env._private is not None
    findings = [_finding(issue) for issue in env._private.truth.issues]
    for _ in range(4):
        env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 50}})
    observation, reward, terminated, _, _ = _submit(env, findings)
    assert observation.ok and terminated
    assert reward == official_reward(1.0, 4) >= 0.9


def test_env_scenario_a_cutoff_finding_alone_is_true_positive() -> None:
    env = _reset()
    assert env._private is not None
    _, cutoff = env._private.truth.issues
    for _ in range(4):
        env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 50}})
    observation, reward, terminated, _, _ = _submit(env, [_finding(cutoff)])
    assert observation.ok and terminated
    _, _, f1 = compute_metrics(1, 0, 1)
    assert reward == official_reward(f1, 4)  # one TP, not a false positive for the other issue


def test_env_scenario_b_each_finding_alone_is_true_positive() -> None:
    env = _reset(config=SCENARIO_B)
    assert env._private is not None
    issues = env._private.truth.issues
    assert [i.type for i in issues] == ["duplicate_invoice", "fx_error"]
    assert issues[0].currency != issues[1].currency == env.episode.manifest.base_currency
    surfaces = []
    for table in ("invoices", "payments", "ledger"):
        result, *_ = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})
        surfaces.append(result.model_dump_json())
    assert all(token not in surface for token in SECRETS for surface in surfaces)
    _, _, f1_half = compute_metrics(1, 0, 1)
    for issue in issues:
        single = _reset(config=SCENARIO_B)
        for _ in range(3):
            single.step({"tool": "list_records", "args": {"table": "fx", "limit": 50}})
        observation, reward, terminated, _, _ = _submit(single, [_finding(issue)])
        assert observation.ok and terminated
        assert reward == official_reward(f1_half, 3)
    both = _reset(config=SCENARIO_B)
    observation, reward, terminated, _, _ = _submit(both, [_finding(i) for i in issues])
    assert observation.ok and terminated and reward == official_reward(1.0, 0)
