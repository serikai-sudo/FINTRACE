"""Contract tests for the four baselines: random, rule_based (thoughts), mark_everything, oracle."""

from __future__ import annotations

from fintrace.agents.base import Policy
from fintrace.agents.mark_everything import MarkEverythingAttack
from fintrace.agents.random_agent import MAX_RANDOM_STEPS, RandomInvestigator
from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.constants import ISSUE_TYPES, MAX_FINDINGS, IssueType
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.runner import (
    BASELINE_FACTORIES,
    EpisodeResult,
    load_split_config,
    run_baseline,
    run_split,
)
from fintrace.evaluator.testing import oracle_for
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    Finding,
    InjectorParams,
    JsonObject,
)

FORBIDDEN_IN_THOUGHTS = ("truth", "provenance", "issue_id", "expected_amount", "ISS-", "seed")
DEV_SEED = 10000


def _config(kind: IssueType) -> EpisodeConfig:
    return EpisodeConfig(errors=ErrorInjectionConfig(**{kind: InjectorParams(count=1)}))


def _run(
    policy: Policy, seed: int, config: EpisodeConfig | None = None
) -> tuple[FintraceEnv, EpisodeResult, list[JsonObject]]:
    env = FintraceEnv()
    trace: list[JsonObject] = []
    result = run_baseline(policy, env, seed, config, trace=trace)
    return env, result, trace


def _submitted(trace: list[JsonObject]) -> list[Finding]:
    submits = [action for action in trace if action["tool"] == "submit"]
    assert submits, "the policy never submitted"
    findings = submits[-1]["args"]["findings"]
    assert isinstance(findings, list)
    return [Finding.model_validate(f) for f in findings]


def _tp_fp_fn(env: FintraceEnv, trace: list[JsonObject]) -> tuple[int, int, int]:
    assert env._private is not None  # test-side oracle access
    return match(
        _submitted(trace), env._private.truth.issues, public_ids=public_ids_of(env._private.public)
    )


# ---------------------------------------------------------------- random


def test_random_agent_deterministic() -> None:
    runs = [_run(RandomInvestigator(seed=0), 42) for _ in range(2)]
    assert runs[0][2] == runs[1][2]  # identical action sequences
    assert runs[0][1]["reward"] == runs[1][1]["reward"]
    other = _run(RandomInvestigator(seed=1), 42)[2]
    assert other != runs[0][2]


def test_random_agent_respects_max_steps() -> None:
    for seed in range(5):
        _, result, trace = _run(RandomInvestigator(seed=seed), 42)
        assert 10 <= result["steps"] <= MAX_RANDOM_STEPS
        assert len(trace) == result["steps"]
        assert all(action["tool"] != "submit" for action in trace[:-1])


def test_random_agent_eventually_submits() -> None:
    env, result, trace = _run(RandomInvestigator(seed=3), DEV_SEED, load_split_config("dev"))
    assert result["terminated"] and not result["truncated"]
    assert trace[-1]["tool"] == "submit"
    assert trace[-1]["args"]["episode_id"] == env.episode.manifest.episode_id
    assert result["reward"] < 0.0  # self-check 9: the lower bound loses on erroneous books


# ---------------------------------------------------------------- rule based


def test_rule_based_finds_duplicate() -> None:
    env, result, trace = _run(RuleBasedReconciliation(), 42, _config("duplicate_invoice"))
    assert _tp_fp_fn(env, trace) == (1, 0, 0)
    finding = _submitted(trace)[0]
    assert env._private is not None
    issue = env._private.truth.issues[0]
    assert finding.type == "duplicate_invoice"
    assert tuple(finding.affected_ids) == issue.affected_ids
    assert finding.claimed_amount_minor == issue.expected_amount_minor
    assert result["reward"] > 0.5


def test_rule_based_finds_partial_payment() -> None:
    hits = 0
    for seed in (42, 43, 44):
        env, _, trace = _run(RuleBasedReconciliation(), seed, _config("payment_allocation"))
        tp, fp, _ = _tp_fp_fn(env, trace)
        assert fp == 0  # never a false alarm on an otherwise clean book
        hits += tp
    assert hits >= 2  # the document scan reaches most payments within the budget


def test_rule_based_finds_fx_error() -> None:
    env, _, trace = _run(RuleBasedReconciliation(), 42, _config("fx_error"))
    assert _tp_fp_fn(env, trace) == (1, 0, 0)
    finding = next(f for f in _submitted(trace) if f.type == "fx_error")
    assert env._private is not None
    issue = env._private.truth.issues[0]
    assert finding.currency == env.episode.manifest.base_currency
    assert finding.claimed_amount_minor == issue.expected_amount_minor
    assert tuple(finding.affected_ids) == issue.affected_ids


def test_rule_based_finds_period_misclassification() -> None:
    env, _, trace = _run(RuleBasedReconciliation(), 42, _config("period_misclassification"))
    assert _tp_fp_fn(env, trace) == (1, 0, 0)


def test_rule_based_writes_thoughts() -> None:
    env, _, trace = _run(RuleBasedReconciliation(), DEV_SEED, load_split_config("dev"))
    assert len(trace) >= 20
    for action in trace:
        thought = action.get("thought")
        assert isinstance(thought, str) and 10 <= len(thought) <= 500, action
        lowered = thought.lower()
        assert not any(token.lower() in lowered for token in FORBIDDEN_IN_THOUGHTS), thought
    assert trace[0]["thought"].startswith("Starting investigation of ep_")
    assert "submitting" in trace[-1]["thought"]
    # the narrative names the records it checks
    assert any("INV-" in str(a["thought"]) for a in trace)
    assert any("PAY-" in str(a["thought"]) for a in trace)
    tp, fp, _ = _tp_fp_fn(env, trace)
    assert tp >= 1 and fp == 0  # self-check 6: F1 > 0 on the first dev episode
    # clean books: the same agent raises no finding at all
    _, result, clean_trace = _run(RuleBasedReconciliation(), 42, EpisodeConfig())
    assert _submitted(clean_trace) == [] and result["terminated"]


# ---------------------------------------------------------------- mark everything


def test_mark_everything_submits_30_findings() -> None:
    env, result, trace = _run(MarkEverythingAttack(), DEV_SEED, load_split_config("dev"))
    findings = _submitted(trace)
    assert len(findings) == MAX_FINDINGS == 30
    assert {f.type for f in findings} == set(ISSUE_TYPES)
    assert result["n_findings"] == 30 and result["terminated"]
    assert result["steps"] <= 8  # two tables, a few pages each, then submit
    assert _tp_fp_fn(env, trace)[0] == 0


def test_mark_everything_scores_negative_on_all_dev_episodes() -> None:
    results = run_split(BASELINE_FACTORIES["mark_everything"], "dev")
    assert len(results) == 10
    rewards = [r["reward"] for r in results]
    assert sum(rewards) / len(rewards) < -0.5
    assert all(r["n_findings"] == 30 and r["terminated"] for r in results)


# ---------------------------------------------------------------- oracle


def test_oracle_achieves_f1_one_on_dev_episodes() -> None:
    results = run_split(BASELINE_FACTORIES["oracle"], "dev")
    assert len(results) == 10
    for result in results:
        assert result["reward"] > 0.9, result
        assert result["terminated"] and result["n_findings"] == 8
    env = FintraceEnv()
    env.reset(seed=DEV_SEED, config=load_split_config("dev"))
    oracle = oracle_for(env)
    trace: list[JsonObject] = []
    result = run_baseline(oracle, env, DEV_SEED, load_split_config("dev"), trace=trace)
    assert result["steps"] == len(oracle.evidence_documents) + 1
    assert _tp_fp_fn(env, trace) == (8, 0, 0)
    assert all(a["tool"] == "get_document" for a in trace[:-1]) and trace[-1]["tool"] == "submit"
