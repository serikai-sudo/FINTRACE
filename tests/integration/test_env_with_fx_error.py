"""End-to-end: env.reset applies the fx_error injector; all three injectors combine cleanly."""

from __future__ import annotations

from fintrace.environment.env import FintraceEnv
from fintrace.generator.injectors import duplicate, fx_error, payment
from fintrace.models import EpisodeConfig, ErrorInjectionConfig, InjectorParams, JsonObject
from fintrace.reward.official import official_reward

FX_ONE = EpisodeConfig(errors=ErrorInjectionConfig(fx_error=InjectorParams(count=1)))
ALL_THREE = EpisodeConfig(
    errors=ErrorInjectionConfig(
        duplicate_invoice=InjectorParams(count=1),
        payment_allocation=InjectorParams(count=1),
        fx_error=InjectorParams(count=1),
    )
)
CHECKS = {
    "duplicate_invoice": duplicate.independent_check,
    "payment_allocation": payment.independent_check,
    "fx_error": fx_error.independent_check,
}


def _reset(config: EpisodeConfig, seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=config)
    return env


def test_env_reset_with_fx_config_produces_truth_issue() -> None:
    env = _reset(FX_ONE)
    assert env._private is not None
    issues = env._private.truth.issues
    assert len(issues) == 1 and issues[0].type == "fx_error" and issues[0].currency == "USD"
    assert env._private.truth.episode_id == env.episode.manifest.episode_id
    clean = FintraceEnv()
    clean.reset(seed=42)
    assert clean.episode.manifest.episode_id != env.episode.manifest.episode_id
    assert len(clean.episode.ledger) == len(env.episode.ledger)  # rewritten, nothing added
    observation, info = env.reset(seed=42, config=FX_ONE)
    surfaces = [observation.model_dump_json(), str(info)]
    for table in ("ledger", "fx"):
        result, *_ = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})
        surfaces.append(result.model_dump_json())
    secrets = ("ISS-", "variant", "truth", "provenance", "wrong_rate", "policy_rate", "seed")
    for secret in secrets:
        assert all(secret not in surface for surface in surfaces), secret


def test_env_correct_findings_against_fx_gets_high_reward() -> None:
    env = _reset(FX_ONE)
    assert env._private is not None
    issue = env._private.truth.issues[0]
    for _ in range(5):
        env.step({"tool": "list_records", "args": {"table": "fx", "limit": 50}})
    finding: JsonObject = {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "payment journal converted at a rate that is not the policy-date rate",
    }
    submission = {"episode_id": env.episode.manifest.episode_id, "findings": [finding]}
    observation, reward, terminated, truncated, _ = env.step({"tool": "submit", "args": submission})
    assert observation.ok and terminated and not truncated
    assert reward == official_reward(1.0, 5) >= 0.9


def test_env_all_three_injectors_combined() -> None:
    env = _reset(ALL_THREE)
    assert env._private is not None
    issues = env._private.truth.issues
    assert [issue.type for issue in issues] == [
        "duplicate_invoice", "payment_allocation", "fx_error",
    ]  # fmt: skip
    for issue in issues:
        CHECKS[issue.type](env.episode, issue)  # each still rediscoverable on the final books
    affected = [set(issue.affected_ids) for issue in issues]
    assert not (affected[0] & affected[1]) and not (affected[1] & affected[2])
    assert not (affected[0] & affected[2])
    assert len({issue.issue_id for issue in issues}) == 3
