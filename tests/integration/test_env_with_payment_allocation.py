"""End-to-end: env.reset applies the payment_allocation injector; rewards follow the real truth."""

from __future__ import annotations

from fintrace.environment.env import FintraceEnv
from fintrace.generator.injectors.payment import VARIANTS
from fintrace.models import EpisodeConfig, ErrorInjectionConfig, InjectorParams, JsonObject
from fintrace.reward.official import official_reward

PAYMENT_ONE = EpisodeConfig(
    errors=ErrorInjectionConfig(payment_allocation=InjectorParams(count=1))
)
VARIANT_SEEDS = range(20)  # first seeds on which every variant shows up at least once


def _reset(seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=PAYMENT_ONE)
    return env


def test_env_reset_with_payment_config_produces_truth_issue() -> None:
    env = _reset()
    assert env._private is not None
    issues = env._private.truth.issues
    assert len(issues) == 1 and issues[0].type == "payment_allocation"
    assert env._private.truth.episode_id == env.episode.manifest.episode_id
    clean = FintraceEnv()
    clean.reset(seed=42)
    assert clean.episode.manifest.episode_id != env.episode.manifest.episode_id
    assert len(clean.episode.payments) == len(env.episode.payments)  # nothing added, one changed
    observation, info = env.reset(seed=42, config=PAYMENT_ONE)
    surfaces = [observation.model_dump_json(), str(info)]
    result, *_ = env.step({"tool": "list_records", "args": {"table": "payments", "limit": 50}})
    surfaces.append(result.model_dump_json())
    for secret in ("ISS-", "variant", "truth", "provenance", "transposition", "seed"):
        assert all(secret not in surface for surface in surfaces)


def test_env_correct_findings_against_payment_gets_high_reward() -> None:
    env = _reset()
    assert env._private is not None
    issue = env._private.truth.issues[0]
    for _ in range(4):
        env.step({"tool": "list_records", "args": {"table": "payments", "limit": 50}})
    finding: JsonObject = {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "payment allocation does not reconcile with the payment amount",
    }
    submission = {"episode_id": env.episode.manifest.episode_id, "findings": [finding]}
    observation, reward, terminated, truncated, _ = env.step({"tool": "submit", "args": submission})
    assert observation.ok and terminated and not truncated
    assert reward == official_reward(1.0, 4) >= 0.9


def test_env_reset_with_all_four_variants_across_seeds() -> None:
    seen: set[str] = set()
    for seed in VARIANT_SEEDS:
        env = _reset(seed)
        assert env._private is not None
        entry = env._private.provenance.injections[0]
        assert entry.op == "payment_allocation"
        seen.add(str(entry.modified["variant"]))
    assert seen == set(VARIANTS)
