"""End-to-end: env.reset applies the duplicate injector and rewards the agent against real truth."""

from __future__ import annotations

from fintrace.environment.env import FintraceEnv
from fintrace.models import EpisodeConfig, ErrorInjectionConfig, InjectorParams, JsonObject
from fintrace.reward.official import official_reward

DUPLICATE_ONE = EpisodeConfig(
    errors=ErrorInjectionConfig(duplicate_invoice=InjectorParams(count=1))
)


def _reset(seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=DUPLICATE_ONE)
    return env


def _submit(env: FintraceEnv, findings: list[JsonObject]) -> tuple:
    episode_id = env.episode.manifest.episode_id
    return env.step({"tool": "submit", "args": {"episode_id": episode_id, "findings": findings}})


def test_env_reset_with_duplicate_config_produces_truth_issue() -> None:
    env = _reset()
    assert env._private is not None
    issues = env._private.truth.issues
    assert len(issues) == 1 and issues[0].type == "duplicate_invoice"
    assert env._private.truth.episode_id == env.episode.manifest.episode_id
    clean = FintraceEnv()
    clean.reset(seed=42)
    assert clean._private is not None and clean._private.truth.issues == ()
    assert clean.episode.manifest.episode_id != env.episode.manifest.episode_id
    # neither the reset observation nor any tool output carries anything about the injection
    observation, info = env.reset(seed=42, config=DUPLICATE_ONE)
    issue = issues[0]
    secrets = ("ISS-", "truth", "provenance", "expected_amount", "duplicate_invoice", "seed")
    surfaces = [observation.model_dump_json(), str(info)]
    for table in ("ledger", "invoices", "payments", "fx"):
        result, *_ = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})
        surfaces.append(result.model_dump_json())
    for invoice_id in issue.affected_ids:
        result, *_ = env.step(
            {"tool": "get_record", "args": {"table": "invoices", "id": invoice_id}}
        )
        assert result.ok and invoice_id in result.model_dump_json()
        surfaces.append(result.model_dump_json())
    for surface in surfaces:
        assert not any(secret in surface for secret in secrets), surface[:200]


def test_env_correct_findings_against_duplicate_gets_high_reward() -> None:
    env = _reset()
    assert env._private is not None
    issue = env._private.truth.issues[0]
    for _ in range(3):
        env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 50}})
    finding: JsonObject = {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "two invoices from one supplier with identical totals and lines",
    }
    observation, reward, terminated, truncated, _ = _submit(env, [finding])
    assert observation.ok and terminated and not truncated
    assert reward == official_reward(1.0, 3) >= 0.9


def test_env_silent_on_episode_with_duplicate_gets_low_reward() -> None:
    env = _reset()
    for _ in range(3):
        env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 50}})
    _, reward, terminated, _, _ = _submit(env, [])
    assert terminated and reward == -1.0  # F1 = 0 against one issue, clipped
