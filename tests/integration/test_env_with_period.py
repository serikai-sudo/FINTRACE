"""End-to-end: env.reset applies the period injector; all four injectors combine cleanly."""

from __future__ import annotations

from fintrace.environment.env import FintraceEnv
from fintrace.generator.injectors import duplicate, fx_error, payment, period
from fintrace.models import EpisodeConfig, ErrorInjectionConfig, InjectorParams, JsonObject
from fintrace.reward.official import official_reward

PERIOD_ONE = EpisodeConfig(
    errors=ErrorInjectionConfig(period_misclassification=InjectorParams(count=1))
)
ALL_FOUR = EpisodeConfig(
    errors=ErrorInjectionConfig(
        duplicate_invoice=InjectorParams(count=1),
        payment_allocation=InjectorParams(count=1),
        fx_error=InjectorParams(count=1),
        period_misclassification=InjectorParams(count=1),
    )
)
CHECKS = {
    "duplicate_invoice": duplicate.independent_check,
    "payment_allocation": payment.independent_check,
    "fx_error": fx_error.independent_check,
    "period_misclassification": period.independent_check,
}


def _reset(config: EpisodeConfig, seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=config)
    return env


def test_env_reset_with_period_config_produces_truth_issue() -> None:
    env = _reset(PERIOD_ONE)
    assert env._private is not None
    issues = env._private.truth.issues
    assert len(issues) == 1 and issues[0].type == "period_misclassification"
    invoice = next(i for i in env.episode.invoices if i.invoice_id in issues[0].affected_ids)
    assert issues[0].currency == invoice.currency
    assert issues[0].expected_amount_minor == invoice.total_minor
    observation, info = env.reset(seed=42, config=PERIOD_ONE)
    surfaces = [observation.model_dump_json(), str(info)]
    for table in ("ledger", "invoices"):
        result, *_ = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})
        surfaces.append(result.model_dump_json())
    secrets = ("ISS-", "variant", "truth", "provenance", "boundary", "shift", "seed")
    for secret in secrets:
        assert all(secret not in surface for surface in surfaces), secret


def test_env_correct_findings_against_period_gets_high_reward() -> None:
    env = _reset(PERIOD_ONE)
    assert env._private is not None
    issue = env._private.truth.issues[0]
    for _ in range(6):
        env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 50}})
    finding: JsonObject = {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "invoice and its journal fall in different periods in the policy timezone",
    }
    submission = {"episode_id": env.episode.manifest.episode_id, "findings": [finding]}
    observation, reward, terminated, truncated, _ = env.step({"tool": "submit", "args": submission})
    assert observation.ok and terminated and not truncated
    assert reward == official_reward(1.0, 6) >= 0.9


def test_env_all_four_injectors_combined() -> None:
    env = _reset(ALL_FOUR)
    assert env._private is not None
    issues = env._private.truth.issues
    assert [issue.type for issue in issues] == [
        "duplicate_invoice", "payment_allocation", "fx_error", "period_misclassification",
    ]  # fmt: skip
    for issue in issues:
        CHECKS[issue.type](env.episode, issue)  # each still rediscoverable on the final books
    affected = [set(issue.affected_ids) for issue in issues]
    for i, first in enumerate(affected):
        for second in affected[i + 1 :]:
            assert not first & second
    assert len({issue.issue_id for issue in issues}) == 4
