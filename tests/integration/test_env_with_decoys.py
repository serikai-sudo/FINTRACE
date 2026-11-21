"""End-to-end: decoy-rich episodes punish mark-everything and still reward correct findings."""

from __future__ import annotations

from fintrace.constants import MAX_FINDINGS
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.generator.injectors.base import legitimate_pattern_ids
from fintrace.models import (
    DecoyCounts,
    EpisodeConfig,
    ErrorInjectionConfig,
    Finding,
    InjectorParams,
    Issue,
    JsonObject,
    ToolResult,
)
from fintrace.reward.official import official_reward

FULL = DecoyCounts(repeated_payment_amounts=2, split_payments=1, refunds=1, similar_invoices=1)
DECOYS_AND_ERRORS = EpisodeConfig(
    decoys=FULL,
    errors=ErrorInjectionConfig(
        duplicate_invoice=InjectorParams(count=1),
        payment_allocation=InjectorParams(count=1),
        fx_error=InjectorParams(count=1),
        period_misclassification=InjectorParams(count=1),
    ),
)


def _reset(seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=DECOYS_AND_ERRORS)
    return env


def _finding(issue: Issue) -> JsonObject:
    return {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "verified against the source documents and the ledger",
    }


Step = tuple[ToolResult, float, bool, bool, JsonObject]


def _submit(env: FintraceEnv, findings: list[JsonObject]) -> Step:
    args: JsonObject = {"episode_id": env.episode.manifest.episode_id, "findings": findings}
    return env.step({"tool": "submit", "args": args})


def test_env_with_all_decoys_and_all_injectors() -> None:
    env = _reset()
    assert env._private is not None
    issues = env._private.truth.issues
    assert [i.type for i in issues] == [
        "duplicate_invoice", "payment_allocation", "fx_error", "period_misclassification",
    ]  # fmt: skip
    legit = legitimate_pattern_ids(env._private)
    assert len(legit) >= 10
    for issue in issues:
        assert not set(issue.affected_ids) & legit
    clean = FintraceEnv()
    clean.reset(seed=42)
    assert len(env.episode.payments) > len(clean.episode.payments)  # decoys added documents
    observation, info = env.reset(seed=42, config=DECOYS_AND_ERRORS)
    surfaces = [observation.model_dump_json(), str(info)]
    for table in ("invoices", "payments", "ledger"):
        result, *_ = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})
        surfaces.append(result.model_dump_json())
    secrets = ("legitimate", "decoy", "ISS-", "truth", "provenance", "variant", "seed")
    assert all(token not in surface for token in secrets for surface in surfaces)


def test_mark_everything_on_decoy_episode_loses() -> None:
    env = _reset()
    assert env._private is not None
    issues = env._private.truth.issues
    invoice_by_id = {i.invoice_id: i for i in env.episode.invoices}
    shotgun: list[JsonObject] = []
    for pay in env.episode.payments[:MAX_FINDINGS]:  # flag every payment as misallocated
        invoice = invoice_by_id[pay.invoice_refs[0].invoice_id] if pay.invoice_refs else None
        shotgun.append(
            {
                "type": "payment_allocation",
                "affected_ids": [pay.payment_id, *([invoice.invoice_id] if invoice else [])],
                "claimed_amount_minor": pay.amount_minor,
                "currency": pay.currency,
                "evidence_ids": [pay.doc_id, *([invoice.doc_id] if invoice else [])],
                "explanation": "looks suspicious",
            }
        )
    assert len(shotgun) == MAX_FINDINGS
    findings = [Finding.model_validate(f) for f in shotgun]
    tp, fp, fn = match(findings, issues, public_ids=public_ids_of(env.episode))
    _, _, f1 = compute_metrics(tp, fp, fn)
    assert tp <= 1 and fp >= MAX_FINDINGS - 1 and f1 < 0.1  # at most the one real allocation issue
    observation, reward, terminated, _, _ = _submit(env, shotgun)
    assert observation.ok and terminated
    assert reward == official_reward(f1, 0) <= -0.9


def test_rule_based_correct_on_decoy_episode_wins() -> None:
    env = _reset()
    assert env._private is not None
    findings = [_finding(issue) for issue in env._private.truth.issues]
    for _ in range(6):
        env.step({"tool": "list_records", "args": {"table": "payments", "limit": 50}})
    observation, reward, terminated, _, _ = _submit(env, findings)
    assert observation.ok and terminated
    assert reward == official_reward(1.0, 6) >= 0.9
