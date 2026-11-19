"""M1 end-to-end: env -> submit -> matcher -> metrics -> official reward, and layer isolation.

Truth is empty until the Day-2 injectors exist, so these tests register a synthetic Truth built
from real public ids into the environment's TruthStore (test-only "oracle" access).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fintrace.constants import MAX_STEPS
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.metrics import compute_metrics
from fintrace.models import EpisodeConfig, Issue, JsonObject, Truth
from fintrace.reward.official import official_reward

SRC = Path(__file__).resolve().parents[2] / "src" / "fintrace"
LIST_LEDGER_1: JsonObject = {"tool": "list_records", "args": {"table": "ledger", "limit": 1}}


def _env(seed: int = 42) -> FintraceEnv:
    env = FintraceEnv()
    env.reset(seed=seed, config=EpisodeConfig())
    return env


def _register_synthetic_truth(env: FintraceEnv, n_issues: int) -> Truth:
    """Oracle-side helper: one duplicate_invoice issue per real invoice, evidence = its doc_id."""
    invoices = env.episode.invoices[:n_issues]
    assert len(invoices) == n_issues
    issues = tuple(
        Issue(
            issue_id=f"ISS-{k + 1:02d}",
            type="duplicate_invoice",
            affected_ids=(invoice.invoice_id,),
            expected_amount_minor=invoice.total_minor,
            currency=invoice.currency,
            acceptable_evidence_sets=((invoice.doc_id,),),
        )
        for k, invoice in enumerate(invoices)
    )
    truth = Truth(episode_id=env.episode.manifest.episode_id, issues=issues)
    env._truth_store.register(truth)  # test-only: mimics the oracle's view of the truth
    return truth


def _finding_for(issue: Issue, **over: object) -> JsonObject:
    data: JsonObject = {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.acceptable_evidence_sets[0]),
        "explanation": "oracle",
    }
    data.update(over)  # type: ignore[arg-type]
    return data


def _submit(env: FintraceEnv, findings: list[JsonObject]) -> tuple:
    episode_id = env.episode.manifest.episode_id
    return env.step({"tool": "submit", "args": {"episode_id": episode_id, "findings": findings}})


def test_full_loop_perfect_submit() -> None:
    env = _env()
    truth = _register_synthetic_truth(env, 4)
    for _ in range(3):
        env.step(LIST_LEDGER_1)
    observation, reward, terminated, truncated, info = _submit(
        env, [_finding_for(issue) for issue in truth.issues]
    )
    assert observation.ok and observation.result == {"status": "accepted", "findings_count": 4}
    assert terminated and not truncated and info == {"max_steps": 100, "version": "1.0.0"}
    assert reward == official_reward(1.0, 3) == pytest.approx(0.994, abs=1e-12)
    assert reward > 0.9


def test_full_loop_silent_on_clean_episode() -> None:
    env = _env(seed=7)  # default EpisodeConfig: clean books, empty truth
    assert env._private is not None and env._private.truth.issues == ()
    for _ in range(3):
        env.step(LIST_LEDGER_1)
    _, reward, terminated, truncated, _ = _submit(env, [])
    assert terminated and not truncated
    assert reward == official_reward(1.0, 3) > 0.9
    accuser = _env(seed=7)
    invoice = accuser.episode.invoices[0]  # a plausible but wrong claim about a real invoice
    accusation: JsonObject = {
        "type": "duplicate_invoice",
        "affected_ids": [invoice.invoice_id],
        "claimed_amount_minor": invoice.total_minor,
        "currency": invoice.currency,
        "evidence_ids": [invoice.doc_id],
        "explanation": "false accusation",
    }
    _, reward, *_ = _submit(accuser, [accusation])
    assert reward == -1.0  # false accusation on clean books
    # reset bookkeeping: exactly one truth registered at any time, keyed by the live episode
    store = accuser._truth_store
    assert len(store) == 1 and store.get(accuser.episode.manifest.episode_id) is not None
    accuser.reset(seed=8)
    assert len(store) == 1 and store.get(accuser.episode.manifest.episode_id) is not None
    accuser.reset(seed=8)
    assert len(store) == 1


def test_full_loop_truncation_gives_negative_reward() -> None:
    env = _env(seed=3)
    _register_synthetic_truth(env, 5)
    for step in range(MAX_STEPS - 1):
        _, reward, terminated, truncated, _ = env.step(LIST_LEDGER_1)
        assert (reward, terminated, truncated) == (0.0, False, False), step
    _, reward, terminated, truncated, _ = env.step(LIST_LEDGER_1)
    assert truncated and not terminated and reward == -1.0  # F1 = 0 over 5 issues, -1.2 clipped
    assert env.step(LIST_LEDGER_1)[1] == 0.0  # EPISODE_OVER no-op: no reward twice
    # an invalid action as the 100th step truncates with the same reward
    invalid_last = _env(seed=3)
    _register_synthetic_truth(invalid_last, 5)
    for _ in range(MAX_STEPS - 1):
        invalid_last.step(LIST_LEDGER_1)
    _, reward, _, truncated, _ = invalid_last.step({"tool": "nonsense", "args": {}})
    assert truncated and reward == -1.0
    # clean books: truncation is a silent perfect audit minus the full step cost
    clean = _env(seed=3)
    for _ in range(MAX_STEPS):
        _, reward, _, truncated, _ = clean.step(LIST_LEDGER_1)
    assert truncated and reward == official_reward(1.0, MAX_STEPS) == pytest.approx(0.8)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_full_loop_tools_cannot_import_evaluator() -> None:
    offenders: list[str] = []
    guarded = [*sorted((SRC / "tools").glob("*.py")), *sorted((SRC / "agents").rglob("*.py"))]
    assert guarded, "expected tools/ and agents/ sources to scan"
    for path in guarded:
        for module in sorted(_imported_modules(path)):
            if "evaluator" in module or "reward" in module:
                offenders.append(f"{path.relative_to(SRC)}: {module}")
    assert not offenders, f"tools/ and agents/ must not import evaluator or reward: {offenders}"
    env_modules = _imported_modules(SRC / "environment" / "env.py")
    assert {"fintrace.evaluator.matcher", "fintrace.reward.official"} <= env_modules
    assert "fintrace.reward.experimental" not in env_modules


def test_full_loop_submit_returns_canonical_reward() -> None:
    env = _env(seed=9)
    truth = _register_synthetic_truth(env, 4)
    for _ in range(6):
        env.step(LIST_LEDGER_1)
    wrong = _finding_for(truth.issues[2], claimed_amount_minor=1)  # amount outside tolerance
    _, reward, terminated, _, _ = _submit(
        env, [_finding_for(truth.issues[0]), _finding_for(truth.issues[1]), wrong]
    )
    assert terminated
    _, _, f1 = compute_metrics(2, 1, 2)
    assert f1 == pytest.approx(4 / 7)
    assert reward == official_reward(f1, 6)
    assert reward == pytest.approx(2 * 4 / 7 - 1 - 0.012)
