"""Return-type contract of the evaluator API: plain ints and floats only.

Pinning the shapes keeps the scorer's public surface free of issue objects, ids or amounts; the
leak scans over observations live in tests/security/.
"""

from __future__ import annotations

from fintrace.evaluator.matcher import build_compat_matrix, match
from fintrace.evaluator.metrics import compute_metrics
from fintrace.models import Finding, Issue
from fintrace.reward.official import official_reward

TRUTH = Issue(
    issue_id="ISS-0001",
    type="duplicate_invoice",
    affected_ids=("INV-A", "INV-B"),
    expected_amount_minor=1000,
    currency="USD",
    acceptable_evidence_sets=(("DOC-A", "DOC-B"),),
)
IDS = frozenset({"INV-A", "INV-B", "DOC-A", "DOC-B"})
FINDING = Finding(
    type="duplicate_invoice",
    affected_ids=["INV-A", "INV-B"],
    claimed_amount_minor=1000,
    currency="USD",
    evidence_ids=["DOC-A", "DOC-B"],
    explanation="",
)


def test_tp_fp_fn_tuple_only() -> None:
    """match() returns exactly (tp, fp, fn) as plain ints, nothing else."""
    for findings in ([], [FINDING], [FINDING, FINDING]):
        result = match(findings, (TRUTH,), public_ids=IDS)
        assert type(result) is tuple and len(result) == 3
        assert all(type(value) is int for value in result)
    assert match([], (TRUTH,), public_ids=IDS) == (0, 0, 1)
    matrix = build_compat_matrix([FINDING], (TRUTH,), public_ids=IDS)
    assert matrix == {0: [0]}  # indices only: no Issue, no id, no amount
    assert all(type(k) is int and all(type(j) is int for j in v) for k, v in matrix.items())


def test_compute_metrics_returns_only_floats() -> None:
    result = compute_metrics(3, 1, 2)
    assert type(result) is tuple and len(result) == 3
    assert all(type(value) is float for value in result)
    assert result == (0.75, 0.6, 2 * 3 / (2 * 3 + 1 + 2))


def test_official_reward_returns_only_float() -> None:
    reward = official_reward(0.5, 10)
    assert type(reward) is float
    assert reward == 2 * 0.5 - 1 - 0.002 * 10
