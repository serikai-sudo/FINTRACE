"""Matcher: normalisation, published tolerance, evidence, the five TP criteria, max matching."""

from __future__ import annotations

import pytest

from fintrace.constants import AMOUNT_TOLERANCE_MINOR, ISSUE_TYPES
from fintrace.evaluator.matcher import (
    amount_within_tolerance,
    build_compat_matrix,
    canonical_affected_ids,
    has_acceptable_evidence,
    is_true_positive,
    match,
    tolerance_for,
)
from fintrace.models import Finding, Issue

# every id the fixtures below may cite; anything else "does not exist" for the matcher
IDS = frozenset({f"INV-00{k}" for k in (1, 2, 3)} | {f"DOC-00{k}" for k in (1, 2, 3)})


def _issue(**over: object) -> Issue:
    data: dict[str, object] = {
        "issue_id": "ISS-01",
        "type": "duplicate_invoice",
        "affected_ids": ("INV-001", "INV-002"),
        "expected_amount_minor": 125000,
        "currency": "USD",
        "acceptable_evidence_sets": (("DOC-001", "DOC-002"),),
    }
    data.update(over)
    return Issue.model_validate(data)


def _finding(**over: object) -> Finding:
    data: dict[str, object] = {
        "type": "duplicate_invoice",
        "affected_ids": ["INV-002", "INV-001"],  # unsorted on purpose
        "claimed_amount_minor": 125000,
        "currency": "USD",
        "evidence_ids": ["DOC-002", "DOC-001", "DOC-003"],
        "explanation": "second invoice duplicates the first",
    }
    data.update(over)
    return Finding.model_validate(data)


def _match(findings: list[Finding], issues: tuple[Issue, ...]) -> tuple[int, int, int]:
    return match(findings, issues, public_ids=IDS)


def test_canonical_affected_ids_dedup_and_sort() -> None:
    assert canonical_affected_ids(["B", "A", "A", "B"]) == ("A", "B")
    assert canonical_affected_ids(("INV-010", "INV-002")) == ("INV-002", "INV-010")
    assert canonical_affected_ids(iter([])) == ()
    assert canonical_affected_ids(_finding().affected_ids) == _issue().affected_ids


def test_amount_tolerance_default_is_exact() -> None:
    """Jury default tau = 0: the integer minor-unit amounts must be equal."""
    assert amount_within_tolerance(1000, 1000)
    assert amount_within_tolerance(0, 0)
    assert not amount_within_tolerance(1001, 1000)
    assert not amount_within_tolerance(999, 1000)
    assert not amount_within_tolerance(11, 10) and not amount_within_tolerance(9, 10)
    assert not amount_within_tolerance(1, 0)
    assert all(AMOUNT_TOLERANCE_MINOR[t] == 0 for t in ISSUE_TYPES)
    assert all(tolerance_for(t) == 0 for t in ISSUE_TYPES) and tolerance_for("made_up") == 0


def test_amount_tolerance_published_integer_is_inclusive() -> None:
    """A documented per-type tau is an inclusive integer bound, never a fraction with a floor."""
    assert amount_within_tolerance(99900, 100000, 100)
    assert amount_within_tolerance(100100, 100000, 100)
    assert not amount_within_tolerance(99899, 100000, 100)
    assert not amount_within_tolerance(100101, 100000, 100)
    assert amount_within_tolerance(10001, 10000, 1) and not amount_within_tolerance(10002, 10000, 1)
    with pytest.raises(ValueError, match="non-negative"):
        amount_within_tolerance(1, 1, -1)


def test_evidence_superset_single_set() -> None:
    acceptable = (("A", "B"),)
    assert has_acceptable_evidence(frozenset({"A", "B", "C"}), acceptable)
    assert has_acceptable_evidence(frozenset({"A", "B"}), acceptable)
    assert not has_acceptable_evidence(frozenset({"A"}), acceptable)
    assert not has_acceptable_evidence(frozenset({"A", "B"}), ())  # no acceptable set: never
    assert has_acceptable_evidence(frozenset(), ((),))  # empty acceptable set: always


def test_evidence_superset_multi_sets() -> None:
    acceptable = (("A", "B"), ("A", "C"))
    assert has_acceptable_evidence(frozenset({"A", "C", "D"}), acceptable)
    assert has_acceptable_evidence(frozenset({"B", "A"}), acceptable)
    assert not has_acceptable_evidence(frozenset({"B", "C"}), acceptable)


def test_is_tp_all_five_criteria() -> None:
    issue = _issue()
    assert is_true_positive(_finding(), issue, public_ids=IDS)
    assert not is_true_positive(_finding(type="fx_error"), issue, public_ids=IDS)  # 1 type
    assert not is_true_positive(_finding(type="duplicate_invoices"), issue, public_ids=IDS)
    assert not is_true_positive(_finding(affected_ids=["INV-001"]), issue, public_ids=IDS)  # 2 ids
    padded = _finding(affected_ids=["INV-001", "INV-002", "INV-003"])
    assert not is_true_positive(padded, issue, public_ids=IDS)
    repeated = _finding(affected_ids=["INV-001", "INV-002", "INV-001"])
    assert is_true_positive(repeated, issue, public_ids=IDS)
    assert not is_true_positive(_finding(claimed_amount_minor=125001), issue, public_ids=IDS)  # 3
    assert not is_true_positive(_finding(claimed_amount_minor=124999), issue, public_ids=IDS)
    assert not is_true_positive(_finding(currency="EUR"), issue, public_ids=IDS)  # 4 currency
    assert not is_true_positive(_finding(currency="usd"), issue, public_ids=IDS)
    assert not is_true_positive(_finding(evidence_ids=["DOC-001"]), issue, public_ids=IDS)  # 5
    assert is_true_positive(_finding(evidence_ids=["DOC-002", "DOC-001"]), issue, public_ids=IDS)
    # an evidence id that does not exist in the episode disqualifies the finding (jury M04)
    ghost = _finding(evidence_ids=["DOC-001", "DOC-002", "DOC-999"])
    assert not is_true_positive(ghost, issue, public_ids=IDS)
    assert not is_true_positive(_finding(), _issue(acceptable_evidence_sets=()), public_ids=IDS)
    bare = _finding(evidence_ids=[])
    assert is_true_positive(bare, _issue(acceptable_evidence_sets=((),)), public_ids=IDS)


def test_match_simple_perfect() -> None:
    issues = tuple(
        _issue(
            issue_id=f"ISS-0{k}",
            affected_ids=(f"INV-00{k}",),
            acceptable_evidence_sets=((f"DOC-00{k}",),),
        )
        for k in (1, 2, 3)
    )
    findings = [
        _finding(affected_ids=[f"INV-00{k}"], evidence_ids=[f"DOC-00{k}"]) for k in (3, 1, 2)
    ]
    assert _match(findings, issues) == (3, 0, 0)
    assert build_compat_matrix(findings, issues, public_ids=IDS) == {0: [2], 1: [0], 2: [1]}
    assert _match([], issues) == (0, 0, 3)
    assert _match(findings, ()) == (0, 3, 0)
    assert _match([_finding(type="fx_error")], issues) == (0, 1, 3)


def test_match_duplicate_finding() -> None:
    issues = (_issue(),)
    assert _match([_finding(), _finding()], issues) == (1, 1, 0)
    assert _match([_finding(), _finding(), _finding()], issues) == (1, 2, 0)
    # a near-miss on evidence is a false positive and leaves the issue unmatched
    assert _match([_finding(evidence_ids=["DOC-001"])], issues) == (0, 1, 1)


def test_match_ambiguous_prefers_max_tp() -> None:
    # t1 and t2 differ only by their acceptable evidence; pred1 carries both sets, pred2 only
    # t1's. Greedy would give pred1 -> t1 and strand pred2 (1 TP); the augmenting path moves
    # pred1 to t2 and yields 2 TP.
    t1 = _issue(issue_id="ISS-01", acceptable_evidence_sets=(("DOC-001", "DOC-002"),))
    t2 = _issue(issue_id="ISS-02", acceptable_evidence_sets=(("DOC-003",),))
    pred1 = _finding(evidence_ids=["DOC-001", "DOC-002", "DOC-003"])
    pred2 = _finding(evidence_ids=["DOC-001", "DOC-002"])
    assert build_compat_matrix([pred1, pred2], (t1, t2), public_ids=IDS) == {0: [0, 1], 1: [0]}
    assert _match([pred1, pred2], (t1, t2)) == (2, 0, 0)
    assert _match([pred2, pred1], (t1, t2)) == (2, 0, 0)  # order of submission is irrelevant
    assert _match([pred1, pred2, pred2], (t1, t2)) == (2, 1, 0)
    # a three-link chain: f0 -> {i0, i1}, f1 -> {i1, i2}, f2 -> {i2}; greedy strands f2,
    # the augmenting path re-seats f1 and f0 -> 3 TP in every submission order
    i0, i1, i2 = (
        _issue(issue_id=f"ISS-0{k}", acceptable_evidence_sets=((f"DOC-00{k}",),)) for k in (1, 2, 3)
    )
    f0 = _finding(evidence_ids=["DOC-001", "DOC-002"])  # i0 and i1
    f1 = _finding(evidence_ids=["DOC-002", "DOC-003"])  # i1 and i2
    f2 = _finding(evidence_ids=["DOC-003"])  # i2 only
    assert build_compat_matrix([f0, f1, f2], (i0, i1, i2), public_ids=IDS) == {
        0: [0, 1],
        1: [1, 2],
        2: [2],
    }
    for order in ([f0, f1, f2], [f2, f1, f0], [f1, f2, f0], [f0, f2, f1]):
        assert _match(order, (i0, i1, i2)) == (3, 0, 0)
    # worst case at the caps: 30 identical findings against 12 identical-content issues
    many_issues = tuple(_issue(issue_id=f"ISS-{k:02d}") for k in range(1, 13))
    assert _match([_finding()] * 30, many_issues) == (12, 18, 0)
