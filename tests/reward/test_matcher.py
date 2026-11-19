"""Matcher: normalisation, tolerance, evidence, the five TP criteria, maximum matching."""

from __future__ import annotations

from decimal import Decimal

from fintrace.evaluator.matcher import (
    amount_within_tolerance,
    build_compat_matrix,
    canonical_affected_ids,
    has_acceptable_evidence,
    is_true_positive,
    match,
)
from fintrace.models import Finding, Issue


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


def test_canonical_affected_ids_dedup_and_sort() -> None:
    assert canonical_affected_ids(["B", "A", "A", "B"]) == ("A", "B")
    assert canonical_affected_ids(("INV-010", "INV-002")) == ("INV-002", "INV-010")
    assert canonical_affected_ids(iter([])) == ()
    assert canonical_affected_ids(_finding().affected_ids) == _issue().affected_ids


def test_amount_within_tolerance_exact() -> None:
    assert amount_within_tolerance(1000, 1000)
    assert amount_within_tolerance(0, 0)


def test_amount_within_tolerance_boundary() -> None:
    frac = Decimal("0.001")  # expected 100000 -> tolerance 100 minor units
    assert amount_within_tolerance(99900, 100000, frac)
    assert amount_within_tolerance(100100, 100000, frac)
    assert not amount_within_tolerance(99899, 100000, frac)
    assert not amount_within_tolerance(100101, 100000, frac)
    # floor, not round: expected 1999 -> 1.999 -> 1
    assert amount_within_tolerance(1998, 1999, frac)
    assert not amount_within_tolerance(1997, 1999, frac)


def test_amount_tolerance_min_floor() -> None:
    assert amount_within_tolerance(11, 10)  # max(1, floor(0.01)) == 1
    assert amount_within_tolerance(9, 10)
    assert not amount_within_tolerance(12, 10)
    assert amount_within_tolerance(1, 0) and not amount_within_tolerance(2, 0)


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
    assert is_true_positive(_finding(), issue)
    assert not is_true_positive(_finding(type="fx_error"), issue)  # 1 type
    assert not is_true_positive(_finding(affected_ids=["INV-001"]), issue)  # 2 ids: missing
    assert not is_true_positive(_finding(affected_ids=["INV-001", "INV-002", "INV-003"]), issue)
    assert is_true_positive(_finding(affected_ids=["INV-001", "INV-002", "INV-001"]), issue)
    assert not is_true_positive(_finding(claimed_amount_minor=125126), issue)  # 3 amount: 126 > 125
    assert is_true_positive(_finding(claimed_amount_minor=125125), issue)
    assert not is_true_positive(_finding(currency="EUR"), issue)  # 4 currency
    assert not is_true_positive(_finding(evidence_ids=["DOC-001"]), issue)  # 5 evidence
    assert is_true_positive(_finding(evidence_ids=["DOC-002", "DOC-001"]), issue)
    assert not is_true_positive(_finding(), _issue(acceptable_evidence_sets=()))
    assert is_true_positive(_finding(evidence_ids=[]), _issue(acceptable_evidence_sets=((),)))


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
    assert match(findings, issues) == (3, 0, 0)
    assert build_compat_matrix(findings, issues) == {0: [2], 1: [0], 2: [1]}
    assert match([], issues) == (0, 0, 3)
    assert match(findings, ()) == (0, 3, 0)
    assert match([_finding(type="fx_error")], issues) == (0, 1, 3)


def test_match_duplicate_finding() -> None:
    issues = (_issue(),)
    assert match([_finding(), _finding()], issues) == (1, 1, 0)
    assert match([_finding(), _finding(), _finding()], issues) == (1, 2, 0)
    # a near-miss on evidence is a false positive and leaves the issue unmatched
    assert match([_finding(evidence_ids=["DOC-001"])], issues) == (0, 1, 1)


def test_match_ambiguous_prefers_max_tp() -> None:
    # t1 and t2 differ only by amount; pred1 (100050) lies within tolerance of both,
    # pred2 (99950) only of t1. Greedy would give pred1 -> t1 and strand pred2 (1 TP);
    # the augmenting path moves pred1 to t2 and yields 2 TP.
    t1 = _issue(issue_id="ISS-01", expected_amount_minor=100000)
    t2 = _issue(issue_id="ISS-02", expected_amount_minor=100100)
    pred1 = _finding(claimed_amount_minor=100050)
    pred2 = _finding(claimed_amount_minor=99950)
    assert build_compat_matrix([pred1, pred2], (t1, t2)) == {0: [0, 1], 1: [0]}
    assert match([pred1, pred2], (t1, t2)) == (2, 0, 0)
    assert match([pred2, pred1], (t1, t2)) == (2, 0, 0)  # order of submission is irrelevant
    assert match([pred1, pred2, pred2], (t1, t2)) == (2, 1, 0)
    # a three-link chain: f0 -> {i0, i1}, f1 -> {i1, i2}, f2 -> {i2}; greedy strands f2,
    # the augmenting path re-seats f1 and f0 -> 3 TP in every submission order
    i0, i1, i2 = (
        _issue(issue_id=f"ISS-0{k}", expected_amount_minor=100000 + 100 * k) for k in (1, 2, 3)
    )
    f0 = _finding(claimed_amount_minor=100050)  # within 100 of i0 (100100) and i1 (100200)
    f1 = _finding(claimed_amount_minor=100250)  # within 100 of i1 and i2 (100300)
    f2 = _finding(claimed_amount_minor=100350)  # i2 only
    assert build_compat_matrix([f0, f1, f2], (i0, i1, i2)) == {0: [0, 1], 1: [1, 2], 2: [2]}
    for order in ([f0, f1, f2], [f2, f1, f0], [f1, f2, f0], [f0, f2, f1]):
        assert match(order, (i0, i1, i2)) == (3, 0, 0)
    # worst case at the caps: 30 identical findings against 12 identical-content issues
    many_issues = tuple(_issue(issue_id=f"ISS-{k:02d}") for k in range(1, 13))
    assert match([_finding()] * 30, many_issues) == (12, 18, 0)
