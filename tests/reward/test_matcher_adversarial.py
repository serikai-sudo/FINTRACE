"""Adversarial matcher tests: the edge cases attack agents will probe.

Every test takes known findings and truths, runs ``match`` (and where it matters
``compute_metrics`` + ``official_reward``) and pins the exact expected behaviour. A failure here
is a reward exploit in production.
"""

from __future__ import annotations

import random
import time

import pytest
from pydantic import ValidationError

from fintrace.evaluator.matcher import amount_within_tolerance, build_compat_matrix, match
from fintrace.evaluator.metrics import compute_metrics
from fintrace.models import Finding, Issue, IssueType
from fintrace.reward.official import official_reward

EvidenceSets = tuple[tuple[str, ...], ...]
DUP: IssueType = "duplicate_invoice"
AB = ("INV-A", "INV-B")
DOCS_AB: EvidenceSets = (("DOC-A", "DOC-B"),)


def _issue(
    issue_id: str,
    issue_type: IssueType,
    affected: tuple[str, ...],
    amount: int,
    currency: str,
    sets: EvidenceSets,
) -> Issue:
    return Issue(
        issue_id=issue_id,
        type=issue_type,
        affected_ids=tuple(sorted(affected)),
        expected_amount_minor=amount,
        currency=currency,
        acceptable_evidence_sets=sets,
    )


def _finding(
    issue_type: IssueType,
    affected: list[str],
    amount: int,
    currency: str,
    evidence: list[str],
    explanation: str = "",
) -> Finding:
    return Finding(
        type=issue_type,
        affected_ids=list(affected),
        claimed_amount_minor=amount,
        currency=currency,
        evidence_ids=list(evidence),
        explanation=explanation,
    )


# ------------------------------------------------------------------ A.1 ambiguous matching


def test_two_findings_one_truth_only_one_matches() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    first = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    second = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([first, second], (truth,)) == (1, 1, 0)


def test_two_truths_one_finding_matches_at_most_one() -> None:
    first = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    second = _issue("ISS-02", DUP, AB, 1000, "USD", DOCS_AB)  # pathological twin truths
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([finding], (first, second)) == (1, 0, 1)


def test_augmenting_path_three_link_chain() -> None:
    """A real ambiguity through the amount tolerance: greedy gives 2 TP, the matcher must give 3.

    t1 expects 100000 and t2 expects 100050 on the same documents; f1 claims 100040 and fits
    both (40 and 10 within the 100-minor tolerance), f2 claims 99901 and fits t1 only (99 vs
    149), f3 belongs to an unrelated t3. Tried in order, f1 grabs t1 and f2 would be stranded
    unless f1 is moved over to t2.
    """
    t1 = _issue("ISS-01", DUP, AB, 100_000, "USD", DOCS_AB)
    t2 = _issue("ISS-02", DUP, AB, 100_050, "USD", DOCS_AB)
    t3 = _issue("ISS-03", "fx_error", ("INV-E", "PAY-E"), 500, "USD", (("DOC-E", "DOC-PE"),))
    f1 = _finding(DUP, [*AB], 100_040, "USD", ["DOC-A", "DOC-B"])
    f2 = _finding(DUP, [*AB], 99_901, "USD", ["DOC-A", "DOC-B"])
    f3 = _finding("fx_error", ["INV-E", "PAY-E"], 500, "USD", ["DOC-E", "DOC-PE"])
    assert match([f1, f2, f3], (t1, t2, t3)) == (3, 0, 0)
    assert match([f3, f2, f1], (t1, t2, t3)) == (3, 0, 0)  # order of submission is irrelevant
    assert match([f1, f2], (t1, t2, t3)) == (2, 0, 1)
    assert match([f2, f2], (t1, t2, t3)) == (1, 1, 2)  # f2 fits only t1: the copy is a FP


def test_type_confusion_on_shared_ids_claims_one_issue_only() -> None:
    """Two issues of different types on the same ids: a finding claims only its own type."""
    dup_truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    fx_truth = _issue("ISS-02", "fx_error", AB, 1000, "USD", DOCS_AB)
    dup_finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    fx_finding = _finding("fx_error", [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([dup_finding], (dup_truth, fx_truth)) == (1, 0, 1)
    assert match([dup_finding, dup_finding], (dup_truth, fx_truth)) == (1, 1, 1)
    assert match([dup_finding, fx_finding], (dup_truth, fx_truth)) == (2, 0, 0)
    relabelled = _finding("payment_allocation", [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([relabelled], (dup_truth, fx_truth)) == (0, 1, 2)  # a third type claims none


def _max_matching_brute_force(compat: dict[int, list[int]], n_findings: int) -> int:
    """Exhaustive maximum bipartite matching (tiny instances only), the oracle for Kuhn."""

    def best(index: int, taken: frozenset[int]) -> int:
        if index == n_findings:
            return 0
        skip = best(index + 1, taken)
        options = [
            1 + best(index + 1, taken | {issue}) for issue in compat[index] if issue not in taken
        ]
        return max([skip, *options])

    return best(0, frozenset())


def test_matching_is_maximum_and_order_independent_over_random_shuffles() -> None:
    """Overlapping tolerance windows: Kuhn equals the brute-force optimum in every order."""
    truths = tuple(
        _issue(f"ISS-{k:02d}", DUP, AB, 100_000 + 60 * k, "USD", DOCS_AB) for k in range(6)
    )  # tolerance 100 each: neighbouring windows overlap by 40 minor units
    claims = [100_030, 100_090, 100_150, 100_210, 100_270, 100_330, 100_040, 99_850]
    findings = [_finding(DUP, [*AB], claim, "USD", ["DOC-A", "DOC-B"]) for claim in claims]
    compat = build_compat_matrix(findings, truths)
    assert any(len(options) >= 2 for options in compat.values())  # the instance is ambiguous
    optimum = _max_matching_brute_force(compat, len(findings))
    assert 1 <= optimum <= 6
    rng = random.Random(0)
    for _ in range(50):
        shuffled = findings[:]
        rng.shuffle(shuffled)
        tp, fp, fn = match(shuffled, truths)
        assert tp == optimum and fp == len(findings) - tp and fn == len(truths) - tp


def test_30x12_massive_match_bounded_time() -> None:
    truths = tuple(
        _issue(f"ISS-{i:02d}", DUP, (f"INV-A{i}", f"INV-B{i}"), 1000 + i, "USD",
               ((f"DOC-A{i}", f"DOC-B{i}"),))
        for i in range(12)
    )  # fmt: skip
    findings = [
        _finding(DUP, [f"INV-A{i}", f"INV-B{i}"], 1000 + i, "USD", [f"DOC-A{i}", f"DOC-B{i}"])
        for i in range(12)
    ]
    findings += [
        _finding(DUP, [f"INV-X{i}", f"INV-Y{i}"], 100, "USD", [f"DOC-X{i}", f"DOC-Y{i}"])
        for i in range(18)
    ]
    result = match(findings, truths)  # warm-up
    assert result == (12, 18, 0)
    assert _ms_per_call(findings, truths) < 10.0
    # dense worst case: every one of 30 findings is compatible with all 12 twin truths
    twins = tuple(_issue(f"ISS-{i:02d}", DUP, AB, 1000, "USD", DOCS_AB) for i in range(12))
    flood = [_finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])] * 30
    assert match(flood, twins) == (12, 18, 0)
    assert _ms_per_call(flood, twins) < 10.0


def _ms_per_call(findings: list[Finding], truths: tuple[Issue, ...]) -> float:
    """Best of three batches of 100 calls, in milliseconds per call (robust to CI jitter)."""
    batches = []
    for _ in range(3):
        started = time.monotonic()
        for _ in range(100):
            match(findings, truths)
        batches.append((time.monotonic() - started) * 10)
    return min(batches)


# ------------------------------------------------------------------ A.2 duplicate findings


def test_identical_duplicate_findings_only_one_tp() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([finding, finding], (truth,)) == (1, 1, 0)


def test_near_duplicate_with_different_explanations_second_is_fp() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    first = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"], explanation="first")
    second = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"], explanation="second")
    assert match([first, second], (truth,)) == (1, 1, 0)


def test_30_duplicate_findings_against_one_truth() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    tp, fp, fn = match([finding] * 30, (truth,))
    assert (tp, fp, fn) == (1, 29, 0)
    _, _, f1 = compute_metrics(tp, fp, fn)
    assert abs(f1 - 2 / 31) < 1e-12
    assert official_reward(f1, 50) < -0.5  # a duplicate flood is heavily penalised


# ------------------------------------------------------------------ A.3 canonical normalisation


def test_affected_ids_order_irrelevant() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, ["INV-B", "INV-A"], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([finding], (truth,)) == (1, 0, 0)


def test_duplicate_affected_ids_dedup() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, ["INV-A", "INV-A", "INV-B"], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([finding], (truth,)) == (1, 0, 0)
    extra = _finding(DUP, ["INV-A", "INV-B", "INV-C"], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([extra], (truth,)) == (0, 1, 1)  # a padded id set is not the canonical one


def test_currency_lowercase_rejected_and_other_currency_is_fp() -> None:
    """Lower-case currency never reaches the matcher: the contract model rejects it."""
    with pytest.raises(ValidationError):
        _finding(DUP, [*AB], 1000, "usd", ["DOC-A", "DOC-B"])
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    other = _finding(DUP, [*AB], 1000, "EUR", ["DOC-A", "DOC-B"])
    assert match([other], (truth,)) == (0, 1, 1)  # a valid but different currency is a FP


# ------------------------------------------------------------------ A.4 tolerance boundary


def test_amount_tolerance_exact_boundary() -> None:
    """expected 100000 -> tolerance max(1, floor(0.001 * 100000)) = 100, inclusive."""
    assert amount_within_tolerance(100_100, 100_000) is True
    assert amount_within_tolerance(100_101, 100_000) is False
    assert amount_within_tolerance(99_900, 100_000) is True
    assert amount_within_tolerance(99_899, 100_000) is False
    truth = _issue("ISS-01", DUP, AB, 100_000, "USD", DOCS_AB)
    inside = _finding(DUP, [*AB], 100_100, "USD", ["DOC-A", "DOC-B"])
    outside = _finding(DUP, [*AB], 100_101, "USD", ["DOC-A", "DOC-B"])
    assert match([inside], (truth,)) == (1, 0, 0)
    assert match([outside], (truth,)) == (0, 1, 1)


def test_amount_tolerance_small_amounts_min_floor() -> None:
    """expected 10 -> floor(0.01) = 0, lifted to the 1-minor floor."""
    assert amount_within_tolerance(10, 10) is True
    assert amount_within_tolerance(11, 10) is True
    assert amount_within_tolerance(9, 10) is True
    assert amount_within_tolerance(12, 10) is False
    assert amount_within_tolerance(8, 10) is False
    assert amount_within_tolerance(0, 1) is True and amount_within_tolerance(2, 1) is True
    assert amount_within_tolerance(3, 1) is False


# ------------------------------------------------------------------ A.5 evidence edges


def test_evidence_empty_acceptable_set_impossible() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", ())
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([finding], (truth,)) == (0, 1, 1)
    anything_goes = _issue("ISS-02", DUP, AB, 1000, "USD", ((),))  # no evidence required
    bare = _finding(DUP, [*AB], 1000, "USD", [])
    assert match([bare], (anything_goes,)) == (1, 0, 0)


def test_evidence_duplicate_ids_dedup() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-A", "DOC-B"])
    assert match([finding], (truth,)) == (1, 0, 0)
    partial = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-A", "DOC-A"])
    assert match([partial], (truth,)) == (0, 1, 1)  # repeating one id never completes a set


def test_evidence_multi_sets_one_satisfied_is_enough() -> None:
    acceptable: EvidenceSets = (
        ("DOC-A", "DOC-B", "L-999"),  # needs an extra ledger row
        ("DOC-A", "DOC-B"),  # the simpler set
        ("DOC-X", "DOC-Y"),  # unrelated
    )
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", acceptable)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert match([finding], (truth,)) == (1, 0, 0)
    noisy = _finding(DUP, [*AB], 1000, "USD", ["DOC-Q", "DOC-A", "DOC-Z", "DOC-B", "L-1"])
    assert match([noisy], (truth,)) == (1, 0, 0)  # a superset with noise still qualifies
    mixed = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-X"])
    assert match([mixed], (truth,)) == (0, 1, 1)  # half of two sets is no set
