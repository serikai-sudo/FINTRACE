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
# the public id universe of these fixtures: every id a finding may cite and still be matched
IDS = frozenset(
    {"INV-A", "INV-B", "INV-C", "INV-E", "PAY-E", "DOC-A", "DOC-B", "DOC-E", "DOC-PE", "DOC-X",
     "DOC-Y", "DOC-Q", "DOC-Z", "L-1", "L-999"}
    | {f"DOC-{k}" for k in range(10)}
    | {
        f"{fam}{i}"
        for i in range(18)
        for fam in ("INV-A", "INV-B", "DOC-A", "DOC-B", "INV-X", "INV-Y", "DOC-X", "DOC-Y")
    }
)  # fmt: skip


def _match(findings: list[Finding], truths: tuple[Issue, ...]) -> tuple[int, int, int]:
    return match(findings, truths, public_ids=IDS)


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
    assert _match([first, second], (truth,)) == (1, 1, 0)


def test_two_truths_one_finding_matches_at_most_one() -> None:
    first = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    second = _issue("ISS-02", DUP, AB, 1000, "USD", DOCS_AB)  # pathological twin truths
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([finding], (first, second)) == (1, 0, 1)


def test_augmenting_path_three_link_chain() -> None:
    """A real ambiguity through the evidence sets: greedy gives 2 TP, the matcher must give 3.

    t1 accepts {DOC-A, DOC-B} and t2 accepts {DOC-A, DOC-B, DOC-X} on the same ids and amount;
    f1 cites all three documents and fits both, f2 cites two and fits t1 only, f3 belongs to an
    unrelated t3. Tried in order, f1 grabs t1 and f2 would be stranded unless f1 is moved to t2.
    """
    t1 = _issue("ISS-01", DUP, AB, 100_000, "USD", DOCS_AB)
    t2 = _issue("ISS-02", DUP, AB, 100_000, "USD", (("DOC-A", "DOC-B", "DOC-X"),))
    t3 = _issue("ISS-03", "fx_error", ("INV-E", "PAY-E"), 500, "USD", (("DOC-E", "DOC-PE"),))
    f1 = _finding(DUP, [*AB], 100_000, "USD", ["DOC-A", "DOC-B", "DOC-X"])
    f2 = _finding(DUP, [*AB], 100_000, "USD", ["DOC-A", "DOC-B"])
    f3 = _finding("fx_error", ["INV-E", "PAY-E"], 500, "USD", ["DOC-E", "DOC-PE"])
    assert _match([f1, f2, f3], (t1, t2, t3)) == (3, 0, 0)
    assert _match([f3, f2, f1], (t1, t2, t3)) == (3, 0, 0)  # order of submission is irrelevant
    assert _match([f1, f2], (t1, t2, t3)) == (2, 0, 1)
    assert _match([f2, f2], (t1, t2, t3)) == (1, 1, 2)  # f2 fits only t1: the copy is a FP


def test_type_confusion_on_shared_ids_claims_one_issue_only() -> None:
    """Two issues of different types on the same ids: a finding claims only its own type."""
    dup_truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    fx_truth = _issue("ISS-02", "fx_error", AB, 1000, "USD", DOCS_AB)
    dup_finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    fx_finding = _finding("fx_error", [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([dup_finding], (dup_truth, fx_truth)) == (1, 0, 1)
    assert _match([dup_finding, dup_finding], (dup_truth, fx_truth)) == (1, 1, 1)
    assert _match([dup_finding, fx_finding], (dup_truth, fx_truth)) == (2, 0, 0)
    relabelled = _finding("payment_allocation", [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([relabelled], (dup_truth, fx_truth)) == (0, 1, 2)  # a third type claims none


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
    """Overlapping evidence windows: Kuhn equals the brute-force optimum in every order."""
    truths = tuple(
        _issue(f"ISS-{k:02d}", DUP, AB, 100_000, "USD", ((f"DOC-{k}", f"DOC-{k + 1}"),))
        for k in range(6)
    )  # truth k accepts {DOC-k, DOC-k+1}: a finding citing three consecutive documents fits two
    windows = [[f"DOC-{k}", f"DOC-{k + 1}", f"DOC-{k + 2}"] for k in range(6)]
    windows += [["DOC-0", "DOC-1"], ["DOC-9"]]  # truth 0 only; nobody
    findings = [_finding(DUP, [*AB], 100_000, "USD", docs) for docs in windows]
    compat = build_compat_matrix(findings, truths, public_ids=IDS)
    assert any(len(options) >= 2 for options in compat.values())  # the instance is ambiguous
    optimum = _max_matching_brute_force(compat, len(findings))
    assert 1 <= optimum <= 6
    rng = random.Random(0)
    for _ in range(50):
        shuffled = findings[:]
        rng.shuffle(shuffled)
        tp, fp, fn = _match(shuffled, truths)
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
    result = _match(findings, truths)  # warm-up
    assert result == (12, 18, 0)
    assert _ms_per_call(findings, truths) < 10.0
    # dense worst case: every one of 30 findings is compatible with all 12 twin truths
    twins = tuple(_issue(f"ISS-{i:02d}", DUP, AB, 1000, "USD", DOCS_AB) for i in range(12))
    flood = [_finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])] * 30
    assert _match(flood, twins) == (12, 18, 0)
    assert _ms_per_call(flood, twins) < 10.0


def _ms_per_call(findings: list[Finding], truths: tuple[Issue, ...]) -> float:
    """Best of three batches of 100 calls, in milliseconds per call (robust to CI jitter)."""
    batches = []
    for _ in range(3):
        started = time.monotonic()
        for _ in range(100):
            _match(findings, truths)
        batches.append((time.monotonic() - started) * 10)
    return min(batches)


# ------------------------------------------------------------------ A.2 duplicate findings


def test_identical_duplicate_findings_only_one_tp() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([finding, finding], (truth,)) == (1, 1, 0)


def test_near_duplicate_with_different_explanations_second_is_fp() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    first = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"], explanation="first")
    second = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"], explanation="second")
    assert _match([first, second], (truth,)) == (1, 1, 0)


def test_30_duplicate_findings_against_one_truth() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    tp, fp, fn = _match([finding] * 30, (truth,))
    assert (tp, fp, fn) == (1, 29, 0)
    _, _, f1 = compute_metrics(tp, fp, fn)
    assert abs(f1 - 2 / 31) < 1e-12
    assert official_reward(f1, 50) < -0.5  # a duplicate flood is heavily penalised


# ------------------------------------------------------------------ A.3 canonical normalisation


def test_affected_ids_order_irrelevant() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, ["INV-B", "INV-A"], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([finding], (truth,)) == (1, 0, 0)


def test_duplicate_affected_ids_dedup() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, ["INV-A", "INV-A", "INV-B"], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([finding], (truth,)) == (1, 0, 0)
    extra = _finding(DUP, ["INV-A", "INV-B", "INV-C"], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([extra], (truth,)) == (0, 1, 1)  # a padded id set is not the canonical one


def test_currency_case_or_space_is_one_fp_not_structural() -> None:
    """Jury §5 / M06: a currency that differs by case or whitespace is compared exactly and fails
    as an ordinary false positive; only a non-string currency is a structural error."""
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    for currency in ("usd", " USD", "USD ", "EUR"):
        other = _finding(DUP, [*AB], 1000, currency, ["DOC-A", "DOC-B"])
        assert _match([other], (truth,)) == (0, 1, 1), currency
    correct = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    lower = _finding(DUP, [*AB], 1000, "usd", ["DOC-A", "DOC-B"])
    assert _match([correct, lower], (truth,)) == (1, 1, 0)  # the good finding still counts
    with pytest.raises(ValidationError):
        _finding(DUP, [*AB], 1000, 840, ["DOC-A", "DOC-B"])  # type: ignore[arg-type]


# ------------------------------------------------------------------ A.4 tolerance boundary


def test_amount_tolerance_exact_boundary() -> None:
    """Jury default tau = 0 (test 8 / M07): one minor unit off is a miss; a published integer tau
    is inclusive (M08)."""
    assert amount_within_tolerance(100_000, 100_000) is True
    assert amount_within_tolerance(100_001, 100_000) is False
    assert amount_within_tolerance(99_999, 100_000) is False
    assert amount_within_tolerance(100_100, 100_000, 100) is True  # only with a published tau
    truth = _issue("ISS-01", DUP, AB, 100_000, "USD", DOCS_AB)
    exact = _finding(DUP, [*AB], 100_000, "USD", ["DOC-A", "DOC-B"])
    above = _finding(DUP, [*AB], 100_001, "USD", ["DOC-A", "DOC-B"])
    below = _finding(DUP, [*AB], 99_999, "USD", ["DOC-A", "DOC-B"])
    assert _match([exact], (truth,)) == (1, 0, 0)
    assert _match([above], (truth,)) == (0, 1, 1)
    assert _match([below], (truth,)) == (0, 1, 1)


def test_amount_tolerance_small_amounts_no_floor() -> None:
    """There is no 1-unit floor any more: 10 matches 10 only."""
    assert amount_within_tolerance(10, 10) is True
    assert amount_within_tolerance(11, 10) is False
    assert amount_within_tolerance(9, 10) is False
    assert amount_within_tolerance(1, 1) is True
    assert amount_within_tolerance(0, 1) is False and amount_within_tolerance(2, 1) is False


# ------------------------------------------------------------------ A.5 evidence edges


def test_evidence_empty_acceptable_set_impossible() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", ())
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([finding], (truth,)) == (0, 1, 1)
    anything_goes = _issue("ISS-02", DUP, AB, 1000, "USD", ((),))  # no evidence required
    bare = _finding(DUP, [*AB], 1000, "USD", [])
    assert _match([bare], (anything_goes,)) == (1, 0, 0)


def test_evidence_duplicate_ids_dedup() -> None:
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", DOCS_AB)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-A", "DOC-B"])
    assert _match([finding], (truth,)) == (1, 0, 0)
    partial = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-A", "DOC-A"])
    assert _match([partial], (truth,)) == (0, 1, 1)  # repeating one id never completes a set


def test_evidence_multi_sets_one_satisfied_is_enough() -> None:
    acceptable: EvidenceSets = (
        ("DOC-A", "DOC-B", "L-999"),  # needs an extra ledger row
        ("DOC-A", "DOC-B"),  # the simpler set
        ("DOC-X", "DOC-Y"),  # unrelated
    )
    truth = _issue("ISS-01", DUP, AB, 1000, "USD", acceptable)
    finding = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B"])
    assert _match([finding], (truth,)) == (1, 0, 0)
    noisy = _finding(DUP, [*AB], 1000, "USD", ["DOC-Q", "DOC-A", "DOC-Z", "DOC-B", "L-1"])
    assert _match([noisy], (truth,)) == (1, 0, 0)  # existing extra ids still qualify (M03)
    ghost = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-B", "DOC-NOPE"])
    assert _match([ghost], (truth,)) == (0, 1, 1)  # a non-existent evidence id disqualifies (M04)
    mixed = _finding(DUP, [*AB], 1000, "USD", ["DOC-A", "DOC-X"])
    assert _match([mixed], (truth,)) == (0, 1, 1)  # half of two sets is no set
