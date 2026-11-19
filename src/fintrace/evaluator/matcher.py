"""One-to-one matching of submitted findings against truth issues (evaluator-only).

A finding is a true positive for an issue iff all five hold: same type, same canonical affected
ids, amount within tolerance, same currency, and evidence that is a superset of at least one
acceptable evidence set. Maximum bipartite matching (Kuhn's augmenting paths) makes the counts
independent of submission order and turns a duplicate finding into a false positive instead of
letting it double-claim one issue.

Evidence semantics for injectors: ``acceptable_evidence_sets == ()`` makes an issue unmatchable;
``((),)`` means "no evidence required" (every set is a superset of the empty set).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import ROUND_FLOOR, Decimal, localcontext
from typing import Final

from fintrace.constants import DEFAULT_TOLERANCE_FRAC, MIN_TOLERANCE_MINOR
from fintrace.models import Finding, Issue

_ONE: Final[Decimal] = Decimal(1)
_TOLERANCE_PRECISION: Final[int] = 60  # exact for any realistic minor-unit amount


def canonical_affected_ids(ids: Iterable[str]) -> tuple[str, ...]:
    """Deduplicated, lexicographically sorted tuple; the form Issue.affected_ids already has."""
    return tuple(sorted(set(ids)))


def has_acceptable_evidence(
    evidence_ids: frozenset[str], acceptable_sets: Sequence[Sequence[str]]
) -> bool:
    """True iff ``evidence_ids`` is a superset of at least one acceptable set."""
    return any(evidence_ids.issuperset(acceptable) for acceptable in acceptable_sets)


def amount_within_tolerance(
    claimed: int, expected: int, tolerance_frac: Decimal = DEFAULT_TOLERANCE_FRAC
) -> bool:
    """|claimed - expected| <= max(MIN_TOLERANCE_MINOR, floor(tolerance_frac * expected))."""
    with localcontext() as context:
        context.prec = _TOLERANCE_PRECISION
        scaled = (Decimal(expected) * tolerance_frac).quantize(_ONE, rounding=ROUND_FLOOR)
    tolerance = max(MIN_TOLERANCE_MINOR, int(scaled))
    return abs(claimed - expected) <= tolerance


def is_true_positive(
    finding: Finding, issue: Issue, tolerance_frac: Decimal = DEFAULT_TOLERANCE_FRAC
) -> bool:
    """All five criteria: type, canonical ids, amount tolerance, currency, acceptable evidence."""
    return (
        finding.type == issue.type
        and canonical_affected_ids(finding.affected_ids) == issue.affected_ids
        and amount_within_tolerance(
            finding.claimed_amount_minor, issue.expected_amount_minor, tolerance_frac
        )
        and finding.currency == issue.currency
        and has_acceptable_evidence(
            frozenset(finding.evidence_ids), issue.acceptable_evidence_sets
        )
    )


def build_compat_matrix(
    findings: Sequence[Finding],
    issues: Sequence[Issue],
    tolerance_frac: Decimal = DEFAULT_TOLERANCE_FRAC,
) -> dict[int, list[int]]:
    """{finding_index: [compatible issue indices, ascending]}."""
    return {
        i: [j for j, issue in enumerate(issues) if is_true_positive(finding, issue, tolerance_frac)]
        for i, finding in enumerate(findings)
    }


def match(
    findings: Sequence[Finding],
    issues: Sequence[Issue],
    tolerance_frac: Decimal = DEFAULT_TOLERANCE_FRAC,
) -> tuple[int, int, int]:
    """Maximum one-to-one matching via augmenting paths; returns ``(tp, fp, fn)``.

    Each issue is owned by at most one finding. A finding that cannot be placed (duplicate of a
    matched one, or compatible with nothing) is a false positive; an issue nobody owns is a false
    negative. Findings are tried in submission order with ascending issue indices, so the result
    is deterministic.
    """
    compat = build_compat_matrix(findings, issues, tolerance_frac)
    owner: dict[int, int] = {}  # issue index -> finding index

    def try_assign(finding_index: int, seen: set[int]) -> bool:
        for issue_index in compat[finding_index]:
            if issue_index in seen:
                continue
            seen.add(issue_index)
            if issue_index not in owner or try_assign(owner[issue_index], seen):
                owner[issue_index] = finding_index
                return True
        return False

    for finding_index in range(len(findings)):
        try_assign(finding_index, set())
    tp = len(owner)
    return tp, len(findings) - tp, len(issues) - tp


__all__ = [
    "amount_within_tolerance",
    "build_compat_matrix",
    "canonical_affected_ids",
    "has_acceptable_evidence",
    "is_true_positive",
    "match",
]
