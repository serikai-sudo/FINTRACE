"""One-to-one matching of submitted findings against truth issues (evaluator-only).

A finding is a true positive for an issue iff all five hold (the jury's matcher conventions):
the same type string, the same canonical affected ids (exact strings, set semantics), an amount
within the published per-type tolerance (``AMOUNT_TOLERANCE_MINOR``: 0 minor units for every
type, i.e. an exact match), the same currency string, and evidence ids that all exist in the
public episode and include at least one acceptable evidence set. Every comparison is exact: no
case folding, trimming, prefix stripping or fuzzy ids. Maximum bipartite matching (Kuhn's
augmenting paths) makes the counts independent of submission order and turns a duplicate finding
into a false positive instead of letting it double-claim one issue. Among several maximum
matchings, ``match_pairs`` picks the lexicographically smallest sorted sequence of
``(issue_id, finding index)`` pairs (issue ids ordered by Unicode code points, finding indices
counted from 0 in submission order), the jury's deterministic tie-break.

Evidence semantics for injectors: ``acceptable_evidence_sets == ()`` makes an issue unmatchable;
``((),)`` means "no evidence required" (every set is a superset of the empty set).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from fintrace.constants import AMOUNT_TOLERANCE_MINOR
from fintrace.models import Finding, Issue, PublicEpisode


def canonical_affected_ids(ids: Iterable[str]) -> tuple[str, ...]:
    """Deduplicated, lexicographically sorted tuple; the form Issue.affected_ids already has."""
    return tuple(sorted(set(ids)))


def public_ids_of(public: PublicEpisode) -> frozenset[str]:
    """Every identifier an agent can see or fetch in ``public``.

    Ledger row, journal, entity, invoice, payment and document ids, plus the synthetic FX ids
    ``<source>-<target>-<date>`` that ``get_record`` serves (the same format as
    ``tools.list_records.fx_id``). An evidence id outside this set does not exist for the matcher:
    the finding that cites it is never a true positive.
    """
    ids: set[str] = set()
    for row in public.ledger:
        ids.add(row.row_id)
        ids.add(row.journal_id)
        ids.update(ref for ref in (row.entity_id, row.invoice_id, row.payment_id) if ref)
    for invoice in public.invoices:
        ids.update((invoice.doc_id, invoice.invoice_id, invoice.entity_id))
    for payment in public.payments:
        ids.update((payment.doc_id, payment.payment_id))
        ids.update(ref.invoice_id for ref in payment.invoice_refs)
    ids.update(f"{rate.source_currency}-{rate.target_currency}-{rate.date}" for rate in public.fx)
    return frozenset(ids)


def has_acceptable_evidence(
    evidence_ids: frozenset[str], acceptable_sets: Sequence[Sequence[str]]
) -> bool:
    """True iff ``evidence_ids`` is a superset of at least one acceptable set."""
    return any(evidence_ids.issuperset(acceptable) for acceptable in acceptable_sets)


def tolerance_for(issue_type: str) -> int:
    """The published tolerance of one issue type in minor units (0 unless documented otherwise)."""
    return AMOUNT_TOLERANCE_MINOR.get(issue_type, 0)


def amount_within_tolerance(claimed: int, expected: int, tolerance_minor: int = 0) -> bool:
    """|claimed - expected| <= tolerance_minor, an integer number of minor units (0 = exact)."""
    if tolerance_minor < 0:
        raise ValueError("tolerance_minor must be a non-negative integer")
    return abs(claimed - expected) <= tolerance_minor


def is_true_positive(finding: Finding, issue: Issue, *, public_ids: frozenset[str]) -> bool:
    """All five criteria: type, canonical ids, amount, currency, existing acceptable evidence."""
    evidence = frozenset(finding.evidence_ids)
    return (
        finding.type == issue.type
        and canonical_affected_ids(finding.affected_ids) == issue.affected_ids
        and amount_within_tolerance(
            finding.claimed_amount_minor, issue.expected_amount_minor, tolerance_for(issue.type)
        )
        and finding.currency == issue.currency
        and evidence <= public_ids
        and has_acceptable_evidence(evidence, issue.acceptable_evidence_sets)
    )


def build_compat_matrix(
    findings: Sequence[Finding], issues: Sequence[Issue], *, public_ids: frozenset[str]
) -> dict[int, list[int]]:
    """{finding_index: [compatible issue indices, ascending]}."""
    return {
        i: [
            j
            for j, issue in enumerate(issues)
            if is_true_positive(finding, issue, public_ids=public_ids)
        ]
        for i, finding in enumerate(findings)
    }


def _max_cardinality(compat: Mapping[int, Sequence[int]]) -> int:
    """Size of a maximum matching of the bipartite graph ``compat`` (Kuhn's augmenting paths)."""
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

    for finding_index in sorted(compat):
        try_assign(finding_index, set())
    return len(owner)


def match(
    findings: Sequence[Finding], issues: Sequence[Issue], *, public_ids: frozenset[str]
) -> tuple[int, int, int]:
    """Maximum one-to-one matching; returns ``(tp, fp, fn)``.

    Each issue is owned by at most one finding. A finding that cannot be placed (duplicate of a
    matched one, or compatible with nothing) is a false positive; an issue nobody owns is a false
    negative. The counts do not depend on which maximum matching is chosen; ``match_pairs`` gives
    the jury's canonical one.
    """
    tp = _max_cardinality(build_compat_matrix(findings, issues, public_ids=public_ids))
    return tp, len(findings) - tp, len(issues) - tp


def match_pairs(
    findings: Sequence[Finding], issues: Sequence[Issue], *, public_ids: frozenset[str]
) -> tuple[tuple[str, int], ...]:
    """The canonical maximum matching as sorted ``(issue_id, finding_index)`` pairs.

    Jury tie-break: issue ids are taken in Unicode code-point order and each receives the
    smallest finding index that still leaves a maximum matching for the rest, which yields the
    lexicographically smallest sorted pair sequence among all maximum matchings. Reordering the
    submitted findings can change which identical finding carries a pair, never the counts.
    Assumes distinct ``issue_id`` values within one truth, which the generator guarantees.
    """
    compat = build_compat_matrix(findings, issues, public_ids=public_ids)
    target = _max_cardinality(compat)
    by_issue: dict[int, list[int]] = {j: [] for j in range(len(issues))}
    for finding_index in sorted(compat):
        for issue_index in compat[finding_index]:
            by_issue[issue_index].append(finding_index)
    order = sorted(range(len(issues)), key=lambda j: (issues[j].issue_id, j))
    fixed_issues: set[int] = set()
    fixed_findings: set[int] = set()
    pairs: list[tuple[str, int]] = []
    for issue_index in order:
        if len(pairs) == target:
            break
        for finding_index in by_issue[issue_index]:
            if finding_index in fixed_findings:
                continue
            rest = {
                i: [j for j in compat[i] if j != issue_index and j not in fixed_issues]
                for i in compat
                if i != finding_index and i not in fixed_findings
            }
            if len(pairs) + 1 + _max_cardinality(rest) == target:
                pairs.append((issues[issue_index].issue_id, finding_index))
                fixed_issues.add(issue_index)
                fixed_findings.add(finding_index)
                break
    return tuple(sorted(pairs))


__all__ = [
    "amount_within_tolerance",
    "build_compat_matrix",
    "canonical_affected_ids",
    "has_acceptable_evidence",
    "is_true_positive",
    "match",
    "match_pairs",
    "public_ids_of",
    "tolerance_for",
]
