"""The truth-blind audit: what it catches that the injectors' own checks cannot.

Every injector ships an ``independent_check``, but it is handed the issue it is meant to confirm,
looks only where ``affected_ids`` points and compares against ``expected_amount_minor`` — so a
discrepancy no injector recorded passes it unseen, and an arithmetic bug shared with the injector
cancels out. ``audit.verify_declared`` rescans the clean books and the injected books with its own
arithmetic and requires the difference to be exactly what truth declares, in both directions.
These tests fix both directions, the one import rule that makes "truth-blind" more than a
docstring, and the hazard that direction found: a misallocation that also moves a payment's FX
policy date would carry a second, undeclared conversion gap on the same records.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fintrace.evaluator.runner import load_split_config
from fintrace.exporters.episode_files import read_episode
from fintrace.generator.audit import AuditError, Detected, scan, verify_declared
from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.fx import build_fx_index, convert_minor, policy_date, rate_on
from fintrace.generator.injectors import duplicate, fx_error, payment, period
from fintrace.generator.injectors.base import INJECTOR_ORDER, InjectorError
from fintrace.generator.injectors.fx_error import noise_floor
from fintrace.generator.injectors.payment import _moves_fx_policy_date
from fintrace.models import (
    Invoice,
    Issue,
    IssueType,
    Payment,
    PrivateEpisode,
    PublicEpisode,
    Truth,
)

INJECTORS: dict[IssueType, object] = {
    "duplicate_invoice": duplicate,
    "payment_allocation": payment,
    "fx_error": fx_error,
    "period_misclassification": period,
}

SEED = 10000
AUDIT_SOURCE = Path("src/fintrace/generator/audit.py")
DATA_DIR = Path("data")


@pytest.fixture(scope="module")
def episode() -> tuple[PublicEpisode, PrivateEpisode]:
    """(clean baseline, injected episode) of one dev seed; the two states the audit compares."""
    config = load_split_config("dev")
    return build_clean_episode(SEED, config).public, build_episode(SEED, config)


def _declared(truth: Truth) -> set[tuple[str, tuple[str, ...]]]:
    return {(issue.type, issue.affected_ids) for issue in truth.issues}


# ---------------------------------------------------------------- truth-blind by construction


def test_audit_never_imports_the_generator_it_audits() -> None:
    """The one rule behind the word: models and the standard library, nothing of the generator.

    An import of ``fintrace.generator.fx`` would make the conversion check agree with the injector
    by sharing its arithmetic, which is the defect this module exists to close.
    """
    tree = ast.parse(AUDIT_SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    fintrace = [name for name in imported if name.startswith("fintrace")]
    assert fintrace == ["fintrace.models"], fintrace
    assert not any(name.startswith("fintrace.generator") for name in imported)
    assert not any(node.level for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))


def test_scan_takes_the_public_episode_and_nothing_else() -> None:
    """No truth, no provenance, no private episode in the signature: the scan cannot peek."""
    annotations = scan.__annotations__
    assert annotations == {"public": "PublicEpisode", "return": "tuple[Detected, ...]"}


def test_clean_books_carry_no_discrepancy(episode: tuple[PublicEpisode, PrivateEpisode]) -> None:
    baseline, _ = episode
    assert scan(baseline) == ()


# ---------------------------------------------------------------- the committed dataset


def test_every_committed_episode_scans_to_exactly_its_truth() -> None:
    """The shipped data, re-derived without reading truth first: same issues, same amounts.

    This is the claim the old per-issue check could not make — it confirmed the issues truth named
    and nothing else.
    """
    splits = [split for split in sorted(DATA_DIR.iterdir()) if split.is_dir()]
    episode_dirs = [ep for split in splits for ep in sorted(split.iterdir()) if ep.is_dir()]
    assert len(episode_dirs) == 20
    for ep_dir in episode_dirs:
        private = read_episode(ep_dir)
        found = {item.key: item for item in scan(private.public)}
        declared = {
            (issue.type, issue.affected_ids): Detected(
                issue.type, issue.affected_ids, issue.expected_amount_minor, issue.currency
            )
            for issue in private.truth.issues
        }
        assert found == declared, ep_dir.as_posix()


# ---------------------------------------------------------------- both directions of the gate


def test_the_gate_passes_the_episode_the_generator_built(
    episode: tuple[PublicEpisode, PrivateEpisode],
) -> None:
    baseline, private = episode
    assert len(private.truth.issues) >= len(INJECTOR_ORDER)
    verify_declared(baseline, private.public, private.truth)


def test_a_discrepancy_truth_does_not_declare_fails_the_gate(
    episode: tuple[PublicEpisode, PrivateEpisode],
) -> None:
    """The case the injectors' own checks cannot see: drop an issue and keep the books.

    Every remaining issue still passes its own ``independent_check`` — nothing points at the one
    that went missing — so without this gate the episode would ship with an unlabelled error.
    """
    baseline, private = episode
    dropped, *kept = private.truth.issues
    thinner = private.truth.model_copy(update={"issues": tuple(kept)})
    for issue in kept:  # the second opinion is blind to the gap, by construction
        INJECTORS[issue.type].independent_check(private.public, issue)
    with pytest.raises(AuditError, match="present in the books but not declared") as raised:
        verify_declared(baseline, private.public, thinner)
    assert dropped.type in str(raised.value)
    assert _declared(thinner) < _declared(private.truth)


def test_an_issue_the_books_do_not_show_fails_the_gate(
    episode: tuple[PublicEpisode, PrivateEpisode],
) -> None:
    """A truth record with no counterpart in the books: an injection that did not land."""
    baseline, private = episode
    invented = Issue(
        issue_id="ISS-9999",
        type="duplicate_invoice",
        affected_ids=("INV-0001", "INV-0002"),
        expected_amount_minor=123456,
        currency=private.public.manifest.base_currency,
        acceptable_evidence_sets=(("INV-0001",),),
    )
    fatter = private.truth.model_copy(update={"issues": (*private.truth.issues, invented)})
    with pytest.raises(AuditError, match="declared but not present in the books"):
        verify_declared(baseline, private.public, fatter)


def test_a_wrong_amount_in_truth_fails_the_gate(
    episode: tuple[PublicEpisode, PrivateEpisode],
) -> None:
    """The arithmetic direction: the records match, the number does not."""
    baseline, private = episode
    first, *rest = private.truth.issues
    bent = first.model_copy(update={"expected_amount_minor": first.expected_amount_minor + 1})
    truth = private.truth.model_copy(update={"issues": (bent, *rest)})
    with pytest.raises(AuditError, match="books say"):
        verify_declared(baseline, private.public, truth)


# ---------------------------------------------------------------- the hazard the gate found


def test_the_misallocation_guard_stands_exactly_on_the_published_fx_floor() -> None:
    """Variant (c) on a foreign-currency payment re-keys its journal on another invoice's issue
    date, so the books would carry an fx_error nobody declared — once the gap clears the floor.

    Both halves matter. Refusing above the floor closes the hazard; NOT refusing below it keeps
    the draw the generator always made, because a sub-floor gap is rounding, no fx_error exists
    to go undeclared, and a needless refusal would move the RNG on and rewrite correct episodes.
    """
    public = build_episode(SEED, load_split_config("dev")).public
    base = public.manifest.base_currency
    invoices = {invoice.invoice_id: invoice for invoice in public.invoices}
    above = below = 0
    for pay in public.payments:
        if pay.currency == base or len(pay.invoice_refs) != 1:
            continue
        current = invoices.get(pay.invoice_refs[0].invoice_id)
        if current is None:
            continue
        for target in public.invoices:
            if target.currency != pay.currency or target.invoice_id == current.invoice_id:
                continue
            fires = _moves_fx_policy_date(public, pay, target)
            booked, expected, floor = _pair_amounts(public, pay, current, target)
            if abs(booked - expected) >= floor:
                above += 1
                assert fires, (pay.payment_id, target.invoice_id, booked, expected, floor)
            else:
                below += 1
                assert not fires, (pay.payment_id, target.invoice_id, booked, expected, floor)
    assert above > 0 and below > 0, (above, below)


def _pair_amounts(
    public: PublicEpisode, pay: Payment, current: Invoice, target: Invoice
) -> tuple[int, int, int]:
    """(booked base amount, the policy amount after re-pointing, the floor between them).

    Computed with the generator's own helpers on purpose: the question here is whether the guard
    and the fx injector stand on one bar, not whether the conversion itself is right - the audit
    answers that with its own arithmetic.
    """
    base = public.manifest.base_currency
    index = build_fx_index(public.fx)
    rates = [
        rate_on(
            index,
            policy_date(public.policy, invoice.issued_at, pay.timestamp),
            pay.currency,
            base,
            public.policy,
        )
        for invoice in (current, target)
    ]
    booked, expected = (convert_minor(pay.amount_minor, rate, public.policy) for rate in rates)
    return booked, expected, noise_floor(expected)


@pytest.mark.slow
def test_the_gate_holds_over_a_range_of_seeds() -> None:
    """``build_episode`` raises when the books and truth disagree, so building is the assertion.

    Before the guard, 3 of 54 dev episodes carried an fx_error no injector had recorded.
    """
    config = load_split_config("dev")
    built = 0
    for seed in range(20000, 20024):
        try:
            private = build_episode(seed, config)
        except InjectorError:
            continue  # a seed the generator itself refuses: not this gate's business
        built += 1
        verify_declared(build_clean_episode(seed, config).public, private.public, private.truth)
    assert built >= 18, built
