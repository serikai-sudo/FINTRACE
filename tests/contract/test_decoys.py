"""Decoys: legitimate patterns that look like errors; checkers silent, injectors keep away."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pytest

from fintrace.generator.build import build_clean_episode, build_episode
from fintrace.generator.company import CREDIT_NOTE_PREFIX
from fintrace.generator.injectors import duplicate, fx_error, payment, period
from fintrace.generator.injectors.base import inconsistent_ids, legitimate_pattern_ids
from fintrace.generator.validate import ConsistencyError, check_balanced, validate_clean_history
from fintrace.models import (
    DecoyCounts,
    EpisodeConfig,
    ErrorInjectionConfig,
    InjectorParams,
    InteractingScenarioParams,
    Invoice,
    Issue,
    JsonObject,
    Payment,
    PrivateEpisode,
)

FULL = DecoyCounts(repeated_payment_amounts=2, split_payments=1, refunds=1, similar_invoices=1)
CHECKS = {
    "duplicate_invoice": duplicate.independent_check,
    "payment_allocation": payment.independent_check,
    "fx_error": fx_error.independent_check,
    "period_misclassification": period.independent_check,
}


@pytest.fixture(scope="module")
def decoy_episode() -> PrivateEpisode:
    return build_episode(42, EpisodeConfig(decoys=FULL))


def _events(episode: PrivateEpisode, op: str) -> list[JsonObject]:
    return [e for e in episode.provenance.history if e.get("op") == f"generate_decoy_{op}"]


def _ids(event: JsonObject, key: str) -> list[str]:
    values = event[key]
    assert isinstance(values, list)
    return [str(v) for v in values]


def _invoice(episode: PrivateEpisode, invoice_id: str) -> Invoice:
    return next(i for i in episode.public.invoices if i.invoice_id == invoice_id)


def _payment(episode: PrivateEpisode, payment_id: str) -> Payment:
    return next(p for p in episode.public.payments if p.payment_id == payment_id)


def _journal_of(episode: PrivateEpisode, **ref: str) -> str:
    (field, value), = ref.items()
    rows = [r for r in episode.public.ledger if getattr(r, field) == value]
    journals = {r.journal_id for r in rows}
    assert len(journals) >= 1
    return sorted(journals)[0]


def _issue(kind: str, affected: tuple[str, ...], amount: int, currency: str) -> Issue:
    return Issue(
        issue_id="ISS-0001",
        type=kind,  # type: ignore[arg-type]
        affected_ids=tuple(sorted(affected)),
        expected_amount_minor=amount,
        currency=currency,
        acceptable_evidence_sets=((),),
    )


def test_repeated_payment_amounts_generated(decoy_episode: PrivateEpisode) -> None:
    events = _events(decoy_episode, "repeated_amounts")
    assert len(events) == 2
    for event in events:
        payments = [_payment(decoy_episode, p) for p in _ids(event, "payment_ids")]
        invoices = [_invoice(decoy_episode, i) for i in _ids(event, "invoice_ids")]
        assert len(payments) == 2 and len(invoices) == 2
        assert payments[0].amount_minor == payments[1].amount_minor  # the suspicious part
        assert payments[0].timestamp[:10] != payments[1].timestamp[:10]  # different dates
        key = duplicate._line_items_key
        assert key(invoices[0].line_items) != key(invoices[1].line_items)  # never exact twins
        assert {p.invoice_refs[0].invoice_id for p in payments} == {i.invoice_id for i in invoices}
        for pay in payments:  # ...but each settles its own invoice in full
            invoice = _invoice(decoy_episode, pay.invoice_refs[0].invoice_id)
            assert pay.invoice_refs[0].allocated_minor == pay.amount_minor == invoice.total_minor
        assert event["legitimate_pattern"] is True
        assert len(_ids(event, "row_ids")) == 8 and len(_ids(event, "journal_ids")) == 4


def test_repeated_payment_amounts_pass_validation(decoy_episode: PrivateEpisode) -> None:
    validate_clean_history(decoy_episode)  # no relaxation needed: the books are clean
    assert decoy_episode.truth.issues == () and decoy_episode.provenance.injections == ()
    base = decoy_episode.public.manifest.base_currency
    for event in _events(decoy_episode, "repeated_amounts"):
        invoice_ids = _ids(event, "invoice_ids")
        for payment_id in _ids(event, "payment_ids"):
            pay = _payment(decoy_episode, payment_id)
            invoice_id = pay.invoice_refs[0].invoice_id
            allocation = (payment_id, invoice_id)
            synthetic = (
                _issue("payment_allocation", allocation, pay.amount_minor, pay.currency),
                _issue("fx_error", (invoice_id, payment_id), 1, base),
                _issue(
                    "period_misclassification",
                    (payment_id, _journal_of(decoy_episode, payment_id=payment_id)),
                    pay.amount_minor,
                    pay.currency,
                ),
            )
            for issue in synthetic:
                with pytest.raises(ConsistencyError):
                    CHECKS[issue.type](decoy_episode.public, issue)
        twin = _issue(
            "duplicate_invoice", tuple(invoice_ids),
            _invoice(decoy_episode, invoice_ids[0]).total_minor,
            _invoice(decoy_episode, invoice_ids[0]).currency,
        )  # fmt: skip
        with pytest.raises(ConsistencyError):
            duplicate.independent_check(decoy_episode.public, twin)


def test_split_payments_generated(decoy_episode: PrivateEpisode) -> None:
    (event,) = _events(decoy_episode, "split_payment")
    (invoice_id,) = _ids(event, "invoice_ids")
    invoice = _invoice(decoy_episode, invoice_id)
    payments = [
        p for p in decoy_episode.public.payments if p.invoice_refs[0].invoice_id == invoice_id
    ]
    assert len(payments) == 2
    assert {p.payment_id for p in payments} == set(_ids(event, "payment_ids"))
    assert sum(p.invoice_refs[0].allocated_minor for p in payments) == invoice.total_minor
    assert all(p.amount_minor == p.invoice_refs[0].allocated_minor for p in payments)
    assert payments[0].timestamp != payments[1].timestamp
    assert all(p.amount_minor >= 100 for p in payments)


def test_split_payments_no_payment_allocation_issue(decoy_episode: PrivateEpisode) -> None:
    (event,) = _events(decoy_episode, "split_payment")
    (invoice_id,) = _ids(event, "invoice_ids")
    base = decoy_episode.public.manifest.base_currency
    for payment_id in _ids(event, "payment_ids"):
        pay = _payment(decoy_episode, payment_id)
        for amount in (pay.amount_minor, _invoice(decoy_episode, invoice_id).total_minor):
            synthetic = _issue("payment_allocation", (payment_id, invoice_id), amount, pay.currency)
            with pytest.raises(ConsistencyError):
                payment.independent_check(decoy_episode.public, synthetic)
        probes = (
            _issue("fx_error", (invoice_id, payment_id), 1, base),
            _issue(
                "period_misclassification",
                (payment_id, _journal_of(decoy_episode, payment_id=payment_id)),
                pay.amount_minor,
                pay.currency,
            ),
        )
        for probe in probes:
            with pytest.raises(ConsistencyError):
                CHECKS[probe.type](decoy_episode.public, probe)


def test_refunds_generated(decoy_episode: PrivateEpisode) -> None:
    (event,) = _events(decoy_episode, "refund")
    original_id, note_id = _ids(event, "invoice_ids")
    original, note = _invoice(decoy_episode, original_id), _invoice(decoy_episode, note_id)
    if not all(li.desc.startswith(CREDIT_NOTE_PREFIX) for li in note.line_items):
        original, note = note, original  # ids are permuted; identify the note by its lines
    assert all(li.desc.startswith(CREDIT_NOTE_PREFIX) for li in note.line_items)
    assert not any(li.desc.startswith(CREDIT_NOTE_PREFIX) for li in original.line_items)
    assert note.entity_id == original.entity_id and note.currency == original.currency
    assert datetime.fromisoformat(note.issued_at) > datetime.fromisoformat(original.issued_at)
    assert 0 < note.total_minor <= original.total_minor
    assert note.line_items[0].desc == f"{CREDIT_NOTE_PREFIX}{original.line_items[0].desc}"
    assert _ids(event, "payment_ids") == []
    # no synthetic issue over the pair is accepted: not a duplicate, not a cut-off error
    pair = (original.invoice_id, note.invoice_id)
    probes = [
        _issue("duplicate_invoice", pair, original.total_minor, original.currency),
        _issue("duplicate_invoice", pair, note.total_minor, note.currency),
        *(
            _issue(
                "period_misclassification",
                (doc.invoice_id, _journal_of(decoy_episode, invoice_id=doc.invoice_id)),
                doc.total_minor,
                doc.currency,
            )
            for doc in (original, note)
        ),
    ]
    for probe in probes:
        with pytest.raises(ConsistencyError):
            CHECKS[probe.type](decoy_episode.public, probe)
    # defence in depth: a payment allocated to a credit note is rejected by the validator
    donor = decoy_episode.public.payments[0]
    refs = (donor.invoice_refs[0].model_copy(update={"invoice_id": note.invoice_id}),)
    paid_note = donor.model_copy(update={"invoice_refs": refs, "currency": note.currency})
    payments = tuple(
        paid_note if p.payment_id == donor.payment_id else p for p in decoy_episode.public.payments
    )
    with pytest.raises(ConsistencyError, match="payment_allocation"):
        validate_clean_history(decoy_episode.public.model_copy(update={"payments": payments}))


def test_refunds_balanced(decoy_episode: PrivateEpisode) -> None:
    (event,) = _events(decoy_episode, "refund")
    notes = [
        _invoice(decoy_episode, i)
        for i in _ids(event, "invoice_ids")
        if _invoice(decoy_episode, i).line_items[0].desc.startswith(CREDIT_NOTE_PREFIX)
    ]
    (note,) = notes
    rows = [r for r in decoy_episode.public.ledger if r.invoice_id == note.invoice_id]
    assert len(rows) == 2 and len({r.journal_id for r in rows}) == 1
    debit = next(r for r in rows if r.debit_minor)
    credit = next(r for r in rows if r.credit_minor)
    assert debit.debit_minor == credit.credit_minor == note.total_minor
    assert (debit.account, credit.account) in {("AP", "EXPENSE"), ("REVENUE", "AR")}  # reversal
    check_balanced(decoy_episode.public.ledger)
    validate_clean_history(decoy_episode)
    # a reversal on an ordinary invoice (no refund lines) is still rejected by the validator
    original_id = next(i for i in _ids(event, "invoice_ids") if i != note.invoice_id)
    plain_rows = [r for r in decoy_episode.public.ledger if r.invoice_id == original_id]
    swapped = {
        r.row_id: r.model_copy(
            update={"debit_minor": r.credit_minor, "credit_minor": r.debit_minor}
        )
        for r in plain_rows
    }
    ledger = tuple(swapped.get(r.row_id, r) for r in decoy_episode.public.ledger)
    with pytest.raises(ConsistencyError, match="ledger_matches_documents"):
        validate_clean_history(decoy_episode.public.model_copy(update={"ledger": ledger}))


def test_similar_invoices_generated(decoy_episode: PrivateEpisode) -> None:
    (event,) = _events(decoy_episode, "similar_invoices")
    first, second = (_invoice(decoy_episode, i) for i in _ids(event, "invoice_ids"))
    assert (first.entity_id, first.total_minor, first.currency) == (
        second.entity_id, second.total_minor, second.currency,
    )  # fmt: skip
    key = duplicate._line_items_key
    assert key(first.line_items) != key(second.line_items)
    gap = abs(datetime.fromisoformat(first.issued_at) - datetime.fromisoformat(second.issued_at))
    assert gap >= timedelta(days=14)


def test_similar_invoices_not_caught_as_duplicate(decoy_episode: PrivateEpisode) -> None:
    (event,) = _events(decoy_episode, "similar_invoices")
    ids = tuple(_ids(event, "invoice_ids"))
    first = _invoice(decoy_episode, ids[0])
    synthetic = _issue("duplicate_invoice", ids, first.total_minor, first.currency)
    with pytest.raises(ConsistencyError, match="duplicate_check"):
        duplicate.independent_check(decoy_episode.public, synthetic)
    # and the duplicate injector never treats the pair as a source (unique-group rule + legit ids)
    assert not set(ids) & {i.invoice_id for i in duplicate._eligible_invoices(decoy_episode)}


def test_decoys_not_selected_as_injector_source() -> None:
    config = EpisodeConfig(
        decoys=FULL,
        errors=ErrorInjectionConfig(
            duplicate_invoice=InjectorParams(count=1),
            payment_allocation=InjectorParams(count=1),
            fx_error=InjectorParams(count=1),
            period_misclassification=InjectorParams(count=1),
            interacting=InteractingScenarioParams(scenario_a=1, scenario_b=1),
        ),
    )
    episode = build_episode(42, config)
    legit = legitimate_pattern_ids(episode)
    assert len(legit) >= 5 * 2 and len(episode.truth.issues) == 8
    for issue in episode.truth.issues:
        assert not set(issue.affected_ids) & legit, issue.issue_id
        for evidence in issue.acceptable_evidence_sets:
            assert not set(evidence) & legit
        CHECKS[issue.type](episode.public, issue)
    for entry in episode.provenance.injections:
        assert entry.source_id not in legit
        assert not set(entry.created_ids) & legit
        relaxed = entry.modified.get("relaxed_ids")
        assert not (set(map(str, relaxed)) & legit if isinstance(relaxed, list) else set())
    validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
    # every decoy group survived the injections untouched
    ops = Counter(str(e["op"]) for e in episode.provenance.history if e.get("legitimate_pattern"))
    assert ops == {
        "generate_decoy_repeated_amounts": 2,
        "generate_decoy_split_payment": 1,
        "generate_decoy_refund": 1,
        "generate_decoy_similar_invoices": 1,
    }


def test_backward_compat_empty_decoys(decoy_episode: PrivateEpisode) -> None:
    assert DecoyCounts().total() == 0 and EpisodeConfig().decoys == DecoyCounts()
    clean = build_clean_episode(42)
    assert build_episode(42).model_dump_json() == clean.model_dump_json()
    assert not any(e.get("legitimate_pattern") for e in clean.provenance.history)
    # a decoy configuration shares the seed's clean base: same company period, policy and FX
    assert decoy_episode.provenance.seed_hash == clean.provenance.seed_hash
    assert decoy_episode.public.policy == clean.public.policy
    assert decoy_episode.public.fx == clean.public.fx
    assert decoy_episode.public.manifest.period_start == clean.public.manifest.period_start
    assert decoy_episode.public.manifest.episode_id != clean.public.manifest.episode_id
    # decoy configurations are deterministic too, and the document range still holds
    again = build_episode(42, EpisodeConfig(decoys=FULL))
    assert again.model_dump_json() == decoy_episode.model_dump_json()
    lo, hi = EpisodeConfig().n_documents_range
    assert lo <= decoy_episode.public.manifest.n_documents <= hi
    # a configuration whose document budget cannot hold the decoys fails loudly
    tiny = EpisodeConfig(
        n_ledger_rows_range=(20, 20), n_documents_range=(2, 5),
        decoys=DecoyCounts(repeated_payment_amounts=2),
    )  # fmt: skip
    with pytest.raises(ValueError, match="document budget"):
        build_episode(1, tiny)
