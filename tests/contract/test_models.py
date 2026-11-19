"""Contract tests for the frozen data models in fintrace.models."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from fintrace.models import (
    CalculatorArgs,
    Error,
    Finding,
    FXRate,
    Invoice,
    Issue,
    LedgerRow,
    LineItem,
    Manifest,
    Payment,
    Policy,
    PrivateEpisode,
    Provenance,
    ProvenanceEntry,
    PublicEpisode,
    Submission,
    ToolResult,
    Truth,
)

EPISODE_ID = "ep_0123456789ab"
TS = "2026-02-03T10:15:00+05:00"

# ---------------------------------------------------------------- factories (valid baselines)


def _manifest(**over: object) -> Manifest:
    data: dict[str, object] = {
        "schema_version": "1.0.0",
        "episode_id": EPISODE_ID,
        "base_currency": "KZT",
        "period_start": "2026-01-01T00:00:00+05:00",
        "period_end": "2026-04-01T00:00:00+05:00",
        "timezone": "+05:00",
        "n_ledger_rows": 2,
        "n_documents": 2,
        "rounding": {"mode": "ROUND_HALF_EVEN", "stage": "per_total"},
    }
    data.update(over)
    return Manifest.model_validate(data)


def _policy() -> Policy:
    return Policy.model_validate(
        {
            "fx": {
                "rate_date_rule": "transaction_date",
                "fallback": "previous_available_date",
                "rate_direction": "source_to_target",
            },
            "period": {
                "start_inclusive": True,
                "end_exclusive": True,
                "timezone": "Asia/Almaty",
                "assignment_field": "issued_at",
            },
            "rounding": {
                "mode": "ROUND_HALF_EVEN",
                "minor_unit_digits": 2,
                "apply_to": "converted_total",
            },
        }
    )


def _ledger_row(**over: object) -> LedgerRow:
    data: dict[str, object] = {
        "row_id": "L-0001",
        "timestamp": TS,
        "account": "AR",
        "debit_minor": 125000,
        "credit_minor": 0,
        "currency": "KZT",
        "entity_id": "ENT-001",
        "invoice_id": "INV-001",
        "payment_id": None,
        "journal_id": "J-0001",
    }
    data.update(over)
    return LedgerRow.model_validate(data)


def _line_item(**over: object) -> LineItem:
    data: dict[str, object] = {
        "desc": "Widget",
        "qty": 2,
        "unit_minor": 50000,
        "amount_minor": 100000,
    }
    data.update(over)
    return LineItem.model_validate(data)


def _invoice(**over: object) -> Invoice:
    data: dict[str, object] = {
        "doc_id": "DOC-INV-001",
        "invoice_id": "INV-001",
        "entity_id": "ENT-001",
        "issued_at": TS,
        "due_at": "2026-03-05T10:15:00+05:00",
        "currency": "KZT",
        "total_minor": 125000,
        "line_items": [
            {"desc": "Widget", "qty": 2, "unit_minor": 50000, "amount_minor": 100000},
            {"desc": "Delivery", "qty": 1, "unit_minor": 25000, "amount_minor": 25000},
        ],
    }
    data.update(over)
    return Invoice.model_validate(data)


def _payment() -> Payment:
    return Payment.model_validate(
        {
            "doc_id": "DOC-PAY-001",
            "payment_id": "PAY-001",
            "timestamp": "2026-02-20T09:00:00+05:00",
            "currency": "KZT",
            "amount_minor": 125000,
            "invoice_refs": [{"invoice_id": "INV-001", "allocated_minor": 125000}],
        }
    )


def _fx(**over: object) -> FXRate:
    data: dict[str, object] = {
        "date": "2026-02-03",
        "source_currency": "USD",
        "target_currency": "KZT",
        "rate": Decimal("512.35"),
    }
    data.update(over)
    return FXRate.model_validate(data)


def _public_episode() -> PublicEpisode:
    return PublicEpisode(
        manifest=_manifest(),
        policy=_policy(),
        ledger=(
            _ledger_row(),
            _ledger_row(row_id="L-0002", account="REVENUE", debit_minor=0, credit_minor=125000),
        ),
        invoices=(_invoice(),),
        payments=(_payment(),),
        fx=(_fx(),),
    )


def _private_episode() -> PrivateEpisode:
    issue = Issue(
        issue_id="ISS-01",
        type="duplicate_invoice",
        affected_ids=("INV-001", "INV-002"),
        expected_amount_minor=125000,
        currency="KZT",
        acceptable_evidence_sets=(("INV-001", "INV-002"), ("DOC-INV-001", "DOC-INV-002")),
    )
    entry = ProvenanceEntry(
        issue_id="ISS-01",
        op="duplicate_invoice",
        source_id="INV-001",
        created_ids=("INV-002",),
        original={"invoice_id": "INV-001"},
        modified={"invoice_id": "INV-002"},
        caused_by=(),
    )
    return PrivateEpisode(
        public=_public_episode(),
        truth=Truth(episode_id=EPISODE_ID, issues=(issue,)),
        provenance=Provenance(
            generator_version="0.1.0",
            seed_hash="3f2a9c",
            history=({"event": "generate_base"}, {"event": "inject", "issue_id": "ISS-01"}),
            injections=(entry,),
        ),
    )


def _finding(**over: object) -> dict[str, object]:
    data: dict[str, object] = {
        "type": "duplicate_invoice",
        "affected_ids": ["INV-001", "INV-002"],
        "claimed_amount_minor": 125000,
        "currency": "KZT",
        "evidence_ids": ["DOC-INV-001", "DOC-INV-002"],
        "explanation": "Same supplier, amount and date; second invoice is a duplicate.",
    }
    data.update(over)
    return data


# ---------------------------------------------------------------- Finding / Submission


def test_finding_rejects_extra_fields() -> None:
    assert Finding.model_validate(_finding()).type == "duplicate_invoice"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Finding.model_validate(_finding(seed=42))


def test_finding_affected_ids_capped() -> None:
    at_cap = Finding.model_validate(_finding(affected_ids=[f"ID-{i}" for i in range(20)]))
    assert len(at_cap.affected_ids) == 20
    with pytest.raises(ValidationError, match="too_long"):
        Finding.model_validate(_finding(affected_ids=[f"ID-{i}" for i in range(21)]))


def test_finding_evidence_capped() -> None:
    at_cap = Finding.model_validate(_finding(evidence_ids=[f"E-{i}" for i in range(10)]))
    assert len(at_cap.evidence_ids) == 10
    with pytest.raises(ValidationError, match="too_long"):
        Finding.model_validate(_finding(evidence_ids=[f"E-{i}" for i in range(11)]))


def test_finding_explanation_capped() -> None:
    assert len(Finding.model_validate(_finding(explanation="x" * 1000)).explanation) == 1000
    with pytest.raises(ValidationError, match="string_too_long"):
        Finding.model_validate(_finding(explanation="x" * 1001))


@pytest.mark.parametrize("currency", ["usd", "USDD", "US", "U$D", ""])
def test_finding_wrong_currency_pattern(currency: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        Finding.model_validate(_finding(currency=currency))


def test_finding_wrong_type_enum() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        Finding.model_validate(_finding(type="duplicates"))


def test_submission_cap_30() -> None:
    assert len(Submission(episode_id=EPISODE_ID, findings=[_finding()] * 30).findings) == 30
    with pytest.raises(ValidationError, match="too_long"):
        Submission(episode_id=EPISODE_ID, findings=[_finding()] * 31)


@pytest.mark.parametrize("episode_id", ["episode_1", "ep_", "ep_XYZ", "EP_0123456789ab"])
def test_submission_wrong_episode_id(episode_id: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        Submission(episode_id=episode_id, findings=[])


# ---------------------------------------------------------------- ledger / documents


def test_ledger_row_balance() -> None:
    assert _ledger_row().debit_minor == 125000
    assert _ledger_row(debit_minor=0, credit_minor=1).credit_minor == 1
    with pytest.raises(ValidationError, match="exactly one of debit_minor / credit_minor"):
        _ledger_row(debit_minor=0, credit_minor=0)
    with pytest.raises(ValidationError, match="exactly one of debit_minor / credit_minor"):
        _ledger_row(debit_minor=100, credit_minor=100)


def test_invoice_total_mismatch() -> None:
    assert _invoice().total_minor == 125000
    with pytest.raises(ValidationError, match="total_minor must equal the sum"):
        _invoice(total_minor=125001)


def test_line_item_amount_mismatch() -> None:
    assert _line_item().amount_minor == 100000
    with pytest.raises(ValidationError, match="amount_minor must equal qty"):
        _line_item(amount_minor=99999)


def test_fx_rate_decimal_not_float() -> None:
    exact = Decimal("1.08420000000001")  # 15 significant digits, beyond float round-trip safety
    rate = _fx(rate=exact).rate
    assert isinstance(rate, Decimal)
    assert rate == exact
    assert str(rate) == "1.08420000000001"
    assert _fx(rate="1.0842").rate == Decimal("1.0842")  # str input (what JSON files carry)
    assert _fx().model_dump_json().count('"rate":"512.35"') == 1  # serialised as a string
    from_json = FXRate.model_validate_json(
        '{"date":"2026-02-03","source_currency":"USD","target_currency":"KZT","rate":"1.0842"}'
    )
    assert from_json.rate == Decimal("1.0842")
    with pytest.raises(ValidationError, match="never a float"):
        _fx(rate=1.0842)
    with pytest.raises(ValidationError, match="never a float"):
        FXRate.model_validate_json(
            '{"date":"2026-02-03","source_currency":"USD","target_currency":"KZT","rate":1.0842}'
        )


# ---------------------------------------------------------------- ISO-8601


@pytest.mark.parametrize(
    "timestamp",
    ["2026-02-03T10:15:00", "2026-02-03 10:15:00+05:00", "2026-02-03", "2026-02-30T10:15:00Z"],
)
def test_iso8601_rejects_no_offset(timestamp: str) -> None:
    with pytest.raises(ValidationError):
        _ledger_row(timestamp=timestamp)


def test_iso8601_accepts_z() -> None:
    assert _ledger_row(timestamp="2026-02-03T10:15:00Z").timestamp == "2026-02-03T10:15:00+00:00"
    with_fraction = _ledger_row(timestamp="2026-02-03T10:15:00.250Z").timestamp
    assert with_fraction == "2026-02-03T10:15:00.250+00:00"  # fraction digits preserved
    assert _ledger_row(timestamp=TS).timestamp == TS
    assert _fx(date="2026-02-03").date == "2026-02-03"
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        _fx(date="20260203")


# ---------------------------------------------------------------- tool results / args


def test_tool_result_ok_without_result() -> None:
    assert ToolResult(ok=True, tool="get_record", result={}, steps_used=1, steps_remaining=99).ok
    with pytest.raises(ValidationError, match="ok=True requires result"):
        ToolResult(ok=True, tool="get_record", result=None, steps_used=1, steps_remaining=99)
    with pytest.raises(ValidationError, match="ok=True requires result"):
        ToolResult(
            ok=True,
            tool="get_record",
            result={},
            error=Error(code="NOT_FOUND"),
            steps_used=1,
            steps_remaining=99,
        )


def test_tool_result_err_with_result() -> None:
    err = Error(code="NOT_FOUND")
    failed = ToolResult(ok=False, tool="get_record", error=err, steps_used=1, steps_remaining=99)
    assert failed.error is err and failed.result is None
    with pytest.raises(ValidationError, match="ok=False requires error"):
        ToolResult(
            ok=False,
            tool="get_record",
            result={"x": 1},
            error=None,
            steps_used=1,
            steps_remaining=99,
        )
    with pytest.raises(ValidationError, match="ok=False requires error"):
        ToolResult(
            ok=False,
            tool="get_record",
            result={"x": 1},
            error=err,
            steps_used=1,
            steps_remaining=99,
        )


def test_calculator_args_200_char_limit() -> None:
    assert len(CalculatorArgs(expression="1+" * 100).expression) == 200
    with pytest.raises(ValidationError, match="string_too_long"):
        CalculatorArgs(expression="1" * 201)


# ---------------------------------------------------------------- episode shapes


def test_public_episode_is_frozen() -> None:
    episode = _public_episode()
    with pytest.raises(ValidationError, match="frozen_instance"):
        episode.manifest = _manifest(n_documents=99)  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen_instance"):
        episode.ledger[0].debit_minor = 1  # type: ignore[misc]
    assert isinstance(episode.ledger, tuple)
    assert isinstance(episode.invoices[0].line_items, tuple)


def test_private_episode_shape() -> None:
    private = _private_episode()
    assert isinstance(private.truth.issues, tuple)
    assert not isinstance(private.truth.issues, list)
    assert isinstance(private.truth.issues[0].affected_ids, tuple)
    assert isinstance(private.truth.issues[0].acceptable_evidence_sets[0], tuple)
    assert isinstance(private.provenance.injections, tuple)
    assert private.public.manifest.episode_id == private.truth.episode_id
    with pytest.raises(ValidationError, match="sorted and unique"):
        Issue(
            issue_id="ISS-02",
            type="fx_error",
            affected_ids=("L-0002", "L-0001"),
            expected_amount_minor=1,
            currency="KZT",
            acceptable_evidence_sets=(("L-0001",),),
        )
