"""Contract tests for the frozen data models in fintrace.models."""

from __future__ import annotations

from decimal import Decimal
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

import fintrace.models as models
from fintrace.constants import (
    ERROR_CODES,
    ISSUE_TYPES,
    ROUNDING_MODES,
    TABLE_NAMES,
    TOOL_NAMES,
    ErrorCode,
    IssueType,
    RoundingMode,
    TableName,
    ToolName,
)
from fintrace.models import (
    Action,
    CalculatorArgs,
    EpisodeState,
    Error,
    Finding,
    FXRate,
    Invoice,
    Issue,
    LedgerRow,
    LineItem,
    ListRecordsArgs,
    Manifest,
    Payment,
    Policy,
    PolicyPeriod,
    PrivateEpisode,
    Provenance,
    ProvenanceEntry,
    PublicEpisode,
    StepResult,
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


def _issue(**over: object) -> Issue:
    data: dict[str, object] = {
        "issue_id": "ISS-01",
        "type": "duplicate_invoice",
        "affected_ids": ("INV-001", "INV-002"),
        "expected_amount_minor": 125000,
        "currency": "KZT",
        "acceptable_evidence_sets": (("INV-001", "INV-002"), ("DOC-INV-001", "DOC-INV-002")),
    }
    data.update(over)
    return Issue.model_validate(data)


def _private_episode(**over: object) -> PrivateEpisode:
    entry = ProvenanceEntry(
        issue_id="ISS-01",
        op="duplicate_invoice",
        source_id="INV-001",
        created_ids=("INV-002",),
        original={"invoice_id": "INV-001", "total_minor": 125000},
        modified={"invoice_id": "INV-002", "total_minor": 125000},
        caused_by=(),
    )
    data: dict[str, object] = {
        "public": _public_episode(),
        "truth": Truth(episode_id=EPISODE_ID, issues=(_issue(),)),
        "provenance": Provenance(
            generator_version="0.1.0",
            seed_hash="3f2a9c",
            history=({"event": "generate_base"}, {"event": "inject", "issue_id": "ISS-01"}),
            injections=(entry,),
        ),
    }
    data.update(over)
    return PrivateEpisode.model_validate(data)


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


def _tool_result(**over: object) -> ToolResult:
    data: dict[str, object] = {
        "ok": True,
        "tool": "get_record",
        "result": {"row_id": "L-0001"},
        "steps_used": 1,
        "steps_remaining": 99,
    }
    data.update(over)
    return ToolResult.model_validate(data)


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


@pytest.mark.parametrize("currency", ["usd", "USDD", "US", "U$D", "", "USD\n"])
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


@pytest.mark.parametrize(
    "episode_id",
    ["episode_1", "ep_", "ep_XYZ", "EP_0123456789ab", "ep_012345678", "ep_0123456789abcdef0"],
)
def test_submission_wrong_episode_id(episode_id: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        Submission(episode_id=episode_id, findings=[])


def test_submission_episode_id_boundaries() -> None:
    assert Submission(episode_id="ep_0123456789", findings=[]).episode_id == "ep_0123456789"
    sixteen = "ep_0123456789abcdef"
    assert Submission(episode_id=sixteen, findings=[]).episode_id == sixteen


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


@pytest.mark.parametrize("bad", [125000.0, True, "125000"])
def test_money_fields_are_strict_int(bad: object) -> None:
    with pytest.raises(ValidationError, match="int_type"):
        _ledger_row(debit_minor=bad)
    with pytest.raises(ValidationError, match="int_type"):
        Finding.model_validate(_finding(claimed_amount_minor=bad))
    with pytest.raises(ValidationError, match="int_type"):
        _line_item(qty=bad)


def test_money_fields_reject_json_float() -> None:
    with pytest.raises(ValidationError, match="int_type"):
        LineItem.model_validate_json(
            '{"desc":"x","qty":1,"unit_minor":100.0,"amount_minor":100}'
        )


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


@pytest.mark.parametrize(
    ("bad", "error_type"),
    [("1e400", "decimal_max_digits"), ("NaN", "finite_number"), ("0", "greater_than"),
     ("-1", "greater_than")],
)
def test_fx_rate_bounds(bad: str, error_type: str) -> None:
    with pytest.raises(ValidationError, match=error_type):
        _fx(rate=bad)


# ---------------------------------------------------------------- ISO-8601


@pytest.mark.parametrize(
    "timestamp",
    ["2026-02-03T10:15:00", "2026-02-03 10:15:00+05:00", "2026-02-03", "2026-02-03T10:15:00 Z"],
)
def test_iso8601_rejects_no_offset(timestamp: str) -> None:
    with pytest.raises(ValidationError, match="explicit UTC offset"):
        _ledger_row(timestamp=timestamp)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-02-30T10:15:00Z",  # invalid calendar date
        "2026-02-03T10:15:00+24:00",  # offset hour out of range
        "2026-02-03T10:15:00+05:60",  # offset minute out of range
        "2026-02-03T10:15:00+" + chr(0x660) + chr(0x665) + ":00",  # non-ASCII digits
    ],
)
def test_iso8601_rejects_invalid_calendar_or_offset(timestamp: str) -> None:
    with pytest.raises(ValidationError):
        _ledger_row(timestamp=timestamp)


def test_iso8601_accepts_z() -> None:
    assert _ledger_row(timestamp="2026-02-03T10:15:00Z").timestamp == "2026-02-03T10:15:00+00:00"
    with_fraction = _ledger_row(timestamp="2026-02-03T10:15:00.250Z").timestamp
    assert with_fraction == "2026-02-03T10:15:00.250+00:00"  # fraction digits preserved
    minus_zero = _ledger_row(timestamp="2026-02-03T10:15:00-00:00").timestamp
    assert minus_zero == "2026-02-03T10:15:00+00:00"
    assert _ledger_row(timestamp=TS).timestamp == TS
    assert _fx(date="2026-02-03").date == "2026-02-03"
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        _fx(date="20260203")


# ---------------------------------------------------------------- tool results / args


def test_tool_result_ok_without_result() -> None:
    assert _tool_result().ok
    with pytest.raises(ValidationError, match="ok=True requires result"):
        _tool_result(result=None)
    with pytest.raises(ValidationError, match="ok=True requires result"):
        _tool_result(error=Error(code="NOT_FOUND"))


def test_tool_result_err_with_result() -> None:
    err = Error(code="NOT_FOUND")
    failed = _tool_result(ok=False, result=None, error=err)
    assert failed.error is err and failed.result is None
    with pytest.raises(ValidationError, match="ok=False requires error"):
        _tool_result(ok=False, result={"x": 1}, error=None)
    with pytest.raises(ValidationError, match="ok=False requires error"):
        _tool_result(ok=False, result={"x": 1}, error=err)
    with pytest.raises(ValidationError, match="ok=False requires error"):
        _tool_result(ok=False, result=None, error=None)


def test_tool_result_steps_bounded() -> None:
    assert _tool_result(steps_used=100, steps_remaining=0).steps_used == 100
    with pytest.raises(ValidationError, match="less_than_equal"):
        _tool_result(steps_used=101)
    with pytest.raises(ValidationError, match="greater_than_equal"):
        _tool_result(steps_remaining=-1)


def test_error_is_code_only() -> None:
    assert Error(code="NOT_FOUND").model_dump() == {"code": "NOT_FOUND"}
    assert Error(code="NOT_FOUND").model_dump_json() == '{"code":"NOT_FOUND"}'
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Error.model_validate({"code": "NOT_FOUND", "message": "Traceback (most recent call last)"})
    with pytest.raises(ValidationError, match="literal_error"):
        Error.model_validate({"code": "SOMETHING_ELSE"})


def test_payload_fields_are_json_only() -> None:
    assert Action(tool="calculator", args={"expression": "1+1"}).args == {"expression": "1+1"}
    with pytest.raises(ValidationError, match="invalid-json-value"):
        _tool_result(result={"rate": Decimal("1.5")})
    with pytest.raises(ValidationError, match="invalid-json-value"):
        _tool_result(result={"ids": {"a", "b"}})
    with pytest.raises(ValidationError, match="invalid-json-value"):
        Action(tool="submit", args={"findings": ("x",)})


def test_calculator_args_200_char_limit() -> None:
    assert len(CalculatorArgs(expression="1+" * 100).expression) == 200
    with pytest.raises(ValidationError, match="string_too_long"):
        CalculatorArgs(expression="1" * 201)


def test_list_records_args_bounded() -> None:
    assert ListRecordsArgs(table="ledger").limit == 50
    assert ListRecordsArgs(table="ledger", cursor="offset:50", limit=1).cursor == "offset:50"
    with pytest.raises(ValidationError, match="string_too_long"):
        ListRecordsArgs(table="ledger", cursor="c" * 65)
    with pytest.raises(ValidationError, match="less_than_equal"):
        ListRecordsArgs(table="ledger", limit=51)
    with pytest.raises(ValidationError, match="int_type"):
        ListRecordsArgs(table="ledger", limit="5")
    with pytest.raises(ValidationError, match="literal_error"):
        ListRecordsArgs(table="truth")


# ---------------------------------------------------------------- step result / state


def test_step_result_info_exact_keys_and_reward_bounds() -> None:
    ok = StepResult(observation=_tool_result(), info={"max_steps": 100, "version": "1.0.0"})
    assert ok.reward == 0.0 and ok.info == {"max_steps": 100, "version": "1.0.0"}
    with pytest.raises(ValidationError, match="exactly the keys"):
        StepResult(
            observation=_tool_result(),
            info={"max_steps": 100, "version": "1.0.0", "seed": 42},
        )
    with pytest.raises(ValidationError, match="exactly the keys"):
        StepResult(observation=_tool_result(), info={"max_steps": 100})
    with pytest.raises(ValidationError, match="max_steps must be an int"):
        StepResult(observation=_tool_result(), info={"max_steps": True, "version": "1.0.0"})
    with pytest.raises(ValidationError, match="version must be a str"):
        StepResult(observation=_tool_result(), info={"max_steps": 100, "version": 1})
    info = {"max_steps": 100, "version": "1.0.0"}
    assert StepResult(observation=_tool_result(), reward=-1.0, info=info).reward == -1.0
    with pytest.raises(ValidationError, match="finite_number"):
        StepResult(observation=_tool_result(), reward=float("nan"), info=info)
    with pytest.raises(ValidationError, match="less_than_equal"):
        StepResult(observation=_tool_result(), reward=1.5, info=info)


def test_episode_state_counters() -> None:
    state = EpisodeState(episode_id=EPISODE_ID)
    assert (state.step_count, state.submitted, state.terminated, state.truncated) == (0, 0, 0, 0)
    with pytest.raises(ValidationError, match="greater_than_equal"):
        EpisodeState(episode_id=EPISODE_ID, step_count=-5)
    with pytest.raises(ValidationError, match="less_than_equal"):
        EpisodeState(episode_id=EPISODE_ID, step_count=101)


# ---------------------------------------------------------------- episode shapes


def test_public_episode_is_frozen() -> None:
    episode = _public_episode()
    with pytest.raises(ValidationError, match="frozen_instance"):
        episode.manifest = _manifest(n_documents=99)  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen_instance"):
        episode.ledger[0].debit_minor = 1  # type: ignore[misc]
    assert isinstance(episode.ledger, tuple)
    assert isinstance(episode.invoices[0].line_items, tuple)
    assert hash(episode) == hash(_public_episode())  # frozen + tuples => hashable, value-based


def test_private_episode_shape() -> None:
    private = _private_episode()
    assert isinstance(private.truth.issues, tuple)
    assert not isinstance(private.truth.issues, list)
    assert isinstance(private.truth.issues[0].affected_ids, tuple)
    assert isinstance(private.truth.issues[0].acceptable_evidence_sets[0], tuple)
    assert isinstance(private.provenance.injections, tuple)
    assert private.public.manifest.episode_id == private.truth.episode_id
    with pytest.raises(ValidationError, match="sorted and unique"):
        _issue(issue_id="ISS-02", affected_ids=("L-0002", "L-0001"))
    with pytest.raises(ValidationError, match="sorted and unique"):
        _issue(issue_id="ISS-02", affected_ids=("L-0001", "L-0001"))
    with pytest.raises(ValidationError, match=r"truth\.episode_id must equal"):
        _private_episode(truth=Truth(episode_id="ep_ffffffffff", issues=()))


def test_private_episode_json_round_trip() -> None:
    private = _private_episode()
    dumped = private.model_dump_json()
    restored = PrivateEpisode.model_validate_json(dumped)
    assert restored == private
    assert restored.model_dump_json() == dumped
    assert '"seed"' not in private.public.model_dump_json()


# ---------------------------------------------------------------- patterns / schema / mirrors


@pytest.mark.parametrize("bad_id", ["INV 001", "INV-001\n", "", "x" * 65, "INV/001"])
def test_opaque_id_pattern_edges(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        _ledger_row(row_id=bad_id)


@pytest.mark.parametrize(
    "bad_issue_id", ["ISS-1", "ISS-01\n", "ISS-" + chr(0x660) + chr(0x661), "iss-01"]
)
def test_issue_id_pattern_edges(bad_issue_id: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        _issue(issue_id=bad_issue_id)


@pytest.mark.parametrize("tz", ["+05:00", "-03:30", "UTC", "Asia/Almaty", "Etc/GMT+5",
                                "America/Argentina/Buenos_Aires"])
def test_timezone_pattern_accepts(tz: str) -> None:
    assert _manifest(timezone=tz).timezone == tz


@pytest.mark.parametrize("tz", ["+99:99", "+05:60", "+5:00", "Almaty", "utc", "Asia/Almaty\n"])
def test_timezone_pattern_rejects(tz: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        PolicyPeriod(
            start_inclusive=True, end_exclusive=True, timezone=tz, assignment_field="issued_at"
        )


def test_literal_aliases_match_tuple_mirrors() -> None:
    assert get_args(IssueType) == ISSUE_TYPES
    assert get_args(ErrorCode) == ERROR_CODES
    assert get_args(ToolName) == TOOL_NAMES
    assert get_args(TableName) == TABLE_NAMES
    assert get_args(RoundingMode) == ROUNDING_MODES


def test_every_model_generates_json_schema() -> None:
    model_classes = [
        obj
        for obj in (getattr(models, name) for name in models.__all__)
        if isinstance(obj, type) and issubclass(obj, BaseModel)
    ]
    # 31 blueprint models + FrozenModel/StrictModel + 3 config models
    # (EpisodeConfig, InjectorParams, ErrorInjectionConfig)
    assert len(model_classes) == 37  # + InteractingScenarioParams
    for cls in model_classes:
        schema = cls.model_json_schema()
        assert schema.get("additionalProperties") is False, cls.__name__
