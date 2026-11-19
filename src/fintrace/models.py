"""Pydantic v2 data models: the frozen contract shared by every FINTRACE module.

Money is ``int`` minor units or ``Decimal`` (FX rates) — never ``float``. Timestamps are ISO-8601
strings with an explicit UTC offset (JSON-first; no ``datetime`` objects cross a model boundary).
``PublicEpisode`` is all the agent may ever see; ``PrivateEpisode`` is evaluator-only.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Final, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .constants import (
    CURRENCY_ISO4217_PATTERN,
    EPISODE_ID_PATTERN,
    ISSUE_ID_PATTERN,
    MAX_AFFECTED_IDS_PER_FINDING,
    MAX_CALCULATOR_EXPR_CHARS,
    MAX_EVIDENCE_IDS_PER_FINDING,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    MAX_ID_CHARS,
    MAX_PAGE_SIZE,
    OPAQUE_ID_PATTERN,
    TIMEZONE_PATTERN,
    ErrorCode,
    IssueType,
    RoundingMode,
    TableName,
    ToolName,
)

# ---------------------------------------------------------------- bases


class FrozenModel(BaseModel):
    """Immutable record: unknown fields rejected, attributes frozen after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StrictModel(BaseModel):
    """Mutable or agent-supplied payload: unknown fields rejected."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- reusable field types

OpaqueId: TypeAlias = Annotated[
    str, StringConstraints(pattern=OPAQUE_ID_PATTERN, max_length=MAX_ID_CHARS)
]
IssueId: TypeAlias = Annotated[str, StringConstraints(pattern=ISSUE_ID_PATTERN)]
CurrencyCode: TypeAlias = Annotated[str, StringConstraints(pattern=CURRENCY_ISO4217_PATTERN)]
AccountName: TypeAlias = Literal["AP", "AR", "CASH", "REVENUE", "EXPENSE", "FX_GAIN_LOSS"]

# ---------------------------------------------------------------- ISO-8601 validation

_ISO8601_OFFSET_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)$"
)
_ISO8601_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_iso8601_with_offset(v: str) -> str:
    """Accept ``YYYY-MM-DDTHH:MM:SS[.f](±HH:MM|Z)``; a trailing ``Z`` becomes ``+00:00``."""
    if _ISO8601_OFFSET_RE.fullmatch(v) is None:
        raise ValueError("timestamp must be ISO-8601 with an explicit UTC offset")
    datetime.fromisoformat(v)  # calendar validity and offset range
    return f"{v[:-1]}+00:00" if v.endswith("Z") else v


def validate_iso8601_date(v: str) -> str:
    """Accept a calendar date in the canonical ``YYYY-MM-DD`` form only."""
    if _ISO8601_DATE_RE.fullmatch(v) is None:
        raise ValueError("date must be ISO-8601 YYYY-MM-DD")
    date.fromisoformat(v)  # calendar validity
    return v


# ---------------------------------------------------------------- public data models


class Rounding(FrozenModel):
    """Rounding convention declared in the manifest."""

    mode: RoundingMode
    stage: Literal["per_line", "per_total"]


class PolicyPeriod(FrozenModel):
    """How a document is assigned to the reporting period."""

    start_inclusive: bool
    end_exclusive: bool
    timezone: str = Field(pattern=TIMEZONE_PATTERN)
    assignment_field: Literal["issued_at", "posted_at"]


class PolicyFX(FrozenModel):
    """Which FX rate applies to a transaction and how gaps are handled."""

    rate_date_rule: Literal["transaction_date", "posting_date"]
    fallback: Literal["previous_available_date", "next_available_date", "error"]
    rate_direction: Literal["source_to_target", "target_to_source"]


class PolicyRounding(FrozenModel):
    """Rounding applied to converted amounts."""

    mode: RoundingMode
    minor_unit_digits: int = Field(ge=0, le=6)
    apply_to: Literal["converted_total", "per_line"]


class Policy(FrozenModel):
    """Accounting policy of the episode's company (public)."""

    fx: PolicyFX
    period: PolicyPeriod
    rounding: PolicyRounding


class Manifest(FrozenModel):
    """Episode header: identity, period, sizes. Contains no seed."""

    schema_version: str
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    base_currency: CurrencyCode
    period_start: str
    period_end: str
    timezone: str = Field(pattern=TIMEZONE_PATTERN)
    n_ledger_rows: int = Field(ge=0)
    n_documents: int = Field(ge=0)
    rounding: Rounding

    @field_validator("period_start", "period_end")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        return validate_iso8601_with_offset(v)


class LedgerRow(FrozenModel):
    """One journal line; exactly one of debit/credit is non-zero."""

    row_id: OpaqueId
    timestamp: str
    account: AccountName
    debit_minor: int = Field(ge=0)
    credit_minor: int = Field(ge=0)
    currency: CurrencyCode
    entity_id: OpaqueId | None
    invoice_id: OpaqueId | None
    payment_id: OpaqueId | None
    journal_id: OpaqueId

    @field_validator("timestamp")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        return validate_iso8601_with_offset(v)

    @model_validator(mode="after")
    def _one_side_only(self) -> Self:
        if (self.debit_minor > 0) == (self.credit_minor > 0):
            raise ValueError("exactly one of debit_minor / credit_minor must be > 0")
        return self


class LineItem(FrozenModel):
    """Invoice line; amount is derived from quantity and unit price."""

    desc: str = Field(max_length=256)
    qty: int = Field(ge=1)
    unit_minor: int = Field(ge=0)
    amount_minor: int = Field(ge=0)

    @model_validator(mode="after")
    def _amount_is_qty_times_unit(self) -> Self:
        if self.amount_minor != self.qty * self.unit_minor:
            raise ValueError("amount_minor must equal qty * unit_minor")
        return self


class Invoice(FrozenModel):
    """Invoice document; total equals the sum of its line items."""

    doc_id: OpaqueId
    invoice_id: OpaqueId
    entity_id: OpaqueId
    issued_at: str
    due_at: str
    currency: CurrencyCode
    total_minor: int = Field(ge=0)
    line_items: tuple[LineItem, ...]

    @field_validator("issued_at", "due_at")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        return validate_iso8601_with_offset(v)

    @model_validator(mode="after")
    def _total_is_sum_of_lines(self) -> Self:
        if self.total_minor != sum(li.amount_minor for li in self.line_items):
            raise ValueError("total_minor must equal the sum of line_items.amount_minor")
        return self


class InvoiceAllocation(FrozenModel):
    """Portion of a payment allocated to one invoice."""

    invoice_id: OpaqueId
    allocated_minor: int = Field(ge=0)


class Payment(FrozenModel):
    """Payment document with its invoice allocations."""

    doc_id: OpaqueId
    payment_id: OpaqueId
    timestamp: str
    currency: CurrencyCode
    amount_minor: int = Field(ge=0)
    invoice_refs: tuple[InvoiceAllocation, ...]

    @field_validator("timestamp")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        return validate_iso8601_with_offset(v)


class FXRate(FrozenModel):
    """Daily exchange rate; the rate is an exact Decimal parsed from a string, never a float."""

    date: str
    source_currency: CurrencyCode
    target_currency: CurrencyCode
    rate: Decimal = Field(gt=0)

    @field_validator("date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        return validate_iso8601_date(v)

    @field_validator("rate", mode="before")
    @classmethod
    def _reject_float(cls, v: object) -> object:
        if isinstance(v, float):
            raise ValueError("rate must be a string or Decimal, never a float")
        return v


class PublicEpisode(FrozenModel):
    """Everything the agent may see. Tools and the environment read only this."""

    manifest: Manifest
    policy: Policy
    ledger: tuple[LedgerRow, ...]
    invoices: tuple[Invoice, ...]
    payments: tuple[Payment, ...]
    fx: tuple[FXRate, ...]


# ---------------------------------------------------------------- private (evaluator-only) models


class Issue(FrozenModel):
    """One injected error in canonical form; affected_ids are sorted and unique."""

    issue_id: IssueId
    type: IssueType
    affected_ids: tuple[OpaqueId, ...]
    expected_amount_minor: int = Field(ge=0)
    currency: CurrencyCode
    acceptable_evidence_sets: tuple[tuple[OpaqueId, ...], ...]
    interacts_with: tuple[IssueId, ...] = ()

    @field_validator("affected_ids")
    @classmethod
    def _canonical(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(set(v)):
            raise ValueError("affected_ids must be sorted and unique (canonical form)")
        return v


class Truth(FrozenModel):
    """Ground truth for one episode. Never reaches the agent surface."""

    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    issues: tuple[Issue, ...]


class ProvenanceEntry(FrozenModel):
    """Append-only record of one injection operation."""

    issue_id: IssueId
    op: str
    source_id: OpaqueId | None
    created_ids: tuple[OpaqueId, ...]
    original: dict[str, Any]
    modified: dict[str, Any]
    caused_by: tuple[str, ...]
    note: str | None = None


class Provenance(FrozenModel):
    """Generation log: how the episode and its injections were produced."""

    generator_version: str
    seed_hash: str
    history: tuple[dict[str, Any], ...]
    injections: tuple[ProvenanceEntry, ...]


class PrivateEpisode(FrozenModel):
    """Public episode plus truth and provenance. Evaluator-only."""

    public: PublicEpisode
    truth: Truth
    provenance: Provenance


# ---------------------------------------------------------------- submission models


class Finding(StrictModel):
    """One reported issue in an agent submission."""

    type: IssueType
    affected_ids: list[OpaqueId] = Field(max_length=MAX_AFFECTED_IDS_PER_FINDING)
    claimed_amount_minor: int = Field(ge=0)
    currency: CurrencyCode
    evidence_ids: list[OpaqueId] = Field(max_length=MAX_EVIDENCE_IDS_PER_FINDING)
    explanation: str = Field(max_length=MAX_EXPLANATION_CHARS)


class Submission(StrictModel):
    """Final answer of the agent for one episode."""

    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    findings: list[Finding] = Field(max_length=MAX_FINDINGS)


# ---------------------------------------------------------------- tool argument models


class ListRecordsArgs(StrictModel):
    """Arguments of the list_records tool."""

    table: TableName
    cursor: str | None = None
    limit: int = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class GetRecordArgs(StrictModel):
    """Arguments of the get_record tool."""

    table: TableName
    id: OpaqueId


class GetDocumentArgs(StrictModel):
    """Arguments of the get_document tool."""

    doc_id: OpaqueId


class CalculatorArgs(StrictModel):
    """Arguments of the calculator tool."""

    expression: str = Field(max_length=MAX_CALCULATOR_EXPR_CHARS)


class SubmitArgs(StrictModel):
    """Arguments of the submit tool."""

    findings: Submission


class Action(StrictModel):
    """Single entry point of env.step(); args are discriminated by the environment."""

    tool: ToolName
    args: dict[str, Any]


# ---------------------------------------------------------------- observation / state models


class Error(FrozenModel):
    """Environment error; code only, message stays generic."""

    code: ErrorCode
    message: str = ""


class ToolResult(FrozenModel):
    """Observation returned after every step: exactly one of result / error is set."""

    ok: bool
    tool: ToolName
    result: dict[str, Any] | None = None
    error: Error | None = None
    steps_used: int = Field(ge=0)
    steps_remaining: int = Field(ge=0)

    @model_validator(mode="after")
    def _result_xor_error(self) -> Self:
        if self.ok and (self.result is None or self.error is not None):
            raise ValueError("ok=True requires result and no error")
        if not self.ok and (self.result is not None or self.error is None):
            raise ValueError("ok=False requires error and no result")
        return self


Observation: TypeAlias = ToolResult


class EpisodeState(StrictModel):
    """Mutable per-episode counters owned by the environment."""

    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    step_count: int = 0
    submitted: bool = False
    terminated: bool = False
    truncated: bool = False


class StepResult(FrozenModel):
    """What env.step() returns; info is exactly {"max_steps", "version"} — never the seed."""

    observation: ToolResult
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any]


__all__ = [
    "AccountName",
    "Action",
    "CalculatorArgs",
    "CurrencyCode",
    "EpisodeState",
    "Error",
    "FXRate",
    "Finding",
    "FrozenModel",
    "GetDocumentArgs",
    "GetRecordArgs",
    "Invoice",
    "InvoiceAllocation",
    "Issue",
    "IssueId",
    "LedgerRow",
    "LineItem",
    "ListRecordsArgs",
    "Manifest",
    "Observation",
    "OpaqueId",
    "Payment",
    "Policy",
    "PolicyFX",
    "PolicyPeriod",
    "PolicyRounding",
    "PrivateEpisode",
    "Provenance",
    "ProvenanceEntry",
    "PublicEpisode",
    "Rounding",
    "StepResult",
    "StrictModel",
    "Submission",
    "SubmitArgs",
    "ToolResult",
    "Truth",
    "validate_iso8601_date",
    "validate_iso8601_with_offset",
]
