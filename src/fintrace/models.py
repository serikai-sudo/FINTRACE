"""Pydantic v2 data models: the frozen contract shared by every FINTRACE module.

Money is strict ``int`` minor units (``MinorUnits``) or ``Decimal`` (FX rates) — never ``float``.
Timestamps are ISO-8601 strings with an explicit UTC offset (JSON-first; no ``datetime`` objects
cross a model boundary). Free-form payloads (``JsonObject``) accept JSON values only — no
``Decimal``, ``set`` or ``tuple`` — and are immutable by convention: producers build them once
from ``model_dump(mode="json")`` and never mutate them afterwards.
``PublicEpisode`` is all the agent may ever see; ``PrivateEpisode`` is evaluator-only.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
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
    MAX_LEDGER_ROWS,
    MAX_PAGE_SIZE,
    MAX_STEPS,
    MIN_LEDGER_ROWS,
    OPAQUE_ID_PATTERN,
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    SCHEMA_VERSION,
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
MinorUnits: TypeAlias = Annotated[StrictInt, Field(ge=0)]  # money: no bool/float/str coercion
NonNegInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]  # counters and sizes
StepCount: TypeAlias = Annotated[StrictInt, Field(ge=0, le=MAX_STEPS)]
JsonObject: TypeAlias = dict[str, JsonValue]  # JSON-only payload (no Decimal / set / tuple)

# ---------------------------------------------------------------- ISO-8601 validation

_ISO8601_OFFSET_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?"
    r"([+-](?:[01][0-9]|2[0-3]):[0-5][0-9]|Z)$"
)
_ISO8601_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def validate_iso8601_with_offset(v: str) -> str:
    """Accept ``YYYY-MM-DDTHH:MM:SS[.f](±HH:MM|Z)``; ``Z`` and ``-00:00`` become ``+00:00``."""
    if _ISO8601_OFFSET_RE.fullmatch(v) is None:
        raise ValueError("timestamp must be ISO-8601 with an explicit UTC offset")
    datetime.fromisoformat(v)  # calendar validity
    if v.endswith("Z"):
        return f"{v[:-1]}+00:00"
    if v.endswith("-00:00"):
        return f"{v[:-6]}+00:00"
    return v


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
    minor_unit_digits: int = Field(strict=True, ge=0, le=6)
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
    n_ledger_rows: NonNegInt
    n_documents: NonNegInt
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
    debit_minor: MinorUnits
    credit_minor: MinorUnits
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
    qty: int = Field(strict=True, ge=1)
    unit_minor: MinorUnits
    amount_minor: MinorUnits

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
    total_minor: MinorUnits
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
    allocated_minor: MinorUnits


class Payment(FrozenModel):
    """Payment document with its invoice allocations."""

    doc_id: OpaqueId
    payment_id: OpaqueId
    timestamp: str
    currency: CurrencyCode
    amount_minor: MinorUnits
    invoice_refs: tuple[InvoiceAllocation, ...]

    @field_validator("timestamp")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        return validate_iso8601_with_offset(v)


class FXRate(FrozenModel):
    """Daily FX rate; an exact Decimal parsed from a string (generator quantizes), never a float."""

    date: str
    source_currency: CurrencyCode
    target_currency: CurrencyCode
    rate: Decimal = Field(gt=0, max_digits=24, decimal_places=16)

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
    expected_amount_minor: MinorUnits
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
    original: JsonObject
    modified: JsonObject
    caused_by: tuple[str, ...]
    note: str | None = None


class Provenance(FrozenModel):
    """Generation log: how the episode and its injections were produced."""

    generator_version: str
    seed_hash: str
    history: tuple[JsonObject, ...]
    injections: tuple[ProvenanceEntry, ...]


class PrivateEpisode(FrozenModel):
    """Public episode plus truth and provenance. Evaluator-only."""

    public: PublicEpisode
    truth: Truth
    provenance: Provenance

    @model_validator(mode="after")
    def _same_episode(self) -> Self:
        if self.truth.episode_id != self.public.manifest.episode_id:
            raise ValueError("truth.episode_id must equal public.manifest.episode_id")
        return self


# ---------------------------------------------------------------- submission models


class Finding(StrictModel):
    """One reported issue in an agent submission."""

    type: IssueType
    affected_ids: list[OpaqueId] = Field(max_length=MAX_AFFECTED_IDS_PER_FINDING)
    claimed_amount_minor: MinorUnits
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
    cursor: OpaqueId | None = None
    limit: int = Field(default=MAX_PAGE_SIZE, strict=True, ge=1, le=MAX_PAGE_SIZE)


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
    args: JsonObject
    # agent scratchpad, not a sixth tool: the environment ignores it, the trajectory logger keeps it
    thought: str | None = Field(default=None, max_length=500)


# ---------------------------------------------------------------- observation / state models


class Error(FrozenModel):
    """Environment error: a code and nothing else (no message, no details, no trace)."""

    code: ErrorCode


class ToolResult(FrozenModel):
    """Observation returned after every step: exactly one of result / error is set."""

    ok: bool
    tool: ToolName
    result: JsonObject | None = None
    error: Error | None = None
    steps_used: StepCount
    steps_remaining: StepCount

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
    step_count: StepCount = 0
    submitted: bool = False
    terminated: bool = False
    truncated: bool = False


class StepResult(FrozenModel):
    """Return value of env.step(); info holds exactly max_steps and version — never the seed."""

    observation: ToolResult
    reward: float = Field(
        default=0.0, allow_inf_nan=False, ge=REWARD_CLIP_MIN, le=REWARD_CLIP_MAX
    )
    terminated: bool = False
    truncated: bool = False
    info: JsonObject

    @field_validator("info")
    @classmethod
    def _info_exact_keys(cls, v: JsonObject) -> JsonObject:
        if set(v) != {"max_steps", "version"}:
            raise ValueError('info must contain exactly the keys "max_steps" and "version"')
        max_steps, version = v["max_steps"], v["version"]
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise ValueError("info.max_steps must be an int")
        if not isinstance(version, str):
            raise ValueError("info.version must be a str")
        return v


# ---------------------------------------------------------------- generator configuration


class EpisodeConfig(FrozenModel):
    """Configuration for generating one episode. Deterministic given (seed, config, VERSION)."""

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    n_ledger_rows_range: tuple[NonNegInt, NonNegInt] = (MIN_LEDGER_ROWS, MAX_LEDGER_ROWS)
    n_documents_range: tuple[NonNegInt, NonNegInt] = (20, 200)
    n_entities_range: tuple[NonNegInt, NonNegInt] = (5, 30)
    base_currency: CurrencyCode = "USD"
    foreign_currencies: tuple[CurrencyCode, ...] = ("EUR", "GBP")
    period_days: int = Field(default=90, ge=28, le=366)
    timezone: str = Field(default="+05:00", pattern=TIMEZONE_PATTERN)
    recurring_supplier_count: int = Field(default=3, ge=0, le=10)  # recurring suppliers = decoys
    split_payment_rate: Decimal = Field(default=Decimal("0.15"), ge=Decimal("0"), le=Decimal("0.5"))
    foreign_invoice_rate: Decimal = Field(
        default=Decimal("0.20"), ge=Decimal("0"), le=Decimal("0.6")
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # (floor, ceiling): rows >= 20 keep the misc fill-up solvable, entities <= 50 is the
        # size of the fictional-name table, 5 is the number of supported foreign currencies.
        bounds = {
            "n_ledger_rows_range": (20, 20_000),
            "n_documents_range": (2, 5_000),
            "n_entities_range": (1, 50),
        }
        for name, (floor, ceiling) in bounds.items():
            lo, hi = getattr(self, name)
            if not floor <= lo <= hi <= ceiling:
                raise ValueError(f"{name} must satisfy {floor} <= lo <= hi <= {ceiling}")
        if self.base_currency in self.foreign_currencies:
            raise ValueError("base_currency must not appear in foreign_currencies")
        if len(set(self.foreign_currencies)) != len(self.foreign_currencies):
            raise ValueError("foreign_currencies must be unique")
        if len(self.foreign_currencies) > 5:
            raise ValueError("at most 5 foreign currencies are supported")
        return self


__all__ = [
    "AccountName",
    "Action",
    "CalculatorArgs",
    "CurrencyCode",
    "EpisodeConfig",
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
    "JsonObject",
    "LedgerRow",
    "LineItem",
    "ListRecordsArgs",
    "Manifest",
    "MinorUnits",
    "NonNegInt",
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
    "StepCount",
    "StepResult",
    "StrictModel",
    "Submission",
    "SubmitArgs",
    "ToolResult",
    "Truth",
    "validate_iso8601_date",
    "validate_iso8601_with_offset",
]
