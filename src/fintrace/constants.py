"""Project-wide constants: the frozen contract every FINTRACE module imports. No functions here."""

from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

# ---------------------------------------------------------------- 1. Version
SCHEMA_VERSION: Final[str] = "1.0.0"  # on-disk episode artefacts (manifest, truth, provenance)
ENV_VERSION: Final[str] = "1.0.0"  # environment contract (tools, observation, reward)

# ---------------------------------------------------------------- 2. Limits
MAX_STEPS: Final[int] = 100  # tool calls per episode (jury limit)
MAX_PAGE_SIZE: Final[int] = 50  # records per list_records page
MAX_FINDINGS: Final[int] = 30  # findings per submission
MAX_EVIDENCE_IDS_PER_FINDING: Final[int] = 10
MAX_AFFECTED_IDS_PER_FINDING: Final[int] = 20
MAX_EXPLANATION_CHARS: Final[int] = 1000
MAX_CALCULATOR_EXPR_CHARS: Final[int] = 200
MAX_ID_CHARS: Final[int] = 64  # any opaque identifier (row, document, invoice, payment, cursor)
MIN_LEDGER_ROWS: Final[int] = 200  # default lower bound of EpisodeConfig.n_ledger_rows_range
MAX_LEDGER_ROWS: Final[int] = 2000  # default upper bound of EpisodeConfig.n_ledger_rows_range
INJECTOR_MAX_RETRY_ATTEMPTS: Final[int] = 10  # inject retries before build_episode gives up
INJECTOR_DUPLICATE_DATE_SHIFT_MAX_DAYS: Final[int] = 10  # duplicate issued 0..10 days later
TIME_BUDGET_SECONDS: Final[float] = 60.0  # wall-clock safety timeout per episode (jury C18)

# ---------------------------------------------------------------- 3. Tolerance defaults
DEFAULT_TOLERANCE_FRAC: Final[Decimal] = Decimal("0.001")  # 0.1 % relative tolerance on amounts
MIN_TOLERANCE_MINOR: Final[int] = 1  # absolute floor of the tolerance, in minor units

# ---------------------------------------------------------------- 4. Benford (blueprint 4.3)
BENFORD_CHI_SQUARED_CRITICAL: Final[Decimal] = Decimal("15.507")  # p = 0.05, 8 degrees of freedom
BENFORD_MIN_SAMPLE_SIZE: Final[int] = 100  # below this the Benford signal is not reported
# Sampler shape: generated amounts live in decades 10**LO .. 10**HI major units (10 .. 100 000),
# with a Pareto(alpha) upper tail. float on purpose: these are arguments of math.pow / math.log10
# inside the sampler and are never stored — every emitted amount is an int in minor units.
BENFORD_LO_EXPONENT: Final[float] = 1.0
BENFORD_HI_EXPONENT: Final[float] = 5.0
BENFORD_PARETO_ALPHA: Final[float] = 1.5

# ---------------------------------------------------------------- 5. Reward (blueprint 7.4)
# reward = clip(F1_COEF * F1 - F1_BIAS - STEP_COST * inspection_steps, CLIP_MIN, CLIP_MAX)
REWARD_F1_COEF: Final[Decimal] = Decimal("2")
REWARD_F1_BIAS: Final[Decimal] = Decimal("1")
REWARD_STEP_COST: Final[Decimal] = Decimal("0.002")
REWARD_CLIP_MIN: Final[float] = -1.0  # reward is the one float in the system (not money)
REWARD_CLIP_MAX: Final[float] = 1.0

# ---------------------------------------------------------------- 6. Issue types
IssueType: TypeAlias = Literal[
    "duplicate_invoice",
    "payment_allocation",
    "fx_error",
    "period_misclassification",
]
ISSUE_TYPES: Final[tuple[IssueType, ...]] = (
    "duplicate_invoice",
    "payment_allocation",
    "fx_error",
    "period_misclassification",
)

# ---------------------------------------------------------------- 7. Error codes
# The only error payload the environment ever emits is {"error": {"code": <ErrorCode>}}.
ErrorCode: TypeAlias = Literal[
    "INVALID_ACTION",
    "INVALID_TABLE",
    "NOT_FOUND",
    "INVALID_CURSOR",
    "CALC_ERROR",
    "EPISODE_OVER",
    "INVALID_SUBMISSION",
]
ERROR_CODES: Final[tuple[ErrorCode, ...]] = (
    "INVALID_ACTION",
    "INVALID_TABLE",
    "NOT_FOUND",
    "INVALID_CURSOR",
    "CALC_ERROR",
    "EPISODE_OVER",
    "INVALID_SUBMISSION",
)

# ---------------------------------------------------------------- 8. Tool names
ToolName: TypeAlias = Literal["list_records", "get_record", "get_document", "calculator", "submit"]
TOOL_NAMES: Final[tuple[ToolName, ...]] = (
    "list_records",
    "get_record",
    "get_document",
    "calculator",
    "submit",
)

# ---------------------------------------------------------------- 9. Table names (whitelist)
TableName: TypeAlias = Literal["ledger", "invoices", "payments", "fx"]
TABLE_NAMES: Final[tuple[TableName, ...]] = ("ledger", "invoices", "payments", "fx")

# ---------------------------------------------------------------- 10. Regex patterns
# Used as Pydantic Field(pattern=...) / StringConstraints(pattern=...) in fintrace.models.
# Pydantic applies patterns with search semantics, so every pattern is anchored with ^ and $.
# ASCII classes ([0-9]) only: \d in the Rust regex engine also matches non-ASCII digits.
OPAQUE_ID_PATTERN: Final[str] = r"^[A-Za-z0-9_\-:.]+$"
CURRENCY_ISO4217_PATTERN: Final[str] = r"^[A-Z]{3}$"
# Generator note: derive the id from a keyed/salted hash of the seed, never a bare hash prefix —
# the id is public and a bare prefix of a small integer seed is brute-forceable.
EPISODE_ID_PATTERN: Final[str] = r"^ep_[0-9a-f]{10,16}$"
ISSUE_ID_PATTERN: Final[str] = r"^ISS-[0-9]{2,4}$"
# Fixed offset ("+05:00", hours 00-23, minutes 00-59), "UTC", or an IANA zone name
# ("Asia/Almaty", "Etc/GMT+5", "America/Argentina/Buenos_Aires").
TIMEZONE_PATTERN: Final[str] = (
    r"^(?:[+-](?:[01][0-9]|2[0-3]):[0-5][0-9]|UTC|[A-Za-z_]+(?:/[A-Za-z0-9_+\-]+)+)$"
)

# ---------------------------------------------------------------- 11. Reporting / observation
REWARD_INTERMEDIATE: Final[float] = 0.0  # reward of every non-terminal step
RoundingMode: TypeAlias = Literal["ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_DOWN"]
ROUNDING_MODES: Final[tuple[RoundingMode, ...]] = (
    "ROUND_HALF_EVEN",
    "ROUND_HALF_UP",
    "ROUND_DOWN",
)
DEFAULT_ROUNDING_MODE: Final[RoundingMode] = "ROUND_HALF_EVEN"
MINOR_UNIT_DIGITS_DEFAULT: Final[int] = 2  # USD / EUR / KZT

# ---------------------------------------------------------------- 11b. OCEL 2.0 export
# Object types of the exported log (display order); event types are TOOL_NAMES. Trajectories carry
# no clock, so event times are synthetic: OCEL_TIME_BASE + step * OCEL_STEP_SECONDS (offset-aware,
# deterministic, never the wall clock). Object attribute times use the pm4py sentinel for initial
# values.
OCEL_OBJECT_TYPES: Final[tuple[str, ...]] = ("episode", "invoice", "payment", "fx_rate", "finding")
OCEL_TIME_BASE: Final[str] = "2026-04-01T09:00:00+05:00"
OCEL_STEP_SECONDS: Final[int] = 1
OCEL_OBJECT_TIME: Final[str] = "1970-01-01T00:00:00Z"

# ---------------------------------------------------------------- 12. Demo / jury smoke config
# One episode built in memory by `python -m fintrace` (the Docker ENTRYPOINT) and by the jury
# protocol smoke run: dev-like sizes, one injected issue of each type among four decoy patterns.
# A plain dict, not an EpisodeConfig: models.py imports this module, not the other way round.
DEMO_SEED: Final[int] = 10000  # feasible for DEMO_TEAM_CONFIG; the demo output is byte-stable
DEMO_TEAM_CONFIG: Final[dict[str, Any]] = {
    "n_ledger_rows_range": [500, 1500],
    "n_documents_range": [30, 150],
    "n_entities_range": [8, 20],
    "errors": {
        "duplicate_invoice": {"count": 1},
        "payment_allocation": {"count": 1},
        "fx_error": {"count": 1},
        "period_misclassification": {"count": 1},
    },
    "decoys": {
        "repeated_payment_amounts": 1,
        "split_payments": 1,
        "refunds": 1,
        "similar_invoices": 1,
    },
}

# ---------------------------------------------------------------- 13. Paths (relative to repo root)
CONFIG_DIR: Final[Path] = Path("config")
DATA_DIR: Final[Path] = Path("data")
SCHEMAS_DIR: Final[Path] = Path("schemas")
TRAJECTORIES_DIR: Final[Path] = Path("trajectories")

__all__ = [
    "BENFORD_CHI_SQUARED_CRITICAL",
    "BENFORD_HI_EXPONENT",
    "BENFORD_LO_EXPONENT",
    "BENFORD_MIN_SAMPLE_SIZE",
    "BENFORD_PARETO_ALPHA",
    "CONFIG_DIR",
    "CURRENCY_ISO4217_PATTERN",
    "DATA_DIR",
    "DEFAULT_ROUNDING_MODE",
    "DEFAULT_TOLERANCE_FRAC",
    "DEMO_SEED",
    "DEMO_TEAM_CONFIG",
    "ENV_VERSION",
    "EPISODE_ID_PATTERN",
    "ERROR_CODES",
    "INJECTOR_DUPLICATE_DATE_SHIFT_MAX_DAYS",
    "INJECTOR_MAX_RETRY_ATTEMPTS",
    "ISSUE_ID_PATTERN",
    "ISSUE_TYPES",
    "MAX_AFFECTED_IDS_PER_FINDING",
    "MAX_CALCULATOR_EXPR_CHARS",
    "MAX_EVIDENCE_IDS_PER_FINDING",
    "MAX_EXPLANATION_CHARS",
    "MAX_FINDINGS",
    "MAX_ID_CHARS",
    "MAX_LEDGER_ROWS",
    "MAX_PAGE_SIZE",
    "MAX_STEPS",
    "MINOR_UNIT_DIGITS_DEFAULT",
    "MIN_LEDGER_ROWS",
    "MIN_TOLERANCE_MINOR",
    "OCEL_OBJECT_TIME",
    "OCEL_OBJECT_TYPES",
    "OCEL_STEP_SECONDS",
    "OCEL_TIME_BASE",
    "OPAQUE_ID_PATTERN",
    "REWARD_CLIP_MAX",
    "REWARD_CLIP_MIN",
    "REWARD_F1_BIAS",
    "REWARD_F1_COEF",
    "REWARD_INTERMEDIATE",
    "REWARD_STEP_COST",
    "ROUNDING_MODES",
    "SCHEMAS_DIR",
    "SCHEMA_VERSION",
    "TABLE_NAMES",
    "TIMEZONE_PATTERN",
    "TIME_BUDGET_SECONDS",
    "TOOL_NAMES",
    "TRAJECTORIES_DIR",
    "ErrorCode",
    "IssueType",
    "RoundingMode",
    "TableName",
    "ToolName",
]
