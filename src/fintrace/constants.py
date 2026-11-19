"""Project-wide constants: the frozen contract every FINTRACE module imports. No functions here."""

from decimal import Decimal
from pathlib import Path
from typing import Final, Literal, TypeAlias

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
MAX_ID_CHARS: Final[int] = 64  # any opaque identifier (row, document, invoice, payment, entity)

# ---------------------------------------------------------------- 3. Tolerance defaults
DEFAULT_TOLERANCE_FRAC: Final[Decimal] = Decimal("0.001")  # 0.1 % relative tolerance on amounts
MIN_TOLERANCE_MINOR: Final[int] = 1  # absolute floor of the tolerance, in minor units

# ---------------------------------------------------------------- 4. Benford (blueprint 4.3)
BENFORD_CHI_SQUARED_CRITICAL: Final[Decimal] = Decimal("15.507")  # p = 0.05, 8 degrees of freedom
BENFORD_MIN_SAMPLE_SIZE: Final[int] = 100  # below this the Benford signal is not reported

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
OPAQUE_ID_PATTERN: Final[str] = r"^[A-Za-z0-9_\-:.]+$"
CURRENCY_ISO4217_PATTERN: Final[str] = r"^[A-Z]{3}$"
EPISODE_ID_PATTERN: Final[str] = r"^ep_[0-9a-f]{10,16}$"
ISSUE_ID_PATTERN: Final[str] = r"^ISS-\d{2,4}$"
# Fixed offset ("+05:00") or IANA zone name ("Asia/Almaty").
TIMEZONE_PATTERN: Final[str] = r"^[+-]\d{2}:\d{2}$|^[A-Za-z_]+/[A-Za-z_]+$"

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

# ---------------------------------------------------------------- 12. Paths (relative to repo root)
CONFIG_DIR: Final[Path] = Path("config")
DATA_DIR: Final[Path] = Path("data")
SCHEMAS_DIR: Final[Path] = Path("schemas")
TRAJECTORIES_DIR: Final[Path] = Path("trajectories")

__all__ = [
    "BENFORD_CHI_SQUARED_CRITICAL",
    "BENFORD_MIN_SAMPLE_SIZE",
    "CONFIG_DIR",
    "CURRENCY_ISO4217_PATTERN",
    "DATA_DIR",
    "DEFAULT_ROUNDING_MODE",
    "DEFAULT_TOLERANCE_FRAC",
    "ENV_VERSION",
    "EPISODE_ID_PATTERN",
    "ERROR_CODES",
    "ISSUE_ID_PATTERN",
    "ISSUE_TYPES",
    "MAX_AFFECTED_IDS_PER_FINDING",
    "MAX_CALCULATOR_EXPR_CHARS",
    "MAX_EVIDENCE_IDS_PER_FINDING",
    "MAX_EXPLANATION_CHARS",
    "MAX_FINDINGS",
    "MAX_ID_CHARS",
    "MAX_PAGE_SIZE",
    "MAX_STEPS",
    "MINOR_UNIT_DIGITS_DEFAULT",
    "MIN_TOLERANCE_MINOR",
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
    "TOOL_NAMES",
    "TRAJECTORIES_DIR",
    "ErrorCode",
    "IssueType",
    "RoundingMode",
    "TableName",
    "ToolName",
]
