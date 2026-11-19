"""Tool registry: argument schemas, handlers, action parsing and dispatch.

Handlers are pure functions ``(args, state, episode) -> ToolResult`` over the PublicEpisode only.
The registry never imports fintrace.evaluator; the matcher reaches submit through the
environment (dependency injection, step 6).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from fintrace.models import (
    Action,
    CalculatorArgs,
    EpisodeState,
    ErrorCode,
    GetDocumentArgs,
    GetRecordArgs,
    ListRecordsArgs,
    PublicEpisode,
    Submission,
    ToolName,
    ToolResult,
)

from .calculator import handle_calculator
from .get_document import handle_get_document
from .get_record import handle_get_record
from .list_records import handle_list_records
from .submit import handle_submit

ToolHandler = Callable[[Any, EpisodeState, PublicEpisode], ToolResult]

TOOL_SCHEMAS: Final[dict[ToolName, type[BaseModel]]] = {
    "list_records": ListRecordsArgs,
    "get_record": GetRecordArgs,
    "get_document": GetDocumentArgs,
    "calculator": CalculatorArgs,
    "submit": Submission,
}

# (tool, argument field) -> error code when that field alone fails a value check
_FIELD_ERROR_CODES: Final[dict[tuple[ToolName, str], ErrorCode]] = {
    ("list_records", "table"): "INVALID_TABLE",
    ("get_record", "table"): "INVALID_TABLE",
    ("list_records", "cursor"): "INVALID_CURSOR",
    ("calculator", "expression"): "CALC_ERROR",
}
_STRUCTURAL_ERRORS: Final[frozenset[str]] = frozenset({"extra_forbidden", "missing"})

TOOL_HANDLERS: Final[dict[ToolName, ToolHandler]] = {
    "list_records": handle_list_records,
    "get_record": handle_get_record,
    "get_document": handle_get_document,
    "calculator": handle_calculator,
    "submit": handle_submit,
}


def _argument_error_code(tool: ToolName, exc: ValidationError) -> ErrorCode:
    """Value errors confined to one known field get that field's code; the rest INVALID_ACTION.

    ``table`` -> INVALID_TABLE, ``cursor`` -> INVALID_CURSOR, ``expression`` -> CALC_ERROR (an
    over-long expression is a calculator error, not a protocol error). Missing or unexpected
    fields are protocol errors regardless of their name.
    """
    errors = exc.errors()
    fields = {error["loc"][0] for error in errors if error["loc"]}
    if len(fields) != 1 or any(error["type"] in _STRUCTURAL_ERRORS for error in errors):
        return "INVALID_ACTION"
    field = next(iter(fields))
    return _FIELD_ERROR_CODES.get((tool, str(field)), "INVALID_ACTION")


def parse_action(action: object) -> tuple[ToolName, BaseModel] | ErrorCode:
    """Validate a raw action into ``(tool, typed args)`` or return an error code. Never raises."""
    try:
        envelope = Action.model_validate(action)
    except ValidationError:
        return "INVALID_ACTION"
    try:
        args = TOOL_SCHEMAS[envelope.tool].model_validate(envelope.args)
    except ValidationError as exc:
        return _argument_error_code(envelope.tool, exc)
    return envelope.tool, args


def dispatch(
    tool: ToolName, args: BaseModel, state: EpisodeState, episode: PublicEpisode
) -> ToolResult:
    """Run the handler for ``tool``; the caller has already validated ``args``."""
    return TOOL_HANDLERS[tool](args, state, episode)


__all__ = ["TOOL_HANDLERS", "TOOL_SCHEMAS", "ToolHandler", "dispatch", "parse_action"]
