"""Policy protocol and action helpers shared by every baseline and attack agent.

Agents see only the public surface: observations (``ToolResult`` and the reset observation,
typed structurally below so this package depends on ``fintrace.models`` alone) and the five
tool actions. Nothing under ``fintrace.agents`` may depend on the scorer, the reward formula,
the env loop or the generator; ``tests/contract/test_runner.py`` enforces it with an AST scan.
"""

from __future__ import annotations

from typing import Final, Protocol

from fintrace.constants import MAX_FINDINGS, MAX_PAGE_SIZE, TableName
from fintrace.models import JsonObject

BASELINE_NAMES: Final[tuple[str, ...]] = ("random", "rule_based", "mark_everything", "oracle")
THOUGHT_MAX_CHARS: Final[int] = 500  # Action.thought max_length in fintrace.models
_ELLIPSIS: Final[str] = "..."


class Observation(Protocol):
    """What ``reset`` and ``step`` hand to a policy: the public fields of both observation types."""

    ok: bool
    tool: str
    result: JsonObject | None
    steps_used: int
    steps_remaining: int


class Policy(Protocol):
    """A baseline or attack agent: ``reset`` before each episode, ``act`` once per step."""

    name: str

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> JsonObject: ...


def clip_thought(text: str) -> str:
    """Collapse whitespace and cut to the ``Action.thought`` limit (never an invalid action)."""
    text = " ".join(text.split())
    if len(text) <= THOUGHT_MAX_CHARS:
        return text
    return text[: THOUGHT_MAX_CHARS - len(_ELLIPSIS)] + _ELLIPSIS


def episode_id_of(observation: Observation) -> str | None:
    """The episode id carried by the reset observation (``result.manifest.episode_id``)."""
    result = observation.result
    if not isinstance(result, dict):
        return None
    manifest = result.get("manifest")
    if not isinstance(manifest, dict):
        return None
    episode_id = manifest.get("episode_id")
    return episode_id if isinstance(episode_id, str) else None


def records_of(observation: Observation) -> list[JsonObject]:
    """The page of a successful ``list_records`` observation, else an empty list."""
    if not observation.ok or observation.tool != "list_records" or observation.result is None:
        return []
    records = observation.result.get("records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def records_total_of(observation: Observation) -> int | None:
    """``records_total`` of a successful ``list_records`` observation."""
    if not observation.ok or observation.result is None:
        return None
    total = observation.result.get("records_total")
    return total if isinstance(total, int) else None


def next_cursor_of(observation: Observation) -> str | None:
    """The next page cursor of a successful ``list_records`` observation, ``None`` at the end."""
    if not observation.ok or observation.result is None:
        return None
    cursor = observation.result.get("next_cursor")
    return cursor if isinstance(cursor, str) else None


def list_action(
    table: TableName, cursor: str | None = None, limit: int = MAX_PAGE_SIZE, *, thought: str
) -> JsonObject:
    """``list_records`` action."""
    args: JsonObject = {"table": table, "cursor": cursor, "limit": limit}
    return {"tool": "list_records", "args": args, "thought": clip_thought(thought)}


def record_action(table: TableName, record_id: str, *, thought: str) -> JsonObject:
    """``get_record`` action."""
    args: JsonObject = {"table": table, "id": record_id}
    return {"tool": "get_record", "args": args, "thought": clip_thought(thought)}


def document_action(doc_id: str, *, thought: str) -> JsonObject:
    """``get_document`` action."""
    args: JsonObject = {"doc_id": doc_id}
    return {"tool": "get_document", "args": args, "thought": clip_thought(thought)}


def calculator_action(expression: str, *, thought: str) -> JsonObject:
    """``calculator`` action."""
    args: JsonObject = {"expression": expression}
    return {"tool": "calculator", "args": args, "thought": clip_thought(thought)}


def submit_action(episode_id: str, findings: list[JsonObject], *, thought: str) -> JsonObject:
    """``submit`` action; findings beyond ``MAX_FINDINGS`` are dropped (never an invalid action)."""
    args: JsonObject = {"episode_id": episode_id, "findings": list(findings[:MAX_FINDINGS])}
    return {"tool": "submit", "args": args, "thought": clip_thought(thought)}


__all__ = [
    "BASELINE_NAMES",
    "THOUGHT_MAX_CHARS",
    "Observation",
    "Policy",
    "calculator_action",
    "clip_thought",
    "document_action",
    "episode_id_of",
    "list_action",
    "next_cursor_of",
    "record_action",
    "records_of",
    "records_total_of",
    "submit_action",
]
