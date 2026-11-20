"""Torch-free featurizer for the behaviour-cloning proof: trajectory prefix -> feature vector.

For step ``t`` of a recorded trajectory the features are a deterministic function of the
observations of steps ``1 .. t-1`` (the state is updated *after* the vector is emitted), so a
policy that only sees the public tool results can be trained to predict what the expert did
next. Labels: the tool the expert called (``TOOLS``, 5 classes) and a finer target that also
names the table of a ``list_records`` call (``FINE_CLASSES``, 8 classes). Pure Python, no torch,
no numpy: the main test suite imports this module to pin the feature layout and its determinism.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from fintrace.constants import MAX_STEPS, TABLE_NAMES, TOOL_NAMES
from fintrace.exporters.ocel import read_trajectory
from fintrace.models import JsonObject

TOOLS: Final[tuple[str, ...]] = tuple(TOOL_NAMES)
TABLES: Final[tuple[str, ...]] = tuple(TABLE_NAMES)
FINE_CLASSES: Final[tuple[str, ...]] = (
    *(f"list_records:{table}" for table in TABLES),
    "get_record",
    "get_document",
    "calculator",
    "submit",
)
# scales that map raw counters into roughly [0, 1] (the MLP standardises again on train statistics)
PAGE_SCALE: Final[int] = 20
RECORDS_SCALE: Final[int] = 1000
GET_RECORD_SCALE: Final[int] = 20
GET_DOCUMENT_SCALE: Final[int] = 60
CALCULATOR_SCALE: Final[int] = 5
ERROR_SCALE: Final[int] = 5
DUPLICATE_SCALE: Final[int] = 5
FOREIGN_SCALE: Final[int] = 20
UNREAD_SCALE: Final[int] = 70
TABLE_FEATURES: Final[tuple[str, ...]] = ("pages", "seen_fraction", "listing_done", "records_total")
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "step_fraction",
    "remaining_fraction",
    *(f"{table}_{name}" for table in TABLES for name in TABLE_FEATURES),
    "n_get_record",
    "n_get_document",
    "n_calculator",
    "n_errors",
    *(f"last_tool_{tool}" for tool in TOOLS),
    *(f"last_table_{table}" for table in TABLES),
    "last_ok",
    "last_document_invoice",
    "last_document_payment",
    "duplicate_candidate_groups",
    "foreign_payments_seen",
    "unread_payment_documents",
)
N_FEATURES: Final[int] = len(FEATURE_NAMES)


def tool_label(action: object) -> int:
    """Index into ``TOOLS`` of the expert's action (``ValueError`` for a non-tool action)."""
    tool = action.get("tool") if isinstance(action, dict) else None
    if tool not in TOOLS:
        raise ValueError(f"not a tool action: {action!r}")
    return TOOLS.index(str(tool))


def fine_label(action: object) -> int:
    """Index into ``FINE_CLASSES``: the tool, with the table for ``list_records``."""
    tool = TOOLS[tool_label(action)]
    if tool != "list_records":
        return FINE_CLASSES.index(tool)
    args = action.get("args") if isinstance(action, dict) else None
    table = args.get("table") if isinstance(args, dict) else None
    if table not in TABLES:
        raise ValueError(f"list_records without a known table: {action!r}")
    return FINE_CLASSES.index(f"list_records:{table}")


class _PrefixState:
    """What a public-only observer knows after the steps seen so far."""

    def __init__(self) -> None:
        self.pages: dict[str, int] = dict.fromkeys(TABLES, 0)
        self.seen: dict[str, int] = dict.fromkeys(TABLES, 0)
        self.total: dict[str, int] = dict.fromkeys(TABLES, 0)
        self.done: dict[str, int] = dict.fromkeys(TABLES, 0)
        self.n_get_record = 0
        self.n_get_document = 0
        self.n_calculator = 0
        self.n_errors = 0
        self.last_tool: str | None = None
        self.last_table: str | None = None
        self.last_ok = 1
        self.last_kind: str | None = None
        self.invoice_groups: Counter[tuple[str, int, str]] = Counter()
        self.invoice_currencies: Counter[str] = Counter()
        self.payments_seen = 0
        self.foreign_payments = 0
        self.documents_read: set[str] = set()

    def features(self, step: int, remaining: int) -> list[float]:
        vector: list[float] = [(step - 1) / MAX_STEPS, remaining / MAX_STEPS]
        for table in TABLES:
            total = self.total[table]
            vector += [
                min(self.pages[table], PAGE_SCALE) / PAGE_SCALE,
                (self.seen[table] / total) if total else 0.0,
                float(self.done[table]),
                min(total, RECORDS_SCALE) / RECORDS_SCALE,
            ]
        vector += [
            min(self.n_get_record, GET_RECORD_SCALE) / GET_RECORD_SCALE,
            min(self.n_get_document, GET_DOCUMENT_SCALE) / GET_DOCUMENT_SCALE,
            min(self.n_calculator, CALCULATOR_SCALE) / CALCULATOR_SCALE,
            min(self.n_errors, ERROR_SCALE) / ERROR_SCALE,
        ]
        vector += [1.0 if self.last_tool == tool else 0.0 for tool in TOOLS]
        vector += [1.0 if self.last_table == table else 0.0 for table in TABLES]
        vector += [
            float(self.last_ok),
            1.0 if self.last_kind == "invoice" else 0.0,
            1.0 if self.last_kind == "payment" else 0.0,
        ]
        duplicates = sum(1 for count in self.invoice_groups.values() if count >= 2)
        unread = max(0, self.payments_seen - len(self.documents_read))
        vector += [
            min(duplicates, DUPLICATE_SCALE) / DUPLICATE_SCALE,
            min(self.foreign_payments, FOREIGN_SCALE) / FOREIGN_SCALE,
            min(unread, UNREAD_SCALE) / UNREAD_SCALE,
        ]
        return vector

    def _dominant_currency(self) -> str | None:
        if not self.invoice_currencies:
            return None
        return min(self.invoice_currencies, key=lambda c: (-self.invoice_currencies[c], c))

    def update(self, entry: JsonObject) -> None:
        action = entry.get("action")
        args = action.get("args") if isinstance(action, dict) else None
        args = args if isinstance(args, dict) else {}
        observation = entry["observation"]
        ok = observation.get("ok") is True
        result = observation.get("result") if ok else None
        result = result if isinstance(result, dict) else {}
        tool = action.get("tool") if isinstance(action, dict) else None
        table = args.get("table") if args.get("table") in TABLES else None
        if tool == "list_records" and ok and table is not None:
            records = result.get("records")
            records = records if isinstance(records, list) else []
            self.pages[table] += 1
            self.seen[table] += len(records)
            total = result.get("records_total")
            if isinstance(total, int) and not isinstance(total, bool):
                self.total[table] = total
            if result.get("next_cursor") is None:
                self.done[table] = 1
            if table == "invoices":
                for record in records:
                    if isinstance(record, dict):
                        key = (
                            str(record.get("entity_id")),
                            int(record.get("total_minor") or 0),
                            str(record.get("currency")),
                        )
                        self.invoice_groups[key] += 1
                        self.invoice_currencies[str(record.get("currency"))] += 1
            elif table == "payments":
                dominant = self._dominant_currency()
                for record in records:
                    if isinstance(record, dict):
                        self.payments_seen += 1
                        if dominant is not None and record.get("currency") != dominant:
                            self.foreign_payments += 1
        elif tool == "get_record":
            self.n_get_record += 1
        elif tool == "get_document":
            self.n_get_document += 1
            if ok:
                kind = result.get("kind")
                self.last_kind = str(kind) if kind in ("invoice", "payment") else self.last_kind
                self.documents_read.add(str(args.get("doc_id")))
        elif tool == "calculator":
            self.n_calculator += 1
        if not ok:
            self.n_errors += 1
        self.last_tool = str(tool) if tool in TOOLS else None
        self.last_table = table
        self.last_ok = 1 if ok else 0


def featurize_steps(
    steps: Sequence[JsonObject],
) -> tuple[list[list[float]], list[int], list[int]]:
    """``(X, y_tool, y_fine)`` for one trajectory: one row per step, labels = the expert's action."""
    state = _PrefixState()
    rows: list[list[float]] = []
    tools: list[int] = []
    fine: list[int] = []
    for entry in steps:
        remaining = int(entry["observation"]["steps_remaining"]) + 1  # before this step was taken
        rows.append(state.features(int(entry["step"]), remaining))
        tools.append(tool_label(entry.get("action")))
        fine.append(fine_label(entry.get("action")))
        state.update(entry)
    return rows, tools, fine


def featurize_file(path: Path) -> tuple[list[list[float]], list[int], list[int]]:
    """``featurize_steps`` over a trajectory JSONL file (validated by ``read_trajectory``)."""
    return featurize_steps(read_trajectory(path))


__all__ = [
    "FEATURE_NAMES",
    "FINE_CLASSES",
    "N_FEATURES",
    "TABLES",
    "TOOLS",
    "featurize_file",
    "featurize_steps",
    "fine_label",
    "tool_label",
]
