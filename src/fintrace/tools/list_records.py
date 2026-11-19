"""list_records: stable-sorted, paginated, compact views of the four public tables.

Also hosts the per-episode view cache shared by get_record / get_document: sorting and indexing
happen once per episode (keyed by episode_id, cleared by ``FintraceEnv.reset``). Only
``PublicEpisode`` is ever read here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from fintrace.environment.observations import build_error_result, build_ok_result
from fintrace.models import (
    EpisodeState,
    FXRate,
    Invoice,
    JsonObject,
    LedgerRow,
    ListRecordsArgs,
    Payment,
    PublicEpisode,
    TableName,
    ToolResult,
)

Record = LedgerRow | Invoice | Payment | FXRate

# Topline fields returned by list_records; details (debit/credit, journal_id, line_items,
# invoice_refs) cost a separate get_record / get_document step.
_COMPACT_FIELDS: Final[dict[TableName, tuple[str, ...]]] = {
    "ledger": (
        "row_id", "timestamp", "account", "currency", "entity_id", "invoice_id", "payment_id",
    ),
    "invoices": (
        "doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor",
    ),
    "payments": ("doc_id", "payment_id", "timestamp", "currency", "amount_minor"),
    "fx": ("date", "source_currency", "target_currency", "rate"),
}  # fmt: skip
_CURSOR_RE: Final[re.Pattern[str]] = re.compile(r"^c_(0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class EpisodeView:
    """Sorted rows and id indexes of one PublicEpisode; built once, never mutated."""

    episode_id: str
    rows: dict[TableName, tuple[Record, ...]]
    by_id: dict[TableName, dict[str, Record]]
    documents: dict[str, Invoice | Payment]


_SORTED_CACHE: dict[str, EpisodeView] = {}  # episode_id -> view; FintraceEnv.reset clears it
_CACHE_LIMIT: Final[int] = 8  # envs discarded without reset() must not grow the cache unbounded


def fx_id(rate: FXRate) -> str:
    """Synthetic id for an FX row (the model has none): ``<source>-<target>-<date>``."""
    return f"{rate.source_currency}-{rate.target_currency}-{rate.date}"


def _build_view(episode: PublicEpisode) -> EpisodeView:
    ledger = tuple(sorted(episode.ledger, key=lambda r: (r.timestamp, r.row_id)))
    invoices = tuple(sorted(episode.invoices, key=lambda i: (i.issued_at, i.invoice_id)))
    payments = tuple(sorted(episode.payments, key=lambda p: (p.timestamp, p.payment_id)))
    fx = tuple(sorted(episode.fx, key=lambda f: (f.date, f.source_currency, f.target_currency)))
    rows: dict[TableName, tuple[Record, ...]] = {
        "ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx,
    }  # fmt: skip
    by_id: dict[TableName, dict[str, Record]] = {
        "ledger": {r.row_id: r for r in ledger},
        "invoices": {i.invoice_id: i for i in invoices},
        "payments": {p.payment_id: p for p in payments},
        "fx": {fx_id(f): f for f in fx},
    }
    documents: dict[str, Invoice | Payment] = {i.doc_id: i for i in invoices}
    documents.update({p.doc_id: p for p in payments})
    return EpisodeView(episode.manifest.episode_id, rows, by_id, documents)


def episode_view(episode: PublicEpisode) -> EpisodeView:
    """Cached sorted/indexed view of ``episode`` (built on first use)."""
    view = _SORTED_CACHE.get(episode.manifest.episode_id)
    if view is None:
        view = _build_view(episode)
        if len(_SORTED_CACHE) >= _CACHE_LIMIT:
            _SORTED_CACHE.pop(next(iter(_SORTED_CACHE)))  # oldest insertion
        _SORTED_CACHE[episode.manifest.episode_id] = view
    return view


def clear_cache(episode_id: str | None) -> None:
    """Drop the cached view of one episode (called by the environment on reset)."""
    if episode_id is not None:
        _SORTED_CACHE.pop(episode_id, None)


def parse_cursor(cursor: str | None) -> int | None:
    """``None`` -> offset 0; ``c_<offset>`` -> offset; anything else -> None (invalid)."""
    if cursor is None:
        return 0
    match = _CURSOR_RE.fullmatch(cursor)
    return int(match.group(1)) if match else None


def compact_record(table: TableName, record: Record) -> JsonObject:
    """Topline JSON view of one record (FX rows gain a synthetic ``fx_id``)."""
    data = record.model_dump(mode="json")
    compact: JsonObject = {field: data[field] for field in _COMPACT_FIELDS[table]}
    if isinstance(record, FXRate):
        return {"fx_id": fx_id(record), **compact}
    return compact


def handle_list_records(
    args: ListRecordsArgs, state: EpisodeState, episode: PublicEpisode
) -> ToolResult:
    """At most ``limit`` (<= MAX_PAGE_SIZE) compact records from a stable-sorted table.

    Cursor ``c_<offset>``; a missing cursor starts at 0; an offset past the end (or malformed)
    is INVALID_CURSOR. Result: ``{records, next_cursor, records_total}``.
    """
    rows = episode_view(episode).rows[args.table]
    total = len(rows)
    offset = parse_cursor(args.cursor)
    if offset is None or (offset >= total and not (total == 0 and offset == 0)):
        return build_error_result("list_records", "INVALID_CURSOR", "cursor out of range", state)
    page = rows[offset : offset + args.limit]
    end = offset + len(page)
    result: JsonObject = {
        "records": [compact_record(args.table, record) for record in page],
        "next_cursor": f"c_{end}" if end < total else None,
        "records_total": total,
    }
    return build_ok_result("list_records", result, state)


__all__ = [
    "EpisodeView",
    "Record",
    "clear_cache",
    "compact_record",
    "episode_view",
    "fx_id",
    "handle_list_records",
    "parse_cursor",
]
