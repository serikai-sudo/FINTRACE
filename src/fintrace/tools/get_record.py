"""get_record: one full record by (table, id) plus the ids it references."""

from __future__ import annotations

from fintrace.environment.observations import build_error_result, build_ok_result
from fintrace.models import (
    EpisodeState,
    FXRate,
    GetRecordArgs,
    Invoice,
    JsonObject,
    LedgerRow,
    Payment,
    PublicEpisode,
    ToolResult,
)

from .list_records import Record, episode_view, fx_id


def related_ids(record: Record) -> list[str]:
    """Ids a record points at, in field order, nulls dropped (FX rows reference nothing)."""
    if isinstance(record, LedgerRow):
        candidates = (record.entity_id, record.invoice_id, record.payment_id, record.journal_id)
    elif isinstance(record, Invoice):
        candidates = (record.entity_id,)
    elif isinstance(record, Payment):
        candidates = tuple(ref.invoice_id for ref in record.invoice_refs)
    else:
        candidates = ()
    return [identifier for identifier in candidates if identifier is not None]


def handle_get_record(
    args: GetRecordArgs, state: EpisodeState, episode: PublicEpisode
) -> ToolResult:
    """Full record (``model_dump(mode="json")``) and ``related_ids``; unknown id -> NOT_FOUND."""
    record = episode_view(episode).by_id[args.table].get(args.id)
    if record is None:
        return build_error_result("get_record", "NOT_FOUND", "unknown record id", state)
    data: JsonObject = record.model_dump(mode="json")
    if isinstance(record, FXRate):
        data = {"fx_id": fx_id(record), **data}
    result: JsonObject = {"table": args.table, "record": data, "related_ids": related_ids(record)}
    return build_ok_result("get_record", result, state)


__all__ = ["handle_get_record", "related_ids"]
