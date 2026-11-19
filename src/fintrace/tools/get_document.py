"""get_document: the full invoice or payment behind a doc_id."""

from __future__ import annotations

from fintrace.environment.observations import build_error_result, build_ok_result
from fintrace.models import (
    EpisodeState,
    GetDocumentArgs,
    Invoice,
    JsonObject,
    PublicEpisode,
    ToolResult,
)

from .list_records import episode_view


def handle_get_document(
    args: GetDocumentArgs, state: EpisodeState, episode: PublicEpisode
) -> ToolResult:
    """``{kind: "invoice" | "payment", document: {...}}``; unknown doc_id -> NOT_FOUND."""
    document = episode_view(episode).documents.get(args.doc_id)
    if document is None:
        return build_error_result("get_document", "NOT_FOUND", "unknown document id", state)
    kind = "invoice" if isinstance(document, Invoice) else "payment"
    result: JsonObject = {"kind": kind, "document": document.model_dump(mode="json")}
    return build_ok_result("get_document", result, state)


__all__ = ["handle_get_document"]
