"""Injector protocol and the helpers every concrete injector shares.

Injector contract (MUST hold for every concrete injector):
  1. it reads only the public episode plus provenance / truth (to avoid colliding with earlier
     injections and with the recurring decoys);
  2. it returns a new PrivateEpisode whose public files are modified, together with the Issue it
     created and a ProvenanceEntry (build.py appends both);
  3. ``independent_check(public, issue)`` recomputes the discrepancy from public files alone and
     must agree with ``issue.expected_amount_minor`` exactly, else raises ConsistencyError;
  4. it never touches records that belong to issues already present.
A concrete injector is a module exposing ``issue_type``, ``inject`` and ``independent_check``
(duck-typed against ``Injector``); nothing here imports the scorer or the environment.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import Final, Protocol

from fintrace.generator.company import EntityKind
from fintrace.models import (
    InjectorParams,
    Invoice,
    Issue,
    IssueType,
    PrivateEpisode,
    ProvenanceEntry,
    PublicEpisode,
)

INJECTOR_ORDER: Final[tuple[IssueType, ...]] = (
    "duplicate_invoice",
    "payment_allocation",
    "fx_error",
    "period_misclassification",
)
_ISSUE_ID_MODULUS: Final[int] = 10_000  # ISS-0000 .. ISS-9999 (frozen ISSUE_ID_PATTERN)


class InjectorError(Exception):
    """An injection cannot be applied cleanly; build.py retries on a fresh sub-RNG, then raises."""


class Injector(Protocol):
    """Shape of an injector module."""

    issue_type: IssueType

    def inject(
        self, episode: PrivateEpisode, rng: random.Random, params: InjectorParams
    ) -> tuple[PrivateEpisode, Issue, ProvenanceEntry]: ...

    def independent_check(self, public: PublicEpisode, issue: Issue) -> None: ...


def used_labels(public: PublicEpisode) -> set[str]:
    """Every public identifier of the episode (for membership checks only)."""
    labels: set[str] = set()
    for row in public.ledger:
        labels.update((row.row_id, row.journal_id))
        labels.update(x for x in (row.entity_id, row.invoice_id, row.payment_id) if x is not None)
    for invoice in public.invoices:
        labels.update((invoice.doc_id, invoice.invoice_id, invoice.entity_id))
    for payment in public.payments:
        labels.update((payment.doc_id, payment.payment_id))
    return labels


def fresh_label(family: str, taken: set[str], rng: random.Random) -> str:
    """An unused ``<family>-<digits>`` label drawn from the gaps below the family's current maximum.

    Labels share the format and range of the existing ones, so an injected record cannot be
    told apart by its id. The caller adds the result to ``taken`` before asking again.
    """
    prefix = f"{family}-"
    suffixes = [label[len(prefix) :] for label in taken if label.startswith(prefix)]
    numbers = {int(suffix) for suffix in suffixes if suffix.isdigit()}  # membership only
    if not numbers:
        raise InjectorError(f"no existing labels in family {family!r}")
    width = max(len(suffix) for suffix in suffixes)  # order-independent
    gaps = [n for n in range(1, max(numbers) + 1) if n not in numbers]
    if not gaps:
        raise InjectorError(f"no free {family!r} labels below the current maximum")
    return f"{prefix}{rng.choice(gaps):0{width}d}"


def shift_timestamp(iso: str, days: int) -> str:
    """Move an ISO-8601 timestamp by whole days; the UTC offset is preserved."""
    return (datetime.fromisoformat(iso) + timedelta(days=days)).isoformat()


def invoice_side(public: PublicEpisode, invoice: Invoice) -> EntityKind:
    """supplier (EXPENSE/AP journal) or customer (AR/REVENUE journal), read from the ledger."""
    for row in public.ledger:
        if row.invoice_id == invoice.invoice_id and row.payment_id is None and row.debit_minor:
            if row.account == "EXPENSE":
                return "supplier"
            if row.account == "AR":
                return "customer"
    raise InjectorError(f"invoice {invoice.invoice_id} has no recognisable journal")


def derive_issue_id(episode_id: str, source_id: str, taken: set[str]) -> str:
    """Deterministic ``ISS-dddd`` from sha256(episode_id:source_id); re-hashed on collision."""
    for n in range(_ISSUE_ID_MODULUS):
        suffix = f":{n}" if n else ""
        digest = hashlib.sha256(f"{episode_id}:{source_id}{suffix}".encode()).hexdigest()
        candidate = f"ISS-{int(digest[:8], 16) % _ISSUE_ID_MODULUS:04d}"
        if candidate not in taken:
            return candidate
    raise InjectorError("issue id space exhausted")


def touched_ids(private: PrivateEpisode) -> set[str]:
    """Ids already claimed by earlier issues: affected, created and source ids (membership)."""
    ids: set[str] = set()
    for issue in private.truth.issues:
        ids.update(issue.affected_ids)
    for entry in private.provenance.injections:
        ids.update(entry.created_ids)
        if entry.source_id is not None:
            ids.add(entry.source_id)
    return ids


def append_truth_and_provenance(
    private: PrivateEpisode, issue: Issue, entry: ProvenanceEntry
) -> PrivateEpisode:
    """Return the episode with ``issue`` and ``entry`` appended (the only way truth grows)."""
    truth = private.truth.model_copy(update={"issues": (*private.truth.issues, issue)})
    provenance = private.provenance.model_copy(
        update={"injections": (*private.provenance.injections, entry)}
    )
    return private.model_copy(update={"truth": truth, "provenance": provenance})


__all__ = [
    "INJECTOR_ORDER",
    "Injector",
    "InjectorError",
    "append_truth_and_provenance",
    "derive_issue_id",
    "fresh_label",
    "invoice_side",
    "shift_timestamp",
    "touched_ids",
    "used_labels",
]
