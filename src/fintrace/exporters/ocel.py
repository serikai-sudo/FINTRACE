"""OCEL 2.0 export: an agent's investigation of a public episode as an object-centric event log.

Objects are the public records (invoices, payments, FX rates), the episode itself and the
findings the agent submitted; events are the agent's tool calls, one per trajectory step, in the
OCEL 2.0 JSON exchange format (``schemas/ocel.schema.json``) that process-mining tools such as
PM4Py and Celonis import. The log is built from a ``PublicEpisode`` and a trajectory JSONL only
(``fintrace.logging.trajectory``): nothing private is reachable from here (CONTRIBUTING.md 6.2) and
``scan_ocel`` re-checks the result with the trajectory leak rules. Trajectories carry no clock,
so event times are synthetic and deterministic: ``OCEL_TIME_BASE + step * OCEL_STEP_SECONDS``;
object attribute times use the ``1970-01-01T00:00:00Z`` sentinel pm4py uses for initial values.
All collections are emitted in an explicit order, so a rerun reproduces the file byte for byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

from fintrace.constants import (
    MAX_STEPS,
    OCEL_OBJECT_TIME,
    OCEL_OBJECT_TYPES,
    OCEL_STEP_SECONDS,
    OCEL_TIME_BASE,
    TOOL_NAMES,
)
from fintrace.logging.trajectory import (
    FORBIDDEN_KEY_FRAGMENTS,
    FORBIDDEN_KEYS,
    forbidden_keys,
    forbidden_tokens,
)
from fintrace.models import JsonObject, PublicEpisode

TYPE_EPISODE: Final[str] = "episode"
TYPE_INVOICE: Final[str] = "invoice"
TYPE_PAYMENT: Final[str] = "payment"
TYPE_FX_RATE: Final[str] = "fx_rate"
TYPE_FINDING: Final[str] = "finding"
# event -> object qualifiers
QUALIFIER_EPISODE: Final[str] = "episode"  # every event belongs to the investigated episode
QUALIFIER_LISTED: Final[str] = "listed"  # list_records page contained the object
QUALIFIER_INSPECTED: Final[str] = "inspected"  # get_record
QUALIFIER_READ: Final[str] = "read"  # get_document
QUALIFIER_FLAGS: Final[str] = "flags"  # submit -> finding
# object -> object qualifiers
QUALIFIER_ALLOCATED_TO: Final[str] = "allocated_to"  # payment -> invoice
QUALIFIER_AFFECTS: Final[str] = "affects"  # finding -> invoice / payment
QUALIFIER_EVIDENCED_BY: Final[str] = "evidenced_by"  # finding -> document object
JSON_INDENT: Final[int] = 1

_EVENT_COMMON: Final[dict[str, str]] = {
    "step": "integer",
    "ok": "boolean",
    "error_code": "string",
    "reward": "float",
    "terminated": "boolean",
    "truncated": "boolean",
    "thought": "string",
}
OBJECT_ATTRIBUTES: Final[dict[str, dict[str, str]]] = {
    TYPE_EPISODE: {
        "schema_version": "string",
        "base_currency": "string",
        "period_start": "time",
        "period_end": "time",
        "timezone": "string",
        "n_ledger_rows": "integer",
        "n_documents": "integer",
        "rounding_mode": "string",
        "rounding_stage": "string",
        "fx_rate_date_rule": "string",
        "fx_fallback": "string",
        "fx_rate_direction": "string",
        "period_assignment_field": "string",
    },
    TYPE_INVOICE: {
        "doc_id": "string",
        "entity_id": "string",
        "issued_at": "time",
        "due_at": "time",
        "currency": "string",
        "total_minor": "string",
        "n_line_items": "integer",
    },
    TYPE_PAYMENT: {
        "doc_id": "string",
        "timestamp": "time",
        "currency": "string",
        "amount_minor": "string",
        "n_invoice_refs": "integer",
    },
    TYPE_FX_RATE: {
        "date": "string",
        "source_currency": "string",
        "target_currency": "string",
        "rate": "string",
    },
    TYPE_FINDING: {
        "issue_type": "string",
        "claimed_amount_minor": "string",
        "currency": "string",
        "explanation": "string",
        "n_affected_ids": "integer",
        "n_evidence_ids": "integer",
        "n_unresolved_ids": "integer",
        "step": "integer",
    },
}
EVENT_ATTRIBUTES: Final[dict[str, dict[str, str]]] = {
    "list_records": {
        **_EVENT_COMMON,
        "table": "string",
        "cursor": "string",
        "limit": "integer",
        "n_records": "integer",
        "next_cursor": "string",
        "records_total": "integer",
    },
    "get_record": {
        **_EVENT_COMMON,
        "table": "string",
        "record_id": "string",
        "n_related_ids": "integer",
    },
    "get_document": {**_EVENT_COMMON, "doc_id": "string", "kind": "string"},
    "calculator": {**_EVENT_COMMON, "expression": "string", "value": "string"},
    "submit": {**_EVENT_COMMON, "n_findings": "integer", "status": "string"},
}
_RECORD_ID_FIELDS: Final[tuple[str, ...]] = ("invoice_id", "payment_id", "fx_id")


# ---------------------------------------------------------------- inputs
def read_trajectory(path: Path) -> list[JsonObject]:
    """The steps of a trajectory JSONL in order; ``ValueError`` on anything that is not one."""
    steps: list[JsonObject] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            raise ValueError(f"{path.name}:{number}: not JSON") from None
        step = entry.get("step") if isinstance(entry, dict) else None
        if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= MAX_STEPS:
            raise ValueError(f"{path.name}:{number}: not a trajectory step")
        if not isinstance(entry.get("observation"), dict):
            raise ValueError(f"{path.name}:{number}: step without an observation")
        if steps and step <= int(steps[-1]["step"]):
            raise ValueError(f"{path.name}:{number}: steps out of order")
        steps.append(entry)
    return steps


def event_time(step: int) -> str:
    """Synthetic, offset-aware time of trajectory step ``step`` (deterministic; no wall clock)."""
    if step < 0:
        raise ValueError("step must be non-negative")
    base = datetime.fromisoformat(OCEL_TIME_BASE)  # carries its offset: never naive
    return (base + timedelta(seconds=step * OCEL_STEP_SECONDS)).isoformat()


def fx_object_id(source_currency: str, target_currency: str, date: str) -> str:
    """The id of an FX-rate object; the same id ``list_records`` shows the agent as ``fx_id``."""
    return f"{source_currency}-{target_currency}-{date}"


# ---------------------------------------------------------------- typed attribute values
def _typed(value: object, declared: str) -> object | None:
    """``value`` if it fits the declared OCEL attribute type, else ``None`` (attribute dropped)."""
    if isinstance(value, bool):
        return value if declared == "boolean" else None
    if isinstance(value, int):
        if declared == "integer":
            return value
        return float(value) if declared == "float" else None
    if isinstance(value, float):
        return value if declared == "float" else None
    if isinstance(value, str):
        if declared == "string":
            return value
        return value if declared == "time" and _offset_aware(value) else None
    return None


def _offset_aware(text: str) -> bool:
    try:
        return datetime.fromisoformat(text).tzinfo is not None
    except ValueError:
        return False


def _object_attributes(kind: str, values: Mapping[str, object]) -> list[JsonObject]:
    declared = OBJECT_ATTRIBUTES[kind]
    out: list[JsonObject] = []
    for name in sorted(declared):
        value = _typed(values.get(name), declared[name])
        if value is not None:
            out.append({"name": name, "time": OCEL_OBJECT_TIME, "value": value})
    return out


def _event_attributes(tool: str, values: Mapping[str, object]) -> list[JsonObject]:
    declared = EVENT_ATTRIBUTES[tool]
    out: list[JsonObject] = []
    for name in sorted(declared):
        value = _typed(values.get(name), declared[name])
        if value is not None:
            out.append({"name": name, "value": value})
    return out


def _relationships(pairs: set[tuple[str, str]]) -> list[JsonObject]:
    return [{"objectId": object_id, "qualifier": q} for q, object_id in sorted(pairs)]


# ---------------------------------------------------------------- id resolution (public only)
class _Index:
    """Public ids -> OCEL object ids: records by their own id, documents and ledger rows/journals
    by the invoice/payment they belong to."""

    def __init__(self, public: PublicEpisode) -> None:
        self.object_ids: set[str] = set()
        self.by_alias: dict[str, tuple[str, ...]] = {}
        for invoice in public.invoices:
            self.object_ids.add(invoice.invoice_id)
            self.by_alias[invoice.doc_id] = (invoice.invoice_id,)
        for payment in public.payments:
            self.object_ids.add(payment.payment_id)
            self.by_alias[payment.doc_id] = (payment.payment_id,)
        for rate in public.fx:
            self.object_ids.add(fx_object_id(rate.source_currency, rate.target_currency, rate.date))
        journals: dict[str, set[str]] = {}
        for row in public.ledger:
            candidates = (row.invoice_id, row.payment_id)
            linked = tuple(i for i in candidates if i is not None and i in self.object_ids)
            self.by_alias[row.row_id] = linked
            journals.setdefault(row.journal_id, set()).update(linked)
        for journal_id, linked_ids in journals.items():
            self.by_alias.setdefault(journal_id, tuple(sorted(linked_ids)))

    def resolve(self, ref: object) -> tuple[str, ...]:
        """Object ids a public id refers to (empty for unknown ids: entities, foreign ids, junk)."""
        if not isinstance(ref, str):
            return ()
        if ref in self.object_ids:
            return (ref,)
        return self.by_alias.get(ref, ())


# ---------------------------------------------------------------- objects
def _episode_object(public: PublicEpisode) -> JsonObject:
    manifest = public.manifest
    policy = public.policy
    values = {
        "schema_version": manifest.schema_version,
        "base_currency": manifest.base_currency,
        "period_start": manifest.period_start,
        "period_end": manifest.period_end,
        "timezone": manifest.timezone,
        "n_ledger_rows": manifest.n_ledger_rows,
        "n_documents": manifest.n_documents,
        "rounding_mode": manifest.rounding.mode,
        "rounding_stage": manifest.rounding.stage,
        "fx_rate_date_rule": policy.fx.rate_date_rule,
        "fx_fallback": policy.fx.fallback,
        "fx_rate_direction": policy.fx.rate_direction,
        "period_assignment_field": policy.period.assignment_field,
    }
    return {
        "id": manifest.episode_id,
        "type": TYPE_EPISODE,
        "attributes": _object_attributes(TYPE_EPISODE, values),
        "relationships": [],
    }


def _record_objects(public: PublicEpisode, index: _Index) -> list[JsonObject]:
    objects: list[JsonObject] = []
    for invoice in sorted(public.invoices, key=lambda i: i.invoice_id):
        values = {
            "doc_id": invoice.doc_id,
            "entity_id": invoice.entity_id,
            "issued_at": invoice.issued_at,
            "due_at": invoice.due_at,
            "currency": invoice.currency,
            "total_minor": str(invoice.total_minor),
            "n_line_items": len(invoice.line_items),
        }
        objects.append(
            {
                "id": invoice.invoice_id,
                "type": TYPE_INVOICE,
                "attributes": _object_attributes(TYPE_INVOICE, values),
                "relationships": [],
            }
        )
    for payment in sorted(public.payments, key=lambda p: p.payment_id):
        values = {
            "doc_id": payment.doc_id,
            "timestamp": payment.timestamp,
            "currency": payment.currency,
            "amount_minor": str(payment.amount_minor),
            "n_invoice_refs": len(payment.invoice_refs),
        }
        allocated = {
            (QUALIFIER_ALLOCATED_TO, ref.invoice_id)
            for ref in payment.invoice_refs
            if ref.invoice_id in index.object_ids
        }
        objects.append(
            {
                "id": payment.payment_id,
                "type": TYPE_PAYMENT,
                "attributes": _object_attributes(TYPE_PAYMENT, values),
                "relationships": _relationships(allocated),
            }
        )
    rates = sorted(
        public.fx, key=lambda r: fx_object_id(r.source_currency, r.target_currency, r.date)
    )
    for rate in rates:
        values = {
            "date": rate.date,
            "source_currency": rate.source_currency,
            "target_currency": rate.target_currency,
            "rate": str(rate.rate),
        }
        objects.append(
            {
                "id": fx_object_id(rate.source_currency, rate.target_currency, rate.date),
                "type": TYPE_FX_RATE,
                "attributes": _object_attributes(TYPE_FX_RATE, values),
                "relationships": [],
            }
        )
    return objects


def _finding_objects(step: int, findings: object, index: _Index) -> list[JsonObject]:
    """One object per finding of an accepted submission (ids ``finding-<step>-<index>``)."""
    if not isinstance(findings, list):
        return []
    objects: list[JsonObject] = []
    for position, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        affected = finding.get("affected_ids")
        evidence = finding.get("evidence_ids")
        affected_ids = affected if isinstance(affected, list) else []
        evidence_ids = evidence if isinstance(evidence, list) else []
        links: set[tuple[str, str]] = set()
        unresolved = 0
        groups = ((QUALIFIER_AFFECTS, affected_ids), (QUALIFIER_EVIDENCED_BY, evidence_ids))
        for qualifier, refs in groups:
            for ref in refs:
                targets = index.resolve(ref)
                if not targets:
                    unresolved += 1
                links.update((qualifier, target) for target in targets)
        amount = finding.get("claimed_amount_minor")
        values = {
            "issue_type": finding.get("type"),
            "claimed_amount_minor": (
                str(amount) if isinstance(amount, int) and not isinstance(amount, bool) else None
            ),
            "currency": finding.get("currency"),
            "explanation": finding.get("explanation"),
            "n_affected_ids": len(affected_ids),
            "n_evidence_ids": len(evidence_ids),
            "n_unresolved_ids": unresolved,
            "step": step,
        }
        objects.append(
            {
                "id": f"finding-{step:04d}-{position:02d}",
                "type": TYPE_FINDING,
                "attributes": _object_attributes(TYPE_FINDING, values),
                "relationships": _relationships(links),
            }
        )
    return objects


# ---------------------------------------------------------------- events
def _record_targets(record: object, index: _Index) -> set[str]:
    """Objects a listed / fetched record points at (itself, or the documents of a ledger row)."""
    if not isinstance(record, dict):
        return set()
    targets: set[str] = set()
    for field in _RECORD_ID_FIELDS:
        targets.update(index.resolve(record.get(field)))
    return targets


def _event(
    entry: JsonObject, episode_id: str, index: _Index, findings: Sequence[JsonObject]
) -> JsonObject:
    step = int(entry["step"])
    observation = entry["observation"]
    tool = observation.get("tool")
    if tool not in TOOL_NAMES:
        raise ValueError(f"step {step}: unknown tool {tool!r}")
    action = entry.get("action")
    args = action.get("args") if isinstance(action, dict) else None
    args = args if isinstance(args, dict) else {}
    ok = observation.get("ok") is True
    result = observation.get("result") if ok else None
    result = result if isinstance(result, dict) else {}
    error = observation.get("error")
    values: dict[str, object] = {
        "step": step,
        "ok": ok,
        "error_code": error.get("code") if isinstance(error, dict) else None,
        "reward": entry.get("reward"),
        "terminated": entry.get("terminated"),
        "truncated": entry.get("truncated"),
        "thought": action.get("thought") if isinstance(action, dict) else None,
    }
    links: set[tuple[str, str]] = {(QUALIFIER_EPISODE, episode_id)}
    if tool == "list_records":
        records = result.get("records")
        records = records if isinstance(records, list) else []
        values.update(
            table=args.get("table"),
            cursor=args.get("cursor"),
            limit=args.get("limit"),
            n_records=len(records) if ok else None,
            next_cursor=result.get("next_cursor"),
            records_total=result.get("records_total"),
        )
        for record in records:
            links.update((QUALIFIER_LISTED, t) for t in _record_targets(record, index))
    elif tool == "get_record":
        related = result.get("related_ids")
        values.update(
            table=args.get("table"),
            record_id=args.get("id"),
            n_related_ids=len(related) if isinstance(related, list) else None,
        )
        targets = _record_targets(result.get("record"), index) | set(index.resolve(args.get("id")))
        links.update((QUALIFIER_INSPECTED, t) for t in targets if ok)
    elif tool == "get_document":
        values.update(doc_id=args.get("doc_id"), kind=result.get("kind"))
        targets = _record_targets(result.get("document"), index)
        targets |= set(index.resolve(args.get("doc_id")))
        links.update((QUALIFIER_READ, t) for t in targets if ok)
    elif tool == "calculator":
        values.update(expression=args.get("expression"), value=result.get("value"))
    else:  # submit
        submitted = args.get("findings")
        values.update(
            n_findings=len(submitted) if isinstance(submitted, list) else None,
            status=result.get("status"),
        )
        links.update((QUALIFIER_FLAGS, str(f["id"])) for f in findings)
    return {
        "id": f"evt-{step:04d}",
        "type": tool,
        "time": event_time(step),
        "attributes": _event_attributes(tool, values),
        "relationships": _relationships(links),
    }


# ---------------------------------------------------------------- the log
def build_ocel_log(public: PublicEpisode, steps: Sequence[JsonObject]) -> JsonObject:
    """The OCEL 2.0 log of ``steps`` (a trajectory, see ``read_trajectory``) over ``public``.

    Raises ``ValueError`` when a step belongs to another episode or names an unknown tool.
    """
    episode_id = public.manifest.episode_id
    index = _Index(public)
    objects = [_episode_object(public), *_record_objects(public, index)]
    events: list[JsonObject] = []
    finding_objects: list[JsonObject] = []
    for entry in steps:
        if entry.get("episode_id") not in (None, episode_id):
            raise ValueError(f"step {entry.get('step')}: trajectory of another episode")
        observation = entry["observation"]
        step_findings: list[JsonObject] = []
        if observation.get("tool") == "submit" and observation.get("ok") is True:
            action = entry.get("action")
            args = action.get("args") if isinstance(action, dict) else None
            submitted = args.get("findings") if isinstance(args, dict) else None
            step_findings = _finding_objects(int(entry["step"]), submitted, index)
        finding_objects.extend(step_findings)
        events.append(_event(entry, episode_id, index, step_findings))
    objects.extend(finding_objects)
    return {
        "objectTypes": [
            {
                "name": kind,
                "attributes": [
                    {"name": name, "type": OBJECT_ATTRIBUTES[kind][name]}
                    for name in sorted(OBJECT_ATTRIBUTES[kind])
                ],
            }
            for kind in OCEL_OBJECT_TYPES
        ],
        "eventTypes": [
            {
                "name": tool,
                "attributes": [
                    {"name": name, "type": EVENT_ATTRIBUTES[tool][name]}
                    for name in sorted(EVENT_ATTRIBUTES[tool])
                ],
            }
            for tool in TOOL_NAMES
        ],
        "objects": objects,
        "events": events,
    }


def ocel_json(log: Mapping[str, Any]) -> str:
    """The one JSON form on disk: sorted keys, light indentation, UTF-8 as-is, trailing newline."""
    return (
        json.dumps(log, sort_keys=True, indent=JSON_INDENT, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def export_ocel(public: PublicEpisode, trajectory: Path, out: Path) -> JsonObject:
    """Read ``trajectory``, build the log over ``public``, write it to ``out`` and return it."""
    log = build_ocel_log(public, read_trajectory(trajectory))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ocel_json(log).encode("utf-8"))
    return log


# ---------------------------------------------------------------- checks
def _declared(definitions: object) -> dict[str, dict[str, str]]:
    declared: dict[str, dict[str, str]] = {}
    if isinstance(definitions, list):
        for item in definitions:
            if isinstance(item, dict) and isinstance(item.get("attributes"), list):
                declared[str(item.get("name"))] = {
                    str(a.get("name")): str(a.get("type"))
                    for a in item["attributes"]
                    if isinstance(a, dict)
                }
    return declared


def _check_attributes(
    owner: str, attributes: object, declared: Mapping[str, str], *, timed: bool
) -> list[str]:
    problems: list[str] = []
    if not isinstance(attributes, list):
        return [f"{owner}: attributes is not a list"]
    for attribute in attributes:
        if not isinstance(attribute, dict):
            problems.append(f"{owner}: attribute is not an object")
            continue
        name = str(attribute.get("name"))
        if name not in declared:
            problems.append(f"{owner}: attribute {name} is not declared for its type")
            continue
        if _typed(attribute.get("value"), declared[name]) is None:
            problems.append(f"{owner}: attribute {name} does not fit type {declared[name]}")
        if timed and not (
            isinstance(attribute.get("time"), str) and _offset_aware(str(attribute["time"]))
        ):
            problems.append(f"{owner}: attribute {name} has no offset-aware time")
    return problems


def validate_ocel(log: Mapping[str, Any]) -> list[str]:
    """Structural OCEL 2.0 invariants beyond the JSON Schema (empty list = sound).

    Unique object and event ids; every object/event type declared; every attribute declared for
    its type with a value of the declared type; every relationship target an existing object;
    every time offset-aware; event times non-decreasing in log order.
    """
    problems: list[str] = []
    for key in ("objectTypes", "eventTypes", "objects", "events"):
        if not isinstance(log.get(key), list):
            problems.append(f"{key} missing or not a list")
    if problems:
        return problems
    object_types = _declared(log["objectTypes"])
    event_types = _declared(log["eventTypes"])
    object_ids: set[str] = set()
    for obj in log["objects"]:
        object_id = str(obj.get("id"))
        if object_id in object_ids:
            problems.append(f"object {object_id}: duplicate id")
        object_ids.add(object_id)
        kind = str(obj.get("type"))
        if kind not in object_types:
            problems.append(f"object {object_id}: type {kind} not declared")
            continue
        problems += _check_attributes(
            f"object {object_id}", obj.get("attributes", []), object_types[kind], timed=True
        )
    for obj in log["objects"]:
        for rel in obj.get("relationships", []):
            if str(rel.get("objectId")) not in object_ids:
                target = rel.get("objectId")
                problems.append(f"object {obj.get('id')}: relationship to unknown {target}")
    event_ids: set[str] = set()
    previous: datetime | None = None
    for event in log["events"]:
        event_id = str(event.get("id"))
        if event_id in event_ids:
            problems.append(f"event {event_id}: duplicate id")
        event_ids.add(event_id)
        kind = str(event.get("type"))
        if kind not in event_types:
            problems.append(f"event {event_id}: type {kind} not declared")
        else:
            problems += _check_attributes(
                f"event {event_id}", event.get("attributes", []), event_types[kind], timed=False
            )
        time_text = event.get("time")
        if not (isinstance(time_text, str) and _offset_aware(time_text)):
            problems.append(f"event {event_id}: time is not offset-aware")
        else:
            moment = datetime.fromisoformat(time_text)
            if previous is not None and moment < previous:
                problems.append(f"event {event_id}: time goes backwards")
            previous = moment
        for rel in event.get("relationships", []):
            if str(rel.get("objectId")) not in object_ids:
                problems.append(f"event {event_id}: relationship to unknown {rel.get('objectId')}")
    return problems


def scan_ocel(log: Mapping[str, Any]) -> list[str]:
    """Leak scan of a log with the trajectory rules; OCEL attribute names count as keys."""
    hits = [f"token {token}" for token in forbidden_tokens(ocel_json(log))]
    hits += [f"key {key}" for key in forbidden_keys(dict(log))]
    names: set[str] = set()
    for section in ("objectTypes", "eventTypes", "objects", "events"):
        for item in log.get(section, []) if isinstance(log.get(section), list) else []:
            attributes = item.get("attributes", []) if isinstance(item, dict) else []
            for attribute in attributes if isinstance(attributes, list) else []:
                if isinstance(attribute, dict):
                    names.add(str(attribute.get("name")).lower())
    hits += [
        f"attribute {name}"
        for name in sorted(names)
        if name in FORBIDDEN_KEYS or any(f in name for f in FORBIDDEN_KEY_FRAGMENTS)
    ]
    return hits


__all__ = [
    "EVENT_ATTRIBUTES",
    "JSON_INDENT",
    "OBJECT_ATTRIBUTES",
    "QUALIFIER_AFFECTS",
    "QUALIFIER_ALLOCATED_TO",
    "QUALIFIER_EPISODE",
    "QUALIFIER_EVIDENCED_BY",
    "QUALIFIER_FLAGS",
    "QUALIFIER_INSPECTED",
    "QUALIFIER_LISTED",
    "QUALIFIER_READ",
    "TYPE_EPISODE",
    "TYPE_FINDING",
    "TYPE_FX_RATE",
    "TYPE_INVOICE",
    "TYPE_PAYMENT",
    "build_ocel_log",
    "event_time",
    "export_ocel",
    "fx_object_id",
    "ocel_json",
    "read_trajectory",
    "scan_ocel",
    "validate_ocel",
]
