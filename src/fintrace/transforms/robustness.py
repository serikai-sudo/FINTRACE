"""Semantics-preserving transformations of a PrivateEpisode for robustness testing (blueprint 15).

Each transform maps ``(episode, seed) -> (new_episode, mapping)``. The new episode tells the same
story in different clothes, so an agent that generalises scores the same on both, while the
``TransformMapping`` translates ids between the two worlds for the paired equivalence checks:

    remap_ids        every opaque id (INV-/DOC-/PAY-/E-/L-/J-) renamed to a fresh id of the same
                     family and width; every cross-reference, the truth and the provenance follow
    shuffle_records  storage order of the four tables shuffled; the sorted tool views must not
                     notice
    shift_time       the whole calendar moved by 1..7 days: period, documents, postings and FX
                     dates together, so relative chronology, period membership and rates are kept

Transforms are deterministic in ``(transform name, episode id, seed)``. Each one gives the result
a derived ``episode_id`` so that a transformed world is never confused with the original in logs,
reports and the per-environment truth store (the tool view cache is keyed by the loaded episode
object, not by its id, so even a copy that keeps the original id is served correctly). Nothing
here imports the evaluator, the environment or the agents.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Final, NamedTuple, TypeVar

from fintrace.models import (
    InvoiceAllocation,
    Issue,
    JsonObject,
    PrivateEpisode,
    Provenance,
    ProvenanceEntry,
    PublicEpisode,
    Truth,
)

# whole days only: FX dates (YYYY-MM-DD) and timestamps must stay aligned
SHIFT_DAYS_RANGE: Final[tuple[int, int]] = (1, 7)
_SECONDS_PER_DAY: Final[int] = 86_400
_EPISODE_ID_HEX: Final[int] = 12
_ID_RE: Final[re.Pattern[str]] = re.compile(r"^([A-Z]+)-([0-9]+)$")
_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_T = TypeVar("_T")


class TransformMapping(NamedTuple):
    """How to carry ids (and time) from the original episode into the transformed one."""

    id_map: dict[str, str]
    time_shift_seconds: int = 0

    def translate_id(self, original: str) -> str:
        return self.id_map.get(original, original)

    def translate_ids(self, originals: Iterable[str]) -> tuple[str, ...]:
        return tuple(self.translate_id(item) for item in originals)

    def translate_issue(self, issue: Issue) -> Issue:
        """The issue as it reads in the transformed world (issue ids are not renamed)."""
        return issue.model_copy(
            update={
                "affected_ids": tuple(sorted(self.translate_ids(issue.affected_ids))),
                "acceptable_evidence_sets": tuple(
                    self.translate_ids(evidence) for evidence in issue.acceptable_evidence_sets
                ),
            }
        )

    def translate_finding(self, finding: JsonObject) -> JsonObject:
        """A submission finding ported to the transformed world (ids only; amounts unchanged)."""
        ported = dict(finding)
        for key in ("affected_ids", "evidence_ids"):
            value = finding.get(key)
            if isinstance(value, list):
                ported[key] = [self.translate_id(str(item)) for item in value]
        return ported


Transform = Callable[[PrivateEpisode, int], tuple[PrivateEpisode, TransformMapping]]


# ---------------------------------------------------------------- shared helpers
def _derive_rng(seed: int, name: str, episode_id: str) -> random.Random:
    digest = hashlib.sha256(f"{name}:{episode_id}:{seed}".encode()).hexdigest()
    return random.Random(int(digest, 16))


def _derived_episode_id(episode_id: str, name: str, seed: int) -> str:
    digest = hashlib.sha256(f"transform:{name}:{episode_id}:{seed}".encode()).hexdigest()
    return f"ep_{digest[:_EPISODE_ID_HEX]}"


def _with_episode_id(episode: PrivateEpisode, episode_id: str) -> PrivateEpisode:
    manifest = episode.public.manifest.model_copy(update={"episode_id": episode_id})
    public = episode.public.model_copy(update={"manifest": manifest})
    truth = episode.truth.model_copy(update={"episode_id": episode_id})
    return episode.model_copy(update={"public": public, "truth": truth})


def _canonical_order(public: PublicEpisode) -> PublicEpisode:
    """The generator's storage order: (timestamp, id) per table."""
    return public.model_copy(
        update={
            "ledger": tuple(sorted(public.ledger, key=lambda r: (r.timestamp, r.row_id))),
            "invoices": tuple(sorted(public.invoices, key=lambda i: (i.issued_at, i.invoice_id))),
            "payments": tuple(sorted(public.payments, key=lambda p: (p.timestamp, p.payment_id))),
            "fx": tuple(
                sorted(public.fx, key=lambda f: (f.date, f.source_currency, f.target_currency))
            ),
        }
    )


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:  # noqa: ANN401 - JSON tree
    """Apply ``fn`` to every string value anywhere in a JSON tree (keys untouched)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_map_strings(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_strings(item, fn) for key, item in value.items()}
    return value


def _translate_json(value: Any, id_map: dict[str, str]) -> Any:  # noqa: ANN401 - JSON tree
    """Replace every string equal to a mapped id, anywhere in a JSON tree."""
    return _map_strings(value, lambda text: id_map.get(text, text))


def _referenced_ids(episode: PrivateEpisode) -> set[str]:
    """Every id-shaped string the truth and the provenance point at."""
    found: set[str] = set()
    for issue in episode.truth.issues:
        found.update(issue.affected_ids)
        for evidence in issue.acceptable_evidence_sets:
            found.update(evidence)
    for entry in episode.provenance.injections:
        if entry.source_id is not None:
            found.add(entry.source_id)
        found.update(entry.created_ids)
        _map_strings((entry.original, entry.modified), found.add)
    _map_strings(list(episode.provenance.history), found.add)
    return {item for item in found if _ID_RE.fullmatch(item)}


# ---------------------------------------------------------------- remap_ids
def _collect_ids(public: PublicEpisode) -> dict[str, list[str]]:
    """Every opaque id of the public episode grouped by family, sorted, unique."""
    found: set[str] = set()
    for row in public.ledger:
        found.update(
            v for v in (row.row_id, row.entity_id, row.invoice_id, row.payment_id, row.journal_id)
            if v is not None
        )  # fmt: skip
    for invoice in public.invoices:
        found.update((invoice.doc_id, invoice.invoice_id, invoice.entity_id))
    for payment in public.payments:
        found.add(payment.doc_id)
        found.add(payment.payment_id)
        found.update(ref.invoice_id for ref in payment.invoice_refs)
    families: dict[str, list[str]] = {}
    for item in sorted(found):
        match = _ID_RE.fullmatch(item)
        if match is None:
            raise ValueError(f"id {item!r} is not of the form FAMILY-digits")
        families.setdefault(match.group(1), []).append(item)
    return families


def _build_id_remap(public: PublicEpisode, rng: random.Random) -> dict[str, str]:
    """Fresh unused ids of the same family and digit width for every id (a bijection)."""
    id_map: dict[str, str] = {}
    for family, originals in sorted(_collect_ids(public).items()):
        widths = {len(item.split("-", 1)[1]) for item in originals}
        if len(widths) != 1:
            raise ValueError(f"family {family} mixes digit widths {sorted(widths)}")
        width = widths.pop()
        used = {int(item.split("-", 1)[1]) for item in originals}
        free = [number for number in range(10**width) if number not in used]
        if len(free) < len(originals):
            raise ValueError(f"family {family}: not enough fresh {width}-digit ids")
        fresh = rng.sample(free, len(originals))
        for original, number in zip(originals, fresh, strict=True):
            id_map[original] = f"{family}-{number:0{width}d}"
    targets = set(id_map.values())
    if len(targets) != len(id_map) or targets & set(id_map):
        raise ValueError("id remap is not a collision-free bijection")  # pre-check, never expected
    return id_map


def _rewrite_public(public: PublicEpisode, id_map: dict[str, str]) -> PublicEpisode:
    def t(value: str | None) -> str | None:
        return None if value is None else id_map.get(value, value)

    ledger = tuple(
        row.model_copy(
            update={
                "row_id": t(row.row_id),
                "entity_id": t(row.entity_id),
                "invoice_id": t(row.invoice_id),
                "payment_id": t(row.payment_id),
                "journal_id": t(row.journal_id),
            }
        )
        for row in public.ledger
    )
    invoices = tuple(
        invoice.model_copy(
            update={
                "doc_id": t(invoice.doc_id),
                "invoice_id": t(invoice.invoice_id),
                "entity_id": t(invoice.entity_id),
            }
        )
        for invoice in public.invoices
    )
    payments = tuple(
        payment.model_copy(
            update={
                "doc_id": t(payment.doc_id),
                "payment_id": t(payment.payment_id),
                "invoice_refs": tuple(
                    InvoiceAllocation(
                        invoice_id=id_map.get(ref.invoice_id, ref.invoice_id),
                        allocated_minor=ref.allocated_minor,
                    )
                    for ref in payment.invoice_refs
                ),
            }
        )
        for payment in public.payments
    )
    return _canonical_order(
        public.model_copy(update={"ledger": ledger, "invoices": invoices, "payments": payments})
    )


def _rewrite_truth(truth: Truth, mapping: TransformMapping) -> Truth:
    return truth.model_copy(
        update={"issues": tuple(mapping.translate_issue(issue) for issue in truth.issues)}
    )


def _translate_optional(value: str | None, id_map: dict[str, str]) -> str | None:
    return None if value is None else id_map.get(value, value)


def _rewrite_provenance(provenance: Provenance, id_map: dict[str, str]) -> Provenance:
    """Provenance keeps pointing at the right records (relaxed ids, sources, created ids)."""
    injections = tuple(
        ProvenanceEntry(
            issue_id=entry.issue_id,
            op=entry.op,
            source_id=_translate_optional(entry.source_id, id_map),
            created_ids=tuple(id_map.get(item, item) for item in entry.created_ids),
            original=_translate_json(entry.original, id_map),
            modified=_translate_json(entry.modified, id_map),
            caused_by=entry.caused_by,
            note=entry.note,
        )
        for entry in provenance.injections
    )
    history = tuple(_translate_json(event, id_map) for event in provenance.history)
    return provenance.model_copy(update={"injections": injections, "history": history})


def remap_ids(episode: PrivateEpisode, seed: int) -> tuple[PrivateEpisode, TransformMapping]:
    """Rename every opaque id consistently (documents, records, journals, counterparties)."""
    old_id = episode.public.manifest.episode_id
    rng = _derive_rng(seed, "remap_ids", old_id)
    mapping = TransformMapping(id_map=_build_id_remap(episode.public, rng))
    unmapped = sorted(_referenced_ids(episode) - set(mapping.id_map))
    if unmapped:  # the truth or provenance names a record the public episode does not have
        raise ValueError(f"ids referenced outside the public episode: {unmapped[:5]}")
    transformed = episode.model_copy(
        update={
            "public": _rewrite_public(episode.public, mapping.id_map),
            "truth": _rewrite_truth(episode.truth, mapping),
            "provenance": _rewrite_provenance(episode.provenance, mapping.id_map),
        }
    )
    return _with_episode_id(transformed, _derived_episode_id(old_id, "remap_ids", seed)), mapping


# ---------------------------------------------------------------- shuffle_records
def _shuffle_tuple(items: Sequence[_T], rng: random.Random) -> tuple[_T, ...]:
    order = list(range(len(items)))
    rng.shuffle(order)
    return tuple(items[index] for index in order)


def shuffle_records(episode: PrivateEpisode, seed: int) -> tuple[PrivateEpisode, TransformMapping]:
    """Shuffle the storage order of every table; ids, truth and provenance are untouched."""
    old_id = episode.public.manifest.episode_id
    rng = _derive_rng(seed, "shuffle_records", old_id)
    public = episode.public.model_copy(
        update={
            "ledger": _shuffle_tuple(episode.public.ledger, rng),
            "invoices": _shuffle_tuple(episode.public.invoices, rng),
            "payments": _shuffle_tuple(episode.public.payments, rng),
            "fx": _shuffle_tuple(episode.public.fx, rng),
        }
    )
    transformed = episode.model_copy(update={"public": public})
    derived = _derived_episode_id(old_id, "shuffle_records", seed)
    return _with_episode_id(transformed, derived), TransformMapping(id_map={})


# ---------------------------------------------------------------- shift_time
def _shift_iso(timestamp: str, seconds: int) -> str:
    """Shift an aware ISO-8601 timestamp, keeping its own UTC offset."""
    return (datetime.fromisoformat(timestamp) + timedelta(seconds=seconds)).isoformat()


def _shift_date(day: str, seconds: int) -> str:
    return (date.fromisoformat(day) + timedelta(seconds=seconds)).isoformat()


def _shift_text(text: str, seconds: int) -> str:
    """Shift a string that is exactly an ISO date or an aware ISO timestamp; others untouched."""
    if _DATE_RE.fullmatch(text):
        return _shift_date(text, seconds)
    if _TIMESTAMP_RE.fullmatch(text):
        return _shift_iso(text, seconds)
    return text


def _shift_provenance(provenance: Provenance, seconds: int) -> Provenance:
    """Dates and timestamps recorded by the injectors follow the calendar (evaluator-side only)."""
    injections = tuple(
        entry.model_copy(
            update={
                "original": _map_strings(entry.original, lambda t: _shift_text(t, seconds)),
                "modified": _map_strings(entry.modified, lambda t: _shift_text(t, seconds)),
            }
        )
        for entry in provenance.injections
    )
    history = tuple(
        _map_strings(event, lambda t: _shift_text(t, seconds)) for event in provenance.history
    )
    return provenance.model_copy(update={"injections": injections, "history": history})


def shift_time(episode: PrivateEpisode, seed: int) -> tuple[PrivateEpisode, TransformMapping]:
    """Move the whole calendar by 1..7 days: period, documents, postings and FX rates together."""
    old_id = episode.public.manifest.episode_id
    rng = _derive_rng(seed, "shift_time", old_id)
    seconds = rng.randint(*SHIFT_DAYS_RANGE) * _SECONDS_PER_DAY
    public = episode.public
    manifest = public.manifest.model_copy(
        update={
            "period_start": _shift_iso(public.manifest.period_start, seconds),
            "period_end": _shift_iso(public.manifest.period_end, seconds),
        }
    )
    ledger = tuple(
        row.model_copy(update={"timestamp": _shift_iso(row.timestamp, seconds)})
        for row in public.ledger
    )
    invoices = tuple(
        invoice.model_copy(
            update={
                "issued_at": _shift_iso(invoice.issued_at, seconds),
                "due_at": _shift_iso(invoice.due_at, seconds),
            }
        )
        for invoice in public.invoices
    )
    payments = tuple(
        payment.model_copy(update={"timestamp": _shift_iso(payment.timestamp, seconds)})
        for payment in public.payments
    )
    fx = tuple(
        rate.model_copy(update={"date": _shift_date(rate.date, seconds)}) for rate in public.fx
    )
    shifted = public.model_copy(
        update={
            "manifest": manifest,
            "ledger": ledger,
            "invoices": invoices,
            "payments": payments,
            "fx": fx,
        }
    )
    transformed = episode.model_copy(
        update={"public": shifted, "provenance": _shift_provenance(episode.provenance, seconds)}
    )
    derived = _derived_episode_id(old_id, "shift_time", seed)
    mapping = TransformMapping(id_map={}, time_shift_seconds=seconds)
    return _with_episode_id(transformed, derived), mapping


# ---------------------------------------------------------------- composition and paired checks
TRANSFORMS: Final[dict[str, Transform]] = {
    "remap_ids": remap_ids,
    "shuffle_records": shuffle_records,
    "shift_time": shift_time,
}


def apply_transforms(
    episode: PrivateEpisode, seed: int, names: Sequence[str]
) -> tuple[PrivateEpisode, TransformMapping]:
    """Apply transforms in order; the returned mapping composes their id maps and time shifts."""
    id_map: dict[str, str] = {}
    shift = 0
    current = episode
    for name in names:
        current, mapping = TRANSFORMS[name](current, seed)
        id_map = {
            original: mapping.translate_id(target) for original, target in id_map.items()
        } | {k: v for k, v in mapping.id_map.items() if k not in id_map.values()}
        shift += mapping.time_shift_seconds
    return current, TransformMapping(id_map=id_map, time_shift_seconds=shift)


def oracle_findings(truth: Truth) -> list[JsonObject]:
    """The exact findings of the truth (first evidence set), shaped like a submission."""
    findings: list[JsonObject] = []
    for issue in truth.issues:
        evidence = issue.acceptable_evidence_sets[0] if issue.acceptable_evidence_sets else ()
        findings.append(
            {
                "type": issue.type,
                "affected_ids": list(issue.affected_ids),
                "claimed_amount_minor": issue.expected_amount_minor,
                "currency": issue.currency,
                "evidence_ids": list(evidence),
                "explanation": "",
            }
        )
    return findings


def translate_findings(
    findings: Sequence[JsonObject], mapping: TransformMapping
) -> list[JsonObject]:
    """Findings of the original world as they must read in the transformed one."""
    return [mapping.translate_finding(finding) for finding in findings]


def canonical_findings(findings: Sequence[JsonObject]) -> str:
    """Order-free canonical text of a finding list (ids sorted inside each finding)."""
    normalised = []
    for finding in findings:
        item = dict(finding)
        for key in ("affected_ids", "evidence_ids"):
            value = item.get(key)
            if isinstance(value, list):
                item[key] = sorted(str(v) for v in value)
        item.pop("explanation", None)
        normalised.append(item)
    normalised.sort(key=lambda f: json.dumps(f, sort_keys=True))
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"))


__all__ = [
    "SHIFT_DAYS_RANGE",
    "TRANSFORMS",
    "Transform",
    "TransformMapping",
    "apply_transforms",
    "canonical_findings",
    "oracle_findings",
    "remap_ids",
    "shift_time",
    "shuffle_records",
    "translate_findings",
]
