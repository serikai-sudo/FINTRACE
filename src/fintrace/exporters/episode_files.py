"""Episode files: the on-disk format of one FINTRACE episode and its exact inverse.

Ten byte-stable files per episode directory (``data/<split>/ep_<episode_id>/`` or a directory the
jury names): the six public files the agent-side tools read (``manifest.json``, ``policy.json``,
``invoices.jsonl``, ``payments.jsonl``, ``ledger.csv``, ``fx.csv``), the two private files only the
evaluator may read (``truth.json``, ``provenance.json``), ``seed.txt`` and ``checksums.sha256`` over
the public files. Canonical JSON (sorted keys, compact separators, ``ensure_ascii=False``), CSV via
``csv.DictWriter`` with sorted field names and LF line ends, one record per JSONL line.
``write_episode`` and ``read_episode`` are exact inverses; ``read_public_episode`` reads the six
public files only; ``verify_episode`` re-hashes the public files. Used by scripts/generate_data.py,
the jury adapter and the OCEL exporter.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Final

from fintrace.models import (
    FXRate,
    Invoice,
    LedgerRow,
    Manifest,
    Payment,
    Policy,
    PrivateEpisode,
    Provenance,
    PublicEpisode,
    Truth,
)

PUBLIC_FILES: Final[tuple[str, ...]] = (  # the agent-visible artefacts; the only files hashed
    "fx.csv",
    "invoices.jsonl",
    "ledger.csv",
    "manifest.json",
    "payments.jsonl",
    "policy.json",
)
PRIVATE_FILES: Final[tuple[str, ...]] = ("provenance.json", "truth.json")  # evaluator-only
SEED_FILE: Final[str] = "seed.txt"
CHECKSUMS_FILE: Final[str] = "checksums.sha256"
EPISODE_FILES: Final[frozenset[str]] = frozenset(
    (*PUBLIC_FILES, *PRIVATE_FILES, SEED_FILE, CHECKSUMS_FILE)
)
LEDGER_INT_COLUMNS: Final[tuple[str, ...]] = ("credit_minor", "debit_minor")
LEDGER_NULLABLE_COLUMNS: Final[tuple[str, ...]] = ("entity_id", "invoice_id", "payment_id")


# ---------------------------------------------------------------- serialisation primitives
def canonical_json(obj: object) -> str:
    """The one JSON form on disk: sorted keys, compact separators, UTF-8 as-is, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _jsonl_text(records: list[dict[str, Any]]) -> str:
    """One canonical JSON object per line; empty input gives an empty file."""
    return "".join(canonical_json(record) + "\n" for record in records)


def _csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    """CSV with the given (sorted) header, LF line ends, ``None`` written as the empty string."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    return buffer.getvalue()


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))  # bytes: LF on every OS, no locale encoding


def _write_json(path: Path, obj: object) -> None:
    write_text(path, canonical_json(obj) + "\n")


def _read_json(path: Path) -> Any:  # noqa: ANN401 - JSON payload, validated by the caller's model
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())



# ---------------------------------------------------------------- one episode on disk
def _records(items: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def _fieldnames(model: type[LedgerRow] | type[FXRate]) -> list[str]:
    return sorted(model.model_fields)


def write_episode(ep: PrivateEpisode, ep_dir: Path, seed: int) -> None:
    """Write the ten episode files into ``ep_dir`` (created if needed), byte-stably."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    public = ep.public
    _write_json(ep_dir / "manifest.json", public.manifest.model_dump(mode="json"))
    _write_json(ep_dir / "policy.json", public.policy.model_dump(mode="json"))
    write_text(ep_dir / "invoices.jsonl", _jsonl_text(_records(public.invoices)))
    write_text(ep_dir / "payments.jsonl", _jsonl_text(_records(public.payments)))
    write_text(ep_dir / "ledger.csv", _csv_text(_records(public.ledger), _fieldnames(LedgerRow)))
    write_text(ep_dir / "fx.csv", _csv_text(_records(public.fx), _fieldnames(FXRate)))
    _write_json(ep_dir / "truth.json", ep.truth.model_dump(mode="json"))
    _write_json(ep_dir / "provenance.json", ep.provenance.model_dump(mode="json"))
    write_text(ep_dir / SEED_FILE, f"{seed}\n")
    lines = [f"{sha256_file(ep_dir / name)}  {name}\n" for name in sorted(PUBLIC_FILES)]
    write_text(ep_dir / CHECKSUMS_FILE, "".join(lines))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # split on "\n" only: str.splitlines() would also break records at U+2028 / U+0085, which
    # ensure_ascii=False writes verbatim inside string fields
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ledger_row(raw: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = dict(raw)
    for column in LEDGER_INT_COLUMNS:
        row[column] = int(row[column])
    for column in LEDGER_NULLABLE_COLUMNS:
        row[column] = row[column] or None
    return row


def read_public_episode(ep_dir: Path) -> PublicEpisode:
    """The ``PublicEpisode`` from the six public files only; truth/provenance/seed stay unread.

    The agent-side readers (exporters, demos) use this: nothing private is opened at all.
    """
    ledger_rows = _read_csv(ep_dir / "ledger.csv")
    return PublicEpisode(
        manifest=Manifest.model_validate(_read_json(ep_dir / "manifest.json")),
        policy=Policy.model_validate(_read_json(ep_dir / "policy.json")),
        ledger=tuple(LedgerRow.model_validate(_ledger_row(row)) for row in ledger_rows),
        invoices=tuple(Invoice.model_validate(r) for r in _read_jsonl(ep_dir / "invoices.jsonl")),
        payments=tuple(Payment.model_validate(r) for r in _read_jsonl(ep_dir / "payments.jsonl")),
        fx=tuple(FXRate.model_validate(r) for r in _read_csv(ep_dir / "fx.csv")),
    )


def read_episode(ep_dir: Path) -> PrivateEpisode:
    """Rebuild the ``PrivateEpisode`` from an episode directory (inverse of ``write_episode``)."""
    return PrivateEpisode(
        public=read_public_episode(ep_dir),
        truth=Truth.model_validate(_read_json(ep_dir / "truth.json")),
        provenance=Provenance.model_validate(_read_json(ep_dir / "provenance.json")),
    )


def read_seed(ep_dir: Path) -> int:
    """The seed recorded in ``seed.txt``."""
    return int((ep_dir / SEED_FILE).read_text(encoding="utf-8").strip())


def _parse_checksums(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        entries[name] = digest
    return entries


def verify_episode(ep_dir: Path) -> bool:
    """True iff ``checksums.sha256`` lists exactly the public files and every digest matches."""
    checksums = ep_dir / CHECKSUMS_FILE
    if not checksums.is_file():
        return False
    expected = _parse_checksums(checksums.read_text(encoding="utf-8"))
    if set(expected) != set(PUBLIC_FILES):
        return False
    return all(
        (ep_dir / name).is_file() and sha256_file(ep_dir / name) == digest
        for name, digest in expected.items()
    )


__all__ = [
    "CHECKSUMS_FILE",
    "EPISODE_FILES",
    "LEDGER_INT_COLUMNS",
    "LEDGER_NULLABLE_COLUMNS",
    "PRIVATE_FILES",
    "PUBLIC_FILES",
    "SEED_FILE",
    "canonical_json",
    "read_episode",
    "read_public_episode",
    "read_seed",
    "sha256_file",
    "verify_episode",
    "write_episode",
    "write_text",
]
