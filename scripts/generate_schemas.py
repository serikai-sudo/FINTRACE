"""Generate JSON Schema (draft 2020-12) files for the Pydantic contract models.

Usage: ``python scripts/generate_schemas.py [--check]`` (or ``make schemas``).

Output is deterministic — sorted keys, 2-space indent, trailing newline — so
``tests/contract/test_schemas.py`` pins the committed files byte-for-byte to ``fintrace.models``.
Never hand-edit a generated file: change ``models.py`` and regenerate. The ``trajectory`` and
``ocel`` schemas are hand-written and are not touched by this script.

The schemas are a structural filter, not the authority: JSON Schema cannot express the models'
timestamp validators, cross-field invariants (``total_minor == sum(lines)``), strict-int rejection
of ``5.0``, or Rust-regex ``$`` semantics. Pydantic validation in ``fintrace.models`` is final.
Run under the pinned dependencies (``make install``): output depends on the pydantic version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from pydantic import BaseModel, JsonValue
from pydantic.json_schema import GenerateJsonSchema

from fintrace.constants import SCHEMA_VERSION, SCHEMAS_DIR
from fintrace.models import Invoice, Manifest, Payment, Submission

JSON_SCHEMA_DIALECT: Final[str] = "https://json-schema.org/draft/2020-12/schema"
GENERATED: Final[dict[str, type[BaseModel]]] = {
    "manifest": Manifest,
    "invoice": Invoice,
    "payment": Payment,
    "submission": Submission,
}
HAND_WRITTEN: Final[tuple[str, ...]] = ("trajectory", "ocel")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


class LeanJsonSchema(GenerateJsonSchema):
    """Generator that omits per-field ``title`` keys; model-level title/description are kept."""

    # `schema` is pydantic's private CoreSchemaOrField union (TYPE_CHECKING-only export);
    # `object` is the public, contravariance-safe annotation for this override.
    def field_title_should_be_set(self, schema: object) -> bool:
        return False


def schema_id(name: str) -> str:
    """Return the canonical ``$id`` for a schema file: ``urn:fintrace:schema:<name>:<version>``."""
    return f"urn:fintrace:schema:{name}:{SCHEMA_VERSION}"


def build_schema(name: str, model: type[BaseModel]) -> dict[str, JsonValue]:
    """Build the draft 2020-12 schema document for ``model``."""
    schema = model.model_json_schema(schema_generator=LeanJsonSchema, mode="validation")
    return {"$schema": JSON_SCHEMA_DIALECT, "$id": schema_id(name), **schema}


def render(schema: dict[str, JsonValue]) -> str:
    """Serialise deterministically: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def schema_path(name: str, schemas_dir: Path = REPO_ROOT / SCHEMAS_DIR) -> Path:
    """Return the on-disk path of ``<name>.schema.json``."""
    return schemas_dir / f"{name}.schema.json"


def main(argv: list[str] | None = None) -> int:
    """Write the generated schemas, or with ``--check`` report stale files and exit 1."""
    parser = argparse.ArgumentParser(
        description="Generate schemas/*.schema.json from the Pydantic contract models."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any committed generated schema differs from the models",
    )
    args = parser.parse_args(argv)

    stale: list[str] = []
    for name, model in GENERATED.items():
        payload = render(build_schema(name, model)).encode("utf-8")  # bytes: LF on every OS
        path = schema_path(name)
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                stale.append(name)
            continue
        path.write_bytes(payload)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(payload)} bytes)")

    if stale:
        print(f"stale schemas (run `make schemas`): {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "GENERATED",
    "HAND_WRITTEN",
    "JSON_SCHEMA_DIALECT",
    "REPO_ROOT",
    "LeanJsonSchema",
    "build_schema",
    "main",
    "render",
    "schema_id",
    "schema_path",
]

if __name__ == "__main__":
    raise SystemExit(main())
