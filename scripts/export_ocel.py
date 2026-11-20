"""Export one agent trajectory plus its public episode as an OCEL 2.0 object-centric event log.

Usage::

    python scripts/export_ocel.py [--trajectory trajectories/rule_based_dev_10000.jsonl]
                                  [--episode-dir data/dev/ep_<id>] [--data-dir data]
                                  [--out docs/sample_ocel_log.json]
                                  [--schema schemas/ocel.schema.json]

The episode directory defaults to the committed dataset entry whose ``episode_id`` matches the
trajectory (``data/manifest.json``; only ``dir``/``episode_id`` are read). Only the six public
files of the episode are opened. Before anything is written the log must pass the structural
invariants (``validate_ocel``), the JSON Schema (when ``jsonschema`` is installed) and the leak
scan (``scan_ocel``); exit code 1 on any failure, 2 when the inputs cannot be resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Final

from fintrace.constants import DATA_DIR, SCHEMAS_DIR
from fintrace.exporters.episode_files import read_public_episode
from fintrace.exporters.ocel import (
    build_ocel_log,
    ocel_json,
    read_trajectory,
    scan_ocel,
    validate_ocel,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_MANIFEST: Final[str] = "manifest.json"
DEFAULT_TRAJECTORY: Final[Path] = REPO_ROOT / "trajectories" / "rule_based_dev_10000.jsonl"
DEFAULT_OUT: Final[Path] = REPO_ROOT / "docs" / "sample_ocel_log.json"
DEFAULT_SCHEMA: Final[Path] = REPO_ROOT / SCHEMAS_DIR / "ocel.schema.json"
SCHEMA_SKIPPED: Final[str] = "skipped (jsonschema not installed)"
SCHEMA_OK: Final[str] = "PASS"  # verdict text of a schema-valid log


def episode_dir_for(episode_id: str, data_dir: Path) -> Path | None:
    """The committed episode directory for ``episode_id`` (``data/manifest.json``), if any."""
    manifest = json.loads((data_dir / DATA_MANIFEST).read_text(encoding="utf-8"))
    for split in manifest["splits"]:
        for entry in split["episodes"]:
            if entry["episode_id"] == episode_id:
                return data_dir / str(entry["dir"])
    return None


def schema_check(log: dict[str, Any], schema_path: Path) -> str:
    """``PASS``, ``skipped (...)`` or ``FAIL: <first error>`` against the OCEL JSON Schema."""
    if importlib.util.find_spec("jsonschema") is None:
        return SCHEMA_SKIPPED
    from jsonschema import Draft202012Validator  # dev dependency; optional at runtime

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(log), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        return f"FAIL: {len(errors)} error(s); first at {list(first.path)}: {first.message}"
    return SCHEMA_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 iff a sound, schema-valid, leak-free log was written."""
    parser = argparse.ArgumentParser(description="Export a trajectory as an OCEL 2.0 log.")
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--episode-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)
    try:
        steps = read_trajectory(args.trajectory)
        if not steps:
            raise ValueError(f"{args.trajectory}: empty trajectory")
        episode_id = str(steps[0].get("episode_id"))
        episode_dir = args.episode_dir or episode_dir_for(episode_id, args.data_dir)
        if episode_dir is None:
            raise ValueError(f"no committed episode {episode_id} under {args.data_dir}")
        public = read_public_episode(episode_dir)
    except (OSError, ValueError) as exc:  # unreadable, malformed or unresolvable inputs
        print(str(exc), file=sys.stderr)
        return 2
    if public.manifest.episode_id != episode_id:
        held = public.manifest.episode_id
        print(f"{episode_dir} holds {held}, not {episode_id}", file=sys.stderr)
        return 2
    log = build_ocel_log(public, steps)
    problems = validate_ocel(log)
    schema_status = schema_check(log, args.schema)
    leaks = scan_ocel(log)
    counts: dict[str, int] = {}
    for obj in log["objects"]:
        counts[str(obj["type"])] = counts.get(str(obj["type"]), 0) + 1
    relationships = sum(len(o["relationships"]) for o in log["objects"]) + sum(
        len(e["relationships"]) for e in log["events"]
    )
    print(f"episode {episode_id} from {episode_dir}; trajectory {args.trajectory.name}")
    print(f"events: {len(log['events'])}; objects: {len(log['objects'])} {counts}; "
          f"relationships: {relationships}")
    print("first event times: " + ", ".join(e["time"] for e in log["events"][:3]))
    print(f"structural invariants: {'PASS' if not problems else f'{len(problems)} problem(s)'}")
    print(f"schema {args.schema.name}: {schema_status}")
    print(f"leak scan: {'clean' if not leaks else f'{len(leaks)} hit(s)'}")
    if problems or leaks or schema_status.startswith("FAIL"):
        for line in (*problems, *leaks):
            print(line, file=sys.stderr)
        if schema_status.startswith("FAIL"):
            print(schema_status, file=sys.stderr)
        print("nothing written", file=sys.stderr)
        return 1
    payload = ocel_json(log).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(f"sha256 {args.out.name}: {hashlib.sha256(payload).hexdigest()}")
    print(f"wrote {args.out} ({len(payload)} bytes)")
    return 0


__all__ = [
    "DATA_MANIFEST",
    "DEFAULT_OUT",
    "DEFAULT_SCHEMA",
    "DEFAULT_TRAJECTORY",
    "REPO_ROOT",
    "SCHEMA_OK",
    "SCHEMA_SKIPPED",
    "episode_dir_for",
    "main",
    "schema_check",
]

if __name__ == "__main__":
    raise SystemExit(main())
