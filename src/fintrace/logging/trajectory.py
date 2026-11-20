"""Trajectory logger: one canonical JSONL line per ``env.step()`` call, thoughts preserved.

Each line is ``{"action", "episode_id", "observation", "reward", "step", "terminated",
"truncated"}`` serialised with ``sort_keys=True, separators=(",", ":"), ensure_ascii=False`` so a
rerun of the same policy on the same episode reproduces the file byte for byte. ``action`` is the
dict the agent sent (its optional ``thought`` included), ``observation`` is the ``ToolResult`` the
environment returned, ``step`` counts from 1. Nothing else is written: no seed, no wall clock, no
evaluator diagnostics, no ``info``. ``scan_trajectory`` is the leak check the recording script
and the security tests run over every produced file.

One logger serves one episode: ``open`` truncates the file, the environment closes the logger on
the terminal step, and a logger is never shared between threads or episodes.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import IO, Final, Self

from fintrace.models import JsonObject, ToolResult

FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "truth",
    "provenance",
    "expected_amount",
    "issue_id",
    "acceptable_evidence",
    "seed_hash",
    "generator_version",
    "interacts_with",
    "caused_by",
    "legitimate_pattern",
)
# field names that must not exist at any depth of a trajectory line (evaluator-side vocabulary)
FORBIDDEN_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "seed",
    "metric",
    "precision",
    "recall",
    "config",
)
FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset({"f1", "info", "tp", "fp", "fn", "private"})
ENTRY_FIELDS: Final[tuple[str, ...]] = (
    "action",
    "episode_id",
    "observation",
    "reward",
    "step",
    "terminated",
    "truncated",
)
TRAJECTORY_SUFFIX: Final[str] = ".jsonl"


def canonical_line(entry: JsonObject) -> str:
    """The one JSON form on disk (no trailing newline)."""
    return json.dumps(
        entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def trajectory_path(root: Path, policy: str, split: str, seed: int) -> Path:
    """``<root>/<policy>_<split>_<seed>.jsonl`` (the seed lives in the name, never in the file)."""
    return root / f"{policy}_{split}_{seed}{TRAJECTORY_SUFFIX}"


def _json_default(value: object) -> str:
    """Deterministic stand-in for non-JSON values: Decimals keep their text, others their type."""
    if isinstance(value, Decimal):
        return str(value)
    return f"<{type(value).__name__}>"  # never repr(): no memory addresses in the file


def sanitize_action(action: object) -> JsonObject:
    """A JSON-safe, UTF-8-encodable copy of what the agent sent; junk is recorded by kind only.

    Never raises: a non-dict, non-string keys, NaN, a self-referential or absurdly nested
    structure and lone surrogates all degrade to ``{"invalid_action": ...}`` so a hostile action
    can never break the environment step that records it.
    """
    if not isinstance(action, dict):
        return {"invalid_action": type(action).__name__}
    try:
        text = json.dumps(
            action, sort_keys=True, default=_json_default, allow_nan=False, ensure_ascii=False
        )
        text.encode("utf-8")  # a lone surrogate renders with ensure_ascii=False but fails here
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError):
        return {"invalid_action": "unserialisable"}
    copy = json.loads(text)
    return copy if isinstance(copy, dict) else {"invalid_action": "unserialisable"}


def _keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(k) for k in value] + [k for v in value.values() for k in _keys(v)]
    if isinstance(value, list):
        return [k for item in value for k in _keys(item)]
    return []


def scan_trajectory(path: Path) -> list[str]:
    """``<file>:<line>: <what>`` for every forbidden token or key fragment (case-insensitive).

    Tokens are matched as substrings of the whole line; key fragments against every JSON key at
    any depth (``seed``, ``metric``, ... must not even exist as field names). The key check also
    covers agent-authored keys inside ``action``: a policy that names a field ``info`` or ``tp``
    is reported too, which is the intended strictness for files that may train other agents.
    """
    hits: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            lowered = line.lower()
            hits.extend(f"{path.name}:{number}: {t}" for t in FORBIDDEN_TOKENS if t in lowered)
            try:
                keys = {key.lower() for key in _keys(json.loads(line))}
            except ValueError:
                hits.append(f"{path.name}:{number}: not a JSON object")
                continue
            for key in sorted(keys):
                if key in FORBIDDEN_KEYS or any(f in key for f in FORBIDDEN_KEY_FRAGMENTS):
                    hits.append(f"{path.name}:{number}: key {key}")
    return hits


class TrajectoryLogger:
    """Append-only JSONL writer for one episode (opened fresh with ``"w"``)."""

    def __init__(self, output_path: Path, episode_id: str | None = None) -> None:
        self._path = output_path
        self._episode_id = episode_id
        self._file: IO[str] | None = None
        self._step_count = 0
        self._error: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def error(self) -> str | None:
        """Exception class name if recording was aborted, else None."""
        return self._error

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def open(self, episode_id: str | None = None) -> None:
        """Bind the episode id (if not given at construction) and truncate the file.

        Opening twice is a no-op; binding a different episode id than the one given raises.
        """
        if episode_id is not None:
            if self._episode_id is not None and self._episode_id != episode_id:
                raise ValueError(
                    f"logger bound to {self._episode_id}, cannot record {episode_id}"
                )
            self._episode_id = episode_id
        if self._episode_id is None:
            raise ValueError("an episode id is required before the first write")
        if self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8", newline="\n")
        self._step_count = 0

    def log_step(
        self,
        action: object,
        observation: ToolResult,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Serialise one step exactly as the environment returned it."""
        if self._file is None or self._episode_id is None:
            raise RuntimeError("TrajectoryLogger.log_step called before open()")
        self._step_count += 1
        entry: JsonObject = {
            "episode_id": self._episode_id,
            "step": self._step_count,
            "action": sanitize_action(action),
            "observation": observation.model_dump(mode="json"),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        self._file.write(canonical_line(entry) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def abort(self, exc: BaseException) -> None:
        """Stop recording after a write failure; the file keeps the steps written so far."""
        self._error = type(exc).__name__
        self.close()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "ENTRY_FIELDS",
    "FORBIDDEN_KEYS",
    "FORBIDDEN_KEY_FRAGMENTS",
    "FORBIDDEN_TOKENS",
    "TRAJECTORY_SUFFIX",
    "TrajectoryLogger",
    "canonical_line",
    "sanitize_action",
    "scan_trajectory",
    "trajectory_path",
]
