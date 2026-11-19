"""Contract tests: schemas/*.schema.json are draft 2020-12, strict, pinned to models/constants.

The schemas are a structural filter; ``fintrace.models`` is authoritative. Known, accepted gaps:
JSON Schema cannot express the timestamp validators (timestamps are plain strings here), the
cross-field invariants (``total_minor == sum(lines)``, debit xor credit), strict-int rejection of
``5.0``, or the Rust-regex ``$`` semantics (``re`` accepts a trailing newline). Anything that passes
a schema must still pass Pydantic before it is trusted.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from fintrace.constants import (
    EPISODE_ID_PATTERN,
    ERROR_CODES,
    ISSUE_TYPES,
    MAX_AFFECTED_IDS_PER_FINDING,
    MAX_EVIDENCE_IDS_PER_FINDING,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    MAX_ID_CHARS,
    MAX_STEPS,
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    SCHEMA_VERSION,
    SCHEMAS_DIR,
    TOOL_NAMES,
)
from fintrace.models import Action, Error, StepResult, Submission, ToolResult

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = REPO_ROOT / SCHEMAS_DIR
DIALECT = "https://json-schema.org/draft/2020-12/schema"
GENERATED_NAMES = ("manifest", "invoice", "payment", "submission")
HAND_WRITTEN_NAMES = ("trajectory", "ocel")
ALL_NAMES = GENERATED_NAMES + HAND_WRITTEN_NAMES


def _load_generator() -> ModuleType:
    path = REPO_ROOT / "scripts" / "generate_schemas.py"
    spec = importlib.util.spec_from_file_location("generate_schemas", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


def _raw(name: str) -> bytes:
    return (SCHEMAS / f"{name}.schema.json").read_bytes()  # bytes: no newline translation


def _schema(name: str) -> dict[str, Any]:
    loaded = json.loads(_raw(name))
    assert isinstance(loaded, dict)
    return loaded


def _walk(node: object) -> Iterator[dict[str, Any]]:
    """Yield every dict node of a JSON tree, depth-first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _ref_targets(schema: dict[str, Any]) -> set[str]:
    return {node["$ref"] for node in _walk(schema) if isinstance(node.get("$ref"), str)}


# ---------------------------------------------------------------- 1-3: every file


def test_all_six_schemas_parse() -> None:
    assert tuple(GEN.GENERATED) + GEN.HAND_WRITTEN == ALL_NAMES
    for name in ALL_NAMES:
        schema = _schema(name)
        assert schema["type"] == "object", name
        assert len(_raw(name)) > 500, name  # no placeholders left
        assert b"\r" not in _raw(name), name  # LF only, on every platform
        for ref in _ref_targets(schema):  # every $ref resolves inside the file
            assert ref.startswith("#/$defs/") and ref.split("/")[-1] in schema["$defs"], (name, ref)


def test_schema_dialect_is_draft_2020_12() -> None:
    for name in ALL_NAMES:
        assert _schema(name)["$schema"] == DIALECT, name
        text = _raw(name).decode("utf-8")
        assert "draft-07" not in text and "2019-09" not in text, name
        assert "#/definitions/" not in text, name  # 2020-12 uses $defs


def test_schema_ids_follow_convention() -> None:
    for name in ALL_NAMES:
        schema = _schema(name)
        assert schema["$id"] == f"urn:fintrace:schema:{name}:{SCHEMA_VERSION}", name
        assert schema["$id"] == GEN.schema_id(name)
        assert isinstance(schema["title"], str) and schema["title"], name


# ---------------------------------------------------------------- 4-8: generated files


def test_generated_schemas_match_committed() -> None:
    for name, model in GEN.GENERATED.items():
        expected = GEN.render(GEN.build_schema(name, model))
        assert _raw(name) == expected.encode("utf-8"), f"{name}: stale — run `make schemas`"
        assert expected == GEN.render(json.loads(expected))  # render is a fixed point
    assert GEN.main(["--check"]) == 0


def test_no_additional_properties_true() -> None:
    for name in ALL_NAMES:
        for node in _walk(_schema(name)):
            assert node.get("additionalProperties") is not True, name
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False, (name, sorted(node["properties"]))


def test_generated_schemas_have_no_field_titles() -> None:
    for name in GENERATED_NAMES:
        schema = _schema(name)
        assert "title" in schema and "description" in schema  # top-level only
        for node in _walk(schema):
            for field_name, field_schema in node.get("properties", {}).items():
                assert "title" not in field_schema, (name, field_name)
                assert "description" not in field_schema, (name, field_name)


def test_required_matches_model_fields() -> None:
    for name, model in GEN.GENERATED.items():
        schema = _schema(name)
        required_in_model = [f for f, info in model.model_fields.items() if info.is_required()]
        assert schema["required"] == required_in_model, name
        assert sorted(schema["properties"]) == sorted(model.model_fields), name
    dumped = Submission(episode_id="ep_0123456789ab", findings=[]).model_dump(mode="json")
    assert sorted(dumped) == sorted(_schema("submission")["properties"])


def test_submission_schema_enforces_caps() -> None:
    schema = _schema("submission")
    assert schema["properties"]["episode_id"]["pattern"] == EPISODE_ID_PATTERN
    assert schema["properties"]["findings"]["maxItems"] == MAX_FINDINGS
    assert schema["properties"]["findings"]["items"] == {"$ref": "#/$defs/Finding"}
    finding = schema["$defs"]["Finding"]
    assert finding["additionalProperties"] is False
    props = finding["properties"]
    assert props["type"]["enum"] == list(ISSUE_TYPES)
    assert props["affected_ids"]["maxItems"] == MAX_AFFECTED_IDS_PER_FINDING
    assert props["affected_ids"]["items"]["maxLength"] == MAX_ID_CHARS
    assert props["evidence_ids"]["maxItems"] == MAX_EVIDENCE_IDS_PER_FINDING
    assert props["explanation"]["maxLength"] == MAX_EXPLANATION_CHARS
    assert props["claimed_amount_minor"] == {"minimum": 0, "type": "integer"}
    assert props["currency"]["pattern"] == "^[A-Z]{3}$"


# ---------------------------------------------------------------- 9-10: hand-written files


def test_trajectory_schema_consistent_with_constants_and_models() -> None:
    schema = _schema("trajectory")
    defs = schema["$defs"]
    assert schema["properties"]["episode_id"]["pattern"] == EPISODE_ID_PATTERN
    assert schema["properties"]["steps"]["maxItems"] == MAX_STEPS
    assert defs["Action"]["properties"]["tool"]["enum"] == list(TOOL_NAMES)
    assert defs["Observation"]["properties"]["tool"]["enum"] == list(TOOL_NAMES)
    assert defs["Error"]["properties"]["code"]["enum"] == list(ERROR_CODES)
    for reward in (schema["properties"]["final_reward"], defs["Step"]["properties"]["reward"]):
        assert (reward["minimum"], reward["maximum"]) == (REWARD_CLIP_MIN, REWARD_CLIP_MAX)
    for counter in ("steps_used", "steps_remaining"):
        assert defs["Observation"]["properties"][counter]["maximum"] == MAX_STEPS
    assert defs["Step"]["properties"]["step_index"]["maximum"] == MAX_STEPS - 1
    # pinned to the models: same field names as Action / ToolResult / Error / StepResult
    assert set(defs["Action"]["properties"]) == set(Action.model_fields)
    assert set(defs["Observation"]["properties"]) == set(ToolResult.model_fields)
    assert set(defs["Error"]["properties"]) == set(Error.model_fields) == {"code"}
    step_fields = set(defs["Step"]["properties"]) - {"step_index", "action"}
    assert step_fields == set(StepResult.model_fields)
    assert sorted(defs["Info"]["properties"]) == sorted(defs["Info"]["required"])
    assert sorted(defs["Info"]["required"]) == ["max_steps", "version"]
    # ok <=> result present and error null (ToolResult invariant), expressed as if/then/else
    assert defs["Observation"]["if"] == {"properties": {"ok": {"const": True}}}
    assert defs["Observation"]["then"]["properties"]["error"] == {"type": "null"}
    assert defs["Observation"]["else"]["properties"]["result"] == {"type": "null"}
    # no private-episode surface besides the explicitly labelled, optional `metrics`
    assert "metrics" not in schema["required"]
    lowered = _raw("trajectory").decode("utf-8").lower()
    for forbidden in ("seed", "provenance", "expected_amount", "issue_id", "acceptable_evidence"):
        assert forbidden not in lowered, forbidden


def test_ocel_schema_structure() -> None:
    schema = _schema("ocel")
    assert schema["required"] == ["objectTypes", "eventTypes", "objects", "events"]
    for key in schema["required"]:
        assert schema["properties"][key]["type"] == "array", key
        assert schema["properties"][key]["items"]["$ref"].startswith("#/$defs/"), key
    defs = schema["$defs"]
    assert defs["AttributeType"]["enum"] == ["string", "time", "integer", "float", "boolean"]
    assert defs["AttributeValue"]["type"] == ["string", "number", "boolean", "null"]
    assert defs["Event"]["required"] == ["id", "type", "time"]  # OCEL 2.0: attrs/rels optional
    assert defs["Object"]["required"] == ["id", "type"]
    assert set(defs["Event"]["properties"]) == {"id", "type", "time", "attributes", "relationships"}
    assert set(defs["Object"]["properties"]) == {"id", "type", "attributes", "relationships"}
    assert defs["Relationship"]["required"] == ["objectId", "qualifier"]
    assert defs["ObjectAttribute"]["required"] == ["name", "time", "value"]
    timestamp = re.compile(defs["Timestamp"]["pattern"])
    assert timestamp.fullmatch("2026-02-03T10:15:00+05:00")
    assert timestamp.fullmatch("2026-02-03T10:15:00.250Z")
    assert timestamp.fullmatch("2026-02-03T10:15:00") is None  # naive timestamps are bugs
    assert timestamp.fullmatch("2026-02-03T10:15:00+24:00") is None
