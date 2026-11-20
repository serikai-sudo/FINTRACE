"""OCEL 2.0 export (D2): JSON Schema validity, structural invariants, coverage of the public
episode, one event per trajectory step, determinism, synthetic offset-aware times; slow: the four
baseline trajectories over the dev split all export to valid, leak-free logs."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import fintrace.environment.env as env_module
from fintrace.agents.base import BASELINE_NAMES
from fintrace.constants import (
    OCEL_OBJECT_TIME,
    OCEL_OBJECT_TYPES,
    OCEL_STEP_SECONDS,
    OCEL_TIME_BASE,
    TOOL_NAMES,
)
from fintrace.evaluator.runner import BASELINE_FACTORIES, run_split
from fintrace.exporters.episode_files import read_episode, read_public_episode
from fintrace.exporters.ocel import (
    QUALIFIER_ALLOCATED_TO,
    QUALIFIER_EPISODE,
    QUALIFIER_FLAGS,
    QUALIFIER_READ,
    TYPE_EPISODE,
    TYPE_FINDING,
    TYPE_FX_RATE,
    TYPE_INVOICE,
    TYPE_PAYMENT,
    build_ocel_log,
    fx_object_id,
    ocel_json,
    read_trajectory,
    scan_ocel,
    validate_ocel,
)
from fintrace.generator.build import build_episode
from fintrace.models import JsonObject, PublicEpisode

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_TRAJECTORY = REPO_ROOT / "trajectories" / "rule_based_dev_10000.jsonl"
SAMPLE_EPISODE_DIR = REPO_ROOT / "data" / "dev" / "ep_b2f2e0741428"
SAMPLE_LOG = REPO_ROOT / "docs" / "sample_ocel_log.json"
SCHEMA_PATH = REPO_ROOT / "schemas" / "ocel.schema.json"
DATA_MANIFEST = REPO_ROOT / "data" / "manifest.json"


@cache
def _public() -> PublicEpisode:
    return read_public_episode(SAMPLE_EPISODE_DIR)


@cache
def _steps() -> tuple[JsonObject, ...]:
    return tuple(read_trajectory(SAMPLE_TRAJECTORY))


@cache
def _log() -> JsonObject:
    """Shared read-only log of the committed sample; tests that tamper take a deepcopy."""
    return build_ocel_log(_public(), _steps())


@cache
def _schema() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def _schema_errors(log: JsonObject) -> list[str]:
    return [e.message for e in Draft202012Validator(_schema()).iter_errors(log)]


def _of_type(log: JsonObject, kind: str) -> list[JsonObject]:
    return [o for o in log["objects"] if o["type"] == kind]


def _attribute(item: JsonObject, name: str) -> object:
    return next(a["value"] for a in item["attributes"] if a["name"] == name)


def _step(
    step: int, observation: JsonObject, action: object = None, **extra: object
) -> JsonObject:
    base: JsonObject = {
        "episode_id": _public().manifest.episode_id,
        "step": step,
        "action": action if action is not None else {"invalid_action": "str"},
        "observation": {
            "error": None if observation.get("ok") else {"code": "INVALID_ACTION"},
            "steps_used": step,
            "steps_remaining": 100 - step,
            **observation,
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    return {**base, **extra}


@cache
def _hostile_steps() -> tuple[JsonObject, ...]:
    episode_id = _public().manifest.episode_id
    bad_finding = {
        "type": "fx_error",
        "affected_ids": [["nested"], "E-9999"],
        "claimed_amount_minor": True,
        "currency": "USD",
        "evidence_ids": None,
        "explanation": "",
    }
    return (
        _step(1, {"tool": "list_records", "ok": False, "result": None}),  # junk action
        _step(
            2,
            {"tool": "list_records", "ok": True, "result": {"records": [1, {"invoice_id": 3}]}},
            {"tool": "list_records", "args": {"table": 7, "limit": "50", "cursor": None}},
        ),
        _step(3, {"tool": "get_record", "ok": False, "result": None}, {"tool": "get_record"}),
        _step(
            4,
            {
                "tool": "submit",
                "ok": False,
                "result": None,
                "error": {"code": "INVALID_SUBMISSION"},
            },
            {"tool": "submit", "args": {"episode_id": "ep_0000000000", "findings": [bad_finding]}},
        ),
        _step(
            5,
            {"tool": "submit", "ok": True, "result": {"status": "accepted", "findings_count": 1}},
            {"tool": "submit", "args": {"episode_id": episode_id, "findings": [bad_finding]}},
            terminated=True,
        ),
    )


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "export_ocel.py"
    spec = importlib.util.spec_from_file_location("export_ocel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ocel_schema_valid() -> None:
    log = _log()
    assert _schema_errors(log) == []
    assert [t["name"] for t in log["objectTypes"]] == list(OCEL_OBJECT_TYPES)
    assert [t["name"] for t in log["eventTypes"]] == list(TOOL_NAMES)
    allowed = set(_schema()["$defs"]["AttributeType"]["enum"])
    used = {a["type"] for t in (*log["objectTypes"], *log["eventTypes"]) for a in t["attributes"]}
    assert used <= allowed and "time" in used


def test_ocel_structural_invariants() -> None:
    log = _log()
    assert validate_ocel(log) == []
    # the same invariants recomputed here, independently of validate_ocel
    object_ids = [o["id"] for o in log["objects"]]
    assert len(object_ids) == len(set(object_ids))
    ids = set(object_ids)
    object_types = {t["name"]: {a["name"] for a in t["attributes"]} for t in log["objectTypes"]}
    event_types = {t["name"]: {a["name"] for a in t["attributes"]} for t in log["eventTypes"]}
    for obj in log["objects"]:
        assert obj["type"] in object_types
        assert {a["name"] for a in obj["attributes"]} <= object_types[obj["type"]]
        assert all(r["objectId"] in ids for r in obj["relationships"])
    event_ids = [e["id"] for e in log["events"]]
    assert len(event_ids) == len(set(event_ids))
    for event in log["events"]:
        assert event["type"] in event_types
        assert {a["name"] for a in event["attributes"]} <= event_types[event["type"]]
        assert all(r["objectId"] in ids for r in event["relationships"])
        assert all(a["value"] is not None for a in event["attributes"])
    # positive controls: the checker reports a broken log
    broken = copy.deepcopy(log)
    broken["events"][0]["relationships"].append({"objectId": "nope", "qualifier": "episode"})
    broken["objects"][1]["type"] = "ghost"
    broken["events"][1]["attributes"].append({"name": "undeclared", "value": 1})
    broken["events"][2]["time"] = "2026-04-01T09:00:03"  # naive
    broken["objects"][2]["id"] = broken["objects"][3]["id"]  # duplicate
    problems = validate_ocel(broken)
    assert any("unknown nope" in p for p in problems)
    assert any("type ghost not declared" in p for p in problems)
    assert any("undeclared" in p for p in problems)
    assert any("offset-aware" in p for p in problems)
    assert any("duplicate id" in p for p in problems)
    # hostile / odd steps: junk actions, rejected calls, wrong arg types, a rejected submit and an
    # accepted submit with malformed findings still give a sound, schema-valid, leak-free log
    hostile = build_ocel_log(_public(), _hostile_steps())
    assert validate_ocel(hostile) == [] and scan_ocel(hostile) == []
    assert _schema_errors(hostile) == []
    expected_types = [s["observation"]["tool"] for s in _hostile_steps()]
    assert [e["type"] for e in hostile["events"]] == expected_types
    assert len(_of_type(hostile, TYPE_FINDING)) == 1  # only the accepted submit creates findings
    finding = _of_type(hostile, TYPE_FINDING)[0]
    assert _attribute(finding, "n_unresolved_ids") == 2 and finding["relationships"] == []
    assert "claimed_amount_minor" not in {a["name"] for a in finding["attributes"]}  # bool dropped
    with pytest.raises(ValueError, match="another episode"):
        build_ocel_log(_public(), [{**_hostile_steps()[0], "episode_id": "ep_0000000000"}])
    with pytest.raises(ValueError, match="unknown tool"):
        build_ocel_log(_public(), [_step(1, {"tool": "nope", "ok": False, "result": None})])


def test_ocel_objects_cover_episode() -> None:
    log, public = _log(), _public()
    assert read_episode(SAMPLE_EPISODE_DIR).public == public  # the public-only reader is exact
    assert len(_of_type(log, TYPE_EPISODE)) == 1
    assert _of_type(log, TYPE_EPISODE)[0]["id"] == public.manifest.episode_id
    assert {o["id"] for o in _of_type(log, TYPE_INVOICE)} == {i.invoice_id for i in public.invoices}
    assert {o["id"] for o in _of_type(log, TYPE_PAYMENT)} == {p.payment_id for p in public.payments}
    expected_fx = {fx_object_id(r.source_currency, r.target_currency, r.date) for r in public.fx}
    assert {o["id"] for o in _of_type(log, TYPE_FX_RATE)} == expected_fx
    assert len(_of_type(log, TYPE_INVOICE)) == len(public.invoices) == 86
    assert len(_of_type(log, TYPE_PAYMENT)) == len(public.payments) == 66
    assert len(_of_type(log, TYPE_FX_RATE)) == len(public.fx) == 182
    submit = next(s for s in _steps() if s["observation"]["tool"] == "submit")
    assert len(_of_type(log, TYPE_FINDING)) == submit["observation"]["result"]["findings_count"]
    allocations = sum(
        1
        for o in _of_type(log, TYPE_PAYMENT)
        for r in o["relationships"]
        if r["qualifier"] == QUALIFIER_ALLOCATED_TO
    )
    assert allocations == sum(len(p.invoice_refs) for p in public.payments) > 0
    invoice = public.invoices[0]
    obj = next(o for o in _of_type(log, TYPE_INVOICE) if o["id"] == invoice.invoice_id)
    assert _attribute(obj, "total_minor") == str(invoice.total_minor)  # money as a string
    assert _attribute(obj, "issued_at") == invoice.issued_at
    assert all(a["time"] == OCEL_OBJECT_TIME for o in log["objects"] for a in o["attributes"])


def test_ocel_events_match_trajectory() -> None:
    log, steps = _log(), _steps()
    assert len(log["events"]) == len(steps) == 100
    assert [e["id"] for e in log["events"]] == [f"evt-{s['step']:04d}" for s in steps]
    assert [e["type"] for e in log["events"]] == [s["observation"]["tool"] for s in steps]
    assert {e["type"] for e in log["events"]} <= set(TOOL_NAMES)
    episode_id = _public().manifest.episode_id
    for event, step in zip(log["events"], steps, strict=True):
        assert {"objectId": episode_id, "qualifier": QUALIFIER_EPISODE} in event["relationships"]
        assert _attribute(event, "step") == step["step"]
        assert _attribute(event, "ok") is step["observation"]["ok"]
        if step["observation"]["tool"] == "list_records" and step["observation"]["ok"]:
            records = step["observation"]["result"]["records"]
            assert _attribute(event, "n_records") == len(records)
    submit_event = log["events"][-1]
    flagged = {
        r["objectId"] for r in submit_event["relationships"] if r["qualifier"] == QUALIFIER_FLAGS
    }
    assert flagged == {o["id"] for o in _of_type(log, TYPE_FINDING)} and len(flagged) == 8
    first_doc = next(
        (e, s)
        for e, s in zip(log["events"], steps, strict=True)
        if s["observation"]["tool"] == "get_document" and s["observation"]["ok"]
    )
    event, step = first_doc
    document = step["observation"]["result"]["document"]
    target = document.get("invoice_id") or document.get("payment_id")
    assert {"objectId": target, "qualifier": QUALIFIER_READ} in event["relationships"]
    assert _attribute(event, "doc_id") == step["action"]["args"]["doc_id"]


def test_ocel_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    texts = [
        ocel_json(
            build_ocel_log(
                read_public_episode(SAMPLE_EPISODE_DIR), read_trajectory(SAMPLE_TRAJECTORY)
            )
        )
        for _ in range(2)
    ]
    digests = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts}
    assert len(digests) == 1
    script = _load_script()
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for out in outputs:
        assert script.main(["--trajectory", str(SAMPLE_TRAJECTORY), "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "schema ocel.schema.json: PASS" in stdout and "leak scan: clean" in stdout
    assert outputs[0].read_bytes() == outputs[1].read_bytes() == texts[0].encode("utf-8")
    assert SAMPLE_LOG.is_file(), "run `make ocel` and commit docs/sample_ocel_log.json"
    assert SAMPLE_LOG.read_bytes() == outputs[0].read_bytes(), "docs/sample_ocel_log.json drifted"


def test_ocel_timestamps_offset_aware() -> None:
    log = _log()
    base = datetime.fromisoformat(OCEL_TIME_BASE)
    assert base.tzinfo is not None
    previous: datetime | None = None
    for event, step in zip(log["events"], _steps(), strict=True):
        moment = datetime.fromisoformat(event["time"])
        assert moment.tzinfo is not None and event["time"].endswith("+05:00")
        assert moment == base + timedelta(seconds=step["step"] * OCEL_STEP_SECONDS)
        assert previous is None or moment > previous
        previous = moment
    assert log["events"][0]["time"] == "2026-04-01T09:00:01+05:00"
    for obj in log["objects"]:
        for attribute in obj["attributes"]:
            assert datetime.fromisoformat(attribute["time"]).tzinfo is not None


@pytest.fixture
def _cached_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    store: dict[tuple[int, str], object] = {}

    def cached(seed: int, config: object = None) -> object:
        key = (seed, "" if config is None else config.model_dump_json())  # type: ignore[attr-defined]
        if key not in store:
            store[key] = build_episode(seed, config)  # type: ignore[arg-type]
        return store[key]

    monkeypatch.setattr(env_module, "build_episode", cached)


def _episode_dirs() -> dict[str, Path]:
    """episode_id -> directory from data/manifest.json (only ``dir``/``episode_id`` are read)."""
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    return {
        str(entry["episode_id"]): REPO_ROOT / "data" / str(entry["dir"])
        for split in manifest["splits"]
        for entry in split["episodes"]
    }


@pytest.mark.slow
@pytest.mark.usefixtures("_cached_builds")
def test_ocel_all_baselines(tmp_path: Path) -> None:
    """Every baseline trajectory over the dev split exports to a valid, sound, leak-free log."""
    for name in BASELINE_NAMES:
        run_split(BASELINE_FACTORIES[name], "dev", trajectory_dir=tmp_path)
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 4 * 10
    dirs = _episode_dirs()
    for path in files:
        steps = read_trajectory(path)
        public = read_public_episode(dirs[str(steps[0]["episode_id"])])
        log = build_ocel_log(public, steps)
        assert _schema_errors(log) == [], path.name
        assert validate_ocel(log) == [], path.name
        assert scan_ocel(log) == [], path.name
        assert len(log["events"]) == len(steps), path.name
        assert len(_of_type(log, TYPE_INVOICE)) == len(public.invoices), path.name
