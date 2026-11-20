"""The OCEL export is built from public data only: leak scan of the log, a private-side check
against the hidden episode (test side only), scanner positive controls and import isolation of
the exporter module."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from fintrace.constants import OCEL_OBJECT_TIME
from fintrace.exporters.episode_files import read_episode, read_public_episode
from fintrace.exporters.ocel import build_ocel_log, ocel_json, read_trajectory, scan_ocel
from fintrace.logging.trajectory import FORBIDDEN_KEY_FRAGMENTS, FORBIDDEN_KEYS, FORBIDDEN_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_TRAJECTORY = REPO_ROOT / "trajectories" / "rule_based_dev_10000.jsonl"
SAMPLE_EPISODE_DIR = REPO_ROOT / "data" / "dev" / "ep_b2f2e0741428"
EXPORTER_SOURCE = REPO_ROOT / "src" / "fintrace" / "exporters" / "ocel.py"
FORBIDDEN_IMPORT_PREFIXES = (
    "fintrace.evaluator",
    "fintrace.generator",
    "fintrace.environment",
    "fintrace.exporters.episode_files",  # can open truth.json / provenance.json
)
FORBIDDEN_SOURCE_WORDS = ("privateepisode", "read_episode", "truth", "provenance", "seed")


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        found = {str(k).lower() for k in value}
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found |= _keys(item)
        return found
    return set()


def _bad_name(name: str) -> bool:
    return name in FORBIDDEN_KEYS or any(f in name for f in FORBIDDEN_KEY_FRAGMENTS)


def test_ocel_no_leak() -> None:
    public = read_public_episode(SAMPLE_EPISODE_DIR)
    log = build_ocel_log(public, read_trajectory(SAMPLE_TRAJECTORY))
    text = ocel_json(log)
    lowered = text.lower()
    assert scan_ocel(log) == []
    for token in (*FORBIDDEN_TOKENS, "iss-", '"seed"', "truth.json", "provenance.json"):
        assert token not in lowered, token
    assert not [k for k in _keys(log) if _bad_name(k)]
    names = {
        str(a["name"]).lower()
        for section in ("objectTypes", "eventTypes", "objects", "events")
        for item in log[section]
        for a in item["attributes"]
    }
    assert not [n for n in names if _bad_name(n)]
    # private side (test only): nothing of the hidden episode made it into the log
    private = read_episode(SAMPLE_EPISODE_DIR)
    assert private.truth.issues, "the sample episode carries injected issues"
    assert private.provenance.seed_hash not in text
    for issue in private.truth.issues:
        assert issue.issue_id not in text
    assert str(private.provenance.history[0]["company"]) not in text
    # positive controls: the scanner fires on a planted attribute and on a planted key
    planted = copy.deepcopy(log)
    planted["objects"][0]["attributes"].append(
        {"name": "seed_count", "time": OCEL_OBJECT_TIME, "value": "ground truth"}
    )
    hits = scan_ocel(planted)
    assert "token truth" in hits and "attribute seed_count" in hits, hits
    planted = copy.deepcopy(log)
    planted["episode_seed"] = 10000
    assert "key episode_seed" in scan_ocel(planted)
    # import isolation: the exporter sees the public episode and the trajectory, nothing else
    source = EXPORTER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    plain_imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    modules |= {alias.name for node in plain_imports for alias in node.names}
    assert not [m for m in modules if m.startswith(FORBIDDEN_IMPORT_PREFIXES)], modules
    from_imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported = {alias.name for node in from_imports for alias in node.names}
    assert "PublicEpisode" in imported and "PrivateEpisode" not in imported
    lowered_source = source.lower()
    assert not [w for w in FORBIDDEN_SOURCE_WORDS if w in lowered_source]
