"""Environment Card (D5): required sections, numbers equal to their sources, byte-stable markdown,
hard failure when a source is missing, structurally valid one-page PDF, PDF library dev-only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import tomllib
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
CARD_MD = DOCS / "environment_card.md"
CARD_PDF = DOCS / "environment_card.pdf"
BASELINES = ("oracle", "rule_based", "mark_everything", "random")
BASELINE_ROW = re.compile(r"^\| (oracle|rule_based|mark_everything|random) \| ([+-]\d\.\d{3}) \|")
LEAK_TEST = REPO_ROOT / "tests" / "security" / "test_leak_scan_full.py"
FUZZ_TEST = REPO_ROOT / "tests" / "contract" / "test_env_fuzz.py"


@cache
def _script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "build_environment_card.py"
    spec = importlib.util.spec_from_file_location("build_environment_card", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _metrics() -> dict[str, Any]:
    return _script().collect_metrics()


@cache
def _markdown() -> str:
    return _script().render_markdown(_metrics())


def _constant(path: Path, name: str) -> int:
    match = re.search(rf"^{name} = ([0-9_]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, (path, name)
    return int(match.group(1))


def _json(name: str) -> dict[str, Any]:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def test_card_md_has_required_sections() -> None:
    script = _script()
    markdown = _markdown()
    assert markdown.startswith("# FINTRACE Environment Card\n")
    for heading in script.SECTIONS:
        assert f"\n{heading}\n" in markdown, heading
    assert f"card date {script.CARD_DATE}" in markdown and "datetime" not in markdown
    assert CARD_MD.is_file(), "run `make card` and commit docs/environment_card.md"
    committed = CARD_MD.read_text(encoding="utf-8")
    for heading in script.SECTIONS:
        assert f"\n{heading}\n" in committed, heading


def test_card_numbers_match_sources() -> None:
    """Every figure in the card equals the value read here, independently, from its source."""
    markdown = _markdown()
    attacks = _json("results_attacks.json")
    rows = attacks["attackers"]
    defeated = sum(1 for r in rows if r["defeated"])
    worst = max(float(r["stats"]["mean_reward"]) for r in rows)
    floor = sum(1 for r in rows if float(r["stats"]["max_reward"]) <= -1.0)
    reference = float(attacks["reference"]["stats"]["mean_reward"])
    assert f"{defeated}/{len(rows)} attackers defeated" in markdown
    assert f"worst mean {worst:+.3f}" in markdown
    assert f"{floor}/{len(rows)} on the -1.000 floor" in markdown
    assert f"reference {attacks['reference']['name']} {reference:+.3f}" in markdown
    gap = _json("oracle_gap_results.json")
    error_types = [t for t in gap["types"] if t != "clean"]
    rule_based = {t: float(gap["gap"]["rule_based"][t]) for t in error_types}
    spread = max(rule_based.values()) - min(rule_based.values())
    assert f"spread {spread:.2f}" in markdown
    for error_type in error_types:
        assert f"{error_type}: {rule_based[error_type]:.2f}" in markdown
    attackers = [p for p in gap["policies"] if p not in BASELINES]
    attacker_gaps = [float(gap["gap"][p][t]) for p in attackers for t in error_types]
    assert f"attackers in [{min(attacker_gaps):.2f}, {max(attacker_gaps):.2f}]" in markdown
    design = f"{len(gap['policies'])} policies x {len(gap['types'])} types x {gap['n_per_type']}"
    assert f"{design} paired episodes" in markdown
    baselines: dict[str, float] = {}
    for line in (DOCS / "results_dev.md").read_text(encoding="utf-8").splitlines():
        match = BASELINE_ROW.match(line)
        if match and match.group(1) not in baselines:
            baselines[match.group(1)] = float(match.group(2))
    assert set(baselines) == set(BASELINES)
    for policy, mean in baselines.items():
        assert f"{policy} {mean:+.3f}" in markdown, policy
    manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    counts = {s["split"]: len(s["episodes"]) for s in manifest["splits"]}
    assert f"{counts['train']} train + {counts['dev']} dev committed episodes" in markdown
    assert f"`{manifest['config_hash']}`" in markdown
    episodes = _constant(LEAK_TEST, "EPISODES_REQUIRED")
    steps = _constant(LEAK_TEST, "STEPS_PER_EPISODE")
    fuzz = _constant(FUZZ_TEST, "FUZZ_ITERATIONS")
    surfaces = f"{episodes} episodes x {steps + 1} surfaces ({episodes * (steps + 1)})"
    assert f"enforced by tests scanning {surfaces} and {fuzz} fuzzed actions; any hit" in markdown
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    digest = re.search(r"@sha256:([0-9a-f]{64})", dockerfile)
    assert digest is not None and digest.group(1) in markdown
    # the provenance table names a source for every number of the Metrics section
    provenance = _script().provenance_rows(_metrics())
    assert len(provenance) >= 15 and all(src for _, _, src in provenance)
    for name, value, src in provenance:
        assert f"| {name} | {value} | `{src}` |" in markdown, name


def test_card_md_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _script()
    first, second = script.render_markdown(script.collect_metrics()), _markdown()
    assert first == second
    outputs = [tmp_path / "a.md", tmp_path / "b.md"]
    for out in outputs:
        assert script.main(["--out-md", str(out), "--no-pdf"]) == 0
    capsys.readouterr()
    digests = {hashlib.sha256(out.read_bytes()).hexdigest() for out in outputs}
    assert len(digests) == 1 and outputs[0].read_bytes() == first.encode("utf-8")


def test_card_fails_without_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _script()
    partial = tmp_path / "docs"
    partial.mkdir()
    for name in ("oracle_gap_results.json", "results_dev.md"):
        shutil.copy(DOCS / name, partial / name)  # results_attacks.json deliberately absent
    out = tmp_path / "card.md"
    args = ["--docs-dir", str(partial), "--out-md", str(out), "--no-pdf"]
    assert script.main(args) == script.EXIT_SOURCE
    assert "results_attacks.json" in capsys.readouterr().err and not out.exists()
    shutil.copy(DOCS / "results_attacks.json", partial / "results_attacks.json")
    broken = json.loads((partial / "oracle_gap_results.json").read_text(encoding="utf-8"))
    del broken["gap"]
    (partial / "oracle_gap_results.json").write_text(json.dumps(broken), encoding="utf-8")
    assert script.main(args) == script.EXIT_SOURCE
    assert "missing field 'gap'" in capsys.readouterr().err and not out.exists()
    with pytest.raises(script.CardSourceError):
        script.leak_scan_metrics(tmp_path)  # no test files -> no scan constants -> no number


@pytest.mark.slow
def test_card_pdf_valid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """PDF checked structurally (magic, trailer, size, one page, date); committed md pinned."""
    script = _script()
    out_pdf = tmp_path / "card.pdf"
    args = ["--out-md", str(tmp_path / "card.md"), "--out-pdf", str(out_pdf)]
    assert script.main(args) == 0
    assert "1 page" in capsys.readouterr().out
    data = out_pdf.read_bytes()
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    assert len(data) > 2_000  # compressed streams: a one-page card is small
    count = re.search(rb"/Type /Pages[^>]*?/Count (\d+)", data) or re.search(rb"/Count (\d+)", data)
    assert count is not None and int(count.group(1)) == 1
    assert b"/CreationDate (D:" + script.CARD_DATE.replace("-", "").encode() in data
    assert CARD_PDF.is_file(), "run `make card` and commit docs/environment_card.pdf"
    committed = CARD_PDF.read_bytes()
    assert committed.startswith(b"%PDF-") and len(committed) > 2_000
    committed_count = re.search(rb"/Count (\d+)", committed)
    assert committed_count is not None and int(committed_count.group(1)) == 1
    assert b"/CreationDate (D:" + script.CARD_DATE.replace("-", "").encode() in committed
    # the committed markdown is pinned here, with the other regenerated artefacts (slow suite)
    rebuilt = (tmp_path / "card.md").read_bytes()
    assert CARD_MD.read_bytes() == rebuilt, "docs/environment_card.md drifted: run `make card`"


def test_pdf_library_is_dev_only() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "fpdf" not in requirements and "reportlab" not in requirements
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any(dep.lower().startswith("fpdf") for dep in project["dependencies"])
    assert any(dep.startswith("fpdf2==") for dep in project["optional-dependencies"]["dev"])
