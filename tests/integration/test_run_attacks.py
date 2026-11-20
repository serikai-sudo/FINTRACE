"""Slow end-to-end: the attack sweep script over the dev split and its committed artefacts."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType

import pytest

import fintrace.environment.env as env_module
from fintrace.generator.build import build_episode

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_attacks.py"
    spec = importlib.util.spec_from_file_location("run_attacks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _cached_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 20 attackers reset the same 10 dev episodes 200 times: build each once."""
    store: dict[tuple[int, str], object] = {}

    def cached(seed: int, config: object = None) -> object:
        key = (seed, "" if config is None else config.model_dump_json())  # type: ignore[attr-defined]
        if key not in store:
            store[key] = build_episode(seed, config)  # type: ignore[arg-type]
        return store[key]

    monkeypatch.setattr(env_module, "build_episode", cached)


@pytest.mark.slow
def test_run_attacks_full_dev_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    assert script.main(["--split", "dev", "--docs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count("DEFEATED") == 20 and "HOLE" not in out
    results = json.loads((tmp_path / "results_attacks.json").read_text(encoding="utf-8"))
    assert len(results["attackers"]) == 20
    assert sum(len(row["episodes"]) for row in results["attackers"]) == 200
    assert all(float(row["stats"]["mean_reward"]) <= -0.3 for row in results["attackers"])
    assert all(row["defeated"] for row in results["attackers"])
    report = (tmp_path / "reward_robustness_report.md").read_text(encoding="utf-8")
    assert report.startswith("# FINTRACE Reward Robustness Report")
    assert "Pillar 4 (Adversarial Robustness)" in report
    header = "| Attacker | Strategy | Mean reward | Std | Min | Max | Mean steps | Verdict |"
    assert header in report
    assert report.count("| DEFEATED |") == 20 and "HOLE" not in report
    png = (tmp_path / "figures" / "robustness_chart.png").read_bytes()
    assert png[:8] == PNG_MAGIC and len(png) > 10_000
    assert struct.unpack(">II", png[16:24]) == (1200, 800)
    # the committed artefacts are exactly what this run produced (json + report; png by size)
    for name in ("results_attacks.json", "reward_robustness_report.md"):
        committed = DOCS / name
        assert committed.is_file(), f"run `make attacks` and commit docs/{name}"
        assert committed.read_bytes() == (tmp_path / name).read_bytes(), f"{name} drifted"
    assert (DOCS / "figures" / "robustness_chart.png").stat().st_size == len(png)
