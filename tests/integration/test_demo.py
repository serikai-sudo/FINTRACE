"""``python -m fintrace`` / ``make demo``: the live demo episode runs end to end, prints the agent's
notes and the seven score fields, is byte-stable, needs nothing but ``src/`` and makes no network
call (the socket guard of the jury-protocol test)."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fintrace.__main__ import SCORE_KEYS, main
from fintrace.constants import DEMO_SEED, MAX_STEPS

from .test_jury_protocol_offline import Attempt, _positive_control
from .test_jury_protocol_offline import _no_network as no_network_guard  # noqa: F401  # fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_LINE = re.compile(
    r"^  tp (\d+)  fp (\d+)  fn (\d+)  precision ([0-9.]+)  recall ([0-9.]+)  f1 ([0-9.]+)"
    r"  return (-?[0-9.]+)$",
    re.MULTILINE,
)


def _run_in_process(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    code = main(argv, out=buffer)
    return code, buffer.getvalue()


def test_demo_runs_and_scores(tmp_path: Path) -> None:
    code, text = _run_in_process([])
    assert code == 0
    assert f"seed {DEMO_SEED}" in text
    assert "  step   1  list_records invoices" in text  # the investigation starts with a scan
    assert re.search(r"^  step \d+  submit \(\d+ findings\)$", text, re.MULTILINE)
    assert "recorded" in text  # at least one of the agent's own conclusions is shown
    assert re.search(r"^Findings submitted \([1-9]\d*\)$", text, re.MULTILINE)
    assert "The agent never sees this line." in text
    scores = SCORE_LINE.findall(text)
    assert len(scores) == 1
    tp, fp, fn = (int(v) for v in scores[0][:3])
    precision, recall, f1, ret = (float(v) for v in scores[0][3:])
    assert tuple(SCORE_KEYS) == ("tp", "fp", "fn", "precision", "recall", "f1", "return")
    assert precision == round(tp / (tp + fp), 3) if tp + fp else precision == 0.0
    assert recall == round(tp / (tp + fn), 3) if tp + fn else recall == 0.0
    expected_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    assert abs(f1 - expected_f1) < 0.002  # both sides rounded to three decimals
    assert -1.0 <= ret <= 1.0 and ret > 0.5  # a real investigation, not an empty submission
    assert "(equal)" in text and "MISMATCH" not in text
    assert "inspection steps" in text and f"of {MAX_STEPS}" in text
    # the same text from `python -m fintrace` with only a copy of src/ on the path and an empty
    # working directory: no config/, data/ or scripts/ are needed (what the image ships)
    shutil.copytree(REPO_ROOT / "src" / "fintrace", tmp_path / "pkg" / "fintrace")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path / "pkg"), "PYTHONHASHSEED": "1"}
    (tmp_path / "empty_cwd").mkdir()
    result = subprocess.run(  # fixed interpreter and argv, no shell
        [sys.executable, "-m", "fintrace"],
        cwd=tmp_path / "empty_cwd",
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == text


def test_demo_deterministic() -> None:
    first = _run_in_process([])
    second = _run_in_process([])
    assert first == second and first[0] == 0
    full = _run_in_process(["--full"])
    assert full[0] == 0 and full == _run_in_process(["--full"])
    assert full[1].count("\n  step ") == MAX_STEPS  # every step is shown with --full
    assert first[1].count("\n  step ") < MAX_STEPS  # the default shows key steps only
    assert SCORE_LINE.findall(full[1]) == SCORE_LINE.findall(first[1])  # same score either way


def test_demo_runs_offline(request: pytest.FixtureRequest) -> None:
    attempts: list[Attempt] = request.getfixturevalue("no_network_guard")  # the socket guard
    _positive_control(attempts)
    code, text = _run_in_process([])
    assert code == 0 and SCORE_LINE.search(text) is not None
    assert attempts == [], f"the demo tried to use the network: {attempts}"
