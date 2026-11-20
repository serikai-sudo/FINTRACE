"""D4 guard rails in the torch-free main suite: torch never reaches src/, the adapter, the
scripts, the Dockerfile, the runtime or dev dependencies or the lint scope; importing the runtime
loads neither torch nor numpy; the BC featurizer is deterministic and imports no torch."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
BC_DIR = REPO_ROOT / "experiments" / "bc_proof"
SAMPLE_TRAJECTORY = REPO_ROOT / "trajectories" / "rule_based_dev_10000.jsonl"
SCANNED = (
    *sorted((REPO_ROOT / "src" / "fintrace").rglob("*.py")),
    REPO_ROOT / "judge_adapter.py",
    *sorted((REPO_ROOT / "scripts").glob("*.py")),
)


def _imports_torch(path: Path) -> bool:
    """Static or dynamic: `import torch`, `from torch...`, and the literal "torch" that
    importlib.import_module / __import__ would need."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "torch" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            return True
        elif isinstance(node, ast.Constant) and node.value == "torch":
            return True
    return False


def _load_featurizer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bc_featurize", BC_DIR / "featurize.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bc_proof_isolated() -> None:
    assert len(SCANNED) > 40
    offenders = [str(p.relative_to(REPO_ROOT)) for p in SCANNED if _imports_torch(p)]
    assert offenders == [], offenders
    runtime_pins = [
        line.strip().lower()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")  # the header comment mentions torch by name
    ]
    assert runtime_pins and not any(pin.startswith("torch") for pin in runtime_pins), runtime_pins
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
    assert not any(dep.lower().startswith("torch") for dep in declared), declared
    bc_requirements = (BC_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert any(line.startswith("torch==") for line in bc_requirements.splitlines())
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    lint_line = next(line for line in makefile.splitlines() if "ruff check" in line)
    assert "experiments" not in lint_line
    assert "torch" not in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").lower()
    probe = (
        "import importlib, pkgutil, sys, fintrace, judge_adapter\n"
        "for info in pkgutil.walk_packages(fintrace.__path__, 'fintrace.'):\n"
        "    importlib.import_module(info.name)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'numpy')))\n"
    )
    env = {"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}", "PATH": "/usr/bin:/bin"}
    result = subprocess.run(  # noqa: S603  # fixed interpreter and argv, no shell
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "[]", result.stdout  # no torch, no numpy in the runtime graph
    assert "torch" not in sys.modules  # and this suite itself never imported it


def test_bc_featurizer_deterministic() -> None:
    before = {m for m in sys.modules if m.split(".")[0] in ("torch", "numpy")}
    featurize = _load_featurizer()
    after = {m for m in sys.modules if m.split(".")[0] in ("torch", "numpy")}
    assert after == before  # the featurizer pulls in neither torch nor numpy
    assert len(featurize.FEATURE_NAMES) == featurize.N_FEATURES == 37
    assert len(set(featurize.FEATURE_NAMES)) == featurize.N_FEATURES
    assert featurize.TOOLS == ("list_records", "get_record", "get_document", "calculator", "submit")
    assert len(featurize.FINE_CLASSES) == 8
    first = featurize.featurize_file(SAMPLE_TRAJECTORY)
    second = featurize.featurize_file(SAMPLE_TRAJECTORY)
    assert first == second
    rows, tools, fine = first
    assert len(rows) == len(tools) == len(fine) == 100
    assert all(len(row) == featurize.N_FEATURES for row in rows)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for row in rows for v in row)
    assert all(0 <= t < len(featurize.TOOLS) for t in tools)
    assert all(0 <= f < len(featurize.FINE_CLASSES) for f in fine)
    assert featurize.TOOLS[tools[-1]] == "submit" and featurize.FINE_CLASSES[fine[-1]] == "submit"
    assert featurize.FINE_CLASSES[fine[0]] == "list_records:invoices"
    # the first row is the empty prefix: no pages, no documents, nothing listed
    assert rows[0][0] == 0.0 and rows[0][1] == 1.0 and sum(rows[0][2:]) == 1.0  # only last_ok = 1
    assert rows[-1][0] > rows[0][0] and rows[-1][1] < rows[0][1]
