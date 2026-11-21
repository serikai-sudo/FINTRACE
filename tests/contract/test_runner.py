"""Runner contract and the agents -> evaluator import firewall (static scan + runtime check)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from fintrace.agents.random_agent import RandomInvestigator
from fintrace.constants import MAX_STEPS
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import (
    BASELINE_FACTORIES,
    load_split_config,
    run_baseline,
    run_split,
    split_seeds,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
AGENTS_DIR = SRC_DIR / "fintrace" / "agents"
FORBIDDEN_PACKAGES = ("evaluator", "reward", "environment", "generator")
# escape hatches an agent could use instead of a plain import statement
FORBIDDEN_STDLIB = ("importlib", "sys", "os", "pathlib", "io", "shutil", "subprocess")
FORBIDDEN_NAMES = ("__import__", "import_module", "open", "DATA_DIR", "CONFIG_DIR")
FORBIDDEN_ATTRIBUTES = (
    "read_text", "read_bytes", "_private", "_truth_store", "_journal", "evaluator_journal",
)  # fmt: skip
RESULT_KEYS = {
    "policy", "seed", "episode_id", "steps", "reward", "terminated", "truncated", "n_findings",
    "elapsed_sec",
}  # fmt: skip


def test_run_baseline_returns_metrics() -> None:
    env = FintraceEnv()
    result = run_baseline(RandomInvestigator(seed=0), env, 42)
    assert set(result) == RESULT_KEYS
    assert result["policy"] == "random" and result["seed"] == 42
    assert result["episode_id"] == env.episode.manifest.episode_id
    assert 1 <= result["steps"] <= MAX_STEPS
    assert isinstance(result["reward"], float) and -1.0 <= result["reward"] <= 1.0
    assert result["terminated"] or result["truncated"]
    assert isinstance(result["elapsed_sec"], float) and result["elapsed_sec"] >= 0.0
    stats = summarize([result])
    assert stats["n"] == 1 and stats["mean_reward"] == result["reward"]
    assert stats["std_reward"] == 0.0 and stats["mean_steps"] == result["steps"]
    assert summarize([])["n"] == 0


def test_run_split_covers_all_episodes() -> None:
    results = run_split(BASELINE_FACTORIES["random"], "dev")
    assert len(results) == 10
    assert [r["seed"] for r in results] == split_seeds("dev")
    assert len({r["episode_id"] for r in results}) == 10
    assert [r["seed"] for r in run_split(BASELINE_FACTORIES["random"], "dev", limit=3)] == (
        split_seeds("dev")[:3]
    )
    assert load_split_config("dev").errors.total() == 8
    with pytest.raises(KeyError):
        split_seeds("test")


# ---------------------------------------------------------------- import firewall


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for an attribute chain rooted at a name, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_forbidden_module(name: str) -> bool:
    parts = name.split(".")
    if parts[0] in FORBIDDEN_STDLIB:
        return True
    return parts[0] == "fintrace" and len(parts) > 1 and parts[1] in FORBIDDEN_PACKAGES


def forbidden_uses(path: Path, package: tuple[str, ...]) -> list[str]:
    """Every way ``path`` (a module inside ``package``) could reach a forbidden package.

    Covers absolute and relative imports (resolved against ``package``, so ``from .. import
    evaluator`` inside ``fintrace.agents`` is ``fintrace.evaluator``), attribute chains such as
    ``fintrace.evaluator.testing``, dynamic imports, string constants naming a forbidden module,
    and the file / private-attribute escape hatches.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: level 1 = this package, each extra level one parent up
                base = package[: len(package) - (node.level - 1)]
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                module = node.module or ""
            names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            names = [dotted] if dotted else []
            if node.attr in FORBIDDEN_ATTRIBUTES:
                hits.append(f"{path.name}:{node.lineno}: attribute .{node.attr}")
        elif isinstance(node, ast.Name):
            names = []
            if node.id in FORBIDDEN_NAMES:
                hits.append(f"{path.name}:{node.lineno}: name {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value] if node.value.startswith("fintrace.") else []
            if "evaluator.testing" in node.value:
                hits.append(f"{path.name}:{node.lineno}: string {node.value!r}")
        else:
            continue
        for name in names:
            if _is_forbidden_module(name):
                hits.append(f"{path.name}:{node.lineno}: {name}")
    return hits


def _package_of(path: Path) -> tuple[str, ...]:
    return tuple(path.relative_to(SRC_DIR).with_suffix("").parts[:-1])


def test_oracle_not_accessible_from_agents() -> None:
    """No module under fintrace.agents can reach the evaluator, reward, environment or generator."""
    files = sorted(AGENTS_DIR.rglob("*.py"))
    assert len(files) >= 5
    hits = [hit for path in files for hit in forbidden_uses(path, _package_of(path))]
    assert hits == [], "agents must stay on the public surface:\n" + "\n".join(hits)
    # runtime: importing every agent module in a fresh interpreter loads none of the packages
    modules = [".".join(path.relative_to(SRC_DIR).with_suffix("").parts) for path in files]
    code = (
        "import sys\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
        "print(sorted(m for m in sys.modules if m.startswith('fintrace.')))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter, literal code, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    loaded = completed.stdout.strip()
    for package in FORBIDDEN_PACKAGES:
        assert f"fintrace.{package}" not in loaded, loaded


BAD_SNIPPETS = (
    "import fintrace.evaluator.testing as t\n",
    "from fintrace.evaluator import testing\n",
    "from fintrace import evaluator\n",
    "from ..evaluator import testing\n",
    "from .. import evaluator\n",
    "from ..reward.official import official_reward\n",
    "import fintrace\nx = fintrace.evaluator.testing.oracle_for\n",
    "import importlib\n",
    "m = __import__('fintrace.evaluator.testing')\n",
    "name = 'fintrace.evaluator.testing'\n",
    "def f(env):\n    return env._private\n",
    "from pathlib import Path\nPath('data').read_text()\n",
)
CLEAN_SNIPPETS = (
    "from .base import Policy\n",
    "from fintrace.models import JsonObject\nfrom fintrace.constants import MAX_STEPS\n",
    "import random\nrng = random.Random(0)\n",
)


@pytest.mark.parametrize("snippet", BAD_SNIPPETS)
def test_firewall_scanner_catches_escape_hatches(tmp_path: Path, snippet: str) -> None:
    path = tmp_path / "bad_agent.py"
    path.write_text(snippet, encoding="utf-8")
    assert forbidden_uses(path, ("fintrace", "agents")), snippet


@pytest.mark.parametrize("snippet", CLEAN_SNIPPETS)
def test_firewall_scanner_accepts_public_surface(tmp_path: Path, snippet: str) -> None:
    path = tmp_path / "good_agent.py"
    path.write_text(snippet, encoding="utf-8")
    assert forbidden_uses(path, ("fintrace", "agents")) == []


def test_firewall_resolves_relative_imports_against_nested_packages(tmp_path: Path) -> None:
    """``from ..x`` means fintrace.agents.x inside fintrace.agents.attacks, fintrace.x in agents."""
    path = tmp_path / "attack.py"
    path.write_text("from ..rule_based import RuleBasedReconciliation\n", encoding="utf-8")
    assert forbidden_uses(path, ("fintrace", "agents", "attacks")) == []
    path.write_text("from ...evaluator import testing\n", encoding="utf-8")
    assert forbidden_uses(path, ("fintrace", "agents", "attacks"))
