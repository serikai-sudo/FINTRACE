"""The attackers live on the public surface: no evaluator, reward, environment or generator.

The fresh-interpreter import check in test_runner.py already covers every module under
fintrace.agents, attacks/ included; this file adds the static scan and the literal-word check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTACKS_DIR = REPO_ROOT / "src" / "fintrace" / "agents" / "attacks"


def _firewall() -> ModuleType:
    """The AST scanner of the baselines test, loaded by path (tests/ is not a package)."""
    path = REPO_ROOT / "tests" / "contract" / "test_runner.py"
    spec = importlib.util.spec_from_file_location("firewall", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_evaluator_imports_in_attacks() -> None:
    firewall = _firewall()
    files = sorted(ATTACKS_DIR.rglob("*.py"))
    assert len(files) == 22  # base, registry, 20 attackers
    hits = [
        hit for path in files for hit in firewall.forbidden_uses(path, firewall._package_of(path))
    ]
    assert hits == [], "attackers must stay on the public surface:\n" + "\n".join(hits)
    for path in files:  # belt and braces: the literal words never appear at all
        text = path.read_text(encoding="utf-8")
        assert "evaluator" not in text and "_private" not in text and "truth" not in text, path.name
