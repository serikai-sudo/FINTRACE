"""Build the FINTRACE Environment Card from committed artefacts (no number is typed by hand).

Usage::

    python scripts/build_environment_card.py [--docs-dir docs] [--out-md docs/environment_card.md]
                                             [--out-pdf docs/environment_card.pdf] [--no-pdf]

Every metric is read from a committed source: ``docs/results_attacks.json`` (adversarial
robustness), ``docs/oracle_gap_results.json`` (differential signal), ``docs/results_dev.md``
(baselines), ``data/manifest.json`` (dataset), the constants of the leak-scan and fuzz tests
(scan volumes), the test tree itself (test inventory), ``config/dev.yaml`` and the package
constants (environment facts) and the ``Dockerfile`` (image digest). A missing source or field
raises ``CardSourceError`` and the build exits with code 2 instead of inventing a value.

The markdown is a pure function of those sources (``CARD_DATE`` is a constant, never the wall
clock), so a rebuild is byte-identical and the committed card is pinned by a test. The PDF
(fpdf2, a dev-only pure-Python dependency) carries the same fixed creation date; it is checked
structurally, not byte for byte.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

import fintrace
from fintrace.constants import (
    ISSUE_TYPES,
    MAX_FINDINGS,
    MAX_PAGE_SIZE,
    MAX_STEPS,
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    REWARD_F1_BIAS,
    REWARD_F1_COEF,
    REWARD_STEP_COST,
    TIME_BUDGET_SECONDS,
    TOOL_NAMES,
)
from fintrace.models import DecoyCounts

CARD_DATE: Final[str] = "2026-11-20"  # fixed build date: the card never reads the wall clock
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DOCS_DIR: Final[Path] = REPO_ROOT / "docs"
DATA_DIR: Final[Path] = REPO_ROOT / "data"
TESTS_DIR: Final[Path] = REPO_ROOT / "tests"
CONFIG_DIR: Final[Path] = REPO_ROOT / "config"
OUT_MD: Final[Path] = DOCS_DIR / "environment_card.md"
OUT_PDF: Final[Path] = DOCS_DIR / "environment_card.pdf"
ATTACKS_FILE: Final[str] = "results_attacks.json"
GAP_FILE: Final[str] = "oracle_gap_results.json"
BASELINES_FILE: Final[str] = "results_dev.md"
BASELINE_ORDER: Final[tuple[str, ...]] = ("oracle", "rule_based", "mark_everything", "random")
LEAK_TEST: Final[str] = "security/test_leak_scan_full.py"
FUZZ_TEST: Final[str] = "contract/test_env_fuzz.py"
JURY_CPUS: Final[int] = 2  # CLAUDE.md section 1, compute budget on the jury machine
JURY_MEMORY_GB: Final[int] = 4
SECTIONS: Final[tuple[str, ...]] = (
    "## Environment details",
    "## Data",
    "## Metrics",
    "## Intended use",
    "## Out of scope",
    "## Limitations",
    "## Reproducibility",
    "## Provenance of the numbers",
)
EXIT_SOURCE: Final[int] = 2
EXIT_PDF: Final[int] = 1
PDF_FONT: Final[str] = "Helvetica"
Metrics = dict[str, Any]


class CardSourceError(RuntimeError):
    """A source file or field the card needs is missing: the build stops, nothing is invented."""


# ---------------------------------------------------------------- source readers
def _load_json(path: Path) -> Any:  # noqa: ANN401
    if not path.is_file():
        raise CardSourceError(f"missing source {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CardSourceError(f"{path} is not valid JSON") from exc


def _require(mapping: Any, key: str, where: str) -> Any:  # noqa: ANN401
    if not isinstance(mapping, dict) or key not in mapping:
        raise CardSourceError(f"{where}: missing field {key!r}")
    return mapping[key]


def attack_metrics(docs_dir: Path) -> Metrics:
    """docs/results_attacks.json -> defeated count, worst/best attacker mean, floor, reference."""
    path = docs_dir / ATTACKS_FILE
    where = str(path)
    data = _load_json(path)
    rows = _require(data, "attackers", where)
    if not isinstance(rows, list) or not rows:
        raise CardSourceError(f"{path}: no attackers")
    stats = [_require(r, "stats", where) for r in rows]
    means = [float(_require(s, "mean_reward", where)) for s in stats]
    maxima = [float(_require(s, "max_reward", where)) for s in stats]
    reference = _require(data, "reference", where)
    reference_stats = _require(reference, "stats", where)
    worst = max(range(len(rows)), key=lambda i: (means[i], str(rows[i].get("name"))))
    return {
        "n_attackers": len(rows),
        "defeated": sum(1 for r in rows if _require(r, "defeated", where) is True),
        "worst_mean": means[worst],
        "worst_name": str(_require(rows[worst], "name", where)),
        "best_mean": min(means),
        "floor_count": sum(1 for m in maxima if m <= float(REWARD_CLIP_MIN)),
        "reference_name": str(_require(reference, "name", where)),
        "reference_mean": float(_require(reference_stats, "mean_reward", where)),
        "threshold": float(_require(data, "threshold", where)),
        "episodes_per_attacker": len(_require(rows[0], "episodes", where)),
        "split": str(_require(data, "split", where)),
    }


def oracle_gap_metrics(docs_dir: Path) -> Metrics:
    """docs/oracle_gap_results.json -> rule_based per-type gaps and spread, attacker floor."""
    path = docs_dir / GAP_FILE
    where = str(path)
    data = _load_json(path)
    types = [str(t) for t in _require(data, "types", where)]
    policies = [str(p) for p in _require(data, "policies", where)]
    gap = _require(data, "gap", where)
    error_types = [t for t in types if t != "clean"]
    if not error_types or "rule_based" not in policies or "oracle" not in policies:
        raise CardSourceError(f"{path}: unexpected types or policies")
    rule_based_row = _require(gap, "rule_based", where)
    rule_based = {t: float(_require(rule_based_row, t, where)) for t in error_types}
    attackers = [p for p in policies if p not in BASELINE_ORDER]
    attacker_gaps = [
        float(_require(_require(gap, p, where), t, where)) for p in attackers for t in error_types
    ]
    if not attacker_gaps:
        raise CardSourceError(f"{path}: no attacker rows")
    hardest = max(error_types, key=lambda t: (rule_based[t], t))
    easiest = min(error_types, key=lambda t: (rule_based[t], t))
    oracle_row = _require(_require(data, "mean_reward", where), "oracle", where)
    return {
        "n_policies": len(policies),
        "n_attackers": len(attackers),
        "n_types": len(types),
        "n_per_type": int(_require(data, "n_per_type", where)),
        "seeds": [int(s) for s in _require(data, "seeds", where)],
        "error_types": error_types,
        "rule_based_gaps": rule_based,
        "spread": rule_based[hardest] - rule_based[easiest],
        "hardest": hardest,
        "easiest": easiest,
        "attacker_gap_min": min(attacker_gaps),
        "attacker_gap_max": max(attacker_gaps),
        "oracle_clean_mean": float(_require(oracle_row, "clean", where)),
    }


_BASELINE_ROW: Final[re.Pattern[str]] = re.compile(
    r"^\| (?P<policy>[a-z_]+) \| (?P<mean>[+-]\d\.\d{3}) \| (?P<std>\d\.\d{3}) \| "
    r"(?P<min>[+-]\d\.\d{3}) \| (?P<max>[+-]\d\.\d{3}) \| (?P<steps>\d+\.\d) \|$"
)
_CONFIG_HASH: Final[re.Pattern[str]] = re.compile(r"config hash `([0-9a-f]{12})`")
_FIELDS: Final[tuple[str, ...]] = ("mean", "std", "min", "max", "steps")


def baseline_metrics(docs_dir: Path) -> Metrics:
    """docs/results_dev.md -> mean/std/min/max/steps per baseline from the Summary table."""
    path = docs_dir / BASELINES_FILE
    if not path.is_file():
        raise CardSourceError(f"missing source {path}")
    text = path.read_text(encoding="utf-8")
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _BASELINE_ROW.match(line)
        if match and match["policy"] not in rows:
            rows[match["policy"]] = {k: float(match[k]) for k in _FIELDS}
    missing = [p for p in BASELINE_ORDER if p not in rows]
    if missing:
        raise CardSourceError(f"{path}: no Summary row for {missing}")
    hash_match = _CONFIG_HASH.search(text)
    if hash_match is None:
        raise CardSourceError(f"{path}: config hash not found")
    return {"rows": {p: rows[p] for p in BASELINE_ORDER}, "config_hash": hash_match.group(1)}


def dataset_metrics(data_dir: Path) -> Metrics:
    """data/manifest.json -> episodes and skipped seeds per split, config hash."""
    path = data_dir / "manifest.json"
    where = str(path)
    data = _load_json(path)
    splits: dict[str, dict[str, Any]] = {}
    for report in _require(data, "splits", where):
        name = str(_require(report, "split", where))
        skipped = _require(report, "skipped", where)
        splits[name] = {
            "episodes": len(_require(report, "episodes", where)),
            "skipped": [int(_require(s, "seed", where)) for s in skipped],
            "seed_base": int(_require(report, "seed_base", where)),
        }
    if not {"train", "dev"} <= set(splits):
        raise CardSourceError(f"{path}: train/dev splits missing")
    return {"config_hash": str(_require(data, "config_hash", where)), "splits": splits}


def _int_constant(path: Path, name: str) -> int:
    """The integer bound to module-level ``name`` in a test file (read by AST, never imported)."""
    if not path.is_file():
        raise CardSourceError(f"missing source {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError as exc:
                raise CardSourceError(f"{path}: constant {name} is not a literal") from exc
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    raise CardSourceError(f"{path}: constant {name} not found")


def leak_scan_metrics(tests_dir: Path) -> Metrics:
    """Scan volumes from the constants of the leak-scan and fuzz tests."""
    episodes = _int_constant(tests_dir / LEAK_TEST, "EPISODES_REQUIRED")
    steps = _int_constant(tests_dir / LEAK_TEST, "STEPS_PER_EPISODE")
    return {
        "episodes": episodes,
        "surfaces_per_episode": steps + 1,  # the reset observation plus every step
        "surfaces": episodes * (steps + 1),
        "fuzz_iterations": _int_constant(tests_dir / FUZZ_TEST, "FUZZ_ITERATIONS"),
    }


def count_tests(tests_dir: Path) -> dict[str, int]:
    """Test files, test functions (module level or in classes) and slow-marked ones, by AST."""
    files = sorted(tests_dir.rglob("test_*.py"))
    if not files:
        raise CardSourceError(f"no test files under {tests_dir}")
    functions = slow = 0
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                functions += 1
                decorators = node.decorator_list
                slow += any(isinstance(d, ast.Attribute) and d.attr == "slow" for d in decorators)
    return {"files": len(files), "functions": functions, "slow": slow}


def dev_config_metrics(config_dir: Path) -> Metrics:
    """config/dev.yaml -> injected issue counts and decoy counts of the committed dev episodes."""
    path = config_dir / "dev.yaml"
    where = str(path)
    if not path.is_file():
        raise CardSourceError(f"missing source {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = _require(raw, "errors", where)
    counts = {t: int(_require(_require(errors, t, where), "count", where)) for t in ISSUE_TYPES}
    scenarios = {k: int(v) for k, v in sorted(_require(errors, "interacting", where).items())}
    decoys = _require(raw, "decoys", where)
    decoy_counts = {name: int(decoys.get(name, 0)) for name in DecoyCounts.model_fields}
    return {
        "issue_counts": counts,
        "scenarios": scenarios,
        "decoys": decoy_counts,
        "issues_per_episode": sum(counts.values()) + 2 * sum(scenarios.values()),
    }


_DIGEST: Final[re.Pattern[str]] = re.compile(
    r"ARG PYTHON_BASE=(?P<image>[\w.:/-]+)@sha256:(?P<digest>[0-9a-f]{64})"
)


def image_metrics(repo_root: Path) -> dict[str, str]:
    """Dockerfile -> digest-pinned base image."""
    path = repo_root / "Dockerfile"
    if not path.is_file():
        raise CardSourceError(f"missing source {path}")
    match = _DIGEST.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise CardSourceError(f"{path}: digest-pinned base image not found")
    return {"base_image": match["image"], "digest": match["digest"]}


def reward_formula() -> str:
    """The official reward as text, from the reward constants."""
    return (
        f"clip({REWARD_F1_COEF} * F1 - {REWARD_F1_BIAS} - {REWARD_STEP_COST} * inspection_steps,"
        f" {REWARD_CLIP_MIN:g}, {REWARD_CLIP_MAX:g})"
    )


def collect_metrics(
    *,
    docs_dir: Path = DOCS_DIR,
    data_dir: Path = DATA_DIR,
    tests_dir: Path = TESTS_DIR,
    config_dir: Path = CONFIG_DIR,
    repo_root: Path = REPO_ROOT,
) -> Metrics:
    """Everything the card prints, from its sources; raises CardSourceError when one is missing."""
    dataset = dataset_metrics(data_dir)
    baselines = baseline_metrics(docs_dir)
    if baselines["config_hash"] != dataset["config_hash"]:
        raise CardSourceError(
            f"docs/{BASELINES_FILE} was produced for config {baselines['config_hash']}, the "
            f"committed data has {dataset['config_hash']}: rerun `make baselines`"
        )
    return {
        "version": fintrace.__version__,
        "date": CARD_DATE,
        "env": {
            "tools": list(TOOL_NAMES),
            "max_steps": MAX_STEPS,
            "time_budget_s": TIME_BUDGET_SECONDS,
            "max_findings": MAX_FINDINGS,
            "max_page_size": MAX_PAGE_SIZE,
            "reward": reward_formula(),
            "cpus": JURY_CPUS,
            "memory_gb": JURY_MEMORY_GB,
        },
        "issue_types": list(ISSUE_TYPES),
        "dev_config": dev_config_metrics(config_dir),
        "dataset": dataset,
        "attacks": attack_metrics(docs_dir),
        "gap": oracle_gap_metrics(docs_dir),
        "baselines": baselines,
        "leak": leak_scan_metrics(tests_dir),
        "tests": count_tests(tests_dir),
        "image": image_metrics(repo_root),
    }


# ---------------------------------------------------------------- derived strings
def _reward(value: float) -> str:
    return f"{value:+.3f}"


def _counts(values: dict[str, int]) -> str:
    return ", ".join(f"{k} ({v})" for k, v in values.items())


class _Text:
    """The formatted phrases shared by the markdown and the PDF (one source for both)."""

    def __init__(self, m: Metrics) -> None:
        a, g, b, d, leak, t = (
            m["attacks"], m["gap"], m["baselines"], m["dataset"], m["leak"], m["tests"],
        )  # fmt: skip
        train, dev = d["splits"]["train"], d["splits"]["dev"]
        rows = b["rows"]
        self.n_episodes = train["episodes"] + dev["episodes"]
        self.split = f"{train['episodes']} train + {dev['episodes']} dev"
        self.skipped = f"train {train['skipped']}, dev {dev['skipped']}"
        self.seed_bases = f"{train['seed_base']} / {dev['seed_base']}"
        self.config_hash = d["config_hash"]
        self.defeated = f"{a['defeated']}/{a['n_attackers']} attackers defeated"
        self.worst = f"worst mean {_reward(a['worst_mean'])} ({a['worst_name']})"
        floor_value = _reward(REWARD_CLIP_MIN)
        self.floor = f"{a['floor_count']}/{a['n_attackers']} on the {floor_value} floor"
        self.reference = f"reference {a['reference_name']} {_reward(a['reference_mean'])}"
        self.threshold = f"{a['threshold']:+.1f}"
        self.attack_split = f"{a['episodes_per_attacker']} {a['split']} episodes"
        gaps = g["rule_based_gaps"]
        self.gaps = ", ".join(f"{t_}: {gaps[t_]:.2f}" for t_ in g["error_types"])
        self.spread = f"spread {g['spread']:.2f} ({g['hardest']} vs {g['easiest']})"
        self.spread_detail = (
            f"{g['spread']:.2f} ({g['hardest']} {gaps[g['hardest']]:.2f} vs "
            f"{g['easiest']} {gaps[g['easiest']]:.2f})"
        )
        self.attacker_range = (
            f"attackers in [{g['attacker_gap_min']:.2f}, {g['attacker_gap_max']:.2f}]"
        )
        self.gap_design = (
            f"{g['n_policies']} policies x {g['n_types']} types x {g['n_per_type']} paired episodes"
        )
        self.oracle_clean = _reward(g["oracle_clean_mean"])
        self.baselines = ", ".join(f"{p} {_reward(rows[p]['mean'])}" for p in BASELINE_ORDER)
        self.rule_based_mean = _reward(rows["rule_based"]["mean"])
        self.oracle_mean = _reward(rows["oracle"]["mean"])
        self.leak = (
            f"enforced by tests scanning {leak['episodes']} episodes x "
            f"{leak['surfaces_per_episode']} surfaces ({leak['surfaces']}) and "
            f"{leak['fuzz_iterations']} fuzzed actions; any hit fails the suite"
        )
        self.leak_short = (
            f"tests scan {leak['episodes']} x {leak['surfaces_per_episode']} = {leak['surfaces']} "
            f"surfaces and {leak['fuzz_iterations']} fuzzed actions; any hit fails"
        )
        self.tests = f"{t['functions']} test functions in {t['files']} files ({t['slow']} slow)"
        self.determinism = (
            f"byte-identical regeneration of all {self.n_episodes} committed episodes and of "
            "every report"
        )
        issue_counts = m["dev_config"]["issue_counts"]
        self.issue_counts = _counts({k: issue_counts[k] for k in m["issue_types"]})
        self.n_scenarios = len(m["dev_config"]["scenarios"])
        self.n_attackers = a["n_attackers"]
        self.scenarios = _counts(m["dev_config"]["scenarios"])
        self.decoys = _counts(m["dev_config"]["decoys"])
        self.issues_per_episode = m["dev_config"]["issues_per_episode"]
        self.digest = m["image"]["digest"]
        self.base_image = m["image"]["base_image"]


def provenance_rows(m: Metrics) -> list[tuple[str, str, str]]:
    """(number, value, source) for every figure printed in the Metrics section."""
    a, g, b, d = m["attacks"], m["gap"], m["baselines"], m["dataset"]
    leak, env = m["leak"], m["env"]
    x = _Text(m)
    attacks = f"docs/{ATTACKS_FILE}"
    gap = f"docs/{GAP_FILE}"
    rows: list[tuple[str, str, str]] = [
        ("attackers defeated", f"{a['defeated']}/{a['n_attackers']}",
         f"{attacks}: attackers[].defeated"),
        ("worst attacker mean reward", f"{_reward(a['worst_mean'])} ({a['worst_name']})",
         f"{attacks}: max(attackers[].stats.mean_reward)"),
        ("attackers on the -1.000 floor", f"{a['floor_count']}/{a['n_attackers']}",
         f"{attacks}: attackers[].stats.max_reward <= -1"),
        ("reference heuristic mean reward",
         f"{_reward(a['reference_mean'])} ({a['reference_name']})",
         f"{attacks}: reference.stats.mean_reward"),
        ("defeat threshold", x.threshold, f"{attacks}: threshold"),
        ("rule_based oracle-gap spread", x.spread_detail,
         f"{gap}: gap.rule_based[type] over the error types"),
        ("attacker oracle gap range", f"[{g['attacker_gap_min']:.2f}, {g['attacker_gap_max']:.2f}]",
         f"{gap}: gap[attacker][error type]"),
        ("oracle on clean books", x.oracle_clean, f"{gap}: mean_reward.oracle.clean"),
        ("oracle-gap design", x.gap_design, f"{gap}: policies, types, n_per_type"),
    ]  # fmt: skip
    for policy in BASELINE_ORDER:
        mean = _reward(b["rows"][policy]["mean"])
        source = f"docs/{BASELINES_FILE}: Summary table"
        rows.append((f"baseline {policy} mean reward", mean, source))
    rows += [
        ("dataset", f"{x.split} episodes", "data/manifest.json: splits[].episodes"),
        ("skipped seeds", x.skipped, "data/manifest.json: splits[].skipped"),
        ("config hash", d["config_hash"], "data/manifest.json: config_hash"),
        ("leak-scan surfaces",
         f"{leak['episodes']} episodes x {leak['surfaces_per_episode']} = {leak['surfaces']}",
         f"tests/{LEAK_TEST}: EPISODES_REQUIRED, STEPS_PER_EPISODE (+ reset)"),
        ("fuzzed actions", f"{leak['fuzz_iterations']}", f"tests/{FUZZ_TEST}: FUZZ_ITERATIONS"),
        ("test inventory", x.tests, "tests/**/test_*.py (AST)"),
        ("image base digest", f"sha256:{x.digest[:16]}...", "Dockerfile: ARG PYTHON_BASE"),
        ("jury compute budget", f"{env['cpus']} CPU / {env['memory_gb']} GB RAM, no network",
         "CLAUDE.md section 1 (declared constants JURY_CPUS, JURY_MEMORY_GB)"),
        ("step / time budgets", f"{env['max_steps']} actions, {env['time_budget_s']:g} s",
         "fintrace.constants: MAX_STEPS, TIME_BUDGET_SECONDS"),
    ]  # fmt: skip
    return rows


def render_markdown(m: Metrics) -> str:
    """The card as markdown; a pure function of the collected metrics (byte-stable)."""
    env, b = m["env"], m["baselines"]
    x = _Text(m)
    tools = "`, `".join(env["tools"])
    baseline_rows = "\n".join(
        f"| {p} | {_reward(r['mean'])} | {r['std']:.3f} | {_reward(r['min'])} | "
        f"{_reward(r['max'])} | {r['steps']:.1f} |"
        for p, r in b["rows"].items()
    )
    provenance = "\n".join(f"| {n} | {v} | `{s}` |" for n, v, s in provenance_rows(m))
    n_issue_types = len(m["issue_types"])
    repro_tests = (
        "`tests/integration/test_data_integration.py::test_generate_data_timing`, "
        "`tests/contract/test_generator.py::test_deterministic_regeneration`"
    )
    leak_tests = (
        "`tests/security/test_leak_scan_full.py::test_leak_scan_20_seeds_100_steps_each`, "
        "`tests/contract/test_env_fuzz.py::test_fuzz_10000_random_actions_no_crash_no_leak`"
    )
    lines = [
        "# FINTRACE Environment Card",
        "",
        f"Version `{m['version']}`, card date {m['date']} (fixed build date; every number below is "
        "read from a committed artefact named in the last section, none is typed by hand).",
        "",
        "## Environment details",
        "",
        "- Task: audit a seeded synthetic ledger against its source documents (invoices, payments, "
        "FX rates) and submit the injected errors as findings; a hidden evaluator scores them "
        "against the exact ground truth.",
        "- Observation space (jury protocol): reset returns `{task, manifest, policy, "
        "remaining_steps}`; every step returns `{result, error, remaining_steps}` with `error` "
        "null or `{code, message}` (jury adapter envelope; the internal environment returns "
        "the error code only).",
        f"- Action space: {len(env['tools'])} tools, `{tools}`, sent as `{{\"name\", \"args\"}}`; "
        f"`list_records` pages hold at most {env['max_page_size']} records; a submission holds at "
        f"most {env['max_findings']} findings over {n_issue_types} issue types.",
        f"- Reward: `{env['reward']}`; every step before the terminal submit is an inspection "
        "step; in the jury's closed mode the policy sees reward 0 and the return is computed by "
        "`score_episode` only.",
        f"- Budgets: {env['max_steps']} actions per episode, {env['time_budget_s']:g} s of wall "
        f"clock per episode, {env['cpus']} CPU / {env['memory_gb']} GB RAM, no network (jury "
        "machine).",
        "",
        "## Data",
        "",
        "- Generator: seeded synthetic double-entry books (`random.Random(seed)` only), validated "
        "for balance, totals, allocations, period bounds and Benford statistics before any error "
        "is injected.",
        f"- Injected error types (count per committed dev episode): {x.issue_counts}; interacting "
        f"scenarios: {x.scenarios}; issues per dev episode: {x.issues_per_episode}.",
        f"- Decoys (legitimate look-alikes, per dev episode): {x.decoys}.",
        f"- Split: {x.split} committed episodes (seed bases {x.seed_bases}; infeasible seeds "
        f"skipped: {x.skipped}); config hash `{x.config_hash}`; every public file checksummed.",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Evidence |",
        "|---|---|---|",
        f"| Determinism | {x.determinism} | {repro_tests} |",
        f"| Data leakage | {x.leak} | {leak_tests} |",
        f"| Adversarial robustness | {x.defeated} on {x.attack_split} (threshold {x.threshold}); "
        f"{x.worst}; {x.floor}; {x.reference} | `docs/{ATTACKS_FILE}`, "
        "`docs/reward_robustness_report.md` |",
        f"| Differential signal | rule_based oracle gap {x.gaps}; {x.spread}; {x.attacker_range} "
        f"on every error type; {x.gap_design} | `docs/{GAP_FILE}`, `docs/oracle_gap_report.md` |",
        f"| Baselines ({m['attacks']['split']} split) | {x.baselines} (mean reward) | "
        f"`docs/{BASELINES_FILE}` |",
        f"| Test coverage | {x.tests}; parametrisation expands the collected count (`make test` / "
        "`make test-slow`) | `tests/` |",
        "",
        "Baseline detail (mean reward, std, min, max, mean steps):",
        "",
        "| Policy | Mean | Std | Min | Max | Steps |",
        "|---|---:|---:|---:|---:|---:|",
        baseline_rows,
        "",
        "## Intended use",
        "",
        "- Training and evaluating audit agents (RL fine-tuning, behaviour cloning, prompted LLM "
        "agents) on reconciliation errors with an exact, leak-free, reproducible reward.",
        f"- Benchmarking reward robustness: the {x.n_attackers} attack policies and the oracle-gap "
        "diagnostic are part of the environment and rerun with one command each.",
        "",
        "## Out of scope",
        "",
        "- Auditing real books: the ledgers are synthetic and the error taxonomy is fixed; the "
        "environment certifies agents against this taxonomy, not against a company's accounts.",
        "- Production deployment: the shipped image is a jury/evaluation artefact (offline, "
        "CPU-only).",
        "",
        "## Limitations",
        "",
        f"- Synthetic data: {n_issue_types} error classes plus {x.n_scenarios} interacting "
        "scenarios; realistic rates and structure, but no document images, free text or human "
        "noise.",
        "- The NetworkX graph layer (`src/fintrace/graph/`) is declared in the architecture and "
        "not yet used; relationships in the OCEL export come from the public records directly.",
        "- No learned policy is shipped (the CPU behaviour-cloning proof, D4, is planned); the "
        "baselines are a rule-based auditor, a random investigator, a mark-everything attacker "
        "and a truth-reading oracle.",
        "- The rule-based baseline spends its whole step budget, so late allocation issues can be "
        f"missed by design (its mean on the {m['attacks']['split']} split is {x.rule_based_mean} "
        f"against the oracle's {x.oracle_mean}).",
        "",
        "## Reproducibility",
        "",
        "- Same seed and config give byte-identical episode files and identical rewards; all "
        "randomness flows from one seeded generator; wall-clock and global-RNG APIs are banned by "
        "lint.",
        "- Offline by test: the complete jury protocol runs under a socket guard on every CI run; "
        f"the image is built from `{x.base_image}` pinned by digest `sha256:{x.digest}`, runs as a "
        "non-root user and installs the pinned runtime requirements only.",
        "- Regenerate everything: `make data`, `make baselines`, `make attacks`, "
        "`make oracle-gap`, `make ocel`, `make card`; verify: `make data-verify`, `make test-all`.",
        "",
        "## Provenance of the numbers",
        "",
        "| Number | Value | Source |",
        "|---|---|---|",
        provenance,
        "",
    ]
    return "\n".join(lines)


def render_pdf(m: Metrics, out: Path) -> int:
    """One-page PDF of the card with a fixed creation date; returns the page count."""
    from fpdf import FPDF  # dev-only dependency, imported here so the markdown never needs it

    env = m["env"]
    x = _Text(m)
    tools = ", ".join(env["tools"])
    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime.fromisoformat(f"{m['date']}T00:00:00+00:00").astimezone(UTC))
    pdf.set_title("FINTRACE Environment Card")
    pdf.set_author("FINTRACE")
    pdf.set_producer("FINTRACE scripts/build_environment_card.py")
    pdf.set_margins(12, 10, 12)
    pdf.set_auto_page_break(auto=True, margin=10)  # overflow -> page 2 -> the build fails
    pdf.add_page()

    def heading(text: str) -> None:
        pdf.set_font(PDF_FONT, "B", 10)
        pdf.cell(0, 5.2, text, new_x="LMARGIN", new_y="NEXT")

    def para(text: str) -> None:
        pdf.set_font(PDF_FONT, "", 7.6)
        pdf.multi_cell(0, 3.5, text, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(PDF_FONT, "B", 15)
    pdf.cell(0, 8, "FINTRACE Environment Card", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(PDF_FONT, "", 8)
    subtitle = (
        f"Version {m['version']}  |  card date {m['date']}  |  reproducible, offline RL "
        "environment for financial-audit agents (Genesis Hackathon 2026)"
    )
    pdf.cell(0, 4.5, subtitle, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    heading("Environment details")
    para(
        "Task: audit a seeded synthetic ledger against its invoices, payments and FX rates and "
        "submit the injected errors as findings; a hidden evaluator scores them against the exact "
        "ground truth. Observation: reset -> {task, manifest, policy, remaining_steps}; step -> "
        f"{{result, error, remaining_steps}}. Actions: {tools} as {{name, args}}; pages <= "
        f"{env['max_page_size']} records; <= {env['max_findings']} findings per submission "
        f"over {len(m['issue_types'])} issue types. Reward: {env['reward']}; closed mode returns 0 "
        f"to the policy, score_episode holds the return. Budgets: {env['max_steps']} actions, "
        f"{env['time_budget_s']:g} s wall clock, {env['cpus']} CPU / {env['memory_gb']} GB, no "
        "network."
    )
    heading("Data")
    para(
        "Seeded synthetic double-entry books validated (balance, totals, allocations, period, "
        f"Benford) before injection. Error types per dev episode: {x.issue_counts}; interacting "
        f"scenarios {x.scenarios}; {x.issues_per_episode} issues per episode. Decoys: {x.decoys}. "
        f"Split: {x.split} episodes, config hash {x.config_hash}, infeasible seeds skipped "
        f"({x.skipped})."
    )
    heading("Metrics (every value read from a committed artefact)")
    pdf.set_font(PDF_FONT, "", 7.2)
    metric_rows = [
        ("Determinism", x.determinism, "repro tests"),
        ("Data leakage", x.leak_short, "security + fuzz tests"),
        ("Adversarial robustness", f"{x.defeated}; {x.worst}; {x.floor}; {x.reference}",
         f"docs/{ATTACKS_FILE}"),
        ("Differential signal",
         f"rule_based gap spread {x.spread_detail}; {x.attacker_range}", f"docs/{GAP_FILE}"),
        ("Baselines (dev)", x.baselines, f"docs/{BASELINES_FILE}"),
        ("Test coverage", x.tests, "tests/ (AST)"),
    ]  # fmt: skip
    with pdf.table(
        col_widths=(28, 118, 40), line_height=3.6, text_align="LEFT", first_row_as_headings=False
    ) as table:
        for row_values in metric_rows:
            row = table.row()
            for value in row_values:
                row.cell(value)
    pdf.ln(1.5)
    heading("Intended use / out of scope")
    para(
        "Intended: training and evaluating audit agents on reconciliation errors with an exact, "
        "leak-free, reproducible reward; benchmarking reward robustness with the shipped attackers "
        "and the oracle-gap diagnostic. Out of scope: auditing real books (synthetic data, fixed "
        "taxonomy) and production deployment (the image is an offline evaluation artefact)."
    )
    heading("Limitations")
    para(
        f"Synthetic data with {len(m['issue_types'])} error classes plus {x.n_scenarios} "
        "interacting scenarios; the NetworkX graph layer is declared but unused; no learned "
        "policy is shipped "
        "(CPU behaviour-cloning proof planned); the rule-based baseline spends its whole step "
        f"budget ({x.rule_based_mean} vs oracle {x.oracle_mean} on dev)."
    )
    heading("Reproducibility")
    para(
        "One seeded generator, canonical serialisation, checksummed dataset, banned wall-clock and "
        "global-RNG APIs. Offline by test (socket guard around the whole jury protocol). Image: "
        f"{x.base_image} pinned by digest sha256:{x.digest[:24]}..., non-root, pinned runtime "
        "requirements only. Regenerate: make data / baselines / attacks / oracle-gap / ocel / "
        "card; verify: make data-verify, make test-all. Full provenance of every number: "
        "docs/environment_card.md."
    )
    pages = int(pdf.page)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    return pages


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 on success, 2 when a source is missing, 1 when the PDF cannot be built."""
    parser = argparse.ArgumentParser(description="Build the FINTRACE Environment Card.")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tests-dir", type=Path, default=TESTS_DIR)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    try:
        metrics = collect_metrics(
            docs_dir=args.docs_dir,
            data_dir=args.data_dir,
            tests_dir=args.tests_dir,
            config_dir=args.config_dir,
        )
    except CardSourceError as exc:
        print(f"environment card: {exc} (nothing written)", file=sys.stderr)
        return EXIT_SOURCE
    markdown = render_markdown(metrics)
    payload = markdown.encode("utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_bytes(payload)
    print(f"{'number':34s} {'value':46s} source")
    for number, value, source in provenance_rows(metrics):
        print(f"{number:34s} {value:46s} {source}")
    digest = hashlib.sha256(payload).hexdigest()
    print(f"wrote {args.out_md} ({len(payload)} bytes, sha256 {digest})")
    if args.no_pdf:
        return 0
    try:
        pages = render_pdf(metrics, args.out_pdf)
    except ImportError:
        print(
            "environment card: fpdf2 is not installed (dev extra); markdown written, PDF skipped",
            file=sys.stderr,
        )
        return EXIT_PDF
    if pages != 1:
        print(f"environment card: PDF has {pages} pages, expected 1", file=sys.stderr)
        return EXIT_PDF
    print(f"wrote {args.out_pdf} ({args.out_pdf.stat().st_size} bytes, {pages} page)")
    return 0


__all__ = [
    "ATTACKS_FILE",
    "BASELINES_FILE",
    "BASELINE_ORDER",
    "CARD_DATE",
    "DOCS_DIR",
    "EXIT_PDF",
    "EXIT_SOURCE",
    "GAP_FILE",
    "OUT_MD",
    "OUT_PDF",
    "SECTIONS",
    "CardSourceError",
    "attack_metrics",
    "baseline_metrics",
    "collect_metrics",
    "count_tests",
    "dataset_metrics",
    "dev_config_metrics",
    "image_metrics",
    "leak_scan_metrics",
    "main",
    "oracle_gap_metrics",
    "provenance_rows",
    "render_markdown",
    "render_pdf",
    "reward_formula",
]

if __name__ == "__main__":
    raise SystemExit(main())
