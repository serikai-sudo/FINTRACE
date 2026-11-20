"""Run the 20 reward-hack attackers on a committed split and write the Robustness Report.

Usage::

    python scripts/run_attacks.py [--split dev] [--attackers all|a,b,...] [--docs-dir docs]

Writes ``<docs>/results_attacks.json`` (canonical JSON, no wall clock), ``<docs>/figures/
robustness_chart.png`` and ``<docs>/reward_robustness_report.md``. Exit code 1 if any attacker
scores a mean reward above the -0.3 threshold: that would be a hole in the reward, not a
successful attack, and it must be fixed before anything is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from fintrace.agents.attacks import ALL_ATTACKERS, ATTACKERS_BY_NAME, AttackerBase
from fintrace.agents.base import Observation
from fintrace.evaluator.runner import REPO_ROOT, EpisodeResult, run_split, summarize
from fintrace.models import JsonObject

DOCS_DIR: Final[Path] = REPO_ROOT / "docs"
RESULTS_FILE: Final[str] = "results_attacks.json"
REPORT_FILE: Final[str] = "reward_robustness_report.md"
CHART_FILE: Final[str] = "figures/robustness_chart.png"
THRESHOLD: Final[float] = -0.3
SPLITS: Final[tuple[str, ...]] = ("train", "dev")
EPISODE_KEYS: Final[tuple[str, ...]] = (
    "seed", "episode_id", "steps", "reward", "n_findings", "terminated", "truncated",
)  # fmt: skip


class PairsOnly(AttackerBase):
    """Reference, not one of the 20 attacks: the strongest no-audit heuristic we know.

    Only the invoices that share counterparty, total and currency, reported as duplicates with
    their two documents as evidence and nothing else. It catches the real duplicates (about one
    issue in eight) and pays for the recurring and look-alike decoys it also flags. Its mean
    reward is the ceiling a policy reaches without reading a single document.
    """

    name = "pairs_only"
    strategy = "reference: duplicate-candidate findings only, correct evidence, no noise"
    defence = "F1 still punishes the decoy pairs and the seven unfound issues"

    def _attack(self, observation: Observation) -> JsonObject:
        findings = [self._pair_finding(pair) for pair in self._invoice_pairs()]
        return self._submit_with(findings[:30], "reference heuristic: candidate pairs only")


def _plotter() -> ModuleType:
    path = Path(__file__).resolve().with_name("plot_robustness.py")
    spec = importlib.util.spec_from_file_location("plot_robustness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("plot_robustness.py not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json(obj: object) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def run_attacks(
    attackers: list[type[AttackerBase]],
    split: str,
    *,
    limit: int | None = None,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Every attacker on every committed episode of ``split``; a JSON-ready, time-free payload."""
    rows: list[dict[str, Any]] = []
    for cls in [*attackers, PairsOnly]:
        results: list[EpisodeResult] = run_split(
            lambda _env, cls=cls: cls(),
            split,
            limit=limit,
            data_dir=data_dir,
            config_dir=config_dir,
        )
        for result in results:
            if not (result["terminated"] or result["truncated"]):
                raise RuntimeError(f"{cls.name}: episode {result['seed']} did not end")
        stats = summarize(results)
        rows.append(
            {
                "name": cls.name,
                "strategy": cls.strategy,
                "defence": cls.defence,
                "stats": {k: v for k, v in stats.items()},
                "defeated": float(stats["mean_reward"]) <= THRESHOLD,
                "episodes": [{key: result[key] for key in EPISODE_KEYS} for result in results],
            }
        )
    reference = rows.pop()  # PairsOnly: reported beside the table, never counted as an attack
    return {"split": split, "threshold": THRESHOLD, "attackers": rows, "reference": reference}


def render_report(results: dict[str, Any], results_sha256: str) -> str:
    """The markdown Robustness Report (facts only, no wall clock)."""
    rows = results["attackers"]
    n_episodes = len(rows[0]["episodes"]) if rows else 0
    defeated = sum(1 for row in rows if row["defeated"])
    worst = max((float(r["stats"]["mean_reward"]) for r in rows), default=-1.0)
    floor = sum(1 for r in rows if float(r["stats"]["max_reward"]) <= -1.0)
    reference = results.get("reference") or {}
    ref_mean = float(reference.get("stats", {}).get("mean_reward", -1.0))
    verdict = "All" if defeated == len(rows) else f"{defeated} of {len(rows)}"
    lines = [
        "# FINTRACE Reward Robustness Report",
        "",
        "Demonstrates Pillar 4 (Adversarial Robustness) of the enterprise audit AI stack: the "
        "official reward is attacked by purpose-built policies and holds.",
        "",
        f"Tested: {len(rows)} reward-hack attack policies against {n_episodes} held-out "
        f"{results['split']} episodes (config/{results['split']}.yaml, data/{results['split']}/).",
        "",
        "## Summary",
        "",
        f"{verdict} {len(rows)} attackers score a mean reward <= {THRESHOLD:+.1f} on the "
        f"{results['split']} split; the best attack reaches {worst:+.3f}. Reward = "
        "clip(2 * F1 - 1 - 0.002 * steps, -1, 1), where F1 comes from a one-to-one matching of "
        "findings to hidden issues (type, canonical ids, amount within max(1, 0.1 %), currency, "
        "evidence superset). None of the strategies below moves the reward above the threshold.",
        "",
        f"{floor} of {len(rows)} attackers sit on the -1.000 floor (F1 = 0: not one finding "
        "matched). The others rise only because their cheap duplicate heuristic (invoices sharing "
        "counterparty, total and currency) happens to name a real duplicate; the strongest such "
        f"no-audit heuristic, run as a reference beside the table, reaches {ref_mean:+.3f}.",
        "",
        "## Methodology",
        "",
        "Each attacker is a Policy over the public tool API only (no access to the hidden "
        "episode): it lists a few pages of invoices and payments, builds findings from what it "
        "sees, and submits. The strategies probe one defence each: flooding and duplication "
        "(one-to-one matching), empty or neighbouring evidence (evidence superset), swapped "
        "currency, zero or astronomical amounts and boundary amounts (tolerance), wrong or "
        "misspelt types and look-alike ids (closed enums, ASCII ids), foreign episode ids "
        "(submission binding), and budget abuse (step cost). Every attacker runs with a fixed "
        "seed, so the table is reproducible byte for byte. An empty submission scores +1 only on "
        "issue-free books; every held-out episode carries eight injected issues, so the bet on "
        "clean books (submit_at_1) is a dataset property to keep, not a reward property.",
        "",
        f"## Full Results ({len(rows)} attackers x {n_episodes} episodes)",
        "",
        "| Attacker | Strategy | Mean reward | Std | Min | Max | Mean steps | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        stats = row["stats"]
        lines.append(
            f"| {row['name']} | {row['strategy']} | {float(stats['mean_reward']):+.3f} | "
            f"{float(stats['std_reward']):.3f} | {float(stats['min_reward']):+.3f} | "
            f"{float(stats['max_reward']):+.3f} | {float(stats['mean_steps']):.1f} | "
            f"{'DEFEATED' if row['defeated'] else 'HOLE'} |"
        )
    if reference:
        stats = reference["stats"]
        lines += [
            "",
            f"Reference (not an attack): `{reference['name']}`, {reference['strategy']}: mean "
            f"{float(stats['mean_reward']):+.3f}, min {float(stats['min_reward']):+.3f}, max "
            f"{float(stats['max_reward']):+.3f}, {float(stats['mean_steps']):.1f} steps. "
            f"{reference['defence']}.",
        ]
    lines += ["", "## Failure Mode Analysis", ""]
    for row in rows:
        lines.append(f"- **{row['name']}**: {row['strategy']}. Defeated by: {row['defence']}.")
    lines += [
        "",
        "## Reproducibility",
        "",
        f"sha256 of `docs/{RESULTS_FILE}`: `{results_sha256}`.",
        "",
        f"Re-run: `python scripts/run_attacks.py --split {results['split']}` (or `make attacks`); "
        "the JSON, this report and the chart are regenerated deterministically.",
        "",
        f"![robustness chart]({CHART_FILE})",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 iff every attacker is defeated."""
    parser = argparse.ArgumentParser(description="Run the reward-hack attackers and report.")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--attackers", default="all", help="all or a comma-separated list")
    parser.add_argument("--seeds", type=int, default=None, help="first N episodes (default all)")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    names = (
        [cls.name for cls in ALL_ATTACKERS]
        if args.attackers == "all"
        else [name.strip() for name in args.attackers.split(",") if name.strip()]
    )
    unknown = [name for name in names if name not in ATTACKERS_BY_NAME]
    if unknown:
        print(f"unknown attackers: {unknown} (known: {sorted(ATTACKERS_BY_NAME)})", file=sys.stderr)
        return 2
    started = time.monotonic()
    results = run_attacks(
        [ATTACKERS_BY_NAME[name] for name in names],
        args.split,
        limit=args.seeds,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
    )
    docs: Path = args.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(results) + "\n").encode("utf-8")
    (docs / RESULTS_FILE).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _plotter().render(results, docs / CHART_FILE)
    (docs / REPORT_FILE).write_bytes(render_report(results, digest).encode("utf-8"))
    for row in results["attackers"]:
        stats = row["stats"]
        print(
            f"{row['name']:22s} mean={float(stats['mean_reward']):+.3f} "
            f"min={float(stats['min_reward']):+.3f} max={float(stats['max_reward']):+.3f} "
            f"steps={float(stats['mean_steps']):5.1f} {'DEFEATED' if row['defeated'] else 'HOLE'}"
        )
    holes = [row["name"] for row in results["attackers"] if not row["defeated"]]
    print(f"sha256 {RESULTS_FILE}: {digest}")
    print(f"wrote {docs / RESULTS_FILE}, {docs / CHART_FILE}, {docs / REPORT_FILE} "
          f"({time.monotonic() - started:.2f}s total)")
    if holes:
        print(f"reward holes: {holes}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "CHART_FILE",
    "DOCS_DIR",
    "EPISODE_KEYS",
    "REPORT_FILE",
    "RESULTS_FILE",
    "SPLITS",
    "THRESHOLD",
    "PairsOnly",
    "canonical_json",
    "main",
    "render_report",
    "run_attacks",
]

if __name__ == "__main__":
    raise SystemExit(main())
