"""Build the diagnostic set, run the 24 policies and write the Oracle Gap report, JSON and heatmap.

Usage::

    python scripts/run_oracle_gap.py [--n-per-type 3] [--seed-base 30000]
                                     [--policies all|a,b,...] [--docs-dir docs]
                                     [--config-dir config]

Writes ``<docs>/oracle_gap_results.json`` (canonical JSON, no wall clock), ``<docs>/figures/
oracle_gap_matrix.png`` and ``<docs>/oracle_gap_report.md``. Exit code 1 if the matrix fails
its sanity checks (oracle below +0.9 in a column, or an attacker earning a positive reward on
erroneous books): that would be a hole to fix, not a result to commit. Exit code 2 for an
unknown policy name or a policy list without the oracle (the gap reference). The artefacts are
written either way (a failing run reports its sanity failures in the report).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from fintrace.constants import ISSUE_TYPES
from fintrace.diagnostics.oracle_gap import (
    ATTACKER_FLOOR_GAP,
    BASE_SPLIT,
    BASELINE_ORDER,
    CLEAN_TYPE,
    DEFAULT_N_PER_TYPE,
    DIAGNOSTIC_SEED_BASE,
    ORACLE,
    attacker_floor,
    attacker_rows,
    build_diagnostic_set,
    check_results,
    compute_oracle_gap,
    differential_spread,
    policy_factories,
)
from fintrace.evaluator.runner import REPO_ROOT, load_split_config
from fintrace.exporters.episode_files import canonical_json

DOCS_DIR: Final[Path] = REPO_ROOT / "docs"
RESULTS_FILE: Final[str] = "oracle_gap_results.json"
REPORT_FILE: Final[str] = "oracle_gap_report.md"
CHART_FILE: Final[str] = "figures/oracle_gap_matrix.png"
FLOOR_GAP: Final[float] = 1.9  # a cell this far from the oracle is the -1.000 floor (F1 = 0)
SILENT_GAP: Final[float] = 0.25  # within this of the oracle on clean books = stayed silent
FLAT_SPREAD: Final[float] = 0.05  # a gap row varying less than this over the types is flat


def _plotter() -> ModuleType:
    path = Path(__file__).resolve().with_name("plot_oracle_gap.py")
    spec = importlib.util.spec_from_file_location("plot_oracle_gap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("plot_oracle_gap.py not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(results: dict[str, Any], key: str, fmt: str) -> list[str]:
    types = [str(t) for t in results["types"]]
    lines = [
        "| Policy | " + " | ".join(types) + " |",
        "|---|" + "---:|" * len(types),
    ]
    for policy in results["policies"]:
        cells = " | ".join(format(float(results[key][policy][t]), fmt) for t in types)
        lines.append(f"| {policy} | {cells} |")
    return lines


def _names(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "none"


def _insight_rule_based(results: dict[str, Any]) -> list[str]:
    if "rule_based" not in results["policies"]:
        return []
    gaps = {t: float(results["gap"]["rule_based"][t]) for t in ISSUE_TYPES}
    hardest = max(gaps, key=lambda t: (gaps[t], t))
    easiest = min(gaps, key=lambda t: (gaps[t], t))
    per_type = ", ".join(f"{gaps[t]:.2f} on {t}" for t in ISSUE_TYPES)
    spread = differential_spread(results, "rule_based")
    if spread <= FLAT_SPREAD:
        return [
            f"- **rule_based** (the auditing baseline): gap {per_type}, spread {spread:.2f}: no "
            "differential across the four error types in this run.",
        ]
    if gaps[easiest] > 0:
        ratio = gaps[hardest] / gaps[easiest]
        comparison = f"the same strategy is {ratio:.1f}x further from the oracle on {hardest} "
        comparison += f"than on {easiest}"
    else:
        comparison = f"the same strategy matches the oracle on {easiest} and trails it by "
        comparison += f"{gaps[hardest]:.2f} on {hardest}"
    return [
        f"- **rule_based** (the auditing baseline): gap {per_type}. Hardest type {hardest} "
        f"({gaps[hardest]:.2f}), easiest {easiest} ({gaps[easiest]:.2f}), spread {spread:.2f}: "
        f"{comparison}. That is a gradient of difficulty for a learner to climb, not a "
        "pass/fail outcome.",
    ]


def _insight_differential(results: dict[str, Any]) -> list[str]:
    policies = [str(p) for p in results["policies"] if p != ORACLE]
    varying = [p for p in policies if differential_spread(results, p) > FLAT_SPREAD]
    detail = ", ".join(f"`{p}` {differential_spread(results, p):.2f}" for p in varying)
    return [
        f"- {len(varying)} of {len(policies)} non-oracle policies have a gap that varies across "
        f"the four error types (spread > {FLAT_SPREAD}): {detail or 'none'}. A pass/fail quiz "
        "would give every policy one constant row.",
    ]


def _insight_clean(results: dict[str, Any]) -> list[str]:
    gap = results["gap"]
    oracle_clean = float(results["mean_reward"][ORACLE][CLEAN_TYPE])
    others = [str(p) for p in results["policies"] if p != ORACLE]
    silent = [p for p in others if float(gap[p][CLEAN_TYPE]) <= SILENT_GAP]
    flagging = [p for p in others if float(gap[p][CLEAN_TYPE]) >= FLOOR_GAP]
    return [
        "- **clean books** measure false-positive propensity. The oracle scores "
        f"{oracle_clean:+.3f} "
        f"by submitting nothing; {len(silent)} policies stay within {SILENT_GAP} of it ("
        f"{_names(silent)}): they submit nothing or next to nothing there, and on clean books "
        f"silence is the correct answer. The {len(flagging)} policies that flag non-existent "
        f"errors sit at the floor ({_names(flagging)}).",
    ]


def _insight_attackers(results: dict[str, Any]) -> list[str]:
    attackers = attacker_rows(results)
    if not attackers:
        return []
    floor = attacker_floor(results)
    min_gap = float(floor["min_gap"])
    max_gap = float(floor["max_gap"])
    cells = [(name, t, float(results["gap"][name][t])) for name in attackers for t in ISSUE_TYPES]
    at_floor = sum(1 for _, _, g in cells if g >= FLOOR_GAP)
    if min_gap >= ATTACKER_FLOOR_GAP:
        verdict = (
            f"i.e. >= {ATTACKER_FLOOR_GAP:.1f}: no attacker ever earns a positive reward on "
            "erroneous books, whatever the type"
        )
    else:
        verdict = (
            f"i.e. below {ATTACKER_FLOOR_GAP:.1f}: an attacker earns a positive reward on "
            "erroneous books, which is a reward hole to fix (see the sanity failures)"
        )
    lines = [
        f"- **{len(attackers)} reward-hack attackers**: on the four error types every gap lies in "
        f"[{min_gap:.2f}, {max_gap:.2f}], {verdict}; {at_floor} of {len(cells)} attacker cells "
        f"sit at the -1.000 floor (gap >= {FLOOR_GAP}).",
    ]
    name, error_type, gap = min(cells, key=lambda cell: (cell[2], cell[0], cell[1]))
    if gap < FLOOR_GAP:
        rewards = [float(r) for r in results["rewards"][name][error_type]]
        hits = sum(1 for r in rewards if r > 0)
        shown = ", ".join(f"{r:+.3f}" for r in rewards)
        others = [float(results["gap"][name][t]) for t in ISSUE_TYPES if t != error_type]
        lines.append(
            f"- The smallest attacker gap is `{name}` on {error_type} ({gap:.2f}): its guess "
            f"coincides with the one injected issue in {hits} of {len(rewards)} books (per-episode "
            f"rewards {shown}). A correct finding earns reward, as it should; on the other three "
            f"types the same policy's gap is {min(others):.2f} to {max(others):.2f}."
        )
    return lines


def _sanity_section(failures: list[str]) -> list[str]:
    if not failures:
        return []
    return ["", "## Sanity failures", "", *(f"- {failure}" for failure in failures)]


def render_report(
    results: dict[str, Any], results_sha256: str, failures: list[str] | None = None
) -> str:
    """The markdown Oracle Gap report (facts only, no wall clock, narrative from the data).

    ``failures`` (from ``check_results``) are listed in their own section so a failing run can
    never leave a clean-looking report behind.
    """
    failures = list(failures) if failures is not None else check_results(results)
    types = [str(t) for t in results["types"]]
    policies = [str(p) for p in results["policies"]]
    n_baselines = sum(1 for p in policies if p in BASELINE_ORDER)
    n_attackers = len(attacker_rows(results))
    n_per_type = int(results["n_per_type"])
    seeds = ", ".join(str(seed) for seed in results["seeds"])
    skipped = results.get("skipped", [])
    lines = [
        "# FINTRACE Oracle Gap Diagnostic",
        "",
        "Demonstrates that FINTRACE produces a differential training signal: the same policy earns "
        "different rewards on different error types, so the environment is trainable, not a "
        "binary pass/fail quiz.",
        "",
        f"Tested: {len(policies)} policies ({n_baselines} baselines + {n_attackers} reward-hack "
        f"attackers) x {len(types)} diagnostic episode types x {n_per_type} paired episodes per "
        f"type (seeds {seeds}).",
        "",
        "## Methodology",
        "",
        "The committed dev episodes mix all four error types, two interacting scenarios and "
        "decoys, "
        "so they cannot attribute a reward to one error type. The diagnostic set isolates one: for "
        f"each seed the clean books (sizes, currencies and decoys of config/{BASE_SPLIT}.yaml, no "
        "injected error) and four variants with exactly one injected issue of one type. The "
        "generator's RNG key excludes the error plan, so the five columns of a seed share the same "
        "base books and differ only in the injected error (a paired design); a seed is kept only "
        f"if all five columns build ({len(skipped)} skipped). Every policy runs through the "
        "standard train-mode runner (`fintrace.evaluator.runner.run_baseline`, official reward "
        "clip(2 * F1 - 1 - 0.002 * steps, -1, 1), 100-step budget) with a fixed seed, so the "
        f"matrix is reproducible byte for byte. Cell = mean reward over the {n_per_type} episodes "
        "of a type; oracle gap = mean(oracle) - mean(policy), where the oracle reads the hidden "
        "truth, fetches its evidence documents and submits exactly the injected issues (and "
        "nothing on clean books). The diagnostic episodes are regenerated in memory and never "
        "committed.",
        "",
        f"## Matrix: mean reward ({len(policies)} policies x {len(types)} types)",
        "",
        *_table(results, "mean_reward", "+.3f"),
        "",
        f"## Oracle gap ({len(policies)} policies x {len(types)} types)",
        "",
        "Cell = mean(oracle) - mean(policy) on that type: 0.00 matches the oracle, 2.00 is the "
        "-1.000 floor against a +1 oracle. The oracle row is the reference.",
        "",
        *_table(results, "gap", ".2f"),
        "",
        "## Key insight",
        "",
        *_insight_rule_based(results),
        *_insight_differential(results),
        *_insight_clean(results),
        *_insight_attackers(results),
        *_sanity_section(failures),
        "",
        "## Reproducibility",
        "",
        f"sha256 of `docs/{RESULTS_FILE}`: `{results_sha256}`.",
        "",
        "Re-run: `python scripts/run_oracle_gap.py` (or `make oracle-gap`); the JSON, this report "
        f"and the heatmap are regenerated deterministically from seeds {seeds}.",
        "",
        f"![oracle gap matrix]({CHART_FILE})",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 iff the matrix passes its sanity checks."""
    parser = argparse.ArgumentParser(description="Compute the oracle-gap matrix and report.")
    parser.add_argument("--n-per-type", type=int, default=DEFAULT_N_PER_TYPE)
    parser.add_argument("--seed-base", type=int, default=DIAGNOSTIC_SEED_BASE)
    parser.add_argument("--policies", default="all", help="all or a comma-separated list")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    names = (
        None
        if args.policies == "all"
        else [name.strip() for name in args.policies.split(",") if name.strip()]
    )
    if names is not None and ORACLE not in names:
        print(f"--policies must include {ORACLE!r} (the gap reference)", file=sys.stderr)
        return 2
    try:
        factories = policy_factories(names)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    started = time.monotonic()
    base = load_split_config(BASE_SPLIT, args.config_dir) if args.config_dir else None
    episodes, skipped = build_diagnostic_set(
        args.n_per_type, base_config=base, seed_base=args.seed_base
    )
    results = compute_oracle_gap(episodes, factories, skipped=skipped)
    failures = check_results(results)
    docs: Path = args.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(results) + "\n").encode("utf-8")
    (docs / RESULTS_FILE).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _plotter().render(results, docs / CHART_FILE)
    report = render_report(results, digest, failures)
    (docs / REPORT_FILE).write_bytes(report.encode("utf-8"))
    types = [str(t) for t in results["types"]]
    print(f"{'policy':22s}" + "".join(f"{t[:12]:>13s}" for t in types) + "   gap")
    for policy in results["policies"]:
        rewards = "".join(f"{float(results['mean_reward'][policy][t]):+13.3f}" for t in types)
        gaps = " ".join(f"{float(results['gap'][policy][t]):.2f}" for t in types)
        print(f"{policy:22s}{rewards}   {gaps}")
    if "rule_based" in results["policies"]:
        gaps = ", ".join(f"{t}={float(results['gap']['rule_based'][t]):.2f}" for t in types)
        print(f"rule_based gaps: {gaps} (spread {differential_spread(results, 'rule_based'):.2f})")
    floor = attacker_floor(results)
    print(f"attacker floor over the error types: {floor}")
    print(f"skipped seeds: {results['skipped']}")
    print(f"sha256 {RESULTS_FILE}: {digest}")
    print(f"wrote {docs / RESULTS_FILE}, {docs / CHART_FILE}, {docs / REPORT_FILE} "
          f"({time.monotonic() - started:.2f}s total)")
    if failures:
        print("sanity failures: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


__all__ = [
    "CHART_FILE",
    "DOCS_DIR",
    "FLAT_SPREAD",
    "FLOOR_GAP",
    "REPORT_FILE",
    "RESULTS_FILE",
    "SILENT_GAP",
    "main",
    "render_report",
]

if __name__ == "__main__":
    raise SystemExit(main())
