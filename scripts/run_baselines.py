"""Run the four baselines on a committed split and write ``docs/results_<split>.md``.

Usage::

    python scripts/run_baselines.py [--split {train,dev}] [--seeds N] [--out PATH]
                                    [--baselines random,rule_based,mark_everything,oracle]

The markdown has two parts. Everything above ``TIMING_MARKER`` is deterministic: precision, recall,
F1, return and steps per policy, the per-episode tables and the per-type failure table, so a rerun
on unchanged code reproduces it byte for byte (the slow integration test pins exactly that part).
Below the marker comes the wall clock of this run, which differs per machine and is not pinned.
Precision, recall and F1 come from the evaluator's own ``match`` and ``compute_metrics`` applied to
the hidden truth of the committed episode directory, and the recomputed official return is checked
against the reward the environment paid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from fintrace.agents.base import BASELINE_NAMES
from fintrace.constants import DATA_DIR, ISSUE_TYPES, MAX_STEPS
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.evaluator.runner import (
    BASELINE_FACTORIES,
    REPO_ROOT,
    EpisodeResult,
    run_split,
    summarize,
)
from fintrace.exporters.episode_files import read_episode
from fintrace.models import JsonObject, PrivateEpisode, Submission
from fintrace.reward.official import official_reward

DOCS_DIR: Final[Path] = REPO_ROOT / "docs"
CONFIG_HASH_HEX: Final[int] = 12
SPLITS: Final[tuple[str, ...]] = ("train", "dev")
MANIFEST_FILE: Final[str] = "manifest.json"  # data/manifest.json, as written by `make data`
TIMING_MARKER: Final[str] = "<!-- wall-clock timing below: not byte-pinned, differs per machine -->"
REWARD_TOLERANCE: Final[float] = 1e-9
SEE_ALSO: Final[str] = (
    "See also: `docs/oracle_gap_report.md` (gap to the oracle per error type on paired episodes) "
    "and `experiments/bc_proof/bc_proof_report.md` (per-class recall of the CPU imitator)."
)

EpisodeScore = dict[str, Any]
Timing = dict[str, float]


def config_hash(config_dir: Path) -> str:
    """Same recipe as scripts/generate_data.py: sha256 over train.yaml + dev.yaml bytes."""
    digest = hashlib.sha256()
    for split in SPLITS:
        digest.update((config_dir / f"{split}.yaml").read_bytes())
    return digest.hexdigest()[:CONFIG_HASH_HEX]


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


# ---------------------------------------------------------------- scoring (evaluator functions)
def episode_dirs(split: str, data_dir: Path) -> dict[int, Path]:
    """seed -> committed episode directory of ``split`` (from data/manifest.json)."""
    manifest = json.loads((data_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    for report in manifest["splits"]:
        if report["split"] == split:
            return {int(e["seed"]): data_dir / str(e["dir"]) for e in report["episodes"]}
    raise KeyError(f"split {split!r} not in {MANIFEST_FILE}")


def split_traces(results: list[EpisodeResult], trace: list[JsonObject]) -> list[list[JsonObject]]:
    """One action list per episode: every ``policy.act()`` is followed by exactly one
    ``env.step()``, so the flat trace of ``run_split`` splits on the cumulative step counts."""
    traces: list[list[JsonObject]] = []
    start = 0
    for result in results:
        end = start + int(result["steps"])
        traces.append(trace[start:end])
        start = end
    if start != len(trace):
        raise RuntimeError(f"trace holds {len(trace)} actions, the step counts sum to {start}")
    return traces


def _submitted_findings(actions: list[JsonObject], terminated: bool) -> list[JsonObject]:
    """The findings the environment accepted: the final submit of a terminated episode."""
    if not terminated or not actions or actions[-1].get("tool") != "submit":
        return []
    args = actions[-1].get("args")
    findings = args.get("findings") if isinstance(args, dict) else None
    return list(findings) if isinstance(findings, list) else []


def _submission_refused(actions: list[JsonObject], terminated: bool) -> bool:
    """True when the episode ended because the environment refused the final submit.

    The environment ends an episode at -1 when a well-formed submit carries a payload the schema
    refuses (jury Block 3, tests 18-20). That is decided on the raw action the policy sent, not on
    the two fields ``_submitted_findings`` keeps, so extra keys and missing fields are caught too.
    """
    if not terminated or not actions or actions[-1].get("tool") != "submit":
        return False
    args = actions[-1].get("args")
    if not isinstance(args, dict):
        return True
    try:
        Submission.model_validate(args)
    except ValidationError:
        return True
    return False


def _refused_submission_score(n_issues: int, reward: float) -> EpisodeScore:
    """Score of an episode the environment ended because it refused the submission."""
    return {
        "tp": 0,
        "fp": 0,
        "fn": n_issues,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "return": reward,
        "n_findings": 0,
        "submitted": False,
        "per_type": {t: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for t in ISSUE_TYPES},
    }


def score_episode(
    result: EpisodeResult, actions: list[JsonObject], private: PrivateEpisode
) -> EpisodeScore:
    """tp/fp/fn (overall and per issue type), precision/recall/F1 and the official return of one
    episode, computed with the evaluator's ``match`` / ``compute_metrics`` / ``official_reward``
    and checked against the reward the environment paid."""
    if private.truth.episode_id != result["episode_id"]:
        raise RuntimeError(
            f"seed {result['seed']}: the environment built {result['episode_id']}, the committed "
            f"directory holds {private.truth.episode_id}"
        )
    terminated = bool(result["terminated"])
    if _submission_refused(actions, terminated):
        # the environment refused this submission and ended the episode at -1 (jury tests 18-20);
        # no finding was scored, so every planted issue stays unfound
        return _refused_submission_score(len(private.truth.issues), float(result["reward"]))
    submission = Submission.model_validate(
        {"episode_id": result["episode_id"], "findings": _submitted_findings(actions, terminated)}
    )
    public_ids = public_ids_of(private.public)
    per_type: dict[str, dict[str, int]] = {}
    tp = fp = fn = 0
    for issue_type in ISSUE_TYPES:  # a finding only ever matches an issue of its own type, so the
        findings = [f for f in submission.findings if f.type == issue_type]  # matching splits
        issues = [i for i in private.truth.issues if i.type == issue_type]  # exactly by type
        t_tp, t_fp, t_fn = match(findings, issues, public_ids=public_ids)
        per_type[issue_type] = {"tp": t_tp, "fp": t_fp, "fn": t_fn, "total": len(issues)}
        tp, fp, fn = tp + t_tp, fp + t_fp, fn + t_fn
    fp += sum(1 for f in submission.findings if f.type not in ISSUE_TYPES)  # unknown type: FP
    if (tp, fp, fn) != match(submission.findings, private.truth.issues, public_ids=public_ids):
        raise RuntimeError(f"seed {result['seed']}: per-type counts do not add up to the match")
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    inspection_steps = int(result["steps"]) - 1 if terminated else MAX_STEPS
    ret = official_reward(f1, inspection_steps)
    if abs(ret - float(result["reward"])) > REWARD_TOLERANCE:
        raise RuntimeError(
            f"seed {result['seed']}: recomputed return {ret:.6f} != environment reward "
            f"{float(result['reward']):.6f}"
        )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "return": ret,
        "n_findings": len(submission.findings),
        "submitted": terminated,
        "per_type": per_type,
    }


def pooled(scores: list[EpisodeScore]) -> dict[str, float | int]:
    """tp/fp/fn summed over the episodes and the precision/recall/F1 of the sums."""
    tp = sum(int(s["tp"]) for s in scores)
    fp = sum(int(s["fp"]) for s in scores)
    fn = sum(int(s["fn"]) for s in scores)
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def pooled_per_type(scores: list[EpisodeScore]) -> dict[str, dict[str, int]]:
    out = {t: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for t in ISSUE_TYPES}
    for score in scores:
        for issue_type, counts in score["per_type"].items():
            for key in ("tp", "fp", "fn", "total"):
                out[issue_type][key] += int(counts[key])
    return out


def run_all(
    split: str,
    names: list[str],
    *,
    limit: int | None = None,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
) -> tuple[dict[str, list[EpisodeResult]], dict[str, list[EpisodeScore]], Timing]:
    """Run every named baseline on the split; results, per-episode scores and wall-clock timing."""
    data_root = data_dir or REPO_ROOT / DATA_DIR
    dirs = episode_dirs(split, data_root)
    privates: dict[int, PrivateEpisode] = {}
    results: dict[str, list[EpisodeResult]] = {}
    scores: dict[str, list[EpisodeScore]] = {}
    timing: Timing = {}
    total_started = time.monotonic()
    for name in names:
        started = time.monotonic()
        trace: list[JsonObject] = []
        rows = run_split(
            BASELINE_FACTORIES[name],
            split,
            limit=limit,
            data_dir=data_dir,
            config_dir=config_dir,
            trace=trace,
        )
        episode_scores: list[EpisodeScore] = []
        for row, actions in zip(rows, split_traces(rows, trace), strict=True):
            seed = int(row["seed"])
            if seed not in privates:
                privates[seed] = read_episode(dirs[seed])
            episode_scores.append(score_episode(row, actions, privates[seed]))
        results[name] = rows
        scores[name] = episode_scores
        timing[name] = time.monotonic() - started
    timing["total"] = time.monotonic() - total_started
    return results, scores, timing


# ---------------------------------------------------------------- rendering
def _per_episode_table(
    title: str, names: list[str], episodes: list[EpisodeResult], cell: Callable[[str, int], str]
) -> list[str]:
    header = "| Seed | Episode | " + " | ".join(names) + " |"
    rule = "|---:|---|" + "---:|" * len(names)
    lines = ["", f"## {title}", "", header, rule]
    for index, episode in enumerate(episodes):
        cells = " | ".join(cell(name, index) for name in names)
        lines.append(f"| {episode['seed']} | {episode['episode_id']} | {cells} |")
    return lines


def _failure_line(name: str, scores: list[EpisodeScore]) -> str:
    agg = pooled(scores)
    per_type = pooled_per_type(scores)
    planted = sum(c["total"] for c in per_type.values())
    weakest = min(
        (t for t in ISSUE_TYPES if per_type[t]["total"]),
        key=lambda t: (per_type[t]["tp"] / per_type[t]["total"], ISSUE_TYPES.index(t)),
        default=None,
    )
    if weakest is None or per_type[weakest]["tp"] == per_type[weakest]["total"]:
        weak_text = "no planted issue missed"
    elif int(agg["tp"]) == 0:
        weak_text = "no planted issue of any type found"
    else:
        counts = per_type[weakest]
        weak_text = f"weakest type `{weakest}` ({counts['tp']}/{counts['total']})"
    submitted = sum(1 for s in scores if s["submitted"])
    findings = sum(int(s["n_findings"]) for s in scores) / max(1, len(scores))
    return (
        f"- **{name}**: finds {agg['tp']} of {planted} planted issues (recall "
        f"{float(agg['recall']):.3f}) with {agg['fp']} false positives (precision "
        f"{float(agg['precision']):.3f}); {weak_text}; valid submission in "
        f"{submitted}/{len(scores)} episodes, {findings:.1f} findings per episode on average."
    )


def render_markdown(
    split: str,
    results: dict[str, list[EpisodeResult]],
    scores: dict[str, list[EpisodeScore]],
    config_hash_hex: str,
) -> str:
    """The deterministic report (everything above ``TIMING_MARKER``)."""
    names = list(results)
    episodes = results[names[0]] if names else []
    lines = [
        f"# Baseline results: {split} split",
        "",
        f"Generated by `scripts/run_baselines.py --split {split}` on {len(episodes)} committed "
        f"episodes (config hash `{config_hash_hex}`). "
        "Return = clip(2 * F1 - 1 - 0.002 * inspection_steps, -1, 1); the oracle reads the "
        "evidence documents before submitting, so even a perfect audit pays for its steps. The "
        "rule-based agent keeps reading payment documents until its step budget runs out, so "
        "allocation issues late in the payment list can be missed by design.",
        "",
        "## Summary",
        "",
        "Precision, recall and F1 are pooled over the episodes (sums of tp / fp / fn, scored with "
        "the evaluator's own matcher against the hidden truth); Return is the mean official "
        "reward, Steps the mean number of actions.",
        "",
        "| Policy | Precision | Recall | F1 | Return | Std | Min | Max | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        stats = summarize(results[name])
        agg = pooled(scores[name])
        lines.append(
            f"| {name} | {float(agg['precision']):.3f} | {float(agg['recall']):.3f} | "
            f"{float(agg['f1']):.3f} | {_fmt(float(stats['mean_reward']))} | "
            f"{float(stats['std_reward']):.3f} | {_fmt(float(stats['min_reward']))} | "
            f"{_fmt(float(stats['max_reward']))} | {float(stats['mean_steps']):.1f} |"
        )
    lines += _per_episode_table(
        "Reward per episode", names, episodes, lambda n, i: _fmt(float(results[n][i]["reward"]))
    )
    lines += _per_episode_table(
        "F1 per episode", names, episodes, lambda n, i: f"{float(scores[n][i]['f1']):.3f}"
    )
    lines += _per_episode_table(
        "Steps per episode", names, episodes, lambda n, i: str(results[n][i]["steps"])
    )
    lines += [
        "",
        "## Where each policy fails",
        "",
        f"Pooled over the {len(episodes)} episodes: planted issues found per error type "
        "(tp/total) and false positives reported under that type.",
        "",
        "| Policy | " + " | ".join(ISSUE_TYPES) + " |",
        "|---|" + "---:|" * len(ISSUE_TYPES),
    ]
    for name in names:
        per_type = pooled_per_type(scores[name])
        cells = " | ".join(
            f"{per_type[t]['tp']}/{per_type[t]['total']} found, {per_type[t]['fp']} fp"
            for t in ISSUE_TYPES
        )
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    lines += [_failure_line(name, scores[name]) for name in names]
    lines += ["", SEE_ALSO]
    return "\n".join(lines) + "\n"


def compose_document(markdown: str, timing_text: str) -> str:
    """The file: pinned markdown, a blank line, the marker, a blank line, the timing section."""
    return f"{markdown}\n{TIMING_MARKER}\n\n{timing_text}"


def split_document(document: str) -> tuple[str, str]:
    """Inverse of ``compose_document``: (pinned markdown, timing section)."""
    head, separator, tail = document.partition(f"\n{TIMING_MARKER}\n\n")
    if not separator:
        raise ValueError("no timing marker in the document")
    return head, tail


def render_timing(results: dict[str, list[EpisodeResult]], timing: Timing) -> str:
    """Wall clock of this run (below the marker, not pinned): loop time per episode and total."""
    lines = [
        "## Timing (this machine, not byte-stable)",
        "",
        "Loop = the policy's act/step loop per episode as measured by the runner; Wall = episode "
        "generation + loop + scoring for the whole split. Measured with `time.monotonic`.",
        "",
        "| Policy | Loop, s per episode (mean) | Wall, s |",
        "|---|---:|---:|",
    ]
    for name, rows in results.items():
        loop = sum(float(r["elapsed_sec"]) for r in rows) / max(1, len(rows))
        lines.append(f"| {name} | {loop:.3f} | {timing[name]:.2f} |")
    lines.append(f"| total |  | {timing['total']:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Run the baselines on a committed split.")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--seeds", type=int, default=None, help="first N episodes (default all)")
    parser.add_argument("--out", type=Path, default=None, help="default docs/results_<split>.md")
    parser.add_argument(
        "--baselines", default=",".join(BASELINE_NAMES), help="comma-separated policy names"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.baselines.split(",") if name.strip()]
    unknown = [name for name in names if name not in BASELINE_FACTORIES]
    if unknown:
        known = sorted(BASELINE_FACTORIES)
        print(f"unknown baselines: {unknown} (known: {known})", file=sys.stderr)
        return 2
    config_dir = args.config_dir or REPO_ROOT / "config"
    results, scores, timing = run_all(
        args.split, names, limit=args.seeds, data_dir=args.data_dir, config_dir=config_dir
    )
    for name in names:
        stats = summarize(results[name])
        agg = pooled(scores[name])
        print(
            f"{name:16s} n={stats['n']:2d} P={float(agg['precision']):.3f} "
            f"R={float(agg['recall']):.3f} F1={float(agg['f1']):.3f} "
            f"return={float(stats['mean_reward']):+.3f} std={float(stats['std_reward']):.3f} "
            f"steps={float(stats['mean_steps']):.1f} ({timing[name]:.2f}s)"
        )
    out = args.out or DOCS_DIR / f"results_{args.split}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(args.split, results, scores, config_hash(config_dir))
    out.write_bytes(compose_document(markdown, render_timing(results, timing)).encode("utf-8"))
    print(f"wrote {out} ({timing['total']:.2f}s total)")
    return 0


__all__ = [
    "DOCS_DIR",
    "SPLITS",
    "TIMING_MARKER",
    "compose_document",
    "config_hash",
    "episode_dirs",
    "main",
    "pooled",
    "pooled_per_type",
    "render_markdown",
    "render_timing",
    "run_all",
    "score_episode",
    "split_document",
    "split_traces",
]

if __name__ == "__main__":
    raise SystemExit(main())
