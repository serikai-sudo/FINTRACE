"""``python -m fintrace``: a live demo episode (the Docker ENTRYPOINT).

One episode is built in memory from the embedded demo config (``constants.DEMO_TEAM_CONFIG``; no
``config/``, ``data/`` or network, so it runs inside the image), investigated by the rule-based
auditor through the real environment, and scored against the hidden truth with the same three calls
``judge_adapter.score_episode`` makes after reading an episode directory: ``match`` ->
``compute_metrics`` -> ``official_reward``. The output shows the agent's own step notes, the
submitted findings and the seven score fields; it is byte-stable for a given seed (no wall clock,
no paths). Output is written to the given stream from this entry point only: no ``print()`` in
``src/`` (CONTRIBUTING.md §4).

Usage::

    python -m fintrace [--seed 10000] [--full]

Exit 0 when the recomputed return equals the environment's terminal reward, 1 on a mismatch,
2 when the seed is infeasible for the demo config.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from typing import Any, Final, TextIO

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.constants import DEMO_SEED, DEMO_TEAM_CONFIG, ISSUE_TYPES, MAX_STEPS, TOOL_NAMES
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.matcher import match, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.evaluator.runner import run_baseline
from fintrace.generator.build import build_episode
from fintrace.generator.injectors.base import InjectorError
from fintrace.models import EpisodeConfig, JsonObject, Submission
from fintrace.reward.official import official_reward

SCORE_KEYS: Final[tuple[str, ...]] = ("tp", "fp", "fn", "precision", "recall", "f1", "return")
FINDING_MARK: Final[str] = "recorded"  # the rule-based auditor's word for a finding it just logged
WIDTH: Final[int] = 100
NOTE_INDENT: Final[str] = " " * 12
REWARD_TOLERANCE: Final[float] = 1e-9

DemoResult = dict[str, Any]


# ---------------------------------------------------------------- run
def _submitted_findings(steps: list[JsonObject]) -> list[JsonObject]:
    """The findings of the final ``submit`` action, or none when the episode was truncated."""
    if not steps or steps[-1].get("tool") != "submit":
        return []
    args = steps[-1].get("args")
    findings = args.get("findings") if isinstance(args, dict) else None
    return list(findings) if isinstance(findings, list) else []


def run_demo(seed: int = DEMO_SEED, config: EpisodeConfig | None = None) -> DemoResult:
    """Build, investigate and score one episode in memory.

    Raises ``InjectorError`` when ``seed`` is infeasible for the config. The books are built twice
    from the same seed (once inside ``env.reset``, once here for the truth); §6.5 determinism makes
    them identical, and the episode ids are compared to prove it.
    """
    cfg = EpisodeConfig.model_validate(DEMO_TEAM_CONFIG) if config is None else config
    env = FintraceEnv()
    steps: list[JsonObject] = []
    result = run_baseline(RuleBasedReconciliation(), env, seed, cfg, trace=steps)
    manifest = env.episode.manifest
    private = build_episode(seed, cfg)
    if private.truth.episode_id != result["episode_id"]:
        raise RuntimeError("the episode rebuilt for scoring differs from the one investigated")
    findings = _submitted_findings(steps) if result["terminated"] else []
    submission = Submission.model_validate(
        {"episode_id": result["episode_id"], "findings": findings}
    )
    tp, fp, fn = match(
        submission.findings, private.truth.issues, public_ids=public_ids_of(private.public)
    )
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    inspection_steps = int(result["steps"]) - 1 if result["terminated"] else MAX_STEPS
    score: dict[str, int | float] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "return": official_reward(f1, inspection_steps),
    }
    return {
        "seed": seed,
        "config": cfg,
        "manifest": manifest,
        "steps": steps,
        "terminated": bool(result["terminated"]),
        "truncated": bool(result["truncated"]),
        "findings": [finding.model_dump(mode="json") for finding in submission.findings],
        "inspection_steps": inspection_steps,
        "score": score,
        "env_reward": float(result["reward"]),
    }


# ---------------------------------------------------------------- render
def _phase(action: JsonObject) -> str:
    tool = str(action.get("tool"))
    args = action.get("args")
    table = args.get("table") if isinstance(args, dict) else None
    return f"{tool}({table})" if table else tool


def _label(action: JsonObject) -> str:
    tool = str(action.get("tool"))
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if tool == "list_records":
        cursor = args.get("cursor")
        return f"{tool} {args.get('table')} {cursor or 'first page'}"
    if tool == "get_record":
        return f"{tool} {args.get('table')} {args.get('id')}"
    if tool == "get_document":
        return f"{tool} {args.get('doc_id')}"
    if tool == "calculator":
        return f"{tool} {str(args.get('expression', ''))[:40]}"
    if tool == "submit":
        findings = args.get("findings")
        return f"{tool} ({len(findings) if isinstance(findings, list) else 0} findings)"
    return tool


def key_step_indexes(steps: list[JsonObject], *, full: bool = False) -> list[int]:
    """Steps worth showing: the first, the last, every phase change (tool or table) and every step
    whose note says a finding was recorded. ``full`` keeps them all."""
    if full:
        return list(range(len(steps)))
    keep: list[int] = []
    previous = ""
    for index, action in enumerate(steps):
        phase = _phase(action)
        note = str(action.get("thought", ""))
        if index in (0, len(steps) - 1) or phase != previous or FINDING_MARK in note:
            keep.append(index)
        previous = phase
    return keep


def _wrap(text: str, indent: str = NOTE_INDENT, first: str | None = None) -> list[str]:
    initial = indent if first is None else first
    lines = textwrap.wrap(text, width=WIDTH, initial_indent=initial, subsequent_indent=indent)
    return lines or [indent.rstrip()]


def _header(demo: DemoResult) -> list[str]:
    cfg: EpisodeConfig = demo["config"]
    manifest = demo["manifest"]
    counts = ", ".join(f"{name}={getattr(cfg.errors, name).count}" for name in ISSUE_TYPES)
    decoys = ", ".join(f"{k}={v}" for k, v in cfg.decoys.model_dump().items())
    title = "FINTRACE demo: one seeded episode, the rule-based auditor, the official score"
    return [
        title,
        "=" * len(title),
        f"Episode   {manifest.episode_id}   seed {demo['seed']}   built in memory from the "
        "embedded demo config",
        *_wrap(
            f"{cfg.errors.total()} injected issues ({counts}); decoys {decoys}. "
            "The agent never sees this line.",
            indent=" " * 10,
            first="Config    ",
        ),
        *_wrap(
            f"{manifest.n_ledger_rows} ledger rows, {manifest.n_documents} documents, base "
            f"currency {manifest.base_currency}, period {manifest.period_start[:10]} -> "
            f"{manifest.period_end[:10]} ({manifest.timezone}). From the public manifest, "
            "as the agent sees it.",
            indent=" " * 10,
            first="Books     ",
        ),
        f"Budget    {MAX_STEPS} steps; tools: {', '.join(TOOL_NAMES)}",
    ]


def _investigation(demo: DemoResult, *, full: bool) -> list[str]:
    steps: list[JsonObject] = demo["steps"]
    shown = key_step_indexes(steps, full=full)
    scope = "every step" if full else "key steps"
    lines = [
        "",
        f"Investigation: {len(shown)} {scope} of {len(steps)} (the agent's own notes; "
        "--full prints every step)",
    ]
    previous = -1
    for index in shown:
        skipped = index - previous - 1
        if skipped > 0:
            plural = "step" if skipped == 1 else "steps"
            lines.append(f"{NOTE_INDENT}... {skipped} more {_phase(steps[previous + 1])} {plural}")
        action = steps[index]
        lines.append(f"  step {index + 1:3d}  {_label(action)}")
        note = str(action.get("thought") or "")
        if note:
            lines.extend(_wrap(note))
        previous = index
    if demo["truncated"] and not demo["terminated"]:
        lines.append(f"{NOTE_INDENT}(step budget exhausted without a submission)")
    return lines


def _findings(demo: DemoResult) -> list[str]:
    findings: list[JsonObject] = demo["findings"]
    lines = ["", f"Findings submitted ({len(findings)})"]
    for number, finding in enumerate(findings, start=1):
        lines.append(
            f"  {number}. {finding['type']:<26} affected {', '.join(finding['affected_ids'])}   "
            f"amount {finding['claimed_amount_minor']} {finding['currency']} (minor units)"
        )
        lines.append(f"     evidence {', '.join(finding['evidence_ids'])}")
        lines.extend(_wrap(str(finding["explanation"]), indent="     "))
    return lines


def _score(demo: DemoResult, *, consistent: bool) -> list[str]:
    score = demo["score"]
    cells = [
        f"{key} {score[key]}" if isinstance(score[key], int) else f"{key} {score[key]:.3f}"
        for key in SCORE_KEYS
    ]
    verdict = "equal" if consistent else "MISMATCH"
    return [
        "",
        *_wrap(
            "Score against the hidden truth (match -> compute_metrics -> official_reward, exactly "
            "the calls judge_adapter.score_episode makes)",
            indent="",
        ),
        "  " + "  ".join(cells),
        *_wrap(
            f"return = clip(2*f1 - 1 - 0.002*inspection_steps, -1, 1) with "
            f"{demo['inspection_steps']} inspection steps; the environment's terminal reward is "
            f"{demo['env_reward']:.3f} ({verdict}).",
            indent="  ",
        ),
        "",
        "Reproduce: make demo | python -m fintrace [--seed N] [--full] |",
        "           docker run --rm --network none fintrace:0.1.0",
    ]


def render_demo(demo: DemoResult, *, full: bool = False, consistent: bool = True) -> list[str]:
    """The complete demo text as lines (ASCII, no wall clock, no paths: byte-stable per seed)."""
    return [
        *_header(demo),
        *_investigation(demo, full=full),
        *_findings(demo),
        *_score(demo, consistent=consistent),
    ]


# ---------------------------------------------------------------- entry point
def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    """Run the demo and write it to ``out`` (default: stdout); return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m fintrace",
        description="Live demo: one seeded in-memory episode, the rule-based auditor, the score.",
    )
    parser.add_argument("--seed", type=int, default=DEMO_SEED, help=f"episode seed ({DEMO_SEED})")
    parser.add_argument("--full", action="store_true", help="print every step, not only key ones")
    args = parser.parse_args(argv)
    stream = sys.stdout if out is None else out
    try:
        demo = run_demo(args.seed)
    except InjectorError:
        sys.stderr.write(f"seed {args.seed} is infeasible for the demo config; try another seed\n")
        return 2
    consistent = math.isclose(demo["score"]["return"], demo["env_reward"], abs_tol=REWARD_TOLERANCE)
    stream.write("\n".join(render_demo(demo, full=args.full, consistent=consistent)) + "\n")
    return 0 if consistent else 1


__all__ = ["SCORE_KEYS", "key_step_indexes", "main", "render_demo", "run_demo"]

if __name__ == "__main__":
    raise SystemExit(main())
