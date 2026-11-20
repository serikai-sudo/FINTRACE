"""Drive the complete jury protocol once, through the public ``judge_adapter`` interface only.

Usage::

    python scripts/jury_protocol_smoke.py [--seed 10000] [--work-dir DIR]
                                          [--team-config config/dev.yaml]
    docker run --rm -i --network none --cpus 2 --memory 4g --entrypoint python \
        fintrace:0.1.0 - < scripts/jury_protocol_smoke.py   # no scripts/ in the image: stdin

generate_episode -> make_env -> reset (closed mode) -> the real rule-based baseline until the
episode ends (at most 100 steps) -> score_episode. Prints a JSON summary with the seven score
fields and ``PASS`` or ``FAIL`` (exit 1). Needs no network and nothing outside ``src/`` +
``judge_adapter.py``: without ``--team-config`` an embedded team config (four single injections)
is used, and a seed that is infeasible for it is skipped for the next one.
``tests/integration/test_jury_protocol_offline.py`` runs ``run_protocol`` under a socket guard;
``make docker-verify`` runs this file inside the image. No ``__file__`` use: the file may be
executed from stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Final

import judge_adapter

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.constants import MAX_STEPS
from fintrace.generator.injectors.base import InjectorError

SCORE_KEYS: Final[tuple[str, ...]] = ("tp", "fp", "fn", "precision", "recall", "f1", "return")
MAX_SEED_TRIES: Final[int] = 10
# dev-like sizes with one injected issue of each type; no config file needed inside the image
EMBEDDED_TEAM_CONFIG: Final[dict[str, Any]] = {
    "n_ledger_rows_range": [500, 1500],
    "n_documents_range": [30, 150],
    "n_entities_range": [8, 20],
    "errors": {
        "duplicate_invoice": {"count": 1},
        "payment_allocation": {"count": 1},
        "fx_error": {"count": 1},
        "period_misclassification": {"count": 1},
    },
    "decoys": {
        "repeated_payment_amounts": 1,
        "split_payments": 1,
        "refunds": 1,
        "similar_invoices": 1,
    },
}


class _View:
    """Observation view for the rule-based agent built from the jury envelope."""

    def __init__(self, tool: str, payload: dict[str, Any]) -> None:
        self.ok = payload.get("error") is None
        self.tool = tool
        self.result = payload.get("result") if tool != "reset" else payload
        self.steps_remaining = int(payload["remaining_steps"])
        self.steps_used = MAX_STEPS - self.steps_remaining


def _rewrap(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Undo the adapter's result projection so the baseline sees the internal tool shapes."""
    if payload.get("error") is not None:
        return payload
    result = payload["result"]
    if tool == "list_records":
        result = {**result, "records_total": len(result["records"])}
    elif tool == "get_record":
        result = {"record": result}
    elif tool == "get_document":
        result = {"document": result}
    return {**payload, "result": result}


def _generate(seed: int, team_config: dict[str, Any], episode_dir: Path, tries: int) -> int:
    """Write the episode for the first feasible seed in ``seed, seed + 1, ...``; return it."""
    last: Exception | None = None
    for candidate in range(seed, seed + tries):
        try:
            judge_adapter.generate_episode(
                candidate, {"team_config": team_config}, str(episode_dir)
            )
        except InjectorError as exc:  # infeasible for this config: try the next seed
            last = exc
            continue
        return candidate
    raise RuntimeError(f"no feasible seed in {seed}..{seed + tries - 1}") from last


def run_protocol(
    seed: int,
    work_dir: Path,
    team_config: dict[str, Any] | None = None,
    *,
    max_seed_tries: int = MAX_SEED_TRIES,
) -> dict[str, Any]:
    """One complete jury episode; returns a JSON-ready summary (no wall clock inside)."""
    config = EMBEDDED_TEAM_CONFIG if team_config is None else team_config
    episode_dir = work_dir / "episode"
    used_seed = _generate(seed, config, episode_dir, max_seed_tries)
    env = judge_adapter.make_env()
    first, info = env.reset(0, {"mode": "closed", "episode_dir": str(episode_dir)})
    agent = RuleBasedReconciliation()
    agent.reset()
    view = _View("reset", first)
    rewards: list[float] = []
    terminated = truncated = False
    findings: list[Any] = []
    for _ in range(MAX_STEPS):
        action = agent.act(view)
        tool = str(action["tool"])
        jury_action: dict[str, Any] = {"name": tool, "args": action["args"]}
        if tool == "submit":
            findings = list(action["args"]["findings"])
            jury_action["args"] = {"findings": {"findings": findings}}
        payload, reward, terminated, truncated, info = env.step(jury_action)
        rewards.append(float(reward))
        view = _View(tool, _rewrap(tool, payload))
        if terminated or truncated:
            break
    inspection_steps = max(0, len(rewards) - 1) if terminated else len(rewards)
    score = judge_adapter.score_episode(str(episode_dir), {"findings": findings}, inspection_steps)
    return {
        "seed": used_seed,
        "episode_id": first["manifest"]["episode_id"],
        "steps": len(rewards),
        "terminated": terminated,
        "truncated": truncated,
        "policy_rewards_zero": all(r == 0.0 for r in rewards),
        "info_empty": info == {},
        "n_findings": len(findings),
        "score": {key: score[key] for key in SCORE_KEYS},
    }


def check_summary(summary: dict[str, Any]) -> list[str]:
    """Why ``summary`` is not a sound protocol run (empty = PASS)."""
    problems: list[str] = []
    if not (summary["terminated"] or summary["truncated"]):
        problems.append("episode did not end")
    if summary["steps"] > MAX_STEPS:
        problems.append("more than MAX_STEPS steps")
    if not summary["policy_rewards_zero"]:
        problems.append("closed mode leaked a non-zero reward to the policy")
    score = summary["score"]
    if any(score[key] is None for key in SCORE_KEYS):
        problems.append("score has null metrics (invalid submission)")
    elif not -1.0 <= float(score["return"]) <= 1.0:
        problems.append("return outside [-1, 1]")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 0 iff the protocol completed with a sound score."""
    parser = argparse.ArgumentParser(description="Run the jury protocol once, offline.")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--team-config", type=Path, default=None, help="YAML EpisodeConfig")
    args = parser.parse_args(argv)
    team_config: dict[str, Any] | None = None
    if args.team_config is not None:
        import yaml  # only for an explicit config file; the image path never needs it

        team_config = yaml.safe_load(args.team_config.read_text(encoding="utf-8"))
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="fintrace_jury_"))
    started = time.monotonic()  # wall clock for the operator's cold-start reading only
    summary = run_protocol(args.seed, work_dir, team_config)
    elapsed = time.monotonic() - started
    problems = check_summary(summary)
    print(json.dumps(summary, sort_keys=True))
    print(f"protocol wall time: {elapsed:.2f}s (generate + {summary['steps']} steps + score)")
    if problems:
        print("FAIL: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("PASS")
    return 0


__all__ = [
    "EMBEDDED_TEAM_CONFIG",
    "MAX_SEED_TRIES",
    "SCORE_KEYS",
    "check_summary",
    "main",
    "run_protocol",
]

if __name__ == "__main__":
    raise SystemExit(main())
