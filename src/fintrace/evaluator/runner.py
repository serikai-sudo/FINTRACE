"""Baseline runner: one policy, one environment, one episode (or a whole split).

The runner sees exactly what an external harness sees: the observations and the terminal
reward returned by ``FintraceEnv.step``. It never reads the private episode; the oracle gets
its truth through ``fintrace.evaluator.testing.oracle_for``, which is why factories receive
the environment *after* ``reset``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import yaml

from fintrace.agents.base import Policy
from fintrace.agents.mark_everything import MarkEverythingAttack
from fintrace.agents.random_agent import RandomInvestigator
from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.constants import CONFIG_DIR, DATA_DIR, MAX_STEPS
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.testing import oracle_for
from fintrace.models import EpisodeConfig, JsonObject

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_MANIFEST: Final[str] = "manifest.json"
EpisodeResult = dict[str, Any]
PolicyFactory = Callable[[FintraceEnv], Policy]

BASELINE_FACTORIES: Final[dict[str, PolicyFactory]] = {
    "random": lambda _env: RandomInvestigator(seed=0),
    "rule_based": lambda _env: RuleBasedReconciliation(),
    "mark_everything": lambda _env: MarkEverythingAttack(),
    "oracle": oracle_for,
}


def _run(
    factory: PolicyFactory,
    env: FintraceEnv,
    seed: int,
    config: EpisodeConfig | None,
    trace: list[JsonObject] | None,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed, config=config)
    policy = factory(env)
    policy.reset()
    started = time.monotonic()
    steps = 0
    reward = 0.0
    terminated = truncated = False
    n_findings = 0
    while steps < MAX_STEPS and not (terminated or truncated):
        action = policy.act(observation)
        if trace is not None:
            trace.append(action)
        if action.get("tool") == "submit":
            args = action.get("args")
            findings = args.get("findings") if isinstance(args, dict) else None
            n_findings = len(findings) if isinstance(findings, list) else 0
        observation, reward, terminated, truncated, _ = env.step(action)
        steps += 1
    return {
        "policy": policy.name,
        "seed": seed,
        "episode_id": env.episode.manifest.episode_id,
        "steps": steps,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "n_findings": n_findings,
        "elapsed_sec": time.monotonic() - started,
    }


def run_baseline(
    policy: Policy,
    env: FintraceEnv,
    seed: int,
    config: EpisodeConfig | None = None,
    *,
    trace: list[JsonObject] | None = None,
) -> EpisodeResult:
    """Run one episode with a ready policy; ``trace`` (if given) collects every action dict."""
    return _run(lambda _env: policy, env, seed, config, trace)


def load_split_config(split: str, config_dir: Path | None = None) -> EpisodeConfig:
    """``config/<split>.yaml`` as an ``EpisodeConfig``."""
    path = (config_dir or REPO_ROOT / CONFIG_DIR) / f"{split}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return EpisodeConfig.model_validate(raw)


def split_seeds(split: str, data_dir: Path | None = None) -> list[int]:
    """The seeds of the committed episodes of ``split`` (from ``data/manifest.json``)."""
    manifest = json.loads(
        ((data_dir or REPO_ROOT / DATA_DIR) / DATA_MANIFEST).read_text(encoding="utf-8")
    )
    for report in manifest["splits"]:
        if report["split"] == split:
            return [int(entry["seed"]) for entry in report["episodes"]]
    raise KeyError(f"split {split!r} not in {DATA_MANIFEST}")


def run_split(
    policy_factory: PolicyFactory,
    split: str,
    *,
    limit: int | None = None,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
    trace: list[JsonObject] | None = None,
) -> list[EpisodeResult]:
    """Run a policy (built per episode) on the first ``limit`` committed episodes of a split."""
    seeds = split_seeds(split, data_dir)[:limit]
    config = load_split_config(split, config_dir)
    env = FintraceEnv()
    return [_run(policy_factory, env, seed, config, trace) for seed in seeds]


def summarize(results: list[EpisodeResult]) -> dict[str, float | int]:
    """Mean / population std / min / max reward and mean steps over ``results``."""
    rewards = [float(r["reward"]) for r in results]
    steps = [int(r["steps"]) for r in results]
    if not rewards:
        return {
            "n": 0, "mean_reward": 0.0, "std_reward": 0.0, "min_reward": 0.0, "max_reward": 0.0,
            "mean_steps": 0.0,
        }  # fmt: skip
    return {
        "n": len(rewards),
        "mean_reward": statistics.fmean(rewards),
        "std_reward": statistics.pstdev(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_steps": statistics.fmean(steps),
    }


__all__ = [
    "BASELINE_FACTORIES",
    "DATA_MANIFEST",
    "REPO_ROOT",
    "EpisodeResult",
    "PolicyFactory",
    "load_split_config",
    "run_baseline",
    "run_split",
    "split_seeds",
    "summarize",
]
