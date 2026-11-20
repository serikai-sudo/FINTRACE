"""Oracle-gap diagnostic: per-error-type reward gap between the oracle and every policy.

The committed dev episodes mix all four error types, two interacting scenarios and decoys, so
they cannot say *which* error type a policy struggles with. This module builds a diagnostic set
of single-error episodes in memory (never committed): for every seed the clean books (dev sizes
and decoys, no injected error) and four variants with exactly one injected issue of one type.
The generator's RNG key excludes the error plan, so the five columns of a seed share the same
base books and differ only in the injected error (a paired design). Every policy then runs
through the standard train-mode runner and the cell value is its mean official reward on that
column; ``oracle gap = mean(oracle) - mean(policy)`` with the truth-reading oracle as the
reference row (gap 0). A pass/fail quiz would give every policy the same gap in every column;
different gaps per column are the differential training signal this diagnostic measures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Final, NamedTuple

from fintrace.agents.attacks import ALL_ATTACKERS
from fintrace.constants import ISSUE_TYPES
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import (
    BASELINE_FACTORIES,
    PolicyFactory,
    load_split_config,
    run_baseline,
)
from fintrace.generator.build import build_episode
from fintrace.generator.injectors.base import InjectorError
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    InjectorParams,
    JsonObject,
    PrivateEpisode,
)

CLEAN_TYPE: Final[str] = "clean"
DIAGNOSTIC_TYPES: Final[tuple[str, ...]] = (CLEAN_TYPE, *ISSUE_TYPES)
BASE_SPLIT: Final[str] = "dev"  # sizes, currencies and decoys of the diagnostic books
DIAGNOSTIC_SEED_BASE: Final[int] = 30000  # disjoint from the train (1000+) and dev (10000+) seeds
DEFAULT_N_PER_TYPE: Final[int] = 3
MAX_CANDIDATE_SEEDS: Final[int] = 60
SKIP_REASON_CHARS: Final[int] = 120
ORACLE: Final[str] = "oracle"
BASELINE_ORDER: Final[tuple[str, ...]] = (ORACLE, "rule_based", "mark_everything", "random")
ORACLE_MIN_REWARD: Final[float] = 0.9  # the reference row must be a near-perfect audit everywhere
ATTACKER_FLOOR_GAP: Final[float] = 1.0  # gap >= 1 <=> the attacker never earns a positive reward


class DiagnosticEpisode(NamedTuple):
    """One column cell of the diagnostic set: the episode and how to rebuild it."""

    error_type: str
    seed: int
    config: EpisodeConfig
    episode: PrivateEpisode


def diagnostic_config(base: EpisodeConfig, error_type: str) -> EpisodeConfig:
    """``base`` with its error plan replaced by one ``error_type`` issue (none for clean)."""
    if error_type == CLEAN_TYPE:
        errors = ErrorInjectionConfig()
    elif error_type in ISSUE_TYPES:
        errors = ErrorInjectionConfig(**{error_type: InjectorParams(count=1)})
    else:
        raise ValueError(f"unknown diagnostic type {error_type!r}; expected {DIAGNOSTIC_TYPES}")
    return EpisodeConfig.model_validate({**base.model_dump(), "errors": errors})


def build_diagnostic_set(
    n_per_type: int = DEFAULT_N_PER_TYPE,
    *,
    base_config: EpisodeConfig | None = None,
    seed_base: int = DIAGNOSTIC_SEED_BASE,
    max_candidates: int = MAX_CANDIDATE_SEEDS,
) -> tuple[list[DiagnosticEpisode], list[JsonObject]]:
    """``n_per_type`` paired seeds x the five columns, plus the seeds skipped as infeasible.

    A seed is kept only when every column builds; an ``InjectorError`` in any column skips the
    whole seed (recorded with its reason), so the columns stay paired. Any other exception is a
    generator bug and propagates. Raises ``RuntimeError`` when the candidate range runs dry.
    """
    if n_per_type < 1:
        raise ValueError("n_per_type must be at least 1")
    base = load_split_config(BASE_SPLIT) if base_config is None else base_config
    configs = {error_type: diagnostic_config(base, error_type) for error_type in DIAGNOSTIC_TYPES}
    episodes: list[DiagnosticEpisode] = []
    skipped: list[JsonObject] = []
    kept = 0
    for seed in range(seed_base, seed_base + max_candidates):
        column: list[DiagnosticEpisode] = []
        try:
            for error_type in DIAGNOSTIC_TYPES:
                config = configs[error_type]
                episode = build_episode(seed, config)
                column.append(DiagnosticEpisode(error_type, seed, config, episode))
        except InjectorError as exc:
            skipped.append({"seed": seed, "reason": str(exc)[:SKIP_REASON_CHARS]})
            continue
        episodes.extend(column)
        kept += 1
        if kept == n_per_type:
            return episodes, skipped
    last = seed_base + max_candidates - 1
    raise RuntimeError(f"only {kept} feasible seeds in {seed_base}..{last} (need {n_per_type})")


def policy_factories(names: Sequence[str] | None = None) -> dict[str, PolicyFactory]:
    """The diagnostic rows in display order: the 4 baselines, then the 20 attackers.

    Both come from the existing registries (``BASELINE_FACTORIES``, ``ALL_ATTACKERS``); every
    factory takes the reset environment, which only the oracle reads (for its truth).
    """
    registry: dict[str, PolicyFactory] = {name: BASELINE_FACTORIES[name] for name in BASELINE_ORDER}
    for cls in ALL_ATTACKERS:
        registry[cls.name] = lambda _env, cls=cls: cls()
    if names is None:
        return registry
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise KeyError(f"unknown policies {unknown}; known: {list(registry)}")
    return {name: registry[name] for name in names}


def compute_oracle_gap(
    diagnostic_set: Sequence[DiagnosticEpisode],
    policies: Mapping[str, PolicyFactory],
    *,
    skipped: Sequence[JsonObject] = (),
) -> JsonObject:
    """Run every policy on every diagnostic episode; return the time-free, JSON-ready matrix.

    Payload: ``types`` and ``policies`` (display order), ``n_per_type``, ``seeds``, ``skipped``,
    ``episodes`` (seed, episode_id, n_issues per column), per-episode ``rewards`` and ``steps``,
    ``mean_reward[policy][type]`` and ``gap[policy][type] = mean(oracle) - mean(policy)``.
    Same set + same policies => identical payload (every policy and the generator are seeded).
    """
    if ORACLE not in policies:
        raise ValueError("the oracle row is the gap reference: include 'oracle' in policies")
    by_type: dict[str, list[DiagnosticEpisode]] = {t: [] for t in DIAGNOSTIC_TYPES}
    for item in diagnostic_set:
        if item.error_type not in by_type:
            raise ValueError(f"unknown diagnostic type {item.error_type!r}")
        by_type[item.error_type].append(item)
    counts = {len(items) for items in by_type.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("every diagnostic type needs the same, non-zero number of episodes")
    env = FintraceEnv()
    rewards: dict[str, dict[str, list[float]]] = {}
    steps: dict[str, dict[str, list[int]]] = {}
    for name, factory in policies.items():
        rewards[name] = {}
        steps[name] = {}
        for error_type in DIAGNOSTIC_TYPES:
            column_rewards: list[float] = []
            column_steps: list[int] = []
            for item in by_type[error_type]:
                # prime the env with the prebuilt episode so the factory (the oracle) sees its
                # truth; run_baseline then takes the real path, env.reset(seed, config), and
                # rebuilds the identical episode (same seed + config => same id, checked below)
                env.reset_loaded(item.episode)
                result = run_baseline(factory(env), env, item.seed, item.config)
                expected = item.episode.public.manifest.episode_id
                if result["episode_id"] != expected:
                    raise RuntimeError(
                        f"{name}/{error_type}: seed {item.seed} rebuilt a different episode"
                    )
                if not (result["terminated"] or result["truncated"]):
                    raise RuntimeError(f"{name}/{error_type}: seed {item.seed} did not end")
                column_rewards.append(float(result["reward"]))
                column_steps.append(int(result["steps"]))
            rewards[name][error_type] = column_rewards
            steps[name][error_type] = column_steps
    mean_reward = {
        name: {t: fmean(rewards[name][t]) for t in DIAGNOSTIC_TYPES} for name in policies
    }
    gap = {
        name: {t: mean_reward[ORACLE][t] - mean_reward[name][t] for t in DIAGNOSTIC_TYPES}
        for name in policies
    }
    return {
        "types": list(DIAGNOSTIC_TYPES),
        "policies": list(policies),
        "n_per_type": counts.pop(),
        "seeds": sorted({item.seed for item in diagnostic_set}),
        "skipped": [dict(entry) for entry in skipped],
        "episodes": {
            t: [
                {
                    "seed": item.seed,
                    "episode_id": item.episode.public.manifest.episode_id,
                    "n_issues": len(item.episode.truth.issues),
                }
                for item in sorted(by_type[t], key=lambda item: item.seed)
            ]
            for t in DIAGNOSTIC_TYPES
        },
        "rewards": rewards,
        "steps": steps,
        "mean_reward": mean_reward,
        "gap": gap,
    }


def differential_spread(results: JsonObject, policy: str) -> float:
    """max - min of ``policy``'s gap over the four error types (0 = flat = pass/fail-like)."""
    gaps = [float(results["gap"][policy][t]) for t in ISSUE_TYPES]
    return max(gaps) - min(gaps)


def attacker_rows(results: JsonObject) -> list[str]:
    """The policies of ``results`` that are not baselines, in display order."""
    return [str(name) for name in results["policies"] if name not in BASELINE_ORDER]


def attacker_floor(results: JsonObject) -> JsonObject:
    """min/max attacker gap over the four error types (``None`` when no attacker ran)."""
    names = attacker_rows(results)
    gaps = [float(results["gap"][name][t]) for name in names for t in ISSUE_TYPES]
    return {
        "n_attackers": len(names),
        "min_gap": min(gaps) if gaps else None,
        "max_gap": max(gaps) if gaps else None,
    }


def check_results(results: JsonObject) -> list[str]:
    """Sanity findings on a matrix (empty = sound): oracle near-perfect, attackers never win."""
    failures: list[str] = []
    for t in DIAGNOSTIC_TYPES:
        oracle_mean = float(results["mean_reward"][ORACLE][t])
        if oracle_mean < ORACLE_MIN_REWARD:
            failures.append(f"oracle mean {oracle_mean:+.3f} on {t} is below {ORACLE_MIN_REWARD}")
    for name in attacker_rows(results):
        for t in ISSUE_TYPES:
            gap = float(results["gap"][name][t])
            if gap < ATTACKER_FLOOR_GAP:
                failures.append(f"{name} gap {gap:.3f} on {t} is below {ATTACKER_FLOOR_GAP}")
        if float(results["gap"][name][CLEAN_TYPE]) < 0.0:
            failures.append(f"{name} beats the oracle on clean books")
    return failures


__all__ = [
    "ATTACKER_FLOOR_GAP",
    "BASELINE_ORDER",
    "BASE_SPLIT",
    "CLEAN_TYPE",
    "DEFAULT_N_PER_TYPE",
    "DIAGNOSTIC_SEED_BASE",
    "DIAGNOSTIC_TYPES",
    "MAX_CANDIDATE_SEEDS",
    "ORACLE",
    "ORACLE_MIN_REWARD",
    "SKIP_REASON_CHARS",
    "DiagnosticEpisode",
    "attacker_floor",
    "attacker_rows",
    "build_diagnostic_set",
    "check_results",
    "compute_oracle_gap",
    "diagnostic_config",
    "differential_spread",
    "policy_factories",
]
