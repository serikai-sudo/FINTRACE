"""Every reward-hack attacker is defeated on the dev split; the base helpers behave."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

import fintrace.environment.env as env_module
from fintrace.agents.attacks import ALL_ATTACKERS, ATTACKERS_BY_NAME, AttackerBase
from fintrace.agents.attacks.base import MAX_SCAN_PAGES
from fintrace.constants import MAX_FINDINGS, MAX_STEPS
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import load_split_config, run_baseline, split_seeds
from fintrace.generator.build import build_episode
from fintrace.models import EpisodeConfig, JsonObject, PrivateEpisode, Submission

THRESHOLD = -0.3
N_SEEDS = 5
MAX_ATTACKER_LINES = 70
ATTACKS_DIR = Path(__file__).resolve().parents[2] / "src" / "fintrace" / "agents" / "attacks"


@cache
def _config() -> EpisodeConfig:
    return load_split_config("dev")


@cache
def _episode(seed: int) -> PrivateEpisode:
    return build_episode(seed, _config())


@pytest.fixture(autouse=True)
def _cached_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build each dev episode once per session: the attackers are cheap, the generator is not."""
    monkeypatch.setattr(env_module, "build_episode", lambda seed, _config=None: _episode(seed))


def _run(attacker: AttackerBase, seed: int) -> tuple[dict[str, object], list[JsonObject]]:
    trace: list[JsonObject] = []
    result = run_baseline(attacker, FintraceEnv(), seed, _config(), trace=trace)
    return result, trace


@pytest.mark.parametrize("attacker_cls", ALL_ATTACKERS, ids=[cls.name for cls in ALL_ATTACKERS])
def test_attacker_scores_negative_on_dev(attacker_cls: type[AttackerBase]) -> None:
    """Mean reward <= -0.3 on the first dev episodes; every episode ends, nothing crashes."""
    rewards: list[float] = []
    for seed in split_seeds("dev")[:N_SEEDS]:
        result, trace = _run(attacker_cls(), seed)
        rewards.append(float(result["reward"]))  # type: ignore[arg-type]
        assert result["terminated"] or result["truncated"], attacker_cls.name
        assert 1 <= int(result["steps"]) <= MAX_STEPS  # type: ignore[call-overload]
        submits = [a for a in trace if a["tool"] == "submit"]
        assert submits
        assert all(len(a["args"]["findings"]) <= MAX_FINDINGS for a in submits)  # type: ignore[index]
        assert all(isinstance(a.get("thought"), str) and a["thought"] for a in trace)
    mean = sum(rewards) / len(rewards)
    assert mean <= THRESHOLD, f"{attacker_cls.__name__}: mean reward {mean:.3f} > {THRESHOLD}"


def test_attackers_deterministic() -> None:
    seed = split_seeds("dev")[0]
    for cls in ALL_ATTACKERS:
        first, trace_a = _run(cls(), seed)
        second, trace_b = _run(cls(), seed)
        assert first["reward"] == second["reward"] and trace_a == trace_b, cls.name


def test_base_extract_episode_id() -> None:
    env = FintraceEnv()
    observation, _ = env.reset(seed=split_seeds("dev")[0], config=_config())
    attacker = ATTACKERS_BY_NAME["shotgun"]()
    assert attacker._extract_episode_id(observation) == env.episode.manifest.episode_id
    later, *_ = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 5}})
    assert attacker._extract_episode_id(later) == env.episode.manifest.episode_id  # sticky
    fresh = ATTACKERS_BY_NAME["shotgun"]()
    assert fresh._extract_episode_id(later) is None  # a list page carries no manifest


def test_base_collect_records() -> None:
    env = FintraceEnv()
    env.reset(seed=split_seeds("dev")[0], config=_config())
    attacker = ATTACKERS_BY_NAME["shotgun"]()
    store: list[JsonObject] = []
    page, *_ = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 7}})
    assert attacker._collect_records(page, store) == 7 and len(store) == 7
    error, *_ = env.step({"tool": "list_records", "args": {"table": "nope"}})
    assert attacker._collect_records(error, store) == 0 and len(store) == 7
    attacker._invoices = store
    pairs = attacker._invoice_pairs()
    for a, b in pairs:
        assert a["entity_id"] == b["entity_id"] and a["total_minor"] == b["total_minor"]
    finding = attacker._pair_finding(pairs[0]) if pairs else attacker._random_pair_finding()
    assert finding is not None
    episode_id = env.episode.manifest.episode_id
    Submission.model_validate({"episode_id": episode_id, "findings": [finding]})


def test_registry_complete() -> None:
    assert len(ALL_ATTACKERS) == 20 and len(ATTACKERS_BY_NAME) == 20
    for cls in ALL_ATTACKERS:
        assert issubclass(cls, AttackerBase) and cls.strategy and cls.defence, cls
    skipped = ("__init__.py", "base.py")
    files = sorted(p for p in ATTACKS_DIR.glob("*.py") if p.name not in skipped)
    assert len(files) == 20
    for path in files:
        assert len(path.read_text(encoding="utf-8").splitlines()) < MAX_ATTACKER_LINES, path.name
    assert MAX_SCAN_PAGES <= 3  # reconnaissance stays cheap
