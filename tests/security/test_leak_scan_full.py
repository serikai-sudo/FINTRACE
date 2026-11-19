"""Full leak scan: 20 injected episodes x ~100 valid steps; no private token ever surfaces.

Scanned per episode, over every observation and ``info`` of the trajectory:
  - forbidden words (truth, provenance, caused_by, generator_version, seed_hash, variant, ...)
  - the literal seed (distinctive 9-digit seeds, so a substring hit cannot be a coincidence)
  - the seed hash and the private company name
  - every issue_id of the episode's truth
  - expected_amount_minor values as whole numbers in the reset surface (manifest, policy,
    budget, info); ledger amounts may coincide with an expected amount legitimately, so the
    later surfaces are not scanned for them
A positive-control test proves the scanners fire. The 20-episode scan is ``slow``.
"""

from __future__ import annotations

import json
import random
import re

import pytest

from fintrace.environment.env import FintraceEnv
from fintrace.environment.observations import ResetObservation
from fintrace.generator.injectors.base import InjectorError
from fintrace.models import (
    EpisodeConfig,
    ErrorInjectionConfig,
    InjectorParams,
    JsonObject,
    ToolResult,
)

FORBIDDEN_LOWERCASE = (
    "truth",
    "provenance",
    "truth.json",
    "provenance.json",
    "caused_by",
    "original_",
    "generator_version",
    "seed_hash",
    "legitimate_pattern",
    "variant",
    "interacts_with",
    "\"seed\"",
)
EPISODES_REQUIRED = 20
STEPS_PER_EPISODE = 99  # plus the reset observation = 100 surfaces
# distinctive 9-digit seeds: a substring hit on the seed itself cannot be a coincidence
CANDIDATE_SEEDS = tuple(918273645 + 1000003 * k for k in range(40))
ID_FIELD = {"invoices": "invoice_id", "payments": "payment_id", "ledger": "row_id"}


def _combined_config() -> EpisodeConfig:
    return EpisodeConfig(
        errors=ErrorInjectionConfig(
            duplicate_invoice=InjectorParams(count=1),
            payment_allocation=InjectorParams(count=1),
            fx_error=InjectorParams(count=1),
            period_misclassification=InjectorParams(count=1),
        )
    )


def _random_valid_action(rng: random.Random, env: FintraceEnv) -> JsonObject:
    """One of the four read tools with arguments an agent could realistically produce."""
    tool = rng.choice(["list_records", "get_record", "get_document", "calculator"])
    public = env.episode
    if tool == "list_records":
        table = rng.choice(["ledger", "invoices", "payments", "fx"])
        return {"tool": tool, "args": {"table": table, "cursor": None, "limit": rng.randint(1, 50)}}
    if tool == "get_record":
        table = rng.choice(list(ID_FIELD))
        records = getattr(public, table)
        record = rng.choice(records)
        return {"tool": tool, "args": {"table": table, "id": getattr(record, ID_FIELD[table])}}
    if tool == "get_document":
        document = rng.choice([*public.invoices, *public.payments])
        return {"tool": tool, "args": {"doc_id": document.doc_id}}
    a, b = rng.randint(1, 10_000), rng.randint(1, 10_000)
    return {"tool": tool, "args": {"expression": f"{a} {rng.choice(['+', '-', '*', '/'])} {b}"}}


def _surface(observation: ToolResult | ResetObservation, info: JsonObject) -> str:
    return json.dumps(observation.model_dump(mode="json")) + json.dumps(info)


def _hits(blob: str, secrets: tuple[str, ...], *, case_insensitive: bool = False) -> list[str]:
    """The secrets that occur in ``blob`` (the scanner itself; see the positive control)."""
    haystack = blob.lower() if case_insensitive else blob
    return [s for s in secrets if (s.lower() if case_insensitive else s) in haystack]


def _number_hits(blob: str, numbers: tuple[int, ...]) -> list[int]:
    """Numbers that occur in ``blob`` as whole JSON numbers or strings, not inside longer digits."""
    return [n for n in numbers if re.search(rf"(?<![0-9]){n}(?![0-9])", blob)]


def test_scanner_positive_control() -> None:
    """The scanners fire: a planted token, seed, issue id and amount are all caught."""
    planted = '{"truth": 1, "x": "ep_1 seed 918273645 ISS-0042", "n_documents": 70323}'
    assert _hits(planted, FORBIDDEN_LOWERCASE, case_insensitive=True) == ["truth"]
    assert _hits(planted, ("918273645", "ISS-0042", "absent")) == ["918273645", "ISS-0042"]
    assert _number_hits(planted, (70323, 323, 170323)) == [70323]


@pytest.mark.slow
def test_leak_scan_20_seeds_100_steps_each() -> None:
    scanned: list[int] = []
    infeasible: list[int] = []
    steps_total = 0
    for seed in CANDIDATE_SEEDS:
        if len(scanned) == EPISODES_REQUIRED:
            break
        env = FintraceEnv()
        try:
            observation, info = env.reset(seed=seed, config=_combined_config())
        except InjectorError:
            infeasible.append(seed)
            continue
        private = env._private  # the test deliberately inspects the evaluator side
        assert private is not None and len(private.truth.issues) == 4
        head = _surface(observation, info)
        surfaces = [head]
        rng = random.Random(seed + 10_000)
        for _ in range(STEPS_PER_EPISODE):
            observation, reward, terminated, truncated, info = env.step(
                _random_valid_action(rng, env)
            )
            assert observation.ok, observation
            assert (reward, terminated, truncated) == (0.0, False, False)
            surfaces.append(_surface(observation, info))
            steps_total += 1
        blob = "\n".join(surfaces)
        assert _hits(blob, FORBIDDEN_LOWERCASE, case_insensitive=True) == [], seed
        company = str(private.provenance.history[0]["company"])
        secrets = (str(seed), private.provenance.seed_hash, company)
        assert _hits(blob, secrets) == [], seed
        issue_ids = tuple(issue.issue_id for issue in private.truth.issues)
        assert _hits(blob, issue_ids) == [], seed
        # the reset surface (manifest, policy, budget, info) must not carry any truth amount;
        # later surfaces list ledger amounts, where a coincidence with an expected amount is
        # legitimate and therefore not scanned
        amounts = tuple(issue.expected_amount_minor for issue in private.truth.issues)
        assert _number_hits(head, amounts) == [], seed
        assert _hits(head, ("expected_amount", "generator_version", "injections")) == []
        scanned.append(seed)
    assert scanned == list(CANDIDATE_SEEDS[:EPISODES_REQUIRED]) and infeasible == []
    assert steps_total == EPISODES_REQUIRED * STEPS_PER_EPISODE
