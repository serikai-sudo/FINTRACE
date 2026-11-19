"""Fuzz testing: 10,000 random/malformed actions sent to FintraceEnv (blueprint 6.5).

The env must NEVER raise an uncaught exception, NEVER leak private tokens, and every step must
come back as a ToolResult with a bounded float reward. The 10k run is ``slow``; the envelope test
is fast.
"""

from __future__ import annotations

import json
import random
import string
from collections import Counter

import pytest

from fintrace.constants import ERROR_CODES, TOOL_NAMES
from fintrace.environment.env import FintraceEnv
from fintrace.models import EpisodeConfig, ErrorInjectionConfig, InjectorParams

PRIVATE_TOKENS = (
    "truth",
    "provenance",
    "expected_amount_minor",
    "acceptable_evidence_sets",
    "generator_version",
    "seed_hash",
    "ISS-",
    "caused_by",
    "original",
    "modified",
    "legitimate_pattern",
    "\"seed\"",
)
FUZZ_ITERATIONS = 10_000
RESET_EVERY = 500
CATEGORIES = 11
GREEK = "ΑΒΓΔΕΖΗΘΙΚΛμνξοπ"
# category 10: adversarial scalars and envelopes outside the JSON model
_ADVERSARIAL: tuple[object, ...] = (
    {"tool": "list_records", "args": {"table": "ledger", "limit": float("nan")}},
    {"tool": "list_records", "args": {"table": "ledger", "limit": float("inf")}},
    {"tool": "list_records", "args": {"table": "ledger", "limit": 10**5000}},
    {"tool": "list_records", "args": {"table": "ledger", "limit": True}},
    {"tool": "list_records", "args": {1: "x"}},
    {"tool": "list_records", "args": {"table": b"ledger"}},
    {"tool": "list_records", "args": {"table": "ledger"}, "thought": "x" * 100_000},
    {"tool": "list_records", "args": {"table": "ledger"}, "thought": {"nested": True}},
    {"tool": "calculator", "args": {"expression": "9**9**9**9"}},
    {"tool": "calculator", "args": {"expression": "(" * 150 + "1" + ")" * 150}},
    {"tool": "\ud800list_records", "args": {"table": "ledger"}},
    {"tool": "get_record", "args": {"table": "ledger", "id": "L-\udfff0001"}},
    {"tool": "submit", "args": {"episode_id": "ep_0123456789ab", "findings": [1, 2, 3]}},
    object(),
    (),
    {"tool": ["list_records"], "args": {}},
)


def _combined_config() -> EpisodeConfig:
    """One of each injector: the fuzz runs against a non-empty truth."""
    return EpisodeConfig(
        errors=ErrorInjectionConfig(
            duplicate_invoice=InjectorParams(count=1),
            payment_allocation=InjectorParams(count=1),
            fx_error=InjectorParams(count=1),
            period_misclassification=InjectorParams(count=1),
        )
    )


def _generate_random_action(rng: random.Random) -> tuple[int, object]:
    """(category, action): eleven categories of malformed input, chosen uniformly."""
    kind = rng.randint(0, CATEGORIES - 1)
    if kind == 0:  # unknown tool name
        name = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 20)))
        return kind, {"tool": name, "args": {}}
    if kind == 1:  # valid tool, args missing
        return kind, {"tool": rng.choice(TOOL_NAMES)}
    if kind == 2:  # valid tool, extra field
        args = {"table": "ledger", "cursor": None, "limit": 50, "extra_field": "bomb"}
        return kind, {"tool": "list_records", "args": args}
    if kind == 3:  # deeply nested garbage
        nested: dict[str, object] = {"x": "y"}
        for _ in range(rng.randint(10, 100)):
            nested = {"n": nested}
        return kind, {"tool": "list_records", "args": nested}
    if kind == 4:  # not a dict at all
        return kind, rng.choice([None, [], 42, "string_action", True, 3.14])
    if kind == 5:  # calculator fed random printable and control bytes
        alphabet = string.printable + "\x00\x01\x02"
        expression = "".join(rng.choices(alphabet, k=rng.randint(1, 250)))
        return kind, {"tool": "calculator", "args": {"expression": expression}}
    if kind == 6:  # cursor of wild length
        cursor = "c_" + "9" * rng.randint(1, 5000)
        args = {"table": "ledger", "cursor": cursor, "limit": 50}
        return kind, {"tool": "list_records", "args": args}
    if kind == 7:  # unicode garbage id
        record_id = "".join(rng.choices(GREEK, k=rng.randint(1, 30)))
        return kind, {"tool": "get_record", "args": {"table": "ledger", "id": record_id}}
    if kind == 8:  # structurally invalid submission
        args = {"episode_id": "bogus", "findings": [{"garbage": "field"}]}
        return kind, {"tool": "submit", "args": args}
    if kind == 10:  # adversarial scalars: NaN/inf/huge ints, non-string keys, bytes, surrogates
        return kind, rng.choice(_ADVERSARIAL)
    key = rng.choice(["table", "garbage"])  # random key / value pairs
    value = rng.choice([None, [], {}, "bomb"])
    return kind, {"tool": rng.choice(["list_records", "submit"]), "args": {key: value}}


def _leak_scan(blob: str, where: str) -> None:
    lowered = blob.lower()
    for token in PRIVATE_TOKENS:
        assert token.lower() not in lowered, f"leak of {token!r} at {where}"


def test_leak_scanner_positive_control() -> None:
    """The scanner fires on a planted token (guards against it silently becoming a no-op)."""
    _leak_scan('{"ok": true, "result": {"records": []}}', "clean")  # no raise
    for planted in ('{"truth": 1}', '{"x": "ISS-0042"}', '{"Seed": 42}', '{"PROVENANCE": {}}'):
        with pytest.raises(AssertionError, match="leak of"):
            _leak_scan(planted, "planted")


@pytest.mark.slow
def test_fuzz_10000_random_actions_no_crash_no_leak() -> None:
    """Core fuzz: random actions always come back as a ToolResult and never leak."""
    rng = random.Random(99)
    env = FintraceEnv()
    env.reset(seed=42, config=_combined_config())
    kinds: Counter[int] = Counter()
    codes: Counter[str] = Counter()
    resets = 0
    for i in range(FUZZ_ITERATIONS):
        kind, action = _generate_random_action(rng)
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:  # the contract under test: no exception may escape step()
            pytest.fail(f"env crashed on action #{i} (kind {kind}): {action!r} -> {exc!r}")
        kinds[kind] += 1
        assert hasattr(observation, "ok")
        assert isinstance(reward, float) and -1.0 <= reward <= 1.0
        if observation.ok:
            codes["ok"] += 1
        else:
            assert observation.error is not None and observation.error.code in ERROR_CODES
            codes[observation.error.code] += 1
        _leak_scan(json.dumps(observation.model_dump(mode="json")) + json.dumps(info), f"#{i}")
        if terminated or truncated or i % RESET_EVERY == RESET_EVERY - 1:
            env.reset(seed=42 + (i % 100))  # clean episodes: always feasible
            resets += 1
    assert sum(kinds.values()) == FUZZ_ITERATIONS and set(kinds) == set(range(CATEGORIES))
    assert sum(codes.values()) == FUZZ_ITERATIONS
    assert codes["INVALID_ACTION"] > 0 and codes["CALC_ERROR"] > 0 and codes["INVALID_CURSOR"] > 0
    assert codes["INVALID_TABLE"] > 0 and codes["EPISODE_OVER"] == 0  # always reset after an end
    assert resets >= FUZZ_ITERATIONS // RESET_EVERY


def test_fuzz_action_envelopes_never_crash() -> None:
    """Specific pathological envelopes: every one is rejected with a contract error code."""
    env = FintraceEnv()
    env.reset(seed=1)
    too_many_ids = {
        "type": "duplicate_invoice",
        "affected_ids": ["INV-0001"] * 25,  # exceeds MAX_AFFECTED_IDS_PER_FINDING
        "claimed_amount_minor": 0,
        "currency": "USD",
        "evidence_ids": [],
        "explanation": "",
    }
    pathological: list[object] = [
        {},
        {"tool": None, "args": None},
        {"tool": "list_records"},
        {"args": {"table": "ledger"}},
        {"tool": "list_records", "args": None},
        {"tool": "list_records", "args": {"table": None}},
        {"tool": "list_records", "args": {"table": "ledger", "limit": -1}},
        {"tool": "list_records", "args": {"table": "ledger", "limit": 1_000_000}},
        {"tool": "calculator", "args": {"expression": ""}},
        {"tool": "calculator", "args": {"expression": "1" * 10_000}},
        {"tool": "submit", "args": {"episode_id": "", "findings": []}},
        {
            "tool": "submit",
            "args": {"episode_id": env.episode.manifest.episode_id, "findings": [too_many_ids]},
        },
    ]
    for action in pathological:
        env.reset(seed=1)  # fresh state for each envelope
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:  # the contract under test: no exception may escape step()
            pytest.fail(f"pathological action crashed: {action!r} -> {exc!r}")
        assert observation.ok is False and observation.error is not None
        assert observation.error.code in ERROR_CODES and observation.error.code != "EPISODE_OVER"
        assert (reward, terminated, truncated) == (0.0, False, False)
        _leak_scan(json.dumps(observation.model_dump(mode="json")) + json.dumps(info), repr(action))
