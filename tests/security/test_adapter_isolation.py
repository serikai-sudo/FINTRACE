"""Through the jury adapter a policy never sees a non-zero closed-mode reward or any closed data."""

from __future__ import annotations

import json
from pathlib import Path

import judge_adapter

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.evaluator.runner import load_split_config
from fintrace.logging.trajectory import FORBIDDEN_TOKENS

DEV_SEED = 10000


class _View:
    """Minimal observation view for the rule-based agent built from the jury envelope."""

    def __init__(self, tool: str, payload: dict[str, object]) -> None:
        self.ok = payload.get("error") is None
        self.tool = tool
        self.result = payload.get("result") if tool != "reset" else payload
        self.steps_used = 100 - int(payload["remaining_steps"])  # type: ignore[call-overload]
        self.steps_remaining = int(payload["remaining_steps"])  # type: ignore[call-overload]


def test_adapter_no_evaluator_leak_to_policy(tmp_path: Path) -> None:
    team = json.loads(load_split_config("dev").model_dump_json())
    judge_adapter.generate_episode(DEV_SEED, {"team_config": team}, str(tmp_path / "ep"))
    env = judge_adapter.make_env()
    first, info = env.reset(0, {"mode": "closed", "episode_dir": str(tmp_path / "ep")})
    agent = RuleBasedReconciliation()
    agent.reset()
    surfaces = [json.dumps(first), json.dumps(info)]
    rewards: list[float] = []
    view = _View("reset", first)
    for _ in range(100):
        action = agent.act(view)
        jury_action = {"name": action["tool"], "args": action["args"]}
        if action["tool"] == "submit":
            jury_action["args"] = {"findings": {"findings": action["args"]["findings"]}}
        payload, reward, terminated, truncated, info = env.step(jury_action)
        rewards.append(reward)
        surfaces.extend((json.dumps(payload), json.dumps(info)))
        if action["tool"] == "list_records" and payload["error"] is None:
            payload["result"]["records_total"] = len(payload["result"]["records"])
        if action["tool"] == "get_record" and payload["error"] is None:
            payload["result"] = {"record": payload["result"]}
        if action["tool"] == "get_document" and payload["error"] is None:
            payload["result"] = {"document": payload["result"]}
        view = _View(str(action["tool"]), payload)
        if terminated or truncated:
            break
    assert terminated and all(r == 0.0 for r in rewards)  # C19 over a whole real investigation
    blob = "\n".join(surfaces).lower()
    for token in (*FORBIDDEN_TOKENS, "iss-", "f1", "precision", "recall", "\"tp\"", "deadline"):
        assert token not in blob, token
    findings = {"findings": action["args"]["findings"]}
    report = judge_adapter.score_episode(str(tmp_path / "ep"), findings, len(rewards) - 1)
    assert report["return"] > 0.5  # the closed scorer, and only it, sees the real quality
