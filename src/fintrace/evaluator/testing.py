"""Oracle policy: the upper-bound baseline, built from the hidden truth.

Lives on the evaluator side on purpose. ``OraclePolicy`` takes a ``Truth`` explicitly and
``oracle_for`` is the one place that reads an environment's private episode; nothing under
``fintrace.agents`` may import this module (``tests/contract/test_runner.py`` scans for it).
The oracle still pays for its evidence: it reads every document of the first acceptable
evidence set before submitting, so its step count is an honest lower bound on the cost of a
perfect audit, not a free lunch.
"""

from __future__ import annotations

from fintrace.agents.base import Observation, document_action, submit_action
from fintrace.environment.env import FintraceEnv
from fintrace.models import JsonObject, Truth

_DOCUMENT_PREFIX = "DOC"


class OraclePolicy:
    """Submits the exact truth after fetching the documents of its first evidence sets."""

    name = "oracle"

    def __init__(self, truth: Truth) -> None:
        self._truth = truth
        self._episode_id = truth.episode_id
        self._findings: list[JsonObject] = []
        self._evidence_docs: list[str] = []
        self._fetched: set[str] = set()
        for issue in truth.issues:
            evidence = issue.acceptable_evidence_sets[0] if issue.acceptable_evidence_sets else ()
            for item in evidence:
                if item.startswith(_DOCUMENT_PREFIX) and item not in self._evidence_docs:
                    self._evidence_docs.append(item)
            self._findings.append(
                {
                    "type": issue.type,
                    "affected_ids": list(issue.affected_ids),
                    "claimed_amount_minor": issue.expected_amount_minor,
                    "currency": issue.currency,
                    "evidence_ids": list(evidence),
                    "explanation": f"Oracle evidence set {list(evidence[:2])}",
                }
            )

    @property
    def evidence_documents(self) -> tuple[str, ...]:
        """Documents the oracle reads before submitting (one step each)."""
        return tuple(self._evidence_docs)

    def reset(self) -> None:
        self._fetched = set()

    def act(self, observation: Observation) -> JsonObject:
        del observation  # the oracle never needs to look at what it reads
        for doc_id in self._evidence_docs:
            if doc_id not in self._fetched:
                self._fetched.add(doc_id)
                return document_action(doc_id, thought=f"oracle fetching evidence {doc_id}")
        return submit_action(
            self._episode_id, self._findings, thought="oracle submitting its prepared findings"
        )


def oracle_for(env: FintraceEnv) -> OraclePolicy:
    """The oracle of the episode currently loaded in ``env`` (test harness / runner only)."""
    private = env._private  # evaluator side: the single sanctioned read of the hidden episode
    if private is None:
        raise RuntimeError("reset the environment before building its oracle")
    return OraclePolicy(private.truth)


__all__ = ["OraclePolicy", "oracle_for"]
