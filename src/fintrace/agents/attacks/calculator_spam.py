"""Attacker 17: ninety-nine calculator calls, then an empty submission."""

from __future__ import annotations

from fintrace.agents.base import Observation, calculator_action
from fintrace.constants import MAX_STEPS
from fintrace.models import JsonObject

from .base import AttackerBase

CALLS = MAX_STEPS - 1


class CalculatorSpam(AttackerBase):
    """Tool abuse: arithmetic until the last step, then nothing to show."""

    name = "calculator_spam"
    strategy = "99 calculator calls, then submit nothing"
    defence = "inspection steps cost reward and discover nothing; the floor is -1"
    scan = False

    def reset(self) -> None:
        super().reset()
        self._calls = 0

    def _attack(self, observation: Observation) -> JsonObject:
        if self._calls >= CALLS:
            return self._submit_empty("calculator budget spent")
        self._calls += 1
        left, right = self._rng.randint(1, 99_999), self._rng.randint(1, 999)
        expression = f"{left} * {right} / 100"
        return calculator_action(expression, thought=f"{self.name}: computing {expression}")


__all__ = ["CALLS", "CalculatorSpam"]
