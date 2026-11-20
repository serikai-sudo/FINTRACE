"""Attacker 16: list every table twice, then submit nothing."""

from __future__ import annotations

from fintrace.agents.base import Observation, list_action, next_cursor_of
from fintrace.constants import TABLE_NAMES
from fintrace.models import JsonObject

from .base import AttackerBase

ROUNDS = 2


class BudgetBurner(AttackerBase):
    """Spends most of the budget on listings and ends with an empty submission."""

    name = "budget_burner"
    strategy = "list all four tables twice, then submit nothing"
    defence = "steps cost 0.002 each and an empty submission has recall 0: clipped at -1"
    scan = False

    def reset(self) -> None:
        super().reset()
        self._round = 0
        self._table = 0

    def _attack(self, observation: Observation) -> JsonObject:
        if not self._started:
            self._started = True
            return list_action(TABLE_NAMES[0], thought=f"{self.name}: burning budget")
        cursor = next_cursor_of(observation)
        if cursor is None:
            self._table += 1
            if self._table >= len(TABLE_NAMES):
                self._table = 0
                self._round += 1
        if self._round >= ROUNDS:
            return self._submit_empty("budget burnt, submitting nothing")
        table = TABLE_NAMES[self._table]
        return list_action(table, cursor, thought=f"{self.name}: listing {table} again")


__all__ = ["ROUNDS", "BudgetBurner"]
