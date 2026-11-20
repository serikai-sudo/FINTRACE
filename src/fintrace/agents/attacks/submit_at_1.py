"""Attacker 6: submit nothing on the very first step."""

from __future__ import annotations

from fintrace.agents.base import Observation
from fintrace.models import JsonObject

from .base import AttackerBase


class SubmitAtStep1(AttackerBase):
    """Bets on clean books: an empty submission at step 1 scores 1.0 there and -1.0 elsewhere."""

    name = "submit_at_1"
    strategy = "empty submission at step 1 (bet on clean books)"
    defence = "every held-out episode carries issues; an empty submission has recall 0"
    scan = False

    def _attack(self, observation: Observation) -> JsonObject:
        return self._submit_empty("betting on clean books at step 1")


__all__ = ["SubmitAtStep1"]
