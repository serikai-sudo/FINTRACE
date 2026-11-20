"""Registry of the 20 reward-hack attackers (differentiator D1). See base.py."""

from __future__ import annotations

from typing import Final

from .amount_bomb import AmountBombAttack
from .base import AttackerBase
from .budget_burner import BudgetBurner
from .calculator_spam import CalculatorSpam
from .cross_episode import CrossEpisodeReplay
from .currency_swap import CurrencySwapAttack
from .duplicate_flood import DuplicateFlood
from .empty_evidence import EmptyEvidenceSet
from .evidenceless import EvidenceLessFlooder
from .explanation_stuffing import ExplanationStuffing
from .id_permutation import IDPermutationAttack
from .near_miss_evidence import NearMissEvidence
from .oversubmit_same_type import OversubmitSameType
from .shotgun import ShotgunAttack
from .submit_at_1 import SubmitAtStep1
from .tolerance_edge import ToleranceEdgeProbe
from .typo_type import TypoInType
from .undersubmit import UndersubmitBoundary
from .unicode_id import UnicodeIDAttack
from .wrong_type import WrongTypeRightIDs
from .zero_amount import ZeroAmountAttack

ALL_ATTACKERS: Final[tuple[type[AttackerBase], ...]] = (
    ShotgunAttack,
    EvidenceLessFlooder,
    ToleranceEdgeProbe,
    CurrencySwapAttack,
    IDPermutationAttack,
    SubmitAtStep1,
    ExplanationStuffing,
    DuplicateFlood,
    WrongTypeRightIDs,
    NearMissEvidence,
    OversubmitSameType,
    UndersubmitBoundary,
    ZeroAmountAttack,
    UnicodeIDAttack,
    EmptyEvidenceSet,
    BudgetBurner,
    CalculatorSpam,
    TypoInType,
    CrossEpisodeReplay,
    AmountBombAttack,
)
ATTACKERS_BY_NAME: Final[dict[str, type[AttackerBase]]] = {cls.name: cls for cls in ALL_ATTACKERS}

__all__ = ["ALL_ATTACKERS", "ATTACKERS_BY_NAME", "AttackerBase"]
