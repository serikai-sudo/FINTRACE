"""Benford sampler and statistics: distribution quality, first digit, expected frequencies."""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

from fintrace.constants import BENFORD_CHI_SQUARED_CRITICAL
from fintrace.generator.benford import (
    benford_amount_minor,
    benford_amounts_minor,
    chi_squared_benford,
    derive_sub_rng,
    expected_benford_frequencies,
    first_digit,
)


def test_benford_sampler_distribution() -> None:
    rng = random.Random(2026)
    stratified = benford_amounts_minor(rng, 10_000)
    assert len(stratified) == 10_000 and min(stratified) >= 1
    assert chi_squared_benford(stratified) < Decimal(10)
    iid = [benford_amount_minor(rng) for _ in range(10_000)]
    assert chi_squared_benford(iid) < BENFORD_CHI_SQUARED_CRITICAL  # i.i.d. sampler also Benford
    assert all(isinstance(a, int) and not isinstance(a, bool) for a in stratified[:100])
    # determinism: same seed, same stream; sub-RNGs are label-isolated
    again = benford_amounts_minor(random.Random(2026), 10_000)
    assert again == stratified
    parent_a, parent_b = random.Random(1), random.Random(1)
    assert derive_sub_rng(parent_a, "fx").random() == derive_sub_rng(parent_b, "fx").random()
    assert derive_sub_rng(random.Random(1), "fx").random() != derive_sub_rng(
        random.Random(1), "company"
    ).random()


@pytest.mark.parametrize(
    ("n", "digit"),
    [(0, 0), (1, 1), (9, 9), (10, 1), (999, 9), (1000, 1), (-250, 2), (7_000_000, 7)],
)
def test_first_digit_correctness(n: int, digit: int) -> None:
    assert first_digit(n) == digit


def test_expected_frequencies_sum_to_one() -> None:
    freqs = expected_benford_frequencies()
    assert sorted(freqs) == list(range(1, 10))
    total = sum(freqs.values(), Decimal(0))
    assert abs(total - Decimal(1)) <= Decimal("0.000001")
    for digit, value in freqs.items():
        assert abs(float(value) - math.log10(1 + 1 / digit)) < 1e-6
        assert value == value.quantize(Decimal("0.000001"))  # exactly 6 places
    assert freqs[1] == Decimal("0.301030") and freqs[9] == Decimal("0.045757")
