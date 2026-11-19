"""Benford-compliant amount sampling and first-digit statistics. Stdlib only; seeded RNG only.

Magnitudes are handled as integer micro-decades (log10 of the amount times 10**6), so the
sampler's state is integer-only; a float exists solely as a local temporary inside ``10 ** x``,
``math.log10`` and Box-Muller and is converted to ``int`` minor units or ``Decimal`` at once.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from typing import Final

from fintrace.constants import (
    BENFORD_HI_EXPONENT,
    BENFORD_LO_EXPONENT,
    BENFORD_PARETO_ALPHA,
    MINOR_UNIT_DIGITS_DEFAULT,
)
from fintrace.models import MinorUnits

DEFAULT_MEAN_EXPONENT: Final[Decimal] = Decimal("2.5")  # median amount ~ 10**2.5 major units
DEFAULT_SPAN: Final[Decimal] = Decimal("2.0")  # exponent ~ U(mean - span, mean + span)
DEFAULT_PARETO_MIX: Final[Decimal] = Decimal("0.15")  # share of draws that get a Pareto tail

_MICRO: Final[int] = 1_000_000  # micro-decades per decade
_LO_MICRO: Final[int] = round(BENFORD_LO_EXPONENT * _MICRO)
_HI_MICRO: Final[int] = round(BENFORD_HI_EXPONENT * _MICRO)
# log10(d) in micro-decades for d = 1..10: mantissa bin of first digit d is [bounds[d-1], bounds[d])
_DIGIT_BOUNDS_MICRO: Final[tuple[int, ...]] = tuple(
    round(math.log10(d) * _MICRO) for d in range(1, 11)
)
_SIX_PLACES: Final[Decimal] = Decimal("0.000001")
_THREE_PLACES: Final[Decimal] = Decimal("0.001")
# log10(1 + 1/d), quantized to 6 places; the nine values sum to exactly 1.000000.
_BENFORD_FREQUENCIES: Final[dict[int, Decimal]] = {
    digit: Decimal(repr(math.log10(1.0 + 1.0 / digit))).quantize(_SIX_PLACES)
    for digit in range(1, 10)
}


def derive_sub_rng(parent: random.Random, label: str) -> random.Random:
    """Derive an isolated child RNG from 256 parent bits and a label via sha256."""
    material = parent.getrandbits(256).to_bytes(32, "big") + label.encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def normal_like(rng: random.Random) -> Decimal:
    """Standard-normal draw (Box-Muller on two uniforms) returned as a Decimal multiplier."""
    u1 = rng.random()
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)
    return Decimal(repr(z))


def _micro_decades(exponent: Decimal) -> int:
    """Decimal log10 magnitude -> integer micro-decades."""
    return int(exponent * _MICRO)


def _pareto_decades(rng: random.Random, index: int, count: int) -> int:
    """Pareto(alpha) tail as a whole-decade jump (>= 1) at the quantile (index + u) / count.

    Whole decades keep the first digit, so the tail makes upper decades heavier without
    disturbing Benford's law; the fold caps jumps at the top decade.
    """
    quantile = (index + rng.random()) / count
    return max(1, round(-math.log10(1.0 - quantile) / BENFORD_PARETO_ALPHA))


def _apportion_first_digits(n: int) -> list[int]:
    """Largest-remainder apportionment of ``n`` observations to Benford's nine frequencies."""
    quotas = [_BENFORD_FREQUENCIES[d] * n for d in range(1, 10)]
    counts = [int(q) for q in quotas]
    by_remainder = sorted(range(9), key=lambda i: (quotas[i] - counts[i], -i), reverse=True)
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    return counts


def _amount_from_micro(micro: int, minor_unit_digits: int) -> int:
    """Fold by whole decades into [LO, HI) (mantissa preserved) and convert to minor units."""
    while micro < _LO_MICRO:
        micro += _MICRO
    while micro >= _HI_MICRO:
        micro -= _MICRO
    return max(1, round(10.0 ** (micro / _MICRO) * 10**minor_unit_digits))


def benford_amount_minor(
    rng: random.Random,
    minor_unit_digits: int = MINOR_UNIT_DIGITS_DEFAULT,
    mean_exponent: Decimal = DEFAULT_MEAN_EXPONENT,
    span: Decimal = DEFAULT_SPAN,
    pareto_mix: Decimal = DEFAULT_PARETO_MIX,
) -> MinorUnits:
    """Sample one amount in minor units whose first digit follows Benford's Law.

    Base: 10 ** U(mean - span, mean + span); with probability ``pareto_mix`` a Pareto tail
    multiplier is applied; the result is folded by whole decades into the configured range.
    """
    lo, width = _micro_decades(mean_exponent - span), _micro_decades(2 * span)
    micro = lo + rng.randrange(width)
    if rng.random() < float(pareto_mix):
        micro += _pareto_decades(rng, 0, 1) * _MICRO
    return _amount_from_micro(micro, minor_unit_digits)


def benford_amounts_minor(
    rng: random.Random,
    n: int,
    minor_unit_digits: int = MINOR_UNIT_DIGITS_DEFAULT,
    mean_exponent: Decimal = DEFAULT_MEAN_EXPONENT,
    span: Decimal = DEFAULT_SPAN,
    pareto_mix: Decimal = DEFAULT_PARETO_MIX,
) -> list[MinorUnits]:
    """Sample ``n`` amounts whose first digits match Benford's frequencies as closely as integers
    allow (largest-remainder quotas), with the mantissa stratified inside each digit bin and the
    decade cycled evenly over the configured range; the Pareto tail jumps whole decades on evenly
    spaced positions. This removes sampling noise, so the batch itself always passes the
    chi-squared test; derived amounts (splits, repeats) are what adds noise downstream.
    Output order is shuffled.
    """
    if n <= 0:
        return []
    lo_decade = math.floor(float(mean_exponent - span))
    hi_decade = math.ceil(float(mean_exponent + span))
    decades = list(range(lo_decade, max(hi_decade, lo_decade + 1)))
    micros: list[int] = []
    for digit, count in zip(range(1, 10), _apportion_first_digits(n), strict=True):
        lo_m, hi_m = _DIGIT_BOUNDS_MICRO[digit - 1], _DIGIT_BOUNDS_MICRO[digit]
        width = hi_m - lo_m
        offset = rng.randrange(len(decades))
        for i in range(count):
            mantissa = lo_m + (width * i + rng.randrange(width)) // count
            micros.append(decades[(offset + i) % len(decades)] * _MICRO + mantissa)
    n_tail = round(float(pareto_mix) * n)
    if n_tail > 0:
        step = n / n_tail
        for j in range(n_tail):
            micros[int(step * j + step / 2)] += _pareto_decades(rng, j, n_tail) * _MICRO
    rng.shuffle(micros)
    return [_amount_from_micro(m, minor_unit_digits) for m in micros]


def first_digit(n: int) -> int:
    """Leading decimal digit of ``abs(n)``; 0 for 0."""
    return 0 if n == 0 else int(str(abs(n))[0])


def expected_benford_frequencies() -> dict[int, Decimal]:
    """Return {1: 0.301030, 2: 0.176091, ..., 9: 0.045757} (Decimal, 6 places, sums to 1)."""
    return dict(_BENFORD_FREQUENCIES)


def chi_squared_benford(amounts: Sequence[int]) -> Decimal:
    """Pearson chi-squared of first digits against Benford (8 degrees of freedom); zeros skipped."""
    counts = Counter(first_digit(a) for a in amounts if a != 0)
    n = sum(counts.values())
    if n == 0:
        return Decimal(0)
    total = Decimal(0)
    for digit, probability in _BENFORD_FREQUENCIES.items():
        expected = probability * n
        observed = Decimal(counts.get(digit, 0))
        total += (observed - expected) ** 2 / expected
    return total.quantize(_THREE_PLACES)


__all__ = [
    "DEFAULT_MEAN_EXPONENT",
    "DEFAULT_PARETO_MIX",
    "DEFAULT_SPAN",
    "benford_amount_minor",
    "benford_amounts_minor",
    "chi_squared_benford",
    "derive_sub_rng",
    "expected_benford_frequencies",
    "first_digit",
    "normal_like",
]
