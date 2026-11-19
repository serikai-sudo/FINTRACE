"""FX curves (seeded random walk) and the one conversion rule shared by generator and validator.

All supported currencies have 2 minor-unit digits, so minor units convert 1:1 by the rate.
Rates are stored as ``FXRate(source=foreign, target=base)``: 1 foreign = ``rate`` base.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Final, TypeAlias

from fintrace.models import CurrencyCode, FXRate, MinorUnits, Policy

from .benford import derive_sub_rng, normal_like

# Reference rates to USD (1 unit of currency = x USD). Cross rates derive from this table.
USD_RATES: Final[dict[str, Decimal]] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "CHF": Decimal("1.12"),
    "CAD": Decimal("0.73"),
    "AUD": Decimal("0.65"),
}
SUPPORTED_CURRENCIES: Final[tuple[str, ...]] = tuple(USD_RATES)
DAILY_VOLATILITY: Final[Decimal] = Decimal("0.005")  # ~0.5 % daily step

_RATE_QUANTUM: Final[Decimal] = Decimal("0.0001")
_MIN_RATE: Final[Decimal] = Decimal("0.0001")
_ONE: Final[Decimal] = Decimal(1)

# (source, target) -> (sorted ISO dates, rates aligned with the dates)
FxIndex: TypeAlias = dict[tuple[str, str], tuple[list[str], list[Decimal]]]


def cross_rate(source: CurrencyCode, target: CurrencyCode) -> Decimal:
    """Reference cross rate 1 source = x target, 4 places; ValueError for unsupported codes."""
    for code in (source, target):
        if code not in USD_RATES:
            raise ValueError(f"unsupported currency: {code} (supported: {SUPPORTED_CURRENCIES})")
    return (USD_RATES[source] / USD_RATES[target]).quantize(_RATE_QUANTUM)


def _calendar_days(start_date: str, end_date: str) -> list[str]:
    """All ISO dates from start to end inclusive."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


def generate_fx_curve(
    rng: random.Random,
    base_currency: CurrencyCode,
    foreign_currency: CurrencyCode,
    start_date: str,
    end_date: str,
    initial_rate: Decimal,
) -> list[FXRate]:
    """Random-walk curve, one rate per calendar day in [start, end], ~0.5 % daily volatility."""
    rate = initial_rate.quantize(_RATE_QUANTUM)
    curve: list[FXRate] = []
    for day in _calendar_days(start_date, end_date):
        curve.append(
            FXRate(
                date=day,
                source_currency=foreign_currency,
                target_currency=base_currency,
                rate=rate,
            )
        )
        step = _ONE + normal_like(rng) * DAILY_VOLATILITY
        rate = max(_MIN_RATE, (rate * step).quantize(_RATE_QUANTUM))
    return curve


def generate_all_fx_rates(
    rng: random.Random,
    base: CurrencyCode,
    foreigns: tuple[CurrencyCode, ...],
    period_start: str,
    period_end: str,
) -> tuple[FXRate, ...]:
    """One curve per foreign currency (own sub-RNG each), canonical order (date, source, target)."""
    start_date, end_date = period_start[:10], period_end[:10]
    rates: list[FXRate] = []
    for foreign in foreigns:
        sub = derive_sub_rng(rng, f"fx:{foreign}")
        rates.extend(
            generate_fx_curve(sub, base, foreign, start_date, end_date, cross_rate(foreign, base))
        )
    rates.sort(key=lambda r: (r.date, r.source_currency, r.target_currency))
    return tuple(rates)


def build_fx_index(fx_rates: Sequence[FXRate]) -> FxIndex:
    """Index rates by (source, target) for date lookups with bisect."""
    index: FxIndex = {}
    for fx in sorted(fx_rates, key=lambda r: (r.source_currency, r.target_currency, r.date)):
        dates, values = index.setdefault((fx.source_currency, fx.target_currency), ([], []))
        dates.append(fx.date)
        values.append(fx.rate)
    return index


def policy_date(policy: Policy, transaction_timestamp: str, posting_timestamp: str) -> str:
    """Calendar date whose rate applies under ``policy.fx.rate_date_rule``."""
    if policy.fx.rate_date_rule == "transaction_date":
        return transaction_timestamp[:10]
    return posting_timestamp[:10]


def rate_on(
    index: FxIndex, day: str, source: CurrencyCode, target: CurrencyCode, policy: Policy
) -> Decimal:
    """Rate for ``day``; missing days follow ``policy.fx.fallback``; LookupError otherwise."""
    series = index.get((source, target))
    if series is None:
        raise LookupError(f"no FX series {source}->{target}")
    dates, values = series
    i = bisect_left(dates, day)
    if i < len(dates) and dates[i] == day:
        return values[i]
    fallback = policy.fx.fallback
    if fallback == "previous_available_date" and i > 0:
        return values[i - 1]
    if fallback == "next_available_date" and i < len(dates):
        return values[i]
    raise LookupError(f"no FX rate for {source}->{target} on {day} (fallback={fallback})")


def convert_minor(amount_minor: int, rate: Decimal, policy: Policy) -> MinorUnits:
    """Convert minor units by ``rate`` per policy direction, rounded with the policy mode."""
    amount = Decimal(amount_minor)
    converted = amount * rate if policy.fx.rate_direction == "source_to_target" else amount / rate
    return int(converted.quantize(_ONE, rounding=policy.rounding.mode))


__all__ = [
    "DAILY_VOLATILITY",
    "SUPPORTED_CURRENCIES",
    "USD_RATES",
    "FxIndex",
    "build_fx_index",
    "convert_minor",
    "cross_rate",
    "generate_all_fx_rates",
    "generate_fx_curve",
    "policy_date",
    "rate_on",
]
