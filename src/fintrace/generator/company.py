"""Company profile, counterparties and accounting policy for one episode. Fictional names only."""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Final, Literal

from pydantic import Field

from fintrace.constants import MINOR_UNIT_DIGITS_DEFAULT, TIMEZONE_PATTERN
from fintrace.models import (
    AccountName,
    CurrencyCode,
    EpisodeConfig,
    FrozenModel,
    OpaqueId,
    Policy,
    PolicyFX,
    PolicyPeriod,
    PolicyRounding,
    Rounding,
)

COMPANY_STEMS: Final[tuple[str, ...]] = (
    "Aurora", "Beacon", "Cascade", "Delta", "Ember", "Falcon", "Granite", "Horizon", "Ionic",
    "Juniper", "Keystone", "Lumen", "Meridian", "Nimbus", "Orion", "Pinnacle", "Quartz",
    "Riverstone", "Summit", "Tundra",
)  # fmt: skip
COMPANY_SUFFIXES: Final[tuple[str, ...]] = (
    "Industries", "Logistics", "Holdings", "Trading", "Systems", "Partners",
)  # fmt: skip
ENTITY_STEMS: Final[tuple[str, ...]] = (
    "Nova", "Apex", "Ridge", "Harbor", "Atlas", "Birch", "Cobalt", "Drift", "Echo", "Fjord",
    "Glacier", "Hollow", "Iris", "Jade", "Kestrel", "Lotus", "Maple", "Nordic", "Onyx", "Prairie",
    "Quill", "Raven", "Sable", "Timber", "Umber", "Vale", "Willow", "Xenon", "Yarrow", "Zenith",
    "Amber", "Bramble", "Cinder", "Dune", "Elm", "Fathom", "Garnet", "Heron", "Indigo", "Juno",
    "Kite", "Linden", "Moss", "Nectar", "Opal", "Pebble", "Quarry", "Reef", "Slate", "Thistle",
)  # fmt: skip
ENTITY_SUFFIXES: Final[tuple[str, ...]] = ("LLC", "Inc", "Co", "GmbH", "Ltd", "SA")
LINE_ITEM_DESCRIPTIONS: Final[tuple[str, ...]] = (
    "Consulting services", "Office supplies", "Software licence", "Cloud hosting",
    "Freight and delivery", "Packaging materials", "Maintenance contract", "Marketing campaign",
    "Legal services", "Training workshop", "Raw materials", "Spare parts", "Equipment rental",
    "Travel expenses", "Printing", "Telecom services", "Security services", "Cleaning services",
    "Audit fees", "Insurance premium", "Catering", "Warehouse storage", "Design services",
    "Translation services",
)  # fmt: skip
# public convention: a document whose every line starts with this prefix is a credit note and is
# booked as the reversal of the original invoice (AP Dr / EXPENSE Cr, or REVENUE Dr / AR Cr)
CREDIT_NOTE_PREFIX: Final[str] = "Refund: "
RECURRING_SERVICES: Final[tuple[str, ...]] = (
    "Office rent", "Cloud hosting", "Equipment lease", "Insurance premium", "Security services",
    "Payroll software subscription", "Fleet lease", "Warehouse rent", "Telecom plan",
    "Cleaning contract",
)  # fmt: skip
ACCOUNTS: Final[tuple[AccountName, ...]] = (
    "AP", "AR", "CASH", "REVENUE", "EXPENSE", "FX_GAIN_LOSS",
)  # fmt: skip
PERIOD_YEARS: Final[tuple[int, ...]] = (2024, 2025, 2026)

DEFAULT_POLICY: Final[Policy] = Policy(
    fx=PolicyFX(
        rate_date_rule="transaction_date",
        fallback="previous_available_date",
        rate_direction="source_to_target",
    ),
    period=PolicyPeriod(
        start_inclusive=True, end_exclusive=True, timezone="+05:00", assignment_field="issued_at"
    ),
    rounding=PolicyRounding(
        mode="ROUND_HALF_EVEN",
        minor_unit_digits=MINOR_UNIT_DIGITS_DEFAULT,
        apply_to="converted_total",
    ),
)
DEFAULT_ROUNDING: Final[Rounding] = Rounding(mode="ROUND_HALF_EVEN", stage="per_total")

_FIXED_OFFSET_RE: Final[re.Pattern[str]] = re.compile(r"^([+-])([0-9]{2}):([0-9]{2})$")

EntityKind = Literal["customer", "supplier"]


class Entity(FrozenModel):
    """A counterparty: customer (we invoice them) or supplier (they invoice us)."""

    entity_id: OpaqueId
    name: str
    kind: EntityKind
    default_currency: CurrencyCode


class CompanyProfile(FrozenModel):
    """The audited company for one episode: identity, period, chart of accounts, counterparties."""

    company_id: OpaqueId
    name: str
    base_currency: CurrencyCode
    period_start: str
    period_end: str
    timezone: str = Field(pattern=TIMEZONE_PATTERN)
    accounts: tuple[AccountName, ...] = ACCOUNTS
    entities: tuple[Entity, ...]
    policy: Policy = DEFAULT_POLICY


def fixed_offset(tz: str) -> timezone:
    """Parse ``±HH:MM`` into a tzinfo. IANA names are not supported by the generator (v1)."""
    match = _FIXED_OFFSET_RE.fullmatch(tz)
    if match is None:
        raise ValueError(f"generator supports fixed-offset timezones only, got {tz!r}")
    sign = 1 if match.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))


def generate_period(rng: random.Random, config: EpisodeConfig) -> tuple[str, str]:
    """First day of a seeded month in PERIOD_YEARS plus ``period_days``; ISO with offset."""
    tz = fixed_offset(config.timezone)
    start = datetime(rng.choice(PERIOD_YEARS), rng.randrange(1, 13), 1, tzinfo=tz)
    end = start + timedelta(days=config.period_days)
    return start.isoformat(), end.isoformat()


def generate_entities(
    rng: random.Random,
    n: int,
    base_currency: CurrencyCode,
    foreign_currencies: tuple[CurrencyCode, ...],
    foreign_rate: Decimal,
) -> tuple[Entity, ...]:
    """Seeded counterparties: unique names, both kinds, >= 1 foreign-currency entity."""
    if not 1 <= n <= len(ENTITY_STEMS):
        raise ValueError(f"n must be in [1, {len(ENTITY_STEMS)}], got {n}")
    stems = rng.sample(ENTITY_STEMS, k=n)
    kinds: list[EntityKind] = ["supplier" if i % 2 == 0 else "customer" for i in range(n)]
    rng.shuffle(kinds)
    currencies: list[CurrencyCode] = []
    for _ in range(n):
        foreign = bool(foreign_currencies) and rng.random() < float(foreign_rate)
        currencies.append(rng.choice(foreign_currencies) if foreign else base_currency)
    if foreign_currencies and all(c == base_currency for c in currencies):
        currencies[rng.randrange(n)] = rng.choice(foreign_currencies)
    return tuple(
        Entity(
            entity_id=f"E-{i + 1:04d}",
            name=f"{stem} {rng.choice(ENTITY_SUFFIXES)}",
            kind=kind,
            default_currency=currency,
        )
        for i, (stem, kind, currency) in enumerate(zip(stems, kinds, currencies, strict=True))
    )


def generate_company(rng: random.Random, config: EpisodeConfig) -> CompanyProfile:
    """Seeded company: name, period, counterparties, default policy (timezone from config)."""
    name = f"{rng.choice(COMPANY_STEMS)} {rng.choice(COMPANY_SUFFIXES)}"
    period_start, period_end = generate_period(rng, config)
    n_entities = rng.randint(*config.n_entities_range)
    entities = generate_entities(
        rng,
        n_entities,
        config.base_currency,
        config.foreign_currencies,
        config.foreign_invoice_rate,
    )
    policy = DEFAULT_POLICY.model_copy(
        update={"period": DEFAULT_POLICY.period.model_copy(update={"timezone": config.timezone})}
    )
    return CompanyProfile(
        company_id="CO-0001",
        name=name,
        base_currency=config.base_currency,
        period_start=period_start,
        period_end=period_end,
        timezone=config.timezone,
        entities=entities,
        policy=policy,
    )


__all__ = [
    "ACCOUNTS",
    "COMPANY_STEMS",
    "COMPANY_SUFFIXES",
    "CREDIT_NOTE_PREFIX",
    "DEFAULT_POLICY",
    "DEFAULT_ROUNDING",
    "ENTITY_STEMS",
    "ENTITY_SUFFIXES",
    "LINE_ITEM_DESCRIPTIONS",
    "PERIOD_YEARS",
    "RECURRING_SERVICES",
    "CompanyProfile",
    "Entity",
    "EntityKind",
    "fixed_offset",
    "generate_company",
    "generate_entities",
    "generate_period",
]
