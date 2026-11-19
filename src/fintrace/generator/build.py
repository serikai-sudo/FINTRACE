"""Top-level entry point: ``build_clean_episode(seed, config) -> PrivateEpisode`` (no issues yet).

Determinism key = ENV_VERSION : SCHEMA_VERSION : seed : canonical(config). It seeds the root
RNG and every subsystem gets a label-derived sub-RNG. The public ``episode_id`` is a hash prefix
of the generated public CONTENT, not of the key: a key hash would let anyone recover the seed
(and regenerate the truth) by brute force at one sha256 per candidate; a content hash costs a
full episode generation per candidate. Sequential generator ids are permuted at the end so that
id order never reveals generation order.
"""

from __future__ import annotations

import hashlib
import json
import random
from decimal import Decimal
from typing import Any, Final

from fintrace import __version__
from fintrace.constants import ENV_VERSION, SCHEMA_VERSION
from fintrace.models import (
    EpisodeConfig,
    Manifest,
    PrivateEpisode,
    Provenance,
    PublicEpisode,
    Truth,
)

from .benford import derive_sub_rng
from .company import DEFAULT_ROUNDING, generate_company
from .fx import SUPPORTED_CURRENCIES, generate_all_fx_rates
from .history import generate_clean_history
from .validate import validate_clean_history

# id-bearing fields per payload kind; every label is "<FAMILY>-<digits>" and families are
# permuted independently (E, DOC, INV, PAY, J, L)
_ID_FIELDS_BY_KIND: Final[dict[str, tuple[str, ...]]] = {
    "invoices": ("doc_id", "invoice_id", "entity_id"),
    "payments": ("doc_id", "payment_id"),
    "ledger": ("row_id", "journal_id", "entity_id", "invoice_id", "payment_id"),
}
_EPISODE_ID_HEX: Final[int] = 12
_SEED_HASH_HEX: Final[int] = 16

Payload = dict[str, list[dict[str, Any]]]


def _json_default(value: object) -> str:
    """Decimal -> normalised plain string ("0.40" and "0.4" hash alike); nothing else allowed."""
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    raise TypeError(f"not JSON serialisable for hashing: {type(value).__name__}")


def _canonical_json(obj: object) -> str:
    """Sorted keys, compact separators, no NaN, normalised Decimals; the only form hashed."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng_key(seed: int, config: EpisodeConfig) -> str:
    return f"{ENV_VERSION}:{SCHEMA_VERSION}:{seed}:{_canonical_json(config.model_dump())}"


def _key_digest(seed: int, config: EpisodeConfig) -> str:
    return _sha256_hex(_rng_key(seed, config))


def _derive_root_rng(seed: int, config: EpisodeConfig) -> random.Random:
    """Root RNG seeded from sha256 of the determinism key."""
    return random.Random(int(_key_digest(seed, config), 16))


def _episode_id(public_content: dict[str, Any]) -> str:
    """``ep_`` + 12 hex of sha256 over the canonical public content (never over the seed)."""
    return f"ep_{_sha256_hex(_canonical_json(public_content))[:_EPISODE_ID_HEX]}"


def _check_supported_currencies(config: EpisodeConfig) -> None:
    for code in (config.base_currency, *config.foreign_currencies):
        if code not in SUPPORTED_CURRENCIES:
            supported = ", ".join(SUPPORTED_CURRENCIES)
            raise ValueError(f"unsupported currency {code!r}; supported: {supported}")


def _permute_ids(payload: Payload, rng: random.Random) -> Payload:
    """Relabel each id family with a seeded bijection over its own label set, then re-sort.

    Breaks the sequential-id -> generation-order correlation while keeping the label sets (and
    therefore the files' shape) unchanged.
    """
    labels_by_family: dict[str, set[str]] = {}  # sets for membership; sorted before use
    for kind, rows in payload.items():
        fields = _ID_FIELDS_BY_KIND[kind]
        for row in rows:
            for field in fields:
                value = row[field]
                if value is not None:
                    labels_by_family.setdefault(value[: value.index("-")], set()).add(value)
            if kind == "payments":
                for ref in row["invoice_refs"]:
                    labels_by_family.setdefault("INV", set()).add(ref["invoice_id"])
    rename: dict[str, str] = {}
    for family in sorted(labels_by_family):
        labels = sorted(labels_by_family[family])
        shuffled = list(labels)
        rng.shuffle(shuffled)
        rename.update(zip(labels, shuffled, strict=True))
    for kind, rows in payload.items():
        fields = _ID_FIELDS_BY_KIND[kind]
        for row in rows:
            for field in fields:
                value = row[field]
                if value is not None:
                    row[field] = rename[value]
            if kind == "payments":
                for ref in row["invoice_refs"]:
                    ref["invoice_id"] = rename[ref["invoice_id"]]
    payload["invoices"].sort(key=lambda r: (r["issued_at"], r["invoice_id"]))
    payload["payments"].sort(key=lambda r: (r["timestamp"], r["payment_id"]))
    payload["ledger"].sort(key=lambda r: (r["timestamp"], r["row_id"]))
    return payload


def build_clean_episode(seed: int, config: EpisodeConfig | None = None) -> PrivateEpisode:
    """Deterministic seed + config -> validated clean PrivateEpisode (empty truth)."""
    config = config if config is not None else EpisodeConfig()
    _check_supported_currencies(config)
    root = _derive_root_rng(seed, config)
    company_rng = derive_sub_rng(root, "company")
    fx_rng = derive_sub_rng(root, "fx")
    history_rng = derive_sub_rng(root, "history")
    permute_rng = derive_sub_rng(root, "permute")

    company = generate_company(company_rng, config)
    fx_rates = generate_all_fx_rates(
        fx_rng,
        config.base_currency,
        config.foreign_currencies,
        company.period_start,
        company.period_end,
    )
    invoices, payments, ledger = generate_clean_history(company, fx_rates, history_rng, config)

    payload: Payload = {
        "invoices": [invoice.model_dump() for invoice in invoices],
        "payments": [payment.model_dump() for payment in payments],
        "ledger": [row.model_dump() for row in ledger],
    }
    payload = _permute_ids(payload, permute_rng)

    episode_id = _episode_id(
        {
            "base_currency": config.base_currency,
            "period_start": company.period_start,
            "period_end": company.period_end,
            "timezone": config.timezone,
            "policy": company.policy.model_dump(),
            "fx": [fx.model_dump() for fx in fx_rates],
            **payload,
        }
    )
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        base_currency=config.base_currency,
        period_start=company.period_start,
        period_end=company.period_end,
        timezone=config.timezone,
        n_ledger_rows=len(ledger),
        n_documents=len(invoices) + len(payments),
        rounding=DEFAULT_ROUNDING,
    )
    public = PublicEpisode.model_validate(
        {
            "manifest": manifest,
            "policy": company.policy,
            "ledger": payload["ledger"],
            "invoices": payload["invoices"],
            "payments": payload["payments"],
            "fx": fx_rates,
        }
    )
    validate_clean_history(public)
    provenance = Provenance(
        generator_version=__version__,
        seed_hash=_key_digest(seed, config)[:_SEED_HASH_HEX],
        history=(
            {
                "event": "generate_clean",
                "company": company.name,
                "n_entities": len(company.entities),
                "n_invoices": len(invoices),
                "n_payments": len(payments),
                "n_ledger_rows": len(ledger),
                "n_fx_rates": len(fx_rates),
            },
        ),
        injections=(),
    )
    return PrivateEpisode(
        public=public, truth=Truth(episode_id=episode_id, issues=()), provenance=provenance
    )


__all__ = ["build_clean_episode"]
