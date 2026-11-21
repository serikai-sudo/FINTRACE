"""Top-level entry points: ``build_clean_episode`` (clean base) and ``build_episode`` (+ injectors).

Determinism key = ENV_VERSION : SCHEMA_VERSION : seed : canonical(config without ``errors``). The
clean base therefore never depends on which errors are injected (clean and injected episodes of
one seed share their books), and every injection runs on its own label-derived sub-RNG.
The public ``episode_id`` is a hash prefix of the generated public CONTENT, never of the seed, and
is recomputed after injections. Generator ids are relabelled at the end from a slightly larger
label range than needed, so clean and injected episodes both contain unused labels and an
injected document cannot be told apart by its id.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
from collections.abc import Sequence
from decimal import Decimal
from types import ModuleType
from typing import Any, Final

from fintrace import __version__
from fintrace.constants import ENV_VERSION, INJECTOR_MAX_RETRY_ATTEMPTS, SCHEMA_VERSION
from fintrace.models import (
    EpisodeConfig,
    InjectorParams,
    IssueType,
    Manifest,
    PrivateEpisode,
    Provenance,
    PublicEpisode,
    Truth,
)

from .benford import derive_sub_rng
from .company import DEFAULT_ROUNDING, generate_company
from .fx import SUPPORTED_CURRENCIES, generate_all_fx_rates
from .history import DecoyGroup, generate_clean_history
from .injectors.base import (
    INJECTOR_ORDER,
    InjectorError,
    InjectorInfeasible,
    append_truth_and_provenance,
    inconsistent_ids,
)
from .validate import ConsistencyError, validate_clean_history

# id-bearing fields per payload kind; every label is "<FAMILY>-<digits>" and families are
# relabelled independently (E, DOC, INV, PAY, J, L)
_ID_FIELDS_BY_KIND: Final[dict[str, tuple[str, ...]]] = {
    "invoices": ("doc_id", "invoice_id", "entity_id"),
    "payments": ("doc_id", "payment_id"),
    "ledger": ("row_id", "journal_id", "entity_id", "invoice_id", "payment_id"),
    "recurring": ("invoice_id",),
    "decoys": ("id",),  # invoice or payment labels owned by a decoy group
}
# unused labels kept per family so that injected records can take an in-range label
_LABEL_RESERVE: Final[dict[str, int]] = {"INV": 16, "DOC": 32, "PAY": 16, "J": 48, "L": 96}
_MODULE_BY_KIND: Final[dict[IssueType, str]] = {
    "duplicate_invoice": "duplicate",
    "payment_allocation": "payment",
    "fx_error": "fx_error",
    "period_misclassification": "period",
}
# interacting scenarios (two linked issues each), applied after the single injectors
_SCENARIO_FUNCS: Final[dict[str, str]] = {
    "scenario_a": "inject_scenario_a",
    "scenario_b": "inject_scenario_b",
}
_SCENARIO_MODULE: Final[str] = "fintrace.generator.injectors.interacting"
_EPISODE_ID_HEX: Final[int] = 12
_SEED_HASH_HEX: Final[int] = 16

Payload = dict[str, list[dict[str, Any]]]
_INJECTOR_MODULES: dict[str, ModuleType] = {}  # lazily imported injector modules


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
    # errors and decoys never enter the key: the clean base of a seed is one and the same
    # whatever is injected or woven in afterwards
    base_config = _canonical_json(config.model_dump(exclude={"errors", "decoys"}))
    return f"{ENV_VERSION}:{SCHEMA_VERSION}:{seed}:{base_config}"


def _key_digest(seed: int, config: EpisodeConfig) -> str:
    return _sha256_hex(_rng_key(seed, config))


def _derive_root_rng(seed: int, config: EpisodeConfig) -> random.Random:
    """Root RNG seeded from sha256 of the determinism key."""
    return random.Random(int(_key_digest(seed, config), 16))


def _derive_injector_rng(
    seed: int, config: EpisodeConfig, kind: str, index: int, attempt: int
) -> random.Random:
    """Independent RNG per injection (issue type or scenario name, ordinal, retry attempt)."""
    label = f"{_key_digest(seed, config)}:inject:{kind}:{index}:{attempt}"
    return random.Random(int(_sha256_hex(label), 16))


def _episode_id(public_content: dict[str, Any]) -> str:
    """``ep_`` + 12 hex of sha256 over the canonical public content (never over the seed)."""
    return f"ep_{_sha256_hex(_canonical_json(public_content))[:_EPISODE_ID_HEX]}"


def _public_content(public: PublicEpisode) -> dict[str, Any]:
    """The public content that the episode id hashes (everything but the manifest's id itself)."""
    manifest = public.manifest
    return {
        "base_currency": manifest.base_currency,
        "period_start": manifest.period_start,
        "period_end": manifest.period_end,
        "timezone": manifest.timezone,
        "policy": public.policy.model_dump(),
        "fx": [fx.model_dump() for fx in public.fx],
        "invoices": [invoice.model_dump() for invoice in public.invoices],
        "payments": [payment.model_dump() for payment in public.payments],
        "ledger": [row.model_dump() for row in public.ledger],
    }


def _check_supported_currencies(config: EpisodeConfig) -> None:
    for code in (config.base_currency, *config.foreign_currencies):
        if code not in SUPPORTED_CURRENCIES:
            supported = ", ".join(SUPPORTED_CURRENCIES)
            raise ValueError(f"unsupported currency {code!r}; supported: {supported}")


def _permute_ids(payload: Payload, rng: random.Random) -> Payload:
    """Relabel each id family with a seeded injection into a slightly larger label range.

    Breaks the sequential-id -> generation-order correlation and leaves a few unused labels per
    family (``_LABEL_RESERVE``) for injected records, so id gaps are normal in every episode.
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
        width = len(labels[0].split("-", 1)[1])
        numbers = rng.sample(range(1, len(labels) + _LABEL_RESERVE.get(family, 0) + 1), len(labels))
        new_labels = (f"{family}-{number:0{width}d}" for number in numbers)
        rename.update(zip(labels, new_labels, strict=True))
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


def _decoy_events(payload: Payload, decoys: Sequence[DecoyGroup]) -> list[dict[str, Any]]:
    """Provenance markers of the legitimate patterns, with every public id they own (permuted).

    ``legitimate_pattern`` is what ``injectors.base.legitimate_pattern_ids`` keys on, so no
    injector ever builds on a decoy document, journal or row.
    """
    invoice_doc = {row["invoice_id"]: row["doc_id"] for row in payload["invoices"]}
    payment_doc = {row["payment_id"]: row["doc_id"] for row in payload["payments"]}
    events: list[dict[str, Any]] = []
    for k, group in enumerate(decoys):
        labels = sorted(row["id"] for row in payload["decoys"] if row["group"] == k)
        invoice_ids = [label for label in labels if label in invoice_doc]
        payment_ids = [label for label in labels if label in payment_doc]
        rows = [
            row
            for row in payload["ledger"]
            if row["invoice_id"] in invoice_ids or row["payment_id"] in payment_ids
        ]
        events.append(
            {
                "event": f"generate_decoy_{group.op}",
                "op": f"generate_decoy_{group.op}",
                "legitimate_pattern": True,
                "invoice_ids": invoice_ids,
                "payment_ids": payment_ids,
                "doc_ids": sorted(
                    [*(invoice_doc[i] for i in invoice_ids), *(payment_doc[p] for p in payment_ids)]
                ),
                "journal_ids": sorted({row["journal_id"] for row in rows}),
                "row_ids": sorted(row["row_id"] for row in rows),
            }
        )
    return events


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
    history = generate_clean_history(company, fx_rates, history_rng, config)

    payload: Payload = {
        "invoices": [invoice.model_dump() for invoice in history.invoices],
        "payments": [payment.model_dump() for payment in history.payments],
        "ledger": [row.model_dump() for row in history.ledger],
        "recurring": [
            {"invoice_id": invoice_id, "series": k}
            for k, series in enumerate(history.recurring_series)
            for invoice_id in series
        ],
        "decoys": [
            {"op": group.op, "group": k, "id": label}
            for k, group in enumerate(history.decoys)
            for label in (*group.invoice_ids, *group.payment_ids)
        ],
    }
    payload = _permute_ids(payload, permute_rng)
    entity_of = {row["invoice_id"]: row["entity_id"] for row in payload["invoices"]}
    recurring_events: list[dict[str, Any]] = []
    for k in range(len(history.recurring_series)):
        ids = sorted(row["invoice_id"] for row in payload["recurring"] if row["series"] == k)
        recurring_events.append(
            {
                "event": "generate_recurring_series",
                "op": "generate_recurring_series",
                "entity_id": entity_of[ids[0]],
                "invoice_ids": ids,
            }
        )
    decoy_events = _decoy_events(payload, history.decoys)

    episode_id = _episode_id(
        {
            "base_currency": config.base_currency,
            "period_start": company.period_start,
            "period_end": company.period_end,
            "timezone": config.timezone,
            "policy": company.policy.model_dump(),
            "fx": [fx.model_dump() for fx in fx_rates],
            "invoices": payload["invoices"],
            "payments": payload["payments"],
            "ledger": payload["ledger"],
        }
    )
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        base_currency=config.base_currency,
        period_start=company.period_start,
        period_end=company.period_end,
        timezone=config.timezone,
        n_ledger_rows=len(history.ledger),
        n_documents=len(history.invoices) + len(history.payments),
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
                "n_invoices": len(history.invoices),
                "n_payments": len(history.payments),
                "n_ledger_rows": len(history.ledger),
                "n_fx_rates": len(fx_rates),
            },
            *recurring_events,
            *decoy_events,
        ),
        injections=(),
    )
    return PrivateEpisode(
        public=public, truth=Truth(episode_id=episode_id, issues=()), provenance=provenance
    )


def _load_injector(kind: IssueType) -> ModuleType:
    """Import an injector module on first use (clean builds never load them)."""
    module = _INJECTOR_MODULES.get(kind)
    if module is None:
        name = f"fintrace.generator.injectors.{_MODULE_BY_KIND[kind]}"
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            raise InjectorError(f"injector for {kind!r} is not implemented yet") from exc
        _INJECTOR_MODULES[kind] = module
    return module


def _inject_with_retries(
    private: PrivateEpisode,
    seed: int,
    config: EpisodeConfig,
    kind: IssueType,
    index: int,
    params: InjectorParams,
) -> PrivateEpisode:
    """Apply one injection, re-drawing its RNG until the structural and independent checks pass.

    The structural check skips only the documents that injected issues deliberately made
    inconsistent (``inconsistent_ids`` over the provenance, the new entry included); every other
    record, the ones earlier injectors created among them, is still held to the clean contract.
    """
    module = _load_injector(kind)
    last_error: Exception | None = None
    for attempt in range(INJECTOR_MAX_RETRY_ATTEMPTS):
        rng = _derive_injector_rng(seed, config, kind, index, attempt)
        try:
            candidate, issue, entry = module.inject(private, rng, params)
            result = append_truth_and_provenance(candidate, issue, entry)
            validate_clean_history(result, ignore_ids=inconsistent_ids(result))
            module.independent_check(result.public, issue)
        except (ConsistencyError, InjectorError) as exc:
            last_error = exc
            continue
        return result
    raise InjectorError(
        f"{kind} #{index} failed after {INJECTOR_MAX_RETRY_ATTEMPTS} attempts"
    ) from last_error


def _inject_scenario_with_retries(
    private: PrivateEpisode, seed: int, config: EpisodeConfig, name: str, index: int
) -> PrivateEpisode:
    """Apply one interacting scenario (two linked issues) with the same retry discipline.

    Both issues are appended, then the books are checked structurally (relaxed only where the
    provenance says so) and each issue is verified by the ``independent_check`` of its own type:
    a scenario issue is rediscoverable exactly like a single injection of that type.
    """
    module = _INJECTOR_MODULES.get("interacting")  # pseudo-kind key for the scenario module
    if module is None:
        module = importlib.import_module(_SCENARIO_MODULE)
        _INJECTOR_MODULES["interacting"] = module
    inject = getattr(module, _SCENARIO_FUNCS[name])
    last_error: Exception | None = None
    for attempt in range(INJECTOR_MAX_RETRY_ATTEMPTS):
        rng = _derive_injector_rng(seed, config, f"interacting_{name}", index, attempt)
        try:
            # the shared injector signature carries params; a scenario's count lives in the
            # config and is driven by this loop, so a unit value is passed
            result, issues, entries = inject(private, rng, InjectorParams(count=1))
            for issue, entry in zip(issues, entries, strict=True):
                result = append_truth_and_provenance(result, issue, entry)
            validate_clean_history(result, ignore_ids=inconsistent_ids(result))
            for issue in issues:
                _load_injector(issue.type).independent_check(result.public, issue)
        except InjectorInfeasible as exc:  # no candidate at all: retries cannot help
            raise InjectorError(f"{name} #{index} is infeasible for this seed: {exc}") from exc
        except (ConsistencyError, InjectorError) as exc:
            last_error = exc
            continue
        return result
    raise InjectorError(
        f"{name} #{index} failed after {INJECTOR_MAX_RETRY_ATTEMPTS} attempts"
    ) from last_error


def _reassign_episode_id(private: PrivateEpisode) -> PrivateEpisode:
    """Recompute the content-hash episode id after the public content changed."""
    episode_id = _episode_id(_public_content(private.public))
    manifest = private.public.manifest.model_copy(update={"episode_id": episode_id})
    public = private.public.model_copy(update={"manifest": manifest})
    truth = private.truth.model_copy(update={"episode_id": episode_id})
    return private.model_copy(update={"public": public, "truth": truth})


def build_episode(seed: int, config: EpisodeConfig | None = None) -> PrivateEpisode:
    """Clean history, then the injectors of ``config.errors`` in fixed order, each checked.

    Order: duplicate_invoice -> payment_allocation -> fx_error -> period_misclassification, then
    the interacting scenarios (A, B). Every injection must keep the books structurally
    consistent (``validate_clean_history``) and be rediscoverable from public data alone
    (``independent_check``); otherwise it is retried on a fresh sub-RNG and finally raises
    ``InjectorError``. With all counts at zero the result is exactly ``build_clean_episode``.
    """
    config = config if config is not None else EpisodeConfig()
    private = build_clean_episode(seed, config)
    if config.errors.total() == 0:
        return private
    for kind in INJECTOR_ORDER:
        params: InjectorParams = getattr(config.errors, kind)
        for index in range(params.count):
            private = _inject_with_retries(private, seed, config, kind, index, params)
    for name in _SCENARIO_FUNCS:
        for index in range(getattr(config.errors.interacting, name)):
            private = _inject_scenario_with_retries(private, seed, config, name, index)
    # every issue must still be rediscoverable on the final books: a later injection is not
    # allowed to mask an earlier one (eligibility rules prevent it; this makes it a contract)
    for issue in private.truth.issues:
        try:
            _load_injector(issue.type).independent_check(private.public, issue)
        except (ConsistencyError, InjectorError) as exc:
            raise InjectorError(
                f"issue {issue.issue_id} ({issue.type}) failed its independent re-check on the "
                f"final books, i.e. a later injection masked it: {exc}"
            ) from exc
    return _reassign_episode_id(private)


__all__ = ["build_clean_episode", "build_episode"]
