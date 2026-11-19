"""Each consistency check catches its own defect; the orchestrator names the failing check."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fintrace.constants import BENFORD_CHI_SQUARED_CRITICAL
from fintrace.generator.build import build_clean_episode
from fintrace.generator.validate import (
    ConsistencyError,
    check_balanced,
    check_benford,
    check_invoice_totals,
    check_ledger_matches_documents,
    check_payment_allocation,
    check_period,
    validate_clean_history,
)
from fintrace.models import (
    EpisodeConfig,
    InvoiceAllocation,
    LedgerRow,
    Payment,
    PrivateEpisode,
    PublicEpisode,
)

SMALL = EpisodeConfig(n_ledger_rows_range=(200, 300), n_documents_range=(20, 40))


@pytest.fixture(scope="module")
def episode() -> PrivateEpisode:
    return build_clean_episode(seed=7, config=SMALL)


def _with_row(public: PublicEpisode, index: int, **changes: object) -> tuple[LedgerRow, ...]:
    """Ledger with row ``index`` rebuilt (and re-validated) with ``changes``."""
    row = LedgerRow.model_validate({**public.ledger[index].model_dump(), **changes})
    return (*public.ledger[:index], row, *public.ledger[index + 1 :])


def _index_where(public: PublicEpisode, *, credit: bool, with_invoice: bool = False) -> int:
    for i, row in enumerate(public.ledger):
        if (row.credit_minor > 0) == credit and (row.invoice_id is not None or not with_invoice):
            return i
    raise AssertionError("no matching ledger row in fixture")


def test_check_balanced_catches_imbalance(episode: PrivateEpisode) -> None:
    public = episode.public
    check_balanced(public.ledger)  # clean passes
    i = _index_where(public, credit=True)
    with pytest.raises(ConsistencyError) as caught:
        check_balanced(_with_row(public, i, debit_minor=1, credit_minor=0))
    assert caught.value.check_name == "balanced"
    assert public.ledger[i].journal_id in caught.value.items[0]


def test_check_invoice_totals_catches_mismatch(episode: PrivateEpisode) -> None:
    invoices = episode.public.invoices
    check_invoice_totals(invoices)
    invoice = invoices[0]
    bad_total = invoice.model_copy(update={"total_minor": invoice.total_minor + 1})
    with pytest.raises(ConsistencyError) as caught:
        check_invoice_totals((bad_total, *invoices[1:]))
    assert caught.value.check_name == "invoice_totals"
    assert caught.value.items == [invoice.invoice_id]
    line = invoice.line_items[0].model_copy(update={"qty": invoice.line_items[0].qty + 1})
    bad_line = invoice.model_copy(update={"line_items": (line, *invoice.line_items[1:])})
    with pytest.raises(ConsistencyError, match="invoice_totals"):
        check_invoice_totals((bad_line,))


def test_check_payment_allocation_catches_overpay(episode: PrivateEpisode) -> None:
    public = episode.public
    check_payment_allocation(public.payments, public.invoices)
    payment: Payment = public.payments[0]
    invoice = next(i for i in public.invoices if i.invoice_id == payment.invoice_refs[0].invoice_id)
    over = invoice.total_minor + 1
    overpay = payment.model_copy(
        update={
            "amount_minor": over,
            "invoice_refs": (
                InvoiceAllocation(invoice_id=invoice.invoice_id, allocated_minor=over),
            ),
        }
    )
    with pytest.raises(ConsistencyError) as caught:
        check_payment_allocation((overpay, *public.payments[1:]), public.invoices)
    assert caught.value.check_name == "payment_allocation"
    assert "exceed" in caught.value.detail and caught.value.items == [invoice.invoice_id]
    mismatch = payment.model_copy(update={"amount_minor": payment.amount_minor + 1})
    with pytest.raises(ConsistencyError, match="does not match"):
        check_payment_allocation((mismatch,), public.invoices)


def test_check_ledger_matches_documents_catches_orphan(episode: PrivateEpisode) -> None:
    public = episode.public
    base = public.manifest.base_currency
    args = (public.invoices, public.payments, public.policy, public.fx, base)
    check_ledger_matches_documents(public.ledger, *args)
    i = _index_where(public, credit=False, with_invoice=True)
    with pytest.raises(ConsistencyError) as caught:
        check_ledger_matches_documents(_with_row(public, i, invoice_id="INV-ORPHAN"), *args)
    assert caught.value.check_name == "ledger_matches_documents"
    assert "unknown documents" in caught.value.detail
    assert caught.value.items == [public.ledger[i].row_id]
    # dropping a payment's journal entirely is also caught (missing, not orphan)
    without_payment_rows = tuple(r for r in public.ledger if r.payment_id is None)
    with pytest.raises(ConsistencyError, match="payment journal"):
        check_ledger_matches_documents(without_payment_rows, *args)


def test_check_period_catches_out_of_range(episode: PrivateEpisode) -> None:
    public = episode.public
    check_period(public.ledger, public.manifest)
    before = (datetime.fromisoformat(public.manifest.period_start) - timedelta(days=1)).isoformat()
    i = _index_where(public, credit=False)
    with pytest.raises(ConsistencyError) as caught:
        check_period(_with_row(public, i, timestamp=before), public.manifest)
    assert caught.value.check_name == "period"
    assert caught.value.items == [public.ledger[i].row_id]
    at_end = public.manifest.period_end  # exclusive upper bound
    with pytest.raises(ConsistencyError, match="period"):
        check_period(_with_row(public, i, timestamp=at_end), public.manifest)


def test_check_benford_does_not_raise_without_threshold(episode: PrivateEpisode) -> None:
    public = episode.public
    template = public.ledger[_index_where(public, credit=False)]
    skewed = tuple(  # 500 one-row journals, every amount starting with digit 1
        template.model_copy(
            update={
                "debit_minor": 100_000,
                "credit_minor": 0,
                "journal_id": f"J-SKEW{k:04d}",
                "row_id": f"L-SKEW{k:04d}",
            }
        )
        for k in range(500)
    )
    chi = check_benford(skewed, ())
    assert chi > BENFORD_CHI_SQUARED_CRITICAL  # every first digit is 1: badly non-Benford
    with pytest.raises(ConsistencyError) as caught:
        check_benford(skewed, (), threshold=BENFORD_CHI_SQUARED_CRITICAL)
    assert caught.value.check_name == "benford"
    assert check_benford((), ()) == 0  # empty input: nothing to test, no raise


def test_validate_clean_history_orchestrator(episode: PrivateEpisode) -> None:
    validate_clean_history(episode)
    validate_clean_history(episode.public)
    public = episode.public
    before = (datetime.fromisoformat(public.manifest.period_start) - timedelta(days=1)).isoformat()
    # move one whole document-less journal before the period: balance and document checks
    # still hold, so the orchestrator must report the period check
    misc_journal = next(
        r.journal_id for r in public.ledger if r.invoice_id is None and r.payment_id is None
    )
    moved = tuple(
        r.model_copy(update={"timestamp": before}) if r.journal_id == misc_journal else r
        for r in public.ledger
    )
    out_of_period = public.model_copy(update={"ledger": moved})
    with pytest.raises(ConsistencyError) as caught:
        validate_clean_history(out_of_period)
    assert caught.value.check_name == "period"
    j = _index_where(public, credit=True)
    unbalanced = public.model_copy(
        update={"ledger": _with_row(public, j, debit_minor=1, credit_minor=0)}
    )
    with pytest.raises(ConsistencyError) as caught:
        validate_clean_history(unbalanced)
    assert caught.value.check_name == "balanced"
    truncated = public.model_copy(update={"ledger": public.ledger[:-1]})
    with pytest.raises(ConsistencyError) as caught:
        validate_clean_history(truncated)
    assert caught.value.check_name == "manifest"
    duplicate = public.ledger[0].model_copy(update={"row_id": public.ledger[1].row_id})
    duplicated = public.model_copy(update={"ledger": (duplicate, *public.ledger[1:])})
    with pytest.raises(ConsistencyError) as caught:
        validate_clean_history(duplicated)
    assert caught.value.check_name == "unique_ids"
    misc_index = next(
        k for k, r in enumerate(public.ledger) if r.invoice_id is None and r.payment_id is None
    )
    foreign_misc_ledger = _with_row(public, misc_index, currency="EUR")
    with pytest.raises(ConsistencyError) as caught:
        validate_clean_history(public.model_copy(update={"ledger": foreign_misc_ledger}))
    assert caught.value.check_name in ("balanced", "ledger_matches_documents")
