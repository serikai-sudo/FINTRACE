"""The jury's matcher conventions, one test group per rule the environment had to change for.

Fixture as in the jury text ("Общая фикстура", M01-M12): one issue A of a supported type on the
canonical record R1 with 10000 minor units KZT and one acceptable evidence set {D1, D2}; D3 is an
existing extra document, U does not exist anywhere in the episode; the tolerance is 0.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from pydantic import ValidationError

from fintrace.constants import AMOUNT_TOLERANCE_MINOR, ISSUE_TYPES
from fintrace.evaluator.matcher import match, match_pairs, public_ids_of
from fintrace.evaluator.metrics import compute_metrics
from fintrace.generator.build import build_episode
from fintrace.models import EpisodeConfig, Finding, Issue, JsonObject, Submission
from fintrace.reward.official import official_reward
from fintrace.tools.list_records import fx_id

R1, R2, R3 = "INV-0001", "INV-0002", "INV-0003"
D1, D2, D3, D4 = "DOC-0001", "DOC-0002", "DOC-0003", "DOC-0004"
U = "DOC-9999"  # does not exist
IDS = frozenset({R1, R2, R3, D1, D2, D3, D4})
EID = "ep_0123456789ab"
T, T2 = "duplicate_invoice", "fx_error"


def _issue(issue_id: str = "ISS-0001", **over: object) -> Issue:
    data: dict[str, object] = {
        "issue_id": issue_id,
        "type": T,
        "affected_ids": (R1,),
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "acceptable_evidence_sets": ((D1, D2),),
    }
    data.update(over)
    return Issue.model_validate(data)


def _f(**over: object) -> JsonObject:
    data: JsonObject = {
        "type": T,
        "affected_ids": [R1],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": [D1, D2],
        "explanation": "ok",
    }
    data.update(over)
    return data


def _findings(*raw: JsonObject) -> list[Finding]:
    return Submission.model_validate({"episode_id": EID, "findings": list(raw)}).findings


def _score(
    issues: tuple[Issue, ...], *raw: JsonObject, steps: int = 10
) -> tuple[int, int, int, float]:
    tp, fp, fn = match(_findings(*raw), issues, public_ids=IDS)
    _, _, f1 = compute_metrics(tp, fp, fn)
    return tp, fp, fn, official_reward(f1, steps)


A = _issue()


# ---------------------------------------------------------------- 1. tolerance tau = 0


def test_tau_zero_exact_amount_only() -> None:
    """Jury test 8 / M07: 10001 against 10000 at tau = 0 is a false positive, reward -1."""
    assert _score((A,), _f()) == (1, 0, 0, pytest.approx(0.98))
    assert _score((A,), _f(claimed_amount_minor=10001)) == (0, 1, 1, -1.0)
    assert _score((A,), _f(claimed_amount_minor=9999)) == (0, 1, 1, -1.0)
    assert all(AMOUNT_TOLERANCE_MINOR[t] == 0 for t in ISSUE_TYPES)


def test_published_tau_per_type_is_inclusive_m08(monkeypatch: pytest.MonkeyPatch) -> None:
    """M08: a published tau = 1 accepts 10001 and refuses 10002; other types stay exact."""
    monkeypatch.setitem(AMOUNT_TOLERANCE_MINOR, T, 1)
    assert _score((A,), _f(claimed_amount_minor=10001))[:3] == (1, 0, 0)
    assert _score((A,), _f(claimed_amount_minor=10002))[:3] == (0, 1, 1)
    other = _issue(type=T2, affected_ids=(R2,), acceptable_evidence_sets=((D3, D4),))
    off = _f(type=T2, affected_ids=[R2], claimed_amount_minor=10001, evidence_ids=[D3, D4])
    assert _score((other,), off)[:3] == (0, 1, 1)


# ---------------------------------------------------------------- 2. evidence ids must exist


def test_evidence_must_exist_in_the_public_episode() -> None:
    """M03: an extra existing id is fine; M04: a non-existent id disqualifies the finding."""
    assert _score((A,), _f(evidence_ids=[D2, D1, D3]))[:3] == (1, 0, 0)
    assert _score((A,), _f(evidence_ids=[D1, D2, U]))[:3] == (0, 1, 1)
    assert _score((A,), _f(evidence_ids=[D1, U]))[:3] == (0, 1, 1)


def test_public_ids_cover_every_fetchable_and_visible_identifier() -> None:
    """The existence universe: records, documents, journal and entity ids, synthetic FX ids; and
    every acceptable evidence set of a generated truth lies inside it."""
    private = build_episode(7, EpisodeConfig())
    ids = public_ids_of(private.public)
    row = private.public.ledger[0]
    assert {row.row_id, row.journal_id} <= ids
    assert {private.public.invoices[0].doc_id, private.public.payments[0].payment_id} <= ids
    assert all(fx_id(rate) in ids for rate in private.public.fx)
    assert "DOC-9999" not in ids and "" not in ids
    for issue in private.truth.issues:
        for evidence in issue.acceptable_evidence_sets:
            assert set(evidence) <= ids, (issue.issue_id, evidence)


# ---------------------------------------------------------------- 3. per-finding FP vs structural


def test_wrong_values_are_single_false_positives() -> None:
    """Jury §8 / M06: an unknown type string, a currency differing by case or space, or an id that
    names nothing is ONE false positive; the valid findings beside it are still scored."""
    for bad in (
        _f(type="duplicate_invoices"),
        _f(currency="kzt"),
        _f(currency="KZT "),
        _f(affected_ids=["INV 0001"]),
        _f(affected_ids=["INV-0001".replace("I", chr(0x0406), 1)]),
        _f(claimed_amount_minor=-10000),
    ):
        assert _score((A,), bad) == (0, 1, 1, -1.0)
        tp, fp, fn, reward = _score((A,), _f(), bad)
        assert (tp, fp, fn) == (1, 1, 0) and reward == pytest.approx(float(Fraction(47, 150)))


@pytest.mark.parametrize(
    "raw",
    [
        _f(claimed_amount_minor="10000"),  # M09
        _f(claimed_amount_minor=True),  # M09
        _f(claimed_amount_minor=1.5),
        _f(evidence_ids=[D1] * 10 + [D2]),  # M05: 11 ids before dedup
        _f(explanation="a" * 1001),
        _f(unexpected="field"),
        {k: v for k, v in _f().items() if k != "type"},
        _f(type=5),
        _f(affected_ids="INV-0001"),
    ],
)
def test_structural_errors_void_the_submission(raw: JsonObject) -> None:
    """Jury §7-§8: value types, missing fields and size limits are structural (the whole submit)."""
    with pytest.raises(ValidationError):
        _findings(raw)
    with pytest.raises(ValidationError):
        _findings(_f(), raw)  # one bad finding among good ones still voids the submission


def test_thirty_one_findings_are_structural() -> None:
    assert len(_findings(*([_f()] * 30))) == 30
    with pytest.raises(ValidationError):
        _findings(*([_f()] * 31))


# ---------------------------------------------------------------- 4. deterministic tie-break


def test_tie_break_m10_m11_m12() -> None:
    """M10: F0 fits A and B, F1 fits A only -> F1->A, F0->B. M11: both fit both -> A->F0, B->F1.
    M12: two identical correct findings -> one TP, one FP."""
    b10 = _issue("ISS-0002", acceptable_evidence_sets=((D3, D4),))
    f0 = _f(evidence_ids=[D1, D2, D3, D4])
    f1 = _f()
    pairs = match_pairs(_findings(f0, f1), (A, b10), public_ids=IDS)
    assert pairs == (("ISS-0001", 1), ("ISS-0002", 0))
    assert match(_findings(f0, f1), (A, b10), public_ids=IDS) == (2, 0, 0)
    b11 = _issue("ISS-0002")
    expected = (("ISS-0001", 0), ("ISS-0002", 1))
    assert match_pairs(_findings(f1, f1), (A, b11), public_ids=IDS) == expected
    assert match_pairs(_findings(f1, f1), (b11, A), public_ids=IDS) == expected
    assert match(_findings(f1, f1), (b11, A), public_ids=IDS) == (2, 0, 0)  # truth order irrelevant
    assert match(_findings(f1, f1), (A,), public_ids=IDS) == (1, 1, 0)
    assert match_pairs(_findings(f1, f1), (A,), public_ids=IDS) == (("ISS-0001", 0),)


def test_tie_break_is_lexicographically_smallest_pair_sequence() -> None:
    """Issue ids are ordered by code point, not by truth order; each takes the smallest finding
    index that keeps the matching maximum; permuting the findings never changes the counts."""
    z = _issue("ISS-0009", acceptable_evidence_sets=((D3,),))
    a = _issue("ISS-0001", acceptable_evidence_sets=((D1, D2),))
    both = _f(evidence_ids=[D1, D2, D3])  # fits a and z
    only_a = _f()
    # truth order (z, a); findings (both, only_a): a must take only_a (index 1), z gets both (0)
    pairs = match_pairs(_findings(both, only_a), (z, a), public_ids=IDS)
    assert pairs == (("ISS-0001", 1), ("ISS-0009", 0))
    # with two copies of `both` and one `only_a`, a takes the smallest feasible index (0) and z
    # the other copy: ((A,0),(z,1)) is lexicographically smaller than ((A,2),(z,0))
    pairs = match_pairs(_findings(both, both, only_a), (z, a), public_ids=IDS)
    assert pairs == (("ISS-0001", 0), ("ISS-0009", 1))
    rng = random.Random(3)
    base = [both, only_a, _f(evidence_ids=[D3]), _f(type=T2), _f()]
    reference = match(_findings(*base), (z, a), public_ids=IDS)
    for _ in range(20):
        shuffled = base[:]
        rng.shuffle(shuffled)
        findings = _findings(*shuffled)
        assert match(findings, (z, a), public_ids=IDS) == reference
        pairs = match_pairs(findings, (z, a), public_ids=IDS)
        assert len(pairs) == reference[0]
        assert len({i for _, i in pairs}) == len(pairs)
        assert len({iid for iid, _ in pairs}) == len(pairs)
        assert pairs == tuple(sorted(pairs))


def test_match_pairs_count_equals_match_on_random_graphs() -> None:
    rng = random.Random(11)
    docs = [D1, D2, D3, D4]
    for _ in range(40):
        issues = tuple(
            _issue(f"ISS-{k:04d}", acceptable_evidence_sets=((rng.choice(docs),),))
            for k in range(rng.randint(1, 5))
        )
        raw = [
            _f(evidence_ids=rng.sample(docs, rng.randint(0, 3))) for _ in range(rng.randint(0, 7))
        ]
        findings = _findings(*raw)
        tp, fp, fn = match(findings, issues, public_ids=IDS)
        pairs = match_pairs(findings, issues, public_ids=IDS)
        assert len(pairs) == tp and fp == len(findings) - tp and fn == len(issues) - tp
        by_id = {issue.issue_id: issue for issue in issues}
        for issue_id, index in pairs:
            needed = set(by_id[issue_id].acceptable_evidence_sets[0])
            assert needed <= set(findings[index].evidence_ids)


def test_empty_truth_and_empty_findings_row_3() -> None:
    """Jury Block 3 test 3: TP=FP=FN=0 reports P=R=0 with F1=1, and the reward stays 0.98."""
    tp, fp, fn, reward = _score(())
    assert (tp, fp, fn) == (0, 0, 0) and reward == pytest.approx(0.98, abs=1e-9)
    precision, recall, f1 = compute_metrics(0, 0, 0)
    assert (precision, recall, f1) == (0.0, 0.0, 1.0)
    # row 4 for contrast: an empty truth with one finding is a false positive at -1
    assert _score((), _f())[:3] == (0, 1, 0)
    assert _score((), _f())[3] == -1.0
