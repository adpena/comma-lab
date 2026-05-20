# SPDX-License-Identifier: MIT
"""Tests for the SLOT MG-1 RiskAdjustedRanker + UncertaintyAwareCandidateRow."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tac.cathedral_consumers.risk_adjusted_ranking_consumer import (
    CANONICAL_EVIDENCE_GRADES,
    CONSUMER_HOOK_NUMBERS,
    CONSUMER_NAME,
    CONSUMER_VERSION,
    RankedCandidate,
    RiskAdjustedRanker,
    UncertaintyAwareCandidateRow,
    consume_candidate,
    update_from_anchor,
)


# Minimal CandidateRow-shaped fixture (duck-typed; avoids importing
# tools/cathedral_autopilot_autonomous_loop.py to keep tests fast).
@dataclass
class _FakeCandidate:
    candidate_id: str
    predicted_score_delta: float
    estimated_dispatch_cost_usd: float = 1.0


def _make_row(
    candidate_id: str,
    delta: float,
    uncertainty: float | None,
    n_anchors: int = 1,
    grade: str = "predicted",
) -> UncertaintyAwareCandidateRow:
    return UncertaintyAwareCandidateRow(
        candidate=_FakeCandidate(candidate_id, delta),
        predicted_delta_uncertainty=uncertainty,
        n_anchors_consumed=n_anchors,
        evidence_grade=grade,
    )


# ---------------------------------------------------------------------------
# UncertaintyAwareCandidateRow invariants
# ---------------------------------------------------------------------------


def test_uncertainty_aware_candidate_row_happy():
    row = _make_row("c1", -0.005, 0.001)
    assert row.candidate_id == "c1"
    assert row.predicted_score_delta == -0.005
    assert row.estimated_dispatch_cost_usd == 1.0
    assert row.predicted_delta_uncertainty == 0.001
    assert row.n_anchors_consumed == 1
    assert row.evidence_grade == "predicted"


def test_row_refuses_missing_candidate_attrs():
    class _Bad:
        candidate_id = "c1"
        # missing predicted_score_delta + estimated_dispatch_cost_usd
    with pytest.raises(ValueError, match="missing required attribute"):
        UncertaintyAwareCandidateRow(
            candidate=_Bad(),
            predicted_delta_uncertainty=0.001,
            n_anchors_consumed=1,
            evidence_grade="predicted",
        )


def test_row_refuses_nan_uncertainty():
    with pytest.raises(ValueError, match="must not be NaN"):
        UncertaintyAwareCandidateRow(
            candidate=_FakeCandidate("c1", -0.005),
            predicted_delta_uncertainty=float("nan"),
            n_anchors_consumed=1,
            evidence_grade="predicted",
        )


def test_row_refuses_negative_uncertainty():
    with pytest.raises(ValueError, match="must be >= 0"):
        UncertaintyAwareCandidateRow(
            candidate=_FakeCandidate("c1", -0.005),
            predicted_delta_uncertainty=-0.001,
            n_anchors_consumed=1,
            evidence_grade="predicted",
        )


def test_row_allows_none_uncertainty():
    row = _make_row("c1", -0.005, None)
    assert row.predicted_delta_uncertainty is None


def test_row_refuses_unknown_grade():
    with pytest.raises(ValueError, match="canonical taxonomy"):
        UncertaintyAwareCandidateRow(
            candidate=_FakeCandidate("c1", -0.005),
            predicted_delta_uncertainty=0.001,
            n_anchors_consumed=1,
            evidence_grade="totally_made_up",
        )


def test_row_carries_last_updated_utc():
    row = _make_row("c1", -0.005, 0.001)
    # Default is now-UTC ISO with Z suffix.
    assert row.last_updated_utc.endswith("Z")
    assert "T" in row.last_updated_utc


def test_canonical_evidence_grades_includes_promotable():
    assert "promotable_exact_contest_cuda" in CANONICAL_EVIDENCE_GRADES
    assert "promotable_exact_contest_cpu" in CANONICAL_EVIDENCE_GRADES
    assert "predicted" in CANONICAL_EVIDENCE_GRADES
    assert "invalid_byte_identity_artifact" in CANONICAL_EVIDENCE_GRADES


# ---------------------------------------------------------------------------
# RiskAdjustedRanker — lambda values
# ---------------------------------------------------------------------------


def test_ranker_lambda_zero_is_race_mode():
    """Lambda=0 should sort purely by posterior mean (existing behavior)."""
    ranker = RiskAdjustedRanker()
    rows = [
        _make_row("low_risk_small_win", -0.001, 0.0001),
        _make_row("high_risk_big_win", -0.010, 0.005),  # noisy but huge win
        _make_row("certain_medium_win", -0.005, 0.0005),
    ]
    ranked = ranker.rank_candidates(rows, lambda_risk_aversion=0.0)
    # Race-mode: pure mean ranking. Most negative first.
    assert ranked[0].candidate.candidate_id == "high_risk_big_win"
    assert ranked[1].candidate.candidate_id == "certain_medium_win"
    assert ranked[2].candidate.candidate_id == "low_risk_small_win"


def test_ranker_lambda_one_is_conservative():
    """Lambda=1 should penalize uncertain candidates."""
    ranker = RiskAdjustedRanker()
    rows = [
        _make_row("low_risk_small_win", -0.001, 0.0001),
        _make_row("high_risk_big_win", -0.010, 0.020),  # uncertainty > mean
        _make_row("certain_medium_win", -0.005, 0.0005),
    ]
    ranked = ranker.rank_candidates(rows, lambda_risk_aversion=1.0)
    # Conservative: certain_medium_win wins because high_risk's risk-adjusted
    # score is -0.010 + 1.0 * 0.020 = +0.010 (now WORSE than zero).
    # certain_medium_win risk-adjusted: -0.005 + 0.0005 = -0.0045.
    # low_risk_small_win risk-adjusted: -0.001 + 0.0001 = -0.0009.
    assert ranked[0].candidate.candidate_id == "certain_medium_win"
    assert ranked[1].candidate.candidate_id == "low_risk_small_win"
    assert ranked[2].candidate.candidate_id == "high_risk_big_win"


def test_ranker_lambda_half_is_intermediate():
    ranker = RiskAdjustedRanker()
    rows = [
        _make_row("a", -0.005, 0.001),  # risk-adj: -0.005 + 0.5*0.001 = -0.0045
        _make_row("b", -0.004, 0.0001),  # risk-adj: -0.004 + 0.5*0.0001 = -0.00395
    ]
    ranked = ranker.rank_candidates(rows, lambda_risk_aversion=0.5)
    # a wins because -0.0045 < -0.00395 (more negative = better).
    assert ranked[0].candidate.candidate_id == "a"


def test_ranker_handles_none_uncertainty_as_pure_prior():
    """A candidate with None uncertainty should be treated as maximally uncertain."""
    ranker = RiskAdjustedRanker()
    rows = [
        _make_row("anchored", -0.005, 0.001),
        _make_row("unanchored", -0.005, None),  # same mean, no uncertainty info
    ]
    # At lambda=0 they tie on mean -> deterministic tie-break by id.
    ranked_zero = ranker.rank_candidates(rows, lambda_risk_aversion=0.0)
    assert ranked_zero[0].candidate.candidate_id == "anchored"  # alphabetical

    # At lambda=1 unanchored gets full pure-prior penalty.
    ranked_one = ranker.rank_candidates(rows, lambda_risk_aversion=1.0)
    assert ranked_one[0].candidate.candidate_id == "anchored"


def test_ranker_empty_input_returns_empty():
    ranker = RiskAdjustedRanker()
    assert ranker.rank_candidates([], lambda_risk_aversion=0.5) == []


def test_ranker_is_deterministic():
    ranker = RiskAdjustedRanker()
    rows = [_make_row(f"c{i}", -0.005, 0.001) for i in range(5)]
    ranked_a = ranker.rank_candidates(rows, lambda_risk_aversion=0.5)
    ranked_b = ranker.rank_candidates(rows, lambda_risk_aversion=0.5)
    assert [r.candidate.candidate_id for r in ranked_a] == [
        r.candidate.candidate_id for r in ranked_b
    ]


def test_ranker_refuses_negative_lambda():
    ranker = RiskAdjustedRanker()
    with pytest.raises(ValueError, match="must be >= 0"):
        ranker.rank_candidates([], lambda_risk_aversion=-0.5)


def test_ranker_refuses_nan_lambda():
    ranker = RiskAdjustedRanker()
    with pytest.raises(ValueError, match="must not be NaN"):
        ranker.rank_candidates([], lambda_risk_aversion=float("nan"))


def test_ranker_refuses_non_uacr_input():
    ranker = RiskAdjustedRanker()
    with pytest.raises(ValueError, match="UncertaintyAwareCandidateRow"):
        ranker.rank_candidates([_FakeCandidate("c1", -0.005)])


def test_ranked_candidate_records_audit_fields():
    ranker = RiskAdjustedRanker()
    rows = [_make_row("c1", -0.005, 0.001)]
    ranked = ranker.rank_candidates(rows, lambda_risk_aversion=0.5)
    rc = ranked[0]
    assert isinstance(rc, RankedCandidate)
    assert rc.posterior_mean == -0.005
    assert rc.posterior_std == 0.001
    assert rc.lambda_risk_aversion == 0.5
    # risk_adjusted_score = -0.005 + 0.5 * 0.001 = -0.0045
    assert abs(rc.risk_adjusted_score - (-0.0045)) < 1e-9


# ---------------------------------------------------------------------------
# Canonical consumer contract per Catalog #335
# ---------------------------------------------------------------------------


def test_module_level_contract_fields():
    """Catalog #335 requires CONSUMER_NAME / VERSION / HOOK_NUMBERS."""
    assert CONSUMER_NAME == "risk_adjusted_ranking_consumer"
    assert isinstance(CONSUMER_VERSION, str)
    assert len(CONSUMER_HOOK_NUMBERS) > 0


def test_consume_candidate_returns_noop_dict():
    result = consume_candidate({"any": "candidate"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert "rationale" in result
    assert len(result["rationale"]) > 4


def test_update_from_anchor_is_noop():
    # Stateless consumer: no exception, no return value contract.
    assert update_from_anchor({"any": "anchor"}) is None


def test_consumer_satisfies_canonical_contract_protocol():
    """Verify auto-discovery via Catalog #335 validates this consumer."""
    from tac.cathedral.consumer_contract import validate_consumer_module
    import tac.cathedral_consumers.risk_adjusted_ranking_consumer as module

    registration = validate_consumer_module(
        module, module_path="tac.cathedral_consumers.risk_adjusted_ranking_consumer"
    )
    assert registration.contract_compliant, (
        f"validation errors: {registration.validation_errors}"
    )
    assert registration.consumer_name == "risk_adjusted_ranking_consumer"
