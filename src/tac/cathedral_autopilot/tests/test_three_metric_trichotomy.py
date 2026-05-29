# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_autopilot.three_metric_trichotomy (Wave N+46 GAP 1)."""
from __future__ import annotations

import math

import pytest

from tac.cathedral_autopilot.three_metric_trichotomy import (
    CandidateWithThreeMetric,
    DEFAULT_OPERATOR_CANONICAL_METRIC,
    FRONTIER_BREAKING_EV,
    HIGHEST_EV_SHORTEST_WALL_CLOCK,
    HYGIENE_EV,
    ThreeMetricTrichotomyRankingResult,
    VALID_CANONICAL_METRICS,
    _compute_frontier_breaking_ev,
    _compute_highest_ev_shortest_wall_clock_ev,
    _compute_hygiene_ev,
    rank_candidates_via_three_metric_trichotomy,
)


# ---------------------------------------------------------------------------
# CandidateWithThreeMetric invariants
# ---------------------------------------------------------------------------


def test_candidate_with_three_metric_construct_happy_path() -> None:
    c = CandidateWithThreeMetric(
        candidate_id="test",
        hygiene_ev=0.5,
        frontier_breaking_ev=0.1,
        highest_ev_shortest_wall_clock_ev=0.05,
        rank_per_canonical_metric=1,
        rationale="happy path",
    )
    assert c.candidate_id == "test"
    assert c.promotable is False
    assert c.axis_tag == "[predicted]"


def test_candidate_with_three_metric_refuses_empty_id() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        CandidateWithThreeMetric(
            candidate_id="",
            hygiene_ev=0.5,
            frontier_breaking_ev=0.1,
            highest_ev_shortest_wall_clock_ev=0.05,
            rank_per_canonical_metric=1,
            rationale="x",
        )


def test_candidate_with_three_metric_refuses_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        CandidateWithThreeMetric(
            candidate_id="t",
            hygiene_ev=float("nan"),
            frontier_breaking_ev=0.0,
            highest_ev_shortest_wall_clock_ev=0.0,
            rank_per_canonical_metric=1,
            rationale="x",
        )


def test_candidate_with_three_metric_refuses_inf() -> None:
    with pytest.raises(ValueError, match="infinite"):
        CandidateWithThreeMetric(
            candidate_id="t",
            hygiene_ev=0.0,
            frontier_breaking_ev=float("inf"),
            highest_ev_shortest_wall_clock_ev=0.0,
            rank_per_canonical_metric=1,
            rationale="x",
        )


def test_candidate_with_three_metric_refuses_promotable_true() -> None:
    with pytest.raises(ValueError, match="promotable MUST be False"):
        CandidateWithThreeMetric(
            candidate_id="t",
            hygiene_ev=0.0,
            frontier_breaking_ev=0.0,
            highest_ev_shortest_wall_clock_ev=0.0,
            rank_per_canonical_metric=1,
            rationale="x",
            promotable=True,
        )


def test_candidate_with_three_metric_refuses_non_predicted_axis_tag() -> None:
    with pytest.raises(ValueError, match=r"\[predicted\]"):
        CandidateWithThreeMetric(
            candidate_id="t",
            hygiene_ev=0.0,
            frontier_breaking_ev=0.0,
            highest_ev_shortest_wall_clock_ev=0.0,
            rank_per_canonical_metric=1,
            rationale="x",
            axis_tag="[contest-CPU]",
        )


def test_candidate_with_three_metric_as_dict_round_trip() -> None:
    c = CandidateWithThreeMetric(
        candidate_id="t",
        hygiene_ev=0.5,
        frontier_breaking_ev=0.1,
        highest_ev_shortest_wall_clock_ev=0.05,
        rank_per_canonical_metric=1,
        rationale="x",
    )
    d = c.as_dict()
    assert d["candidate_id"] == "t"
    assert d["promotable"] is False
    assert d["axis_tag"] == "[predicted]"
    assert "hygiene_ev" in d


# ---------------------------------------------------------------------------
# ThreeMetricTrichotomyRankingResult invariants
# ---------------------------------------------------------------------------


def test_ranking_result_refuses_invalid_metric() -> None:
    with pytest.raises(ValueError, match="operator_canonical_metric"):
        ThreeMetricTrichotomyRankingResult(
            operator_canonical_metric="invalid",
            candidates_with_three_metrics=(),
            per_metric_top_candidate={},
            rationale="x",
        )


def test_ranking_result_refuses_promotable_true() -> None:
    with pytest.raises(ValueError, match="promotable MUST be False"):
        ThreeMetricTrichotomyRankingResult(
            operator_canonical_metric=HIGHEST_EV_SHORTEST_WALL_CLOCK,
            candidates_with_three_metrics=(),
            per_metric_top_candidate={},
            rationale="x",
            promotable=True,
        )


def test_ranking_result_as_dict_round_trip() -> None:
    r = ThreeMetricTrichotomyRankingResult(
        operator_canonical_metric=HIGHEST_EV_SHORTEST_WALL_CLOCK,
        candidates_with_three_metrics=(),
        per_metric_top_candidate={},
        rationale="empty",
    )
    d = r.as_dict()
    assert d["operator_canonical_metric"] == HIGHEST_EV_SHORTEST_WALL_CLOCK
    assert d["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# _compute_* helper unit tests
# ---------------------------------------------------------------------------


def test_compute_hygiene_ev_default_total_is_13() -> None:
    assert _compute_hygiene_ev({"hygiene_lessons_honored": 13}) == 1.0
    assert _compute_hygiene_ev({"hygiene_lessons_honored": 0}) == 0.0
    assert pytest.approx(
        _compute_hygiene_ev({"hygiene_lessons_honored": 6.5})
    ) == 0.5


def test_compute_hygiene_ev_clamps_above_one() -> None:
    # Honored count above total caps at 1.0
    assert _compute_hygiene_ev({"hygiene_lessons_honored": 20}) == 1.0


def test_compute_hygiene_ev_returns_zero_when_no_signal() -> None:
    assert _compute_hygiene_ev({}) == 0.0
    assert _compute_hygiene_ev({"unrelated_field": 99}) == 0.0


def test_compute_frontier_breaking_ev_consumes_predicted_delta_magnitude() -> None:
    # |delta|=0.5, confidence=0.8, cost=0.5 (smoke -> scale 1.0) -> 0.4
    assert pytest.approx(
        _compute_frontier_breaking_ev({
            "predicted_score_delta": -0.5,
            "empirical_confidence": 0.8,
            "estimated_dispatch_cost_usd": 0.5,
        })
    ) == 0.4


def test_compute_frontier_breaking_ev_cost_class_scaling() -> None:
    # Full class (cost=5.0): scale 0.5; long_burn (cost=20): scale 0.1
    full = _compute_frontier_breaking_ev({
        "predicted_score_delta": -1.0,
        "empirical_confidence": 1.0,
        "estimated_dispatch_cost_usd": 5.0,
    })
    long_burn = _compute_frontier_breaking_ev({
        "predicted_score_delta": -1.0,
        "empirical_confidence": 1.0,
        "estimated_dispatch_cost_usd": 20.0,
    })
    assert pytest.approx(full) == 0.5
    assert pytest.approx(long_burn) == 0.1


def test_compute_highest_ev_shortest_wall_clock_operator_canonical_formula() -> None:
    # (|delta|=0.5 * prob=0.4) / wc=2.0 = 0.1
    ev = _compute_highest_ev_shortest_wall_clock_ev({
        "predicted_score_delta": -0.5,
        "probability_materializes": 0.4,
        "wall_clock_to_validation_hours": 2.0,
    })
    assert pytest.approx(ev) == 0.1


def test_compute_highest_ev_estimates_wc_from_cost_when_missing() -> None:
    # Smoke (cost<1.0) -> wc=0.5; |delta|=0.3 * prob=0.6 / 0.5 = 0.36
    ev = _compute_highest_ev_shortest_wall_clock_ev({
        "predicted_score_delta": -0.3,
        "probability_materializes": 0.6,
        "estimated_dispatch_cost_usd": 0.5,
    })
    assert pytest.approx(ev) == 0.36


# ---------------------------------------------------------------------------
# rank_candidates_via_three_metric_trichotomy end-to-end
# ---------------------------------------------------------------------------


def test_rank_empty_candidates() -> None:
    r = rank_candidates_via_three_metric_trichotomy([])
    assert r.operator_canonical_metric == DEFAULT_OPERATOR_CANONICAL_METRIC
    assert r.candidates_with_three_metrics == ()
    assert r.per_metric_top_candidate == {}


def test_rank_routing_default_is_highest_ev_shortest_wc_per_operator() -> None:
    r = rank_candidates_via_three_metric_trichotomy([])
    assert r.operator_canonical_metric == HIGHEST_EV_SHORTEST_WALL_CLOCK


def test_rank_refuses_invalid_metric() -> None:
    with pytest.raises(ValueError, match="operator_canonical_metric"):
        rank_candidates_via_three_metric_trichotomy(
            [], operator_canonical_metric="bogus"
        )


def test_rank_orthogonality_empirically_demonstrated() -> None:
    """Per the canonical 3-correction sequence: 3 distinct tops across 3 metrics."""
    candidates = [
        # High hygiene, low frontier, low highest_ev
        {
            "candidate_id": "hygiene_top",
            "predicted_score_delta": -0.02,
            "probability_materializes": 0.9,
            "wall_clock_to_validation_hours": 4.0,
            "hygiene_lessons_honored": 13,
            "estimated_dispatch_cost_usd": 5.0,
        },
        # Low hygiene, high frontier, high highest_ev (the variance-acceptance candidate)
        {
            "candidate_id": "high_leverage_low_prob",
            "predicted_score_delta": -3.74,
            "probability_materializes": 0.15,
            "wall_clock_to_validation_hours": 1.5,
            "hygiene_lessons_honored": 3,
            "estimated_dispatch_cost_usd": 0.5,
        },
        # Mid hygiene, very high frontier when cost-class differs
        {
            "candidate_id": "frontier_top",
            "predicted_score_delta": -5.0,
            "empirical_confidence": 1.0,
            "probability_materializes": 0.05,
            "wall_clock_to_validation_hours": 20.0,
            "hygiene_lessons_honored": 8,
            "estimated_dispatch_cost_usd": 0.5,
        },
    ]
    r = rank_candidates_via_three_metric_trichotomy(candidates)
    tops = r.per_metric_top_candidate
    # Hygiene top: highest lessons-honored ratio
    assert tops[HYGIENE_EV] == "hygiene_top"
    # Frontier-breaking top: highest |delta| * confidence * cost_scale
    assert tops[FRONTIER_BREAKING_EV] == "frontier_top"
    # Highest-EV-shortest-WC top: variance-acceptance candidate
    assert tops[HIGHEST_EV_SHORTEST_WALL_CLOCK] == "high_leverage_low_prob"
    # Distinct tops across all 3 metrics -> orthogonality
    assert len(set(tops.values())) == 3


def test_rank_skips_invalid_candidates() -> None:
    candidates = [
        "not_a_mapping",
        {},  # missing candidate_id
        {"candidate_id": ""},  # empty
        {"candidate_id": "valid", "predicted_score_delta": -0.1},
    ]
    r = rank_candidates_via_three_metric_trichotomy(candidates)
    assert len(r.candidates_with_three_metrics) == 1
    assert r.candidates_with_three_metrics[0].candidate_id == "valid"


def test_rank_deterministic_tie_break_by_candidate_id() -> None:
    """No-drift invariant: identical EVs sort by candidate_id ascending."""
    candidates = [
        {"candidate_id": "zzz", "predicted_score_delta": -0.5, "probability_materializes": 0.5, "wall_clock_to_validation_hours": 1.0, "estimated_dispatch_cost_usd": 0.5},
        {"candidate_id": "aaa", "predicted_score_delta": -0.5, "probability_materializes": 0.5, "wall_clock_to_validation_hours": 1.0, "estimated_dispatch_cost_usd": 0.5},
        {"candidate_id": "mmm", "predicted_score_delta": -0.5, "probability_materializes": 0.5, "wall_clock_to_validation_hours": 1.0, "estimated_dispatch_cost_usd": 0.5},
    ]
    r = rank_candidates_via_three_metric_trichotomy(candidates)
    cids = [c.candidate_id for c in r.candidates_with_three_metrics]
    assert cids == ["aaa", "mmm", "zzz"]


def test_rank_rationale_surfaces_orthogonality_signal() -> None:
    candidates = [
        {"candidate_id": "a", "predicted_score_delta": -0.1, "probability_materializes": 0.5, "wall_clock_to_validation_hours": 1.0, "hygiene_lessons_honored": 13},
        {"candidate_id": "b", "predicted_score_delta": -3.0, "probability_materializes": 0.1, "wall_clock_to_validation_hours": 1.0, "hygiene_lessons_honored": 1},
    ]
    r = rank_candidates_via_three_metric_trichotomy(candidates)
    assert "orthogonality" in r.rationale.lower()
    assert "highest_ev_shortest_wall_clock" in r.rationale.lower()


def test_rank_ranking_uses_alternate_metric_when_specified() -> None:
    candidates = [
        {"candidate_id": "high_hygiene", "predicted_score_delta": -0.05, "probability_materializes": 0.5, "wall_clock_to_validation_hours": 10.0, "hygiene_lessons_honored": 13},
        {"candidate_id": "high_leverage", "predicted_score_delta": -1.0, "probability_materializes": 0.5, "wall_clock_to_validation_hours": 1.0, "hygiene_lessons_honored": 1},
    ]
    r = rank_candidates_via_three_metric_trichotomy(
        candidates, operator_canonical_metric=HYGIENE_EV
    )
    # When routing default is HYGIENE_EV, high_hygiene tops the ranking
    assert r.candidates_with_three_metrics[0].candidate_id == "high_hygiene"


def test_valid_canonical_metrics_constants() -> None:
    assert HIGHEST_EV_SHORTEST_WALL_CLOCK in VALID_CANONICAL_METRICS
    assert HYGIENE_EV in VALID_CANONICAL_METRICS
    assert FRONTIER_BREAKING_EV in VALID_CANONICAL_METRICS
    assert len(VALID_CANONICAL_METRICS) == 3


def test_default_operator_canonical_metric_is_highest_ev_shortest_wc() -> None:
    """Per operator binding correction 2026-05-28 ~23:40Z."""
    assert DEFAULT_OPERATOR_CANONICAL_METRIC == HIGHEST_EV_SHORTEST_WALL_CLOCK
