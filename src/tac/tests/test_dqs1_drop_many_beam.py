# SPDX-License-Identifier: MIT
"""Tests for DQS1 drop-many beam planning helper."""

from __future__ import annotations

import pytest

from tac.optimization.dqs1_drop_many_beam import (
    BeamSearchConfig,
    DykstraFeasibilityConfig,
    PairCandidate,
    WaterfillConfig,
    beam_candidate_to_json,
    beam_search_drop_many,
    build_pairwise_interaction_matrix,
    dykstra_alternating_projection_feasibility,
    waterfill_budget_consumed,
)


def _candidate(
    pair_index: int,
    *,
    rate_delta: float = -1.0e-6,
    payload_delta: int = -1,
    budget: float = 1.0e-6,
) -> PairCandidate:
    return PairCandidate(
        pair_index=pair_index,
        rate_score_delta_vs_source_selector=rate_delta,
        predicted_score_mean=0.192,
        payload_bytes_delta_vs_source_selector=payload_delta,
        distortion_repair_budget_score=budget,
    )


def test_beam_search_prefers_synergistic_pair_tuple() -> None:
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    matrix = build_pairwise_interaction_matrix(
        candidates,
        interaction_values={
            "1,3": -5.0e-6,
            "1,2": 2.0e-6,
        },
    )

    rows = beam_search_drop_many(
        candidates,
        matrix,
        config=BeamSearchConfig(width_k=4, depth_d=2, target_depths=(2,)),
        waterfill_config=WaterfillConfig(planning_credit_fraction=0.0),
    )

    assert rows
    assert rows[0].drop_tuple == (1, 3)
    assert rows[0].delta_s_interaction == pytest.approx(-5.0e-6)
    assert rows[0].dykstra_feasible is True
    payload = beam_candidate_to_json(rows[0])
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["axis_decomposition"]["component_response_status"] == (
        "not_measured_component_replay_required_before_budget_spend"
    )


def test_waterfill_budget_is_planning_credit_only() -> None:
    candidates = [_candidate(10, budget=2.0e-6), _candidate(11, budget=3.0e-6)]

    budget = waterfill_budget_consumed(
        (10, 11),
        candidates,
        config=WaterfillConfig(
            segnet_repair_fraction=0.25,
            posenet_repair_fraction=0.25,
            planning_credit_fraction=0.5,
        ),
    )

    assert budget == pytest.approx(1.25e-6)


def test_dykstra_rejects_tuple_without_rate_savings() -> None:
    candidates = [
        _candidate(20, rate_delta=1.0e-6, payload_delta=1, budget=0.0),
        _candidate(21, rate_delta=1.0e-6, payload_delta=1, budget=0.0),
    ]

    assert (
        dykstra_alternating_projection_feasibility(
            (20, 21),
            candidates,
            config=DykstraFeasibilityConfig(rate_min_bytes_saved=1),
        )
        is False
    )


def test_beam_search_filters_to_target_depths() -> None:
    candidates = [_candidate(pair) for pair in range(5)]
    matrix = build_pairwise_interaction_matrix(candidates)

    rows = beam_search_drop_many(
        candidates,
        matrix,
        config=BeamSearchConfig(width_k=8, depth_d=4, target_depths=(3, 4)),
    )

    assert rows
    assert {row.depth for row in rows}.issubset({3, 4})
    assert all(row.delta_s_total < 0.0 for row in rows)

