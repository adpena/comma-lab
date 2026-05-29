# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_autopilot.per_turn_spawn_decision (GAP 3)."""
from __future__ import annotations

import pytest

from tac.cathedral_autopilot.per_turn_spawn_decision import (
    CandidateSelection,
    select_canonical_next_spawn_for_main_thread,
)
from tac.cathedral_autopilot.three_metric_trichotomy import (
    HIGHEST_EV_SHORTEST_WALL_CLOCK,
    HYGIENE_EV,
)


def _make_candidate(
    cid: str,
    *,
    delta: float = -0.1,
    prob: float = 0.5,
    wc: float = 1.0,
    lessons: int = 5,
) -> dict:
    return {
        "candidate_id": cid,
        "predicted_score_delta": delta,
        "probability_materializes": prob,
        "wall_clock_to_validation_hours": wc,
        "hygiene_lessons_honored": lessons,
        "estimated_dispatch_cost_usd": 0.5,
    }


def test_select_basic_proceed_on_empty_in_flight() -> None:
    queue = [_make_candidate("a"), _make_candidate("b")]
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[],
        canonical_queue=queue,
        cap_window_remaining=1,
    )
    assert sel.recommendation == "PROCEED"
    assert sel.selected_candidate_id in ("a", "b")
    assert sel.cap_window_remaining == 0


def test_select_wait_cap_when_cap_zero() -> None:
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[],
        canonical_queue=[_make_candidate("a")],
        cap_window_remaining=0,
    )
    assert sel.recommendation == "WAIT_CAP_EXCEEDED"
    assert sel.selected_candidate_id is None


def test_select_wait_queue_when_all_in_flight() -> None:
    queue = [_make_candidate("a")]
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[{"candidate_id": "a"}],
        canonical_queue=queue,
        cap_window_remaining=1,
    )
    assert sel.recommendation == "WAIT_NO_ELIGIBLE_CANDIDATES"
    assert sel.selected_candidate_id is None


def test_select_excludes_in_flight_from_queue() -> None:
    queue = [_make_candidate("a"), _make_candidate("b")]
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[{"candidate_id": "a"}],
        canonical_queue=queue,
        cap_window_remaining=1,
    )
    assert sel.recommendation == "PROCEED"
    assert sel.selected_candidate_id == "b"


def test_select_picks_highest_ev_shortest_wc_candidate_by_default() -> None:
    """Per operator binding correction 2026-05-28 ~23:40Z."""
    queue = [
        _make_candidate("low_leverage_high_prob", delta=-0.05, prob=0.9, wc=4.0),
        _make_candidate("high_leverage_low_prob", delta=-3.74, prob=0.15, wc=1.5),
    ]
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[],
        canonical_queue=queue,
        cap_window_remaining=1,
    )
    assert sel.recommendation == "PROCEED"
    # high_leverage_low_prob has higher highest_ev_shortest_wc EV
    assert sel.selected_candidate_id == "high_leverage_low_prob"
    assert sel.operator_canonical_metric == HIGHEST_EV_SHORTEST_WALL_CLOCK


def test_select_routing_metric_override() -> None:
    queue = [
        _make_candidate("low_leverage_high_hygiene", delta=-0.05, prob=0.9, wc=4.0, lessons=13),
        _make_candidate("high_leverage_low_hygiene", delta=-3.74, prob=0.15, wc=1.5, lessons=1),
    ]
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[],
        canonical_queue=queue,
        cap_window_remaining=1,
        operator_canonical_metric=HYGIENE_EV,
    )
    assert sel.recommendation == "PROCEED"
    # Override forces routing by HYGIENE_EV
    assert sel.selected_candidate_id == "low_leverage_high_hygiene"


def test_select_refuses_invalid_metric() -> None:
    with pytest.raises(ValueError, match="operator_canonical_metric"):
        select_canonical_next_spawn_for_main_thread(
            in_flight_subagents=[],
            canonical_queue=[],
            operator_canonical_metric="bogus",
        )


def test_select_refuses_negative_cap() -> None:
    with pytest.raises(ValueError, match="cap_window_remaining"):
        select_canonical_next_spawn_for_main_thread(
            in_flight_subagents=[],
            canonical_queue=[],
            cap_window_remaining=-1,
        )


def test_selection_dataclass_refuses_promotable_true() -> None:
    with pytest.raises(ValueError, match="promotable MUST be False"):
        CandidateSelection(
            selected_candidate_id="x",
            operator_canonical_metric=HIGHEST_EV_SHORTEST_WALL_CLOCK,
            cap_window_remaining=0,
            in_flight_sister_count=0,
            rationale="x",
            recommendation="PROCEED",
            promotable=True,
        )


def test_selection_dataclass_refuses_invalid_recommendation() -> None:
    with pytest.raises(ValueError, match="recommendation"):
        CandidateSelection(
            selected_candidate_id="x",
            operator_canonical_metric=HIGHEST_EV_SHORTEST_WALL_CLOCK,
            cap_window_remaining=0,
            in_flight_sister_count=0,
            rationale="x",
            recommendation="BOGUS_RECOMMENDATION",
        )


def test_selection_as_dict_round_trip() -> None:
    sel = CandidateSelection(
        selected_candidate_id="x",
        operator_canonical_metric=HIGHEST_EV_SHORTEST_WALL_CLOCK,
        cap_window_remaining=0,
        in_flight_sister_count=3,
        rationale="x",
        recommendation="PROCEED",
    )
    d = sel.as_dict()
    assert d["selected_candidate_id"] == "x"
    assert d["recommendation"] == "PROCEED"
    assert d["promotable"] is False


def test_select_rationale_mentions_pv_contingency() -> None:
    """Selection rationale must surface that caller must invoke Catalog #378 PV."""
    queue = [_make_candidate("a")]
    sel = select_canonical_next_spawn_for_main_thread(
        in_flight_subagents=[],
        canonical_queue=queue,
        cap_window_remaining=1,
    )
    assert "Catalog #378" in sel.rationale
