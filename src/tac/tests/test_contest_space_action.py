from __future__ import annotations

import math

import pytest

from tac.optimization.contest_space_action import (
    CONTEST_RATE_DENOM_BYTES,
    RATE_SCORE_PER_BYTE,
    ContestSpaceActionError,
    build_contest_space_action_functional,
    build_hydration_contract,
    build_rate_distortion_action_row,
    contest_rate_from_archive_bytes,
    contest_score_from_components,
    rate_score_credit,
)


def test_contest_score_formula_matches_canonical_components() -> None:
    archive_bytes = 178_493
    score = contest_score_from_components(
        avg_segnet_dist=0.00055994,
        avg_posenet_dist=0.00002943,
        archive_bytes=archive_bytes,
    )
    expected = (
        100.0 * 0.00055994
        + math.sqrt(10.0 * 0.00002943)
        + 25.0 * archive_bytes / CONTEST_RATE_DENOM_BYTES
    )
    assert score == expected
    assert contest_rate_from_archive_bytes(archive_bytes) == archive_bytes / CONTEST_RATE_DENOM_BYTES


def test_rate_distortion_action_row_computes_break_even_bytes() -> None:
    hydration = build_hydration_contract(
        video_scope="unit_test_video_scope",
        scorer_axis="[macOS-CPU advisory]",
        archive_axis="unit_test_archive",
        runtime_contract="unit_test_receiver",
        sample_count=1,
    )
    row = build_rate_distortion_action_row(
        candidate_id="candidate_a",
        observed_net_delta_score_units=0.000015,
        saved_bytes=10,
        local_cpu_score=0.192,
        local_cpu_avg_segnet_dist=0.0005,
        local_cpu_avg_posenet_dist=0.00003,
        hydration=hydration,
    )

    assert row["acceptance_state"] == "local_gate_failed"
    assert row["rate_score_credit"] == rate_score_credit(10)
    assert row["estimated_distortion_spend_equation"] == (
        "observed_net_delta_score_units + saved_bytes*rate_score_per_byte"
    )
    assert row["estimated_distortion_spend_score_units"] == 0.000015 + 10 * RATE_SCORE_PER_BYTE
    assert row["extra_saved_bytes_to_break_even"] > 0
    assert row["score_claim"] is False


def test_contest_space_action_functional_aggregates_rows_fail_closed() -> None:
    hydration = build_hydration_contract(
        video_scope="unit_test_video_scope",
        scorer_axis="[macOS-CPU advisory]",
        archive_axis="unit_test_archive",
        runtime_contract="unit_test_receiver",
    )
    rows = [
        build_rate_distortion_action_row(
            candidate_id="a",
            observed_net_delta_score_units=0.1,
            saved_bytes=1,
            hydration=hydration,
        ),
        build_rate_distortion_action_row(
            candidate_id="b",
            observed_net_delta_score_units=-0.01,
            saved_bytes=2,
            hydration=hydration,
        ),
    ]
    functional = build_contest_space_action_functional(rows=rows, hydration=hydration)

    assert functional["row_count"] == 2
    assert functional["local_gate_passed_count"] == 1
    assert functional["best_observed_net_delta_score_units"] == -0.01
    assert functional["saved_bytes_total"] == 3
    assert functional["component_terms"]["rate_denominator_bytes"] == CONTEST_RATE_DENOM_BYTES
    assert functional["ready_for_exact_eval_dispatch"] is False


def test_hydration_contract_rejects_ambiguous_scope() -> None:
    with pytest.raises(ContestSpaceActionError):
        build_hydration_contract(
            video_scope="",
            scorer_axis="[macOS-CPU advisory]",
            archive_axis="unit_test_archive",
            runtime_contract="unit_test_receiver",
        )

    with pytest.raises(ContestSpaceActionError):
        build_hydration_contract(
            video_scope="unit_test_video_scope",
            scorer_axis="[macOS-CPU advisory]",
            archive_axis="unit_test_archive",
            runtime_contract="unit_test_receiver",
            sample_count=-1,
        )
