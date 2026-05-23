# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.decoder_q_selective_selector_pareto import (
    DecoderQSelectiveSelectorParetoError,
    build_selector_pareto_plan,
)


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _bridge() -> dict[str, object]:
    return {
        "schema": "decoder_q_selective_window_bridge_plan.v1",
        **_false_authority(),
        "candidate_generation_only": True,
        "work_units": [
            {
                **_false_authority(),
                "rank": 1,
                "pair_window": [10, 11],
                "observed_mlx_window_gain": 0.010,
                "observed_mlx_gain": 0.010,
                "normalized_full_video_gain": 0.010 / 600.0,
                "full_video_denominator": 600,
                "unit_id": "u10",
            },
            {
                **_false_authority(),
                "rank": 2,
                "pair_window": [20, 21],
                "observed_mlx_window_gain": 0.005,
                "observed_mlx_gain": 0.005,
                "normalized_full_video_gain": 0.005 / 600.0,
                "full_video_denominator": 600,
                "unit_id": "u20",
            },
            {
                **_false_authority(),
                "rank": 3,
                "pair_window": [30, 31],
                "observed_mlx_window_gain": 0.001,
                "observed_mlx_gain": 0.001,
                "normalized_full_video_gain": 0.001 / 600.0,
                "full_video_denominator": 600,
                "unit_id": "u30",
            },
        ],
    }


def test_selector_pareto_builds_prefix_singleton_and_drop_one_rows() -> None:
    plan = build_selector_pareto_plan(
        _bridge(),
        prefix_ks=[1, 3],
        base_score=1.0,
        reference_score=0.99,
    )

    assert plan["score_claim"] is False
    assert plan["summary"] == {
        "candidate_count": 7,
        "drop_one_candidate_count": 3,
        "pareto_frontier_candidate_count": 3,
        "prefix_candidate_count": 2,
        "recommended_selector_id": "prefix_k003",
        "singleton_candidate_count": 2,
    }
    top = plan["candidates"][0]
    assert top["selector_id"] == "prefix_k003"
    assert top["selector_rank"] == 1
    assert top["pareto_rank"] == 1
    assert top["pareto_frontier"] is True
    assert top["selected_pair_indices"] == [10, 20, 30]
    assert top["payload_bytes"] == 14
    assert top["non_authoritative_mlx_window_gain_sum"] == pytest.approx(0.016)
    assert top["non_authoritative_normalized_full_video_gain_sum"] == pytest.approx(
        0.016 / 600.0
    )
    assert top["exact_cpu_calibrated_estimate"]["predicted_score"] == pytest.approx(0.99)
    assert plan["exact_cpu_calibration"]["score_claim"] is False


def test_selector_pareto_rejects_true_authority_in_work_unit() -> None:
    bridge = _bridge()
    bridge["work_units"][0]["score_claim"] = True  # type: ignore[index]

    with pytest.raises(DecoderQSelectiveSelectorParetoError, match="score_claim"):
        build_selector_pareto_plan(bridge)


def test_selector_pareto_rejects_mismatched_normalized_gain() -> None:
    bridge = _bridge()
    bridge["work_units"][0]["normalized_full_video_gain"] = 0.010  # type: ignore[index]

    with pytest.raises(
        DecoderQSelectiveSelectorParetoError,
        match="normalized_full_video_gain_mismatch",
    ):
        build_selector_pareto_plan(bridge)


def test_selector_pareto_rejects_non_integral_pair_window() -> None:
    bridge = _bridge()
    bridge["work_units"][0]["pair_window"] = [10.9, 11.9]  # type: ignore[index]

    with pytest.raises(DecoderQSelectiveSelectorParetoError, match="integral"):
        build_selector_pareto_plan(bridge)
