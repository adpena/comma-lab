# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
    build_modelsize_budget_plan,
)


def test_modelsize_budget_plan_selects_measured_total_score_minimum() -> None:
    rows = [
        {"row_id": "tiny", "archive_bytes": 20_000, "nonrate_score": 0.240},
        {"row_id": "small", "archive_bytes": 40_000, "nonrate_score": 0.205},
        {"row_id": "medium", "archive_bytes": 80_000, "nonrate_score": 0.200},
    ]

    plan = build_modelsize_budget_plan(rows, carrier_id="hi_nerv")

    assert plan["status"] == "measured_modelsize_budget_selected"
    assert plan["selected_point"]["row_id"] == "small"
    assert plan["selected_archive_bytes"] == 40_000
    first, second = plan["marginal_steps"]
    assert first["spend_rule"] == "spend_modelsize_byte"
    assert first["marginal_improvement_per_byte"] > CONTEST_BYTE_PRICE_SCORE
    assert second["spend_rule"] == "stop_or_reallocate_modelsize_byte"
    assert second["marginal_improvement_per_byte"] < CONTEST_BYTE_PRICE_SCORE
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_modelsize_budget_plan_can_extract_nonrate_from_component_distortions() -> None:
    plan = build_modelsize_budget_plan(
        [
            {
                "row_id": "narrow",
                "archive_zip_bytes": 10_000,
                "avg_segnet_dist": 0.002,
                "avg_posenet_dist": 0.030,
            },
            {
                "row_id": "wide",
                "archive_zip_bytes": 12_000,
                "avg_segnet_dist": 0.001,
                "avg_posenet_dist": 0.025,
            },
        ],
        carrier_id="snerv",
    )

    assert plan["carrier_id"] == "snerv"
    assert plan["measured_points"][0]["nonrate_score"] > 0.0
    assert plan["selected_point"]["row_id"] in {"narrow", "wide"}
    assert plan["score_claim"] is False


def test_modelsize_budget_plan_blocks_single_point_ladder() -> None:
    plan = build_modelsize_budget_plan(
        [{"row_id": "only", "archive_bytes": 20_000, "nonrate_score": 0.22}]
    )

    assert plan["status"] == "insufficient_modelsize_ladder"
    assert plan["selected_archive_bytes"] == 20_000
    assert "modelsize_budget_ladder_has_fewer_than_two_points" in plan["blockers"]
    assert plan["score_claim"] is False
