# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
    build_modelsize_budget_plan,
)


def _receiver_proof_fields(label: str) -> dict[str, object]:
    return {
        "receiver_proof_passed": True,
        "receiver_proof_path": f"/Volumes/VertigoDataTier/pact/proofs/{label}.json",
        "receiver_proof_sha256": "a" * 64,
        "archive_sha256": "b" * 64,
        "axis_tag": "[planning/control]",
        "sample_pair_count": 600,
    }


def test_modelsize_budget_plan_selects_measured_total_score_minimum() -> None:
    rows = [
        {
            "row_id": "tiny",
            "archive_bytes": 20_000,
            "nonrate_score": 0.240,
            "modelsize_mparams": 0.04,
            **_receiver_proof_fields("tiny"),
        },
        {
            "row_id": "small",
            "archive_bytes": 40_000,
            "nonrate_score": 0.205,
            "modelsize_mparams": 0.08,
            **_receiver_proof_fields("small"),
        },
        {
            "row_id": "medium",
            "archive_bytes": 80_000,
            "nonrate_score": 0.200,
            "modelsize_mparams": 0.16,
            **_receiver_proof_fields("medium"),
        },
    ]

    plan = build_modelsize_budget_plan(rows, carrier_id="hi_nerv")

    assert plan["status"] == "receiver_closed_modelsize_budget_selected"
    assert plan["decision_basis"] == "receiver_closed_rows"
    assert plan["selected_point"]["row_id"] == "small"
    assert plan["selected_archive_bytes"] == 40_000
    assert plan["receiver_closed_selected_archive_bytes"] == 40_000
    assert plan["point_count_by_evidence"] == {"receiver_closed_measured_bytes": 3}
    first, second = plan["marginal_steps"]
    assert first["spend_rule"] == "spend_modelsize_byte"
    assert first["marginal_improvement_per_byte"] > CONTEST_BYTE_PRICE_SCORE
    assert second["spend_rule"] == "stop_or_reallocate_modelsize_byte"
    assert second["marginal_improvement_per_byte"] < CONTEST_BYTE_PRICE_SCORE
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_modelsize_budget_plan_respects_hard_byte_ceiling() -> None:
    rows = [
        {
            "row_id": "tiny",
            "archive_bytes": 20_000,
            "nonrate_score": 0.240,
            "modelsize_mparams": 0.04,
            **_receiver_proof_fields("tiny"),
        },
        {
            "row_id": "small",
            "archive_bytes": 40_000,
            "nonrate_score": 0.205,
            "modelsize_mparams": 0.08,
            **_receiver_proof_fields("small"),
        },
        {
            "row_id": "medium_over_cap",
            "archive_bytes": 80_000,
            "nonrate_score": 0.010,
            "modelsize_mparams": 0.16,
            **_receiver_proof_fields("medium_over_cap"),
        },
    ]

    plan = build_modelsize_budget_plan(
        rows,
        carrier_id="hi_nerv",
        hard_byte_ceiling=50_000,
    )

    assert plan["hard_byte_ceiling"] == 50_000
    assert plan["selected_point"]["row_id"] == "small"
    assert plan["selected_archive_bytes"] == 40_000
    assert plan["selected_under_hard_byte_ceiling"] is True
    assert plan["decision_under_hard_byte_ceiling_point_count"] == 2
    assert "modelsize_budget_selected_point_over_hard_byte_ceiling" not in plan[
        "blockers"
    ]


def test_modelsize_budget_plan_blocks_when_no_point_fits_hard_byte_ceiling() -> None:
    rows = [
        {
            "row_id": "tiny_over_cap",
            "archive_bytes": 20_000,
            "nonrate_score": 0.240,
            "modelsize_mparams": 0.04,
            **_receiver_proof_fields("tiny_over_cap"),
        },
        {
            "row_id": "small_over_cap",
            "archive_bytes": 40_000,
            "nonrate_score": 0.205,
            "modelsize_mparams": 0.08,
            **_receiver_proof_fields("small_over_cap"),
        },
    ]

    plan = build_modelsize_budget_plan(
        rows,
        carrier_id="snerv",
        hard_byte_ceiling=10_000,
    )

    assert plan["selected_under_hard_byte_ceiling"] is False
    assert plan["under_hard_byte_ceiling_point_count"] == 0
    assert "modelsize_budget_no_candidate_under_hard_byte_ceiling" in plan[
        "blockers"
    ]
    assert "modelsize_budget_selected_point_over_hard_byte_ceiling" in plan[
        "blockers"
    ]
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_modelsize_budget_plan_splits_projected_and_advisory_rows() -> None:
    plan = build_modelsize_budget_plan(
        [
            {
                "row_id": "ideal_curve",
                "projected_archive_bytes_600pair": 36_000,
                "nonrate_score": 0.240,
                "modelsize_mparams": 0.04,
                "lower_bound_only": True,
            },
            {
                "row_id": "zip_without_receiver",
                "archive_zip_bytes": 72_000,
                "nonrate_score": 0.210,
                "fc_dim": 16,
            },
        ],
        carrier_id="snerv",
    )

    assert plan["status"] == "advisory_or_projected_modelsize_budget_selected"
    assert plan["decision_basis"] == "all_rows_advisory_planning_only"
    assert plan["selected_point"]["row_id"] in {
        "ideal_curve",
        "zip_without_receiver",
    }
    assert plan["receiver_closed_selected_point"] is None
    assert plan["receiver_closed_selected_archive_bytes"] is None
    assert plan["point_count_by_evidence"] == {
        "projected_or_lower_bound_bytes": 1,
        "advisory_measured_bytes_without_receiver_proof": 1,
    }
    assert "modelsize_budget_selection_is_advisory_or_projected" in plan["blockers"]
    assert "receiver_closed_byte_proof_missing" in plan["blockers"]
    assert (
        "projected_or_lower_bound_archive_bytes_not_receiver_closed"
        in plan["blockers"]
    )
    assert plan["score_claim"] is False


def test_modelsize_budget_plan_can_extract_nonrate_from_component_distortions() -> None:
    plan = build_modelsize_budget_plan(
        [
            {
                "row_id": "narrow",
                "archive_zip_bytes": 10_000,
                "fc_dim": 8,
                "avg_segnet_dist": 0.002,
                "avg_posenet_dist": 0.030,
            },
            {
                "row_id": "wide",
                "archive_zip_bytes": 12_000,
                "fc_dim": 16,
                "avg_segnet_dist": 0.001,
                "avg_posenet_dist": 0.025,
            },
        ],
        carrier_id="snerv",
    )

    assert plan["carrier_id"] == "snerv"
    assert plan["status"] == "advisory_or_projected_modelsize_budget_selected"
    assert plan["measured_points"] == []
    assert plan["points"][0]["nonrate_score"] > 0.0
    assert plan["receiver_closed_selected_point"] is None
    assert plan["selected_point"]["row_id"] in {"narrow", "wide"}
    assert plan["score_claim"] is False


def test_modelsize_budget_plan_blocks_single_point_ladder() -> None:
    plan = build_modelsize_budget_plan(
        [
            {
                "row_id": "only",
                "archive_bytes": 20_000,
                "nonrate_score": 0.22,
                "modelsize_mparams": 0.04,
            }
        ]
    )

    assert plan["status"] == "insufficient_modelsize_ladder"
    assert plan["selected_archive_bytes"] == 20_000
    assert "modelsize_budget_ladder_has_fewer_than_two_points" in plan["blockers"]
    assert (
        "receiver_closed_modelsize_ladder_has_fewer_than_two_points"
        in plan["blockers"]
    )
    assert plan["score_claim"] is False


def test_modelsize_budget_plan_blocks_unbound_capacity_controls() -> None:
    plan = build_modelsize_budget_plan(
        [
            {
                "row_id": "tiny_unbound",
                "archive_bytes": 20_000,
                "nonrate_score": 0.240,
                **_receiver_proof_fields("tiny_unbound"),
            },
            {
                "row_id": "small_unbound",
                "archive_bytes": 40_000,
                "nonrate_score": 0.205,
                **_receiver_proof_fields("small_unbound"),
            },
        ],
        carrier_id="hi_nerv",
    )

    assert plan["status"] == "receiver_closed_modelsize_budget_selected"
    assert "source_bound_modelsize_or_fc_dim_missing" in plan["blockers"]
    assert all(
        point["source_bound_capacity_control"] is False
        for point in plan["points"]
    )
    assert (
        "emit_source_bound_modelsize_mparams_or_fc_dim_for_budget_points"
        in plan["recommended_next_actions"]
    )


def test_modelsize_budget_plan_rejects_bare_receiver_proof_boolean_as_advisory() -> None:
    plan = build_modelsize_budget_plan(
        [
            {
                "row_id": "tiny_boolean_only",
                "archive_bytes": 20_000,
                "nonrate_score": 0.240,
                "modelsize_mparams": 0.04,
                "receiver_proof_passed": True,
            },
            {
                "row_id": "small_boolean_only",
                "archive_bytes": 40_000,
                "nonrate_score": 0.205,
                "modelsize_mparams": 0.08,
                "receiver_proof_passed": True,
            },
        ],
        carrier_id="snerv",
    )

    assert plan["status"] == "advisory_or_projected_modelsize_budget_selected"
    assert plan["decision_basis"] == "all_rows_advisory_planning_only"
    assert plan["receiver_closed_points"] == []
    assert plan["point_count_by_evidence"] == {
        "advisory_measured_bytes_without_receiver_proof": 2
    }
    assert "receiver_proof_path_missing" in plan["blockers"]
    assert "receiver_proof_sha256_missing_or_invalid" in plan["blockers"]
    assert "archive_sha256_missing_or_invalid" in plan["blockers"]
    assert "receiver_proof_axis_tag_missing" in plan["blockers"]
    assert "receiver_proof_full_sample_count_missing" in plan["blockers"]


def test_modelsize_budget_plan_hard_ceiling_selects_best_under_cap() -> None:
    rows = [
        {
            "row_id": "tiny",
            "archive_bytes": 40_000,
            "nonrate_score": 0.250,
            "modelsize_mparams": 0.04,
            **_receiver_proof_fields("tiny"),
        },
        {
            "row_id": "small",
            "archive_bytes": 100_000,
            "nonrate_score": 0.190,
            "modelsize_mparams": 0.08,
            **_receiver_proof_fields("small"),
        },
        {
            "row_id": "wide_over_cap",
            "archive_bytes": 240_000,
            "nonrate_score": 0.010,
            "modelsize_mparams": 0.20,
            **_receiver_proof_fields("wide_over_cap"),
        },
    ]

    plan = build_modelsize_budget_plan(
        rows,
        carrier_id="hi_nerv",
        hard_byte_ceiling=178_000,
    )

    assert plan["status"] == "receiver_closed_modelsize_budget_selected"
    assert plan["hard_byte_ceiling"] == 178_000
    assert plan["selected_point"]["row_id"] == "small"
    assert plan["selected_archive_bytes"] == 100_000
    assert plan["selected_under_hard_byte_ceiling"] is True
    assert plan["under_hard_byte_ceiling_point_count"] == 2
    assert plan["decision_under_hard_byte_ceiling_point_count"] == 2
    assert "modelsize_budget_selected_point_over_hard_byte_ceiling" not in plan[
        "blockers"
    ]
    assert all(
        step["to_archive_bytes"] <= 178_000 for step in plan["marginal_steps"]
    )


def test_modelsize_budget_plan_hard_ceiling_blocks_when_no_candidate_fits() -> None:
    rows = [
        {
            "row_id": "small_over_cap",
            "archive_bytes": 200_000,
            "nonrate_score": 0.250,
            "modelsize_mparams": 0.08,
            **_receiver_proof_fields("small_over_cap"),
        },
        {
            "row_id": "wide_over_cap",
            "archive_bytes": 260_000,
            "nonrate_score": 0.120,
            "modelsize_mparams": 0.16,
            **_receiver_proof_fields("wide_over_cap"),
        },
    ]

    plan = build_modelsize_budget_plan(
        rows,
        carrier_id="snerv",
        hard_byte_ceiling=178_000,
    )

    assert plan["hard_byte_ceiling"] == 178_000
    assert plan["under_hard_byte_ceiling_point_count"] == 0
    assert plan["decision_under_hard_byte_ceiling_point_count"] == 0
    assert plan["selected_under_hard_byte_ceiling"] is False
    assert "modelsize_budget_no_candidate_under_hard_byte_ceiling" in plan[
        "blockers"
    ]
    assert "modelsize_budget_no_decision_candidate_under_hard_byte_ceiling" in plan[
        "blockers"
    ]
    assert "modelsize_budget_selected_point_over_hard_byte_ceiling" in plan[
        "blockers"
    ]
    assert plan["score_claim"] is False
