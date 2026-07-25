# SPDX-License-Identifier: MIT

import math

import pytest

from tac.optimization.ddm_pf3_finite_price_materialization import (
    PF3MaterializationError,
    build_rd1_fail_closed_backfill,
    conditional_coordinate_price,
    joint_distortion_delta,
    materialized_bucket_report,
)


def test_joint_distortion_delta_uses_integer_seg_and_nonlinear_pose() -> None:
    result = joint_distortion_delta(
        base_errors=100,
        candidate_errors=90,
        base_d_pose=4.0,
        candidate_d_pose=1.0,
    )
    assert result["delta_D_seg"] < 0.0
    assert result["delta_D_pose"] == pytest.approx(math.sqrt(10.0) - math.sqrt(40.0))
    assert result["delta_D_joint"] == pytest.approx(
        result["delta_D_seg"] + result["delta_D_pose"]
    )


@pytest.mark.parametrize(
    ("dimension", "joint", "rate", "status"),
    [
        (1.0, -1.0, 4, "NONACTIONABLE_INTENDED_DIMENSION_NOT_IMPROVED"),
        (-1.0, 1.0, 4, "NONACTIONABLE_JOINT_SPILL_ERASES_DIMENSION_GAIN"),
        (-1.0, -1.0, 0, "DOMINATING_NONPOSITIVE_RATE_DELTA_REQUIRES_EDGE_REORDER"),
        (-2.0, -1.0, 4, "FINITE_CONDITIONAL_COORDINATE_SECANT_NOT_RD1_EDGE"),
    ],
)
def test_conditional_price_is_fail_closed(
    dimension: float,
    joint: float,
    rate: int,
    status: str,
) -> None:
    result = conditional_coordinate_price(
        delta_counted_bytes=rate,
        delta_D_dimension=dimension,
        delta_D_joint=joint,
    )
    assert result["status"] == status
    assert (result["lambda_bytes_per_D_dimension"] is not None) == status.startswith(
        "FINITE_"
    )
    assert result["actionable_for_rd1"] is False


def test_materialized_bucket_report_counts_only_complete_edges() -> None:
    occupied = [
        {
            "bucket_id": f"bucket_{index:02d}",
            "class_pair": "Road--Lane",
            "class_stratum": "cell",
            "g4_temporal_class": "STATIC_IN_IMAGE",
            "event_count": 1,
        }
        for index in range(37)
    ]
    candidates = [
        {
            "candidate_id": "candidate",
            "coordinate_id": "rg3.example",
            "direction_id": "POSITIVE_ONE_QUANTUM",
            "checkpoint_sha256": "a" * 64,
            "five_edges_complete": True,
            "five_pf3_edges": {"candidate_delta": {"delta_D_joint": 1.0}},
            "conditional_coordinate_price": {
                "status": "NO_FINITE_CONDITIONAL_COORDINATE_SECANT"
            },
            "bucket_measurements": [
                {"bucket_id": "bucket_00", "event_count": 1, "event_delta_errors": -1}
            ],
        }
    ]
    result = materialized_bucket_report(occupied, candidates)
    assert result["fully_materialized_occupied_bucket_count"] == 1
    assert result["rows"][0]["blocker"] is None
    assert (
        result["rows"][0]["finite_price_blocker"]
        == "ALL_MEASURED_SAME_OBJECT_COORDINATE_EDGES_WORSEN_JOINT_D"
    )
    assert result["rows"][1]["blocker"] == "NO_EXISTING_RG3_COORDINATE_HIT_THIS_PF2_BUCKET"


def test_rd1_backfill_never_cross_assigns_conditional_coordinate_price() -> None:
    rows = [
        {
            "dual_index": dual,
            "stratum": stratum,
            "scorer_visibility": visibility,
            "g4_temporal_class": temporal,
            "lambda_bytes_per_D_dimension": None,
            "actionable_for_train_decision": False,
        }
        for dual in (1, 2, 3)
        for stratum in ("Road", "Lane", "Undrivable", "Movable", "MyCar", "POSE6_GLOBAL")
        for visibility in ("ker(A)-invisible", "seg-visible", "pose-visible")
        for temporal in ("STATIC_IN_IMAGE", "STATIC_IN_XI_PROXY", "TRANSIENT")
    ]
    result = build_rd1_fail_closed_backfill(
        rows,
        conditional_coordinate_rows=[
            {
                "target_stratum": "Lane",
                "g4_temporal_class": "STATIC_IN_IMAGE",
                "lambda_bytes_per_D_dimension": 12.0,
            }
        ],
        source={"sha256": "b" * 64},
    )
    assert result["source_cell_count"] == 162
    assert result["conditional_coordinate_price_count"] == 1
    assert result["lambda_measured_cell_count"] == 0
    assert all(row["lambda_bytes_per_D_dimension"] is None for row in result["cells"])


def test_bad_rd1_cardinality_refuses() -> None:
    with pytest.raises(PF3MaterializationError, match="162"):
        build_rd1_fail_closed_backfill(
            [],
            conditional_coordinate_rows=[],
            source={},
        )
