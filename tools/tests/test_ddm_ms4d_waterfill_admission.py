from __future__ import annotations

import copy

import pytest

from tac.optimization.ddm_metric_custody_bundle import (
    DIRECT_METRIC_MODE,
    SEG_DIRECT_DATA_SCHEMA,
)
from tac.optimization.ddm_ms4d_waterfill_admission import (
    BLOCKER,
    build_still_null_backfill,
    candidate_materialization_gaps,
    registered_callable_control,
)


def _seg_data() -> dict[str, object]:
    rows = [
        {
            "event_count": 1 if index < 37 else 0,
        }
        for index in range(1_200)
    ]
    blocks = [
        {
            "pair_id": index,
            "bucket_id": f"bucket-{index}",
            "actuation_status": "UNREACHABLE_BY_COUNTED_COORDINATES",
        }
        for index in range(25)
    ]
    return {
        "schema": SEG_DIRECT_DATA_SCHEMA,
        "metric_mode": DIRECT_METRIC_MODE,
        "rows": rows,
        "direct_blocks": blocks,
    }


def test_direct_metrics_do_not_masquerade_as_candidate_materialization() -> None:
    result = candidate_materialization_gaps(_seg_data())
    assert result["candidate_materialization_ready"] is False
    assert result["blocker"] == BLOCKER
    assert result["occupied_metric_bucket_count"] == 37
    assert result["exact_empty_metric_bucket_count"] == 1_163
    assert result["direct_unreachable_pair_bucket_count"] == 25
    assert set(result["materialization_field_counts"].values()) == {0}
    assert result["fully_materialized_occupied_bucket_count"] == 0


def test_partial_materialization_fields_still_refuse() -> None:
    value = _seg_data()
    for row in value["rows"]:  # type: ignore[index]
        row["receiver_object_builder"] = "builder-A"
    result = candidate_materialization_gaps(value)
    assert result["candidate_materialization_ready"] is False
    assert result["materialization_field_counts"]["receiver_object_builder"] == 1_200
    assert result["materialization_field_counts"]["dimension_rate_home"] == 0
    assert result["fully_materialized_occupied_bucket_count"] == 0


def test_unreachable_typing_is_strict() -> None:
    value = copy.deepcopy(_seg_data())
    value["direct_blocks"][0]["actuation_status"] = "REACHABLE"  # type: ignore[index]
    with pytest.raises(ValueError, match="actuation typing"):
        candidate_materialization_gaps(value)


def test_rd1_backfill_records_metric_but_preserves_lambda_null() -> None:
    rows = [
        {
            "dual_index": index // 54 + 1,
            "left_candidate_id": "left",
            "right_candidate_id": "right",
            "stratum": "Road",
            "scorer_visibility": "seg-visible",
            "g4_temporal_class": "STATIC_IN_IMAGE",
            "effective_quantum_D": 0.5,
            "status": "SOURCE_NULL",
        }
        for index in range(162)
    ]
    result = build_still_null_backfill(
        rows,
        source={"sha256": "a" * 64},
        complete_bundle={"sha256": "b" * 64},
    )
    assert result["metric_bundle_context_cell_count"] == 162
    assert result["rung_measured_cell_count"] == 0
    assert result["lambda_measured_cell_count"] == 0
    assert result["still_null_lambda_cell_count"] == 162
    assert all(row["lambda_bytes_per_D_dimension"] is None for row in result["cells"])
    assert all(row["actionable_for_train_decision"] is False for row in result["cells"])
    assert all(
        row["rung_measurement_status"]
        == "STILL_NULL_NO_MATERIALIZED_SAME_OBJECT_RUNG"
        for row in result["cells"]
    )


@pytest.mark.parametrize(
    ("candidate_id", "counted_bytes", "d_seg", "d_pose", "score", "inside"),
    [
        (
            "c1_exact_solved_n600",
            409_526_925,
            0.0001519690619574653,
            0.00010184312078531729,
            272.7342793310384,
            True,
        ),
        (
            "statistics_hard_analytic_composed_frame1",
            138_801,
            0.07051923116048177,
            36.6181847780574,
            26.28022355199344,
            False,
        ),
    ],
)
def test_registered_callable_replays_settled_control_only(
    candidate_id: str,
    counted_bytes: int,
    d_seg: float,
    d_pose: float,
    score: float,
    inside: bool,
) -> None:
    result = registered_callable_control(
        {
            "candidate_id": candidate_id,
            "counted_bytes": counted_bytes,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "S_composed": score,
            "receiver_closure": "archive_receiver_closed",
        },
        role="control",
    )
    assert result["joint_S"] == pytest.approx(score, abs=2e-12)
    assert result["admissible_inside_error_cap"] is inside
    assert result["coder_race_performed"] is False
    assert result["epistemic_status"].endswith("NOT_A_NEW_RUNG")


def test_registered_callable_refuses_unproven_receiver_closure() -> None:
    with pytest.raises(ValueError, match="receiver-closure status"):
        registered_callable_control(
            {
                "candidate_id": "not-closed",
                "counted_bytes": 1,
                "d_seg": 0.0,
                "d_pose": 0.0,
                "S_composed": 25 / 37_545_489,
                "receiver_closure": "UNPROVEN",
            },
            role="control",
        )
