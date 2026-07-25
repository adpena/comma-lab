from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_feature_relay_multiple_shooting import (
    ACCEPTANCE_AUTHORITY,
    DIRECT_METHOD,
    METRIC_KIND,
    RELAY_METHOD,
    FeatureRelayError,
    RelayProblemV1,
    RelaySegmentV1,
    RelayStationV1,
    admit_realized_endpoint,
    compare_realized_radius,
    solve_direct_final_station,
    solve_multiple_shooting,
)

SHA = "1" * 64


def _problem() -> RelayProblemV1:
    stations = (
        RelayStationV1(
            station_id="block2_pre_se",
            layer_path="encoder.model.blocks.1.2.se.forward_pre",
            target_delta=np.array([1.0, -0.5]),
            metric_gram=np.diag([3.0, 1.0]),
            metric_kind=METRIC_KIND,
            evidence_sha256=SHA,
        ),
        RelayStationV1(
            station_id="block3_pre_se",
            layer_path="encoder.model.blocks.2.2.se.forward_pre",
            target_delta=np.array([0.4, 0.8]),
            metric_gram=np.array([[2.0, 0.25], [0.25, 1.0]]),
            metric_kind=METRIC_KIND,
            evidence_sha256=SHA,
        ),
        RelayStationV1(
            station_id="rank4_head",
            layer_path="segmentation_head.rank4_quotient",
            target_delta=np.array([0.3]),
            metric_gram=np.array([[4.0]]),
            metric_kind=METRIC_KIND,
            evidence_sha256=SHA,
        ),
    )
    segments = (
        RelaySegmentV1(
            segment_id="range_a_to_block2",
            source_id="range_a_input",
            target_id="block2_pre_se",
            jacobian=np.array([[1.0, 0.0], [0.5, 1.0]]),
            evidence_sha256=SHA,
        ),
        RelaySegmentV1(
            segment_id="block2_to_block3",
            source_id="block2_pre_se",
            target_id="block3_pre_se",
            jacobian=np.array([[1.0, 0.2], [0.0, 0.5]]),
            evidence_sha256=SHA,
        ),
        RelaySegmentV1(
            segment_id="block3_to_head",
            source_id="block3_pre_se",
            target_id="rank4_head",
            jacobian=np.array([[0.5, -0.25]]),
            evidence_sha256=SHA,
        ),
    )
    return RelayProblemV1(
        stations=stations,
        segments=segments,
        actuator_dimension=2,
        actuator_metric=np.diag([0.1, 0.2]),
    )


def _row(*, d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, object]:
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "receiver_parseback_exact": True,
        "realized_through_r_uint8": True,
        "frozen_scorers": True,
        "num_pairs": 600,
        "score_claim": False,
    }


def test_multiple_shooting_enforces_continuity_and_fisher_primary() -> None:
    result = solve_multiple_shooting(_problem())
    assert result["method"] == RELAY_METHOD
    assert result["metric_primary"] == METRIC_KIND
    assert result["euclidean_authority"] is False
    assert result["predicted_only"] is True
    assert result["realized_acceptance"] is None
    assert result["continuity_residual_l2"] <= result[
        "continuity_tolerance_fp64_derived"
    ]
    assert len(result["station_rows"]) == 3
    assert all(
        row["predicted_fisher_debt_after"] <= row["predicted_fisher_debt_before"]
        + 1e-12
        for row in result["station_rows"]
    )


def test_direct_solve_is_labeled_one_shot_and_prediction_only() -> None:
    result = solve_direct_final_station(_problem())
    assert result["method"] == DIRECT_METHOD
    assert result["predicted_only"] is True
    assert result["realized_acceptance"] is None
    assert result["predicted_final_fisher_reduction"] > 0.0


def test_two_station_chain_is_supported() -> None:
    full = _problem()
    problem = RelayProblemV1(
        stations=full.stations[:2],
        segments=full.segments[:2],
        actuator_dimension=full.actuator_dimension,
        actuator_metric=full.actuator_metric,
    )
    result = solve_multiple_shooting(problem)
    assert len(result["station_rows"]) == 2
    assert result["continuity_residual_l2"] <= result[
        "continuity_tolerance_fp64_derived"
    ]


def test_euclidean_station_metric_is_refused() -> None:
    with pytest.raises(FeatureRelayError, match="custody differs"):
        RelayStationV1(
            station_id="bad",
            layer_path="layer",
            target_delta=np.array([1.0]),
            metric_gram=np.array([[1.0]]),
            metric_kind="euclidean",
            evidence_sha256=SHA,
        )


def test_segment_chain_shape_mismatch_is_refused() -> None:
    problem = _problem()
    bad = RelaySegmentV1(
        segment_id="bad",
        source_id="range_a_input",
        target_id="block2_pre_se",
        jacobian=np.ones((3, 2)),
        evidence_sha256=SHA,
    )
    with pytest.raises(FeatureRelayError, match="segment chain differs"):
        RelayProblemV1(
            stations=problem.stations,
            segments=(bad, *problem.segments[1:]),
            actuator_dimension=2,
            actuator_metric=np.eye(2),
        )


def test_realized_acceptance_uses_joint_end_delta_only() -> None:
    reference = _row(d_seg=0.1, d_pose=4.0, archive_bytes=1000)
    candidate = _row(d_seg=0.099, d_pose=4.0, archive_bytes=1000)
    result = admit_realized_endpoint(
        method=RELAY_METHOD,
        reference=reference,
        candidate=candidate,
    )
    assert result["accepted"] is True
    assert result["intermediate_prediction_used_for_acceptance"] is False
    assert result["delta"]["joint_delta"] == pytest.approx(-0.1)


def test_realized_acceptance_refuses_non_n600_or_parseback_gap() -> None:
    reference = _row(d_seg=0.1, d_pose=4.0, archive_bytes=1000)
    candidate = _row(d_seg=0.099, d_pose=4.0, archive_bytes=1000)
    candidate["receiver_parseback_exact"] = False
    with pytest.raises(FeatureRelayError, match="parse-back"):
        admit_realized_endpoint(
            method=RELAY_METHOD,
            reference=reference,
            candidate=candidate,
        )


def test_realized_acceptance_refuses_fractional_archive_bytes() -> None:
    reference = _row(d_seg=0.1, d_pose=4.0, archive_bytes=1000)
    candidate = _row(d_seg=0.099, d_pose=4.0, archive_bytes=1000)
    candidate["archive_bytes"] = 1000.0
    with pytest.raises(FeatureRelayError, match="exact integer"):
        admit_realized_endpoint(
            method=RELAY_METHOD,
            reference=reference,
            candidate=candidate,
        )


def test_equal_budget_radius_comparison_uses_only_realized_rows() -> None:
    reference = _row(d_seg=0.1, d_pose=4.0, archive_bytes=1000)
    direct = []
    relay = []
    for radius in (1, 2, 3):
        direct_row = admit_realized_endpoint(
            method=DIRECT_METHOD,
            reference=reference,
            candidate=_row(
                d_seg=0.099 if radius <= 2 else 0.101,
                d_pose=4.0,
                archive_bytes=1000,
            ),
        )
        relay_row = admit_realized_endpoint(
            method=RELAY_METHOD,
            reference=reference,
            candidate=_row(d_seg=0.099, d_pose=4.0, archive_bytes=1000),
        )
        direct.append({**direct_row, "radius_quanta": radius})
        relay.append({**relay_row, "radius_quanta": radius})
    comparison = compare_realized_radius(direct_rows=direct, relay_rows=relay)
    assert comparison["direct_radius_quanta"] == 2
    assert comparison["relay_radius_quanta"] == 3
    assert comparison["relay_beats_direct"] is True
    assert comparison["acceptance_authority"] == ACCEPTANCE_AUTHORITY


def test_radius_comparison_refuses_unequal_budget() -> None:
    with pytest.raises(FeatureRelayError, match="equal nonempty"):
        compare_realized_radius(
            direct_rows=[{"schema": "irrelevant"}],
            relay_rows=[],
        )


def test_radius_is_contiguous_accepted_prefix_not_largest_isolated_accept() -> None:
    reference = _row(d_seg=0.1, d_pose=4.0, archive_bytes=1000)
    rows: list[dict[str, object]] = []
    for radius, d_seg in ((1, 0.099), (2, 0.101), (3, 0.099)):
        admission = admit_realized_endpoint(
            method=RELAY_METHOD,
            reference=reference,
            candidate=_row(d_seg=d_seg, d_pose=4.0, archive_bytes=1000),
        )
        rows.append({**admission, "radius_quanta": radius})
    direct_rows = [{**row, "method": DIRECT_METHOD} for row in rows]
    comparison = compare_realized_radius(direct_rows=direct_rows, relay_rows=rows)
    assert comparison["direct_radius_quanta"] == 1
    assert comparison["relay_radius_quanta"] == 1


def test_radius_comparison_refuses_different_ladders() -> None:
    reference = _row(d_seg=0.1, d_pose=4.0, archive_bytes=1000)
    admission = admit_realized_endpoint(
        method=RELAY_METHOD,
        reference=reference,
        candidate=_row(d_seg=0.099, d_pose=4.0, archive_bytes=1000),
    )
    direct = [{**admission, "method": DIRECT_METHOD, "radius_quanta": 1}]
    relay = [{**admission, "radius_quanta": 2}]
    with pytest.raises(FeatureRelayError, match="same realized radius ladder"):
        compare_realized_radius(direct_rows=direct, relay_rows=relay)
