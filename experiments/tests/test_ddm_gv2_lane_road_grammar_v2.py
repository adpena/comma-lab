from __future__ import annotations

import numpy as np

from experiments import ddm_ec1_event_coordinate_producer as ec1
from experiments import ddm_gv2_lane_road_grammar_v2 as gv2
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_connected_segment_stays_on_one_directed_boundary_component() -> None:
    boundary = np.zeros((gv2.H, gv2.W), dtype=bool)
    boundary[100, 40:180] = True
    indices = gv2.connected_segment(boundary, 100, 100, 48, 0.0)
    support = np.zeros_like(boundary)
    support.reshape(-1)[indices] = True
    _labels, count = gv2.ndimage.label(support, structure=gv2.CONNECT8)
    assert len(indices) == 48
    assert count == 1
    assert np.all(boundary.reshape(-1)[indices])


def test_opposite_construction_directions_make_distinct_segments() -> None:
    boundary = np.zeros((gv2.H, gv2.W), dtype=bool)
    boundary[120, 20:220] = True
    positive = gv2.connected_segment(boundary, 120, 120, 24, 0.0)
    negative = gv2.connected_segment(boundary, 120, 120, 24, np.pi)
    assert not np.array_equal(positive, negative)
    assert set(positive.tolist()) & set(negative.tolist())


def test_linear_pose_prediction_reports_nonnegative_global_bound() -> None:
    jacobian = np.zeros((6, 3 * gv2.H * gv2.W), dtype=np.float32)
    jacobian[0, 0] = 2.0
    correction = np.zeros((3, gv2.H, gv2.W), dtype=np.float32)
    correction.reshape(-1)[0] = 0.5
    result = gv2.linear_pose_prediction(
        jacobian,
        correction,
        np.zeros(6, dtype=np.float32),
        np.zeros(6, dtype=np.float64),
    )
    assert result["pose_shift6"][0] == 1.0
    assert result["predicted_delta_d_pose_pair"] == 1.0 / 6.0
    assert result["predicted_nonnegative_pose_bound_global_n600"] == 1.0 / 3600.0


def test_gv2_payload_is_the_unchanged_ec1_wire_format() -> None:
    indices = np.asarray([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=np.int64)
    payload = ec1.proposal_payload(53, 0, 1, indices, gv2.EVENT_TYPE_ID)
    pair, source, target, event_type, decoded = ec1.decode_proposal(payload)
    assert (pair, source, target, event_type) == (53, 0, 1, ec1.EVENT_TYPE["lane_program_delta"])
    assert np.array_equal(decoded, indices)


def test_gv2_passes_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=gv2.REPO,
        strict=False,
        roots=("experiments/ddm_gv2_lane_road_grammar_v2.py",),
    )
    assert findings == []
