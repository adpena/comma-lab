# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_realized_flip_menu import (
    CAMERA_HW,
    SEG_HW,
    RealizedFlipMenuError,
    advisory_objective,
    apply_local_statistics,
    apply_scalar_affine,
    apply_temporal_affine,
    cluster_id,
    compile_menu_rows,
    decode_local_statistics,
    decode_target_masks,
    encode_local_statistics,
    encode_scalar_affine,
    encode_target_masks,
    encode_temporal_affine,
    greedy_telescoping_curve,
    transition_counts,
)


def _solve_row(index: int) -> dict[str, object]:
    return {
        "menu_rank": index + 1,
        "error_count": index + 1,
        "source": "NEVER_DESCRIBED",
        "stratum": "Road",
        "ordered_pair": "Undrivable->Road",
        "boundary_distance_band": f"BAND_{index}",
        "curvature_band": "INTERIOR",
        "curve_availability": "NO_CONTINUOUS_LANE_CURVE",
        "d2_band": "ABOVE_SIDED_Q90",
        "g3_tail_bucket": "G3_TAIL",
        "paint_floor_mechanism": "COARSE_DESCRIPTION",
        "temporal_pattern": "STATIC_IN_IMAGE_HISTORICAL",
    }


def test_compile_menu_is_complete_and_cross_control_fail_closed() -> None:
    rows = [_solve_row(index) for index in range(2_649)]
    compiled = compile_menu_rows(
        rows,
        v19c_residual_errors=2_265_811,
        v19c_total_errors=2_923_991,
        receipt_sha256={"sn1": "a" * 64},
    )
    assert len(compiled) == 2_649 * 6
    assert len({row["row_id"] for row in compiled}) == len(compiled)
    assert all(row["waterfill_eligible"] is False for row in compiled)
    assert all(row["delta_errors_realized"] is None for row in compiled)
    assert compiled[0]["cluster_id"] == cluster_id(rows[0])
    assert compiled[0]["base_residual_errors"] == 2_265_811
    assert compiled[0]["base_total_errors"] == 2_923_991
    assert compiled[0]["byte_partition"]["law"] == "FREE_UNION_NULL_UNION_COUNTED"


def test_local_statistics_payload_roundtrip_and_application() -> None:
    scale = np.ones((5, 2, 3), dtype=np.float32)
    offset = np.zeros_like(scale)
    scale[0, 0] = (2.0, 1.0, 0.5)
    offset[0, 0] = (1.0, 2.0, 3.0)
    payload = encode_local_statistics(scale, offset)
    decoded_scale, decoded_offset = decode_local_statistics(payload)
    assert decoded_scale.dtype == np.float16
    assert np.array_equal(decoded_scale, scale.astype(np.float16))
    assert np.array_equal(decoded_offset, offset.astype(np.float16))
    camera = np.full((1, 2, *CAMERA_HW, 3), 10, dtype=np.uint8)
    cells = np.zeros((1, *SEG_HW), dtype=np.uint8)
    result = apply_local_statistics(camera, cells, payload)
    assert np.array_equal(result[:, 0], camera[:, 0])
    assert tuple(result[0, 1, 0, 0]) == (21, 12, 8)
    assert tuple(result[0, 1, -1, 0]) == (10, 10, 10)


def test_scalar_and_temporal_affine_are_frame1_only_and_parse_back() -> None:
    camera = np.full((2, 2, *CAMERA_HW, 3), 10, dtype=np.uint8)
    scalar_payload = encode_scalar_affine(2.0, 1.0)
    assert len(scalar_payload) == 12
    scalar = apply_scalar_affine(camera, scalar_payload)
    assert np.array_equal(scalar[:, 0], camera[:, 0])
    assert np.all(scalar[:, 1] == 21)
    temporal_payload = encode_temporal_affine(
        np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
    )
    assert len(temporal_payload) == 36
    temporal = apply_temporal_affine(
        camera,
        pair_ids=(0, 1),
        pair_count=2,
        payload=temporal_payload,
    )
    assert np.array_equal(temporal[:, 0], camera[:, 0])
    assert np.all(temporal[0, 1] == 11)
    assert np.all(temporal[1, 1] == 22)


def test_target_mask_codec_roundtrip_and_trailing_byte_refusal() -> None:
    first = np.zeros((1, *SEG_HW), dtype=bool)
    second = np.zeros((2, *SEG_HW), dtype=bool)
    first[0, 2:5, 7:11] = True
    second[1, 100:105, 200:207] = True
    payload = encode_target_masks([(0, first), (1, second)])
    decoded = decode_target_masks(payload)
    assert np.array_equal(decoded[0][1], first)
    assert np.array_equal(decoded[1][1], second)
    with pytest.raises(RealizedFlipMenuError, match="trailing"):
        decode_target_masks(payload + b"x")


def test_transition_and_telescoping_nonadditivity() -> None:
    target = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    before = np.array([[1, 0, 0, 1]], dtype=np.uint8)
    after = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    transition = transition_counts(before=before, after=after, target=target)
    assert transition == {
        "errors_before": 2,
        "errors_after": 2,
        "errors_corrected": 1,
        "errors_introduced": 1,
        "errors_persisting": 1,
        "delta_errors_realized": 0,
    }
    base = {
        "candidate_id": "base",
        "archive_bytes": 100,
        "advisory_objective": 2.0,
    }
    proposals = [
        {
            "candidate_id": "a",
            "parent_candidate_id": "base",
            "archive_bytes": 110,
            "advisory_objective": 1.5,
        },
        {
            "candidate_id": "b",
            "parent_candidate_id": "a",
            "archive_bytes": 120,
            "advisory_objective": 1.6,
        },
    ]
    curve = greedy_telescoping_curve(
        base=base,
        proposals=proposals,
        byte_budget=115,
    )
    assert [row["admitted"] for row in curve] == [True, True, False]
    assert curve[-1]["admission_reason"] == "BYTE_BUDGET_AND_NO_JOINT_GAIN"
    assert curve[-1]["admission_gates"] == {
        "within_byte_budget": False,
        "strict_joint_improvement": False,
    }
    assert advisory_objective(errors=0, sites=1, d_pose=0.0, bytes_=0) == 0.0
