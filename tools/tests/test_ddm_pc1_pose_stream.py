# SPDX-License-Identifier: MIT
"""Regression tests for the typed counted DDM PC1 pose-stream owner."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
import torch

from tac.canonical_equations.ddm_pc1_pose_stream_20260724 import (
    admission_fence,
    ms4d_pose_quadratic,
    non_telescoping_conditional_delta_s,
)
from tac.optimization.ddm_pc1_pose_stream import (
    CAMERA_H,
    CAMERA_W,
    PAIR_H,
    PAIR_W,
    DDMPC1TrainableParameterMapV1,
    PC1PoseStreamError,
    _warp_scorer_frame,
    active_tube_quadratic,
    build_counted_composition_archive,
    fresh_pose_initialization,
    ground_and_movable_depth,
    make_inactive_packet,
    make_zero_active_packet,
    output_effect_owners,
    parse_counted_composition_archive,
    parse_pc1_packet,
    receive_pc1_camera_pairs,
    serialize_pc1_packet,
    solved_plane_yuv6_target,
    verify_unique_output_effect_owners,
)


def _packet(pair_count: int = 2):
    centers = np.asarray(
        [[20.0 + pair, -0.2, 0.4, -0.1, 0.3, -0.5] for pair in range(pair_count)],
        dtype=np.float64,
    )
    xi, scales = fresh_pose_initialization(centers, knot_count=pair_count)
    parameter_map = DDMPC1TrainableParameterMapV1(
        pair_count=pair_count,
        knot_count=pair_count,
        xi_scales=scales,
        residual_scale=0.25,
    )
    return parameter_map, parameter_map.project(
        xi=xi,
        luma_phase=np.zeros((pair_count, 4), dtype=np.float64),
    )


def _gradient_parent() -> np.ndarray:
    x = np.arange(CAMERA_W, dtype=np.uint16)[None, :]
    y = np.arange(CAMERA_H, dtype=np.uint16)[:, None]
    frame = np.stack(
        (
            np.broadcast_to(x % 256, (CAMERA_H, CAMERA_W)),
            np.broadcast_to(y % 256, (CAMERA_H, CAMERA_W)),
            (x + y) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return np.stack((frame, np.flip(frame, axis=1)), axis=0)[None, ...]


def test_packet_roundtrip_is_canonical_and_corruption_fails_closed() -> None:
    _, packet = _packet()
    payload = serialize_pc1_packet(packet)
    parsed = parse_pc1_packet(payload)
    assert serialize_pc1_packet(parsed) == payload
    assert parsed.q_xi.flags.writeable is False
    with pytest.raises(PC1PoseStreamError):
        parse_pc1_packet(payload[:-1] + bytes([payload[-1] ^ 1]))


def test_parameter_map_has_stable_unique_366_coordinates() -> None:
    parameter_map, packet = _packet()
    rows = parameter_map.coordinates()
    assert len(rows) == 20
    assert len({row.coordinate_id for row in rows}) == 20
    assert rows[0].coordinate_id == "ddm.pc1.knot.000.xi.tx"
    assert rows[-1].coordinate_id == "ddm.pc1.knot.001.luma_phase.3"
    assert np.count_nonzero(packet.q_xi) == 12
    assert np.count_nonzero(make_zero_active_packet(packet).q_xi) == 0


def test_inactive_is_exact_and_nonzero_q_changes_both_generated_frames() -> None:
    _, packet = _packet(pair_count=2)
    parent = _gradient_parent()
    mask = np.zeros((1, PAIR_H, PAIR_W), dtype=np.bool_)
    mask[:, 240:320, 210:300] = True
    inactive = receive_pc1_camera_pairs(
        parent_camera=parent,
        packet=make_inactive_packet(packet),
        pair_ids=[0],
        movable_masks=mask,
        torch_module=torch,
    )
    zero_home = receive_pc1_camera_pairs(
        parent_camera=parent,
        packet=make_zero_active_packet(packet),
        pair_ids=[0],
        movable_masks=mask,
        torch_module=torch,
    )
    candidate = receive_pc1_camera_pairs(
        parent_camera=parent,
        packet=packet,
        pair_ids=[0],
        movable_masks=mask,
        torch_module=torch,
    )
    replay = receive_pc1_camera_pairs(
        parent_camera=parent,
        packet=parse_pc1_packet(serialize_pc1_packet(packet)),
        pair_ids=[0],
        movable_masks=mask,
        torch_module=torch,
    )
    assert np.array_equal(inactive, parent)
    assert np.array_equal(candidate, replay)
    assert np.count_nonzero(candidate[:, 0] != zero_home[:, 0]) > 0
    assert np.count_nonzero(candidate[:, 1] != zero_home[:, 1]) > 0
    realized_zero = torch.nn.functional.interpolate(
        torch.from_numpy(zero_home[:, 1]).permute(0, 3, 1, 2).float(),
        size=(PAIR_H, PAIR_W),
        mode="bilinear",
        align_corners=False,
    )
    realized_candidate = torch.nn.functional.interpolate(
        torch.from_numpy(candidate[:, 1]).permute(0, 3, 1, 2).float(),
        size=(PAIR_H, PAIR_W),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.count_nonzero(realized_candidate != realized_zero).item() > 0


def test_depth_contains_continuous_ground_and_movable_contact_stratum() -> None:
    movable = np.zeros((PAIR_H, PAIR_W), dtype=np.bool_)
    movable[250:310, 220:290] = True
    depth = ground_and_movable_depth(movable)
    assert depth.shape == (PAIR_H, PAIR_W)
    assert depth.dtype == np.float32
    assert depth[PAIR_H // 2 - 1, 0] == pytest.approx(120.0)
    assert depth[260, 100] != depth[300, 100]
    assert np.unique(depth[movable]).size == 1
    assert depth[260, 250] != depth[260, 100]


def test_warp_refuses_nonfinite_realization_without_numpy_warnings() -> None:
    frame = torch.zeros((1, 3, PAIR_H, PAIR_W), dtype=torch.float32)
    with pytest.raises(PC1PoseStreamError, match="nonfinite"):
        _warp_scorer_frame(
            frame,
            xi=np.full(6, np.finfo(np.float64).max),
            depth=np.ones((PAIR_H, PAIR_W), dtype=np.float32),
            torch_module=torch,
        )


def test_solved_plane_target_is_exact_yuv6_shape_and_parent_derived() -> None:
    parent = _gradient_parent()
    target = solved_plane_yuv6_target(parent, torch_module=torch)
    assert tuple(target.shape) == (1, 2, 6, PAIR_H // 2, PAIR_W // 2)
    mutated = parent.copy()
    mutated[:, 1, 400:500, 500:600, 0] ^= np.uint8(31)
    mutated_target = solved_plane_yuv6_target(mutated, torch_module=torch)
    assert torch.count_nonzero(mutated_target[:, 1] != target[:, 1]).item() > 0
    assert torch.equal(mutated_target[:, 0], target[:, 0])


def test_active_tube_quadratic_consumes_landed_factor_geometry() -> None:
    candidate = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    center = np.zeros((1, 6), dtype=np.float64)
    factor = np.eye(6, dtype=np.float64)[None, ...] / np.sqrt(6.0)
    result = active_tube_quadratic(
        candidate_pose6=candidate,
        centers=center,
        low_rank_factors=factor,
    )
    assert result[0] == pytest.approx(np.mean(np.square(candidate)))
    assert ms4d_pose_quadratic(candidate[0], center[0], factor[0]) == pytest.approx(result[0])


def test_complete_archive_parseback_and_unique_owner() -> None:
    _, packet = _packet()
    parent = b"exact-parent-archive-bytes"
    import hashlib

    parent_sha = hashlib.sha256(parent).hexdigest()
    archive = build_counted_composition_archive(
        parent_archive=parent,
        parent_sha256=parent_sha,
        packet=packet,
    )
    parsed_parent, parsed_packet, manifest = parse_counted_composition_archive(archive)
    assert parsed_parent == parent
    assert serialize_pc1_packet(parsed_packet) == serialize_pc1_packet(packet)
    assert manifest["owner"] == "ddm.pc1.pose_stream"
    assert verify_unique_output_effect_owners(output_effect_owners())
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        assert handle.namelist() == [
            "manifest/pc1.json",
            "pose/pc1.ddp",
            "parent/ws1.zip",
        ]


def test_admission_and_non_telescoping_score_laws_fail_closed() -> None:
    delta = non_telescoping_conditional_delta_s(
        parent_dseg=0.03,
        parent_dpose=10.0,
        candidate_dseg=0.02,
        candidate_dpose=9.0,
        parent_bytes=100,
        candidate_bytes=200,
    )
    assert np.isfinite(delta)
    assert admission_fence(
        exact_parseback=True,
        inactive_byte_identity=True,
        nonzero_composite_r_support=True,
        both_parents_exact_replay=True,
        unique_effect_owner=True,
        n600_batch32_measured=True,
        descent_was_run=False,
        conditional_delta_s=delta,
    ) == (True, False)
