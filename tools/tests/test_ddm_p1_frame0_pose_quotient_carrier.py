# SPDX-License-Identifier: MIT
"""Regression tests for the counted P1 frame-0 PoseNet quotient carrier."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.ddm_p1_frame0_pose_quotient_carrier_20260725 import (
    DELEGATED_TARGET_D_POSE,
    canonical_rank_law,
    matched_control_fence,
    pose_targeted_actuator,
    reach_curve_disposition,
)
from tac.optimization.ddm_p1_frame0_pose_quotient_carrier import (
    CAMERA_H,
    CAMERA_W,
    GRID_H,
    GRID_W,
    PC1Frame0QuotientError,
    build_counted_composition_archive,
    make_packet,
    packet_typed_stream_tags,
    parse_counted_composition_archive,
    parse_packet,
    receive_frame0_quotient,
    seeded_matched_control_basis,
    serialize_packet,
)
from tools.run_ddm_p1_frame0_pose_quotient_carrier import (
    P1ConfigV1,
    _linear_coefficients,
    _quantize_basis,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _packet(*, treatment: bool, rank: int = 2, active: bool = True):
    basis = seeded_matched_control_basis(seed=41 if treatment else 73, rank=rank)
    coefficients = np.zeros((600, rank), dtype=np.int16)
    coefficients[0, :] = np.arange(1, rank + 1, dtype=np.int16) * 64
    return make_packet(
        treatment=treatment,
        rank=rank,
        q_basis=basis,
        q_coefficients=coefficients,
        active=active,
    )


def _parent(batch: int = 1) -> np.ndarray:
    x = np.arange(CAMERA_W, dtype=np.uint16)[None, :]
    y = np.arange(CAMERA_H, dtype=np.uint16)[:, None]
    frame0 = np.stack(
        (
            np.broadcast_to(x % 256, (CAMERA_H, CAMERA_W)),
            np.broadcast_to(y % 256, (CAMERA_H, CAMERA_W)),
            (x + y) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    frame1 = np.flip(frame0, axis=1).copy()
    return np.repeat(np.stack((frame0, frame1), axis=0)[None, ...], batch, axis=0)


def test_packet_roundtrip_is_exact_and_corruption_fails_closed() -> None:
    packet = _packet(treatment=True)
    payload = serialize_packet(packet)
    parsed = parse_packet(payload)
    assert serialize_packet(parsed) == payload
    assert parsed.q_basis.flags.writeable is False
    with pytest.raises(PC1Frame0QuotientError):
        parse_packet(payload[:-1])


def test_rank6_carrier_is_below_cap_and_treatment_control_bytes_match() -> None:
    treatment = _packet(treatment=True, rank=6)
    control = _packet(treatment=False, rank=6)
    assert len(serialize_packet(treatment)) == len(serialize_packet(control))
    assert len(serialize_packet(treatment)) == treatment.packet_bytes
    assert treatment.packet_bytes <= 30_000


def test_active_receiver_changes_only_frame0_and_inactive_is_exact() -> None:
    parent = _parent()
    active = receive_frame0_quotient(
        parent_camera=parent,
        packet=_packet(treatment=True),
        pair_ids=[0],
    )
    inactive = receive_frame0_quotient(
        parent_camera=parent,
        packet=_packet(treatment=True, active=False),
        pair_ids=[0],
    )
    assert np.count_nonzero(active[:, 0] != parent[:, 0]) > 0
    assert np.array_equal(active[:, 1], parent[:, 1])
    assert np.array_equal(inactive, parent)


def test_counted_archive_parseback_and_exact_matched_budget() -> None:
    parent = b"custodied-g4-parent"
    parent_sha = hashlib.sha256(parent).hexdigest()
    treatment = _packet(treatment=True, rank=3)
    control = _packet(treatment=False, rank=3)
    treatment_archive = build_counted_composition_archive(
        parent_archive=parent,
        parent_sha256=parent_sha,
        packet=treatment,
    )
    control_archive = build_counted_composition_archive(
        parent_archive=parent,
        parent_sha256=parent_sha,
        packet=control,
    )
    parsed_parent, parsed_packet, manifest = parse_counted_composition_archive(treatment_archive)
    assert parsed_parent == parent
    assert serialize_packet(parsed_packet) == serialize_packet(treatment)
    assert manifest["frame0_only"] is True
    assert len(treatment_archive) == len(control_archive)


def test_typed_stream_bytes_reconcile_to_packet() -> None:
    packet = _packet(treatment=True, rank=4)
    tags = packet_typed_stream_tags(packet)
    assert [tag.type.value for tag in tags] == ["SKELETON", "FIBER"]
    assert sum(tag.counted_bytes for tag in tags) == packet.packet_bytes
    assert all(tag.free_receiver_code for tag in tags)


def test_targeted_actuator_and_rank_law_are_not_generic_spatial_menu() -> None:
    jacobian = np.zeros((6, 12), dtype=np.float64)
    jacobian[:, :6] = np.eye(6)
    residual = np.arange(1.0, 7.0)
    actuator = pose_targeted_actuator(jacobian, residual, ridge=1.0e-6)
    assert actuator.shape == (12,)
    assert np.allclose(actuator[:6], residual / (1.0 + 1.0e-6))
    law = canonical_rank_law(
        eigenvalues=(8.0, 4.0, 2.0, 1.0, 0.0, 0.0),
        baseline_d_pose=4.0e-4,
        target_d_pose=DELEGATED_TARGET_D_POSE,
    )
    assert law["selected_rank"] == 3
    assert law["rows"][2]["predicted_linearized_d_pose"] <= DELEGATED_TARGET_D_POSE


def test_falsifier_requires_five_rows_and_is_formulation_scoped() -> None:
    rows = [
        {"rank": rank, "d_pose": 0.1 / rank, "carrier_bytes": 2_000 * rank}
        for rank in range(1, 6)
    ]
    verdict, scope = reach_curve_disposition(rows)
    assert verdict == "P1_SHARED_LOW_RANK_FRAME0_ACTUATOR_FORMULATION_BLOCKED"
    assert scope.startswith("FORMULATION:")
    with pytest.raises(ValueError, match="at least five"):
        reach_curve_disposition(rows[:4])


def test_matched_control_fence_requires_same_exact_frame1_and_packet_budget() -> None:
    digest = "a" * 64
    assert matched_control_fence(
        treatment_packet_bytes=20_000,
        control_packet_bytes=20_000,
        treatment_frame1_sha256=digest,
        control_frame1_sha256=digest,
        parent_frame1_sha256=digest,
        same_rank=True,
        same_precision=True,
        same_solver=True,
    )
    assert not matched_control_fence(
        treatment_packet_bytes=20_000,
        control_packet_bytes=20_001,
        treatment_frame1_sha256=digest,
        control_frame1_sha256=digest,
        parent_frame1_sha256=digest,
        same_rank=True,
        same_precision=True,
        same_solver=True,
    )


def test_sealed_chart_geometry_is_24_by_32() -> None:
    packet = _packet(treatment=True)
    assert packet.q_basis.shape[2:] == (GRID_H, GRID_W)


def test_typed_runner_config_is_canonical_and_seals_solver_budget() -> None:
    config, digest = P1ConfigV1.from_path(
        REPO_ROOT / ".omx/research/configs/ddm_p1_frame0_pose_quotient_carrier_20260725.json"
    )
    assert len(digest) == 64
    assert config.gauss_newton_iterations == 4
    assert config.gauss_newton_max_step == 4096.0
    assert config.measurement.scorer_batch_size == 32


def test_basis_quantization_is_deterministic_and_sign_canonical() -> None:
    coordinates = 3 * GRID_H * GRID_W
    basis = np.zeros((6, coordinates), dtype=np.float64)
    for rank in range(6):
        basis[rank, rank] = -float(rank + 1)
        basis[rank, rank + 6] = 0.5
    first = _quantize_basis(basis)
    second = _quantize_basis(basis)
    assert np.array_equal(first, second)
    assert first.shape == (6, 3, GRID_H, GRID_W)
    assert np.all(first.reshape(6, -1)[np.arange(6), np.arange(6)] == 127)


def test_linear_coefficient_solve_consumes_receiver_scaled_basis() -> None:
    coordinates = 3 * GRID_H * GRID_W
    jacobian = np.zeros((1, 6, coordinates), dtype=np.float32)
    jacobian[0, 0, 0] = 1.0
    residual = np.zeros((1, 6), dtype=np.float64)
    residual[0, 0] = 1.0
    basis = np.zeros((1, 3, GRID_H, GRID_W), dtype=np.int8)
    basis.reshape(1, -1)[0, 0] = 1
    coefficients = _linear_coefficients(
        jacobian=jacobian,
        target_residual=residual,
        q_basis=basis,
        rank=1,
        ridge=1.0e-12,
    )
    assert coefficients.tolist() == [[256]]
