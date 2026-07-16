# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tac.boundary_math.curvelet_placement import (
    NUMPY_AUTHORITY_DTYPE,
    NUMPY_MLX_PARITY_ATOL,
    CurveletPlacementError,
    TaperFoldError,
    apply_orientation_gates_numpy,
    array_sha256,
    fold_taper_into_in_proj_numpy,
    native_orientation_fixed_point_numpy,
    normal_covectors_from_argmax_numpy,
    orientation_gates_numpy,
    orientation_metadata_from_atom_specs,
    projective_jacobian_numpy,
    projective_map_numpy,
    transform_normal_covector_numpy,
    transform_tangent_numpy,
    verify_deploy_taper_fold_receipt,
    verify_taper_fold_numpy,
    verify_taper_fold_receipt,
    verify_taper_fold_receipt_self_consistency,
)
from tac.boundary_math.localized_basis_frames import literal_polar_curvelet_atom_specs


def test_identity_chart_preserves_tangent_and_normal():
    tangent = np.array([[1.0, 2.0], [-3.0, 4.0]], np.float32)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    identity = np.eye(3, dtype=np.float32)
    points = np.array([[0.0, 0.0], [0.2, -0.7]], np.float32)
    jacobian = projective_jacobian_numpy(points, identity)
    expected_t = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)
    expected_n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    assert np.array_equal(projective_map_numpy(points, identity), points)
    assert np.allclose(transform_tangent_numpy(tangent, jacobian), expected_t, atol=1e-7)
    assert np.allclose(transform_normal_covector_numpy(normal, jacobian), expected_n, atol=1e-7)


def test_nontrivial_projective_chart_vector_covector_covariance():
    homography = np.array(
        [[1.2, 0.35, -0.1], [-0.2, 0.9, 0.05], [0.11, -0.07, 1.0]], np.float64
    )
    point = np.array([0.25, -0.4], np.float64)
    tangent = np.array([0.8, 0.6], np.float64)
    normal = np.array([-0.6, 0.8], np.float64)
    jacobian = projective_jacobian_numpy(point, homography)
    tangent_y = transform_tangent_numpy(tangent, jacobian)
    normal_y = transform_normal_covector_numpy(normal, jacobian)
    # The tangent-normal annihilation law is coordinate invariant.
    assert abs(float(np.dot(tangent_y, normal_y))) < 2e-7
    # Push-forward agrees with a finite difference of the actual chart.
    step = 1e-4
    finite_difference = (
        projective_map_numpy(point + step * tangent, homography)
        - projective_map_numpy(point - step * tangent, homography)
    ) / (2.0 * step)
    finite_difference /= np.linalg.norm(finite_difference)
    assert np.allclose(tangent_y, finite_difference, atol=5e-4)
    # A shear makes the covector law observably different from the vector law.
    wrong_normal = transform_tangent_numpy(normal, jacobian)
    assert not np.allclose(normal_y, wrong_normal, atol=1e-3)


def test_batched_projective_jacobian_broadcasts_over_point_grid():
    points = np.zeros((2, 3, 2), np.float32)
    homographies = np.stack(
        (np.eye(3), np.array([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]))
    )
    jacobian = projective_jacobian_numpy(points, homographies)
    assert jacobian.shape == (2, 3, 2, 2)
    assert np.array_equal(jacobian[0], np.broadcast_to(np.eye(2), (3, 2, 2)))
    assert np.array_equal(
        jacobian[1], np.broadcast_to(np.diag([2.0, 0.5]), (3, 2, 2))
    )


def test_numpy_authority_casts_before_arithmetic_and_returns_fp32():
    points = np.array([[0.125, -0.375]], np.float64)
    homography = np.array(
        [[1.0, 0.2, 0.1], [-0.1, 0.9, 0.2], [0.03, -0.04, 1.0]], np.float64
    )
    mapped = projective_map_numpy(points, homography)
    jacobian = projective_jacobian_numpy(points, homography)
    tangent = transform_tangent_numpy(np.array([[1.0, 2.0]], np.float64), jacobian)
    normal = transform_normal_covector_numpy(np.array([[-2.0, 1.0]], np.float64), jacobian)
    placed = apply_orientation_gates_numpy(
        np.ones((1, 2), np.float64), np.ones((1, 2), np.float64)
    )
    assert NUMPY_AUTHORITY_DTYPE is np.float32
    assert mapped.dtype == jacobian.dtype == tangent.dtype == normal.dtype == placed.dtype == np.float32


def _gate_metadata() -> tuple[np.ndarray, np.ndarray]:
    scales = np.array([-1, -1, 0, 0, 0, 0, 2, 2, 2, 2], np.int64)
    angles = np.array(
        [np.nan, np.nan, 0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4,
         0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        np.float32,
    )
    return scales, angles


def test_atom_spec_adapter_preserves_scaling_identity_and_directional_scales():
    specs = (
        SimpleNamespace(column=0, kind="scaling", scale=None, theta=None),
        SimpleNamespace(column=1, kind="directional", scale=0, theta=0.25),
        SimpleNamespace(column=2, kind="directional", scale=2, theta=0.75),
    )
    scales, angles = orientation_metadata_from_atom_specs(specs)
    assert np.array_equal(scales, np.array([-1, 0, 2]))
    assert np.isnan(angles[0])
    assert np.array_equal(angles[1:], np.array([0.25, 0.75], np.float32))


def test_native_orientation_gates_are_per_scale_l2_and_scaling_identity():
    scales, angles = _gate_metadata()
    normals = np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]], np.float32)
    gates = orientation_gates_numpy(normals, scales, angles, kappa=2.5)
    assert gates.shape == (3, 10)
    assert np.array_equal(gates[:, :2], np.ones((3, 2), np.float32))
    for row in range(3):
        for scale in (0, 2):
            directional = gates[row, scales == scale]
            assert abs(float(np.sum(directional * directional)) - 1.0) < 2e-7
    # Normal along theta=0 prefers theta=0; projective n and -n give the same gate.
    assert gates[0, 2] > gates[0, 3] > gates[0, 4]
    assert np.array_equal(gates[0], gates[2])


def test_orientation_gate_application_changes_only_directional_columns():
    scales, angles = _gate_metadata()
    features = np.arange(30, dtype=np.float32).reshape(3, 10)
    gates = orientation_gates_numpy(np.array([[1.0, 0.0]], np.float32), scales, angles, kappa=1.5)
    placed = apply_orientation_gates_numpy(features, gates)
    assert np.array_equal(placed[:, :2], features[:, :2])
    assert not np.array_equal(placed[:, 2:], features[:, 2:])


def test_native_orientation_fixed_point_is_decoder_only_and_same_width():
    scales, angles = orientation_metadata_from_atom_specs(literal_polar_curvelet_atom_specs())
    base = np.ones((9, 80), np.float32)
    calls: list[np.ndarray] = []

    def decode(features: np.ndarray) -> np.ndarray:
        calls.append(features.copy())
        return np.asarray([[0, 0, 1], [0, 1, 1], [2, 2, 1]], dtype=np.int64)

    placed, gates, receipt = native_orientation_fixed_point_numpy(
        base, decode, scales, angles, kappa=1.5, iteration_cap=4
    )
    assert receipt.converged
    assert receipt.iterations_executed == 2
    assert placed.shape == gates.shape == base.shape
    np.testing.assert_array_equal(gates[:, :4], 1.0)
    assert not np.array_equal(gates[:, 4:], 1.0)
    assert receipt.argmax_sha256 and receipt.gate_sha256
    assert len(calls) == 2


def test_argmax_normal_covectors_are_deterministic_and_nonzero():
    labels = np.asarray([[0, 0, 1], [0, 1, 1], [2, 2, 1]], dtype=np.int64)
    first = normal_covectors_from_argmax_numpy(labels)
    second = normal_covectors_from_argmax_numpy(labels.copy())
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_orientation_gates_fail_closed_on_bad_normals_or_metadata():
    scales, angles = _gate_metadata()
    with pytest.raises(CurveletPlacementError, match="zero or singular"):
        orientation_gates_numpy(np.zeros((1, 2)), scales, angles, kappa=1.0)
    with pytest.raises(CurveletPlacementError, match="non-negative"):
        orientation_gates_numpy(np.ones((1, 2)), scales, angles, kappa=-0.1)
    with pytest.raises(CurveletPlacementError, match="angles"):
        orientation_gates_numpy(np.ones((1, 2)), scales, angles[:-1], kappa=1.0)


def test_taper_fold_exact_algebra_receipt_and_double_fold_refusal():
    rng = np.random.default_rng(17)
    weight = rng.standard_normal((7, 10), dtype=np.float32)
    taper = np.linspace(0.2, 1.8, 10, dtype=np.float32)
    features = rng.standard_normal((23, 10), dtype=np.float32)
    source_hash = array_sha256(weight)
    folded, receipt = fold_taper_into_in_proj_numpy(
        weight, taper, expected_source_sha256=source_hash
    )
    assert np.array_equal(folded, weight * taper[None, :])
    assert receipt.source_weight_sha256 == source_hash
    assert receipt.taper_sha256 == array_sha256(taper)
    assert receipt.folded_weight_sha256 == array_sha256(folded)
    assert verify_taper_fold_receipt_self_consistency(receipt)
    assert verify_deploy_taper_fold_receipt(
        receipt, source_weight=weight, taper=taper, folded_weight=folded
    )
    parity = verify_taper_fold_numpy(weight, taper, folded, features)
    assert parity.allclose
    assert parity.max_abs_error < 2e-6
    with pytest.raises(TaperFoldError, match="twice"):
        fold_taper_into_in_proj_numpy(folded, taper, existing_receipt=receipt)


def test_taper_fold_rejects_wrong_source_hash_and_tampered_receipt():
    weight = np.ones((3, 4), np.float32)
    taper = np.ones(4, np.float32)
    with pytest.raises(TaperFoldError, match="custody"):
        fold_taper_into_in_proj_numpy(weight, taper, expected_source_sha256="0" * 64)
    _, receipt = fold_taper_into_in_proj_numpy(weight, taper)
    tampered = receipt.to_dict()
    tampered["taper_length"] = 9
    with pytest.raises(TaperFoldError, match="hash mismatch"):
        verify_taper_fold_receipt(tampered)


def test_deploy_taper_receipt_rejects_missing_or_stale_array_bindings():
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    taper = np.array([0.5, 0.75, 1.25, 1.5], np.float32)
    folded, receipt = fold_taper_into_in_proj_numpy(weight, taper)
    with pytest.raises(TaperFoldError, match="requires source"):
        verify_taper_fold_receipt(receipt, require_array_bindings=True)
    stale_source = weight.copy()
    stale_source[0, 0] += 1.0
    with pytest.raises(TaperFoldError, match="source weight"):
        verify_deploy_taper_fold_receipt(
            receipt, source_weight=stale_source, taper=taper, folded_weight=folded
        )
    stale_taper = taper.copy()
    stale_taper[0] += 0.25
    with pytest.raises(TaperFoldError, match="supplied taper"):
        verify_deploy_taper_fold_receipt(
            receipt, source_weight=weight, taper=stale_taper, folded_weight=folded
        )
    stale_folded = folded.copy()
    stale_folded[0, 0] += 0.25
    with pytest.raises(TaperFoldError, match="folded weight"):
        verify_deploy_taper_fold_receipt(
            receipt, source_weight=weight, taper=taper, folded_weight=stale_folded
        )


def test_numpy_mlx_optional_parity():
    if importlib.util.find_spec("mlx.core") is None:
        pytest.skip("MLX is not installed")
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; mx.eval(mx.array([1.0]))"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"MLX runtime unavailable: {probe.stderr.strip()}")
    import mlx.core as mx

    from tac.boundary_math.curvelet_placement import (
        orientation_gates_mlx,
        projective_jacobian_mlx,
        transform_normal_covector_mlx,
        transform_tangent_mlx,
    )

    homography = np.array(
        [[1.1, 0.15, 0.1], [-0.1, 0.95, -0.2], [0.05, -0.03, 1.0]], np.float32
    )
    points = np.array([[0.1, -0.3], [0.7, 0.2]], np.float32)
    tangent = np.array([[1.0, 2.0], [-2.0, 0.5]], np.float32)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    jacobian_np = projective_jacobian_numpy(points, homography)
    points_mx, homography_mx = mx.array(points), mx.array(homography)
    jacobian_mx = projective_jacobian_mlx(points_mx, homography_mx)
    scales, angles = _gate_metadata()
    gates_np = orientation_gates_numpy(normal, scales, angles, kappa=1.25)
    gates_mx = orientation_gates_mlx(mx.array(normal), scales, angles, kappa=1.25)
    mx.eval(jacobian_mx, gates_mx)
    assert np.allclose(
        np.asarray(jacobian_mx),
        jacobian_np,
        atol=NUMPY_MLX_PARITY_ATOL,
        rtol=NUMPY_MLX_PARITY_ATOL,
    )
    assert np.allclose(
        np.asarray(transform_tangent_mlx(mx.array(tangent), jacobian_mx)),
        transform_tangent_numpy(tangent, jacobian_np),
        atol=NUMPY_MLX_PARITY_ATOL,
        rtol=NUMPY_MLX_PARITY_ATOL,
    )
    assert np.allclose(
        np.asarray(transform_normal_covector_mlx(mx.array(normal), jacobian_mx)),
        transform_normal_covector_numpy(normal, jacobian_np),
        atol=NUMPY_MLX_PARITY_ATOL,
        rtol=NUMPY_MLX_PARITY_ATOL,
    )
    assert np.allclose(
        np.asarray(gates_mx),
        gates_np,
        atol=NUMPY_MLX_PARITY_ATOL,
        rtol=NUMPY_MLX_PARITY_ATOL,
    )
