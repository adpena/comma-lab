# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.lattice_native_pose_null_realizer import (
    LatticeNativeRealizerError,
    apply_private_delta,
    build_default_operator,
    cvp_integer_realize,
    dykstra_integer_realize,
    extract_private_camera_block,
    pose_constraint_matrix,
    private_block_geometry,
    project_scorer_delta_to_pose_null,
    realize_lattice_native_block,
    uniform_round_baseline,
)


def _target() -> np.ndarray:
    raw = np.asarray(
        [
            [[2.35, -1.20, 3.75], [-1.50, 2.10, -0.40]],
            [[0.80, 1.35, -2.25], [-2.00, -0.75, 1.55]],
        ],
        dtype=np.float64,
    )
    return project_scorer_delta_to_pose_null(raw)


def test_pose_projector_matches_sq1_rank_and_kernel():
    a = pose_constraint_matrix()
    target = _target()
    residual = a @ target.reshape(12)
    assert np.linalg.matrix_rank(a) == 6
    assert float(np.abs(residual).max()) < 1e-10


def test_private_geometry_uses_exact_nonuniform_weights_and_matches_operator_apply():
    op = build_default_operator()
    geom = private_block_geometry(op, 20, 40)
    assert geom.denominator > 1
    assert not np.allclose(geom.weights, 0.25)

    delta = np.zeros((2, 2, 2, 2, 3), dtype=np.int16)
    delta[0, 0, 0, 0, 0] = 7
    delta[0, 0, 1, 1, 1] = -3
    got = apply_private_delta(delta, geom)

    full = np.zeros((op.camera_h, op.camera_w, 3), dtype=np.int16)
    for br in range(2):
        rows = geom.row_indices[br]
        for bc in range(2):
            cols = geom.col_indices[bc]
            full[np.ix_(rows, cols, range(3))] = delta[br, bc]
    applied = op.apply(full)
    expected = applied[geom.scorer_row : geom.scorer_row + 2, geom.scorer_col : geom.scorer_col + 2]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_geometry_refuses_odd_pose_block():
    op = build_default_operator()
    with pytest.raises(LatticeNativeRealizerError, match="aligned"):
        private_block_geometry(op, 21, 40)


def test_dykstra_and_cvp_do_real_work_against_naive_round():
    op = build_default_operator()
    geom = private_block_geometry(op, 22, 42)
    base = np.full((2, 2, 2, 2, 3), 128, dtype=np.uint8)
    target = _target()

    naive = uniform_round_baseline(target, geom, base_block=base)
    dykstra = dykstra_integer_realize(target, geom, base_block=base, iterations=6)
    cvp = cvp_integer_realize(
        target,
        geom,
        base_block=base,
        tap_radius=1,
        max_channel_candidates=5,
        max_pixel_candidates=12,
    )

    assert dykstra.method == "dykstra"
    assert cvp.method == "cvp"
    assert dykstra.diagnostics["history"]
    assert cvp.diagnostics["combinations_evaluated"] > 0
    assert (
        cvp.pose_leakage_sq,
        cvp.seg_discrepancy,
    ) <= (
        naive.pose_leakage_sq,
        naive.seg_discrepancy,
    )


def test_cvp_reports_pruned_scope_when_caps_bind():
    op = build_default_operator()
    geom = private_block_geometry(op, 24, 44)
    base = np.full((2, 2, 2, 2, 3), 128, dtype=np.uint8)
    res = cvp_integer_realize(
        _target(),
        geom,
        base_block=base,
        tap_radius=2,
        max_channel_candidates=2,
        max_pixel_candidates=2,
        max_combinations=8,
    )
    assert res.diagnostics["exact_declared_scope"] is False
    assert "PRUNED" in res.diagnostics["candidate_scope"]


def test_realize_lattice_native_block_returns_three_distinct_arms():
    op = build_default_operator()
    geom = private_block_geometry(op, 26, 46)
    base = np.full((2, 2, 2, 2, 3), 128, dtype=np.uint8)
    results = realize_lattice_native_block(_target(), geom, base_block=base)
    assert set(results) == {"naive", "dykstra", "cvp"}
    assert results["naive"].method == "uniform_round"
    assert results["dykstra"].method == "dykstra"
    assert results["cvp"].method == "cvp"


def test_extract_private_camera_block_shape_and_bounds():
    op = build_default_operator()
    geom = private_block_geometry(op, 28, 48)
    frame = np.arange(op.camera_h * op.camera_w * 3, dtype=np.uint32).reshape(
        op.camera_h, op.camera_w, 3
    ) % 256
    block = extract_private_camera_block(frame.astype(np.uint8), geom)
    assert block.shape == (2, 2, 2, 2, 3)
    assert block.dtype == np.uint8
