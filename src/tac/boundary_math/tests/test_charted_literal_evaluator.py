# SPDX-License-Identifier: MIT
"""Sealed charted-evaluation core for the literal polar curvelet (p0_497 gap (a)).

Covers: BasisProgramConfig chart-eval custody (semantics + fine factor), the local
verbatim precompose parity, the sealed ``charted_grid_bilinear_v1`` evaluator
(identity-pair exactness, grid-node exact gather, clamp determinism), the
``GroundFrameChart.build_from_xi`` counted-receiver-program entry, and the trainer
source gates for the counted-chart-payload wiring. All tiny grids — the evaluator's
ACCURACY at the real (384, 512) grid is measured in
``.omx/research/charted_grid_bilinear_v1_receipt_20260717.json`` (advisory receipt),
not asserted here; these tests pin BEHAVIOR/semantics, not tolerance vibes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tac.boundary_math.localized_basis_frames as lbf
from tac.boundary_math.ground_frame_chart import (
    ChartCalibration,
    GroundFrameChart,
    precompose_coords_numpy,
)
from tac.boundary_math.xi_pose_coder import dequantize_xi, quantize_xi

_REPO = Path(__file__).resolve().parents[4]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _enabled_program(fine_factor: int = 2) -> lbf.BasisProgramConfig:
    return lbf.literal_basis_program_config(
        chart_enabled=True,
        chart_pose_dependency="counted_chart_payload",
        chart_eval_semantics=lbf.CHART_EVAL_SEMANTICS_BILINEAR,
        chart_fine_factor=fine_factor,
    )


# --------------------------------------------------------------------------- #
# config custody
# --------------------------------------------------------------------------- #
def test_enabled_chart_requires_sealed_semantics() -> None:
    with pytest.raises(ValueError, match="sealed fast evaluator"):
        lbf.literal_basis_program_config(
            chart_enabled=True, chart_pose_dependency="counted_chart_payload"
        )
    with pytest.raises(ValueError, match="sealed fast evaluator"):
        lbf.literal_basis_program_config(
            chart_enabled=True,
            chart_pose_dependency="counted_chart_payload",
            chart_eval_semantics="some_other_semantics",
        )


def test_enabled_chart_fine_factor_must_be_positive_int() -> None:
    with pytest.raises(ValueError, match="chart_fine_factor"):
        _enabled_program(fine_factor=0)
    with pytest.raises(ValueError, match="chart_fine_factor"):
        lbf.literal_basis_program_config(
            chart_enabled=True,
            chart_pose_dependency="counted_chart_payload",
            chart_eval_semantics=lbf.CHART_EVAL_SEMANTICS_BILINEAR,
            chart_fine_factor=True,  # bools are not admissible factors
        )
    assert _enabled_program(fine_factor=1).chart_fine_factor == 1


def test_disabled_chart_pins_inert_new_fields() -> None:
    with pytest.raises(ValueError, match="chart_eval_semantics='none'"):
        lbf.literal_basis_program_config(
            chart_eval_semantics=lbf.CHART_EVAL_SEMANTICS_BILINEAR
        )
    with pytest.raises(ValueError, match="inert default chart_fine_factor"):
        lbf.literal_basis_program_config(chart_fine_factor=4)
    assert lbf.literal_basis_program_config().chart_eval_semantics == "none"


def test_new_fields_enter_the_canonical_hash() -> None:
    a = _enabled_program(fine_factor=2)
    b = _enabled_program(fine_factor=4)
    assert a.canonical_sha256() != b.canonical_sha256()
    assert "chart_eval_semantics" in a.to_dict()
    assert "chart_fine_factor" in a.to_dict()


def test_from_dict_backcompat_disabled_defaults_enabled_refuses() -> None:
    disabled = lbf.literal_basis_program_config()
    legacy_disabled = disabled.to_dict()
    del legacy_disabled["chart_eval_semantics"]
    del legacy_disabled["chart_fine_factor"]
    restored = lbf.BasisProgramConfig.from_dict(legacy_disabled)
    assert restored.canonical_sha256() == disabled.canonical_sha256()

    enabled = _enabled_program()
    legacy_enabled = enabled.to_dict()
    del legacy_enabled["chart_eval_semantics"]
    del legacy_enabled["chart_fine_factor"]
    # legacy ENABLED chart cannot silently acquire the new sealed semantics
    with pytest.raises(ValueError, match="sealed fast evaluator"):
        lbf.BasisProgramConfig.from_dict(legacy_enabled)

    with pytest.raises(ValueError, match="fields drift"):
        lbf.BasisProgramConfig.from_dict({**disabled.to_dict(), "extra_key": 1})
    truncated = disabled.to_dict()
    del truncated["family"]
    with pytest.raises(ValueError, match="fields drift"):
        lbf.BasisProgramConfig.from_dict(truncated)


# --------------------------------------------------------------------------- #
# local verbatim precompose parity (the receiver cannot import ground_frame_chart)
# --------------------------------------------------------------------------- #
def test_local_precompose_bit_parity_with_ground_frame_chart() -> None:
    rng = np.random.default_rng(7)
    for _ in range(8):
        h_mat = np.eye(3) + rng.normal(scale=0.08, size=(3, 3))
        pts = rng.uniform(-1.4, 1.4, size=(513, 2)).astype(np.float32)
        ours = lbf._precompose_coords_numpy(pts, h_mat)
        theirs = precompose_coords_numpy(pts, h_mat)
        assert np.array_equal(ours, theirs)


def test_local_precompose_identity_fast_path_returns_input_unchanged() -> None:
    pts = np.asarray([[0.25, -0.5], [1.0, 1.0]], dtype=np.float32)
    out = lbf._precompose_coords_numpy(pts, np.eye(3))
    assert out is pts or np.shares_memory(out, pts) or np.array_equal(out, pts)
    assert np.array_equal(out, precompose_coords_numpy(pts, np.eye(3)))


def test_local_precompose_z_guard_is_finite() -> None:
    h_mat = np.eye(3)
    h_mat[2, 0] = 1.0  # z crosses zero inside the box
    pts = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = lbf._precompose_coords_numpy(pts, h_mat)
    assert np.all(np.isfinite(out))
    assert np.array_equal(out, precompose_coords_numpy(pts, h_mat))


# --------------------------------------------------------------------------- #
# sealed bilinear evaluator behavior
# --------------------------------------------------------------------------- #
def test_identity_pair_is_the_exact_uncharted_program() -> None:
    prog = _enabled_program()
    h, w = 9, 11
    cache = lbf.charted_fine_feats_cache_numpy(prog, h, w)
    out = lbf.charted_pair_feats_numpy(prog, h, w, np.eye(3), cache)
    exact = lbf.basis_features_numpy(lbf.inclusive_grid_coords(h, w))
    assert np.array_equal(out, exact)


def test_bilinear_at_fine_grid_nodes_reproduces_the_table_to_fp32_eps() -> None:
    # Sampling AT the fine grid nodes recovers the table up to fp32 coordinate
    # rounding (u = (x+1)·0.5·(Wf−1) lands within one ulp of the integer node, so
    # the interpolation weight is O(eps), not exactly zero). MEASURED: max abs
    # deviation 2.9e-6 on a table with amplitude ~6.3. The bit-exact contract of
    # the sealed program is trainer==receiver (same function, same inputs), which
    # the determinism test below pins — not node-exact gather.
    prog = _enabled_program()
    h, w = 7, 9
    fine_h, fine_w = 2 * h, 2 * w
    cache = lbf.charted_fine_feats_cache_numpy(prog, h, w)
    fine_coords = lbf.inclusive_grid_coords(fine_h, fine_w)
    out = lbf.charted_grid_bilinear_features_numpy(cache, fine_h, fine_w, fine_coords)
    assert out.shape == cache.shape
    assert float(np.max(np.abs(out.astype(np.float64) - cache.astype(np.float64)))) < 1e-5


def test_out_of_box_points_clamp_to_border_values() -> None:
    prog = _enabled_program()
    h, w = 7, 9
    fine_h, fine_w = 2 * h, 2 * w
    cache = lbf.charted_fine_feats_cache_numpy(prog, h, w)
    grid = cache.reshape(fine_h, fine_w, lbf.FEATURE_WIDTH)
    probe = np.asarray(
        [[-9.0, -9.0], [9.0, -9.0], [-9.0, 9.0], [9.0, 9.0]], dtype=np.float32
    )
    out = lbf.charted_grid_bilinear_features_numpy(cache, fine_h, fine_w, probe)
    corners = np.stack(
        [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]], axis=0
    )
    assert np.array_equal(out, corners)
    assert np.all(np.isfinite(out))


def test_charted_pair_feats_is_deterministic_and_shape_checked() -> None:
    prog = _enabled_program()
    h, w = 9, 11
    cache = lbf.charted_fine_feats_cache_numpy(prog, h, w)
    h_mat = np.eye(3)
    h_mat[0, 2] = 0.05
    h_mat[2, 0] = 0.01
    a = lbf.charted_pair_feats_numpy(prog, h, w, h_mat, cache)
    b = lbf.charted_pair_feats_numpy(prog, h, w, h_mat, cache)
    assert np.array_equal(a, b)
    assert a.shape == (h * w, lbf.FEATURE_WIDTH)
    assert a.dtype == np.float32
    with pytest.raises(ValueError, match="fine_feats_cache"):
        lbf.charted_pair_feats_numpy(prog, h, w, h_mat, cache[:-1])
    disabled = lbf.literal_basis_program_config()
    with pytest.raises(ValueError, match="enabled chart"):
        lbf.charted_pair_feats_numpy(disabled, h, w, h_mat, cache)
    with pytest.raises(ValueError, match="enabled chart"):
        lbf.charted_fine_feats_cache_numpy(disabled, h, w)


# --------------------------------------------------------------------------- #
# build_from_xi: the counted receiver program entry
# --------------------------------------------------------------------------- #
def _synthetic_pose_table(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(scale=0.05, size=(n, 6))


def test_build_from_xi_equals_build_field_for_field() -> None:
    poses = _synthetic_pose_table()
    calib = ChartCalibration()
    a = GroundFrameChart.build(poses, ref_pair=0, calib=calib, grid_hw=(9, 11))
    b = GroundFrameChart.build_from_xi(poses, ref_pair=0, calib=calib, grid_hw=(9, 11))
    assert np.array_equal(a.H_chart_norm, b.H_chart_norm)
    assert np.array_equal(a.H_fwd_pix, b.H_fwd_pix)
    assert a.ref_pair == b.ref_pair
    assert a.regime == b.regime
    assert a.grid_hw == b.grid_hw
    assert b.provenance["pose_source"] == "counted_receiver_payload_dequantized"
    assert "build_from_xi" in b.provenance["builder"]


def test_quantize_dequantize_roundtrip_changes_the_homographies() -> None:
    # GT float poses vs the receiver's dequantized table is a REAL distinction the
    # trainer must honor: silently charting from GT would be a train/decode mismatch.
    poses = _synthetic_pose_table()
    q, scales = quantize_xi(poses)
    poses_dq = dequantize_xi(q, scales)
    assert not np.array_equal(poses, poses_dq)
    gt_chart = GroundFrameChart.build(poses, ref_pair=0, grid_hw=(9, 11))
    dq_chart = GroundFrameChart.build_from_xi(poses_dq, ref_pair=0, grid_hw=(9, 11))
    assert not np.array_equal(gt_chart.H_chart_norm, dq_chart.H_chart_norm)
    # and the dequantized build is reproducible bit-for-bit (receiver == trainer)
    dq_chart2 = GroundFrameChart.build_from_xi(poses_dq, ref_pair=0, grid_hw=(9, 11))
    assert np.array_equal(dq_chart.H_chart_norm, dq_chart2.H_chart_norm)


def test_build_from_xi_refuses_malformed_tables() -> None:
    with pytest.raises(Exception, match="P, 6"):
        GroundFrameChart.build_from_xi(np.zeros((4, 3)), ref_pair=0, grid_hw=(9, 11))


# --------------------------------------------------------------------------- #
# trainer source gates (counted-chart-payload wiring; style mirrors existing gates)
# --------------------------------------------------------------------------- #
def test_trainer_wires_counted_chart_payload_and_sealed_evaluator() -> None:
    source = _TRAINER.read_text(encoding="utf-8")
    assert '"counted_chart_payload" if chart_enabled' in source
    assert "--literal-chart-fine-factor" in source
    assert "GroundFrameChart.build_from_xi" in source
    assert "charted_pair_feats_numpy(" in source
    assert "charted_fine_feats_cache_numpy(" in source
    assert '__chart_pose_q"' in source
    assert '__chart_pose_scales"' in source
    # the quantize->dequantize round trip feeds the chart (never raw GT for literal)
    assert "dequantize_xi(_chart_pose_q, _chart_pose_scales)" in source
    # margin compander stays fail-closed for the literal sealed program
    assert "fail-closed for" in source and "literal_polar_curvelet" in source
