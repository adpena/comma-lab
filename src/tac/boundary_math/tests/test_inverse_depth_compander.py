# SPDX-License-Identifier: MIT
"""Tests for the S1 softened-inverse-depth chart composed with GroundFrameChart."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.ground_frame_chart import GroundFrameChart
from tac.boundary_math.inverse_depth_compander import (
    MARGIN_COMPANDER_RESUME_PREFIX,
    MEASURED_HORIZON_ROW,
    MEASURED_SOFTENING_OFFSET_ROWS,
    InverseDepthCompanderError,
    InverseDepthCompanderProfile,
    MarginCompandedGroundChart,
    chart_rows_to_image_rows_numpy,
    compand_coords_mlx,
    compand_coords_numpy,
    image_rows_to_chart_rows_mlx,
    image_rows_to_chart_rows_numpy,
    uncompand_coords_numpy,
)

REPO = Path(__file__).resolve().parents[4]
SEG_H, SEG_W = 384, 512


def _chart() -> GroundFrameChart:
    poses = np.random.default_rng(7).normal(size=(5, 6)).astype(np.float64)
    poses[:, 0] = np.abs(poses[:, 0]) * 30.0 + 5.0
    return GroundFrameChart.build(poses, ref_pair=0, grid_hw=(SEG_H, SEG_W))


def test_measured_profile_matches_s1_receipt_exactly() -> None:
    receipt = json.loads(
        (REPO / ".omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json").read_text()
    )
    payload = receipt["S1_input_chart"]
    assert payload["horizon_row"] == MEASURED_HORIZON_ROW
    assert (
        payload["shifted_family_fits"]["shifted_projective_inverse_depth"]
        ["fitted_softening_offset_rows"]
        == MEASURED_SOFTENING_OFFSET_ROWS
    )


def test_endpoints_monotonicity_and_inverse_round_trip() -> None:
    profile = InverseDepthCompanderProfile()
    rows = np.linspace(-20.0, 420.0, 20001, dtype=np.float32)
    chart = image_rows_to_chart_rows_numpy(rows, profile)
    restored = chart_rows_to_image_rows_numpy(chart, profile)
    assert image_rows_to_chart_rows_numpy(np.array([174.0], np.float32), profile)[0] == 174.0
    assert image_rows_to_chart_rows_numpy(np.array([383.0], np.float32), profile)[0] == 383.0
    assert np.all(np.diff(chart) > 0.0)
    assert np.max(np.abs(restored - rows)) < 2e-3


def test_density_is_concentrated_near_dash_erasure_band() -> None:
    profile = InverseDepthCompanderProfile()
    rows = np.arange(SEG_H, dtype=np.float32)
    chart = image_rows_to_chart_rows_numpy(rows, profile)
    upper_density = np.diff(chart[175:211]).mean()
    lower_tail_density = np.diff(chart[320:383]).mean()
    assert upper_density > lower_tail_density


def test_normalized_coords_preserve_x_and_round_trip() -> None:
    profile = InverseDepthCompanderProfile()
    xy = np.random.default_rng(3).uniform(-1.2, 1.2, size=(1000, 2)).astype(np.float32)
    companded = compand_coords_numpy(xy, profile)
    restored = uncompand_coords_numpy(companded, profile)
    assert np.array_equal(companded[:, 0], xy[:, 0])
    assert np.array_equal(restored[:, 0], xy[:, 0])
    assert np.max(np.abs(restored[:, 1] - xy[:, 1])) < 2e-5


def test_wrapper_composes_after_projective_ground_chart() -> None:
    base = _chart()
    wrapped = MarginCompandedGroundChart.compose(base)
    xy = np.random.default_rng(4).uniform(-1.0, 1.0, size=(257, 2)).astype(np.float32)
    expected = compand_coords_numpy(base.coords_for_pair_numpy(xy, 3), wrapped.profile)
    assert np.array_equal(wrapped.coords_for_pair_numpy(xy, 3), expected)


def test_resume_state_round_trip_and_mismatch_refusal() -> None:
    wrapped = MarginCompandedGroundChart.compose(_chart())
    state = wrapped.state_arrays(MARGIN_COMPANDER_RESUME_PREFIX)
    cfg = {key: value.item() for key, value in state.items()}
    assert wrapped.restore_from_cfg(MARGIN_COMPANDER_RESUME_PREFIX, cfg)
    cfg[f"{MARGIN_COMPANDER_RESUME_PREFIX}softening_offset_rows"] += 1.0
    with pytest.raises(InverseDepthCompanderError, match="resume identity diverged"):
        wrapped.restore_from_cfg(MARGIN_COMPANDER_RESUME_PREFIX, cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 1},
        {"horizon_row": 383.0},
        {"softening_offset_rows": 0.0},
        {"seed": -1},
    ],
)
def test_invalid_profiles_refuse(kwargs: dict[str, object]) -> None:
    with pytest.raises(InverseDepthCompanderError):
        InverseDepthCompanderProfile(**kwargs)


def test_mlx_cpu_bit_parity() -> None:
    mx = pytest.importorskip("mlx.core")
    profile = InverseDepthCompanderProfile()
    rows = np.random.default_rng(9).uniform(-20, 420, size=4097).astype(np.float32)
    xy = np.random.default_rng(10).uniform(-1.2, 1.2, size=(4097, 2)).astype(np.float32)
    try:
        with mx.stream(mx.cpu):
            rows_mx = image_rows_to_chart_rows_mlx(mx.array(rows), profile)
            xy_mx = compand_coords_mlx(mx.array(xy), profile)
            mx.eval(rows_mx, xy_mx)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("managed session cannot initialize MLX CPU stream")
        raise
    assert np.array_equal(image_rows_to_chart_rows_numpy(rows, profile), np.array(rows_mx))
    assert np.array_equal(compand_coords_numpy(xy, profile), np.array(xy_mx))


def test_mlx_default_device_close() -> None:
    mx = pytest.importorskip("mlx.core")
    profile = InverseDepthCompanderProfile()
    xy = np.random.default_rng(11).uniform(-1, 1, size=(257, 2)).astype(np.float32)
    try:
        got = np.array(compand_coords_mlx(mx.array(xy), profile))
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("managed session cannot initialize MLX default device")
        raise
    assert np.allclose(
        compand_coords_numpy(xy, profile),
        got,
        atol=2e-6,
    )
