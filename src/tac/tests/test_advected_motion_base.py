# SPDX-License-Identifier: MIT
"""Focused tests for the counted-xi advected motion base."""

from __future__ import annotations

import numpy as np
import pytest

import tools.measure_predict_project_receiver as measurement_tool
from tac.boundary_math import warp_real_luma_frame0 as g1_warp
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.optimization.predict_project_receiver import (
    PredictProjectReceiverError,
    advect_motion_base,
    counted_planar_xi_series,
)
from tac.optimization.predict_project_schema import build_minimal_constraint_seed


def _seed() -> dict:
    return build_minimal_constraint_seed(
        bytes([0, 1, 2, 3, 4, 0]),
        scorer_height=2,
        scorer_width=3,
        camera_height=4,
        camera_width=6,
    )


def test_counted_planar_trajectory_embeds_without_added_bytes() -> None:
    seed = _seed()
    seed["trajectory"]["controls"][0].update(tx_q=256, ty_q=-512, yaw_q=1_048_576)
    seed["trajectory"]["controls"][1].update(tx_q=256, ty_q=-512, yaw_q=1_048_576)
    xi, custody = counted_planar_xi_series(seed, pair_start=0, pair_end=2)
    assert np.array_equal(xi, np.array([[1.0, 0.0, -2.0, 0.0, 1.0, 0.0]] * 2))
    assert custody["additional_video_derived_bytes"] == 0
    assert custody["decoder_scorer_invocations"] == 0
    assert custody["source_section_raw_bytes"] > 0
    assert len(custody["source_section_sha256"]) == 64


def test_zero_xi_advects_rgb_and_scene_chart_bit_exactly() -> None:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    cells = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    geom = GroundHomographyGeom.eon(native_hw=(2, 3), pitch=0.0)
    result = advect_motion_base(rgb, cells, np.zeros(6), geom)
    assert np.array_equal(result["frame1_base"], rgb)
    assert np.array_equal(result["frame1_cells"], cells)
    assert np.array_equal(result["ground_mask"], np.isin(cells, (0, 1, 2)))
    assert result["additional_video_derived_bytes"] == 0
    assert result["decoder_scorer_invocations"] == 0


def test_advected_base_is_deterministic_and_fails_closed_on_bad_xi() -> None:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    cells = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    geom = GroundHomographyGeom.eon(native_hw=(2, 3), pitch=-0.05)
    xi = np.array([0.01, 0.0, -0.02, 0.0, 0.001, 0.0])
    first = advect_motion_base(rgb, cells, xi, geom)
    second = advect_motion_base(rgb, cells, xi, geom)
    assert np.array_equal(first["frame1_base"], second["frame1_base"])
    assert np.array_equal(first["frame1_cells"], second["frame1_cells"])
    assert first["frame1_base_sha256"] == second["frame1_base_sha256"]
    with pytest.raises(PredictProjectReceiverError, match="translation-first xi"):
        advect_motion_base(rgb, cells, np.zeros(5), geom)


def test_advected_base_warps_ground_and_persists_offground(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    cells = np.array([[0, 3, 0], [3, 0, 3]], dtype=np.uint8)
    geom = GroundHomographyGeom.eon(native_hw=(2, 3), pitch=-0.05)

    def roll_right(values: np.ndarray, _xi: np.ndarray, _geom: GroundHomographyGeom) -> np.ndarray:
        return np.roll(values, 1, axis=1).astype(np.float64)

    monkeypatch.setattr(g1_warp, "warp_frame0_native_numpy", roll_right)
    result = advect_motion_base(rgb, cells, np.ones(6), geom, ground_class_ids=(0,))
    warped_cells = np.roll(cells, 1, axis=1)
    ground = warped_cells == 0
    expected = np.where(ground[..., None], np.roll(rgb, 1, axis=1), rgb)
    assert np.array_equal(result["frame1_cells"], warped_cells)
    assert np.array_equal(result["frame1_base"], expected)
    assert np.array_equal(result["offground_mask"], ~ground)


def test_exact_modular_exception_roundtrips_real_brotli_bytes() -> None:
    base = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    target = np.flip(base, axis=1).copy()
    with pytest.raises(measurement_tool.MeasurementError, match="canonical scorer RGB"):
        measurement_tool._encode_exact_plane_exception(base, target)
    base = np.resize(base, (384, 512, 3)).astype(np.uint8)
    target = np.flip(base, axis=1).copy()
    payload, receipt = measurement_tool._encode_exact_plane_exception(base, target)
    assert receipt["payload_bytes"] == len(payload)
    assert receipt["target_exact"] is True
    assert receipt["changed_values"] > 0
