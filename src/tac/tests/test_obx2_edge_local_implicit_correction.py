# SPDX-License-Identifier: MIT
"""Behaviour tests for the OBX2 Stage-0/Stage-2a runner.

These assert what the code DOES on real geometry: deterministic rung
materialization, exact denominators, exact gate arithmetic, and fail-closed
refusals.  They are not constant checks.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_obx2_edge_local_implicit_correction as obx2  # noqa: E402


def _teacher(pairs: int = 1, *, seed: int = 7) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.integers(
        0, 256, size=(pairs, 2, obx2.CAMERA_H, obx2.CAMERA_W, obx2.CHANNELS), dtype=np.uint8
    )
    tensor = torch.from_numpy(raw.copy()).permute(0, 1, 4, 2, 3).float()
    return tensor, raw


def _gt(pairs: int = 1, *, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros((pairs, obx2.EVAL_H, obx2.EVAL_W), dtype=np.uint8)
    field[:, :, obx2.EVAL_W // 2 :] = 2
    field[:, obx2.EVAL_H // 3, :] = 1
    field ^= rng.integers(0, 2, size=field.shape, dtype=np.uint8) * 0
    return field


def test_gate_arithmetic_matches_the_burn_spec() -> None:
    assert obx2.PACKET_BYTE_GATE == 122_000
    assert abs(obx2.RATE_AT_GATE - 0.08123479228090491) < 1e-15
    assert abs(obx2.DISTORTION_BUDGET_FOR_SUB_012 - 0.03876520771909509) < 1e-15


def test_rung_specification_covers_every_default_rung_and_refuses_unknown() -> None:
    for name in obx2.DEFAULT_RUNGS:
        spec = obx2.rung_specification(name)
        assert spec["name"] == name
        assert spec["kind"] in {
            "identity",
            "render_grid",
            "scorer_plane_grid",
            "scorer_plane_render_noise",
            "uniform_noise",
            "geometric_shift",
        }
    for bad in ("grid_not_a_size", "grid_384", "grid_0x512", "grid_384x512x2", "sp_x", "sp_0x1", "sp384_render_noise_", "sp384_render_noise_x"):
        with pytest.raises(obx2.OBX2Error):
            obx2.rung_specification(bad)
    with pytest.raises(obx2.OBX2Error):
        obx2.rung_specification("mystery")


def test_identity_rung_reproduces_the_teacher_bytes_exactly() -> None:
    teacher, raw = _teacher()
    out, diagnostics, arrays = obx2.apply_rung(
        teacher, obx2.rung_specification("teacher"), pair_ids=[0], gt_chunk=_gt()
    )
    assert diagnostics == {} and arrays == {}
    got = out.to(torch.uint8).numpy()
    want = np.transpose(raw, (0, 1, 4, 2, 3))
    assert np.array_equal(got, want)
    assert obx2.camera_squared_error(got, raw, 0) == 0


def test_render_grid_rung_is_deterministic_and_stays_in_range() -> None:
    teacher, raw = _teacher()
    spec = obx2.rung_specification("grid_384x512")
    first = obx2.apply_rung(teacher, spec, pair_ids=[0], gt_chunk=_gt())[0].to(torch.uint8).numpy()
    second = obx2.apply_rung(teacher, spec, pair_ids=[0], gt_chunk=_gt())[0].to(torch.uint8).numpy()
    assert np.array_equal(first, second)
    assert first.shape == (1, 2, obx2.CHANNELS, obx2.CAMERA_H, obx2.CAMERA_W)
    assert obx2.camera_squared_error(first, raw, 0) > 0


def test_uniform_noise_rung_is_seeded_bounded_and_clamped() -> None:
    teacher, raw = _teacher()
    spec = obx2.rung_specification("noise_4")
    first = obx2.apply_rung(teacher, spec, pair_ids=[3], gt_chunk=_gt())[0].to(torch.uint8).numpy()
    second = obx2.apply_rung(teacher, spec, pair_ids=[3], gt_chunk=_gt())[0].to(torch.uint8).numpy()
    other = obx2.apply_rung(teacher, spec, pair_ids=[4], gt_chunk=_gt())[0].to(torch.uint8).numpy()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    want = np.transpose(raw, (0, 1, 4, 2, 3)).astype(np.int32)
    delta = first.astype(np.int32) - want
    assert int(np.abs(delta).max()) <= 4
    assert int(first.min()) >= 0 and int(first.max()) <= 255


def test_zero_amplitude_noise_rung_refuses() -> None:
    teacher, _ = _teacher()
    spec = dict(obx2.rung_specification("noise_1"))
    spec["amplitude_lsb"] = 0
    with pytest.raises(obx2.OBX2Error):
        obx2.apply_rung(teacher, spec, pair_ids=[0], gt_chunk=_gt())


def test_oracle_structured_rung_leaves_the_boundary_band_untouched() -> None:
    teacher, raw = _teacher()
    gt = _gt()
    spec = obx2.rung_specification("interior_noise_32")
    out = obx2.apply_rung(teacher, spec, pair_ids=[0], gt_chunk=gt)[0].to(torch.uint8).numpy()
    want = np.transpose(raw, (0, 1, 4, 2, 3)).astype(np.int32)
    changed = (out.astype(np.int32) != want).any(axis=(1, 2))[0]
    band = obx2.gate_band(gt, int(spec["band_radius_eval_cells"]))[0]
    band_camera = (
        torch.nn.functional.interpolate(
            torch.from_numpy(band.astype(np.float32))[None, None],
            size=(obx2.CAMERA_H, obx2.CAMERA_W),
            mode="nearest",
        )[0, 0]
        .numpy()
        .astype(bool)
    )
    assert not bool(changed[band_camera].any())
    assert bool(changed[~band_camera].any())


def test_gate_band_marks_four_neighbour_boundaries() -> None:
    field = np.zeros((1, obx2.EVAL_H, obx2.EVAL_W), dtype=np.uint8)
    field[0, :, 10:] = 3
    band = obx2.gate_band(field, 0)
    assert bool(band[0, 5, 9]) and bool(band[0, 5, 10])
    assert not bool(band[0, 5, 0])
    grown = obx2.gate_band(field, 1)
    assert bool(grown[0, 5, 8]) and bool(grown[0, 5, 11])
    assert int(grown.sum()) > int(band.sum())


def test_camera_squared_error_is_exact_integer_arithmetic() -> None:
    raw = np.zeros((1, 2, obx2.CAMERA_H, obx2.CAMERA_W, obx2.CHANNELS), dtype=np.uint8)
    got = np.transpose(raw, (0, 1, 4, 2, 3)).copy()
    got[0, 0, 0, 0, 0] = 3
    got[0, 1, 2, 5, 7] = 4
    assert obx2.camera_squared_error(got, raw, 0) == 9 + 16


def test_chunk_rows_report_per_pair_denominators() -> None:
    raw = np.zeros((2, 2, obx2.CAMERA_H, obx2.CAMERA_W, obx2.CHANNELS), dtype=np.uint8)
    camera = np.transpose(raw, (0, 1, 4, 2, 3)).copy()
    argmax = np.zeros((2, obx2.EVAL_H, obx2.EVAL_W), dtype=np.uint8)
    gt = np.zeros((2, obx2.EVAL_H, obx2.EVAL_W), dtype=np.uint8)
    gt[1, 0, 0] = 4
    pose = np.zeros((2, 6), dtype="<f4")
    target = np.zeros((2, 6), dtype="<f4")
    target[0, 0] = 2.0
    rows = obx2.chunk_rows(
        pair_ids=[0, 1],
        argmax=argmax,
        pose=pose,
        gt_chunk=gt,
        pose_target=target,
        camera_u8=camera,
        teacher_np=raw,
        scorer_plane_squared=[0.0, 0.0],
    )
    assert [row["pair_id"] for row in rows] == [0, 1]
    assert rows[0]["seg_errors"] == 0 and rows[1]["seg_errors"] == 1
    assert rows[0]["pose_squared_error_sum"] == pytest.approx(4.0)
    assert rows[0]["camera_values"] == 2 * obx2.CAMERA_H * obx2.CAMERA_W * obx2.CHANNELS


def _synthetic_rows(seg_errors_per_pair: int, pose_square: float) -> list[dict[str, int | float]]:
    values_per_pair = 2 * obx2.CAMERA_H * obx2.CAMERA_W * obx2.CHANNELS
    return [
        {
            "pair_id": pair,
            "seg_errors": seg_errors_per_pair,
            "seg_pixels": obx2.EVAL_H * obx2.EVAL_W,
            "pose_squared_error_sum": pose_square,
            "pose_values": 6,
            "camera_squared_error_sum": 0,
            "camera_values": values_per_pair,
            "scorer_plane_squared_error_sum": 0.0,
            "scorer_plane_values": 2 * obx2.EVAL_H * obx2.EVAL_W * obx2.CHANNELS,
        }
        for pair in range(obx2.N)
    ]


def test_aggregate_rung_computes_the_exact_contest_components() -> None:
    rows = _synthetic_rows(98_304, 0.06)
    components = obx2.aggregate_rung(rows)
    assert components["d_seg"] == pytest.approx(0.5)
    assert components["d_pose"] == pytest.approx(0.01)
    assert components["distortion"] == pytest.approx(50.0 + math.sqrt(0.1))
    assert components["camera_rmse_vs_teacher"] == 0.0
    assert components["passes_distortion_gate"] is False
    assert components["strict_byte_cap_at_own_distortion"] == 0


def test_aggregate_rung_refuses_a_short_or_reordered_denominator() -> None:
    rows = _synthetic_rows(0, 0.0)
    with pytest.raises(obx2.OBX2Error):
        obx2.aggregate_rung(rows[:-1])
    swapped = list(rows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(obx2.OBX2Error):
        obx2.aggregate_rung(swapped)


def test_aggregate_rung_refuses_a_drifted_camera_denominator() -> None:
    rows = _synthetic_rows(0, 0.0)
    rows[0]["camera_values"] = 1
    with pytest.raises(obx2.OBX2Error):
        obx2.aggregate_rung(rows)


def test_projected_run_bytes_only_charges_requested_rungs() -> None:
    small = obx2.projected_run_bytes(["teacher"])
    large = obx2.projected_run_bytes(list(obx2.DEFAULT_RUNGS))
    assert small["camera_bytes"] == 0
    assert small["render_bytes"] == 0
    assert large["render_bytes"] == obx2.N * 2 * obx2.CHANNELS * obx2.EVAL_H * obx2.EVAL_W
    assert large["camera_bytes"] == len(obx2.RETAIN_CAMERA_RUNGS) * (
        obx2.N * 2 * obx2.CAMERA_H * obx2.CAMERA_W * obx2.CHANNELS
    )
    assert large["projected_run_bytes"] > small["projected_run_bytes"]
    assert large["two_runs_plus_reserve_bytes"] == 2 * large["projected_run_bytes"] + obx2.MINIMUM_FREE_BYTES


def test_storage_preflight_refuses_a_foreign_output_root(tmp_path: Path) -> None:
    with pytest.raises(obx2.OBX2Error):
        obx2.storage_preflight(tmp_path, required=1)


def test_active_claim_refuses_a_foreign_lane_id() -> None:
    with pytest.raises(obx2.OBX2Error):
        obx2.assert_active_claim("ddm_other_lane_20260911")


def test_active_claim_accepts_the_live_obx2_lane() -> None:
    claim = obx2.assert_active_claim(obx2.LANE_ID)
    assert claim["status"].startswith(("building", "active"))


def test_scorer_plane_render_on_the_identity_grid_converges_to_the_teacher() -> None:
    teacher, _ = _teacher()
    flat = teacher.reshape(2, obx2.CHANNELS, obx2.CAMERA_H, obx2.CAMERA_W)
    camera, history, render = obx2.scorer_plane_render(
        flat, height=obx2.CAMERA_H, width=obx2.CAMERA_W, iterations=3, step=1.0
    )
    assert render.shape == flat.shape
    assert len(history) == 4
    assert history[-1] <= history[0]
    assert history[-1] < 1.0
    assert camera.shape == flat.shape


def test_scorer_plane_render_reduces_the_scorer_plane_residual_on_the_qbf_grid() -> None:
    teacher, _ = _teacher()
    flat = teacher.reshape(2, obx2.CHANNELS, obx2.CAMERA_H, obx2.CAMERA_W)
    _, history, _ = obx2.scorer_plane_render(
        flat, height=obx2.EVAL_H, width=obx2.EVAL_W, iterations=obx2.SP_ITERATIONS, step=obx2.SP_STEP
    )
    assert history[-1] < history[0]


def test_sp_render_rung_retains_the_stage_two_distillation_target() -> None:
    teacher, _ = _teacher()
    spec = obx2.rung_specification("sp_384x512")
    assert spec["retain_render"] is True
    _, diagnostics, arrays = obx2.apply_rung(teacher, spec, pair_ids=[0], gt_chunk=_gt())
    assert arrays["render_u8"].shape == (1, 2, obx2.CHANNELS, obx2.EVAL_H, obx2.EVAL_W)
    assert arrays["render_u8"].dtype == np.uint8
    assert len(diagnostics["scorer_plane_residual_rmse_history"]) == obx2.SP_ITERATIONS + 1


def test_render_noise_rung_perturbs_only_the_render_and_is_seeded() -> None:
    teacher, _ = _teacher()
    spec = obx2.rung_specification("sp384_render_noise_4")
    assert spec["amplitude_lsb"] == 4
    first = obx2.apply_rung(teacher, spec, pair_ids=[2], gt_chunk=_gt())[0]
    second = obx2.apply_rung(teacher, spec, pair_ids=[2], gt_chunk=_gt())[0]
    clean = obx2.apply_rung(teacher, obx2.rung_specification("sp_384x512"), pair_ids=[2], gt_chunk=_gt())[0]
    assert torch.equal(first, second)
    assert not torch.equal(first, clean)
    assert float(first.min()) >= 0.0 and float(first.max()) <= 255.0


def test_geometric_shift_rung_moves_only_frame_one_and_scales_with_the_shift() -> None:
    teacher, raw = _teacher()
    small = obx2.apply_rung(teacher, obx2.rung_specification("shift_025"), pair_ids=[0], gt_chunk=_gt())[0]
    large = obx2.apply_rung(teacher, obx2.rung_specification("shift_200"), pair_ids=[0], gt_chunk=_gt())[0]
    want = torch.from_numpy(np.transpose(raw, (0, 1, 4, 2, 3)).copy()).float()
    assert torch.equal(small[:, 0], want[:, 0])
    assert torch.equal(large[:, 0], want[:, 0])
    assert not torch.equal(small[:, 1], want[:, 1])
    small_error = float((small[:, 1] - want[:, 1]).abs().mean())
    large_error = float((large[:, 1] - want[:, 1]).abs().mean())
    assert large_error > small_error > 0.0
    assert obx2.rung_specification("shift_100")["shift_pixels"] == pytest.approx(1.0)
    with pytest.raises(obx2.OBX2Error):
        obx2.rung_specification("shift_x")
