# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "measure_bev_staticity_developability.py"
_SPEC = importlib.util.spec_from_file_location("measure_bev_staticity_developability", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_oriented_shallow_boundary_uses_margin_ratio_and_excludes_deep_side() -> None:
    labels = np.zeros((384, 512), dtype=np.uint8)
    labels[:, 256:] = 1
    margins = np.ones((384, 512), dtype=np.float64)
    margins[:, 255] = 1.0
    margins[:, 256] = 3.0
    class_index = {"Road": 0, "Lane": 1, "Undrivable": 2, "Movable": 3, "MyCar": 4}
    points, all_points, counts = _MODULE.oriented_shallow_boundary_points(labels, margins, class_index)
    assert points["Road"].shape == (384, 2)
    assert np.allclose(points["Road"][:, 0], 255.25)
    assert points["Lane"].shape == (0, 2)
    assert all_points["Road"].shape == (384, 2)
    assert all_points["Lane"].shape == (384, 2)
    assert counts["Road"]["shallow"] == 384
    assert counts["Lane"]["deep"] == 384


def test_static_segments_exclude_event_frames() -> None:
    assert _MODULE.static_segments(8, [2, 5]) == [(0, 2), (3, 5), (6, 8)]


def test_custody_constants_pin_cache_and_ssd_waterfall() -> None:
    assert _MODULE.GT_CACHE_SHA256 == "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
    assert all(str(root).startswith("/Volumes/") for root in _MODULE.SSD_ROOTS)


def test_ruling_summary_distinguishes_static_from_moving() -> None:
    base = np.linspace(-1.0, 1.0, _MODULE.GRID_BINS)
    static = np.stack([np.stack((base, base, base)) for _ in range(12)])
    forward = np.full_like(static, 20.0)
    centers = np.linspace(0.0, 10.0, _MODULE.GRID_BINS)
    static_summary = _MODULE.summarize_stratum(static, forward, centers, ground=False, fx=400.0)
    moving = static + np.arange(12, dtype=np.float64)[:, None, None] * 2.0
    moving_summary = _MODULE.summarize_stratum(moving, forward, centers, ground=False, fx=400.0)
    assert static_summary["near_static"] is True
    assert static_summary["static_fraction_at_1px_floor"] == 1.0
    assert moving_summary["residual_dynamics_fraction"] > 0.0


def test_singleton_label_custody_scores_f0_and_f1_as_separate_calls() -> None:
    f0 = np.zeros((*_MODULE.CAMERA_HW, 3), dtype=np.uint8)
    f1 = np.ones_like(f0)
    expected_f0 = np.full(_MODULE.SCORER_HW, 2, dtype=np.uint8)
    expected_f1 = np.full(_MODULE.SCORER_HW, 3, dtype=np.uint8)
    calls: list[int] = []

    def score_frame(frame: np.ndarray) -> np.ndarray:
        calls.append(int(frame[0, 0, 0]))
        return expected_f0 if len(calls) == 1 else expected_f1

    label0, custody = _MODULE.singleton_label_custody(f0, f1, expected_f1, score_frame)
    assert calls == [0, 1]
    assert np.array_equal(label0, expected_f0)
    assert custody["scorer_call_geometry"] == "one_native_frame_per_call"
    assert custody["singleton_call_count"] == 2
    assert custody["f1_cache_label_mismatches"] == 0
    assert custody["f1_cache_binding_status"] == "EXACT"


def test_absolute_frame_charts_close_every_within_phase() -> None:
    from tac.lie import _se3_numpy as se3

    raw_within = np.zeros((3, 6), dtype=np.float64)
    raw_cross = np.zeros_like(raw_within)
    raw_within[0, 2] = 0.25
    raw_within[1, 2] = 1.0
    raw_within[2, 1] = -0.5
    raw_cross[1, 5] = np.pi / 2.0
    raw_cross[2, 4] = -np.pi / 4.0
    poses_f0, poses_f1, _xi_cross, xi_within, validation = (
        _MODULE.absolute_frame_trajectories(
            raw_within,
            raw_cross,
            s_t=1.0,
            s_r=1.0,
            pitch_rad=0.0,
        )
    )
    assert np.array_equal(poses_f1[0], np.eye(4))
    assert np.allclose(
        poses_f0[0], se3.inverse(se3.exp_se3(xi_within[0])), atol=1e-12, rtol=0.0
    )
    for frame in range(len(raw_within)):
        reconstructed_f1 = se3.compose(poses_f0[frame], se3.exp_se3(xi_within[frame]))
        assert np.allclose(reconstructed_f1, poses_f1[frame], atol=1e-12, rtol=0.0)
    assert validation["within_phase_closure_max_abs"] < 1e-12


def test_absolute_frame_charts_use_noncommuting_cross_then_within_order() -> None:
    from tac.lie import _se3_numpy as se3

    raw_within = np.zeros((2, 6), dtype=np.float64)
    raw_cross = np.zeros_like(raw_within)
    raw_cross[1, 5] = np.pi / 2.0  # rotate around z
    raw_within[1, 2] = 1.0  # pose[2] maps to +x translation
    poses_f0, poses_f1, xi_cross, xi_within, validation = (
        _MODULE.absolute_frame_trajectories(
            raw_within,
            raw_cross,
            s_t=1.0,
            s_r=1.0,
            pitch_rad=0.0,
        )
    )
    expected_f0 = se3.compose(poses_f1[0], se3.exp_se3(xi_cross[1]))
    expected_f1 = se3.compose(expected_f0, se3.exp_se3(xi_within[1]))
    reversed_order = se3.compose(se3.exp_se3(xi_within[1]), se3.exp_se3(xi_cross[1]))
    assert np.allclose(poses_f0[1], expected_f0, atol=1e-12, rtol=0.0)
    assert np.allclose(poses_f1[1], expected_f1, atol=1e-12, rtol=0.0)
    assert not np.allclose(poses_f1[1], reversed_order, atol=1e-12, rtol=0.0)
    assert validation["composition_order"] == "A_f1[t-1] * exp(xi_cross[t]) * exp(xi_within[t])"
    assert validation["already_relative_targets_redifferenced"] is False


def test_v2_motion_custody_preserves_calibration_and_supersedes_proxy_policy() -> None:
    source = {
        "g1_receipt_path": "g1.json",
        "g1_receipt_sha256": "a" * 64,
        "lawref_equation_ids": ["law:s_t", "law:s_r"],
        "lawref_resolutions": {"s_t": {"resolved_value": -0.00143}},
        "pitch_custody": {"resolved_value": -0.05},
        "proxy_limitation": "stale nearest-target-pair proxy text",
    }
    custody = _MODULE.bev_staticity_v2_motion_custody(source)
    assert custody["calibration_authority"]["g1_receipt_sha256"] == "a" * 64
    assert custody["calibration_authority"]["lawref_equation_ids"] == ["law:s_t", "law:s_r"]
    assert custody["transition_authority"]["cross_target_is_nearest_pair_proxy"] is False
    assert custody["supersession"]["supersedes_g1_nearest_target_proxy_limitation"] is True
    assert "proxy_limitation" not in custody["calibration_authority"]


def test_bottom_connected_component_excludes_other_mycar_islands() -> None:
    labels = np.zeros((8, 9), dtype=np.uint8)
    labels[5:, 3:6] = 4
    labels[1:3, 1:3] = 4
    labels[-1, -1] = 4
    hood = _MODULE.bottom_connected_component(labels, 4)
    assert np.all(hood[5:, 3:6])
    assert not np.any(hood[1:3, 1:3])
    assert not hood[-1, -1]
    assert int(np.count_nonzero(hood)) == 9
    boundary = _MODULE.component_boundary_points(hood)
    assert boundary.shape == (8, 2)


def test_hood_world_to_ego_roundtrip_closes_below_floor() -> None:
    from tac.lie import _se3_numpy as se3

    camera = SimpleNamespace(fx_scorer=400.0, fy_scorer=400.0, cx_scorer=256.0)
    boundary = np.array([[200.0, 300.0], [256.0, 350.0], [310.0, 383.0]])
    pose = se3.exp_se3(np.array([1.2, -0.4, 2.5, 0.01, -0.03, 0.02]))
    closure = _MODULE.hood_world_to_ego_closure(boundary, pose, camera)
    assert closure["lifted_point_count"] == 3
    assert closure["max_error_m"] < _MODULE.D0_CLOSURE_FLOOR_M
    assert closure["passed"] is True


def _passing_hood_summary() -> dict[str, object]:
    return {
        "ruling_reconstruction_residual_px": {"p50": 0.75},
        "static_fraction_at_1px_floor": 0.6,
    }


def _valid_trajectory() -> dict[str, object]:
    return {
        "finite_se3": True,
        "homogeneous_last_rows_valid": True,
        "inverse_compose_closure_max_abs": 1e-15,
        "within_phase_closure_max_abs": 1e-15,
        "cross_phase_closure_max_abs": 1e-15,
    }


def test_n64_gate_reports_first_failed_stage_and_refuses_downstream() -> None:
    decision = _MODULE.d0_gate_decision(
        label_mismatches=2,
        trajectory=_valid_trajectory(),
        hood_closure_max_m=1e-15,
        hood_summary=_passing_hood_summary(),
    )
    assert decision["passed"] is False
    assert decision["first_failed_stage"] == "D0.1_SINGLETON_LABEL_CUSTODY"
    assert decision["D1_D3_authorized"] is False


def test_no_d1_d3_authorization_when_any_d0_condition_fails() -> None:
    failing_conditions = (
        {"trajectory": {**_valid_trajectory(), "finite_se3": False}},
        {"trajectory": {**_valid_trajectory(), "within_phase_closure_max_abs": 1e-8}},
        {"hood_closure_max_m": 1e-8},
        {"hood_summary": {**_passing_hood_summary(), "ruling_reconstruction_residual_px": {"p50": 1.01}}},
        {"hood_summary": {**_passing_hood_summary(), "static_fraction_at_1px_floor": 0.49}},
    )
    for override in failing_conditions:
        arguments = {
            "label_mismatches": 0,
            "trajectory": _valid_trajectory(),
            "hood_closure_max_m": 1e-15,
            "hood_summary": _passing_hood_summary(),
            **override,
        }
        decision = _MODULE.d0_gate_decision(**arguments)
        assert decision["passed"] is False
        assert decision["D1_D3_authorized"] is False


def test_d0_gate_authorizes_d1_d3_only_when_every_condition_passes() -> None:
    decision = _MODULE.d0_gate_decision(
        label_mismatches=0,
        trajectory=_valid_trajectory(),
        hood_closure_max_m=1e-15,
        hood_summary=_passing_hood_summary(),
    )
    assert decision["passed"] is True
    assert decision["first_failed_stage"] is None
    assert decision["D1_D3_authorized"] is True
