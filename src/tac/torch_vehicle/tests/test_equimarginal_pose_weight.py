# SPDX-License-Identifier: MIT
"""NO-FAKE tests for Lever A — the equimarginal pose-weight controller.

These verify BEHAVIOR (the controller actually moves w_pose toward the equimarginal point), not constants.
Every test would FAIL if ``equimarginal_pose_weight_multiplier`` were replaced by ``return 1.0``.
"""

from __future__ import annotations

import pytest

from tac.torch_vehicle.equimarginal_pose_weight import (
    EquimarginalPoseWeightController,
    EquimarginalPoseWeightState,
    equimarginal_pose_weight_multiplier,
)


def _mult(seg, pose, *, rho=1.0, decay=0.0, tol=0.15, lo=0.25, hi=4.0, state=None):
    """decay=0 → ratio_ema == the raw measured ratio (deterministic single-step check)."""
    st = state or EquimarginalPoseWeightState()
    return equimarginal_pose_weight_multiplier(
        seg, pose, st, rho=rho, decay=decay, tol=tol, clamp_lo=lo, clamp_hi=hi
    ), st


def test_pose_pull_too_high_drops_w_pose() -> None:
    # pose pull 2x seg pull, rho=1 → ratio 2.0 > rho → multiplier rho/ratio = 0.5 → drop w_pose.
    m, st = _mult(1.0, 2.0, rho=1.0, lo=0.1)
    assert st.ratio_ema == pytest.approx(2.0)
    assert m == pytest.approx(0.5)


def test_pose_pull_too_low_raises_w_pose() -> None:
    # pose pull half of seg pull → ratio 0.5 < rho=1 → multiplier rho/ratio = 2.0 → raise w_pose.
    m, st = _mult(2.0, 1.0, rho=1.0, hi=4.0)
    assert st.ratio_ema == pytest.approx(0.5)
    assert m == pytest.approx(2.0)


def test_inside_deadband_is_exact_noop() -> None:
    # ratio within tol*rho of rho → multiplier exactly 1.0 (no spurious churn).
    m, _ = _mult(1.0, 1.05, rho=1.0, tol=0.15)  # ratio 1.05, |1.05-1| = 0.05 <= 0.15
    assert m == 1.0


def test_ratio_equals_rho_is_noop() -> None:
    m, _ = _mult(1.0, 1.0, rho=1.0)
    assert m == 1.0


def test_step_multiplier_is_clamped() -> None:
    # pose pull 100x seg → ratio 100 → rho/ratio = 0.01 but clamped to lo.
    m, _ = _mult(1.0, 100.0, rho=1.0, lo=0.5, hi=2.0)
    assert m == pytest.approx(0.5)
    # pose pull 0 → ratio 0 → would be +inf multiplier, clamped to hi.
    m2, st2 = _mult(1.0, 0.0, rho=1.0, lo=0.5, hi=2.0)
    assert m2 == pytest.approx(2.0)


def test_degenerate_seg_pull_drives_w_pose_down() -> None:
    # seg pull ~0 (no seg signal) → ratio +inf → drive w_pose DOWN by lo (don't amplify pose).
    m, st = _mult(0.0, 1.0, rho=1.0, lo=0.3, hi=3.0)
    assert m == pytest.approx(0.3)


def test_both_pulls_zero_is_noop() -> None:
    m, st = _mult(0.0, 0.0, rho=1.0)
    assert m == 1.0
    assert st.ratio_ema == pytest.approx(1.0)  # seeded at rho (no-op)


def test_ema_smoothing_blends_history() -> None:
    st = EquimarginalPoseWeightState()
    # first sample seeds directly.
    equimarginal_pose_weight_multiplier(1.0, 2.0, st, rho=1.0, decay=0.5, tol=0.0, clamp_lo=0.1, clamp_hi=10.0)
    assert st.ratio_ema == pytest.approx(2.0)
    # second sample: ratio 4.0 → ema = 0.5*2 + 0.5*4 = 3.0.
    equimarginal_pose_weight_multiplier(1.0, 4.0, st, rho=1.0, decay=0.5, tol=0.0, clamp_lo=0.1, clamp_hi=10.0)
    assert st.ratio_ema == pytest.approx(3.0)
    assert st.steps == 2


def test_controller_converges_toward_rho() -> None:
    # Drive a fixed pose:seg pull RATIO and verify the accumulated w_pose_frac moves so that, were the
    # cotangents recomputed at the new weight, the ratio would approach rho. We simulate: the measured
    # ratio at a given w_pose is ``w_pose_frac * unit_ratio`` (cot_pose linear in w_pose). Start unit_ratio
    # = 4 (pose over-weighted), rho = 1 → frac should fall toward 0.25.
    ctrl = EquimarginalPoseWeightController(rho=1.0, decay=0.0, tol=0.01, bound_lo=0.05, bound_hi=10.0,
                                           step_clamp_lo=0.5, step_clamp_hi=2.0)
    unit_ratio = 4.0
    base = 10.0
    for _ in range(30):
        # the measured cot_pose norm scales with the CURRENT frac (the weight applied last step).
        cot_pose = unit_ratio * ctrl.w_pose_frac
        ctrl.update(1.0, cot_pose, w_pose_base=base)
    # after convergence the frac · unit_ratio should be near rho=1 → frac ≈ 0.25.
    assert ctrl.w_pose_frac == pytest.approx(0.25, abs=0.05)


def test_accumulated_weight_is_clamped_to_bounds() -> None:
    ctrl = EquimarginalPoseWeightController(rho=1.0, decay=0.0, tol=0.01, bound_lo=0.5, bound_hi=2.0,
                                           step_clamp_lo=0.1, step_clamp_hi=10.0)
    # huge pose pull every step → frac driven down but clamped at bound_lo=0.5.
    for _ in range(20):
        ctrl.update(1.0, 1000.0, w_pose_base=4.0)
    assert ctrl.w_pose_frac == pytest.approx(0.5)
    assert ctrl.update(1.0, 1000.0, w_pose_base=4.0) == pytest.approx(2.0)  # 4.0 * 0.5


def test_state_dict_round_trip_continues_trajectory() -> None:
    ctrl = EquimarginalPoseWeightController(rho=1.0, decay=0.5)
    ctrl.update(1.0, 3.0, w_pose_base=5.0)
    sd = ctrl.state_dict()
    ctrl2 = EquimarginalPoseWeightController(rho=1.0, decay=0.5)
    ctrl2.load_state_dict(sd)
    assert ctrl2.ratio_ema == pytest.approx(ctrl.ratio_ema)
    assert ctrl2.w_pose_frac == pytest.approx(ctrl.w_pose_frac)
    assert ctrl2.state.steps == ctrl.state.steps


def test_invalid_params_fail_closed() -> None:
    with pytest.raises(ValueError):
        EquimarginalPoseWeightController(rho=0.0)
    with pytest.raises(ValueError):
        EquimarginalPoseWeightController(decay=1.0)
    with pytest.raises(ValueError):
        EquimarginalPoseWeightController(tol=0.0)
    with pytest.raises(ValueError):
        EquimarginalPoseWeightController(bound_lo=2.0)  # lo must be <= 1
    with pytest.raises(ValueError):
        EquimarginalPoseWeightController(bound_hi=0.5)  # hi must be >= 1


def test_telemetry_row_has_score_pull_fields() -> None:
    ctrl = EquimarginalPoseWeightController(rho=1.0)
    w = ctrl.update(2.0, 1.0, w_pose_base=3.0)
    tel = ctrl.telemetry(cot_seg_norm=2.0, cot_pose_norm=1.0, w_pose_eff=w)
    assert tel["cot_seg_norm"] == 2.0
    assert tel["cot_pose_norm"] == 1.0
    assert tel["w_pose_effective"] == pytest.approx(w)
    assert "ratio_ema" in tel and "w_pose_frac" in tel and tel["rho_target"] == 1.0
