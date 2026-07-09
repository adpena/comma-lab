# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Linear-NCDE trajectory model + hit->solve detector (task #344).

Every test exercises the REAL fitting SOLVE / detector on synthetic dynamics with a KNOWN
answer (recovery), on a synthetic PLATEAU (fires) and a synthetic DESCENT (does not fire), and
on the edge/guard paths. No test would pass if the body were replaced by canonical markers.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.ncde_trajectory import (
    HitSolveConfig,
    LinearNCDE,
    TelemetryPath,
    backtest_holdout,
    detect_hit_solve,
    fit_sliding_windows,
    sliding_onestep_backtest,
)


def _simulate(A, B, c, h0, u, dt=1.0):
    """Euler-roll the true CDE dh/dt = A h + B u + c to build a synthetic telemetry path."""
    A, B, c = np.asarray(A), np.asarray(B), np.asarray(c)
    h0 = np.asarray(h0, dtype=float)
    u = np.asarray(u, dtype=float)
    n = u.shape[0]
    h = np.zeros((n, h0.shape[0]))
    h[0] = h0
    for k in range(n - 1):
        h[k + 1] = h[k] + dt * (A @ h[k] + B @ u[k] + c)
    return h


def _stable_path(n=60, seed=0):
    A = np.array([[-0.30, 0.05], [0.00, -0.15]])
    B = np.array([[0.10], [0.00]])
    c = np.array([0.02, -0.01])
    u = np.full((n, 1), 0.5)
    h = _simulate(A, B, c, [1.0, 0.5], u)
    ep = np.arange(n).astype(float)
    path = TelemetryPath(ep, h, u, ("log_d_seg", "y2"), ("softmax_temp",))
    return path, A, B, c


# --- 1. fit recovers KNOWN synthetic dynamics -------------------------------
def test_fit_recovers_known_A_B_c():
    # B and the affine c are only SEPARATELY identifiable when the control VARIES (constant
    # control is collinear with the affine bias -> only B*u+c is identified). Drive with a
    # varying control so the full (A, B, c) is recoverable.
    A = np.array([[-0.30, 0.05], [0.00, -0.15]])
    B = np.array([[0.10], [0.02]])
    c = np.array([0.02, -0.01])
    k = np.arange(60)
    u = (0.5 + 0.3 * np.sin(0.3 * k)).reshape(-1, 1)  # VARYING control
    h = _simulate(A, B, c, [1.0, 0.5], u)
    path = TelemetryPath(k.astype(float), h, u, ("log_d_seg", "y2"), ("softmax_temp",))
    m = LinearNCDE.fit(path, ridge=1e-10)
    assert m.r2 > 0.999
    assert np.allclose(m.A, A, atol=1e-3)
    assert np.allclose(m.B, B, atol=1e-3)
    assert np.allclose(m.c, c, atol=1e-3)


# --- 2. asymptote (fixed point) recovery ------------------------------------
def test_asymptote_matches_closed_form_fixed_point():
    path, A, B, c = _stable_path()
    m = LinearNCDE.fit(path, ridge=1e-8)
    u_star = np.array([0.5])
    asym = m.asymptote(u_star)
    # true fixed point: A h + B u + c = 0 -> h = -A^{-1}(B u + c)
    true = -np.linalg.solve(A, B @ u_star + c)
    assert asym is not None
    assert np.allclose(asym, true, atol=1e-3)


# --- 3. stability classification (stable vs unstable) -----------------------
def test_stability_classification():
    path, _, _, _ = _stable_path()
    assert LinearNCDE.fit(path, ridge=1e-8).is_stable() is True
    # unstable growing system
    A = np.array([[0.10, 0.0], [0.0, -0.20]])
    u = np.full((40, 1), 0.0)
    h = _simulate(A, np.array([[0.0], [0.0]]), np.array([0.0, 0.0]), [0.1, 0.5], u)
    ep = np.arange(40).astype(float)
    m = LinearNCDE.fit(TelemetryPath(ep, h, u, ("a", "b"), ("u",)), ridge=1e-8)
    assert m.is_stable() is False
    assert m.slowest_time_constant() is not None  # has a stable mode (b) even though a grows
    # asymptote refuses on an unstable system
    assert m.asymptote(np.array([0.0])) is None


# --- 4. time constant is correct sign/scale ---------------------------------
def test_slowest_time_constant_scale():
    path, A, _, _ = _stable_path()
    m = LinearNCDE.fit(path, ridge=1e-8)
    tau = m.slowest_time_constant()
    # slowest eigenvalue of A is -0.15 -> tau ~ 1/0.15 = 6.67
    assert tau == pytest.approx(1.0 / 0.15, rel=0.02)


# --- 5. detector FIRES on a synthetic plateau -------------------------------
def test_detector_fires_on_plateau():
    # a window in the APPROACH-to-asymptote regime: enough dynamics to fit A (r2 high) but the
    # state is already near the fixed point -> small predicted remaining descent -> BASIN fire.
    # (A FULLY-flat window has no dynamics to fit and is correctly refused, per test 7.)
    path, _, _, _ = _stable_path(n=120)
    m = LinearNCDE.fit(path.window(15, 45), ridge=1e-8)
    assert m.is_stable() and m.r2 > 0.9
    v = detect_hit_solve(m, path.state[44], np.array([0.5]),
                         HitSolveConfig(target_state="log_d_seg"))
    assert v.fire is True
    assert v.remaining_descent_frac is not None and v.remaining_descent_frac < 0.05
    assert "BASIN" in v.reason


# --- 6. detector does NOT fire on a still-descending series ------------------
def test_detector_no_fire_on_descent():
    # early window: still far from asymptote (rapid descent)
    path, _, _, _ = _stable_path(n=120)
    m = LinearNCDE.fit(path.window(0, 12), ridge=1e-8)
    v = detect_hit_solve(
        m, path.state[11], np.array([0.5]),
        HitSolveConfig(target_state="log_d_seg", handoff_horizon_epochs=3.0),
    )
    assert v.fire is False
    assert "NO-FIRE" in v.reason


# --- 7. NO-FAKE guard: an unstable fit NEVER fires --------------------------
def test_detector_refuses_untrustworthy_fit():
    # a genuinely UNSTABLE system (growing mode) -> the guard must refuse to fire regardless of
    # r2, because an unstable fit has no meaningful asymptote to reason about.
    A = np.array([[0.08, 0.0], [0.0, -0.20]])  # first mode grows -> unstable
    u = np.zeros((40, 1))
    h = _simulate(A, np.array([[0.0], [0.0]]), np.array([0.0, 0.0]), [0.05, 0.5], u)
    ep = np.arange(40).astype(float)
    m = LinearNCDE.fit(TelemetryPath(ep, h, u, ("log_d_seg", "y"), ("u",)), ridge=1e-9)
    assert m.is_stable() is False
    v = detect_hit_solve(m, h[-1], np.array([0.0]), HitSolveConfig(target_state="log_d_seg"))
    assert v.fire is False
    assert "instrument-invalid" in v.reason or "not trustworthy" in v.reason


# --- 8. plateau_epochs monotonic in eps + positive --------------------------
def test_plateau_epochs_monotonic_in_eps():
    path, _, _, _ = _stable_path()
    m = LinearNCDE.fit(path, ridge=1e-8)
    t_loose = m.plateau_epochs(0.10)
    t_tight = m.plateau_epochs(0.01)
    assert t_loose is not None and t_tight is not None
    assert 0.0 < t_loose < t_tight  # tighter eps -> longer predicted time
    with pytest.raises(ValueError):
        m.plateau_epochs(1.5)


# --- 9. insufficient data raises ------------------------------------------
def test_fit_insufficient_data_raises():
    ep = np.arange(5).astype(float)
    h = np.random.default_rng(0).normal(size=(5, 2))
    u = np.zeros((5, 1))
    with pytest.raises(ValueError, match="insufficient"):
        LinearNCDE.fit(TelemetryPath(ep, h, u, ("a", "b"), ("u",)))


# --- 10. affine term: nonzero asymptote under zero control ------------------
def test_affine_gives_nonzero_asymptote_zero_control():
    A = np.array([[-0.25]])
    c = np.array([0.5])  # fixed point = -c/A = 2.0
    u = np.zeros((40, 1))  # control identically zero
    h = _simulate(A, np.array([[0.0]]), c, [0.0], u)
    ep = np.arange(40).astype(float)
    m = LinearNCDE.fit(TelemetryPath(ep, h, u, ("x",), ("u",)), ridge=1e-9)
    asym = m.asymptote(np.array([0.0]))
    assert asym is not None
    assert asym[0] == pytest.approx(2.0, rel=1e-2)


# --- 11. ridge shrinks coefficients toward zero as lambda grows -------------
def test_ridge_shrinks_coefficients():
    path, _, _, _ = _stable_path()
    small = LinearNCDE.fit(path, ridge=1e-8)
    huge = LinearNCDE.fit(path, ridge=1e6)
    assert np.linalg.norm(huge.A) < np.linalg.norm(small.A)


# --- 12. to_row carries the NON-PROMOTABLE advisory axis + fire semantics ---
def test_to_row_advisory_axis_and_fields():
    path, _, _, _ = _stable_path(n=120)
    m = LinearNCDE.fit(path.window(60, 120), ridge=1e-8)
    v = detect_hit_solve(m, path.state[-1], np.array([0.5]),
                         HitSolveConfig(target_state="log_d_seg"))
    row = v.to_row(epoch=99.0, stage="unify_tau")
    assert row["stage"] == "ncde_trajectory"
    assert row["actuation"] == "NONE"
    assert "NON-PROMOTABLE" in row["axis"]
    assert row["fire"] == v.fire
    assert row["epoch"] == 99.0
    assert row["target"] == "log_d_seg"


# --- 13. sliding-window count + one-step backtest accuracy ------------------
def test_sliding_windows_and_onestep_backtest():
    path, _, _, _ = _stable_path(n=80)
    wf = fit_sliding_windows(path, window=20, stride=10,
                             detector_cfg=HitSolveConfig(target_state="log_d_seg"))
    # windows at starts 0,10,20,...,60 -> 7 windows
    assert len(wf) == len(range(0, path.n - 20 + 1, 10))
    assert all(w.verdict is not None for w in wf)
    bt = sliding_onestep_backtest(path, window=20, target_state="log_d_seg")
    # a well-fit linear CDE predicts its own next step near-exactly
    assert bt["n_windows"] > 0
    assert bt["target_onestep_mape"] < 0.02


# --- 14. predict shape + forward-roll matches Euler-consistency -------------
def test_predict_shape_and_consistency():
    path, A, B, c = _stable_path()
    m = LinearNCDE.fit(path, ridge=1e-8)
    u_future = np.full((5, 1), 0.5)
    pred = m.predict(path.state[0], u_future, 5)
    assert pred.shape == (5, 2)
    # one Euler step from the recovered params must match a hand computation
    h1 = path.state[0] + m.dt * (m.A @ path.state[0] + m.B @ np.array([0.5]) + m.c)
    assert np.allclose(pred[0], h1)


# --- 15. global holdout backtest returns labeled calibration ----------------
def test_backtest_holdout_labeled_calibration():
    path, _, _, _ = _stable_path(n=60)
    bt = backtest_holdout(path, train_frac=0.6, target_state="log_d_seg")
    assert bt["n_holdout"] > 0
    assert "CALIBRATION" in bt["label"]
    assert "NON-PROMOTABLE" in bt["axis"]
    # on a clean stable linear system the holdout MAPE should be small
    assert bt["target_mape"] < 0.1


# --- 16. TelemetryPath validates shape/monotonicity + NaN detection --------
def test_telemetry_path_validation():
    with pytest.raises(ValueError, match="increasing"):
        TelemetryPath(np.array([0.0, 0.0, 1.0]), np.zeros((3, 1)), np.zeros((3, 1)),
                      ("a",), ("u",))
    with pytest.raises(ValueError, match="row mismatch"):
        TelemetryPath(np.array([0.0, 1.0]), np.zeros((3, 1)), np.zeros((2, 1)),
                      ("a",), ("u",))
    p = TelemetryPath(np.array([0.0, 1.0]), np.array([[np.nan], [1.0]]),
                      np.zeros((2, 1)), ("a",), ("u",))
    assert p.is_finite() is False
    with pytest.raises(ValueError, match="non-finite"):
        LinearNCDE.fit(
            TelemetryPath(np.arange(20.0), np.full((20, 1), np.nan), np.zeros((20, 1)),
                          ("a",), ("u",))
        )


# --- 17. fit is robust to a constant (collinear-with-affine) control column -
def test_fit_robust_to_constant_control_column():
    # control identically constant -> collinear with the affine bias; ridge must keep it finite
    A = np.array([[-0.2]])
    u = np.full((40, 1), 1.0)  # constant
    h = _simulate(A, np.array([[0.05]]), np.array([0.01]), [1.0], u)
    ep = np.arange(40).astype(float)
    m = LinearNCDE.fit(TelemetryPath(ep, h, u, ("x",), ("u",)), ridge=1e-3)
    assert np.all(np.isfinite(m.A))
    assert np.all(np.isfinite(m.B))
    assert m.is_stable()
