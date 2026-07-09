# SPDX-License-Identifier: MIT
"""Linear-NCDE trajectory model + hit->solve threshold detector (task #344).

SHADOW-ONLY SENSE ORGAN. Advisory forever: this module NEVER actuates. It models the
training trajectory (the telemetry PATH -- log d_seg / per-class part_frac / loss-term
series / gnorm) as a **linear neural controlled differential equation** and detects when a
stage has entered the quadratic-basin regime where a terminal SOLVE (the #341 GN/CG
finisher) or a #315 stage hand-off should fire. It emits the same advisory-row style as the
costate shadow controller; a downstream controller MAY consume the rows, but the costate
boundary is BINDING (CLAUDE.md): autonomous == advisory-only; heavy/stop/config == operator-GO.

THE MODEL (a SOLVE, not SGD -- fitting the #342 least-squares spirit)
--------------------------------------------------------------------
A linear controlled differential equation drives an observable state ``h in R^d`` by a
control path ``X in R^m`` (the exogenous annealing schedule: softmax_temp, hosc_beta, ...):

    dh/dt = A h + B u + c ,    u = dX/dt   (control increments)

Discretized over epochs (dt in epoch units), this is a linear time-invariant system whose
(A, B, c) are recovered by ONE ridge least-squares SOLVE over a sliding window -- the
control-augmented Dynamic-Mode-Decomposition (DMDc) estimator. Stacking transitions k:

    (h_{k+1} - h_k)/dt  ~=  A h_k + B u_k + c
    Y (N-1 x d)  =  Phi (N-1 x d+m+1) @ C (d+m+1 x d) ,   Phi_k = [h_k | u_k | 1]
    C = (Phi^T Phi + lambda I)^{-1} Phi^T Y      (closed-form ridge -- the SOLVE)

From the fitted dynamics, THREE closed-form predictive laws (no rollout needed):

  * asymptote (predicted within-stage plateau value, per metric):
        h_inf = -A^{-1} (B u_star + c)        (the fixed point, if A is stable/invertible)
  * slowest time constant (predicted descent speed):
        tau_slow = -1 / max_i Re(eig_i(A))    over the stable modes (Re < 0)
  * plateau epochs (predicted epochs to within eps of the asymptote):
        T_eps = -ln(eps) * tau_slow

IDENTIFIABILITY (an honest limit, MEASURED in the tests)
--------------------------------------------------------
A and the asymptote-at-the-training-control are recovered exactly from a clean trajectory.
B and the affine term c are only SEPARATELY identifiable when the control PATH VARIES within
the window -- a constant control is collinear with the affine bias, so only the sum ``B u + c``
is identified (the asymptote at the training u is still exact; extrapolation to a DIFFERENT
u_star is not). The ridge SOLVE keeps a constant-control window finite and well-posed; the
detector's asymptote is therefore trustworthy at the operating control, which is what the
hit->solve verdict uses.

THE HIT->SOLVE DETECTOR
-----------------------
Fires (advisory) on the score-relevant target metric (default log d_seg) when EITHER:

  (a) BASIN condition: the predicted *remaining* within-stage descent
      |h_now - h_inf| relative to the descent already travelled |h_0 - h_now| has fallen
      below ``remaining_descent_frac`` -- i.e. the trajectory has essentially reached its
      within-stage linear asymptote, so a terminal quadratic SOLVE (#341) would capture the
      rest and further first-order descent is spent; OR
  (b) HANDOFF-ETA condition: the NCDE predicts the #315 plateau-slope criterion
      (|rel slope| <= plateau_slope_eps) will be met within ``handoff_horizon_epochs``.

It does NOT fire on a still-descending series (large predicted remaining descent) nor on an
unstable/ill-conditioned fit (guarded by ``min_r2`` + stability check) -- a NO-FAKE guard so
a bad fit cannot manufacture a green.

RECONCILIATION (compose, do not duplicate)
------------------------------------------
* ``powerlaw_exit.py`` fits a UNIVARIATE power-law/exponential TAIL to a single d_seg series.
  This NCDE is a DISTINCT model class: a MIMO linear controlled DE that models the coupled
  multi-observable state AND its response to the annealing control path, yielding per-metric
  asymptotes + the coupling matrix + the control-response. The two are complementary sensors;
  a downstream controller can cross-check (power-law tail-meat vs NCDE remaining-descent).
* Emits ``{stage: "ncde_trajectory", ...}`` advisory rows in the costate-shadow style; the
  shadow controller (``shadow_controller.py``) can ingest them as a new sensor later.
* The #341 basin-finisher is the natural CONSUMER of the BASIN fire.

FORMALIZATION_PENDING: the (A,B,c) estimator is textbook DMDc / linear-CDE; the two closed-
form predictive laws (asymptote = -A^{-1}(Bu+c); T_eps = -ln(eps)*tau_slow) are standard LTI
results. They are NOT yet registered as a canonical equation because their load-bearing
content -- the predicted-vs-empirical *residual* of plateau-epoch / asymptote across runs --
requires N>=5 completed n600 runs to anchor honestly (only 1-2 exist today -> calibration,
not validation). Registering a residual anchor off n=1-2 would be a premature/false anchor
(NO-FAKE surrogate trap). Re-open for canonical registration when >=5 completed run.logs
exist; see DAG FEED-ncde + ``tools/ncde_trajectory_probe.py``.

means != ends: advisory trajectory sensor, NOT a score; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_ADVISORY_AXIS = "[macOS-CPU advisory] NON-PROMOTABLE"

# Minimum transitions above the parameter count for a non-degenerate ridge fit. A linear CDE
# with d states + m controls + 1 affine term has d*(d+m+1) parameters solved column-wise
# (d+m+1 per output); we require at least PARAM_MARGIN transitions beyond (d+m+1) so the
# per-column ridge system is over-determined even at lambda->0.
_PARAM_MARGIN = 2


# ----------------------------------------------------------------------------- data container
@dataclass(frozen=True)
class TelemetryPath:
    """A multivariate telemetry path: state observables + exogenous control drivers.

    ``epochs``: (N,) monotonically increasing epoch coordinates.
    ``state``:  (N, d) observable matrix (e.g. [log d_seg, log ep_loss, ...]).
    ``control``:(N, m) exogenous control matrix (e.g. [softmax_temp, hosc_beta]); may be (N,0).
    """

    epochs: np.ndarray
    state: np.ndarray
    control: np.ndarray
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]

    def __post_init__(self) -> None:
        e = np.asarray(self.epochs, dtype=float).reshape(-1)
        s = np.asarray(self.state, dtype=float)
        c = np.asarray(self.control, dtype=float)
        if s.ndim != 2:
            raise ValueError("state must be 2-D (N, d)")
        if c.ndim != 2:
            raise ValueError("control must be 2-D (N, m)")
        n = e.shape[0]
        if s.shape[0] != n or c.shape[0] != n:
            raise ValueError(
                f"row mismatch: epochs={n} state={s.shape[0]} control={c.shape[0]}"
            )
        if s.shape[1] != len(self.state_names):
            raise ValueError("state_names length != state columns")
        if c.shape[1] != len(self.control_names):
            raise ValueError("control_names length != control columns")
        if n >= 2 and np.any(np.diff(e) <= 0):
            raise ValueError("epochs must be strictly increasing")
        object.__setattr__(self, "epochs", e)
        object.__setattr__(self, "state", s)
        object.__setattr__(self, "control", c)

    @property
    def n(self) -> int:
        return self.epochs.shape[0]

    @property
    def d(self) -> int:
        return self.state.shape[1]

    @property
    def m(self) -> int:
        return self.control.shape[1]

    def is_finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.state)) and np.all(np.isfinite(self.control))
        )

    def window(self, start_idx: int, end_idx: int) -> "TelemetryPath":
        """Return the sub-path [start_idx, end_idx) (half-open)."""
        sl = slice(start_idx, end_idx)
        return TelemetryPath(
            epochs=self.epochs[sl].copy(),
            state=self.state[sl].copy(),
            control=self.control[sl].copy(),
            state_names=self.state_names,
            control_names=self.control_names,
        )


# ------------------------------------------------------------------------------- the estimator
def _ridge_solve(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Closed-form ridge: C = (Phi^T Phi + ridge I)^{-1} Phi^T Y. Shapes (P,), (N,P)/(N,d).

    Scales the ridge by the mean Gram diagonal so a constant control column (collinear with the
    affine bias) is regularized proportionally rather than under-damped -> a finite, well-posed
    SOLVE. A degenerate window that still yields non-finite coefficients is rejected by ``fit``.
    """
    p = phi.shape[1]
    gram = phi.T @ phi
    scale = float(np.mean(np.diag(gram))) or 1.0
    gram = gram + float(ridge) * scale * np.eye(p)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.linalg.solve(gram, phi.T @ y)


@dataclass(frozen=True)
class LinearNCDE:
    """A fitted linear controlled differential equation dh/dt = A h + B u + c."""

    A: np.ndarray  # (d, d)
    B: np.ndarray  # (d, m)
    c: np.ndarray  # (d,)
    dt: float  # mean epoch spacing used to normalize the derivative target
    ridge: float
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    n_fit: int
    rss: float
    r2: float
    h0: np.ndarray  # first state of the fit window (descent-origin reference)
    epoch_span: tuple[float, float]

    # --- fitting -------------------------------------------------------------
    @classmethod
    def fit(cls, path: TelemetryPath, *, ridge: float = 1e-3, dt: float | None = None) -> "LinearNCDE":
        if not path.is_finite():
            raise ValueError("telemetry path has non-finite values -- clean before fit")
        d, m = path.d, path.m
        n = path.n
        n_trans = n - 1
        n_params = d + m + 1
        if n_trans < n_params + _PARAM_MARGIN:
            raise ValueError(
                f"insufficient data: {n_trans} transitions < {n_params + _PARAM_MARGIN} "
                f"required for a {d}-state {m}-control fit"
            )
        de = np.diff(path.epochs)
        dt_eff = float(np.mean(de)) if dt is None else float(dt)
        if dt_eff <= 0:
            raise ValueError("dt must be positive")
        h = path.state
        u = path.control
        # derivative target per-transition, normalized by the ACTUAL local spacing so uneven
        # verdict cadences do not bias the estimator.
        y = (h[1:] - h[:-1]) / de[:, None]  # (N-1, d)
        ones = np.ones((n_trans, 1))
        phi = np.hstack([h[:-1], u[:-1], ones])  # (N-1, d+m+1)
        coef = _ridge_solve(phi, y, ridge)  # (d+m+1, d)
        if not np.all(np.isfinite(coef)):
            raise ValueError("degenerate window: non-finite ridge coefficients")
        A = coef[:d, :].T  # (d, d)
        B = coef[d : d + m, :].T  # (d, m)
        c = coef[d + m, :].copy()  # (d,)
        resid = y - phi @ coef
        rss = float(np.sum(resid**2))
        tss = float(np.sum((y - y.mean(axis=0)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        return cls(
            A=A, B=B, c=c, dt=dt_eff, ridge=float(ridge),
            state_names=path.state_names, control_names=path.control_names,
            n_fit=n_trans, rss=rss, r2=float(r2),
            h0=h[0].copy(), epoch_span=(float(path.epochs[0]), float(path.epochs[-1])),
        )

    # --- prediction / analysis ----------------------------------------------
    def predict(self, h_now: np.ndarray, u_future: np.ndarray, n_steps: int) -> np.ndarray:
        """Forward-roll the CDE from ``h_now`` for ``n_steps`` at the fit dt.

        ``u_future``: (n_steps, m) control values per step (held/scheduled). Returns (n_steps, d).
        """
        h_now = np.asarray(h_now, dtype=float).reshape(-1)
        if h_now.shape[0] != self.A.shape[0]:
            raise ValueError("h_now dimension mismatch")
        u_future = np.asarray(u_future, dtype=float).reshape(n_steps, -1)
        if u_future.shape[1] != self.B.shape[1]:
            raise ValueError("u_future control dimension mismatch")
        out = np.empty((n_steps, self.A.shape[0]))
        h = h_now.copy()
        # Bound the roll: an unstable/ill-conditioned fit can diverge under extrapolation.
        # Cap the magnitude so a NaN/inf blow-up never propagates (the caller reads this as a
        # large residual, which is the honest signal that the fit does not extrapolate).
        cap = 1e6 * (np.max(np.abs(h_now)) + 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(n_steps):
                dh = self.A @ h + self.B @ u_future[k] + self.c
                h = np.clip(h + self.dt * dh, -cap, cap)
                h = np.nan_to_num(h, nan=cap, posinf=cap, neginf=-cap)
                out[k] = h
        return out

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def is_stable(self) -> bool:
        """A is stable (all modes decaying) iff every eigenvalue has Re < 0."""
        return bool(np.all(np.real(self.eigenvalues()) < 0.0))

    def slowest_time_constant(self) -> float | None:
        """tau = -1/max Re(eig) over stable modes (epoch units). None if no stable mode."""
        re = np.real(self.eigenvalues())
        stable = re[re < 0.0]
        if stable.size == 0:
            return None
        slowest = float(np.max(stable))  # closest to 0 (slowest decay)
        return -1.0 / slowest

    def asymptote(self, u_star: np.ndarray) -> np.ndarray | None:
        """Fixed point h_inf = -A^{-1}(B u_star + c). None if A singular/unstable."""
        if not self.is_stable():
            return None
        u_star = np.asarray(u_star, dtype=float).reshape(-1)
        if u_star.shape[0] != self.B.shape[1]:
            raise ValueError("u_star control dimension mismatch")
        rhs = self.B @ u_star + self.c
        try:
            return -np.linalg.solve(self.A, rhs)
        except np.linalg.LinAlgError:
            return None

    def plateau_epochs(self, eps: float = 0.05, u_star: np.ndarray | None = None) -> float | None:
        """Predicted epochs to reach within relative ``eps`` of the asymptote."""
        if not (0.0 < eps < 1.0):
            raise ValueError("eps must be in (0,1)")
        tau = self.slowest_time_constant()
        if tau is None:
            return None
        return -math.log(eps) * tau

    def to_dict(self) -> dict:
        eig = self.eigenvalues()
        return {
            "state_names": list(self.state_names),
            "control_names": list(self.control_names),
            "dt": self.dt,
            "ridge": self.ridge,
            "n_fit": self.n_fit,
            "r2": self.r2,
            "rss": self.rss,
            "stable": self.is_stable(),
            "slowest_time_constant_ep": self.slowest_time_constant(),
            "eig_real": [float(x) for x in np.real(eig)],
            "eig_imag": [float(x) for x in np.imag(eig)],
            "epoch_span": list(self.epoch_span),
        }


# ------------------------------------------------------------------------ hit->solve detector
@dataclass(frozen=True)
class HitSolveConfig:
    """Thresholds for the hit->solve advisory. All shadow-only."""

    target_state: str = "log_d_seg"
    remaining_descent_frac: float = 0.05  # basin: remaining/travelled below this -> fire
    handoff_horizon_epochs: float = 50.0  # eta-handoff window
    plateau_slope_eps: float = 1e-4  # #315 rel-slope handoff criterion
    min_r2: float = 0.5  # NO-FAKE fit-quality guard
    eps_plateau: float = 0.05  # asymptote-approach fraction for T_eps


@dataclass(frozen=True)
class HitSolveVerdict:
    fire: bool
    reason: str
    target_name: str
    current_value: float
    predicted_asymptote: float | None
    predicted_remaining_descent: float | None
    remaining_descent_frac: float | None
    predicted_plateau_epochs: float | None
    eta_handoff_epochs: float | None
    fit_r2: float
    n_fit: int
    stable: bool

    def to_row(self, epoch: float, stage: str | None = None) -> dict:
        return {
            "stage": "ncde_trajectory",
            "epoch": float(epoch),
            "seg_form": stage,
            "fire": self.fire,
            "reason": self.reason,
            "target": self.target_name,
            "current_value": self.current_value,
            "predicted_asymptote": self.predicted_asymptote,
            "predicted_remaining_descent": self.predicted_remaining_descent,
            "remaining_descent_frac": self.remaining_descent_frac,
            "predicted_plateau_epochs": self.predicted_plateau_epochs,
            "eta_handoff_epochs": self.eta_handoff_epochs,
            "fit_r2": self.fit_r2,
            "n_fit": self.n_fit,
            "stable": self.stable,
            "actuation": "NONE",
            "axis": _ADVISORY_AXIS,
        }


def _slope_eta_epochs(
    ncde: LinearNCDE, h_now: np.ndarray, u_star: np.ndarray, target_idx: int,
    slope_eps: float, horizon: float,
) -> float | None:
    """Predicted epochs until |d(target)/dt| / |target| <= slope_eps, by forward roll.

    Returns None if the horizon is exhausted without the slope crossing the threshold.
    """
    n_steps = max(1, int(math.ceil(horizon / max(ncde.dt, 1e-9))))
    u_future = np.tile(np.asarray(u_star, dtype=float).reshape(1, -1), (n_steps, 1))
    traj = ncde.predict(h_now, u_future, n_steps)
    prev = float(h_now[target_idx])
    for k in range(n_steps):
        cur = float(traj[k, target_idx])
        rel_slope = abs(cur - prev) / (abs(cur) + 1e-9)
        if rel_slope <= slope_eps:
            return float((k + 1) * ncde.dt)
        prev = cur
    return None


def detect_hit_solve(
    ncde: LinearNCDE,
    h_now: np.ndarray,
    u_star: np.ndarray,
    cfg: HitSolveConfig = HitSolveConfig(),
) -> HitSolveVerdict:
    """Evaluate the hit->solve advisory on ``ncde`` at the current state ``h_now``.

    Guarded: an unstable or low-r2 fit NEVER fires (NO-FAKE -- a bad fit cannot green).
    """
    h_now = np.asarray(h_now, dtype=float).reshape(-1)
    if cfg.target_state not in ncde.state_names:
        raise ValueError(f"target_state {cfg.target_state!r} not in {ncde.state_names}")
    tidx = ncde.state_names.index(cfg.target_state)
    cur = float(h_now[tidx])
    stable = ncde.is_stable()
    r2 = ncde.r2

    # NO-FAKE guard: refuse to interpret an untrustworthy instrument.
    if not stable or r2 < cfg.min_r2:
        return HitSolveVerdict(
            fire=False,
            reason=(
                f"NO-FIRE: fit not trustworthy (stable={stable}, r2={r2:.3f} "
                f"< min {cfg.min_r2}); instrument-invalid, no verdict"
            ),
            target_name=cfg.target_state, current_value=cur,
            predicted_asymptote=None, predicted_remaining_descent=None,
            remaining_descent_frac=None, predicted_plateau_epochs=None,
            eta_handoff_epochs=None, fit_r2=r2, n_fit=ncde.n_fit, stable=stable,
        )

    asym = ncde.asymptote(u_star)
    asym_t = float(asym[tidx]) if asym is not None else None
    remaining = abs(cur - asym_t) if asym_t is not None else None
    travelled = abs(float(ncde.h0[tidx]) - cur)
    rem_frac = (remaining / travelled) if (remaining is not None and travelled > 1e-12) else None
    t_eps = ncde.plateau_epochs(cfg.eps_plateau, u_star)
    eta_handoff = _slope_eta_epochs(
        ncde, h_now, u_star, tidx, cfg.plateau_slope_eps, cfg.handoff_horizon_epochs
    )

    basin = rem_frac is not None and rem_frac <= cfg.remaining_descent_frac
    handoff_soon = eta_handoff is not None and eta_handoff <= cfg.handoff_horizon_epochs

    if basin:
        reason = (
            f"BASIN: predicted remaining within-stage descent {rem_frac:.4f} <= "
            f"{cfg.remaining_descent_frac} of travelled -> terminal SOLVE (#341) admissible; "
            f"first-order descent spent"
        )
    elif handoff_soon:
        reason = (
            f"HANDOFF-ETA: #315 plateau-slope predicted within {eta_handoff:.1f} ep "
            f"(<= horizon {cfg.handoff_horizon_epochs})"
        )
    else:
        reason = (
            f"NO-FIRE: still descending (remaining_frac="
            f"{rem_frac if rem_frac is not None else float('nan'):.4f}, "
            f"eta_handoff={eta_handoff})"
        )

    return HitSolveVerdict(
        fire=bool(basin or handoff_soon), reason=reason,
        target_name=cfg.target_state, current_value=cur,
        predicted_asymptote=asym_t, predicted_remaining_descent=remaining,
        remaining_descent_frac=rem_frac, predicted_plateau_epochs=t_eps,
        eta_handoff_epochs=eta_handoff, fit_r2=r2, n_fit=ncde.n_fit, stable=stable,
    )


# -------------------------------------------------------------------------- sliding-window API
@dataclass(frozen=True)
class WindowFit:
    end_epoch: float
    ncde: LinearNCDE
    verdict: HitSolveVerdict | None = None


def fit_sliding_windows(
    path: TelemetryPath, *, window: int, stride: int = 1, ridge: float = 1e-3,
    detector_cfg: HitSolveConfig | None = None, u_star: np.ndarray | None = None,
) -> list[WindowFit]:
    """Fit a LinearNCDE on each sliding window [i, i+window) and (optionally) run the detector.

    Skips windows that raise (insufficient/degenerate data) -- honest partial coverage.
    """
    if window < 5:
        raise ValueError("window must be >= 5 to admit a non-degenerate fit")
    out: list[WindowFit] = []
    for start in range(0, path.n - window + 1, stride):
        sub = path.window(start, start + window)
        try:
            ncde = LinearNCDE.fit(sub, ridge=ridge)
        except (ValueError, np.linalg.LinAlgError):
            continue
        verdict = None
        if detector_cfg is not None:
            us = u_star if u_star is not None else sub.control[-1]
            try:
                verdict = detect_hit_solve(ncde, sub.state[-1], us, detector_cfg)
            except ValueError:
                verdict = None
        out.append(WindowFit(end_epoch=float(sub.epochs[-1]), ncde=ncde, verdict=verdict))
    return out


def backtest_holdout(
    path: TelemetryPath, *, train_frac: float = 0.6, ridge: float = 1e-3,
    target_state: str | None = None,
) -> dict:
    """Honest holdout backtest: fit on the first ``train_frac`` of the path, roll the CDE
    forward across the held-out tail using the tail's recorded control, and report the
    predicted-vs-actual MAPE per state (and for ``target_state`` if named).

    n=1-2 completed runs -> this is CALIBRATION, not validation. The caller MUST label it so.
    """
    n = path.n
    n_train = max(6, int(round(train_frac * n)))
    if n_train >= n - 1:
        raise ValueError("not enough held-out points for a backtest")
    train = path.window(0, n_train)
    ncde = LinearNCDE.fit(train, ridge=ridge)
    tail = path.window(n_train - 1, n)  # start at last train point (continuity)
    n_steps = tail.n - 1
    pred = ncde.predict(tail.state[0], tail.control[:-1], n_steps)  # (n_steps, d)
    actual = tail.state[1:]  # (n_steps, d)
    denom = np.abs(actual) + 1e-9
    mape_per_state = np.mean(np.abs(pred - actual) / denom, axis=0)  # (d,)
    result = {
        "n_total": n, "n_train": n_train, "n_holdout": n_steps,
        "train_r2": ncde.r2, "stable": ncde.is_stable(),
        "slowest_time_constant_ep": ncde.slowest_time_constant(),
        "mape_per_state": {
            name: float(v) for name, v in zip(path.state_names, mape_per_state)
        },
        "label": "CALIBRATION (n<=2 completed runs) -- NOT validation",
        "axis": _ADVISORY_AXIS,
    }
    if target_state is not None and target_state in path.state_names:
        ti = path.state_names.index(target_state)
        result["target_state"] = target_state
        result["target_mape"] = float(mape_per_state[ti])
        result["target_pred_final"] = float(pred[-1, ti])
        result["target_actual_final"] = float(actual[-1, ti])
    return result


def sliding_onestep_backtest(
    path: TelemetryPath, *, window: int, ridge: float = 1e-3, target_state: str | None = None,
) -> dict:
    """The sensor's REAL operating accuracy: fit on [i, i+window), predict the ONE next step
    from the recorded control, compare to the actual next state. Averaged over all windows.

    This is the faithful backtest for a sliding-window sensor (unlike a single global holdout,
    which stresses one linear CDE across a whole multi-stage run). n=1-2 runs -> CALIBRATION.
    """
    if window < 5:
        raise ValueError("window must be >= 5")
    errs: list[np.ndarray] = []
    tgt_errs: list[float] = []
    ti = path.state_names.index(target_state) if target_state in path.state_names else None
    n_windows = 0
    for start in range(0, path.n - window):  # need one step beyond the window
        sub = path.window(start, start + window)
        try:
            ncde = LinearNCDE.fit(sub, ridge=ridge)
        except (ValueError, np.linalg.LinAlgError):
            continue
        nxt = path.state[start + window]  # the true next observation
        nxt_ctrl = path.control[start + window - 1]  # control at the last in-window step
        pred = ncde.predict(sub.state[-1], nxt_ctrl.reshape(1, -1), 1)[0]
        denom = np.abs(nxt) + 1e-9
        errs.append(np.abs(pred - nxt) / denom)
        if ti is not None:
            tgt_errs.append(float(abs(pred[ti] - nxt[ti]) / (abs(nxt[ti]) + 1e-9)))
        n_windows += 1
    if not errs:
        return {"n_windows": 0, "note": "no fittable windows", "axis": _ADVISORY_AXIS}
    mape = np.mean(np.vstack(errs), axis=0)
    out = {
        "n_windows": n_windows, "window": window,
        "onestep_mape_per_state": {
            name: float(v) for name, v in zip(path.state_names, mape)
        },
        "label": "CALIBRATION (n<=2 completed runs) -- NOT validation",
        "axis": _ADVISORY_AXIS,
    }
    if ti is not None and tgt_errs:
        out["target_state"] = target_state
        out["target_onestep_mape"] = float(np.mean(tgt_errs))
    return out


__all__ = [
    "TelemetryPath",
    "LinearNCDE",
    "HitSolveConfig",
    "HitSolveVerdict",
    "WindowFit",
    "detect_hit_solve",
    "fit_sliding_windows",
    "backtest_holdout",
    "sliding_onestep_backtest",
]
