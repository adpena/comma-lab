# SPDX-License-Identifier: MIT
"""Unified-ξ estimator-agnostic ego-trajectory seam + the lane-coeff ADVECT operator.

This is the P1 seam of the unified-ξ build (design authority
``unified_xi_design_and_adversarial_review_20260702.md``). There is exactly ONE
shared object -- the ego trajectory ``ξ_ego(t)`` -- and exactly ONE interface
between *how we estimate it* (swappable ``EgoEstimator``) and *what consumes it*
(fixed): the pose warm-start (``warp_real_luma_frame0``) and the lane-coeff
predictor (this module's ``advect_slot_matrix``, consumed by the LBND3
predictive-coding codec in ``analytic_lane_render_band``).

THE DECISIVE REFRAME (design revision #1): L1 is NOT store-in-world-frame +
invertible warp. It is ego-motion-compensated PREDICTIVE CODING (P-frame): the
predictor advects the DECODED previous pair's lane coeffs by ξ, and only the
INNOVATION is coded. No inverse-warp is ever required -> the exact-invertibility
determinism hazard is deleted. ``advect_slot_matrix`` is the sole fp path both
compress and decode call, on identical inputs -> bit-identical by construction
(the numpy-fp64 authority; ZERO mlx/metal in inflate).

Estimator error costs RATE (larger innovation) or pose warm-start CONVERGENCE,
NEVER d_seg/d_pose correctness -- the innovation carries the full residual
losslessly (design §5.4, fault-tolerance). So any estimator is safe to try; the
bake-off just picks the cheapest.

rule-118 accounting: the estimator (VO/SfM/PoseNet-head) runs OFFLINE at compress
time (FREE); ONLY the tiny per-pair planar ego statistic ``(ds, dpsi)`` (or the
SE(3) B-spline control poses that compress it) is COUNTED; the advect + spline
eval at decode are the FREE generic numpy algorithm.

Conventions (fixed once, tested)
--------------------------------
* Lane ground frame: ``lateral(m) = polyval(centerline_coeffs, forward_m)`` (highest-
  degree-first; ``lane_sdf_component`` / ``analytic_lane_render_band`` convention).
* Ego planar motion between consecutive pairs is the FULL 3-DOF SE(2): ``ds`` =
  forward distance driven (metres, >0 = forward), ``dy`` = lateral displacement
  (metres), ``dpsi`` = yaw (radians). The predicted centerline under this ego
  motion is ``c_t(f) = c_{t-1}(f + ds) + dy + dpsi * f`` -- the forward Taylor shift
  (exact), the lateral bulk offset (``+ dy``, absorbed into c0), and the yaw
  first-order term (``+ dpsi * f``, absorbed into c1). The lateral DOF is decisive:
  the ego STEERS to follow the lane, so its lateral offset barely changes even
  as the curve sweeps -- WITHOUT ``dy`` the forward shift over-predicts the c0 bulk
  motion and the innovation GROWS (measured 2026-07-02). ``forward_range`` shifts by
  ``-ds``. Halfwidth / dash period+duty are identity-advected (world-invariant or
  image-frame) -> their innovation is the raw temporal delta (== LBND2).
* SE(3) twist ``xi = (rho, omega)`` translation-first (``tac.lie.CONVENTION``);
  camera x = right (lateral), y = down, z = forward; yaw = rotation about the
  vertical (y) axis. The planar embedding of ``(ds, dy, dpsi)`` is
  ``xi = (dy, 0, ds, 0, dpsi, 0)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from tac.boundary_math.analytic_lane_render_band import _RD_D_SLOT

__all__ = [
    "XiEgoTrajectory",
    "EgoEstimator",
    "taylor_shift_cubic",
    "advect_centerline_coeffs",
    "advect_slot_vec",
    "advect_slot_matrix",
    "LaneOptimalEgoEstimator",
    "PoseTargetEgoEstimator",
    "ConstantVelocityEgoEstimator",
    "fit_se3_bspline_controls",
    "bspline_fit_error_curve",
]

# Slot schema (must match analytic_lane_render_band._line_to_slot_vec):
#   [c3, c2, c1, c0,  hw1, hw0,  dp, dph, dd,  fr0, fr1]
_C_HI = 0   # centerline highest-degree index
_C_LO = 4   # centerline slice end (exclusive) -> [0:4]
_FR0 = 9
_FR1 = 10


# --------------------------------------------------------------------------- #
# The ONE shared object: the ego trajectory (the counted-tiny sufficient stat).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class XiEgoTrajectory:
    """Estimator-agnostic per-pair ego trajectory (the ONE counted seam).

    Canonical payload = the per-pair planar ego statistic ``(ds, dpsi)`` that the
    lane predictor consumes, PLUS the full per-pair SE(3) twist ``dense_xi`` that
    the pose warm-start (``warp_real_luma_frame0``) consumes. ``ds``/``dpsi`` are
    the primary lane-advect knobs; ``dense_xi`` is the planar embedding (or a
    physically-calibrated 6-DOF) for the pose readout. All fp64 (decode authority).

    ``ds[t]`` / ``dpsi[t]`` are the RELATIVE ego motion from pair ``t-1`` to pair
    ``t`` (``ds[0] = dpsi[0] = 0`` by convention: pair 0 has no predecessor).
    """

    n_pairs: int
    ds: np.ndarray            # (n_pairs,) fp64 -- forward distance driven pair t-1 -> t (m)
    dy: np.ndarray            # (n_pairs,) fp64 -- lateral displacement pair t-1 -> t (m)
    dpsi: np.ndarray          # (n_pairs,) fp64 -- yaw pair t-1 -> t (rad)
    dense_xi: np.ndarray      # (n_pairs, 6) fp64 -- SE(3) twist (rho,omega) for the pose readout
    estimator_id: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        P = int(self.n_pairs)
        for name, arr, shp in (("ds", self.ds, (P,)), ("dy", self.dy, (P,)), ("dpsi", self.dpsi, (P,)),
                               ("dense_xi", self.dense_xi, (P, 6))):
            a = np.asarray(arr)
            if a.shape != shp:
                raise ValueError(f"XiEgoTrajectory.{name} shape {a.shape} != {shp}")
            if a.dtype != np.float64:
                raise ValueError(f"XiEgoTrajectory.{name} must be float64 (decode authority); got {a.dtype}")
        if not self.estimator_id:
            raise ValueError("XiEgoTrajectory.estimator_id is required (provenance / rule-118).")

    @staticmethod
    def from_planar(ds: np.ndarray, dy: np.ndarray, dpsi: np.ndarray, estimator_id: str,
                    provenance: dict[str, Any] | None = None) -> "XiEgoTrajectory":
        """Build from per-pair planar ``(ds, dy, dpsi)``; ``dense_xi`` = planar embedding
        ``(dy, 0, ds, 0, dpsi, 0)`` (lateral + forward translation + yaw)."""
        ds = np.asarray(ds, np.float64).reshape(-1)
        dy = np.asarray(dy, np.float64).reshape(-1)
        dpsi = np.asarray(dpsi, np.float64).reshape(-1)
        P = int(ds.shape[0])
        if dy.shape[0] != P or dpsi.shape[0] != P:
            raise ValueError("ds, dy, dpsi must have equal length.")
        dense = np.zeros((P, 6), np.float64)
        dense[:, 0] = dy       # rho_x = lateral
        dense[:, 2] = ds       # rho_z = forward
        dense[:, 4] = dpsi     # omega_y = yaw
        return XiEgoTrajectory(n_pairs=P, ds=ds, dy=dy, dpsi=dpsi, dense_xi=dense,
                               estimator_id=str(estimator_id), provenance=dict(provenance or {}))


class EgoEstimator(Protocol):
    """The swappable producer side of the ONE seam. Any estimator implements this
    single method and returns the estimator-agnostic ``XiEgoTrajectory``."""

    def estimate(self, *args: Any, **kwargs: Any) -> XiEgoTrajectory: ...


# --------------------------------------------------------------------------- #
# The ADVECT operator -- the sole fp path both compress and decode call.
# --------------------------------------------------------------------------- #
def taylor_shift_cubic(coeffs4: np.ndarray, ds: float) -> np.ndarray:
    """``p(f) -> p(f + ds)`` for a degree-3 poly stored highest-first ``[c3,c2,c1,c0]``.

    Exact fp64 Taylor (Pascal) shift -- the closed-form centerline advect under a
    forward ego advance ``ds``. Bit-identical both sides (compress & decode).
    """
    c = np.asarray(coeffs4, np.float64)
    if c.shape != (4,):
        raise ValueError(f"taylor_shift_cubic expects (4,) highest-first coeffs; got {c.shape}")
    c3, c2, c1, c0 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
    d = float(ds)
    d2 = d * d
    d3 = d2 * d
    return np.array(
        [c3,
         c2 + 3.0 * c3 * d,
         c1 + 2.0 * c2 * d + 3.0 * c3 * d2,
         c0 + c1 * d + c2 * d2 + c3 * d3],
        dtype=np.float64,
    )


def advect_centerline_coeffs(coeffs4: np.ndarray, ds: float, dy: float, dpsi: float) -> np.ndarray:
    """Advect a centerline ``[c3,c2,c1,c0]`` by the planar SE(2) ego motion: forward
    ``ds`` (exact Taylor shift ``c(f+ds)``) + lateral ``dy`` (bulk offset -> ``c0 += dy``)
    + yaw ``dpsi`` (first-order lateral-linear term ``+ dpsi*f`` -> ``c1 += dpsi``)."""
    out = taylor_shift_cubic(coeffs4, ds)
    out[2] = out[2] + float(dpsi)   # c1 (linear) slot += dpsi (yaw)
    out[3] = out[3] + float(dy)     # c0 (constant) slot += dy (lateral)
    return out


def advect_slot_vec(vec11: np.ndarray, ds: float, dy: float, dpsi: float) -> np.ndarray:
    """Advect ONE 11-dim slot vector by ``(ds, dy, dpsi)``.

    Centerline ``[0:4]`` -> Taylor-shift + lateral + yaw; forward_range ``[9], [10]`` ->
    ``-ds``; halfwidth / dash-period / dash-phase / dash-duty -> IDENTITY (world-invariant
    or image-frame; their innovation reduces to the LBND2 raw temporal delta). Exact fp64.
    """
    v = np.asarray(vec11, np.float64).copy()
    if v.shape != (_RD_D_SLOT,):
        raise ValueError(f"advect_slot_vec expects ({_RD_D_SLOT},); got {v.shape}")
    v[_C_HI:_C_LO] = advect_centerline_coeffs(v[_C_HI:_C_LO], ds, dy, dpsi)
    v[_FR0] = v[_FR0] - float(ds)
    v[_FR1] = v[_FR1] - float(ds)
    return v


def advect_slot_matrix(row_prev: np.ndarray, ds: float, dy: float, dpsi: float,
                       K: int, present_mask: np.ndarray | None = None) -> np.ndarray:
    """Advect a packed ``(K*11,)`` matrix row by ``(ds, dy, dpsi)``, slot-wise.

    ``present_mask`` (K,) bool: advect only slots that are PRESENT at the target pair
    (a held/absent slot keeps identity -> zero innovation during a hold, matching the
    LBND2 carry-forward semantics; a slot reappearing after a hold is advected by the
    single-pair motion from its last-observed value -- a rate cost, not a correctness
    bug). ``present_mask=None`` -> advect all slots (LBND3-full).
    """
    row = np.asarray(row_prev, np.float64).copy()
    if row.shape != (K * _RD_D_SLOT,):
        raise ValueError(f"advect_slot_matrix expects ({K * _RD_D_SLOT},); got {row.shape}")
    if K == 0:
        return row
    for slot in range(K):
        if present_mask is not None and not bool(present_mask[slot]):
            continue
        sl = slice(slot * _RD_D_SLOT, (slot + 1) * _RD_D_SLOT)
        row[sl] = advect_slot_vec(row[sl], ds, dy, dpsi)
    return row


# --------------------------------------------------------------------------- #
# Estimator (i-lane-optimal): fit (ds, dpsi) from the lane geometry itself.
# --------------------------------------------------------------------------- #
def _fit_planar_ego_from_centerlines(
    prev_slots: list[np.ndarray], cur_slots: list[np.ndarray],
    *, f_samples: np.ndarray, fit_forward: bool,
) -> tuple[float, float, float]:
    """Linearized per-pair LSQ for the shared 3-DOF planar ego ``(ds, dy, dpsi)`` matching
    pair-t's centerlines to pair-(t-1)'s under the advect model. For each matched slot the
    innovation ``c_t(f) - c_{t-1}(f)`` is, to first order,
    ``ds * c'_{t-1}(f)  +  dy * 1  +  dpsi * f`` -> stack the linear system over all shared
    samples and solve (numpy lstsq). The lateral DOF (``dy``, the constant basis) is
    decisive: the ego steers to follow the lane, so its lateral offset absorbs the c0 bulk
    motion the forward shift would otherwise over-predict.

    ``fit_forward=False`` drops the ``ds`` column (2-DOF affine dy+yaw) -- the ablation
    that isolates whether the forward Taylor shift helps or hurts. The APPLIED advect is
    always the EXACT operator; the linearization only ESTIMATES the scalars. Returns
    ``(0,0,0)`` if too few shared samples.
    """
    A_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for cp, cc in zip(prev_slots, cur_slots):
        cp = np.asarray(cp, np.float64)
        gp = np.polyval(np.polyder(cp), f_samples)               # c'_{t-1}(f) (forward basis)
        cp_v = np.polyval(cp, f_samples)
        cc_v = np.polyval(np.asarray(cc, np.float64), f_samples)
        ones = np.ones_like(f_samples)                           # lateral basis
        cols = [gp, ones, f_samples] if fit_forward else [ones, f_samples]
        A_rows.append(np.stack(cols, axis=1))
        b_rows.append(cc_v - cp_v)                               # innovation target
    if not A_rows:
        return 0.0, 0.0, 0.0
    A = np.concatenate(A_rows, axis=0)
    b = np.concatenate(b_rows, axis=0)
    if A.shape[0] < A.shape[1]:
        return 0.0, 0.0, 0.0
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    if fit_forward:
        return float(sol[0]), float(sol[1]), float(sol[2])       # ds, dy, dpsi
    return 0.0, float(sol[0]), float(sol[1])                      # dy, dpsi (ds=0)


class LaneOptimalEgoEstimator:
    """Estimator (i): the LANE-OPTIMAL per-axis ξ. Fits per-pair 3-DOF planar ego
    ``(ds, dy, dpsi)`` that best predicts pair-t's centerlines from pair-(t-1)'s under the
    advect model (lane-data-only, deterministic, $0/offline). This is the ACHIEVABLE-FLOOR
    predictor for the lane-rate axis AND the lane-optimal per-axis optimum the
    shared-vs-per-axis bake-off (design §5.5) must report. NOT circular: it is a
    legitimate estimator whose ξ is one of the two per-axis extremes.

    Consumes the PACKED L4 matrix (from ``_pack_pairs_to_matrix``) so the slot
    correspondence is the canonical lateral-sorted one the codec uses. ``fit_forward``
    toggles the forward Taylor shift (the ablation); ``smooth_win`` optionally
    median-smooths the per-pair estimate (a smooth ego trajectory codes cheaper).
    """

    def __init__(self, *, f_lo: float = 6.0, f_hi: float = 45.0, n_samples: int = 24,
                 ds_clip: float = 30.0, dy_clip: float = 5.0, dpsi_clip: float = 0.35,
                 fit_forward: bool = True, smooth_win: int = 0) -> None:
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)
        self.n_samples = int(n_samples)
        self.ds_clip = float(ds_clip)
        self.dy_clip = float(dy_clip)
        self.dpsi_clip = float(dpsi_clip)
        self.fit_forward = bool(fit_forward)
        self.smooth_win = int(smooth_win)

    def estimate(self, M: np.ndarray, presence: np.ndarray, K: int) -> XiEgoTrajectory:
        """M ``(P, K*11)`` packed matrix, presence ``(P, K)`` bool, K slots."""
        M = np.asarray(M, np.float64)
        presence = np.asarray(presence, bool)
        P = int(M.shape[0])
        f = np.linspace(self.f_lo, self.f_hi, self.n_samples)
        ds = np.zeros(P, np.float64)
        dy = np.zeros(P, np.float64)
        dpsi = np.zeros(P, np.float64)
        for t in range(1, P):
            shared = np.where(presence[t] & presence[t - 1])[0] if K else np.zeros(0, int)
            prev_c, cur_c = [], []
            for slot in shared:
                base = int(slot) * _RD_D_SLOT
                prev_c.append(M[t - 1, base:base + 4])
                cur_c.append(M[t, base:base + 4])
            d_s, d_y, d_psi = _fit_planar_ego_from_centerlines(
                prev_c, cur_c, f_samples=f, fit_forward=self.fit_forward)
            ds[t] = float(np.clip(d_s, -self.ds_clip, self.ds_clip))
            dy[t] = float(np.clip(d_y, -self.dy_clip, self.dy_clip))
            dpsi[t] = float(np.clip(d_psi, -self.dpsi_clip, self.dpsi_clip))
        if self.smooth_win >= 3:
            ds, dy, dpsi = (_median_smooth(a, self.smooth_win) for a in (ds, dy, dpsi))
            ds[0] = dy[0] = dpsi[0] = 0.0
        return XiEgoTrajectory.from_planar(
            ds, dy, dpsi, estimator_id="lane_optimal_lsq",
            provenance={"f_lo": self.f_lo, "f_hi": self.f_hi, "n_samples": self.n_samples,
                        "fit_forward": self.fit_forward, "smooth_win": self.smooth_win,
                        "method": "linearized_shared_planar_ego_lstsq"},
        )


def _median_smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centered median smoothing (odd window), edge-replicated. Deterministic."""
    x = np.asarray(x, np.float64)
    w = int(win) | 1
    h = w // 2
    xp = np.pad(x, h, mode="edge")
    return np.array([float(np.median(xp[i:i + w])) for i in range(x.size)], np.float64)


# --------------------------------------------------------------------------- #
# Estimator (ii-physical): calibrate the frozen PoseNet ego readout into (ds, dpsi).
# --------------------------------------------------------------------------- #
class PoseTargetEgoEstimator:
    """Estimator (ii): the PHYSICAL per-axis ξ from the frozen PoseNet ego readout
    (``gt_poses`` per pair -- the contest scorer's OWN ego-motion, LANE-INDEPENDENT).

    ``gt_poses`` is the PoseNet 6-vec OUTPUT, identifiable to metric ego motion only
    UP-TO-AFFINE (design §4 / LDM Thm 1). This estimator maps ``(forward_channel,
    yaw_channel)`` -> ``(ds, dpsi)`` by a per-clip affine calibration. Two modes:

    * ``calib='geometry'``: physically-grounded fixed scale (``s_t`` on the forward
      channel, ``s_r`` on the yaw channel) -- NO lane data used (the honest physical
      point).
    * ``calib='affine_to_lane'``: least-squares fit the 2x(scale,bias) affine map so
      the physical channels best match a reference lane-optimal ``(ds, dpsi)`` -- the
      up-to-affine calibration MEASURED (used to answer design Q4). Uses lane data
      only for the CALIBRATION scalars (tiny counted descriptor), not per-pair ξ.
    """

    def __init__(self, *, fwd_ch: int = 0, lat_ch: int = 2, yaw_ch: int = 4,
                 s_t: float = 1.0, s_r: float = 1.0, calib: str = "affine_to_lane") -> None:
        self.fwd_ch = int(fwd_ch)
        self.lat_ch = int(lat_ch)
        self.yaw_ch = int(yaw_ch)
        self.s_t = float(s_t)
        self.s_r = float(s_r)
        if calib not in ("geometry", "affine_to_lane"):
            raise ValueError("calib must be 'geometry' or 'affine_to_lane'.")
        self.calib = calib

    def estimate(self, gt_poses: np.ndarray,
                 ref: XiEgoTrajectory | None = None) -> XiEgoTrajectory:
        gp = np.asarray(gt_poses, np.float64)
        need = max(self.fwd_ch, self.lat_ch, self.yaw_ch) + 1
        if gp.ndim != 2 or gp.shape[1] < need:
            raise ValueError(f"gt_poses must be (P, >={need}); got {gp.shape}")
        P = int(gp.shape[0])
        fwd_raw, lat_raw, yaw_raw = gp[:, self.fwd_ch], gp[:, self.lat_ch], gp[:, self.yaw_ch]
        prov: dict[str, Any] = {"fwd_ch": self.fwd_ch, "lat_ch": self.lat_ch,
                                "yaw_ch": self.yaw_ch, "calib": self.calib}
        ds = np.zeros(P, np.float64); dy = np.zeros(P, np.float64); dpsi = np.zeros(P, np.float64)
        if self.calib == "geometry":
            ds[1:] = self.s_t * fwd_raw[1:]
            dy[1:] = self.s_t * lat_raw[1:]
            dpsi[1:] = self.s_r * yaw_raw[1:]
            prov.update({"s_t": self.s_t, "s_r": self.s_r})
        else:
            if ref is None:
                raise ValueError("calib='affine_to_lane' requires a reference XiEgoTrajectory.")
            a_ds, b_ds = _affine_fit(fwd_raw[1:], ref.ds[1:])
            a_dy, b_dy = _affine_fit(lat_raw[1:], ref.dy[1:])
            a_dp, b_dp = _affine_fit(yaw_raw[1:], ref.dpsi[1:])
            ds[1:] = a_ds * fwd_raw[1:] + b_ds
            dy[1:] = a_dy * lat_raw[1:] + b_dy
            dpsi[1:] = a_dp * yaw_raw[1:] + b_dp
            prov.update({"affine_ds": [a_ds, b_ds], "affine_dy": [a_dy, b_dy], "affine_dpsi": [a_dp, b_dp]})
        return XiEgoTrajectory.from_planar(ds, dy, dpsi, estimator_id=f"posenet_{self.calib}", provenance=prov)


def _affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """LSQ affine ``y ~= a*x + b`` (returns (a, b); (0, 0) if degenerate)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 2 or float(np.ptp(x)) < 1e-12:
        return 0.0, float(np.mean(y)) if y.size else 0.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


class ConstantVelocityEgoEstimator:
    """Estimator (iii, baseline): a physically-motivated constant forward speed, no
    yaw. ``ds[t] = ds_const`` for t>=1; ``dpsi = 0``. The zero-information physical
    prior (a genuine estimator with no per-pair data), for the bake-off's lower rung.
    """

    def __init__(self, ds_const: float) -> None:
        self.ds_const = float(ds_const)

    def estimate(self, n_pairs: int) -> XiEgoTrajectory:
        P = int(n_pairs)
        ds = np.full(P, self.ds_const, np.float64)
        ds[0] = 0.0
        return XiEgoTrajectory.from_planar(ds, np.zeros(P, np.float64), np.zeros(P, np.float64),
                                           estimator_id="const_velocity",
                                           provenance={"ds_const": self.ds_const})


# --------------------------------------------------------------------------- #
# SE(3) B-spline compression of the dense ξ (the COUNTED payload alternative) +
# the fit-error-vs-M curve (design Q5).
# --------------------------------------------------------------------------- #
def fit_se3_bspline_controls(dense_xi: np.ndarray, M: int) -> np.ndarray:
    """Fit ``M`` SE(3) control poses to the cumulative absolute ego trajectory implied
    by the per-pair relative twists ``dense_xi`` (n_pairs, 6).

    The absolute pose at pair ``k`` = compose of ``exp_se3(dense_xi[1..k])`` (pair 0 =
    identity). Control poses are SAMPLED from this absolute trajectory at ``M`` evenly
    spaced knot times (a simple, deterministic initial fit; the design's measured
    fit-error curve then picks the smallest ``M`` below tolerance). Uses the
    ``tac.lie`` numpy-fp64 oracle (the authority; ZERO mlx).
    """
    from tac.lie import _se3_numpy as se3

    dxi = np.asarray(dense_xi, np.float64)
    P = int(dxi.shape[0])
    if M < 4:
        raise ValueError("SE(3) cubic B-spline needs M >= 4 control poses.")
    T = np.eye(4, dtype=np.float64)
    abs_T = [T.copy()]
    for k in range(1, P):
        T = se3.compose(T, se3.exp_se3(dxi[k]))
        abs_T.append(T.copy())
    abs_T = np.stack(abs_T, axis=0)              # (P, 4, 4)
    # sample M control poses at evenly spaced pair indices
    idx = np.linspace(0, P - 1, M).round().astype(int)
    return abs_T[idx]


def bspline_fit_error_curve(dense_xi: np.ndarray, M_values: list[int]) -> list[dict[str, Any]]:
    """Design Q5: measured fit-error (RMS forward-translation residual, metres) of the
    SE(3) B-spline reconstruction of the absolute ego trajectory vs ``M`` control poses.

    Returns one row per ``M`` with the reconstruction RMS + the counted control-pose
    payload size estimate (``M * 6`` fp16 = ``M * 12`` bytes). The absolute trajectory
    is the ground truth; the spline is evaluated via ``se3_bspline_eval_numpy`` (fp64).
    """
    from tac.lie import _se3_numpy as se3
    from tac.lie.se3_bspline import se3_bspline_eval_numpy

    dxi = np.asarray(dense_xi, np.float64)
    P = int(dxi.shape[0])
    # ground-truth absolute forward positions
    T = np.eye(4, dtype=np.float64)
    fwd_gt = [0.0]
    for k in range(1, P):
        T = se3.compose(T, se3.exp_se3(dxi[k]))
        fwd_gt.append(float(se3.translation_of(T)[2]))
    fwd_gt = np.asarray(fwd_gt, np.float64)
    rows: list[dict[str, Any]] = []
    for M in M_values:
        if M < 4 or M > P:
            continue
        ctrl = fit_se3_bspline_controls(dxi, M)                    # (M,4,4) control poses
        n_seg = M - 3
        # map each pair index to spline time in [0, n_seg]
        tt = np.linspace(0.0, float(n_seg), P)
        fwd_hat = np.array([float(se3.translation_of(se3_bspline_eval_numpy(ctrl, float(ti)))[2]) for ti in tt])
        rms = float(np.sqrt(np.mean((fwd_hat - fwd_gt) ** 2)))
        rows.append({"M": int(M), "fwd_rms_m": rms, "ctrl_bytes_fp16": int(M * 6 * 2)})
    return rows
