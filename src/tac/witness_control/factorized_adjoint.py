# SPDX-License-Identifier: MIT
"""Exact-factorized costate adjoint with a deliberately tiny learned residual.

This is the load-bearing ``V_exact_factorized_residual`` arm of the existing
``lambda_net`` organ.  It composes three already-canonicalized scorer facts:

* the frozen SegNet head is linear with a rank-4 class-difference subspace;
* the camera-to-margin gain is pair dependent, with ``lambda_cc' proportional
  to 1 / G_cc'``;
* camera coordinates certified as zero-weight by the exact resize/preimage
  analysis are projected to zero, rather than offered to the learner.

The exact class operator is the weighted graph Laplacian

    K = p_visible * B.T diag(||dw_cc'|| / G_cc') B,

where ``B`` is the signed class-pair incidence matrix.  Consequently ``K`` is
rank four and annihilates the all-class gauge direction.  The trajectory fit is
not allowed to relearn ``K``.  It learns only six robust base-drift constants
and five shared scalar amplitude terms (constant, two bounded temporal sensors,
boundary occupancy, regularizer occupancy).  Every learned term multiplies the
exact operator; it cannot invent a new class direction.  At n=1 this is a small
ridge/physics residual, not a neural-network claim.

All outputs are advisory.  No scorer, run, checkpoint, config, or process is
mutated by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.canonical_equations.lane_gain_chain_composed_20260716 import (
    BIRTHS_PER_FRAMESTEP,
    COMPOSED_GAIN_MED,
    DEATHS_PER_FRAMESTEP,
    INTERIOR_ISLANDS_PER_FRAME,
    LANE_ISLAND_FRAC_BELOW_05,
    LANE_ISLAND_PERSISTENCE_MEDIAN,
)
from tac.canonical_equations.realization_necessity_preimage_20260715 import (
    CAMERA_SUPPORT_FRAC,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    HEAD_CENTERED_SINGVALS,
    HEAD_PAIR_NORMS,
)
from tac.witness_control.lambda_net import (
    N_CLASSES,
    PHI_DIM,
    STATE_DIM,
    GPCostatePosteriorAdjoint,
)

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
ARCHITECTURE = "V_exact_factorized_residual"
AXIS_TAG = "[macOS advisory] NON-PROMOTABLE"
RESIDUAL_RIDGE = 10.0
RESIDUAL_RIDGE_SELECTION_SCOPE = (
    "POST_HOC_DEVELOPMENT_ON_205; independent compatible trajectory validation owed")

ZERO_WEIGHT_FRAC = float(CAMERA_SUPPORT_FRAC["certified_free_zero_weight"])
VISIBLE_SUPPORT_FRAC = 1.0 - ZERO_WEIGHT_FRAC


def _pair_indices(pair: str) -> tuple[int, int]:
    a, b = pair.split("-", 1)
    return CLASS_INDEX[a], CLASS_INDEX[b]


# MEASURED inputs, DERIVED composition.  Keep the raw values visible for Rudin
# readback; the matrix builder applies a trace gauge only after composition.
PAIR_FACTOR_RAW = {
    pair: float(HEAD_PAIR_NORMS[pair]) / float(gain)
    for pair, gain in COMPOSED_GAIN_MED.items()
}


def exact_class_operator() -> np.ndarray:
    """Return the rank-4 exact class operator ``K`` (5x5, deterministic).

    A trace-five gauge makes its global scale explicit and leaves the learned
    one-degree amplitude comparable across refits.  The measured visible-support
    mass is then applied once.  Orientation of pair incidence does not affect K.
    """
    pairs = tuple(PAIR_FACTOR_RAW)
    B = np.zeros((len(pairs), N_CLASSES), dtype=np.float64)
    weights = np.empty(len(pairs), dtype=np.float64)
    for r, pair in enumerate(pairs):
        c, cp = _pair_indices(pair)
        B[r, c] = 1.0
        B[r, cp] = -1.0
        weights[r] = PAIR_FACTOR_RAW[pair]
    K = B.T @ np.diag(weights) @ B
    trace = float(np.trace(K))
    if not np.isfinite(trace) or trace <= 0.0:  # fail closed on corrupted constants
        raise ValueError("exact factorized class operator has non-positive trace")
    return K * (float(N_CLASSES) / trace) * VISIBLE_SUPPORT_FRAC


EXACT_CLASS_OPERATOR = exact_class_operator()


def apply_ker_a_mask(camera_costate: Any, zero_weight_mask: Any) -> np.ndarray:
    """Project an explicit camera-space costate onto ``ker(A)^perp`` exactly.

    ``zero_weight_mask=True`` means the exact resize/preimage audit certified the
    coordinate as objective-invisible.  Shape mismatch is refused, never broadcast.
    The aggregate controller uses :data:`VISIBLE_SUPPORT_FRAC`; pixelwise consumers
    use this function with the actual mask artifact.
    """
    value = np.asarray(camera_costate, dtype=np.float64)
    mask = np.asarray(zero_weight_mask, dtype=bool)
    if value.shape != mask.shape:
        raise ValueError(f"camera_costate/mask shape mismatch: {value.shape} != {mask.shape}")
    return np.where(mask, 0.0, value)


def exact_response_direction(phi: Any) -> np.ndarray:
    """Exact response direction for one lever feature vector.

    Only its five class-target coordinates participate.  Pose/rate and the
    boundary/regularizer coordinates do not create a class direction; the latter
    may modulate the learned scalar amplitude in :class:`ExactFactorizedAdjoint`.
    """
    p = np.asarray(phi, dtype=np.float64)
    if p.shape != (PHI_DIM,):
        raise ValueError(f"phi must be ({PHI_DIM},), got {p.shape}")
    out = np.zeros(STATE_DIM, dtype=np.float64)
    out[:N_CLASSES] = -(EXACT_CLASS_OPERATOR @ p[:N_CLASSES])
    return out


def factorization_provenance() -> dict[str, Any]:
    """Machine-readable exact/derived/learned split for digest/dashboard rows."""
    gain_ref = float(np.median([v for k, v in COMPOSED_GAIN_MED.items()
                                if k != "Road-Lane"]))
    combined_ref = float(np.median([v for k, v in PAIR_FACTOR_RAW.items()
                                    if k != "Road-Lane"]))
    return {
        "architecture": ARCHITECTURE,
        "exact": {
            "head_equation": "segnet_head_rank4_linear_flipdist_v1",
            "head_rank": int(np.count_nonzero(np.asarray(HEAD_CENTERED_SINGVALS) > 1e-9)),
            "resize_equation": "realization_necessity_preimage_per_stratum_v1",
            "certified_zero_weight_camera_frac": ZERO_WEIGHT_FRAC,
            "visible_camera_frac": VISIBLE_SUPPORT_FRAC,
            "gain_equation": "lane_gain_chain_composed_v1",
        },
        "derived": {
            "operator": "p_visible * B.T diag(||dw_pair|| / G_pair) B",
            "operator_rank": int(np.linalg.matrix_rank(EXACT_CLASS_OPERATOR, tol=1e-10)),
            "gauge_null_linf": float(np.max(np.abs(
                EXACT_CLASS_OPERATOR @ np.ones(N_CLASSES, dtype=np.float64)))),
            "road_lane_gain_only_lambda_ratio_vs_other_median": (
                gain_ref / float(COMPOSED_GAIN_MED["Road-Lane"])),
            "road_lane_head_over_gain_ratio_vs_other_median": (
                float(PAIR_FACTOR_RAW["Road-Lane"]) / combined_ref),
        },
        "learned": {
            "temporal_residual": (
                "closed-form differentiated-RBF GP posterior (existing T arm; "
                "past-only hyperparameter selection in walk-forward backtest)"),
            "shared_amplitude_parameters": 5,
            "new_class_direction_parameters": 0,
            "residual_ridge": RESIDUAL_RIDGE,
            "residual_ridge_selection_scope": RESIDUAL_RIDGE_SELECTION_SCOPE,
        },
        "axis": AXIS_TAG,
        "score_claim": False,
    }


@dataclass(frozen=True)
class FactorizedFitDiagnostics:
    n_intervals: int
    n_parameters: int
    amplitude: tuple[float, ...]
    fit_rms: float
    clipped_interval_fraction: float
    amplitude_gate: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_intervals": self.n_intervals,
            "n_parameters": self.n_parameters,
            "amplitude": list(self.amplitude),
            "fit_rms": self.fit_rms,
            "clipped_interval_fraction": self.clipped_interval_fraction,
            "amplitude_gate": self.amplitude_gate,
        }


class ExactFactorizedAdjoint:
    """Rank-4 analytic adjoint times a five-scalar bounded residual trunk.

    The base trajectory forcing is the existing closed-form differentiated-RBF
    posterior (arm T), not another learned organ.  The factorized response supplies
    the lever field that forecast-only arm T intentionally lacks; #344 NCDE remains
    a separate stage-boundary timing sensor composed by the shadow controller.
    """

    name = ARCHITECTURE

    def __init__(self, ridge: float = RESIDUAL_RIDGE):
        self.ridge = float(ridge)
        self.temporal = GPCostatePosteriorAdjoint()
        self.beta: np.ndarray | None = None
        self.diagnostics: FactorizedFitDiagnostics | None = None
        self.score_class_weights = np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float64)

    def set_score_composition(self, comp: Any) -> None:
        """Inject the fitted exact d_seg composition for the internal admission gate."""
        w = np.asarray(comp.class_weights, dtype=np.float64)
        if w.shape != (N_CLASSES,) or not np.all(np.isfinite(w)):
            raise ValueError("score composition must carry five finite class weights")
        self.score_class_weights = w

    @staticmethod
    def _amplitude_features(ctx: np.ndarray, phi: np.ndarray) -> np.ndarray:
        # Fixed bounded nonlinear sensors; only their five scalar coefficients learn.
        return np.asarray([
            1.0,
            np.tanh(float(ctx[0])),
            np.tanh(float(ctx[1])),
            float(phi[-2]),              # boundary/event occupancy
            float(phi[-1]),              # regularizer occupancy
        ], dtype=np.float64)

    def _solve_beta(self, intervals, phis: np.ndarray,
                    temporal: GPCostatePosteriorAdjoint) -> np.ndarray:
        """Closed-form five-scalar residual solve for one past-only fit set."""
        Y = np.stack([iv.dxdt() for iv in intervals])
        base = np.stack([temporal.base(iv.x0, iv.ctx, iv.path) for iv in intervals])
        prior = np.stack([exact_response_direction(phi) for phi in phis])
        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        for iv, target, base_rate in zip(intervals, Y, base, strict=True):
            amp = np.stack([self._amplitude_features(iv.ctx, phi) for phi in phis])
            basis = np.einsum("i,is,ik->ks", iv.u_mean, prior, amp)
            for s in range(STATE_DIM):
                x_rows.append(basis[:, s])
                y_rows.append(float(target[s] - base_rate[s]))
        X = np.stack(x_rows)
        y = np.asarray(y_rows, dtype=np.float64)
        gram = X.T @ X
        scale = float(np.mean(np.diag(gram))) or 1.0
        beta = np.linalg.solve(
            gram + self.ridge * scale * np.eye(5, dtype=np.float64), X.T @ y)
        beta[0] = max(float(beta[0]), 0.0)
        return beta

    @classmethod
    def _predict_with(cls, iv, phis: np.ndarray,
                      temporal: GPCostatePosteriorAdjoint,
                      beta: np.ndarray) -> np.ndarray:
        out = temporal.base(iv.x0, iv.ctx, iv.path)
        for u, phi in zip(iv.u_mean, phis, strict=True):
            amp = max(float(beta @ cls._amplitude_features(iv.ctx, phi)), 0.0)
            out = out + float(u) * amp * exact_response_direction(phi)
        return out

    def fit(self, intervals, phis: np.ndarray, seed: int = 0) -> None:
        del seed  # closed-form deterministic solve
        if not intervals:
            raise ValueError("exact-factorized adjoint needs at least one interval")
        P = np.asarray(phis, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != PHI_DIM:
            raise ValueError(f"phis must be (n,{PHI_DIM}), got {P.shape}")
        if P.shape[0] != intervals[0].u_mean.shape[0]:
            raise ValueError("phis/lever-share width mismatch")

        Y = np.stack([iv.dxdt() for iv in intervals])
        self.temporal.fit(intervals, P, seed=0)
        beta = self._solve_beta(intervals, P, self.temporal)

        # Past-only one-step admission for the learned amplitude.  This is inside
        # every outer backtest fold, so it cannot see the fold being predicted.  If
        # the five-scalar residual fails to improve the exact scalar d_seg composition
        # on the last available interval, the exact direction remains observable but
        # its fitted amplitude projects to zero for DECIDE (fail closed).
        gate = "REJECTED_INSUFFICIENT_INTERVALS"
        if len(intervals) >= 3:
            train = intervals[:-1]
            held = intervals[-1]
            temporal_gate = GPCostatePosteriorAdjoint()
            temporal_gate.fit(train, P, seed=0)
            beta_gate = self._solve_beta(train, P, temporal_gate)
            pred0 = temporal_gate.base(held.x0, held.ctx, held.path)
            pred1 = self._predict_with(held, P, temporal_gate, beta_gate)
            meas = held.dxdt()
            w = self.score_class_weights
            err0 = abs(float(w @ (pred0[:N_CLASSES] - meas[:N_CLASSES])))
            err1 = abs(float(w @ (pred1[:N_CLASSES] - meas[:N_CLASSES])))
            if np.isfinite(err1) and err1 <= err0:
                gate = "ADMITTED_PAST_ONLY_ONE_STEP"
            else:
                beta = np.zeros(5, dtype=np.float64)
                gate = "REJECTED_PAST_ONLY_ONE_STEP"
        else:
            beta = np.zeros(5, dtype=np.float64)
        self.beta = beta

        pred = np.stack([self._predict_rate(iv, P) for iv in intervals])
        rms = float(np.sqrt(np.mean((pred - Y) ** 2)))
        raw = []
        for iv in intervals:
            for phi in P:
                raw.append(float(beta @ self._amplitude_features(iv.ctx, phi)))
        clipped = float(np.mean(np.asarray(raw) < 0.0)) if raw else 0.0
        self.diagnostics = FactorizedFitDiagnostics(
            n_intervals=len(intervals), n_parameters=5,
            amplitude=tuple(float(v) for v in beta), fit_rms=rms,
            clipped_interval_fraction=clipped,
            amplitude_gate=gate,
        )

    def _predict_rate(self, iv, phis: np.ndarray) -> np.ndarray:
        out = self.base(iv.x0, iv.ctx, iv.path)
        for u, phi in zip(iv.u_mean, phis, strict=True):
            out = out + float(u) * self.response(iv.x0, iv.ctx, phi, iv.path)
        return out

    def response(self, x: np.ndarray, ctx: np.ndarray, phi: np.ndarray,
                 path: np.ndarray | None = None) -> np.ndarray:
        del x, path
        if self.beta is None:
            raise AssertionError("fit first")
        amplitude = max(float(self.beta @ self._amplitude_features(ctx, phi)), 0.0)
        return amplitude * exact_response_direction(phi)

    def base(self, x: np.ndarray, ctx: np.ndarray,
             path: np.ndarray | None = None) -> np.ndarray:
        return self.temporal.base(x, ctx, path)


def morse_smale_event_prior() -> dict[str, Any]:
    """Measured/derived event warning inputs; no claim that every event is a saddle."""
    turnover = ((float(BIRTHS_PER_FRAMESTEP) + float(DEATHS_PER_FRAMESTEP))
                / float(INTERIOR_ISLANDS_PER_FRAME))
    prov = factorization_provenance()
    return {
        "measured": {
            "lane_island_persistence_median": float(LANE_ISLAND_PERSISTENCE_MEDIAN),
            "lane_island_frac_below_0_5": float(LANE_ISLAND_FRAC_BELOW_05),
            "births_per_framestep_upper_bound": float(BIRTHS_PER_FRAMESTEP),
            "deaths_per_framestep_upper_bound": float(DEATHS_PER_FRAMESTEP),
            "interior_islands_per_frame": float(INTERIOR_ISLANDS_PER_FRAME),
        },
        "derived": {
            "event_turnover_per_island_step_upper_bound": turnover,
            "critical_lambda_pair": "Road-Lane",
            "critical_lambda_ratio_vs_other_major_pair_median": (
                prov["derived"]["road_lane_gain_only_lambda_ratio_vs_other_median"]),
            "interpretation": (
                "birth/death turnover is a Morse-Smale saddle-node warning proxy; "
                "it is not an exact event identity or a fitted optimum"),
        },
        "axis": AXIS_TAG,
        "score_claim": False,
    }


__all__ = [
    "ARCHITECTURE", "AXIS_TAG", "EXACT_CLASS_OPERATOR", "PAIR_FACTOR_RAW",
    "RESIDUAL_RIDGE", "RESIDUAL_RIDGE_SELECTION_SCOPE",
    "VISIBLE_SUPPORT_FRAC", "ZERO_WEIGHT_FRAC", "ExactFactorizedAdjoint",
    "FactorizedFitDiagnostics", "apply_ker_a_mask", "exact_class_operator",
    "exact_response_direction", "factorization_provenance", "morse_smale_event_prior",
]
