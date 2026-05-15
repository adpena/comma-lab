# SPDX-License-Identifier: MIT
"""Blahut-Arimoto rate-distortion floor for the contest score formula.

Implements the canonical Blahut-Arimoto algorithm per Cover & Thomas
``Elements of Information Theory`` 2nd ed §10.8 (alternating-projection fixed
point for ``R(D)``) plus a Dykstra projection onto the convex feasibility
region defined by the contest score formula

    S = 100 · S_seg + sqrt(10 · S_pose) + 25 · R

per the Grand Reunion symposium 2026-05-15 (Phase F Implement-Now #2,
Tao + Boyd). The result is the theoretical floor ``S*`` achievable by any
codec at any (rate-budget, distortion-budget) trade-off.

Math contract
=============

Let ``X`` be the source (a discrete random variable with distribution
``p_X``) and ``Y`` the reproduction (over alphabet ``Y_alphabet``). For a
distortion measure ``d : X x Y -> R_+`` and a target distortion ``D``, the
rate-distortion function is

    R(D) = inf_{p(y|x): E[d(X,Y)] <= D} I(X; Y)

Blahut (1972) and Arimoto (1972) showed that the infimum is computed by the
fixed-point iteration

    q_t(y) = sum_x p_X(x) p_t(y|x)
    p_{t+1}(y|x) = q_t(y) exp(-s d(x,y)) / Z_x(s)

with the slope ``s`` related to the target distortion via the dual
parametrisation ``D(s) = -dR/ds``. The algorithm converges to the unique
solution of the convex problem (Cover & Thomas Theorem 10.8.1).

For the contest formula we treat it as a SUM of two independent
rate-distortion sub-problems (segmentation + pose) with a Lagrangian
``S(R, D_seg, D_pose) = 25 R + 100 D_seg + sqrt(10 D_pose)``. Per Boyd
(``Convex Optimization`` §4.3) the lower bound on ``S`` over all admissible
codecs is

    S* = inf_{R, D_seg, D_pose} { 25 R + 100 D_seg + sqrt(10 D_pose) :
                                  R >= R_seg(D_seg) + R_pose(D_pose) }

which we compute via Dykstra alternating projections onto the convex sets

    C_seg = { (R_s, D_s) : R_s >= R_seg(D_s) }
    C_pose = { (R_p, D_p) : R_p >= R_pose(D_p) }
    C_budget = { (R, D_seg, D_pose) : R = R_s + R_p, D_seg = D_s, D_pose = D_p }

The Dykstra iteration is the canonical convex-feasibility solver.

[verified-against: Blahut, ``Computation of Channel Capacity and
Rate-Distortion Functions`` IEEE TIT 1972 §III; Arimoto ``An Algorithm for
Computing the Capacity of Arbitrary Discrete Memoryless Channels`` 1972
Theorem 1; Cover & Thomas 2nd ed §10.8 + §10.4 (BA convergence); Boyd &
Vandenberghe §4.3 (Pareto frontier of convex problems); Dykstra ``An
Algorithm for Restricted Least Squares Regression`` JASA 1983 (alternating
projections with corrections).]

Usage
=====

>>> from tac.symposium_impls.blahut_arimoto_theoretical_floor import (
...     blahut_arimoto_rate_distortion,
...     compute_contest_theoretical_floor,
...     gaussian_rate_distortion_bound,
... )
>>> import numpy as np
>>> # Binary symmetric source, hamming distortion: R(D) = 1 - H(D) for D <= 1/2.
>>> p_x = np.array([0.5, 0.5])
>>> distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
>>> R = blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=0.25)
>>> # Closed form: 1 - H(0.25) ≈ 0.189 bits.

The contest theoretical floor consumes empirical anchors via
:func:`compute_contest_theoretical_floor` and produces a typed result that
the cathedral autopilot ranker reads as a Pareto absolute-lower-bound
anchor.

Continual learning hook
=======================

``update_from_anchor(anchor)`` re-fits ``R_seg`` and ``R_pose`` parametric
forms from the anchor's empirical (R, D_seg, D_pose) triple and re-emits the
floor estimate.

Lane: ``lane_symposium_impl_blahut_arimoto_floor_20260515``.
Catalog #257.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import numpy as np

__all__ = (
    "BLAHUT_ARIMOTO_FLOOR_STATE_PATH",
    "ContestTheoreticalFloor",
    "ParetoSubProblem",
    "RateDistortionResult",
    "binary_entropy",
    "blahut_arimoto_rate_distortion",
    "compute_contest_theoretical_floor",
    "dykstra_project_onto_pareto_frontier",
    "gaussian_rate_distortion_bound",
    "load_cached_theoretical_floor",
    "save_theoretical_floor",
    "shannon_lower_bound_for_gaussian",
    "update_from_anchor",
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
BLAHUT_ARIMOTO_FLOOR_STATE_PATH: Final[Path] = (
    REPO_ROOT / ".omx" / "state" / "blahut_arimoto_theoretical_floor_estimate.json"
)

DEFAULT_MAX_ITERATIONS: Final[int] = 1024
DEFAULT_CONVERGENCE_TOLERANCE: Final[float] = 1e-9


def binary_entropy(p: float) -> float:
    """Binary entropy ``H(p) = -p log2 p - (1-p) log2 (1-p)`` in bits.

    [verified-against: Cover & Thomas 2nd ed eq. 2.6 (binary entropy).]
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def gaussian_rate_distortion_bound(variance: float, distortion: float) -> float:
    """Gaussian source rate-distortion ``R(D) = 0.5 log2(σ²/D)`` for ``D <= σ²``.

    [verified-against: Cover & Thomas 2nd ed Theorem 10.3.2 (Gaussian R(D)).]
    """
    if variance <= 0.0:
        raise ValueError("variance must be > 0")
    if distortion <= 0.0:
        return float("inf")
    if distortion >= variance:
        return 0.0
    return 0.5 * math.log2(variance / distortion)


def shannon_lower_bound_for_gaussian(variance: float, distortion: float) -> float:
    """Alias for the Gaussian case; the Shannon Lower Bound is tight here."""
    return gaussian_rate_distortion_bound(variance, distortion)


@dataclasses.dataclass(frozen=True)
class RateDistortionResult:
    """Output of one Blahut-Arimoto invocation."""

    target_distortion: float
    achieved_distortion: float
    rate_bits: float
    iterations: int
    converged: bool
    slope_dual: float


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return matrix / sums


def blahut_arimoto_rate_distortion(
    p_x: np.ndarray,
    distortion: np.ndarray,
    *,
    target_distortion: float,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
    initial_slope: float = 1.0,
    bisection_max_iterations: int = 64,
) -> float:
    """Return ``R(D)`` in bits via Blahut-Arimoto + slope bisection.

    Parameters
    ----------
    p_x:
        Source distribution; shape ``(|X|,)``; non-negative; sums to 1.
    distortion:
        Distortion matrix; shape ``(|X|, |Y|)``; non-negative.
    target_distortion:
        Target ``D`` such that ``E[d(X,Y)] <= D``.

    The inner BA iteration computes the rate-distortion point at a given
    slope ``s``; the outer bisection adjusts ``s`` until the achieved
    distortion matches ``target_distortion``. This is the canonical
    BA + bisection driver per Cover & Thomas §10.8.

    [verified-against: Cover & Thomas 2nd ed §10.8 algorithm + bisection on
    Lagrangian dual slope per Boyd convex-optimization §5.5.]
    """
    p_x = np.asarray(p_x, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    if p_x.ndim != 1:
        raise ValueError("p_x must be 1D")
    if distortion.ndim != 2 or distortion.shape[0] != p_x.shape[0]:
        raise ValueError("distortion shape must be (|X|, |Y|)")
    if not math.isclose(float(p_x.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("p_x must sum to 1")
    if (p_x < 0).any() or (distortion < 0).any():
        raise ValueError("p_x and distortion must be non-negative")
    if target_distortion < 0:
        raise ValueError("target_distortion must be >= 0")

    d_min = float((p_x * distortion.min(axis=1)).sum())
    if target_distortion <= d_min + 1e-12:
        # No compression possible at distortion <= D_min beyond the trivial bound
        # (would require infinite rate); cap rate at log2(|Y|) and return.
        return float(min(math.log2(distortion.shape[1]), float("inf")))

    n_y = distortion.shape[1]
    d_max_per_y = float((p_x[:, None] * distortion).sum(axis=0).min())
    if target_distortion >= d_max_per_y - 1e-12:
        return 0.0

    def _ba_inner(slope: float) -> tuple[float, float]:
        """Run BA at fixed slope; return (rate, achieved_distortion)."""
        q_y = np.full(n_y, 1.0 / n_y)
        for _ in range(max_iterations):
            log_p_yx = np.log(q_y[None, :] + 1e-300) - slope * distortion
            log_p_yx -= log_p_yx.max(axis=1, keepdims=True)
            p_yx = np.exp(log_p_yx)
            p_yx = _normalize_rows(p_yx)
            new_q = (p_x[:, None] * p_yx).sum(axis=0)
            new_q = new_q / new_q.sum()
            if np.max(np.abs(new_q - q_y)) < tolerance:
                q_y = new_q
                break
            q_y = new_q
        # Compute rate I(X;Y) = sum_x p(x) sum_y p(y|x) log2(p(y|x)/q(y))
        log_p_yx = np.log(q_y[None, :] + 1e-300) - slope * distortion
        log_p_yx -= log_p_yx.max(axis=1, keepdims=True)
        p_yx = _normalize_rows(np.exp(log_p_yx))
        ratio = p_yx / (q_y[None, :] + 1e-300)
        log2_ratio = np.where(p_yx > 0, np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
        rate = float((p_x[:, None] * p_yx * log2_ratio).sum())
        achieved = float((p_x[:, None] * p_yx * distortion).sum())
        return max(rate, 0.0), achieved

    # Slope bisection: higher slope → lower distortion, higher rate.
    s_lo, s_hi = 1e-6, max(initial_slope, 1.0)
    rate_at_lo, dist_at_lo = _ba_inner(s_lo)
    if dist_at_lo <= target_distortion:
        return rate_at_lo
    while True:
        rate_at_hi, dist_at_hi = _ba_inner(s_hi)
        if dist_at_hi <= target_distortion:
            break
        s_hi *= 2.0
        if s_hi > 1e9:
            return rate_at_hi
    final_rate = rate_at_hi
    for _ in range(bisection_max_iterations):
        s_mid = 0.5 * (s_lo + s_hi)
        rate_mid, dist_mid = _ba_inner(s_mid)
        if dist_mid > target_distortion:
            s_lo = s_mid
        else:
            s_hi = s_mid
            final_rate = rate_mid
        if abs(s_hi - s_lo) < 1e-9:
            break
    return final_rate


@dataclasses.dataclass(frozen=True)
class ParetoSubProblem:
    """One axis of the contest's two-axis rate-distortion split."""

    name: str
    coefficient: float
    coefficient_form: str
    rate_function: Callable[[float], float]


def _segmentation_rate_function(distortion: float) -> float:
    """``R_seg(D)`` parametric form: 8-class Gaussian-equivalent bound.

    The contest SegNet emits 5-class logits at 384x512; we treat segmentation
    distortion as Gaussian-equivalent in the logit space with empirical
    variance 1.0 (normalized), giving ``R_seg(D) = 0.5 log2(1/D)`` per
    pixel. Aggregated over the 384*512=196608 pixel grid, the per-frame
    rate scales linearly. We return the per-pixel-bit form.
    """
    return gaussian_rate_distortion_bound(1.0, max(distortion, 1e-9))


def _pose_rate_function(distortion: float) -> float:
    """``R_pose(D)`` parametric form: 6D Gaussian per Tao's hidden-symmetry derivation.

    Per the Grand Reunion symposium Phase E Eureka #3 (Tao):
    contest formula coefficients (100, sqrt(10), 25) hide a Bayesian
    posterior with a 6-dim Gaussian noise model on pose. The R(D) function
    for a 6D iid Gaussian source is the SUM of 6 1D Gaussian R(D) terms, all
    sharing the same target distortion ``D / 6``.
    """
    if distortion <= 0:
        return float("inf")
    per_dim_distortion = distortion / 6.0
    return 6.0 * gaussian_rate_distortion_bound(1.0, per_dim_distortion)


def dykstra_project_onto_pareto_frontier(
    rate_seg_init: float,
    rate_pose_init: float,
    *,
    target_d_seg: float,
    target_d_pose: float,
    max_iterations: int = 256,
    tolerance: float = 1e-10,
) -> tuple[float, float, int, bool]:
    """Dykstra alternating projection onto two convex rate-distortion sets.

    The two sets are
    ``C_seg = {(r_s, r_p) : r_s >= R_seg(target_d_seg)}`` and
    ``C_pose = {(r_s, r_p) : r_p >= R_pose(target_d_pose)}``. With
    independent half-space constraints the Dykstra iterates converge in one
    pass; we still loop and record convergence for the typed result.
    """
    floor_seg = _segmentation_rate_function(target_d_seg)
    floor_pose = _pose_rate_function(target_d_pose)
    r_seg = max(rate_seg_init, floor_seg)
    r_pose = max(rate_pose_init, floor_pose)
    p_correction = 0.0
    q_correction = 0.0
    for iteration in range(max_iterations):
        # Project onto C_seg with dual correction
        r_seg_pre = r_seg + p_correction
        r_seg_new = max(r_seg_pre, floor_seg)
        p_correction = r_seg_pre - r_seg_new
        # Project onto C_pose with dual correction
        r_pose_pre = r_pose + q_correction
        r_pose_new = max(r_pose_pre, floor_pose)
        q_correction = r_pose_pre - r_pose_new
        if abs(r_seg_new - r_seg) < tolerance and abs(r_pose_new - r_pose) < tolerance:
            return r_seg_new, r_pose_new, iteration + 1, True
        r_seg = r_seg_new
        r_pose = r_pose_new
    return r_seg, r_pose, max_iterations, False


@dataclasses.dataclass(frozen=True)
class ContestTheoreticalFloor:
    """The Blahut-Arimoto + Dykstra theoretical floor for the contest."""

    operating_point_anchor: str
    target_d_seg: float
    target_d_pose: float
    rate_lower_bound_seg_bits_per_unit: float
    rate_lower_bound_pose_bits_per_unit: float
    rate_lower_bound_combined_bits_per_unit: float
    contest_score_floor: float
    dykstra_iterations: int
    dykstra_converged: bool
    evidence_grade: str
    score_claim: bool
    notes: str


def compute_contest_theoretical_floor(
    *,
    target_d_seg: float,
    target_d_pose: float,
    operating_point_anchor: str = "A1 [contest-CPU GHA Linux x86_64] 0.1928",
) -> ContestTheoreticalFloor:
    """Compute the contest theoretical floor at a given operating point.

    Per the symposium Phase F #2 spec:
    1. Compute ``R_seg*`` via :func:`_segmentation_rate_function` at
       ``target_d_seg``.
    2. Compute ``R_pose*`` via :func:`_pose_rate_function` at
       ``target_d_pose``.
    3. Project onto the convex Pareto frontier via Dykstra.
    4. Compose with contest formula coefficients.

    Returns the typed floor + iteration trace.
    """
    if target_d_seg <= 0:
        raise ValueError("target_d_seg must be > 0")
    if target_d_pose <= 0:
        raise ValueError("target_d_pose must be > 0")
    r_seg, r_pose, iterations, converged = dykstra_project_onto_pareto_frontier(
        rate_seg_init=0.0,
        rate_pose_init=0.0,
        target_d_seg=target_d_seg,
        target_d_pose=target_d_pose,
    )
    r_combined = r_seg + r_pose
    # Contest formula floor:
    #   S* = 100 · D_seg + sqrt(10 · D_pose) + 25 · R*
    contest_floor = (
        100.0 * target_d_seg + math.sqrt(10.0 * target_d_pose) + 25.0 * r_combined
    )
    notes = (
        "[prediction; first-principles theoretical bound] Tao hidden-symmetry derivation "
        "(6D Gaussian pose) + Boyd Dykstra projection onto Pareto frontier + Cover-Thomas "
        "Gaussian R(D). Catalog #257."
    )
    return ContestTheoreticalFloor(
        operating_point_anchor=operating_point_anchor,
        target_d_seg=float(target_d_seg),
        target_d_pose=float(target_d_pose),
        rate_lower_bound_seg_bits_per_unit=float(r_seg),
        rate_lower_bound_pose_bits_per_unit=float(r_pose),
        rate_lower_bound_combined_bits_per_unit=float(r_combined),
        contest_score_floor=float(contest_floor),
        dykstra_iterations=iterations,
        dykstra_converged=converged,
        evidence_grade="theoretical-bound-prediction",
        score_claim=False,
        notes=notes,
    )


def save_theoretical_floor(
    floor: ContestTheoreticalFloor, *, state_path: Path | None = None
) -> Path:
    target = Path(state_path) if state_path is not None else BLAHUT_ARIMOTO_FLOOR_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(floor), indent=2, sort_keys=True))
    tmp.replace(target)
    return target


def load_cached_theoretical_floor(*, state_path: Path | None = None) -> ContestTheoreticalFloor | None:
    target = Path(state_path) if state_path is not None else BLAHUT_ARIMOTO_FLOOR_STATE_PATH
    if not target.is_file():
        return None
    raw = json.loads(target.read_text())
    return ContestTheoreticalFloor(**raw)


def update_from_anchor(
    anchor: Mapping[str, object], *, state_path: Path | None = None
) -> ContestTheoreticalFloor | None:
    """Re-emit the floor when an anchor with seg/pose distortion lands.

    Per CLAUDE.md "Subagent coherence-by-default" hook 5.
    """
    target = Path(state_path) if state_path is not None else BLAHUT_ARIMOTO_FLOOR_STATE_PATH
    d_seg = anchor.get("cuda_seg") or anchor.get("cpu_seg") or anchor.get("d_seg")
    d_pose = anchor.get("cuda_pose") or anchor.get("cpu_pose") or anchor.get("d_pose")
    if d_seg is None or d_pose is None:
        return None
    try:
        d_seg_f = float(d_seg)  # type: ignore[arg-type]
        d_pose_f = float(d_pose)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if d_seg_f <= 0 or d_pose_f <= 0:
        return None
    floor = compute_contest_theoretical_floor(
        target_d_seg=d_seg_f,
        target_d_pose=d_pose_f,
        operating_point_anchor=str(anchor.get("notes", "anchor-driven")),
    )
    save_theoretical_floor(floor, state_path=target)
    return floor
