"""Pure-Python score-geometry helpers for the contest objective.

The contest score is:

    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * B / N_REF

where ``N_REF = 37_545_489`` (uncompressed reference video bytes), ``d_seg``
is the SegNet argmax disagreement rate, ``d_pose`` is the PoseNet MSE on the
first 6 dimensions, and ``B`` is the archive byte count.

Because the pose contribution is concave (sqrt), the **marginal value of
each axis depends on the operating point**. At pose_avg ~ 0.18 (the legacy
1.x score regime) the SegNet axis dominates ~77x. At pose_avg ~ 3.4e-5
(the PR106 frontier regime) the pose axis dominates ~2.7x. The crossover
is analytically located at ``pose_avg = (5/100)^2 / 10 = 2.5e-4``.

The canonical differentiable formula and operating-point marginals live in
``tac.contest_rate_distortion_system``. This module is a torch-free companion
for inverse curves and Pareto slack calculations used by planning tools:

  * ``contest_score(d_seg, d_pose, B)`` — exact contest objective
  * ``score_gradient(d_seg, d_pose, B)`` — partial derivatives
  * ``importance_flip_threshold()`` — where SegNet vs PoseNet marginals cross
  * ``marginal_value_per_byte(...)`` — bytes-of-information cost per axis
  * ``information_floor(...)`` — Shannon-style lower bound at d_seg=d_pose=0
  * ``operating_regime(d_pose)`` — classifies a candidate against the flip
  * ``project_onto_pareto_envelope(...)`` — closed-form 3-axis Pareto

Pure math. No torch, no GPU, no scorer load.

Cross-references:

  * ``tac.contest_rate_distortion_system`` canonical formula and marginals
  * ``tools/contest_score_pareto_3axis.py`` evidence-space Pareto ranking
  * ``.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md``
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

CONTEST_REFERENCE_BYTES = 37_545_489
SEG_COEFFICIENT = 100.0
POSE_COEFFICIENT_INSIDE_SQRT = 10.0
RATE_COEFFICIENT = 25.0


@dataclass(frozen=True)
class ScoreDecomposition:
    """The three additive contributions to the contest score."""

    seg_term: float
    pose_term: float
    rate_term: float

    @property
    def total(self) -> float:
        return self.seg_term + self.pose_term + self.rate_term

    @property
    def fractions(self) -> tuple[float, float, float]:
        s = self.total
        if s == 0.0:
            return (0.0, 0.0, 0.0)
        return (self.seg_term / s, self.pose_term / s, self.rate_term / s)


@dataclass(frozen=True)
class ScoreGradient:
    """Partial derivatives of S w.r.t. each axis at one operating point."""

    d_seg: float          # dS/d(d_seg) — constant 100
    d_pose: float         # dS/d(d_pose) — = 5 / sqrt(10*d_pose), undefined at 0
    d_bytes: float        # dS/dB — constant 25/N_REF

    @property
    def seg_over_pose_marginal(self) -> float:
        if self.d_pose == 0.0 or not math.isfinite(self.d_pose):
            return 0.0
        return self.d_seg / self.d_pose

    @property
    def pose_over_seg_marginal(self) -> float:
        if self.d_seg == 0.0:
            return math.inf
        return self.d_pose / self.d_seg


@dataclass(frozen=True)
class OperatingRegime:
    """Classification of a candidate against the importance flip."""

    d_pose: float
    flip_threshold: float
    seg_dominates: bool
    pose_dominates: bool
    crossover_distance_log10: float
    marginal_ratio_seg_over_pose: float
    advice: str


def contest_score(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return the exact contest score.

    >>> contest_score(0.001, 0.0001, 178258)  # doctest: +ELLIPSIS
    0.21037...
    """
    if d_seg < 0.0 or d_pose < 0.0 or archive_bytes < 0:
        raise ValueError("contest score inputs must be non-negative")
    seg_term = SEG_COEFFICIENT * d_seg
    pose_term = math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose)
    rate_term = RATE_COEFFICIENT * archive_bytes / reference_bytes
    return seg_term + pose_term + rate_term


def score_decomposition(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> ScoreDecomposition:
    """Return per-term contribution to contest score."""
    return ScoreDecomposition(
        seg_term=SEG_COEFFICIENT * d_seg,
        pose_term=math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose),
        rate_term=RATE_COEFFICIENT * archive_bytes / reference_bytes,
    )


def score_gradient(
    d_seg: float,
    d_pose: float,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> ScoreGradient:
    """Return marginal sensitivities at the supplied operating point.

    The pose gradient is ``5 / sqrt(10 * d_pose)``, which is unbounded at
    d_pose=0; we return ``math.inf`` in that case so callers can detect
    the singularity rather than hit a ZeroDivisionError.
    """
    if d_pose < 0.0:
        raise ValueError("d_pose must be non-negative")
    d_pose_grad = (
        math.inf
        if d_pose == 0.0
        else 0.5 * math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT / d_pose)
    )
    return ScoreGradient(
        d_seg=SEG_COEFFICIENT,
        d_pose=d_pose_grad,
        d_bytes=RATE_COEFFICIENT / reference_bytes,
    )


def importance_flip_threshold() -> float:
    """The d_pose value where SegNet vs PoseNet marginals cross.

    Setting ``dS/d(d_seg) = dS/d(d_pose)`` gives:

        100 = 5 / sqrt(10 * d_pose)
        sqrt(10 * d_pose) = 0.05
        10 * d_pose = 2.5e-3
        d_pose = 2.5e-4

    Below this threshold the pose axis has a steeper score gradient than
    seg; above it, seg dominates.

    Symbolic derivation (kept symbolic so a future change to coefficients
    reshapes the threshold automatically):

        SEG = 0.5 * sqrt(POSE_INSIDE / x)
        4 * SEG**2 = POSE_INSIDE / x
        x = POSE_INSIDE / (4 * SEG**2)

    With SEG=100, POSE_INSIDE=10: x = 10 / 40000 = 2.5e-4.
    """
    return POSE_COEFFICIENT_INSIDE_SQRT / (4.0 * SEG_COEFFICIENT * SEG_COEFFICIENT)


def operating_regime(
    d_pose: float,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> OperatingRegime:
    """Classify a candidate against the SegNet/PoseNet importance flip."""
    threshold = importance_flip_threshold()
    if d_pose <= 0.0:
        return OperatingRegime(
            d_pose=d_pose,
            flip_threshold=threshold,
            seg_dominates=False,
            pose_dominates=True,
            crossover_distance_log10=math.inf,
            marginal_ratio_seg_over_pose=0.0,
            advice="d_pose at machine zero; pose marginal is unbounded — any pose perturbation is huge",
        )
    grad = score_gradient(0.0, d_pose, reference_bytes=reference_bytes)
    seg_over_pose = grad.seg_over_pose_marginal
    distance = math.log10(d_pose / threshold)
    if d_pose < threshold:
        advice = (
            f"pose-dominated regime (d_pose {d_pose:.2e} < {threshold:.2e}); "
            f"prioritize pose-targeted lanes — they have {1.0/seg_over_pose:.2f}x the "
            "marginal score-per-byte vs seg lanes here"
        )
    else:
        advice = (
            f"seg-dominated regime (d_pose {d_pose:.2e} >= {threshold:.2e}); "
            f"prioritize seg-targeted lanes — they have {seg_over_pose:.2f}x the "
            "marginal score-per-byte vs pose lanes here"
        )
    return OperatingRegime(
        d_pose=d_pose,
        flip_threshold=threshold,
        seg_dominates=d_pose > threshold,
        pose_dominates=d_pose < threshold,
        crossover_distance_log10=distance,
        marginal_ratio_seg_over_pose=seg_over_pose,
        advice=advice,
    )


def marginal_value_per_byte(
    axis: Literal["seg", "pose", "bytes"],
    *,
    d_pose_at_operating_point: float = 0.0,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return the score-points-saved per byte of axis-aligned spend.

    For the bytes axis this is just ``25 / reference_bytes``. For the seg
    and pose axes it is the partial derivative w.r.t. that axis at the
    supplied operating point — interpreted as the score reduction per
    one-unit reduction of that axis. The dispatcher uses this to budget
    candidate spend.
    """
    grad = score_gradient(0.0, d_pose_at_operating_point, reference_bytes=reference_bytes)
    if axis == "seg":
        return grad.d_seg
    if axis == "pose":
        return grad.d_pose
    if axis == "bytes":
        return grad.d_bytes
    raise ValueError(f"unknown axis: {axis!r}")


def information_floor(
    archive_bytes: int,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return the score floor at perfect d_seg=0 + perfect d_pose=0.

    This is the rate-only contribution. It is a *strict lower bound* on the
    achievable score for any archive of this byte budget; you cannot beat
    it without compressing the archive itself.

    For the PR103-on-PR106 active anchor (185,578 bytes), the information
    floor is ~0.12357. Any score above that is paid in d_seg + d_pose
    distortion.
    """
    if archive_bytes < 0:
        raise ValueError("archive_bytes must be non-negative")
    return RATE_COEFFICIENT * archive_bytes / reference_bytes


def project_onto_pareto_envelope(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    *,
    seg_floor: float = 0.0,
    pose_floor: float = 0.0,
    byte_floor: int = 0,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> tuple[float, dict[str, float]]:
    """Closed-form best-case projection onto the 3-axis Pareto envelope.

    Given a candidate (d_seg, d_pose, B) and lower-bound floors on each
    axis (e.g., known information-theoretic limits or empirical floors),
    returns (envelope_score, slack_per_axis).

    ``envelope_score`` is the score achievable IF every axis simultaneously
    reaches its floor. ``slack_per_axis`` reports how much each axis is
    above its floor — this is the available improvement headroom.

    This is the cathedral's "all axes simultaneously optimal" oracle. Real
    candidates cannot reach this point if the floors are mutually
    incompatible, but the slack is a useful planning signal.
    """
    if seg_floor < 0.0 or pose_floor < 0.0 or byte_floor < 0:
        raise ValueError("floors must be non-negative")
    if d_seg < seg_floor or d_pose < pose_floor or archive_bytes < byte_floor:
        raise ValueError(
            "candidate is below stated floor on at least one axis; "
            "either the floor is wrong or the score is invalid"
        )
    envelope = contest_score(seg_floor, pose_floor, byte_floor, reference_bytes=reference_bytes)
    slack = {
        "seg_slack": d_seg - seg_floor,
        "pose_slack": d_pose - pose_floor,
        "byte_slack": float(archive_bytes - byte_floor),
        "seg_score_slack": SEG_COEFFICIENT * (d_seg - seg_floor),
        "pose_score_slack": (
            math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose)
            - math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pose_floor)
        ),
        "byte_score_slack": RATE_COEFFICIENT * (archive_bytes - byte_floor) / reference_bytes,
    }
    return envelope, slack


def equal_score_curve_d_pose(
    target_score: float,
    d_seg: float,
    archive_bytes: int,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float | None:
    """Solve for the d_pose value that makes the score equal target_score.

    Returns None if no non-negative solution exists (i.e., the seg+rate
    terms already exceed target_score).

    Useful for dispatch budgeting: "we need score < 0.190; given d_seg
    and B, how good must pose be?"
    """
    seg_term = SEG_COEFFICIENT * d_seg
    rate_term = RATE_COEFFICIENT * archive_bytes / reference_bytes
    pose_term_required = target_score - seg_term - rate_term
    if pose_term_required < 0.0:
        return None
    return (pose_term_required ** 2) / POSE_COEFFICIENT_INSIDE_SQRT


def equal_score_curve_archive_bytes(
    target_score: float,
    d_seg: float,
    d_pose: float,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> int | None:
    """Solve for the archive byte budget that makes score equal target.

    Returns None if no positive byte budget achieves it (seg+pose already
    exceeds target).
    """
    seg_term = SEG_COEFFICIENT * d_seg
    pose_term = math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose)
    rate_term_required = target_score - seg_term - pose_term
    if rate_term_required < 0.0:
        return None
    return int(rate_term_required * reference_bytes / RATE_COEFFICIENT)


def predict_cpu_axis_marginals(
    archive_features: dict | None = None,
    *,
    d_seg_cuda: float,
    d_pose_cuda: float,
    archive_class: str = "hnerv",
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> dict[str, float]:
    """Compute CPU-axis marginal sensitivities at the operating point.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the legacy
    score-axis marginals computed on CUDA are MISLEADING for the contest
    leaderboard, which ranks by ``--device cpu`` eval. The pose axis on CPU
    has a SATURATED operating point (because of the 5× pose collapse), so
    the same CUDA pose delta translates to a much smaller CPU pose delta.

    This helper rebases the operating point from CUDA to CPU using the
    canonical calibration (:mod:`tac.optimization.cuda_cpu_axis_calibration`)
    and returns the per-axis marginals on the CPU side, plus the seg-vs-pose
    marginal ratio at the CPU operating point.

    Args:
        archive_features: optional dict of archive metadata (currently unused;
            reserved for future per-archive calibration overrides).
        d_seg_cuda: CUDA-axis seg distortion at the operating point.
        d_pose_cuda: CUDA-axis pose distortion at the operating point.
        archive_class: calibration class for R_pose / R_seg (default
            ``"hnerv"``).
        reference_bytes: contest reference bytes (default
            ``CONTEST_REFERENCE_BYTES``).

    Returns:
        Dict with keys:
          * ``pose_marginal``: dS_cpu / d(d_pose_cpu) at the rebased point.
          * ``seg_marginal``: dS_cpu / d(d_seg_cpu) (constant 100).
          * ``bytes_marginal``: dS / dB (constant; identical CUDA/CPU).
          * ``seg_over_pose_marginal_cpu``: ratio at CPU operating point.
          * ``pose_over_seg_marginal_cpu``: inverse ratio.
          * ``cuda_d_pose``, ``cpu_d_pose``: rebased pose values for sanity.
          * ``cuda_d_seg``, ``cpu_d_seg``: rebased seg values for sanity.
    """
    if d_pose_cuda < 0.0 or d_seg_cuda < 0.0:
        raise ValueError("d_pose_cuda and d_seg_cuda must be non-negative")
    from tac.optimization.cuda_cpu_axis_calibration import CudaCpuCalibration
    cal = CudaCpuCalibration(architecture_class=archive_class)
    cpu_d_pose = cal.effective_pose_loss_for_cpu(d_pose_cuda)
    cpu_d_seg = max(0.0, d_seg_cuda / cal.r_seg)
    cpu_grad = score_gradient(cpu_d_seg, cpu_d_pose, reference_bytes=reference_bytes)
    pose_marginal = cpu_grad.d_pose
    seg_marginal = cpu_grad.d_seg
    bytes_marginal = cpu_grad.d_bytes
    seg_over_pose = cpu_grad.seg_over_pose_marginal
    pose_over_seg = cpu_grad.pose_over_seg_marginal
    return {
        "pose_marginal": pose_marginal,
        "seg_marginal": seg_marginal,
        "bytes_marginal": bytes_marginal,
        "seg_over_pose_marginal_cpu": seg_over_pose,
        "pose_over_seg_marginal_cpu": pose_over_seg,
        "cuda_d_pose": d_pose_cuda,
        "cpu_d_pose": cpu_d_pose,
        "cuda_d_seg": d_seg_cuda,
        "cpu_d_seg": cpu_d_seg,
        "calibration_class": archive_class,
    }


__all__ = [
    "CONTEST_REFERENCE_BYTES",
    "POSE_COEFFICIENT_INSIDE_SQRT",
    "RATE_COEFFICIENT",
    "SEG_COEFFICIENT",
    "OperatingRegime",
    "ScoreDecomposition",
    "ScoreGradient",
    "contest_score",
    "equal_score_curve_archive_bytes",
    "equal_score_curve_d_pose",
    "importance_flip_threshold",
    "information_floor",
    "marginal_value_per_byte",
    "operating_regime",
    "predict_cpu_axis_marginals",
    "project_onto_pareto_envelope",
    "score_decomposition",
    "score_gradient",
]
