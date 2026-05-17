# SPDX-License-Identifier: MIT
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


@dataclass(frozen=True)
class TargetByteBudget:
    """Closed-form byte budget for a target score under fixed distortion floors.

    This is a planning artifact, not score evidence. It answers questions like:

        "If CPU pose is at the measured floor and SegNet is held at a chosen
        floor, how small must the archive be to reach score < 0.17?"

    ``max_archive_bytes`` is ``None`` when the supplied distortion floors
    already exceed the target before spending any bytes.
    """

    target_score: float
    d_seg_floor: float
    d_pose_floor: float
    distortion_floor_score: float
    rate_term_budget: float
    max_archive_bytes: int | None
    current_archive_bytes: int | None
    required_savings_bytes: int | None
    feasible_under_floors: bool
    evidence_grade: str = "[prediction; closed-form target byte budget]"
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    @property
    def blocker(self) -> str | None:
        if self.feasible_under_floors:
            return None
        return "distortion_floors_exceed_target_before_rate"


@dataclass(frozen=True)
class RateOnlyDeltaAudit:
    """Closed-form audit of a claimed score gain from byte savings alone.

    This is a false-authority guard for claims like "compress section X from
    A bytes to B bytes, therefore score improves by Y." Since the contest rate
    term is linear, byte-only claims have a hard upper bound independent of
    model quality: ``score_saving = saved_bytes * 25 / N_REF``.
    """

    original_bytes: int
    candidate_bytes: int
    claimed_score_saving: float
    saved_bytes: int
    rate_only_score_saving: float
    required_saved_bytes_for_claim: int
    max_possible_score_saving_if_section_removed: float
    feasible_from_candidate_savings: bool
    feasible_even_if_section_removed: bool
    overclaim_factor_vs_candidate: float
    missing_score_saving_after_candidate: float
    missing_saved_bytes_after_candidate: int
    evidence_grade: str = "[prediction; closed-form rate-only delta audit]"
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    @property
    def blocker(self) -> str | None:
        if self.feasible_from_candidate_savings:
            return None
        if self.feasible_even_if_section_removed:
            return "candidate_byte_savings_below_claim"
        return "claim_exceeds_rate_only_section_capacity"


@dataclass(frozen=True)
class PlannerAxisMarginals:
    """Marginals expressed in the candidate-builder coordinate system.

    ``target_axis="cuda_internal"`` means the candidate delta and the score
    target are both measured on CUDA. ``target_axis="cpu_leaderboard"`` means
    the candidate delta is still expressed as a CUDA-side change, but the
    score response is rebased through the calibrated CPU leaderboard axis via
    the chain rule. This keeps public-leaderboard planning from silently using
    raw CUDA priorities.
    """

    target_axis: Literal["cuda_internal", "cpu_leaderboard"]
    input_axis: Literal["cuda_candidate_delta"]
    cuda_d_seg: float
    cuda_d_pose: float
    effective_d_seg: float
    effective_d_pose: float
    archive_bytes: int
    seg_marginal: float
    pose_marginal: float
    bytes_marginal: float
    seg_chain_scale: float
    pose_chain_scale: float
    priority_axis: Literal["seg", "pose", "bytes"]
    tied_axes: tuple[str, ...]
    calibration_class: str
    evidence_grade: str
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


def _priority_axis_with_ties(
    *,
    seg_marginal: float,
    pose_marginal: float,
    bytes_marginal: float,
    archive_bytes: int,
    tie_rtol: float = 1e-9,
) -> tuple[Literal["seg", "pose", "bytes"], tuple[str, ...]]:
    """Return the highest-priority axis plus ties at the current scale.

    Seg and pose marginals are score-points per distortion-unit. The byte
    marginal is score-points per byte, so it is scaled by current archive
    bytes to keep the legacy planner's "zero this axis" comparison stable.
    """
    if not math.isfinite(pose_marginal):
        return "pose", ("pose",)
    scaled_bytes = bytes_marginal * float(archive_bytes)
    scores: list[tuple[float, Literal["seg", "pose", "bytes"]]] = [
        (seg_marginal, "seg"),
        (pose_marginal, "pose"),
        (scaled_bytes, "bytes"),
    ]
    winner_score = max(score for score, _ in scores)
    tied = tuple(
        name
        for score, name in scores
        if math.isclose(score, winner_score, rel_tol=tie_rtol, abs_tol=tie_rtol)
    )
    winner: Literal["seg", "pose", "bytes"] = tied[0]  # type: ignore[assignment]
    return winner, tied


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
    tie_rtol: float = 1e-9,
) -> OperatingRegime:
    """Classify a candidate against the SegNet/PoseNet importance flip.

    Three branches: pose-dominated (d_pose < threshold), tied (within
    ``tie_rtol`` of threshold), and seg-dominated (d_pose > threshold). Both
    the boolean dominance flags and the advice text use the same comparison
    semantics so the tie case is explicit, not silently routed to one axis.
    """
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
    is_tied = math.isclose(d_pose, threshold, rel_tol=tie_rtol)
    if is_tied:
        advice = (
            f"TIED regime (d_pose {d_pose:.2e} ≈ flip threshold {threshold:.2e} "
            f"within rtol={tie_rtol:.0e}); seg and pose marginals are equal — "
            "consider parallel attack on BOTH axes; dispatch should not route "
            "to a single axis at this operating point"
        )
    elif d_pose < threshold:
        advice = (
            f"pose-dominated regime (d_pose {d_pose:.2e} < {threshold:.2e}); "
            f"prioritize pose-targeted lanes — they have {1.0/seg_over_pose:.2f}x the "
            "marginal score-per-byte vs seg lanes here"
        )
    else:
        advice = (
            f"seg-dominated regime (d_pose {d_pose:.2e} > {threshold:.2e}); "
            f"prioritize seg-targeted lanes — they have {seg_over_pose:.2f}x the "
            "marginal score-per-byte vs pose lanes here"
        )
    # Strict comparison so seg_dominates AND pose_dominates are BOTH False at
    # the tie. Tied state is the (False, False) configuration; the advice
    # text and consumers should branch on `seg_dominates == pose_dominates`
    # to detect ties.
    return OperatingRegime(
        d_pose=d_pose,
        flip_threshold=threshold,
        seg_dominates=(not is_tied) and d_pose > threshold,
        pose_dominates=(not is_tied) and d_pose < threshold,
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


def score_saving_from_byte_savings(
    saved_bytes: int | float,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return the exact score improvement from deleting charged bytes only."""
    if saved_bytes < 0:
        raise ValueError("saved_bytes must be non-negative")
    return RATE_COEFFICIENT * float(saved_bytes) / reference_bytes


def required_byte_savings_for_score_delta(
    score_delta: float,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> int:
    """Return charged bytes required for a byte-only score improvement.

    ``score_delta`` is a positive improvement magnitude. The result is rounded
    up because fractional bytes cannot be removed from an archive.
    """
    if score_delta < 0.0:
        raise ValueError("score_delta must be non-negative")
    return math.ceil(score_delta * reference_bytes / RATE_COEFFICIENT)


def audit_rate_only_delta_claim(
    *,
    original_bytes: int,
    candidate_bytes: int,
    claimed_score_saving: float,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
    tolerance: float = 1e-12,
) -> RateOnlyDeltaAudit:
    """Audit whether a claimed improvement can come from rate savings alone.

    The audit is intentionally axis-independent because the byte term is the
    same on CPU and CUDA. If this fails, a candidate can still be valuable, but
    the missing score delta must come from SegNet/PoseNet component movement or
    from a different archive section, not from the named rate-only shrink.
    """
    if original_bytes < 0 or candidate_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    if claimed_score_saving < 0.0:
        raise ValueError("claimed_score_saving must be non-negative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    saved_bytes = original_bytes - candidate_bytes
    effective_saved_bytes = max(0, saved_bytes)
    rate_saving = score_saving_from_byte_savings(
        effective_saved_bytes,
        reference_bytes=reference_bytes,
    )
    required = required_byte_savings_for_score_delta(
        claimed_score_saving,
        reference_bytes=reference_bytes,
    )
    max_possible = score_saving_from_byte_savings(
        original_bytes,
        reference_bytes=reference_bytes,
    )
    feasible_candidate = rate_saving + tolerance >= claimed_score_saving
    feasible_section = max_possible + tolerance >= claimed_score_saving
    missing_score = max(0.0, claimed_score_saving - rate_saving)
    missing_bytes = required_byte_savings_for_score_delta(
        missing_score,
        reference_bytes=reference_bytes,
    )
    overclaim = (
        math.inf
        if rate_saving == 0.0 and claimed_score_saving > 0.0
        else claimed_score_saving / rate_saving
        if rate_saving > 0.0
        else 1.0
    )
    return RateOnlyDeltaAudit(
        original_bytes=original_bytes,
        candidate_bytes=candidate_bytes,
        claimed_score_saving=claimed_score_saving,
        saved_bytes=saved_bytes,
        rate_only_score_saving=rate_saving,
        required_saved_bytes_for_claim=required,
        max_possible_score_saving_if_section_removed=max_possible,
        feasible_from_candidate_savings=feasible_candidate,
        feasible_even_if_section_removed=feasible_section,
        overclaim_factor_vs_candidate=overclaim,
        missing_score_saving_after_candidate=missing_score,
        missing_saved_bytes_after_candidate=missing_bytes,
    )


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


def target_byte_budget_for_score(
    *,
    target_score: float,
    d_seg_floor: float,
    d_pose_floor: float,
    current_archive_bytes: int | None = None,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> TargetByteBudget:
    """Return the maximum archive byte budget for a target score.

    The algebra is direct from the contest objective:

    ``B_max = floor((S_target - 100*d_seg - sqrt(10*d_pose)) * N / 25)``.

    This function is intentionally CPU/GPU agnostic: callers must pass the
    floor values for the score axis they are planning against. For public
    leaderboard planning, pass CPU-axis floors from paired CPU eval or a
    clearly tagged prediction. For internal promotion, use exact CUDA floors.

    Returns a :class:`TargetByteBudget` tagged as prediction-only; it cannot
    promote, rank, retire, or dispatch a candidate without exact eval custody.
    """
    if target_score < 0.0:
        raise ValueError("target_score must be non-negative")
    if d_seg_floor < 0.0 or d_pose_floor < 0.0:
        raise ValueError("distortion floors must be non-negative")
    if current_archive_bytes is not None and current_archive_bytes < 0:
        raise ValueError("current_archive_bytes must be non-negative")

    distortion_floor_score = (
        SEG_COEFFICIENT * d_seg_floor
        + math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose_floor)
    )
    rate_term_budget = target_score - distortion_floor_score
    feasible = rate_term_budget >= 0.0
    max_bytes = (
        math.floor(rate_term_budget * reference_bytes / RATE_COEFFICIENT)
        if feasible
        else None
    )
    required_savings: int | None = None
    if max_bytes is not None and current_archive_bytes is not None:
        required_savings = max(0, int(current_archive_bytes) - max_bytes)

    return TargetByteBudget(
        target_score=target_score,
        d_seg_floor=d_seg_floor,
        d_pose_floor=d_pose_floor,
        distortion_floor_score=distortion_floor_score,
        rate_term_budget=rate_term_budget,
        max_archive_bytes=max_bytes,
        current_archive_bytes=current_archive_bytes,
        required_savings_bytes=required_savings,
        feasible_under_floors=feasible,
    )


def planner_axis_marginals(
    *,
    target_axis: Literal["cuda_internal", "cpu_leaderboard"],
    cuda_d_seg: float,
    cuda_d_pose: float,
    archive_bytes: int,
    archive_class: str = "hnerv",
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
    tie_rtol: float = 1e-9,
) -> PlannerAxisMarginals:
    """Return target-axis-aware marginals for solver/planner ranking.

    The public leaderboard's CPU axis is not interchangeable with the CUDA
    axis. For CPU leaderboard planning, this helper treats input candidate
    deltas as CUDA-side deltas and applies the calibrated chain-rule scales:

    ``dS_cpu / d(d_seg_cuda) = dS_cpu / d(d_seg_cpu) * (1 / R_seg)``
    ``dS_cpu / d(d_pose_cuda) = dS_cpu / d(d_pose_cpu) * (1 / R_pose)``

    The result is prediction-only planner metadata. It cannot promote, rank,
    retire, dispatch, or claim a score without paired exact-eval custody.
    """
    if cuda_d_seg < 0.0 or cuda_d_pose < 0.0 or archive_bytes < 0:
        raise ValueError("planner axis marginal inputs must be non-negative")

    if target_axis == "cuda_internal":
        grad = score_gradient(cuda_d_seg, cuda_d_pose, reference_bytes=reference_bytes)
        seg_marginal = grad.d_seg
        pose_marginal = grad.d_pose
        bytes_marginal = grad.d_bytes
        effective_d_seg = cuda_d_seg
        effective_d_pose = cuda_d_pose
        seg_scale = 1.0
        pose_scale = 1.0
        calibration_class = archive_class
        evidence_grade = "[prediction; cuda-internal planner marginals]"
    elif target_axis == "cpu_leaderboard":
        from tac.optimization.cuda_cpu_axis_calibration import CudaCpuCalibration

        cal = CudaCpuCalibration(architecture_class=archive_class)
        effective_d_pose = cal.effective_pose_loss_for_cpu(cuda_d_pose)
        effective_d_seg = max(0.0, cuda_d_seg / cal.r_seg)
        grad = score_gradient(
            effective_d_seg,
            effective_d_pose,
            reference_bytes=reference_bytes,
        )
        seg_scale = 1.0 / cal.r_seg
        pose_scale = 1.0 / cal.r_pose
        seg_marginal = grad.d_seg * seg_scale
        pose_marginal = grad.d_pose * pose_scale
        bytes_marginal = grad.d_bytes
        calibration_class = cal.architecture_class
        evidence_grade = "[prediction; cpu-leaderboard chain-rule planner marginals]"
    else:
        raise ValueError(f"unknown target_axis: {target_axis!r}")

    priority_axis, tied_axes = _priority_axis_with_ties(
        seg_marginal=seg_marginal,
        pose_marginal=pose_marginal,
        bytes_marginal=bytes_marginal,
        archive_bytes=archive_bytes,
        tie_rtol=tie_rtol,
    )
    return PlannerAxisMarginals(
        target_axis=target_axis,
        input_axis="cuda_candidate_delta",
        cuda_d_seg=cuda_d_seg,
        cuda_d_pose=cuda_d_pose,
        effective_d_seg=effective_d_seg,
        effective_d_pose=effective_d_pose,
        archive_bytes=archive_bytes,
        seg_marginal=seg_marginal,
        pose_marginal=pose_marginal,
        bytes_marginal=bytes_marginal,
        seg_chain_scale=seg_scale,
        pose_chain_scale=pose_scale,
        priority_axis=priority_axis,
        tied_axes=tied_axes,
        calibration_class=calibration_class,
        evidence_grade=evidence_grade,
    )


def predict_cpu_axis_marginals(
    archive_features: dict | None = None,
    *,
    d_seg_cuda: float,
    d_pose_cuda: float,
    archive_class: str = "hnerv",
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> dict[str, float | bool | str]:
    """Compute CPU-axis marginal sensitivities at the operating point.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the legacy
    score-axis marginals computed on CUDA can be misleading for CPU-axis
    exploration. This helper is a predicted calibration view, not a score
    claim or a promotion artifact.

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
        "evidence_grade": "[prediction; cuda_cpu_axis_calibration]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


@dataclass(frozen=True)
class DualAxisDispatchRecommendation:
    """Operating-point-aware dual-axis (CUDA + CPU) dispatch recommendation.

    Combines :func:`score_gradient` and :func:`predict_cpu_axis_marginals` into
    a single artifact that names the score-cheapest axis on each evaluation
    substrate, flags axis-priority divergence, and maps each axis to the
    Phase A decisions that attack it. Pure prediction; no score claim.

    Cross-ref: ``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``
    Decision 2 (score-gradient) attacks seg+pose; Decision 3 (sensitivity-quant)
    attacks bytes; Decision 1 (co-trained Ballé) attacks bytes via co-design.

    Tie detection: when two or more axes have marginal values that are equal
    within ``tie_rtol``, ``cuda_tied_axes`` / ``cpu_tied_axes`` lists every
    axis at the maximum. The single ``cuda_priority_axis`` /
    ``cpu_priority_axis`` is the deterministic tiebreak (seg-first, then pose,
    then bytes — kept stable for backward compat); operators should branch on
    ``len(tied_axes) > 1`` to surface a parallel-attack recommendation.
    """

    cuda_d_seg: float
    cuda_d_pose: float
    archive_bytes: int
    cuda_seg_marginal: float
    cuda_pose_marginal: float
    cuda_bytes_marginal: float
    cuda_priority_axis: Literal["seg", "pose", "bytes"]
    cpu_d_seg: float
    cpu_d_pose: float
    cpu_seg_marginal: float
    cpu_pose_marginal: float
    cpu_bytes_marginal: float
    cpu_priority_axis: Literal["seg", "pose", "bytes"]
    axis_priority_differs: bool
    target_score_cpu: float | None
    cpu_score_at_operating_point: float
    cpu_score_gap_to_target: float | None
    decision_attack_map: dict[str, list[str]]
    advice: str
    cuda_tied_axes: tuple[str, ...] = ()
    cpu_tied_axes: tuple[str, ...] = ()
    tie_rtol: float = 1e-9

    @property
    def evidence_grade(self) -> str:
        return "[prediction; dual-axis-dispatch-recommendation]"

    @property
    def score_claim(self) -> bool:
        return False

    @property
    def promotion_eligible(self) -> bool:
        return False


def recommend_dispatch_axis_dual(
    *,
    cuda_d_seg: float,
    cuda_d_pose: float,
    archive_bytes: int,
    target_score_cpu: float | None = None,
    archive_class: str = "hnerv",
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> DualAxisDispatchRecommendation:
    """Synthesize a dual-axis (CUDA + CPU) dispatch recommendation.

    The contest leaderboard ranks by CPU eval (per CLAUDE.md "Submission
    auth eval — BOTH CPU AND CUDA"). Our internal CUDA anchor is correlated
    but offset; the constant ~0.033 CUDA-CPU gap on HNeRV-cluster archives
    means optimizing for CUDA usually transfers to CPU, but axis priority
    can diverge near the importance flip threshold (d_pose = 2.5e-4).

    This helper:
      * computes per-axis marginals at the operating point on BOTH substrates;
      * names the score-cheapest axis on each (the priority axis);
      * flags divergence (rare but possible near the flip);
      * maps each axis to Phase A decisions that attack it;
      * computes the CPU score gap to a target if supplied.

    Used by :mod:`tools.dispatch_advisor` to rank Phase A ablations after
    each anchor lands. NOT a score claim and NOT promotion-eligible — this
    is a planning view that surfaces priorities; the actual exact-eval
    artifact must produce a ``[contest-CUDA]`` or ``[contest-CPU]`` tag.
    """
    if cuda_d_seg < 0.0 or cuda_d_pose < 0.0 or archive_bytes < 0:
        raise ValueError("dispatch recommendation inputs must be non-negative")

    cuda_grad = score_gradient(cuda_d_seg, cuda_d_pose, reference_bytes=reference_bytes)
    cpu_view = predict_cpu_axis_marginals(
        d_seg_cuda=cuda_d_seg,
        d_pose_cuda=cuda_d_pose,
        archive_class=archive_class,
        reference_bytes=reference_bytes,
    )
    tie_rtol = 1e-9

    cuda_priority, cuda_tied = _priority_axis_with_ties(
        seg_marginal=cuda_grad.d_seg,
        pose_marginal=cuda_grad.d_pose,
        bytes_marginal=cuda_grad.d_bytes,
        archive_bytes=archive_bytes,
        tie_rtol=tie_rtol,
    )
    cpu_priority, cpu_tied = _priority_axis_with_ties(
        seg_marginal=float(cpu_view["seg_marginal"]),
        pose_marginal=float(cpu_view["pose_marginal"]),
        bytes_marginal=float(cpu_view["bytes_marginal"]),
        archive_bytes=archive_bytes,
        tie_rtol=tie_rtol,
    )
    # Two priorities differ either when their deterministic winner differs OR
    # when one is at a tie and the other is not (parallel-attack vs single-axis).
    differs = (
        cuda_priority != cpu_priority
        or len(cuda_tied) != len(cpu_tied)
    )

    cpu_score_at_op = contest_score(
        float(cpu_view["cpu_d_seg"]),
        float(cpu_view["cpu_d_pose"]),
        archive_bytes,
        reference_bytes=reference_bytes,
    )
    cpu_gap = (
        cpu_score_at_op - target_score_cpu
        if target_score_cpu is not None
        else None
    )

    decision_attack_map: dict[str, list[str]] = {
        "seg": ["A1_score_gradient_segnet_kl", "A3_alt_mallat_wavelet_importance"],
        "pose": ["A1_score_gradient_posenet_mse", "A4_alt_filler_stc_pose_encoding"],
        "bytes": [
            "A2_sensitivity_aware_quantization",
            "A4_charm_co_trained_hyperprior",
            "A5_frame_conditional_bit_budget",
            "A6_block_fp_x_hyperprior_compose",
        ],
    }

    if len(cuda_tied) > 1 or len(cpu_tied) > 1:
        advice = (
            f"AXIS TIE DETECTED at flip threshold: cuda_tied={cuda_tied!r}, "
            f"cpu_tied={cpu_tied!r}. Marginals are equal within rtol={tie_rtol:.0e}; "
            "dispatch should attack ALL tied axes in parallel rather than route "
            "to a single deterministic-tiebreak winner."
        )
    elif differs:
        advice = (
            f"AXIS-PRIORITY DIVERGENCE: CUDA priority is {cuda_priority!r} but "
            f"CPU priority is {cpu_priority!r}. Per leaderboard ranking on the "
            f"CPU axis, prefer {cpu_priority!r}-attacking Phase A decisions. "
            f"Phase A actuator should weight {cpu_priority!r} ablations higher."
        )
    else:
        advice = (
            f"axis priority CONSISTENT across CUDA + CPU: {cuda_priority!r}. "
            f"Phase A should prioritize {cuda_priority!r}-attacking ablations: "
            f"{decision_attack_map[cuda_priority]}"
        )
    if target_score_cpu is not None and cpu_gap is not None:
        advice += (
            f" CPU score gap to target {target_score_cpu:.4f}: "
            f"{cpu_gap:+.4f} (positive = above target, more work needed)."
        )

    return DualAxisDispatchRecommendation(
        cuda_d_seg=cuda_d_seg,
        cuda_d_pose=cuda_d_pose,
        archive_bytes=archive_bytes,
        cuda_seg_marginal=cuda_grad.d_seg,
        cuda_pose_marginal=cuda_grad.d_pose,
        cuda_bytes_marginal=cuda_grad.d_bytes,
        cuda_priority_axis=cuda_priority,
        cpu_d_seg=float(cpu_view["cpu_d_seg"]),
        cpu_d_pose=float(cpu_view["cpu_d_pose"]),
        cpu_seg_marginal=float(cpu_view["seg_marginal"]),
        cpu_pose_marginal=float(cpu_view["pose_marginal"]),
        cpu_bytes_marginal=float(cpu_view["bytes_marginal"]),
        cpu_priority_axis=cpu_priority,
        axis_priority_differs=differs,
        target_score_cpu=target_score_cpu,
        cpu_score_at_operating_point=cpu_score_at_op,
        cpu_score_gap_to_target=cpu_gap,
        decision_attack_map=decision_attack_map,
        advice=advice,
        cuda_tied_axes=cuda_tied,
        cpu_tied_axes=cpu_tied,
        tie_rtol=tie_rtol,
    )


__all__ = [
    "CONTEST_REFERENCE_BYTES",
    "POSE_COEFFICIENT_INSIDE_SQRT",
    "RATE_COEFFICIENT",
    "SEG_COEFFICIENT",
    "DualAxisDispatchRecommendation",
    "OperatingRegime",
    "PlannerAxisMarginals",
    "RateOnlyDeltaAudit",
    "ScoreDecomposition",
    "ScoreGradient",
    "TargetByteBudget",
    "audit_rate_only_delta_claim",
    "contest_score",
    "equal_score_curve_archive_bytes",
    "equal_score_curve_d_pose",
    "importance_flip_threshold",
    "information_floor",
    "marginal_value_per_byte",
    "operating_regime",
    "planner_axis_marginals",
    "predict_cpu_axis_marginals",
    "project_onto_pareto_envelope",
    "recommend_dispatch_axis_dual",
    "required_byte_savings_for_score_delta",
    "score_decomposition",
    "score_gradient",
    "score_saving_from_byte_savings",
    "target_byte_budget_for_score",
]
