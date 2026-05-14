# SPDX-License-Identifier: MIT
"""Canonical CUDA→CPU axis calibration helper.

The contest leaderboard ranks by ``--device cpu`` evaluation, NOT ``--device
cuda``. Empirical sweep across the HNeRV cluster (PR100/101/102/103/105) on
2026-05-08 measured a near-constant axis-rebase from CUDA scores to CPU
scores:

This module is analysis-only and emits no archive bytes; deterministic-bytes acceptable
because byte-match is N/A for this pure calibration transform.

  * ``R_pose = d_pose_cuda / d_pose_cpu ≈ 5.04 ± 0.10``  (HNeRV cluster)
  * ``R_seg  = d_seg_cuda  / d_seg_cpu  ≈ 1.17 ± 0.01``  (HNeRV cluster)
  * ``score(cpu) - score(cuda) ≈ -0.033`` (constant gap, dominated by pose
    sqrt term collapse on CPU)

Mechanism status: the 5× pose ratio is measured; the causal split is still
under investigation. Current hypotheses include GT loader drift
(``DaliVideoDataset`` vs ``AVVideoDataset``), CUDA/CPU reduction-order drift,
and network-internal amplification. Do not use this module as a mechanism
proof. Its operational estimator is the measured per-axis ratio.

This module exposes:

  * ``CudaCpuCalibration`` — class wrapping the calibrated transform with
    optional architecture-class overrides.
  * ``predict_cpu_from_cuda`` — convert a CUDA score to a CPU prediction
    band reflecting σ uncertainty.
  * ``predict_cuda_from_cpu`` — inverse direction.
  * ``is_at_pose_floor`` — whether CUDA pose is inside the empirically observed
    HNeRV floor band. This is advisory only, not a proof that bytes are wasted.
  * ``effective_pose_loss_for_cpu`` — the effective pose loss seen by the
    CPU evaluator, using the measured ratio.

Pure CPU + math. No torch, no scorer load. Safe for solver/recommender hot
paths.

Cross-references:

  * CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
    CONTEST-COMPLIANT HARDWARE" (commit b4919d24)
  * ``feedback_dual_axis_solver_integration_landed_20260508.md`` (memory)
  * ``.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md``
  * ``.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md``
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# -----------------------------------------------------------------------
# Empirical calibration constants (2026-05-08 HNeRV cluster sweep)
# -----------------------------------------------------------------------

R_POSE_HNERV: float = 5.04
"""Empirical mean of d_pose_cuda / d_pose_cpu across HNeRV cluster.

Sources: PR100/101/102/103/105 contest-CPU vs contest-CUDA evals,
2026-05-08. Stable across this cluster (std ≈ 0.10).
"""

R_POSE_HNERV_STD: float = 0.10
"""Standard deviation of R_POSE across HNeRV cluster sweep."""

R_SEG_HNERV: float = 1.17
"""Empirical mean of d_seg_cuda / d_seg_cpu across HNeRV cluster."""

R_SEG_HNERV_STD: float = 0.01
"""Standard deviation of R_SEG across HNeRV cluster sweep."""

SCORE_GAP_HNERV_CONSTANT: float = 0.033
"""Empirical mean of ``contest_score_cpu - contest_score_cuda`` across HNeRV
cluster. Negative because CPU pose is smaller -> CPU pose_term smaller ->
CPU score smaller. Used as a sanity-check residual after axis rebase.
"""

SCORE_GAP_HNERV_STD: float = 0.005
"""Standard deviation of the constant score gap; dominated by per-PR noise
in the constant-gap regression."""

# Below this CUDA pose value, the HNeRV cluster sits in an empirically observed
# CPU/CUDA drift band. This is an advisory trust-region marker for solvers, not
# a hard proof that lower CUDA pose is invisible on the official CPU axis.
CUDA_POSE_PRECISION_FLOOR: float = 1.4e-4
"""Advisory HNeRV CUDA pose floor band.

Empirically derived from paired CPU/CUDA HNeRV anchors. It is used as a
diagnostic marker only; operational CPU prediction uses ``d_pose_cuda/R_pose``.
"""

# Architectures with empirically-different calibration. As the 25-PR sweep
# lands, new entries land here.
KNOWN_ARCHITECTURE_CLASSES: tuple[str, ...] = (
    "hnerv",          # HNeRV cluster (PR100/101/102/103/105) — calibrated
    "qhnerv",         # PR104 — predicted similar; needs verification
    "h3_grayscale",   # PR97 — predicted lower R_pose (different decoder)
    "av1_high_pose",  # PR60 — predicted R_pose ≈ 1.0 at high pose substrate
    "unknown",        # default fallback
)

ARCHITECTURE_CLASS_ALIASES: dict[str, str] = {
    "hnerv_ft_microcodec": "hnerv",
    "hnerv_lc_v2": "hnerv",
    "hnerv_lc_ac": "hnerv",
    "hnerv_microcodec": "hnerv",
    "ff_packed_brotli_hnerv": "hnerv",
    "qhnerv_ft": "qhnerv",
    "qhnerv_ft_best": "qhnerv",
    "h3_av1_grayscale": "h3_grayscale",
    "h3_grayscale": "h3_grayscale",
    "raw_av1_yuv": "av1_high_pose",
}


def normalize_architecture_class(architecture_class: str) -> str:
    """Map registry/public-submission labels into calibration buckets.

    Unknown labels fail closed into the high-uncertainty ``"unknown"`` bucket
    instead of raising. That keeps solvers usable as new profile-registry names
    arrive while preserving the ``extrapolated`` calibration label.
    """
    key = architecture_class.strip().lower().replace("-", "_")
    if not key:
        return "unknown"
    if key in KNOWN_ARCHITECTURE_CLASSES:
        return key
    if key in ARCHITECTURE_CLASS_ALIASES:
        return ARCHITECTURE_CLASS_ALIASES[key]
    if key.startswith("hnerv_") or key.startswith("hnerv"):
        return "hnerv"
    if key.startswith("qhnerv_") or key.startswith("qhnerv"):
        return "qhnerv"
    if key.startswith("h3_") or "h3" in key:
        return "h3_grayscale"
    if "av1" in key and "pose" in key:
        return "av1_high_pose"
    return "unknown"

# -----------------------------------------------------------------------
# Calibration class
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBand:
    """Predicted score band with calibration uncertainty.

    Attributes:
        score_lo: low end of predicted score band (1σ).
        score_hi: high end of predicted score band (1σ).
        score_point: point estimate.
        calibration_quality: ``"hnerv-anchored"`` if architecture is in
            the calibrated cluster, else ``"extrapolated"``.
        sigma: combined σ used to form the band.
        notes: free-form notes from the calibration step.
    """
    score_lo: float
    score_hi: float
    score_point: float
    calibration_quality: str
    sigma: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "score_lo": self.score_lo,
            "score_hi": self.score_hi,
            "score_point": self.score_point,
            "calibration_quality": self.calibration_quality,
            "sigma": self.sigma,
            "notes": list(self.notes),
            "evidence_grade": "[prediction; cuda_cpu_axis_calibration]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


class CudaCpuCalibration:
    """Calibrated CUDA→CPU axis rebase.

    Use the class methods directly for pure functional access; instantiate
    when you want to plug architecture-class overrides for non-HNeRV
    families.

    Args:
        architecture_class: one of :data:`KNOWN_ARCHITECTURE_CLASSES`. The
            default ``"unknown"`` falls back to HNeRV constants but flags
            the calibration as ``"extrapolated"`` so downstream consumers
            can downweight the prediction.
        r_pose_override: replace the calibrated R_pose for this instance.
            Used when a per-architecture sweep has refined the constant.
        r_seg_override: replace the calibrated R_seg.
        score_gap_override: replace the calibrated constant score gap.
        sigma_inflation: multiply the σ band by this factor when the
            architecture is non-HNeRV (default 2.0).
    """

    def __init__(
        self,
        architecture_class: str = "unknown",
        *,
        r_pose_override: float | None = None,
        r_seg_override: float | None = None,
        score_gap_override: float | None = None,
        sigma_inflation: float = 2.0,
    ) -> None:
        self.requested_architecture_class = architecture_class
        self.architecture_class = normalize_architecture_class(architecture_class)
        self.r_pose = r_pose_override if r_pose_override is not None else R_POSE_HNERV
        self.r_seg = r_seg_override if r_seg_override is not None else R_SEG_HNERV
        self.score_gap = (
            score_gap_override if score_gap_override is not None else SCORE_GAP_HNERV_CONSTANT
        )
        self.sigma_inflation = float(sigma_inflation)
        self._is_anchored = self.architecture_class == "hnerv"

    # -- prediction --------------------------------------------------------

    def predict_cpu_from_cuda(
        self,
        cuda_score: float,
        *,
        d_pose_cuda: float | None = None,
        d_seg_cuda: float | None = None,
        archive_bytes: int | None = None,
    ) -> CalibrationBand:
        """Predict CPU score from CUDA score.

        Two prediction modes:

        1. **Constant-gap** (when only ``cuda_score`` is provided): apply
           the empirical constant gap. Fast but coarser.
        2. **Decomposed** (when all three components are provided): rebase
           the pose and seg axes via R_pose / R_seg, recompute the score
           term-by-term. Tighter band.

        The decomposed mode is preferred when the contest-rate-distortion
        decomposition is available.

        Args:
            cuda_score: CUDA score.
            d_pose_cuda: optional CUDA pose distortion.
            d_seg_cuda: optional CUDA seg distortion.
            archive_bytes: optional archive byte count.

        Returns:
            :class:`CalibrationBand`.
        """
        if not math.isfinite(cuda_score):
            raise ValueError("cuda_score must be finite")
        if cuda_score < 0.0:
            raise ValueError("cuda_score must be non-negative")
        notes: list[str] = []
        # Decomposed mode: full rebase.
        if (
            d_pose_cuda is not None
            and d_seg_cuda is not None
            and archive_bytes is not None
        ):
            # Saturation: when CUDA pose is below the precision floor, CPU
            # pose is what the CPU sees with the floor stripped.
            d_pose_cpu = self._rebase_pose(d_pose_cuda)
            d_seg_cpu = max(0.0, d_seg_cuda / self.r_seg)
            # Recompute the contest score on the CPU axes.
            from tac.score_geometry import contest_score
            cpu_score_point = contest_score(
                d_seg=d_seg_cpu,
                d_pose=d_pose_cpu,
                archive_bytes=int(archive_bytes),
            )
            notes.append("mode=decomposed")
            sigma = self._propagate_sigma_decomposed(
                d_pose_cuda=d_pose_cuda,
                d_seg_cuda=d_seg_cuda,
            )
        else:
            # Constant-gap mode.
            cpu_score_point = max(0.0, cuda_score - self.score_gap)
            notes.append("mode=constant_gap")
            sigma = SCORE_GAP_HNERV_STD * self.sigma_inflation if not self._is_anchored else SCORE_GAP_HNERV_STD
        quality = "hnerv-anchored" if self._is_anchored else "extrapolated"
        if not self._is_anchored:
            notes.append(
                f"architecture_class={self.architecture_class}; calibration "
                "extrapolated from HNeRV cluster — verify with per-arch sweep"
            )
        score_lo = max(0.0, cpu_score_point - sigma)
        score_hi = cpu_score_point + sigma
        return CalibrationBand(
            score_lo=score_lo,
            score_hi=score_hi,
            score_point=cpu_score_point,
            calibration_quality=quality,
            sigma=sigma,
            notes=tuple(notes),
        )

    def predict_cuda_from_cpu(
        self,
        cpu_score: float,
        *,
        d_pose_cpu: float | None = None,
        d_seg_cpu: float | None = None,
        archive_bytes: int | None = None,
    ) -> CalibrationBand:
        """Inverse direction: predict CUDA score from CPU score.

        See :meth:`predict_cpu_from_cuda` for mode semantics.
        """
        if not math.isfinite(cpu_score):
            raise ValueError("cpu_score must be finite")
        if cpu_score < 0.0:
            raise ValueError("cpu_score must be non-negative")
        notes: list[str] = []
        if (
            d_pose_cpu is not None
            and d_seg_cpu is not None
            and archive_bytes is not None
        ):
            d_pose_cuda = max(0.0, d_pose_cpu * self.r_pose)
            d_seg_cuda = max(0.0, d_seg_cpu * self.r_seg)
            from tac.score_geometry import contest_score
            cuda_score_point = contest_score(
                d_seg=d_seg_cuda,
                d_pose=d_pose_cuda,
                archive_bytes=int(archive_bytes),
            )
            notes.append("mode=decomposed")
            sigma = self._propagate_sigma_decomposed(
                d_pose_cuda=d_pose_cuda,
                d_seg_cuda=d_seg_cuda,
            )
        else:
            cuda_score_point = max(0.0, cpu_score + self.score_gap)
            notes.append("mode=constant_gap")
            sigma = SCORE_GAP_HNERV_STD * self.sigma_inflation if not self._is_anchored else SCORE_GAP_HNERV_STD
        quality = "hnerv-anchored" if self._is_anchored else "extrapolated"
        if not self._is_anchored:
            notes.append(
                f"architecture_class={self.architecture_class}; calibration "
                "extrapolated from HNeRV cluster — verify with per-arch sweep"
            )
        score_lo = max(0.0, cuda_score_point - sigma)
        score_hi = cuda_score_point + sigma
        return CalibrationBand(
            score_lo=score_lo,
            score_hi=score_hi,
            score_point=cuda_score_point,
            calibration_quality=quality,
            sigma=sigma,
            notes=tuple(notes),
        )

    # -- pose floor saturation --------------------------------------------

    def is_at_pose_floor(self, d_pose_cuda: float) -> bool:
        """Return True iff CUDA pose is inside the observed HNeRV floor band.

        This is an advisory marker for trust-region analysis. It is not a
        promotion/ranking proof and does not imply that additional pose
        improvement is worthless on the official CPU axis.
        """
        if d_pose_cuda < 0.0:
            raise ValueError("d_pose_cuda must be non-negative")
        return d_pose_cuda <= CUDA_POSE_PRECISION_FLOOR

    def effective_pose_loss_for_cpu(self, d_pose_cuda: float) -> float:
        """Return the effective pose loss the CPU evaluator sees.

        Operational estimator: ``d_pose_cuda / R_pose`` for all nonnegative
        values. The floor constant is advisory and is intentionally not
        subtracted here.
        """
        if d_pose_cuda < 0.0:
            raise ValueError("d_pose_cuda must be non-negative")
        return self._rebase_pose(d_pose_cuda)

    # -- internals --------------------------------------------------------

    def _rebase_pose(self, d_pose_cuda: float) -> float:
        """Rebase CUDA pose to CPU pose using the empirical R_pose ratio."""
        if d_pose_cuda < 0.0:
            raise ValueError("d_pose_cuda must be non-negative")
        return d_pose_cuda / self.r_pose

    def _propagate_sigma_decomposed(
        self, *, d_pose_cuda: float, d_seg_cuda: float
    ) -> float:
        """Propagate calibration uncertainty into a score-space σ.

        Uses first-order error propagation through the contest formula:
        d_score / d_R_pose = -0.5 * sqrt(10/d_pose) * (d_pose / R_pose²);
        d_score / d_R_seg = -100 * (d_seg / R_seg²).
        """
        # Pose sigma contribution:
        if d_pose_cuda > 0.0:
            d_pose_cpu = self._rebase_pose(d_pose_cuda)
            if d_pose_cpu > 0.0:
                # d_pose_cpu = d_pose_cuda/R. Therefore
                # |∂ sqrt(10*d_pose_cpu) / ∂R| =
                # 0.5 * sqrt(10*d_pose_cpu) / R.
                pose_grad = 0.5 * math.sqrt(10.0 * d_pose_cpu) / self.r_pose
                pose_sigma_contrib = pose_grad * R_POSE_HNERV_STD
            else:
                pose_sigma_contrib = 0.0
        else:
            pose_sigma_contrib = 0.0
        # Seg sigma contribution (constant 100 coefficient):
        seg_sigma_contrib = 100.0 * d_seg_cuda / (self.r_seg ** 2) * R_SEG_HNERV_STD
        sigma = math.sqrt(pose_sigma_contrib ** 2 + seg_sigma_contrib ** 2)
        if not self._is_anchored:
            sigma = sigma * self.sigma_inflation
        # Sigma floor: never report a band tighter than the empirical gap std.
        return max(sigma, SCORE_GAP_HNERV_STD)


# -----------------------------------------------------------------------
# Module-level convenience wrappers
# -----------------------------------------------------------------------


def predict_cpu_from_cuda(
    cuda_score: float,
    *,
    archive_class: str = "unknown",
    d_pose_cuda: float | None = None,
    d_seg_cuda: float | None = None,
    archive_bytes: int | None = None,
) -> tuple[float, float]:
    """Predict CPU score band from CUDA score.

    Convenience wrapper around :class:`CudaCpuCalibration`. Returns the
    ``(score_lo, score_hi)`` band tuple.

    Args:
        cuda_score: CUDA score.
        archive_class: architecture class string (default ``"unknown"``).
        d_pose_cuda: optional CUDA pose distortion (unlocks decomposed mode).
        d_seg_cuda: optional CUDA seg distortion.
        archive_bytes: optional archive byte count.

    Returns:
        ``(score_lo, score_hi)`` 1σ band on the CPU axis.
    """
    cal = CudaCpuCalibration(architecture_class=archive_class)
    band = cal.predict_cpu_from_cuda(
        cuda_score,
        d_pose_cuda=d_pose_cuda,
        d_seg_cuda=d_seg_cuda,
        archive_bytes=archive_bytes,
    )
    return band.score_lo, band.score_hi


def predict_cuda_from_cpu(
    cpu_score: float,
    *,
    archive_class: str = "unknown",
    d_pose_cpu: float | None = None,
    d_seg_cpu: float | None = None,
    archive_bytes: int | None = None,
) -> tuple[float, float]:
    """Inverse direction. See :func:`predict_cpu_from_cuda`."""
    cal = CudaCpuCalibration(architecture_class=archive_class)
    band = cal.predict_cuda_from_cpu(
        cpu_score,
        d_pose_cpu=d_pose_cpu,
        d_seg_cpu=d_seg_cpu,
        archive_bytes=archive_bytes,
    )
    return band.score_lo, band.score_hi


def is_at_pose_floor(
    d_pose_cuda: float, *, archive_class: str = "unknown"
) -> bool:
    """See :meth:`CudaCpuCalibration.is_at_pose_floor`."""
    cal = CudaCpuCalibration(architecture_class=archive_class)
    return cal.is_at_pose_floor(d_pose_cuda)


def effective_pose_loss_for_cpu(
    d_pose_cuda: float, *, archive_class: str = "unknown"
) -> float:
    """See :meth:`CudaCpuCalibration.effective_pose_loss_for_cpu`."""
    cal = CudaCpuCalibration(architecture_class=archive_class)
    return cal.effective_pose_loss_for_cpu(d_pose_cuda)


# -----------------------------------------------------------------------
# CPU-axis pose floor / theoretical-floor helpers
# -----------------------------------------------------------------------


def cpu_pose_floor_score_contribution(
    d_pose_cpu_floor: float | None = None,
) -> float:
    """Return the CPU pose-axis floor contribution to the contest score.

    This is a descriptive helper for reporting the observed HNeRV CPU floor
    band. It is not a proof that lower pose is unobservable or removable from
    an achievable target.

    Args:
        d_pose_cpu_floor: optional override; default uses the empirical
            HNeRV cluster floor (3.4e-5 — PR106 anchor).

    Returns:
        The pose term (sqrt(10 * d_pose)) at the floor.
    """
    if d_pose_cpu_floor is None:
        # Empirical PR106 anchor d_pose_cpu = 3.4e-5 (after R_pose=5.04 rebase).
        d_pose_cpu_floor = 3.4e-5
    if d_pose_cpu_floor < 0.0:
        raise ValueError("d_pose_cpu_floor must be non-negative")
    return math.sqrt(10.0 * d_pose_cpu_floor)


# -----------------------------------------------------------------------
# Score axis tagging helpers
# -----------------------------------------------------------------------

ScoreAxis = Literal["cuda", "cpu", "unknown"]
"""Score axis label for tagging predicted bands."""


def tag_predicted_band(
    score: float,
    *,
    axis: ScoreAxis,
    calibration_quality: str = "hnerv-anchored",
) -> dict[str, object]:
    """Return a typed dict suitable for solver evidence rows.

    Tags a predicted score with the axis it was computed on, so the
    downstream selector can keep CUDA-axis and CPU-axis predictions
    separate (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA").
    """
    if axis not in {"cuda", "cpu", "unknown"}:
        raise ValueError(f"axis must be 'cuda'/'cpu'/'unknown'; got {axis!r}")
    return {
        "score": float(score),
        "axis": axis,
        "calibration_quality": calibration_quality,
    }


__all__ = [
    "CUDA_POSE_PRECISION_FLOOR",
    "KNOWN_ARCHITECTURE_CLASSES",
    "R_POSE_HNERV",
    "R_POSE_HNERV_STD",
    "R_SEG_HNERV",
    "R_SEG_HNERV_STD",
    "SCORE_GAP_HNERV_CONSTANT",
    "SCORE_GAP_HNERV_STD",
    "CalibrationBand",
    "CudaCpuCalibration",
    "ScoreAxis",
    "cpu_pose_floor_score_contribution",
    "effective_pose_loss_for_cpu",
    "is_at_pose_floor",
    "normalize_architecture_class",
    "predict_cpu_from_cuda",
    "predict_cuda_from_cpu",
    "tag_predicted_band",
]
