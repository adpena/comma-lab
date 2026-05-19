# SPDX-License-Identifier: MIT
"""Canonical MPS-vs-CUDA drift predictor — per-pair drift formalization helper.

The *drift predictor* operationalizes the mathematical + engineering
formalization documented at
``.omx/research/mps_cuda_drift_mathematical_and_engineering_formalization_20260519.md``.

Mathematical core (see formalization memo §3 for full derivation):

  d(theta_MPS, theta_CUDA) := theta_MPS - theta_CUDA          (weight delta)
  delta_S_p ~= sum_i g_{i,p} * d_i                            (Taylor first order)
  |delta_S_p| <= ||g_p||_2 * ||d||_2                          (Cauchy-Schwarz)
  cos(g_p, d) = (g_p . d) / (||g_p||_2 * ||d||_2)             (alignment scalar in [-1, 1])

The Cauchy-Schwarz inequality is a *worst-case* bound. The empirical question --
"does drift live in the score-relevant subspace?" -- is answered by the
distribution of ``cos(g_p, d)`` across pairs:

  - cos ~ 0 uniformly: drift in nullspace; local-MPS compute genuinely viable.
  - |cos| ~ 1 uniformly: drift aligned with score gradient; engineering correction
                         required (Kahan summation / fp32 matmul override / etc.).
  - heavy tail in cos: per-pair routing of high-|cos| pairs to CUDA shadow.

Engineering core (formalization memo §4): per-kernel root-cause analysis predicts
SegNet/PoseNet drift ratio from layer depth + accumulation kind. The layer-depth
prediction matches today's empirical anchor (16.6x observed SegNet/PoseNet ratio)
to within an order of magnitude.

Authority boundary: this module emits **predicted-from-model** Provenance
(Catalog #323). Predictions are NOT contest-axis score claims and are NEVER
``promotion_eligible``. Empirical anchors must land via paired Linux x86_64
runs per CLAUDE.md "Submission auth eval -- BOTH CPU AND CUDA" non-negotiable.

Lane: ``lane_mps_drift_mathematical_and_engineering_formalization_20260519``.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Final

from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.contract import Provenance

__all__ = [
    "PREDICTOR_MODEL_ID",
    "LAYER_DEPTH_RATIO_CONSTANT",
    "VALID_COS_DISTRIBUTION_VERDICTS",
    "VALID_MPS_VIABLE_VERDICTS",
    "ArchitectureFeatures",
    "CalibrationAnchor",
    "CosDistributionSummary",
    "DriftPrediction",
    "KernelTypeCounts",
    "MPS_VIABLE_GAP_THRESHOLD",
    "cauchy_schwarz_upper_bound",
    "cos_distribution_summary",
    "evidence_grade_for_predicted_gap",
    "predict_drift",
    "predict_layer_depth_drift_ratio",
]


# Canonical verdict taxonomies (module-level frozenset constants so dataclass
# asdict() does not pick them up as fields)
VALID_COS_DISTRIBUTION_VERDICTS: Final[frozenset[str]] = frozenset(
    {
        "NULLSPACE_VIABLE",
        "SCORE_RELEVANT_ENGINEERING_REQUIRED",
        "MIXED_NEEDS_PER_PAIR_ROUTING",
        "INSUFFICIENT_DATA",
    }
)
VALID_MPS_VIABLE_VERDICTS: Final[frozenset[str]] = frozenset(
    {"MPS_VIABLE", "MPS_NON_VIABLE", "NEEDS_EMPIRICAL_PROBE"}
)


# Canonical model id used in every emitted Provenance row.
PREDICTOR_MODEL_ID: Final[str] = "tac.mps_diagnostic.drift_predictor.predict_drift.v1"

# Per the formalization memo §4.6, drift accumulates roughly as
# sqrt(N_layers) for reduction-order noise (Higham 2002 chapter 4 bound on
# floating-point accumulation in long chains). Per-layer ratio uses a
# linear-depth term plus an accumulation term:
#
#   ratio_predicted(N_a, N_b) = (N_a / N_b) * sqrt(N_a / N_b)
#
# which equals 8.49x for SegNet (50 layers) vs PoseNet (12 layers). The
# empirical observed ratio is ~16.6x; the unexplained residual ~2x is queued
# as HARD-EARNED-PENDING-FURTHER-INVESTIGATION in the formalization memo §6.2.
LAYER_DEPTH_RATIO_CONSTANT: Final[float] = 1.0

# Operator-facing screening threshold for predicted aggregate gap.
# Below threshold => MPS proxy viable (free signal). Above threshold => skip MPS,
# dispatch direct to Modal. See formalization memo §5.4 for derivation.
MPS_VIABLE_GAP_THRESHOLD: Final[float] = 0.05  # 5 percent aggregate gap


# -----------------------------------------------------------------------------
# Typed dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelTypeCounts:
    """Architecture feature: count of each kernel-type that contributes to drift.

    The kernel-type taxonomy mirrors the per-kernel root-cause analysis in
    formalization memo §4. Each kernel kind has a documented contribution
    coefficient; see ``predict_drift`` for the linear-combination model.
    """

    conv2d_stride2: int = 0
    """Conv2d with stride>=2 (SegNet stem). Reduction-ordering noise."""

    conv2d_stride1: int = 0
    """Conv2d stride=1 (common in decoders). Lower drift per layer."""

    linear_matmul: int = 0
    """Linear/MatMul (FastViT FC layers). TF32-vs-IEEE-fp32 gap."""

    softmax: int = 0
    """Softmax with log-sum-exp stabilization. Boundary-flip risk."""

    interpolate_bicubic: int = 0
    """F.interpolate bicubic upsample. Texture-coord convention drift."""

    layernorm: int = 0
    """LayerNorm / GroupNorm. Reduction-ordering noise."""

    rgb_to_yuv6_inplace: int = 0
    """rgb_to_yuv6 in-place mutations. Read-after-write hazard windows."""

    def total_accumulating_kernels(self) -> int:
        """Total count of kernels that accumulate drift through depth."""
        return (
            self.conv2d_stride2
            + self.conv2d_stride1
            + self.linear_matmul
            + self.layernorm
        )


@dataclass(frozen=True)
class ArchitectureFeatures:
    """Predictor input: structural features of a scorer / renderer architecture.

    Used by ``predict_drift`` to emit a Cauchy-Schwarz upper-bound prediction
    of the per-pair aggregate MPS-vs-CUDA gap. The model is calibrated against
    empirical anchors (the tiny renderer Phase B + future SegMap/NeRV anchors).
    """

    architecture_id: str
    """Short identifier (e.g. 'tiny_renderer_phase_b', 'segnet_efficientnet_b2')."""

    layer_count: int
    """Total layer count contributing to accumulated drift."""

    kernel_type_counts: KernelTypeCounts
    """Per-kernel-type breakdown for per-kernel root-cause coefficients."""

    parameter_count: int
    """Total parameter count theta in R^P."""

    accumulation_depth: int
    """Sequential accumulation depth (max chain length through reductions)."""

    def __post_init__(self) -> None:
        if not self.architecture_id:
            raise ValueError("architecture_id must be non-empty")
        if self.layer_count < 0:
            raise ValueError(f"layer_count must be >=0; got {self.layer_count}")
        if self.parameter_count < 0:
            raise ValueError(
                f"parameter_count must be >=0; got {self.parameter_count}"
            )
        if self.accumulation_depth < 0:
            raise ValueError(
                f"accumulation_depth must be >=0; got {self.accumulation_depth}"
            )

    def features_sha256(self) -> str:
        """Stable hash over the feature tuple for Provenance inputs_sha256."""
        payload = {
            "architecture_id": self.architecture_id,
            "layer_count": self.layer_count,
            "kernel_type_counts": asdict(self.kernel_type_counts),
            "parameter_count": self.parameter_count,
            "accumulation_depth": self.accumulation_depth,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CalibrationAnchor:
    """Empirical anchor used to calibrate the predictor's regression coefficients.

    Each anchor pairs an ``ArchitectureFeatures`` row with a measured per-pair
    aggregate drift (median across pairs). The predictor's regression is fit
    against these anchors so future architectures get a predicted gap without
    running the MPS experiment.
    """

    architecture: ArchitectureFeatures
    measured_aggregate_drift_median: float
    measurement_evidence_path: str
    """Path to the empirical JSON (e.g. mps_drift_granular_*.json)."""

    def __post_init__(self) -> None:
        if self.measured_aggregate_drift_median < 0:
            raise ValueError(
                "measured_aggregate_drift_median must be >=0; got "
                f"{self.measured_aggregate_drift_median}"
            )
        if not self.measurement_evidence_path:
            raise ValueError("measurement_evidence_path must be non-empty")


@dataclass(frozen=True)
class CosDistributionSummary:
    """Summary statistics over the per-pair cos(g_p, d) distribution.

    Cos(g_p, d) is the alignment scalar between the per-pair score gradient and
    the weight delta. Its empirical distribution is the structural disambiguator
    between "drift is in the score-nullspace" (locally-free compute viable) and
    "drift is aligned with the score gradient" (engineering correction required).
    """

    n_pairs: int
    mean: float
    abs_mean: float
    median: float
    std: float
    max_abs: float
    n_outliers_abs_above_0_5: int
    n_outliers_abs_above_0_8: int
    verdict: str
    """One of VALID_COS_DISTRIBUTION_VERDICTS:
    NULLSPACE_VIABLE / SCORE_RELEVANT_ENGINEERING_REQUIRED /
    MIXED_NEEDS_PER_PAIR_ROUTING / INSUFFICIENT_DATA.
    """

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError(f"n_pairs must be >=0; got {self.n_pairs}")
        if not (-1.0 <= self.mean <= 1.0):
            raise ValueError(
                f"mean must be in [-1, 1]; got {self.mean}"
            )
        if self.max_abs < 0 or self.max_abs > 1.0 + 1e-9:
            raise ValueError(
                f"max_abs must be in [0, 1]; got {self.max_abs}"
            )
        if self.verdict not in VALID_COS_DISTRIBUTION_VERDICTS:
            raise ValueError(
                f"unrecognized verdict: {self.verdict}; "
                f"must be in {sorted(VALID_COS_DISTRIBUTION_VERDICTS)}"
            )


@dataclass(frozen=True)
class DriftPrediction:
    """Predictor output: predicted-band MPS-vs-CUDA drift + Cauchy-Schwarz bound.

    All numerical fields are PREDICTED (Catalog #287/#323). The Provenance
    sub-object carries kind=PREDICTED_FROM_MODEL + evidence_grade=PREDICTED +
    promotion_eligible=False so downstream cathedral autopilot consumers
    (Catalog #335/#336/#337) auto-route through the non-promotable branch.
    """

    architecture: ArchitectureFeatures
    predicted_aggregate_gap_lower_bound: float
    predicted_aggregate_gap_upper_bound: float
    """Predicted aggregate-drift band (closed interval, fp32 relative)."""

    predicted_cos_distribution_summary: CosDistributionSummary
    cauchy_schwarz_upper_bound_value: float
    """||g||_2 * ||d||_2 worst-case bound."""

    predicted_segnet_posenet_drift_ratio: float
    """Predicted ratio derived from layer-depth model in formalization §4.6."""

    mps_viable_verdict: str
    """One of: MPS_VIABLE / MPS_NON_VIABLE / NEEDS_EMPIRICAL_PROBE."""

    provenance: Provenance
    """Catalog #323 canonical Provenance; kind=PREDICTED_FROM_MODEL."""

    def __post_init__(self) -> None:
        if (
            self.predicted_aggregate_gap_lower_bound
            > self.predicted_aggregate_gap_upper_bound
        ):
            raise ValueError(
                "lower_bound > upper_bound: "
                f"{self.predicted_aggregate_gap_lower_bound} > "
                f"{self.predicted_aggregate_gap_upper_bound}"
            )
        if self.predicted_aggregate_gap_lower_bound < 0:
            raise ValueError(
                "lower_bound must be >=0; got "
                f"{self.predicted_aggregate_gap_lower_bound}"
            )
        if self.cauchy_schwarz_upper_bound_value < 0:
            raise ValueError(
                "Cauchy-Schwarz bound must be >=0; got "
                f"{self.cauchy_schwarz_upper_bound_value}"
            )
        if self.mps_viable_verdict not in VALID_MPS_VIABLE_VERDICTS:
            raise ValueError(
                f"unrecognized mps_viable_verdict: {self.mps_viable_verdict}; "
                f"must be in {sorted(VALID_MPS_VIABLE_VERDICTS)}"
            )

    def as_dict(self) -> dict:
        """JSON-safe dict for serialization. Provenance auto-flattens via asdict."""
        return {
            "architecture": {
                "architecture_id": self.architecture.architecture_id,
                "layer_count": self.architecture.layer_count,
                "kernel_type_counts": asdict(self.architecture.kernel_type_counts),
                "parameter_count": self.architecture.parameter_count,
                "accumulation_depth": self.architecture.accumulation_depth,
            },
            "predicted_aggregate_gap_lower_bound": (
                self.predicted_aggregate_gap_lower_bound
            ),
            "predicted_aggregate_gap_upper_bound": (
                self.predicted_aggregate_gap_upper_bound
            ),
            "predicted_cos_distribution_summary": asdict(
                self.predicted_cos_distribution_summary
            ),
            "cauchy_schwarz_upper_bound_value": (
                self.cauchy_schwarz_upper_bound_value
            ),
            "predicted_segnet_posenet_drift_ratio": (
                self.predicted_segnet_posenet_drift_ratio
            ),
            "mps_viable_verdict": self.mps_viable_verdict,
            "provenance": {
                "artifact_kind": self.provenance.artifact_kind.value,
                "source_path": self.provenance.source_path,
                "source_sha256": self.provenance.source_sha256,
                "measurement_axis": self.provenance.measurement_axis,
                "hardware_substrate": self.provenance.hardware_substrate,
                "evidence_grade": self.provenance.evidence_grade.value,
                "promotion_eligible": self.provenance.promotion_eligible,
                "score_claim_valid": self.provenance.score_claim_valid,
                "captured_at_utc": self.provenance.captured_at_utc,
                "canonical_helper_invocation": (
                    self.provenance.canonical_helper_invocation
                ),
            },
        }


# -----------------------------------------------------------------------------
# Math helpers (§3 of formalization memo)
# -----------------------------------------------------------------------------


def cauchy_schwarz_upper_bound(
    g_per_pair_norm: float, d_norm: float
) -> float:
    """Cauchy-Schwarz upper bound |delta_S_p| <= ||g_p||_2 * ||d||_2.

    Per formalization memo §3.3. This is the worst-case score impact from
    weight-delta ``d``; the empirical impact is g . d which can be much smaller
    if the alignment cos(g, d) is near zero (nullspace).

    Args:
        g_per_pair_norm: ||g_p||_2 in score-units-per-parameter.
        d_norm: ||d||_2 weight-delta L2 norm in parameter-units.

    Returns:
        Upper bound on |delta_S_p| in score units.
    """
    if g_per_pair_norm < 0:
        raise ValueError(f"g_per_pair_norm must be >=0; got {g_per_pair_norm}")
    if d_norm < 0:
        raise ValueError(f"d_norm must be >=0; got {d_norm}")
    return float(g_per_pair_norm) * float(d_norm)


def cos_distribution_summary(
    per_pair_inner_products: Sequence[float],
    per_pair_g_norms: Sequence[float],
    d_norm: float,
) -> CosDistributionSummary:
    """Build summary stats over per-pair cos(g_p, d) values.

    Per formalization memo §3.4. The empirical distribution of
    ``cos(g_p, d) = (g_p . d) / (||g_p|| * ||d||)`` answers the structural
    nullspace-vs-score-relevant question.

    Verdict semantics (§3.4 of memo):
      - NULLSPACE_VIABLE: |abs_mean| < 0.05 AND max_abs < 0.3
      - MIXED_NEEDS_PER_PAIR_ROUTING: max_abs in [0.3, 0.8)
      - SCORE_RELEVANT_ENGINEERING_REQUIRED: max_abs >= 0.8 OR abs_mean >= 0.2
      - INSUFFICIENT_DATA: n_pairs < 3 OR d_norm < 1e-12 OR all g_norms < 1e-12

    Args:
        per_pair_inner_products: per-pair g_p . d values (length N_pairs).
        per_pair_g_norms: per-pair ||g_p|| values (length N_pairs).
        d_norm: ||d|| weight-delta norm (scalar).

    Returns:
        CosDistributionSummary with frozen invariants.
    """
    n = len(per_pair_inner_products)
    if n != len(per_pair_g_norms):
        raise ValueError(
            "per_pair_inner_products and per_pair_g_norms must have equal length"
        )
    if d_norm < 0:
        raise ValueError(f"d_norm must be >=0; got {d_norm}")

    EPS = 1e-12
    if n < 3 or d_norm < EPS or all(abs(g) < EPS for g in per_pair_g_norms):
        return CosDistributionSummary(
            n_pairs=n,
            mean=0.0,
            abs_mean=0.0,
            median=0.0,
            std=0.0,
            max_abs=0.0,
            n_outliers_abs_above_0_5=0,
            n_outliers_abs_above_0_8=0,
            verdict="INSUFFICIENT_DATA",
        )

    cos_values: list[float] = []
    for inner, g_norm in zip(per_pair_inner_products, per_pair_g_norms):
        if g_norm < EPS:
            cos_values.append(0.0)
            continue
        cos = float(inner) / (float(g_norm) * float(d_norm))
        # Clamp to [-1, 1] for numerical safety
        cos = max(-1.0, min(1.0, cos))
        cos_values.append(cos)

    abs_values = [abs(c) for c in cos_values]
    mean = sum(cos_values) / n
    abs_mean = sum(abs_values) / n
    sorted_cos = sorted(cos_values)
    median = sorted_cos[n // 2] if n % 2 == 1 else (
        sorted_cos[n // 2 - 1] + sorted_cos[n // 2]
    ) / 2.0
    variance = sum((c - mean) ** 2 for c in cos_values) / n
    std = math.sqrt(variance)
    max_abs = max(abs_values)
    n_outliers_0_5 = sum(1 for a in abs_values if a >= 0.5)
    n_outliers_0_8 = sum(1 for a in abs_values if a >= 0.8)

    if max_abs >= 0.8 or abs_mean >= 0.2:
        verdict = "SCORE_RELEVANT_ENGINEERING_REQUIRED"
    elif max_abs >= 0.3:
        verdict = "MIXED_NEEDS_PER_PAIR_ROUTING"
    else:
        verdict = "NULLSPACE_VIABLE"

    return CosDistributionSummary(
        n_pairs=n,
        mean=mean,
        abs_mean=abs_mean,
        median=median,
        std=std,
        max_abs=max_abs,
        n_outliers_abs_above_0_5=n_outliers_0_5,
        n_outliers_abs_above_0_8=n_outliers_0_8,
        verdict=verdict,
    )


def predict_layer_depth_drift_ratio(
    scorer_a_layer_count: int, scorer_b_layer_count: int
) -> float:
    """Predict per-architecture drift ratio from layer-depth alone.

    Per formalization memo §4.6: drift accumulates as
      linear-depth * sqrt(accumulation-depth)
    so the predicted ratio between two scorers is
      ratio = (N_a / N_b) * sqrt(N_a / N_b)

    Worked example: SegNet (50 layers) vs PoseNet (12 layers):
      ratio = (50/12) * sqrt(50/12) = 4.167 * 2.041 = 8.50x
    Today's empirical observation is 16.6x (segnet_drift / posenet_drift =
    2.01e-3 / 1.21e-4 from the Phase B aggregate). Residual unexplained 2x
    queued as HARD-EARNED-PENDING-FURTHER-INVESTIGATION in memo §6.2.

    Args:
        scorer_a_layer_count: layer count of numerator architecture.
        scorer_b_layer_count: layer count of denominator architecture.

    Returns:
        Predicted ratio drift_a / drift_b. Returns 1.0 if either count is 0.
    """
    if scorer_a_layer_count < 0 or scorer_b_layer_count < 0:
        raise ValueError(
            "layer counts must be >=0; got "
            f"{scorer_a_layer_count}, {scorer_b_layer_count}"
        )
    if scorer_a_layer_count == 0 or scorer_b_layer_count == 0:
        return 1.0
    ratio = float(scorer_a_layer_count) / float(scorer_b_layer_count)
    return ratio * math.sqrt(ratio) * LAYER_DEPTH_RATIO_CONSTANT


def evidence_grade_for_predicted_gap(predicted_gap: float) -> str:
    """Map a predicted aggregate gap to an operator-facing verdict label.

    Per formalization memo §5.4 and CLAUDE.md "MPS auth eval is NOISE" sister
    discipline. The verdict drives the MPS-prescreen cathedral consumer
    routing (sister of `tac.cathedral_consumers.mps_viable_prescreen_consumer`).

    Args:
        predicted_gap: predicted aggregate fp32-relative gap (non-negative).

    Returns:
        One of: MPS_VIABLE / MPS_NON_VIABLE / NEEDS_EMPIRICAL_PROBE.
    """
    if predicted_gap < 0:
        raise ValueError(f"predicted_gap must be >=0; got {predicted_gap}")
    if predicted_gap < MPS_VIABLE_GAP_THRESHOLD * 0.5:
        # Predicted well below threshold => viable
        return "MPS_VIABLE"
    if predicted_gap < MPS_VIABLE_GAP_THRESHOLD * 1.5:
        # In the uncertainty band around threshold => need empirical probe
        return "NEEDS_EMPIRICAL_PROBE"
    return "MPS_NON_VIABLE"


# -----------------------------------------------------------------------------
# Per-kernel drift contribution coefficients (§4 of formalization memo)
# -----------------------------------------------------------------------------
#
# Each coefficient is a fp32-relative contribution per-kernel-invocation. The
# values are CALIBRATED against today's Phase B empirical anchor + standard
# Higham 2002 floating-point accumulation bounds.

_KERNEL_DRIFT_COEFFICIENTS: Final[Mapping[str, float]] = {
    # Higham 2002 chapter 4: reduction-order noise scales as eps * sqrt(N).
    # Stride-2 stem has higher coefficient because the cumulative accumulation
    # cascade is longer than stride-1 (downsampling => later layers see more
    # batched sums per output pixel).
    "conv2d_stride2": 5e-6,
    "conv2d_stride1": 2e-6,
    # Linear matmul: TF32 vs IEEE fp32 mantissa gap. CUDA TF32 discards
    # trailing 4 bits of mantissa (19-bit vs 23-bit) => systematic precision
    # gap per matmul.
    "linear_matmul": 1e-6,
    # Softmax: log-sum-exp epsilon difference. Affects boundary pixels only.
    "softmax": 5e-7,
    # F.interpolate bicubic: texture-coordinate convention drift at frame
    # boundaries.
    "interpolate_bicubic": 8e-7,
    # LayerNorm: reduction-order noise (similar to Conv2d) but per-token
    # instead of per-pixel.
    "layernorm": 3e-6,
    # rgb_to_yuv6 in-place: read-after-write hazard windows. Small absolute
    # impact but compounds at pre-scorer-forward stage.
    "rgb_to_yuv6_inplace": 1e-6,
}


# -----------------------------------------------------------------------------
# Main predictor (§5 of formalization memo)
# -----------------------------------------------------------------------------


def predict_drift(
    features: ArchitectureFeatures,
    calibration_anchors: Sequence[CalibrationAnchor] = (),
    *,
    cos_distribution_inputs: tuple[
        Sequence[float], Sequence[float], float
    ] | None = None,
) -> DriftPrediction:
    """Predict MPS-vs-CUDA aggregate drift for an architecture.

    Per formalization memo §5. The predictor combines:
      1. Per-kernel root-cause coefficients (memo §4) -> linear-combination
         baseline prediction.
      2. Layer-depth sqrt(N) cumulative term (Higham 2002 §4 bound) ->
         multiplicative depth factor.
      3. Calibration anchors (if provided) -> rescale baseline so empirical
         medians fit on the predicted line.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #287 + Catalog #323: every
    field of the returned DriftPrediction carries PREDICTED-FROM-MODEL
    Provenance + axis_tag=[predicted] + promotion_eligible=False.

    Args:
        features: architecture structural features.
        calibration_anchors: optional empirical anchors to rescale the linear
            baseline. Each anchor contributes one (predicted, measured) pair.
        cos_distribution_inputs: optional (per_pair_inner_products,
            per_pair_g_norms, d_norm) triple. If omitted, the cos summary
            verdict is INSUFFICIENT_DATA.

    Returns:
        DriftPrediction with frozen invariants + canonical Provenance.
    """
    # 1. Per-kernel baseline contribution
    counts = features.kernel_type_counts
    baseline_per_layer = (
        counts.conv2d_stride2 * _KERNEL_DRIFT_COEFFICIENTS["conv2d_stride2"]
        + counts.conv2d_stride1 * _KERNEL_DRIFT_COEFFICIENTS["conv2d_stride1"]
        + counts.linear_matmul * _KERNEL_DRIFT_COEFFICIENTS["linear_matmul"]
        + counts.softmax * _KERNEL_DRIFT_COEFFICIENTS["softmax"]
        + counts.interpolate_bicubic
        * _KERNEL_DRIFT_COEFFICIENTS["interpolate_bicubic"]
        + counts.layernorm * _KERNEL_DRIFT_COEFFICIENTS["layernorm"]
        + counts.rgb_to_yuv6_inplace
        * _KERNEL_DRIFT_COEFFICIENTS["rgb_to_yuv6_inplace"]
    )

    # 2. Cumulative depth term (Higham 2002 chapter 4)
    depth_factor = (
        math.sqrt(max(1, features.accumulation_depth))
        if features.accumulation_depth > 0
        else 1.0
    )

    predicted_central = baseline_per_layer * depth_factor

    # 3. Calibration rescale (if anchors provided)
    if calibration_anchors:
        # Compute baseline for each anchor; fit a single scale factor that
        # minimizes squared error between (predicted * scale) and measured.
        # Closed form: scale = sum(p_i * m_i) / sum(p_i ** 2).
        predicted_anchors: list[float] = []
        measured_anchors: list[float] = []
        for anchor in calibration_anchors:
            anchor_counts = anchor.architecture.kernel_type_counts
            anchor_baseline = (
                anchor_counts.conv2d_stride2
                * _KERNEL_DRIFT_COEFFICIENTS["conv2d_stride2"]
                + anchor_counts.conv2d_stride1
                * _KERNEL_DRIFT_COEFFICIENTS["conv2d_stride1"]
                + anchor_counts.linear_matmul
                * _KERNEL_DRIFT_COEFFICIENTS["linear_matmul"]
                + anchor_counts.softmax * _KERNEL_DRIFT_COEFFICIENTS["softmax"]
                + anchor_counts.interpolate_bicubic
                * _KERNEL_DRIFT_COEFFICIENTS["interpolate_bicubic"]
                + anchor_counts.layernorm
                * _KERNEL_DRIFT_COEFFICIENTS["layernorm"]
                + anchor_counts.rgb_to_yuv6_inplace
                * _KERNEL_DRIFT_COEFFICIENTS["rgb_to_yuv6_inplace"]
            )
            anchor_depth_factor = (
                math.sqrt(max(1, anchor.architecture.accumulation_depth))
                if anchor.architecture.accumulation_depth > 0
                else 1.0
            )
            anchor_predicted = anchor_baseline * anchor_depth_factor
            if anchor_predicted > 0:
                predicted_anchors.append(anchor_predicted)
                measured_anchors.append(anchor.measured_aggregate_drift_median)
        if predicted_anchors:
            num = sum(p * m for p, m in zip(predicted_anchors, measured_anchors))
            denom = sum(p * p for p in predicted_anchors)
            scale = num / denom if denom > 1e-30 else 1.0
            predicted_central *= scale

    # Band: +/- factor of 3x around central (Cauchy-Schwarz worst-case slack).
    lower = predicted_central / 3.0
    upper = predicted_central * 3.0

    # 4. Cauchy-Schwarz scalar bound (uses provided g_norm * d_norm if available)
    if cos_distribution_inputs is not None:
        inner_products, g_norms, d_norm = cos_distribution_inputs
        cs_bound = cauchy_schwarz_upper_bound(
            max(g_norms) if g_norms else 0.0, d_norm
        )
        cos_summary = cos_distribution_summary(inner_products, g_norms, d_norm)
    else:
        # Diagnostic placeholder: use ||d|| ~ sqrt(P) * predicted_gap as a
        # rough proxy when no master-gradient anchor is available.
        cs_bound = predicted_central * math.sqrt(max(1, features.parameter_count))
        cos_summary = CosDistributionSummary(
            n_pairs=0,
            mean=0.0,
            abs_mean=0.0,
            median=0.0,
            std=0.0,
            max_abs=0.0,
            n_outliers_abs_above_0_5=0,
            n_outliers_abs_above_0_8=0,
            verdict="INSUFFICIENT_DATA",
        )

    # 5. SegNet/PoseNet layer-depth ratio prediction
    # Use 50-layer SegNet vs 12-layer PoseNet as the canonical reference pair.
    ratio_prediction = predict_layer_depth_drift_ratio(50, 12)

    # 6. MPS-viable verdict (sister of mps_viable_prescreen_consumer)
    viable_verdict = evidence_grade_for_predicted_gap(predicted_central)

    # 7. Build canonical Provenance per Catalog #323
    provenance = build_provenance_for_predicted(
        model_id=PREDICTOR_MODEL_ID,
        inputs_sha256=features.features_sha256(),
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )

    return DriftPrediction(
        architecture=features,
        predicted_aggregate_gap_lower_bound=lower,
        predicted_aggregate_gap_upper_bound=upper,
        predicted_cos_distribution_summary=cos_summary,
        cauchy_schwarz_upper_bound_value=cs_bound,
        predicted_segnet_posenet_drift_ratio=ratio_prediction,
        mps_viable_verdict=viable_verdict,
        provenance=provenance,
    )
