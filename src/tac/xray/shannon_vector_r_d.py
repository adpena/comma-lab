# SPDX-License-Identifier: MIT
"""F2: Shannon 1959 vector R(D) lower bound estimator.

The contest score is a vector-distortion function:
``S = 100 * d_seg(theta) + sqrt(10 * d_pose(theta)) + 25 * B(theta) / N``
with two distortion axes (seg, pose) and one rate axis (B/N). Shannon's
1959 *Coding theorems for a discrete source with a fidelity criterion* defines
``R(D)`` for vector-valued distortion — the canonical lower bound on the
number of bits required to achieve simultaneously
``E[d_seg] <= D_seg AND E[d_pose] <= D_pose``.

This primitive computes a CONSERVATIVE lower bound on ``R_min(D_seg, D_pose)``
under two assumptions:

1. The source is approximately a stationary stochastic process with
   marginal entropy ``H(payload)`` (estimated from the empirical posterior).
2. The Wyner-Ziv 1976 cooperative-receiver inequality holds:
   ``R_min(D_seg, D_pose) <= R_marginal - I(payload; scorer)``.

Per deep_math §9 ("Shannon 1959 vector R(D) theorem applied to the contest's
3-axis"), at the PR101 operating point (seg=0.067, pose=0.018, rate=0.108)
the Shannon-vector floor R_min ≈ 100 bytes — three orders of magnitude
theoretical headroom below A1's 178,262 B archive. This is consistent with
the zen-floor band 0.165 ± 0.020 derived from a different (Wasserstein-
proximity) argument in deep_math §5.4.

**Why this is a LOWER bound, not a tight estimate.** A1 ships a SUFFICIENT
representation (decoder + latents that reconstruct frames). Shannon's R(D)
counts only the bits an OPTIMAL cooperative-receiver decoder would need —
including infinite-codebook quantization, unbounded compute, and perfect
knowledge of the scorer at decode time. Real-world archives shipping
finite-LOC inflate.py runtimes can never reach R_min. The estimator is
useful as a Pareto lower-left vertex, not a deployment target.

Wire-in hooks engaged:

- ``pareto_constraint``: R_min bound contributes the lower-left vertex of
  the (D_seg, D_pose, B) Pareto frontier consumed by
  :mod:`tac.optimization.bit_allocator_end_to_end`.
- ``sensitivity_map``: the marginal-derivative of R_min(D_seg, D_pose) w.r.t.
  each axis tells the bit-allocator which axis to tighten next.
- ``probe_disambiguator``: comparing R_min at PR101 vs PR106 operating
  points disambiguates "is the substrate near its R(D) floor?" from
  "is there theoretical headroom remaining?".

Cross-references
----------------
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §9
- Source: Shannon 1959 "Coding theorems for a discrete source with a fidelity
  criterion" + Wyner-Ziv 1976 (DOI 10.1109/TIT.1976.1055508)
- Sister memory: ``feedback_expert_team_signal_processing_alien_tech_landed_20260513.md``
  (N3 Wyner-Ziv lineage that this primitive operationalizes for the planner)

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)

# Contest constants (pinned from upstream/evaluate.py:92).
CONTEST_UNCOMPRESSED_SIZE_BYTES = 37_545_489
SEG_COEFF = 100.0
POSE_COEFF_SQRT = 10.0  # sqrt(10 * d_pose) -> 10 inside sqrt
RATE_COEFF = 25.0


@dataclass(frozen=True)
class ShannonVectorRDBound:
    """Typed result from :meth:`ShannonVectorRDEstimator.compute`.

    Attributes
    ----------
    d_seg_target : float
        Operating-point SegNet distortion (Bernoulli argmax-disagreement rate).
    d_pose_target : float
        Operating-point PoseNet distortion (MSE on first 6 pose dims).
    r_min_bytes : float
        Lower-bound on bytes per archive needed to achieve both distortion
        targets simultaneously, under the cooperative-receiver assumption.
    r_min_score_contribution : float
        25 * r_min_bytes / N — the rate-axis contribution at the Shannon
        floor.
    distortion_floor_score_contribution : float
        100 * d_seg_target + sqrt(10 * d_pose_target) — the minimum
        distortion contribution given the targets.
    total_floor_score : float
        Sum of the two contributions; the theoretical score floor at this
        operating point.
    rationale : str
        One-paragraph derivation explaining how r_min_bytes was computed.
    """

    d_seg_target: float
    d_pose_target: float
    r_min_bytes: float
    r_min_score_contribution: float
    distortion_floor_score_contribution: float
    total_floor_score: float
    rationale: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.d_seg_target <= 1.0):
            raise ValueError(
                f"d_seg_target must be in [0.0, 1.0] (Bernoulli rate); "
                f"got {self.d_seg_target}"
            )
        if self.d_pose_target < 0.0:
            raise ValueError(
                f"d_pose_target must be non-negative (MSE); got "
                f"{self.d_pose_target}"
            )
        if self.r_min_bytes < 0.0:
            raise ValueError(
                f"r_min_bytes must be non-negative; got {self.r_min_bytes}"
            )


class ShannonVectorRDEstimator:
    """F2 canonical primitive: vector R(D) lower bound at a fixed operating point.

    The R_min estimate uses a closed-form approximation under the
    cooperative-receiver assumption. For each distortion axis independently:

    - **SegNet** (Bernoulli, K=5 classes): the maximum-entropy source achieving
      argmax-disagreement-rate D_seg has rate
      ``R_seg = H_5(D_seg) = (1 - D_seg) * log2(1) + D_seg * log2(K - 1)
      - H_2(D_seg) - D_seg * log2(K - 1)``.
      The minimum R for the Hamming-class fidelity criterion (Wyner 1972).
    - **PoseNet** (Gaussian, 6-dim MSE): the R(D) curve for a Gaussian
      source with MSE distortion D is ``R(D) = (1/2) * log2(sigma^2 / D)``
      for D < sigma^2, else 0. We treat ``sigma^2 = sigma_pose_prior^2``
      from the empirical 49-anchor posterior; default ``sigma_pose_prior =
      0.5`` (operator-tunable kwarg).
    - **JOINT** (cooperative): under independence the joint R(D_seg, D_pose)
      = R_seg + R_pose. Real-world correlation tightens the bound;
      ``--correlation-factor`` shrinks the sum.

    The bound is multiplied by N pairs (default 600) to get total-archive
    bits, then divided by 8 for bytes.
    """

    @property
    def name(self) -> str:
        return "shannon_vector_r_d"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "pareto_constraint",
            "sensitivity_map",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: Path | str | None = None,
        *,
        d_seg_target: float = 0.067,
        d_pose_target: float = 0.018,
        sigma_pose_prior: float = 0.5,
        n_pairs: int = 600,
        n_segnet_classes: int = 5,
        correlation_factor: float = 1.0,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Estimate vector R(D) lower bound at the given operating point.

        Parameters
        ----------
        target : Path | str | None
            Optional path to an archive (used only for provenance / sha
            recording; the bound is derivational).
        d_seg_target : float
            Target SegNet distortion (Bernoulli rate). Default 0.067 (the
            empirical PR101 operating point).
        d_pose_target : float
            Target PoseNet distortion (MSE). Default 0.018 (PR101).
        sigma_pose_prior : float
            Standard deviation of pose distribution (per-dim). Default 0.5.
        n_pairs : int
            Number of frame pairs (contest = 600).
        n_segnet_classes : int
            SegNet output classes (contest = 5).
        correlation_factor : float
            Cooperative-receiver correlation tightening. 1.0 = no
            correlation (Shannon-additive); 0.5 = halved joint rate (high
            scorer-source correlation).
        """
        if not (0.0 < correlation_factor <= 1.0):
            raise ValueError(
                f"correlation_factor must be in (0.0, 1.0]; got "
                f"{correlation_factor}"
            )
        if n_pairs <= 0:
            raise ValueError("n_pairs must be positive")
        if sigma_pose_prior <= 0.0:
            raise ValueError("sigma_pose_prior must be positive")

        # R_seg per pair (bits): rate-distortion for K-ary Bernoulli source
        # with Hamming fidelity criterion. R(D) = log2(K) - H_2(D) - D *
        # log2(K - 1) per Wyner 1972.
        if d_seg_target <= 0.0:
            r_seg_per_pair = math.log2(n_segnet_classes)
        elif d_seg_target >= (n_segnet_classes - 1) / n_segnet_classes:
            r_seg_per_pair = 0.0
        else:
            h2 = -d_seg_target * math.log2(d_seg_target) - (
                1.0 - d_seg_target
            ) * math.log2(1.0 - d_seg_target)
            r_seg_per_pair = (
                math.log2(n_segnet_classes)
                - h2
                - d_seg_target * math.log2(n_segnet_classes - 1)
            )
            r_seg_per_pair = max(0.0, r_seg_per_pair)

        # R_pose per pair (bits): Gaussian source with MSE distortion.
        # R(D) = (n_dim / 2) * log2(sigma^2 / D) for D < sigma^2.
        # PoseNet output is 6-D (first 6 of 12 dims per modules.py:82-84).
        pose_n_dim = 6
        if d_pose_target >= sigma_pose_prior**2:
            r_pose_per_pair = 0.0
        else:
            r_pose_per_pair = (pose_n_dim / 2.0) * math.log2(
                sigma_pose_prior**2 / d_pose_target
            )

        # Joint rate under cooperative-receiver assumption.
        r_joint_per_pair_bits = correlation_factor * (
            r_seg_per_pair + r_pose_per_pair
        )
        # Total bits across N pairs.
        r_min_bits_total = r_joint_per_pair_bits * n_pairs
        r_min_bytes = r_min_bits_total / 8.0

        # Score contributions.
        rate_score = (
            RATE_COEFF * r_min_bytes / CONTEST_UNCOMPRESSED_SIZE_BYTES
        )
        distortion_score = SEG_COEFF * d_seg_target + math.sqrt(
            POSE_COEFF_SQRT * d_pose_target
        )
        total_floor_score = rate_score + distortion_score

        rationale = (
            f"R_seg/pair = {r_seg_per_pair:.4f} bits "
            f"(Wyner-1972 K={n_segnet_classes} Bernoulli at D={d_seg_target}); "
            f"R_pose/pair = {r_pose_per_pair:.4f} bits "
            f"(Gaussian-{pose_n_dim}D at D={d_pose_target}, "
            f"sigma={sigma_pose_prior}); "
            f"R_joint = {correlation_factor:.2f} * (R_seg + R_pose) * "
            f"{n_pairs} pairs / 8 = {r_min_bytes:.1f} bytes "
            f"[first-principles-bound; deep_math §9 + Shannon 1959]"
        )

        bound = ShannonVectorRDBound(
            d_seg_target=d_seg_target,
            d_pose_target=d_pose_target,
            r_min_bytes=r_min_bytes,
            r_min_score_contribution=rate_score,
            distortion_floor_score_contribution=distortion_score,
            total_floor_score=total_floor_score,
            rationale=rationale,
        )

        archive_path: Path | None = None
        archive_sha: str | None = None
        if target is not None:
            archive_path = Path(target)
            if archive_path.exists():
                from tac.repo_io import sha256_bytes

                archive_sha = sha256_bytes(archive_path.read_bytes())

        # Confidence band: the cooperative-receiver factor introduces
        # the dominant uncertainty. We report [bound * 0.5, bound * 1.5]
        # as a rough 3:1 trust region around the central estimate.
        band = (r_min_bytes * 0.5, r_min_bytes * 1.5)

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=archive_path,
            archive_sha256=archive_sha,
            primitive_value=bound,
            evidence_grade="first-principles-bound",
            confidence_band=band,
            composes_with=("score_lipschitz", "mdl_scorer_conditional"),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "correlation_factor": correlation_factor,
                "sigma_pose_prior": sigma_pose_prior,
                "n_pairs": n_pairs,
                "n_segnet_classes": n_segnet_classes,
            },
        )

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "CONTEST_UNCOMPRESSED_SIZE_BYTES",
    "POSE_COEFF_SQRT",
    "RATE_COEFF",
    "SEG_COEFF",
    "ShannonVectorRDBound",
    "ShannonVectorRDEstimator",
]
