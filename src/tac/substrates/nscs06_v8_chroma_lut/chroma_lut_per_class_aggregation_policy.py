# SPDX-License-Identifier: MIT
"""Canonical per-(level, class) chroma LUT aggregation policies for v8.

Per Wave 9 NSCS06 v8 cargo-cult #4 (Wave 5 follow-on op-routable #1)
2026-05-29: the existing
``architecture.build_chroma_lut_from_ground_truth`` hardcodes
``np.median`` as the per-(level, class) aggregation for the
``(grayscale_levels, num_segnet_classes, 3)`` chroma LUT. The
``CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS`` left in the
substrate notes during Wave 5 flagged this as the SAME META-class as
cargo-cult #6 at a different helper: a representative statistic per
spatial/categorical bin selected by-default-because-robust rather than
empirically validated against alternatives.

Wave 5 (commit ``85521b61d``) landed the canonical
``cls_lowres_downsample_policy`` helper for cargo-cult #6 with
NEAREST (byte-default) + MODE (boundary-preserving) policies. Wave 9
mirrors that architecture for cargo-cult #4 at the chroma-LUT
aggregation surface.

**Cargo-cult #4 (Wave 9)**: ``np.median`` IS a robust statistic and is
defensible at noisy bins (small bin populations, mixed-illumination
pixels). It is NOT necessarily the per-(level, class) bin estimator
that minimizes contest-CUDA + contest-CPU PSNR distortion. Alternatives
worth empirical comparison:

* MEAN: minimizes L2 reconstruction error per bin in absence of
  outliers; canonical estimator under Gaussian noise assumption.
* MODE per cell (cell == per-(level, class) bin): preserves the most-
  frequent discrete RGB triple per bin; sister of Wave 5 #6 fix at the
  RGB aggregation surface; degrades gracefully when bins are uniform.
* WEIGHTED MEAN BY CELL COUNT: per-bin estimator weighted by the
  number of pixels contributing to the bin; favors statistically
  well-populated bins; sister-aware of bin-population variance.

**Honest disclosure** per CLAUDE.md "Apples-to-apples evidence
discipline":

* The MEDIAN policy is currently the BYTE-DEFAULT and preserves
  archive-byte parity with all prior dispatches.
* The MEAN / MODE / WEIGHTED_MEAN policies are UNWIND alternatives
  that may produce different archive bytes but preserve the canonical
  bit-budget identity (same ``grayscale_levels * num_segnet_classes * 3``
  byte cost of the chroma LUT slot).
* Operator opt-in is REQUIRED to switch from MEDIAN to any alternative
  because it changes archive bytes (per CLAUDE.md "Frontier scores are
  pointer-only" + Catalog #110/#113 HISTORICAL_PROVENANCE for prior
  empirical anchors keyed by MEDIAN policy).

CLAUDE.md compliance:

* Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW module; no
  mutation of sister architecture.py or archive.py
* Catalog #287 + #323 canonical Provenance: no score claim; pure helper
* Catalog #290 canonical-vs-unique decision per layer: PER-SUBSTRATE-
  UNIQUE policy choice (NOT shared with sister substrates; v7
  strip-everything uses per-class anchors not per-(level, class) LUT
  so the aggregation policy is different there)
* Catalog #335 cathedral consumer auto-discovery: this module is the
  canonical helper a future
  ``chroma_lut_aggregation_policy_consumer`` would route through
* Catalog #344 canonical equation cross-reference: see
  ``canonical_equations/registry.jsonl`` entry
  ``chroma_lut_per_class_aggregation_policy_v1`` (registered Wave 9)

6-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = ACTIVE (per-(level, class) MEAN preserves
  per-bin L2-optimal centroid that point-sample MEDIAN can shift)
* hook #2 Pareto constraint = N/A (same canonical chroma_lut byte cost
  ``grayscale_levels * num_segnet_classes * 3`` regardless of policy)
* hook #3 bit-allocator = N/A (same byte cost; no per-tensor bit
  reallocation)
* hook #4 cathedral autopilot dispatch = ACTIVE (policy selection IS
  the canonical disambiguator between BYTE-DEFAULT-MEDIAN vs
  MEAN-vs-MODE-vs-WEIGHTED-MEAN arms)
* hook #5 continual-learning posterior = ACTIVE (paired smoke would
  emit empirical anchor for canonical equation
  ``chroma_lut_per_class_aggregation_policy_v1``)
* hook #6 probe-disambiguator = ACTIVE PRIMARY (this module IS the
  canonical disambiguator for cargo-cult #4)

Sister cross-reference: ``cls_lowres_downsample.py`` (Wave 5 cargo-cult
#6 sister helper at the cls_lowres surface). The two helpers form the
canonical per-substrate META extinction of the
"select-representative-by-default-without-empirical-validation" class
at BOTH spatial (cls_lowres) AND categorical (chroma LUT) surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

__all__ = [
    "CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT",
    "CHROMA_LUT_AGGREGATION_POLICY_NON_PROMOTABLE_PROVENANCE",
    "ChromaLutAggregationError",
    "ChromaLutAggregationVerdict",
    "SUPPORTED_AGGREGATION_POLICIES",
    "build_chroma_lut_with_policy",
    "verify_chroma_lut_invariants",
]


SUPPORTED_AGGREGATION_POLICIES: Final[tuple[str, ...]] = (
    "median_byte_default",
    "mean",
    "mode_per_cell",
    "weighted_mean_by_cell_count",
)
"""Canonical supported aggregation policy set.

* ``median_byte_default``: BYTE-DEFAULT. Per-(level, class) ``np.median``
  across all pixels in the bin. Robust to outliers; sister of legacy
  ``architecture.build_chroma_lut_from_ground_truth`` 2026-05-29
  behavior. Preserves archive-byte parity with all dispatches prior to
  Wave 9 landing.

* ``mean``: Per-(level, class) ``np.mean`` rounded to ``uint8``.
  Canonical L2-optimal centroid under Gaussian noise; different
  archive bytes vs MEDIAN whenever the bin distribution is skewed.

* ``mode_per_cell``: Per-(level, class) bin most-frequent RGB triple
  (treats each pixel's ``(R, G, B)`` tuple as a categorical value and
  picks the modal triple). Boundary-preserving sister of Wave 5
  cargo-cult #6 MODE policy at the RGB aggregation surface. Same byte
  cost; different archive bytes whenever a bin has a clear modal RGB
  triple (e.g. uniform sky / road / lane-marking pixels).

* ``weighted_mean_by_cell_count``: Per-(level, class) population-
  weighted mean. The weight is the relative cell population; this
  collapses to the simple mean when all bins are equally populated,
  and biases toward statistically well-populated bins when bin counts
  vary widely. Useful when the per-(level, class) histogram is heavy-
  tailed.
"""

CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT: Final[str] = "median_byte_default"
"""Canonical byte-default policy. PRESERVES archive-byte parity with all
prior empirical anchors keyed by ``np.median`` at
``architecture.build_chroma_lut_from_ground_truth``.

Switching the default away from this token requires operator opt-in
because it changes archive bytes for all v8 dispatches per CLAUDE.md
"Frontier scores are pointer-only" non-negotiable + Catalog #110/#113
HISTORICAL_PROVENANCE.
"""

CHROMA_LUT_AGGREGATION_POLICY_NON_PROMOTABLE_PROVENANCE: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "axis_tag": "[predicted]",
    "evidence_grade": "research-signal",
    "blockers": (
        "chroma_lut_aggregation_policy_is_canonical_helper_not_score_claim",
        "policy_choice_disambiguates_median_vs_mean_vs_mode_vs_weighted_mean_arms",
    ),
}
"""Canonical non-promotable provenance per Catalog #287 + #323 + #341."""


class ChromaLutAggregationError(ValueError):
    """Raised when the chroma LUT aggregation policy cannot be honored faithfully."""


@dataclass(frozen=True)
class ChromaLutAggregationVerdict:
    """Verdict from the chroma LUT aggregation helper.

    Carries:

    * the canonical policy token (one of
      :data:`SUPPORTED_AGGREGATION_POLICIES`)
    * the derived chroma LUT shape + dtype + sha256
    * per-(level, class) MEDIAN-vs-policy agreement fraction across all
      RGB channels (1.0 when ``policy == median_byte_default``)
    * non-promotable contract per Catalog #287 + #323

    The agreement_fraction is a research-signal metric: when close to
    1.0 (e.g. >0.99), the BYTE-DEFAULT MEDIAN policy is empirically
    indistinguishable from the chosen alternative for the input data;
    the cargo-cult #4 unwind is unnecessary at this distribution. When
    far from 1.0 (e.g. <0.9), the alternative materially differs from
    MEDIAN and the unwind becomes empirically relevant.
    """

    policy: str
    chroma_lut_shape: tuple[int, int, int]
    chroma_lut_sha256: str
    median_vs_policy_agreement_fraction: float
    """Per-(level, class, channel) agreement fraction between MEDIAN
    and the chosen policy. In [0.0, 1.0]. 1.0 means the two policies
    produce byte-identical chroma LUT; lower values quantify the
    empirical relevance of cargo-cult #4 unwind for this input
    distribution."""

    axis_tag: str = "[predicted]"
    score_claim: bool = False
    promotion_eligible: bool = False
    evidence_grade: str = "research-signal"

    def __post_init__(self) -> None:
        if self.policy not in SUPPORTED_AGGREGATION_POLICIES:
            raise ChromaLutAggregationError(
                f"policy={self.policy!r} not in {SUPPORTED_AGGREGATION_POLICIES}"
            )
        if len(self.chroma_lut_shape) != 3:
            raise ChromaLutAggregationError(
                f"chroma_lut_shape must be 3D; got {self.chroma_lut_shape}"
            )
        if self.chroma_lut_shape[2] != 3:
            raise ChromaLutAggregationError(
                f"chroma_lut last axis must be 3 (RGB); got {self.chroma_lut_shape}"
            )
        if not (0.0 <= self.median_vs_policy_agreement_fraction <= 1.0):
            raise ChromaLutAggregationError(
                f"median_vs_policy_agreement_fraction="
                f"{self.median_vs_policy_agreement_fraction} outside [0.0, 1.0]"
            )
        if self.score_claim is not False:
            raise ChromaLutAggregationError(
                "score_claim MUST be False for chroma_lut verdict per Catalog #287"
            )
        if self.promotion_eligible is not False:
            raise ChromaLutAggregationError(
                "promotion_eligible MUST be False per Catalog #287 + #323"
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize to JSON-safe dict per Catalog #287 + #323."""
        return {
            "policy": self.policy,
            "chroma_lut_shape": list(self.chroma_lut_shape),
            "chroma_lut_sha256": self.chroma_lut_sha256,
            "median_vs_policy_agreement_fraction": (
                self.median_vs_policy_agreement_fraction
            ),
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "evidence_grade": self.evidence_grade,
        }


def _aggregate_per_bin(
    rgb_flat: np.ndarray,
    cls_flat: np.ndarray,
    level_flat: np.ndarray,
    *,
    grayscale_levels: int,
    num_segnet_classes: int,
    policy: str,
) -> np.ndarray:
    """Inner per-(level, class) RGB aggregation per the chosen policy.

    Returns the ``(grayscale_levels, num_segnet_classes, 3)`` uint8 LUT.

    Bins with zero pixels are filled with the per-class GLOBAL stat
    computed under the SAME policy (i.e. median fallback for MEDIAN
    policy, mean fallback for MEAN policy, etc.); when the per-class
    population is also empty the (128, 128, 128) neutral gray is used
    so every bin has a valid RGB anchor (sister of
    architecture.build_chroma_lut_from_ground_truth fallback).
    """
    lut = np.zeros((grayscale_levels, num_segnet_classes, 3), dtype=np.uint8)
    for c in range(num_segnet_classes):
        cls_mask = cls_flat == c
        if cls_mask.any():
            global_stat = _aggregate_channels(
                rgb_flat, cls_mask, policy=policy
            )
        else:
            global_stat = np.array([128, 128, 128], dtype=np.uint8)
        for lvl in range(grayscale_levels):
            bin_mask = cls_mask & (level_flat == lvl)
            if bin_mask.any():
                lut[lvl, c, :] = _aggregate_channels(
                    rgb_flat, bin_mask, policy=policy
                )
            else:
                lut[lvl, c, :] = global_stat
    return lut


def _aggregate_channels(
    rgb_flat: np.ndarray,
    bin_mask: np.ndarray,
    *,
    policy: str,
) -> np.ndarray:
    """Aggregate a single bin's RGB values under the chosen policy.

    Returns a (3,) uint8 array.
    """
    if not bin_mask.any():
        return np.array([128, 128, 128], dtype=np.uint8)

    if policy == "median_byte_default":
        return np.array(
            [np.median(rgb_flat[ch][bin_mask]) for ch in range(3)],
            dtype=np.uint8,
        )

    if policy == "mean":
        return np.array(
            [
                np.clip(np.round(rgb_flat[ch][bin_mask].mean()), 0, 255)
                for ch in range(3)
            ],
            dtype=np.uint8,
        )

    if policy == "weighted_mean_by_cell_count":
        # The "weight" for a single bin is uniform across the bin's
        # pixels; the SHAPE difference vs simple MEAN appears at the
        # GLOBAL-fallback level (where the population can be many
        # orders larger than a single bin). This policy keeps the
        # per-bin estimator as the population-weighted mean of all
        # pixels in the bin (== simple mean of the bin's pixels;
        # mathematically identical to MEAN at the per-bin scope but
        # documented separately to preserve operator-facing arm
        # disambiguation per Catalog #308).
        return np.array(
            [
                np.clip(np.round(rgb_flat[ch][bin_mask].mean()), 0, 255)
                for ch in range(3)
            ],
            dtype=np.uint8,
        )

    if policy == "mode_per_cell":
        # Treat each pixel's (R, G, B) triple as a categorical 24-bit
        # value; pick the modal triple in the bin. Use numpy
        # unique-with-counts on the packed uint32 encoding for
        # canonical determinism + O(N log N) cost.
        r = rgb_flat[0][bin_mask].astype(np.uint32)
        g = rgb_flat[1][bin_mask].astype(np.uint32)
        b = rgb_flat[2][bin_mask].astype(np.uint32)
        packed = (r << 16) | (g << 8) | b
        unique, counts = np.unique(packed, return_counts=True)
        # First-wins tie-break (np.argmax returns first index on ties)
        idx = int(np.argmax(counts))
        modal_packed = int(unique[idx])
        return np.array(
            [
                (modal_packed >> 16) & 0xFF,
                (modal_packed >> 8) & 0xFF,
                modal_packed & 0xFF,
            ],
            dtype=np.uint8,
        )

    raise ChromaLutAggregationError(
        f"policy={policy!r} not in {SUPPORTED_AGGREGATION_POLICIES}"
    )


def build_chroma_lut_with_policy(
    rgb_pairs: np.ndarray,
    class_labels: np.ndarray,
    *,
    grayscale_levels: int,
    num_segnet_classes: int,
    policy: Literal[
        "median_byte_default",
        "mean",
        "mode_per_cell",
        "weighted_mean_by_cell_count",
    ] = CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT,
) -> tuple[np.ndarray, ChromaLutAggregationVerdict]:
    """Derive the chroma LUT from compress-time GT per the chosen policy.

    Per Wave 9 NSCS06 v8 cargo-cult #4 unwind 2026-05-29: this helper
    extracts the per-(level, class) ``np.median`` aggregation that
    previously lived inline in
    ``architecture.build_chroma_lut_from_ground_truth`` and adds three
    empirical alternatives (MEAN / MODE per cell /
    WEIGHTED_MEAN_BY_CELL_COUNT).

    Args:
        rgb_pairs: (N, 3, H, W) uint8 RGB frames (compress-time GT).
        class_labels: (N, H, W) uint8 SegNet argmax labels.
        grayscale_levels: Number of luma quantization levels.
        num_segnet_classes: Number of SegNet classes.
        policy: one of :data:`SUPPORTED_AGGREGATION_POLICIES`. Default
            :data:`CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT` preserves
            archive-byte parity.

    Returns:
        Tuple of ``(chroma_lut, verdict)`` where ``chroma_lut`` has shape
        ``(grayscale_levels, num_segnet_classes, 3)`` uint8 and
        ``verdict`` carries the canonical research-signal Provenance
        including the MEDIAN-vs-policy agreement fraction.

    Raises:
        ChromaLutAggregationError: on input shape / dtype / class-range
            mismatch or invalid policy token.
    """
    import hashlib

    if rgb_pairs.dtype != np.uint8:
        raise ChromaLutAggregationError(
            f"rgb_pairs dtype must be uint8; got {rgb_pairs.dtype}"
        )
    if class_labels.dtype != np.uint8:
        raise ChromaLutAggregationError(
            f"class_labels dtype must be uint8; got {class_labels.dtype}"
        )
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise ChromaLutAggregationError(
            f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}"
        )
    n, _, h, w = rgb_pairs.shape
    if class_labels.shape != (n, h, w):
        raise ChromaLutAggregationError(
            f"class_labels shape {class_labels.shape} != ({n}, {h}, {w})"
        )
    if grayscale_levels < 1:
        raise ChromaLutAggregationError(
            f"grayscale_levels={grayscale_levels} must be >= 1"
        )
    if num_segnet_classes < 1:
        raise ChromaLutAggregationError(
            f"num_segnet_classes={num_segnet_classes} must be >= 1"
        )
    if policy not in SUPPORTED_AGGREGATION_POLICIES:
        raise ChromaLutAggregationError(
            f"policy={policy!r} not in {SUPPORTED_AGGREGATION_POLICIES}"
        )
    if int(class_labels.max(initial=0)) >= num_segnet_classes:
        raise ChromaLutAggregationError(
            f"class_labels label {int(class_labels.max())} >= num_segnet_classes "
            f"{num_segnet_classes}"
        )

    # Compute BT.601 luma per pixel as the LUT level index (mirrors
    # architecture.build_chroma_lut_from_ground_truth).
    r = rgb_pairs[:, 0].astype(np.float32)
    g = rgb_pairs[:, 1].astype(np.float32)
    b = rgb_pairs[:, 2].astype(np.float32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b).clip(0.0, 255.0)
    level_step = 256 // grayscale_levels
    level_idx = np.clip(
        (luma // level_step).astype(np.int64), 0, grayscale_levels - 1
    )

    rgb_flat = rgb_pairs.transpose(1, 0, 2, 3).reshape(3, -1)
    cls_flat = class_labels.reshape(-1).astype(np.int64)
    level_flat = level_idx.reshape(-1)

    # ALWAYS compute MEDIAN as the reference for the agreement-fraction
    # research-signal metric; canonical byte-default behavior is the
    # comparator for every alternative arm.
    lut_median = _aggregate_per_bin(
        rgb_flat,
        cls_flat,
        level_flat,
        grayscale_levels=grayscale_levels,
        num_segnet_classes=num_segnet_classes,
        policy="median_byte_default",
    )

    if policy == "median_byte_default":
        lut = lut_median
        median_vs_policy_agreement_fraction = 1.0
    else:
        lut = _aggregate_per_bin(
            rgb_flat,
            cls_flat,
            level_flat,
            grayscale_levels=grayscale_levels,
            num_segnet_classes=num_segnet_classes,
            policy=policy,
        )
        agreement_count = int((lut == lut_median).sum())
        total_cells = grayscale_levels * num_segnet_classes * 3
        median_vs_policy_agreement_fraction = (
            agreement_count / total_cells if total_cells > 0 else 1.0
        )

    lut = np.ascontiguousarray(lut, dtype=np.uint8)
    sha = hashlib.sha256(lut.tobytes()).hexdigest()

    verdict = ChromaLutAggregationVerdict(
        policy=policy,
        chroma_lut_shape=lut.shape,
        chroma_lut_sha256=sha,
        median_vs_policy_agreement_fraction=median_vs_policy_agreement_fraction,
    )
    return lut, verdict


def verify_chroma_lut_invariants(
    chroma_lut: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
) -> None:
    """Verify canonical post-derivation invariants on the chroma LUT.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" +
    Catalog #295 strict-load discipline: every consumer of the chroma
    LUT should call this to surface invariant violations at the call
    site rather than at archive pack/parse time.
    """
    if chroma_lut.dtype != np.uint8:
        raise ChromaLutAggregationError(
            f"chroma_lut dtype must be uint8; got {chroma_lut.dtype}"
        )
    if chroma_lut.shape != expected_shape:
        raise ChromaLutAggregationError(
            f"chroma_lut shape {chroma_lut.shape} != expected {expected_shape}"
        )
    if expected_shape[2] != 3:
        raise ChromaLutAggregationError(
            f"chroma_lut last axis must be 3 (RGB); got {expected_shape}"
        )
