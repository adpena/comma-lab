# SPDX-License-Identifier: MIT
"""UNIWARD-weighted chroma-LUT derivation (7th-order canonical surface).

Per the 6th-order Carmack-dissent verdict: UNIWARD's natural application
is ENTROPY-CODED + QUANTIZED + PER-SYMBOL-ROUTABLE. The NSCS06 v8 chroma
LUT IS the contest's UNIWARD-natural locus: the (16 levels x 5 classes)
bins partition pixels; the per-(level, class) chroma triplet IS the
canonical Fridrich per-symbol routing target.

The v8 canonical LUT derivation uses unweighted MEDIAN per bin (sister of
``architecture.build_chroma_lut_from_ground_truth`` line 293). The UNIWARD-
weighted variant replaces unweighted median with UNIWARD-weighted MEDIAN
(or weighted MEAN as alternative).

The mathematical predicate:

    Unweighted median (canonical v8): rank-50% of pixel chroma values in bin
    UNIWARD-weighted median (7th-order): rank-50% of UNIWARD-weighted-CDF

    cdf(rgb_value) = (sum of UNIWARD weights at pixels in bin with chroma <= rgb_value)
                     / (sum of UNIWARD weights of all pixels in bin)
    weighted_median = first rgb_value where cdf(rgb_value) >= 0.5

The intuition: the LUT entry concentrates on the chroma values that
high-sensitivity pixels prefer; low-sensitivity pixels contribute less to
the entry. The LUT entry preserves precision exactly where the scorer is
sensitive — the canonical Fridrich inverse-steganalysis routing primitive
applied at the per-LUT-symbol granularity.

Per Catalog #230 sister-disjoint: v8 substrate is READ-ONLY consumer-
imported. This module SHADOWS ``build_chroma_lut_from_ground_truth`` with
a UNIWARD-weighted-statistic variant; the canonical v8 derivation is
unchanged.

Per Catalog #287 placeholder rejection: every helper carries substantive
non-placeholder docstring rationale.

Per Catalog #323 canonical Provenance + Catalog #341 canonical-routing
markers: every LUT derivation result is non-promotable
``[macOS-MLX research-signal]``.
"""

from __future__ import annotations

import numpy as np

from .weight_map_per_lut_index import (
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PER_LUT_INDEX_WEIGHT_DTYPE,
    PER_LUT_INDEX_WEIGHT_EPS,
    _compute_luma_quant_level,
)

__all__ = [
    "WeightedMedianResult",
    "weighted_median_per_channel",
    "build_uniward_weighted_chroma_lut",
    "compare_uniward_vs_canonical_lut",
]


class WeightedMedianResult:
    """Container for per-(level, class) weighted median derivation result."""

    __slots__ = (
        "lut_uniward_weighted",
        "lut_canonical_median",
        "per_bin_l2_difference",
        "num_bins_changed",
        "max_per_channel_delta_u8",
    )

    def __init__(
        self,
        *,
        lut_uniward_weighted: np.ndarray,
        lut_canonical_median: np.ndarray,
        per_bin_l2_difference: np.ndarray,
        num_bins_changed: int,
        max_per_channel_delta_u8: int,
    ) -> None:
        self.lut_uniward_weighted = lut_uniward_weighted
        self.lut_canonical_median = lut_canonical_median
        self.per_bin_l2_difference = per_bin_l2_difference
        self.num_bins_changed = num_bins_changed
        self.max_per_channel_delta_u8 = max_per_channel_delta_u8


def weighted_median_per_channel(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Canonical UNIWARD-weighted median for a 1-D channel of pixel values.

    Returns the value where the cumulative-weight CDF first reaches 0.5.
    Sister of np.median but weighted by per-pixel UNIWARD weight.

    Parameters
    ----------
    values : np.ndarray, shape (K,), dtype=uint8
        Per-pixel channel values within a single (level, class) bin.
    weights : np.ndarray, shape (K,)
        Per-pixel UNIWARD weights (non-negative).

    Returns
    -------
    np.ndarray, shape (), dtype=uint8
        Weighted median chroma value.
    """
    if values.size == 0:
        raise ValueError("weighted_median requires at least one value")
    if values.shape != weights.shape:
        raise ValueError(
            f"values shape {values.shape} != weights shape {weights.shape}"
        )
    # Sort by value
    sort_idx = np.argsort(values, kind="stable")
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx].astype(np.float64)
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0.0:
        # Degenerate fallback (all weights zero); use unweighted median per
        # sister discipline (canonical v8 fallback)
        return np.uint8(np.median(values))
    cumulative = np.cumsum(sorted_weights)
    target = 0.5 * total_weight
    # First value where cumulative >= target
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(idx, sorted_values.size - 1)
    return np.uint8(sorted_values[idx])


def build_uniward_weighted_chroma_lut(
    *,
    rgb_pairs: np.ndarray,
    class_labels: np.ndarray,
    per_pixel_uniward_weight: np.ndarray,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
) -> np.ndarray:
    """Derive UNIWARD-weighted-median chroma LUT (7th-order canonical surface).

    For each (level, class) bin: derive the LUT entry via per-channel
    UNIWARD-weighted median over compress-time pixels. Empty bins fall back
    to the per-class GLOBAL UNIWARD-weighted median (sister of canonical
    v8 ``build_chroma_lut_from_ground_truth`` fallback).

    Parameters
    ----------
    rgb_pairs : np.ndarray, shape (N, 3, H, W), dtype=uint8
        Compress-time GT RGB frames (sister of v8 LUT derivation input).
    class_labels : np.ndarray, shape (N, H, W), dtype=uint8
        SegNet argmax labels per pixel.
    per_pixel_uniward_weight : np.ndarray, shape (H, W) OR (N, H, W)
        Per-pixel UNIWARD weight map. 2-D form is broadcast to all N frames.
    grayscale_levels : int
        LUT level dimension (default 16).
    num_segnet_classes : int
        LUT class dimension (default 5).

    Returns
    -------
    np.ndarray, shape (grayscale_levels, num_segnet_classes, 3), dtype=uint8
        UNIWARD-weighted chroma LUT. Byte-comparable to canonical v8 LUT
        (same shape + dtype) so the rest of the v8 inflate pipeline is
        agnostic to which derivation produced it.
    """
    if rgb_pairs.dtype != np.uint8:
        raise ValueError(f"rgb_pairs must be uint8; got {rgb_pairs.dtype}")
    if class_labels.dtype != np.uint8:
        raise ValueError(f"class_labels must be uint8; got {class_labels.dtype}")
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise ValueError(f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}")
    n, _, h, w = rgb_pairs.shape
    if class_labels.shape != (n, h, w):
        raise ValueError(
            f"class_labels shape {class_labels.shape} != ({n}, {h}, {w})"
        )

    weight_array = per_pixel_uniward_weight.astype(PER_LUT_INDEX_WEIGHT_DTYPE)
    if weight_array.ndim == 2:
        if weight_array.shape != (h, w):
            raise ValueError(
                f"weight_map shape {weight_array.shape} != ({h}, {w})"
            )
        weight_array = np.broadcast_to(weight_array, (n, h, w)).copy()
    elif weight_array.ndim == 3:
        if weight_array.shape != (n, h, w):
            raise ValueError(
                f"weight_array shape {weight_array.shape} != ({n}, {h}, {w})"
            )
    else:
        raise ValueError(
            f"per_pixel_uniward_weight must be 2-D or 3-D; got ndim={weight_array.ndim}"
        )

    if (weight_array < 0).any():
        raise ValueError("UNIWARD weights must be non-negative")

    level_idx = _compute_luma_quant_level(rgb_pairs, grayscale_levels)

    lut = np.zeros((grayscale_levels, num_segnet_classes, 3), dtype=np.uint8)
    rgb_flat = rgb_pairs.transpose(1, 0, 2, 3).reshape(3, -1)  # (3, N*H*W)
    cls_flat = class_labels.reshape(-1).astype(np.int64)
    level_flat = level_idx.reshape(-1)
    weight_flat = weight_array.reshape(-1)

    for c in range(num_segnet_classes):
        cls_mask = cls_flat == c
        if cls_mask.any():
            # Per-class UNIWARD-weighted global median as fallback
            cls_weights = weight_flat[cls_mask]
            global_median = np.array(
                [
                    weighted_median_per_channel(rgb_flat[ch][cls_mask], cls_weights)
                    for ch in range(3)
                ],
                dtype=np.uint8,
            )
        else:
            global_median = np.array([128, 128, 128], dtype=np.uint8)
        for lvl in range(grayscale_levels):
            bin_mask = cls_mask & (level_flat == lvl)
            if bin_mask.any():
                bin_weights = weight_flat[bin_mask]
                for ch in range(3):
                    lut[lvl, c, ch] = weighted_median_per_channel(
                        rgb_flat[ch][bin_mask], bin_weights
                    )
            else:
                lut[lvl, c, :] = global_median
    return lut


def compare_uniward_vs_canonical_lut(
    *,
    lut_uniward_weighted: np.ndarray,
    lut_canonical_median: np.ndarray,
) -> WeightedMedianResult:
    """Compare UNIWARD-weighted LUT vs canonical-median LUT for verdict.

    Produces the empirical signal for the 7th-order verdict:
    - byte-identical LUTs across BOTH derivations → PARADIGM-NULL-NO-EFFECT
      EVEN AT ENTROPY-CODED-SIDECAR (final Fridrich-canonical falsification
      for our contest application; triggers Catalog #348 retroactive sweep)
    - measurably different LUTs (num_bins_changed > 0; max_per_channel_delta
      > 0) → PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR (proceed to
      paired-CUDA validation per CLAUDE.md "Submission auth eval - BOTH CPU
      AND CUDA")

    Per Catalog #307 paradigm-vs-implementation classification: the verdict
    here is on the ROUTING-PRIMITIVE-EFFECT-AT-NATURAL-DOMAIN axis, NOT on
    the score-improvement axis. Score-improvement requires paired-CUDA
    auth-eval per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
    non-negotiable + the full v8 contest dispatch pipeline.
    """
    if lut_uniward_weighted.shape != lut_canonical_median.shape:
        raise ValueError(
            f"shape mismatch: uniward={lut_uniward_weighted.shape} "
            f"vs canonical={lut_canonical_median.shape}"
        )
    if lut_uniward_weighted.dtype != np.uint8 or lut_canonical_median.dtype != np.uint8:
        raise ValueError("both LUTs must be uint8 for canonical comparison")

    # Per-bin L2 difference across RGB channels
    diff = lut_uniward_weighted.astype(np.int32) - lut_canonical_median.astype(np.int32)
    per_bin_l2 = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
    num_bins_changed = int((per_bin_l2 > 0).sum())
    max_delta = int(np.abs(diff).max())

    return WeightedMedianResult(
        lut_uniward_weighted=lut_uniward_weighted,
        lut_canonical_median=lut_canonical_median,
        per_bin_l2_difference=per_bin_l2,
        num_bins_changed=num_bins_changed,
        max_per_channel_delta_u8=max_delta,
    )
