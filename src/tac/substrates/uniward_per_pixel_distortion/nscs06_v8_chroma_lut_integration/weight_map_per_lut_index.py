# SPDX-License-Identifier: MIT
"""UNIWARD per-LUT-index weight aggregation (7th-order canonical surface).

Per the 6th-order landing memo Carmack-dissent verdict: UNIWARD's natural
application domain is ENTROPY-CODED + QUANTIZED + PER-SYMBOL-ROUTABLE
surfaces (Holub-Fridrich-Denemark 2014 DEFINED UNIWARD on JPEG DCT
coefficients). The NSCS06 v8 chroma LUT (16 levels x 5 classes x 3 RGB)
IS that surface for our contest: quantized luma indexes the LUT level,
SegNet argmax indexes the class, the (level, class) bin contains the
canonical chroma triplet at that combination.

This module aggregates per-pixel UNIWARD weights into the (level, class)
bins of the v8 LUT, producing per-LUT-index UNIWARD weights. The weights
indicate which (level, class) bins matter most to scorer-conditional
sensitivity — bins with high aggregate weight from high-sensitivity
pixels carry the most score-mass; bins with weight only from low-
sensitivity pixels can tolerate aggressive chroma quantization.

Per Catalog #287 placeholder rejection: every helper carries non-
placeholder docstring rationale. Per Catalog #323 canonical Provenance:
every per-LUT-index weight surface is provenance-tagged
``[macOS-MLX research-signal]`` + ``score_claim=False`` + ``promotable=False``
per Catalog #341.

Per CLAUDE.md MLX-first numpy-portable bridge contract: TRAINING happens
in MLX (this module is numpy-only because per-LUT-index aggregation is
sparse-scatter which numpy handles natively); the canonical sister at
``boostnerv_integration.py`` consumes torch.float32 cached gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Mirror v8 substrate constants per Catalog #230 sister-disjoint (READ-ONLY
# consumer import; no v8 substrate mutation). Sister of v8 architecture.py
# ``GRAYSCALE_LEVELS_DEFAULT`` + ``NUM_SEGNET_CLASSES``.
GRAYSCALE_LEVELS_DEFAULT = 16
NUM_SEGNET_CLASSES = 5

# UNIWARD numerical-stability epsilon per Holub-Fridrich-Denemark 2014.
# Sister of weight_map.WEIGHT_MAP_EPS so per-pixel and per-LUT-index
# layers stay numerically aligned.
PER_LUT_INDEX_WEIGHT_EPS = 1e-6
PER_LUT_INDEX_WEIGHT_DTYPE = np.float32


@dataclass(frozen=True)
class PerLutIndexUniwardWeights:
    """Per-LUT-index UNIWARD weight aggregation result.

    Shape contract:
        weight_per_bin : (grayscale_levels, num_segnet_classes) float32
            Per-(level, class) aggregate UNIWARD weight. Higher = more
            scorer-conditional sensitivity concentrated in that bin.
        pixel_count_per_bin : (grayscale_levels, num_segnet_classes) int64
            Number of compress-time pixels assigned to each bin (for
            observability per Catalog #305).
        weight_sum_per_bin : (grayscale_levels, num_segnet_classes) float32
            Sum of per-pixel UNIWARD weights in each bin (numerator of
            weighted statistic).
        weight_mean_per_bin : (grayscale_levels, num_segnet_classes) float32
            Mean of per-pixel UNIWARD weights in each bin (= weight_sum /
            pixel_count; useful for normalization). NaN for empty bins.
    """

    weight_per_bin: np.ndarray
    pixel_count_per_bin: np.ndarray
    weight_sum_per_bin: np.ndarray
    weight_mean_per_bin: np.ndarray

    @property
    def grayscale_levels(self) -> int:
        return int(self.weight_per_bin.shape[0])

    @property
    def num_segnet_classes(self) -> int:
        return int(self.weight_per_bin.shape[1])

    @property
    def num_nonempty_bins(self) -> int:
        return int((self.pixel_count_per_bin > 0).sum())

    @property
    def dynamic_range_ratio(self) -> float:
        """Ratio of max-bin-weight to min-nonzero-bin-weight (observability)."""
        nonzero = self.weight_per_bin[self.weight_per_bin > 0]
        if nonzero.size == 0:
            return 1.0
        max_w = float(nonzero.max())
        min_w = float(nonzero.min())
        if min_w <= 0.0:
            return 1.0
        return max_w / min_w


def _compute_luma_quant_level(
    rgb_pairs: np.ndarray,
    grayscale_levels: int,
) -> np.ndarray:
    """Compute per-pixel luma quantization level matching v8 substrate convention.

    Sister of v8 ``architecture.build_chroma_lut_from_ground_truth`` luma
    quantization (BT.601 + np.floor; matches inflate-side ``gray_u8 >> shift``).
    """
    if rgb_pairs.dtype != np.uint8:
        raise ValueError(
            f"rgb_pairs must be uint8 (sister of v8 LUT derivation); got {rgb_pairs.dtype}"
        )
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise ValueError(f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}")
    r = rgb_pairs[:, 0].astype(np.float32)
    g = rgb_pairs[:, 1].astype(np.float32)
    b = rgb_pairs[:, 2].astype(np.float32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b).clip(0.0, 255.0)
    level_step = 256 // grayscale_levels
    level_idx = np.clip(
        (luma // level_step).astype(np.int64), 0, grayscale_levels - 1
    )
    return level_idx


def aggregate_per_pixel_uniward_weights_into_lut_bins(
    *,
    rgb_pairs: np.ndarray,
    class_labels: np.ndarray,
    per_pixel_uniward_weight: np.ndarray,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
) -> PerLutIndexUniwardWeights:
    """Aggregate per-pixel UNIWARD weights into per-(level, class) bins.

    For each LUT bin (level, class), sum the per-pixel UNIWARD weights of
    pixels assigned to that bin (per v8 luma quantization + SegNet argmax).
    The aggregate per-bin weight is the canonical UNIWARD weight for that
    LUT index — directly analogous to Fridrich's per-DCT-coefficient
    weighting where coefficients-with-high-aggregate-weight are the
    sensitivity-preserving slots.

    Parameters
    ----------
    rgb_pairs : np.ndarray, shape (N, 3, H, W), dtype=uint8
        Compress-time GT RGB frames (sister of v8 LUT derivation input).
    class_labels : np.ndarray, shape (N, H, W), dtype=uint8
        SegNet argmax labels per pixel.
    per_pixel_uniward_weight : np.ndarray, shape (N, H, W) OR (H, W)
        Per-pixel UNIWARD weight ``w[h,w] = 1 / (eps + (d_seg_grad)^2 +
        (d_pose_grad)^2)`` from canonical
        ``weight_map.compute_per_pixel_uniward_weight_map_numpy``. If 2-D
        (H, W), it is broadcast to all N frames (same per-frame weight
        map applied uniformly — the canonical N+1 sister pattern).
    grayscale_levels : int
        LUT level dimension (default 16, sister of v8 ``GRAYSCALE_LEVELS_DEFAULT``).
    num_segnet_classes : int
        LUT class dimension (default 5, sister of v8 ``NUM_SEGNET_CLASSES``).

    Returns
    -------
    PerLutIndexUniwardWeights
        Frozen dataclass with per-bin aggregate weights + pixel counts +
        observability surfaces per Catalog #305.

    Notes
    -----
    Empty bins (zero pixels) receive ``weight_per_bin=0.0`` and
    ``weight_mean_per_bin=nan``; downstream LUT derivation handles
    empty bins via the canonical per-class-global-median fallback
    (sister of v8 ``build_chroma_lut_from_ground_truth``).
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

    # Broadcast 2-D weight map to all N frames if needed (canonical N+1 pattern)
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
            f"per_pixel_uniward_weight must be 2-D (H,W) or 3-D (N,H,W); "
            f"got ndim={weight_array.ndim}"
        )

    if (weight_array < 0).any():
        raise ValueError("UNIWARD weights must be non-negative")

    level_idx = _compute_luma_quant_level(rgb_pairs, grayscale_levels)
    cls_flat = class_labels.reshape(-1).astype(np.int64)
    level_flat = level_idx.reshape(-1)
    weight_flat = weight_array.reshape(-1)

    # Per-bin aggregation via canonical flat-index + np.bincount (numpy-native;
    # MLX-portable when MLX gains sparse-scatter; falls back to numpy here).
    flat_bin_idx = level_flat * num_segnet_classes + cls_flat
    flat_size = grayscale_levels * num_segnet_classes
    weight_sum_per_bin_flat = np.bincount(
        flat_bin_idx, weights=weight_flat, minlength=flat_size
    ).astype(PER_LUT_INDEX_WEIGHT_DTYPE)
    pixel_count_per_bin_flat = np.bincount(
        flat_bin_idx, minlength=flat_size
    ).astype(np.int64)
    weight_sum_per_bin = weight_sum_per_bin_flat.reshape(
        grayscale_levels, num_segnet_classes
    )
    pixel_count_per_bin = pixel_count_per_bin_flat.reshape(
        grayscale_levels, num_segnet_classes
    )
    # Mean (with NaN for empty bins per docstring contract)
    nonempty = pixel_count_per_bin > 0
    weight_mean_per_bin = np.full_like(weight_sum_per_bin, np.nan)
    weight_mean_per_bin[nonempty] = (
        weight_sum_per_bin[nonempty] / pixel_count_per_bin[nonempty].astype(
            PER_LUT_INDEX_WEIGHT_DTYPE
        )
    )

    # Canonical per-bin weight = sum (the canonical Fridrich aggregate;
    # high-sum bins = high-aggregate-sensitivity = preserve precision)
    weight_per_bin = weight_sum_per_bin.copy()

    return PerLutIndexUniwardWeights(
        weight_per_bin=weight_per_bin,
        pixel_count_per_bin=pixel_count_per_bin,
        weight_sum_per_bin=weight_sum_per_bin,
        weight_mean_per_bin=weight_mean_per_bin,
    )


def build_canonical_provenance_for_per_lut_index_aggregation(
    *,
    per_bin_weights: PerLutIndexUniwardWeights,
    uniward_weight_eps: float = PER_LUT_INDEX_WEIGHT_EPS,
) -> dict:
    """Build canonical Provenance per Catalog #323 + Catalog #341 markers.

    Every per-LUT-index aggregation result carries non-promotable markers
    so the surface cannot leak into score/promotion signals. Sister of
    boostnerv_integration ``build_canonical_provenance_for_integration``.
    """
    return {
        "integration_id": "uniward_per_lut_index_into_nscs06_v8_chroma_lut",
        "integration_version": "v1_2026-05-26_7th_order",
        "consumed_substrate_id": "nscs06_v8_chroma_lut",
        "consumed_substrate_scope": "read_only_consumer_import",
        "grayscale_levels": per_bin_weights.grayscale_levels,
        "num_segnet_classes": per_bin_weights.num_segnet_classes,
        "num_nonempty_bins": per_bin_weights.num_nonempty_bins,
        "dynamic_range_ratio": float(per_bin_weights.dynamic_range_ratio),
        "uniward_weight_eps": float(uniward_weight_eps),
        # Canonical Provenance non-promotable markers per Catalog #341
        "evidence_grade": "macOS-MLX research-signal",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "hardware_substrate_recommendation": "darwin_arm64_m5_max_mlx_local",
        "measurement_axis": "[macOS-MLX research-signal]",
        # Sister hooks per Catalog #125
        "hook_numbers_fired": [1, 5],
        "entropy_position": "P3_entropy_coded_sidecar_per_lut_index_routing",
        # Sister-disjoint discipline acknowledgment per Catalog #230
        "nscs06_v8_substrate_modification_scope": "none_read_only_consumer_import",
    }
