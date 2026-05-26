# SPDX-License-Identifier: MIT
"""Per-pixel UNIWARD weight map computation (Fridrich-canonical).

Canonical formula (Holub-Fridrich-Denemark 2014 UNIWARD; Yousfi-adapter for
contest scorers):

    weight[h, w] = 1.0 / (eps + (d_seg_grad[h, w]) ** 2 + (d_pose_grad[h, w]) ** 2)

Higher weight = LOWER scorer sensitivity = SAFE to perturb (encode aggressively).
Lower weight = HIGHER scorer sensitivity = COSTLY (preserve precision).

The weight map is used in `score_aware_loss.compose_uniward_weighted_score_loss`
to weight per-pixel perturbation cost during training. Per CLAUDE.md "MLX
portable-local-substrate authority": training-time only; output tagged
`[macOS-MLX research-signal]` per Catalog #192/#317.

The weight map is COMPRESS-ONLY (NOT shipped to inflate per Carmack-preferred
budget conservation per HNeRV parity L4). The trained weights themselves embody
the routing.

Drift surface (per MLX↔CUDA bidirectional standing directive 2026-05-26):
- Source 1 (fp16/bf16): weight computation explicitly fp32-cast.
- Source 2 (softmax epsilon): additive eps=1e-6 in denominator.
- Sources 3-5: not applicable (no AdamW state / no bicubic / no EMA at weight
  computation surface).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

WEIGHT_MAP_EPS = 1e-6
WEIGHT_MAP_DTYPE = np.float32


def compute_per_pixel_uniward_weight_map_numpy(
    d_seg_grad_per_pixel: np.ndarray,
    d_pose_grad_per_pixel: np.ndarray,
    eps: float = WEIGHT_MAP_EPS,
) -> np.ndarray:
    """Compute UNIWARD per-pixel weight map from per-pixel scorer gradients.

    Parameters
    ----------
    d_seg_grad_per_pixel : np.ndarray, shape (H, W)
        Per-pixel SegNet gradient magnitude (sqrt of per-pixel d_seg/d_x ** 2
        summed over scorer channels). Must be non-negative.
    d_pose_grad_per_pixel : np.ndarray, shape (H, W)
        Per-pixel PoseNet gradient magnitude. Must be non-negative.
    eps : float
        Numerical stability constant; additive in denominator. Default 1e-6.

    Returns
    -------
    np.ndarray, shape (H, W), dtype=float32
        Per-pixel weight map. Higher values = lower scorer sensitivity = safer
        to perturb. Values are NOT normalized (caller normalizes if needed for
        loss-weighting interpretation).
    """
    if d_seg_grad_per_pixel.shape != d_pose_grad_per_pixel.shape:
        raise ValueError(
            f"shape mismatch: d_seg={d_seg_grad_per_pixel.shape} vs "
            f"d_pose={d_pose_grad_per_pixel.shape}"
        )
    if (d_seg_grad_per_pixel < 0).any() or (d_pose_grad_per_pixel < 0).any():
        raise ValueError("gradients must be non-negative magnitudes")
    seg_sq = d_seg_grad_per_pixel.astype(WEIGHT_MAP_DTYPE) ** 2
    pose_sq = d_pose_grad_per_pixel.astype(WEIGHT_MAP_DTYPE) ** 2
    denom = WEIGHT_MAP_DTYPE(eps) + seg_sq + pose_sq
    weight = WEIGHT_MAP_DTYPE(1.0) / denom
    return weight


def decompose_per_axis_weights(
    d_seg_grad_per_pixel: np.ndarray,
    d_pose_grad_per_pixel: np.ndarray,
    eps: float = WEIGHT_MAP_EPS,
) -> dict[str, np.ndarray]:
    """Return per-axis weight decomposition for observability per Catalog #305.

    Sister of `compute_per_pixel_uniward_weight_map_numpy`; emits seg-only,
    pose-only, and joint variants so the cargo-cult audit "JOINT d_seg + d_pose
    Fisher-info inverse beats single-axis" can be empirically validated.
    """
    seg_sq = d_seg_grad_per_pixel.astype(WEIGHT_MAP_DTYPE) ** 2
    pose_sq = d_pose_grad_per_pixel.astype(WEIGHT_MAP_DTYPE) ** 2
    eps_arr = WEIGHT_MAP_DTYPE(eps)
    return {
        "weight_seg_only": WEIGHT_MAP_DTYPE(1.0) / (eps_arr + seg_sq),
        "weight_pose_only": WEIGHT_MAP_DTYPE(1.0) / (eps_arr + pose_sq),
        "weight_joint": WEIGHT_MAP_DTYPE(1.0) / (eps_arr + seg_sq + pose_sq),
    }


def normalize_weight_map_to_unit_mean(weight_map: np.ndarray) -> np.ndarray:
    """Normalize so mean(weight) == 1.0 for loss-weighting interpretability.

    The unit-mean normalization means a uniform-weighting baseline corresponds
    to `weight = 1.0` everywhere, and the UNIWARD weighting redistributes mass
    around that baseline.
    """
    mean = weight_map.mean()
    if mean < WEIGHT_MAP_EPS:
        raise ValueError(
            f"weight map mean too small for normalization: {mean!r}"
        )
    return weight_map / mean


def histogram_weight_distribution(
    weight_map: np.ndarray,
    bins: int = 20,
) -> dict[str, np.ndarray]:
    """Observability per Catalog #305: per-pixel weight distribution histogram."""
    hist, edges = np.histogram(weight_map.ravel(), bins=bins)
    return {
        "histogram_counts": hist,
        "histogram_edges": edges,
        "weight_min": np.asarray(weight_map.min(), dtype=WEIGHT_MAP_DTYPE),
        "weight_max": np.asarray(weight_map.max(), dtype=WEIGHT_MAP_DTYPE),
        "weight_mean": np.asarray(weight_map.mean(), dtype=WEIGHT_MAP_DTYPE),
        "weight_median": np.asarray(np.median(weight_map), dtype=WEIGHT_MAP_DTYPE),
    }
