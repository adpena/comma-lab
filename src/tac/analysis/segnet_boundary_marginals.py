# SPDX-License-Identifier: MIT
"""SegNet boundary and logit-margin feature helpers.

These helpers are analysis primitives for A5-style bit allocation. They do not
claim scores and do not dispatch jobs. They convert frozen SegNet outputs into
per-pair scalar features that archive builders can consume explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundaryFeatureSummary:
    """Per-sample SegNet boundary feature vectors."""

    boundary_mass: np.ndarray
    low_margin_mass: np.ndarray
    mean_logit_margin: np.ndarray
    p10_logit_margin: np.ndarray

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "per_pair_boundary_mass": _float_list(self.boundary_mass),
            "per_pair_low_margin_mass": _float_list(self.low_margin_mass),
            "per_pair_mean_logit_margin": _float_list(self.mean_logit_margin),
            "per_pair_p10_logit_margin": _float_list(self.p10_logit_margin),
        }


def boundary_mask_from_labels(labels: np.ndarray, *, dilation: int = 1) -> np.ndarray:
    """Return a boundary mask for 2-D labels or a batch of 2-D labels.

    A pixel is marked when its label differs from a horizontal or vertical
    neighbor. The boundary is marked on both sides of the edge, then optionally
    dilated with a square max filter.
    """

    arr = np.asarray(labels)
    if arr.ndim == 2:
        batched = arr[None, ...]
        squeeze = True
    elif arr.ndim == 3:
        batched = arr
        squeeze = False
    else:
        raise ValueError(f"labels must have shape (H, W) or (N, H, W); got {arr.shape}")
    if batched.shape[-2] == 0 or batched.shape[-1] == 0:
        raise ValueError("labels must have non-empty spatial dimensions")

    boundary = np.zeros(batched.shape, dtype=bool)
    horizontal = batched[..., :, 1:] != batched[..., :, :-1]
    boundary[..., :, 1:] |= horizontal
    boundary[..., :, :-1] |= horizontal
    vertical = batched[..., 1:, :] != batched[..., :-1, :]
    boundary[..., 1:, :] |= vertical
    boundary[..., :-1, :] |= vertical

    dilated = dilate_binary_mask(boundary, dilation=dilation)
    return dilated[0] if squeeze else dilated


def dilate_binary_mask(mask: np.ndarray, *, dilation: int = 1) -> np.ndarray:
    """Dilate a boolean mask with a square odd-sized kernel."""

    if not isinstance(dilation, int) or isinstance(dilation, bool) or dilation < 1:
        raise ValueError("dilation must be a positive integer")
    arr = np.asarray(mask, dtype=bool)
    if dilation == 1:
        return arr.copy()
    if arr.ndim == 2:
        batched = arr[None, ...]
        squeeze = True
    elif arr.ndim == 3:
        batched = arr
        squeeze = False
    else:
        raise ValueError(f"mask must have shape (H, W) or (N, H, W); got {arr.shape}")

    radius = dilation // 2
    padded = np.pad(
        batched,
        ((0, 0), (radius, radius), (radius, radius)),
        mode="constant",
        constant_values=False,
    )
    out = np.zeros_like(batched, dtype=bool)
    for dy in range(dilation):
        for dx in range(dilation):
            out |= padded[:, dy : dy + batched.shape[1], dx : dx + batched.shape[2]]
    return out[0] if squeeze else out


def logit_margin(logits: np.ndarray) -> np.ndarray:
    """Return top-1 minus top-2 class logit margin per pixel."""

    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None, ...]
        squeeze = True
    elif arr.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"logits must have shape (C, H, W) or (N, C, H, W); got {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError("logits must contain at least two classes")
    if not np.isfinite(arr).all():
        raise ValueError("logits must contain finite values")
    top2 = np.partition(arr, kth=arr.shape[1] - 2, axis=1)[:, -2:, ...]
    margins = top2[:, 1, ...] - top2[:, 0, ...]
    return margins[0] if squeeze else margins


def summarize_boundary_features(
    *,
    labels: np.ndarray,
    logits: np.ndarray,
    dilation: int = 5,
    low_margin_threshold: float = 1.0,
) -> BoundaryFeatureSummary:
    """Summarize batched SegNet labels/logits into per-pair features."""

    label_arr = np.asarray(labels)
    if label_arr.ndim != 3:
        raise ValueError(f"labels must have shape (N, H, W); got {label_arr.shape}")
    margins = logit_margin(logits)
    if margins.shape != label_arr.shape:
        raise ValueError(
            f"margin shape {margins.shape} does not match labels shape {label_arr.shape}"
        )
    boundary = boundary_mask_from_labels(label_arr, dilation=dilation)
    boundary_mass = boundary.reshape(boundary.shape[0], -1).mean(axis=1)
    low_margin = (margins <= float(low_margin_threshold)).reshape(margins.shape[0], -1)
    flat_margin = margins.reshape(margins.shape[0], -1)
    return BoundaryFeatureSummary(
        boundary_mass=boundary_mass.astype(np.float64),
        low_margin_mass=low_margin.mean(axis=1).astype(np.float64),
        mean_logit_margin=flat_margin.mean(axis=1).astype(np.float64),
        p10_logit_margin=np.percentile(flat_margin, 10.0, axis=1).astype(np.float64),
    )


def merge_feature_summaries(summaries: list[BoundaryFeatureSummary]) -> BoundaryFeatureSummary:
    """Concatenate multiple feature summaries in sample order."""

    if not summaries:
        raise ValueError("at least one summary is required")
    return BoundaryFeatureSummary(
        boundary_mass=np.concatenate([item.boundary_mass for item in summaries]),
        low_margin_mass=np.concatenate([item.low_margin_mass for item in summaries]),
        mean_logit_margin=np.concatenate([item.mean_logit_margin for item in summaries]),
        p10_logit_margin=np.concatenate([item.p10_logit_margin for item in summaries]),
    )


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).tolist()]

