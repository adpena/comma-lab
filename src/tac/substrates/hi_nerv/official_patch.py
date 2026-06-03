# SPDX-License-Identifier: MIT
"""Portable official HiNeRV patch/index primitives.

Official HiNeRV trains and evaluates through 3D video patches.  These helpers
mirror ``models/patch_utils.py`` and ``datasets.py`` in NumPy so local
receiver/adaptor work can prove patch-index custody without importing the
official Torch tree at runtime.

The helpers are source-parity primitives only.  They deliberately carry no
score, promotion, or exact-dispatch authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF: Final[str] = (
    "official_hinerv_patch_index_numpy_v1"
)
HINERV_OFFICIAL_PATCH_INDEX_SOURCE_CONTRACT: Final[str] = (
    "HiNeRV/models/patch_utils.py vidx_to_pidx/video_to_patch/"
    "patch_to_video/compute_pixel_idx_3d plus datasets.py flat patch THW mapping"
)

FALSE_AUTHORITY: Final[dict[str, bool]] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class OfficialPatchIndexError(ValueError):
    """Raised when an official HiNeRV patch/index contract is invalid."""


@dataclass(frozen=True)
class OfficialPixelIndex3D:
    """Pixel coordinates and optional valid-padding masks for 3D patches."""

    pixel_indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None

    def as_jsonable_contract(self) -> dict[str, object]:
        return {
            "schema": "hinerv_official_pixel_index_3d_contract.v1",
            "proof_marker": HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF,
            "source_contract": HINERV_OFFICIAL_PATCH_INDEX_SOURCE_CONTRACT,
            "axis_shapes": [list(arr.shape) for arr in self.pixel_indices],
            "has_masks": self.masks is not None,
            **FALSE_AUTHORITY,
        }


def official_vidx_to_pidx(
    vidx: np.ndarray,
    *,
    vidx_max: tuple[int, int, int],
    pidx_max: tuple[int, int, int],
) -> np.ndarray:
    """Mirror official ``vidx_to_pidx`` for ``(t,h,w)`` video patch indices."""

    idx = _ensure_n3_index(vidx, name="vidx")
    vidx_max_t = _validate_positive_triple(vidx_max, name="vidx_max")
    pidx_max_t = _validate_positive_triple(pidx_max, name="pidx_max")
    scales = []
    for axis, (pmax, vmax) in enumerate(zip(pidx_max_t, vidx_max_t, strict=True)):
        if pmax % vmax != 0:
            raise OfficialPatchIndexError(
                f"pidx_max[{axis}]={pmax} must be divisible by vidx_max[{axis}]={vmax}"
            )
        scales.append(pmax // vmax)
    scale_t, scale_h, scale_w = scales
    pidx_t = idx[:, 0][:, None] * scale_t + np.arange(scale_t, dtype=np.int64)[None, :]
    pidx_h = idx[:, 1][:, None] * scale_h + np.arange(scale_h, dtype=np.int64)[None, :]
    pidx_w = idx[:, 2][:, None] * scale_w + np.arange(scale_w, dtype=np.int64)[None, :]
    expanded = np.stack(
        [
            np.broadcast_to(
                pidx_t[:, :, None, None],
                (idx.shape[0], scale_t, scale_h, scale_w),
            ),
            np.broadcast_to(
                pidx_h[:, None, :, None],
                (idx.shape[0], scale_t, scale_h, scale_w),
            ),
            np.broadcast_to(
                pidx_w[:, None, None, :],
                (idx.shape[0], scale_t, scale_h, scale_w),
            ),
        ],
        axis=-1,
    )
    return np.ascontiguousarray(expanded.reshape(-1, 3).astype(np.int64, copy=False))


def official_video_to_patch(
    video: np.ndarray,
    *,
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    """Convert ``(N,T,H,W,C)`` video batches to official HiNeRV patch order."""

    arr = np.asarray(video)
    if arr.ndim != 5:
        raise OfficialPatchIndexError(f"expected video shape (N,T,H,W,C), got {arr.shape}")
    patch_t, patch_h, patch_w = _validate_positive_triple(patch_size, name="patch_size")
    _, t_size, h_size, w_size, channels = (int(v) for v in arr.shape)
    for axis, (size, patch) in enumerate(
        ((t_size, patch_t), (h_size, patch_h), (w_size, patch_w))
    ):
        if size % patch != 0:
            raise OfficialPatchIndexError(
                f"video axis {axis} size {size} must be divisible by patch size {patch}"
            )
    n_patch_t = t_size // patch_t
    n_patch_h = h_size // patch_h
    n_patch_w = w_size // patch_w
    patches = (
        arr.reshape(
            -1,
            n_patch_t,
            patch_t,
            n_patch_h,
            patch_h,
            n_patch_w,
            patch_w,
            channels,
        )
        .transpose(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(-1, patch_t, patch_h, patch_w, channels)
    )
    return np.ascontiguousarray(patches)


def official_patch_to_video(
    patch: np.ndarray,
    *,
    video_size: tuple[int, int, int],
) -> np.ndarray:
    """Invert ``official_video_to_patch`` for official ``(T,H,W)`` video size."""

    arr = np.asarray(patch)
    if arr.ndim != 5:
        raise OfficialPatchIndexError(f"expected patch shape (P,T,H,W,C), got {arr.shape}")
    t_size, h_size, w_size = _validate_positive_triple(video_size, name="video_size")
    patch_t, patch_h, patch_w = (int(v) for v in arr.shape[1:4])
    channels = int(arr.shape[-1])
    for axis, (size, patch_size_axis) in enumerate(
        ((t_size, patch_t), (h_size, patch_h), (w_size, patch_w))
    ):
        if size % patch_size_axis != 0:
            raise OfficialPatchIndexError(
                f"video axis {axis} size {size} must be divisible by patch size {patch_size_axis}"
            )
    n_patch_t = t_size // patch_t
    n_patch_h = h_size // patch_h
    n_patch_w = w_size // patch_w
    if arr.shape[0] % (n_patch_t * n_patch_h * n_patch_w) != 0:
        raise OfficialPatchIndexError(
            "patch count must be a multiple of patches per video: "
            f"{arr.shape[0]} vs {n_patch_t * n_patch_h * n_patch_w}"
        )
    video = (
        arr.reshape(-1, n_patch_t, n_patch_h, n_patch_w, patch_t, patch_h, patch_w, channels)
        .transpose(0, 1, 4, 2, 5, 3, 6, 7)
        .reshape(-1, t_size, h_size, w_size, channels)
    )
    return np.ascontiguousarray(video)


def official_compute_pixel_idx_3d(
    idx: np.ndarray,
    *,
    idx_max: tuple[int, int, int],
    sizes: tuple[int, int, int],
    padding: tuple[int, int, int],
    clipped: bool = True,
    return_mask: bool = True,
) -> OfficialPixelIndex3D:
    """Mirror official ``compute_pixel_idx_3d`` for patch pixel ranges."""

    index = _ensure_n3_index(idx, name="idx")
    idx_max_t = _validate_positive_triple(idx_max, name="idx_max")
    sizes_t = _validate_positive_triple(sizes, name="sizes")
    padding_t = _validate_nonnegative_triple(padding, name="padding")
    patch_sizes = []
    for axis, (size, max_idx) in enumerate(zip(sizes_t, idx_max_t, strict=True)):
        if size % max_idx != 0:
            raise OfficialPatchIndexError(
                f"sizes[{axis}]={size} must be divisible by idx_max[{axis}]={max_idx}"
            )
        patch_sizes.append(size // max_idx)
    pixel_indices: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for axis, (size, patch_size_axis, pad) in enumerate(
        zip(sizes_t, patch_sizes, padding_t, strict=True)
    ):
        width = patch_size_axis + pad * 2
        raw = (
            index[:, axis][:, None] * patch_size_axis
            - pad
            + np.arange(width, dtype=np.int64)[None, :]
        )
        masks.append(np.ascontiguousarray((raw >= 0) & (raw < size)))
        if clipped:
            raw = np.clip(raw, 0, size - 1)
        pixel_indices.append(np.ascontiguousarray(raw.astype(np.int64, copy=False)))
    return OfficialPixelIndex3D(
        pixel_indices=(pixel_indices[0], pixel_indices[1], pixel_indices[2]),
        masks=(masks[0], masks[1], masks[2]) if return_mask else None,
    )


def official_flat_patch_index_to_thw(
    flat_index: int | np.ndarray,
    *,
    num_patches: tuple[int, int, int],
) -> np.ndarray:
    """Mirror ``VideoDataset.__getitem__`` flat index to ``(t,h,w)`` mapping."""

    n_patch_t, n_patch_h, n_patch_w = _validate_positive_triple(
        num_patches,
        name="num_patches",
    )
    flat = np.asarray(flat_index, dtype=np.int64)
    if np.any(flat < 0) or np.any(flat >= n_patch_t * n_patch_h * n_patch_w):
        raise OfficialPatchIndexError(
            f"flat patch indices must be in [0,{n_patch_t * n_patch_h * n_patch_w})"
        )
    t_idx = flat // (n_patch_h * n_patch_w)
    h_idx = (flat % (n_patch_h * n_patch_w)) // n_patch_w
    w_idx = (flat % (n_patch_h * n_patch_w)) % n_patch_w
    return np.stack([t_idx, h_idx, w_idx], axis=-1).astype(np.int64, copy=False)


def official_patch_index_contract() -> dict[str, object]:
    """Return false-authority contract metadata for patch/index parity."""

    return {
        "schema": "hinerv_official_patch_index_contract.v1",
        "proof_marker": HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF,
        "source_contract": HINERV_OFFICIAL_PATCH_INDEX_SOURCE_CONTRACT,
        **FALSE_AUTHORITY,
    }


def _ensure_n3_index(index: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(index, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise OfficialPatchIndexError(f"{name} must have shape (N,3), got {arr.shape}")
    return np.ascontiguousarray(arr)


def _validate_positive_triple(values: tuple[int, int, int], *, name: str) -> tuple[int, int, int]:
    out = tuple(int(v) for v in values)
    if len(out) != 3:
        raise OfficialPatchIndexError(f"{name} must be a 3-tuple, got {values!r}")
    if any(v <= 0 for v in out):
        raise OfficialPatchIndexError(f"{name} entries must be positive, got {out!r}")
    return out


def _validate_nonnegative_triple(
    values: tuple[int, int, int],
    *,
    name: str,
) -> tuple[int, int, int]:
    out = tuple(int(v) for v in values)
    if len(out) != 3:
        raise OfficialPatchIndexError(f"{name} must be a 3-tuple, got {values!r}")
    if any(v < 0 for v in out):
        raise OfficialPatchIndexError(f"{name} entries must be nonnegative, got {out!r}")
    return out


__all__ = [
    "FALSE_AUTHORITY",
    "HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF",
    "HINERV_OFFICIAL_PATCH_INDEX_SOURCE_CONTRACT",
    "OfficialPatchIndexError",
    "OfficialPixelIndex3D",
    "official_compute_pixel_idx_3d",
    "official_flat_patch_index_to_thw",
    "official_patch_index_contract",
    "official_patch_to_video",
    "official_video_to_patch",
    "official_vidx_to_pidx",
]
