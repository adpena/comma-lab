"""Scorer-free inflate helper for Coord-MLP residual sidecar bytes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.substrates.coord_mlp_residual_sidecar.archive import (
    CoordMlpPatch,
    CoordMlpResidualSidecarError,
    CoordMlpResidualWeights,
    parse_sidecar,
)


@dataclass(frozen=True)
class CoordMlpInflateResult:
    """Result of applying sidecar bytes to decoded RGB frames."""

    frames: np.ndarray
    consumed_bytes: int
    consumed_sha256: str
    applied_pixels: int
    consumed_sections: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "applied_pixels": self.applied_pixels,
            "consumed_bytes": self.consumed_bytes,
            "consumed_sha256": self.consumed_sha256,
            "consumed_sections": list(self.consumed_sections),
            "scorer_at_inflate": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }


def apply_sidecar_to_rgb(
    frames: np.ndarray,
    sidecar_blob: bytes,
    *,
    require_nonzero: bool = True,
) -> CoordMlpInflateResult:
    """Apply a Coord-MLP residual sidecar to ``(T, H, W, 3)`` uint8 frames."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise CoordMlpResidualSidecarError(
            f"frames must have shape (T, H, W, 3); got {frames.shape}"
        )
    if frames.dtype != np.uint8:
        raise CoordMlpResidualSidecarError(f"frames dtype must be uint8; got {frames.dtype}")
    parsed = parse_sidecar(sidecar_blob)
    if require_nonzero and parsed.structural_noop:
        raise CoordMlpResidualSidecarError("Coord-MLP sidecar is structural no-op")

    out = frames.astype(np.int16, copy=True)
    applied_pixels = 0
    for patch in parsed.patches:
        _assert_patch_in_bounds(patch, frames.shape)
        residual = _evaluate_patch_residual(
            parsed.weights,
            patch=patch,
            total_frames=int(frames.shape[0]),
        )
        y0, y1 = patch.y, patch.y + patch.height
        x0, x1 = patch.x, patch.x + patch.width
        out[patch.frame_index, y0:y1, x0:x1, :] = np.clip(
            out[patch.frame_index, y0:y1, x0:x1, :] + residual,
            0,
            255,
        )
        applied_pixels += int(patch.height * patch.width)

    return CoordMlpInflateResult(
        frames=out.astype(np.uint8, copy=False),
        consumed_bytes=len(sidecar_blob),
        consumed_sha256=hashlib.sha256(sidecar_blob).hexdigest(),
        applied_pixels=applied_pixels,
        consumed_sections=tuple(section.name for section in parsed.sections),
    )


def _assert_patch_in_bounds(patch: CoordMlpPatch, shape: tuple[int, ...]) -> None:
    frames, height, width, channels = map(int, shape)
    if channels != 3:
        raise CoordMlpResidualSidecarError("RGB channel count must be 3")
    if patch.frame_index >= frames:
        raise CoordMlpResidualSidecarError(
            f"patch frame_index={patch.frame_index} out of bounds for {frames} frames"
        )
    if patch.y + patch.height > height or patch.x + patch.width > width:
        raise CoordMlpResidualSidecarError(
            f"patch {(patch.y, patch.x, patch.height, patch.width)} outside frame {(height, width)}"
        )


def _evaluate_patch_residual(
    weights: CoordMlpResidualWeights,
    *,
    patch: CoordMlpPatch,
    total_frames: int,
) -> np.ndarray:
    features = _coord_features(patch, total_frames=total_frames)
    w1 = weights.w1_int8.astype(np.float32) / 64.0
    b1 = weights.b1_int16.astype(np.float32) / 1024.0
    w2 = weights.w2_int8.astype(np.float32) / 64.0
    b2 = weights.b2_int16.astype(np.float32) / 16.0
    hidden = np.maximum(features @ w1.T + b1, 0.0)
    residual = hidden @ w2.T + b2
    return np.clip(np.rint(residual), -127, 127).astype(np.int16)


def _coord_features(patch: CoordMlpPatch, *, total_frames: int) -> np.ndarray:
    if patch.width == 1:
        xs = np.array([0.0], dtype=np.float32)
    else:
        xs = np.linspace(-1.0, 1.0, patch.width, dtype=np.float32)
    if patch.height == 1:
        ys = np.array([0.0], dtype=np.float32)
    else:
        ys = np.linspace(-1.0, 1.0, patch.height, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    t_value = 0.0 if total_frames <= 1 else 2.0 * float(patch.frame_index) / float(total_frames - 1) - 1.0
    grid_t = np.full_like(grid_x, t_value, dtype=np.float32)
    return np.stack((grid_t, grid_y, grid_x), axis=-1)


__all__ = ["CoordMlpInflateResult", "apply_sidecar_to_rgb"]
