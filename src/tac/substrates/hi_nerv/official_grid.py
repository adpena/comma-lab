# SPDX-License-Identifier: MIT
"""Portable official HiNeRV ``GridTrilinear3D`` primitive.

Official HiNeRV does not sample local modulo grids the way the contest-local
``HinervSubstrate`` helper does.  Its ``GridTrilinear3D`` layer keeps spatial
resolution fixed and interpolates only along the temporal axis by reshaping the
``(T, H, W, C)`` feature grid to ``(1, 1, T, H*W*C)`` and calling
``torch.nn.functional.interpolate(..., mode="bilinear")``.

This module is the NumPy/portable receiver-side mirror of that exact source
contract.  It is useful for parity work and for future source-faithful HiNeRV
adapters, but it deliberately carries no score or promotion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF: Final[str] = (
    "official_hinerv_grid_trilinear3d_temporal_only_numpy_v1"
)
HINERV_OFFICIAL_GRID_TRILINEAR3D_SOURCE_CONTRACT: Final[str] = (
    "HiNeRV/models/layers.py GridTrilinear3D.forward: "
    "view(1,1,T,H*W*C)->F.interpolate(mode='bilinear')->view(output_size+(C,))"
)

FALSE_AUTHORITY: Final[dict[str, bool]] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class OfficialGridTrilinear3DError(ValueError):
    """Raised when the official HiNeRV grid interpolation contract is invalid."""


@dataclass(frozen=True)
class OfficialGridTrilinear3D:
    """NumPy mirror of official HiNeRV ``GridTrilinear3D``.

    ``output_size`` is ``(T_out, H, W)``.  Official source asserts that the
    input spatial dimensions already match ``H`` and ``W`` because only temporal
    scaling is allowed.
    """

    output_size: tuple[int, int, int]
    align_corners: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_size", _validate_output_size(self.output_size))
        object.__setattr__(self, "align_corners", bool(self.align_corners))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Interpolate ``x`` from ``(T,H,W,C)`` to ``output_size + (C,)``."""

        arr = _ensure_thwc(x)
        t_in, h_in, w_in, channels = (int(v) for v in arr.shape)
        t_out, h_out, w_out = self.output_size
        if h_in != h_out or w_in != w_out:
            raise OfficialGridTrilinear3DError(
                "official HiNeRV GridTrilinear3D only supports temporal scaling: "
                f"input spatial {(h_in, w_in)} != output spatial {(h_out, w_out)}"
            )
        if t_in == t_out:
            return arr.astype(np.float64, copy=True)
        flat = arr.reshape(t_in, h_in * w_in * channels)
        out = _interpolate_temporal_linear(
            flat,
            t_out=t_out,
            align_corners=self.align_corners,
        )
        return out.reshape(t_out, h_out, w_out, channels).astype(np.float64, copy=False)

    def as_jsonable_contract(self) -> dict[str, object]:
        """Return false-authority contract metadata for parity artifacts."""

        return {
            "schema": "hinerv_official_grid_trilinear3d_contract.v1",
            "proof_marker": HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF,
            "source_contract": HINERV_OFFICIAL_GRID_TRILINEAR3D_SOURCE_CONTRACT,
            "output_size": list(self.output_size),
            "align_corners": bool(self.align_corners),
            **FALSE_AUTHORITY,
        }


def official_grid_trilinear3d_forward(
    x: np.ndarray,
    *,
    output_size: tuple[int, int, int],
    align_corners: bool = False,
) -> np.ndarray:
    """Functional wrapper for the official temporal-only grid interpolation."""

    return OfficialGridTrilinear3D(
        output_size=output_size,
        align_corners=align_corners,
    ).forward(x)


def _interpolate_temporal_linear(
    flat: np.ndarray,
    *,
    t_out: int,
    align_corners: bool,
) -> np.ndarray:
    t_in = int(flat.shape[0])
    if t_in <= 0 or t_out <= 0:
        raise OfficialGridTrilinear3DError("temporal dimensions must be positive")
    if t_in == 1:
        return np.repeat(flat.astype(np.float64, copy=False), t_out, axis=0)
    if align_corners:
        if t_out == 1:
            source_positions = np.zeros((1,), dtype=np.float64)
        else:
            source_positions = np.arange(t_out, dtype=np.float64) * (
                float(t_in - 1) / float(t_out - 1)
            )
    else:
        scale = float(t_in) / float(t_out)
        source_positions = (np.arange(t_out, dtype=np.float64) + 0.5) * scale - 0.5
        source_positions = np.maximum(source_positions, 0.0)

    lo = np.floor(source_positions).astype(np.int64)
    hi = np.minimum(lo + 1, t_in - 1)
    lo = np.clip(lo, 0, t_in - 1)
    alpha = (source_positions - lo.astype(np.float64)).reshape(t_out, 1)
    return flat[lo].astype(np.float64) * (1.0 - alpha) + flat[hi].astype(np.float64) * alpha


def _ensure_thwc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 4:
        raise OfficialGridTrilinear3DError(f"expected (T,H,W,C) grid, got shape {arr.shape}")
    if any(int(v) <= 0 for v in arr.shape):
        raise OfficialGridTrilinear3DError(f"all grid dimensions must be positive, got {arr.shape}")
    return arr


def _validate_output_size(output_size: tuple[int, int, int]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in output_size)
    if len(values) != 3:
        raise OfficialGridTrilinear3DError(
            f"output_size must be (T,H,W), got {output_size!r}"
        )
    if any(value <= 0 for value in values):
        raise OfficialGridTrilinear3DError(
            f"output_size dimensions must be positive, got {values!r}"
        )
    return values


__all__ = [
    "FALSE_AUTHORITY",
    "HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF",
    "HINERV_OFFICIAL_GRID_TRILINEAR3D_SOURCE_CONTRACT",
    "OfficialGridTrilinear3D",
    "OfficialGridTrilinear3DError",
    "official_grid_trilinear3d_forward",
]
