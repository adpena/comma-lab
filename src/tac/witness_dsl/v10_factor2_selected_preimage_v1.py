# SPDX-License-Identifier: MIT
"""Exact V10 factor-2 selected-preimage map for NumPy and differentiable MLX.

The public receiver's semantic operand is a scorer-grid ``uint8`` image.  Its
canonical camera-space witness copies each scorer byte to the four disjoint
bilinear support taps and leaves every unowned camera sample at zero.  This
module makes that same map available to the encoder/trainer without a host
round-trip in the differentiable path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
)

SCHEMA: Final = "tac.v10_factor2_selected_preimage.v1"
CAMERA_HW: Final = (874, 1164)
SCORER_HW: Final = (384, 512)
CHANNELS: Final = 3


class V10Factor2SelectedPreimageError(ValueError):
    """The exact factor-2 geometry or a typed operand failed closed."""


@dataclass(frozen=True, slots=True)
class Factor2GatherPlanV1:
    """Host and device gather state for one exact factor-2 geometry."""

    camera_hw: tuple[int, int]
    scorer_hw: tuple[int, int]
    flat_source_indices: np.ndarray
    valid_camera_samples: np.ndarray
    device_indices: Any
    device_valid: Any


def build_numpy_factor2_gather_plan(
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    scorer_hw: tuple[int, int] = SCORER_HW,
) -> tuple[DisjointResizeOperator, np.ndarray, np.ndarray]:
    """Derive the canonical sparse gather map from the certified operator."""

    if (
        type(camera_hw) is not tuple
        or len(camera_hw) != 2
        or type(scorer_hw) is not tuple
        or len(scorer_hw) != 2
        or any(type(value) is not int or value <= 0 for value in (*camera_hw, *scorer_hw))
    ):
        raise V10Factor2SelectedPreimageError("factor-2 geometry must be positive exact integer pairs")
    operator = DisjointResizeOperator.build(
        camera_h=camera_hw[0],
        camera_w=camera_hw[1],
        scorer_h=scorer_hw[0],
        scorer_w=scorer_hw[1],
    )
    indices = np.zeros(camera_hw[0] * camera_hw[1], dtype=np.int32)
    valid = np.zeros(camera_hw[0] * camera_hw[1], dtype=bool)
    for scorer_row, row_support in enumerate(operator.row_supports):
        for scorer_col, col_support in enumerate(operator.col_supports):
            source_index = scorer_row * scorer_hw[1] + scorer_col
            for camera_row in row_support.indices:
                for camera_col in col_support.indices:
                    flat = int(camera_row) * camera_hw[1] + int(camera_col)
                    if valid[flat]:
                        raise V10Factor2SelectedPreimageError(
                            "certified factor-2 supports unexpectedly overlap"
                        )
                    valid[flat] = True
                    indices[flat] = source_index
    expected_owned = scorer_hw[0] * scorer_hw[1] * 4
    if int(np.count_nonzero(valid)) != expected_owned:
        raise V10Factor2SelectedPreimageError(
            "certified factor-2 map does not own exactly four camera taps per scorer sample"
        )
    return operator, indices, valid


def build_mlx_factor2_gather_plan(
    *,
    mlx_module: Any,
    camera_hw: tuple[int, int] = CAMERA_HW,
    scorer_hw: tuple[int, int] = SCORER_HW,
) -> Factor2GatherPlanV1:
    """Build one reusable device gather plan; no video-specific state is stored."""

    if mlx_module is None:
        raise V10Factor2SelectedPreimageError("mlx_module is required")
    _operator, indices, valid = build_numpy_factor2_gather_plan(
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
    )
    return Factor2GatherPlanV1(
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        flat_source_indices=indices,
        valid_camera_samples=valid,
        device_indices=mlx_module.array(indices),
        device_valid=mlx_module.array(valid)[None, :, None],
    )


def realize_factor2_uint8_numpy(
    scorer_rgb: np.ndarray,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    scorer_hw: tuple[int, int] = SCORER_HW,
) -> np.ndarray:
    """Return the exact public V10 camera preimage for one typed scorer plane."""

    raw = np.asarray(scorer_rgb)
    if raw.dtype != np.uint8 or raw.shape != (*scorer_hw, CHANNELS):
        raise V10Factor2SelectedPreimageError(
            f"scorer operand must be uint8[{scorer_hw[0]},{scorer_hw[1]},{CHANNELS}]"
        )
    operator = DisjointResizeOperator.build(
        camera_h=camera_hw[0],
        camera_w=camera_hw[1],
        scorer_h=scorer_hw[0],
        scorer_w=scorer_hw[1],
    )
    try:
        result = realize_factor2_uint8_scorer_plane(operator, np.ascontiguousarray(raw))
    except ValueError as exc:
        raise V10Factor2SelectedPreimageError(
            "certified NumPy factor-2 realization failed"
        ) from exc
    return np.ascontiguousarray(result)


def realize_factor2_scorer_plane_mlx(
    scorer_rgb: Any,
    *,
    mlx_module: Any,
    plan: Factor2GatherPlanV1,
    ste_round: bool = True,
) -> Any:
    """Apply the exact sparse public map on-device with gather gradients.

    ``ste_round=True`` makes the forward operand the final scorer ``uint8``
    lattice value while preserving the identity gradient up to the sparse
    camera gather.  The returned array is floating point but every forward
    value is an exact integer in ``[0,255]``.
    """

    if mlx_module is None or type(plan) is not Factor2GatherPlanV1:
        raise V10Factor2SelectedPreimageError("typed MLX module and gather plan are required")
    shape = tuple(int(value) for value in scorer_rgb.shape)
    squeeze = False
    if shape == (*plan.scorer_hw, CHANNELS):
        source = mlx_module.reshape(scorer_rgb, (1, *shape))
        squeeze = True
    elif len(shape) == 4 and shape[1:] == (*plan.scorer_hw, CHANNELS):
        source = scorer_rgb
    else:
        raise V10Factor2SelectedPreimageError(
            "MLX scorer operand must be [H,W,3] or [N,H,W,3] at the bound scorer geometry"
        )
    if ste_round:
        rounded = mlx_module.clip(mlx_module.round(source), 0.0, 255.0)
        source = source + mlx_module.stop_gradient(rounded - source)
    flat = mlx_module.reshape(
        source,
        (shape[0] if len(shape) == 4 else 1, plan.scorer_hw[0] * plan.scorer_hw[1], CHANNELS),
    )
    gathered = mlx_module.take(flat, plan.device_indices, axis=1)
    camera = mlx_module.where(plan.device_valid, gathered, mlx_module.zeros_like(gathered))
    camera = mlx_module.reshape(
        camera,
        (flat.shape[0], plan.camera_hw[0], plan.camera_hw[1], CHANNELS),
    )
    return camera[0] if squeeze else camera
