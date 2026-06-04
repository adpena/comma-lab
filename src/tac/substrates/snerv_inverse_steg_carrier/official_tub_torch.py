# SPDX-License-Identifier: MIT
"""Torch training bridge for official SNeRV_T ``output_2`` fusion.

The receiver-safe primitive lives in ``official_tub.py`` and deliberately has
no Torch import markers. This module is for training/source-replay use only.
"""

from __future__ import annotations

from typing import Any

from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OfficialTubError,
    _apply_output2_fusion_backend,
    official_output2_fusion_shape,
)


def official_output2_fusion_torch(
    temporal_encoder_concat: Any,
    decoder_output: Any,
    *,
    fc_hw: tuple[int, int],
) -> tuple[Any, Any]:
    """Execute official ``output_2`` split/concat/shuffle with Torch tensors."""

    import torch

    temporal_shape = _torch_nchw_shape(
        temporal_encoder_concat,
        name="temporal_encoder_concat",
    )
    raw_shape = _torch_nchw_shape(decoder_output, name="decoder_output")
    shape = official_output2_fusion_shape(
        temporal_shape,
        fc_hw=fc_hw,
        decoder_output_shape=raw_shape,
    )
    return _apply_output2_fusion_backend(
        temporal_encoder_concat,
        decoder_output,
        shape=shape,
        concat=lambda parts, axis: torch.cat(tuple(parts), axis),
        reshape=lambda value, target_shape: value.reshape(*target_shape),
        transpose=lambda value, axes: value.permute(*axes),
    )


def _torch_nchw_shape(array: Any, *, name: str) -> tuple[int, int, int, int]:
    import torch

    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    if len(shape) != 4:
        raise OfficialTubError(f"{name} must be NCHW rank-4; got {shape}")
    if min(shape) < 1:
        raise OfficialTubError(f"{name} dims must be positive; got {shape}")
    if not bool(torch.isfinite(array).all().item()):
        raise OfficialTubError(f"{name} contains NaN or Inf")
    return shape


__all__ = ["official_output2_fusion_torch"]
