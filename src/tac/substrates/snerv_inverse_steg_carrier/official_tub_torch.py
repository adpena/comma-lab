# SPDX-License-Identifier: MIT
"""Torch training bridge for official SNeRV_T ``output_2`` fusion.

The receiver-safe primitive lives in ``official_tub.py`` and deliberately has
no Torch import markers. This module is for training/source-replay use only.
"""

from __future__ import annotations

from typing import Any

from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OfficialTubError,
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
    emb_ch = int(shape.emb_ch)
    decoder_input = torch.cat(
        (
            temporal_encoder_concat[:, :emb_ch, :, :],
            temporal_encoder_concat[:, emb_ch:, :, :],
        ),
        0,
    )
    fc_h, fc_w = shape.fc_hw
    out_n, _out_c, out_h, out_w = raw_shape
    fused = (
        decoder_output.reshape(out_n, -1, fc_h, fc_w, out_h, out_w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(out_n, -1, fc_h * out_h, fc_w * out_w)
    )
    return decoder_input, fused


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
