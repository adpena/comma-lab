# SPDX-License-Identifier: MIT
"""Portable official SNeRV HFR head primitives.

Official SNeRV's high-frequency restoration is not the deterministic receiver
edge correction in ``carrier.py``.  The upstream source uses three learned
``ConvBlock`` heads fed by ``pyr_out``:

``1x1 Conv2d -> LeakyReLU(0.1) -> 3x3 Conv2d(pad=1)``

This module is NumPy-first and receiver-portable.  It is a real executable
primitive for train/export integration and parity testing, but it carries no
score or promotion authority by itself because trained weights, MFU ``pyr_out``
parity, and byte-closed archive wiring are separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF: Final[str] = (
    "official_snerv_hfr_convblock_three_heads_numpy_nchw"
)
OFFICIAL_SNERV_HFR_SOURCE_SHA: Final[str] = (
    "0844a08f9591eea9625f8b961ed91d08030e06d1"
)
OFFICIAL_SNERV_HFR_SOURCE_CONTRACT: Final[str] = (
    "official_snerv_lines_62_64_91_122_and_layers_144_160_hfr_contract"
)

FALSE_AUTHORITY: Final[dict[str, bool]] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class OfficialSnervHfrError(ValueError):
    """Raised when official HFR primitive tensors violate the source contract."""


@dataclass(frozen=True)
class OfficialConv2dNchw:
    """PyTorch-style ``nn.Conv2d`` weights in NCHW/OIHW layout."""

    weight: np.ndarray
    bias: np.ndarray | None = None
    padding: int = 0
    stride: int = 1

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64)
        bias = None if self.bias is None else np.asarray(self.bias, dtype=np.float64)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "padding", int(self.padding))
        object.__setattr__(self, "stride", int(self.stride))
        if weight.ndim != 4:
            raise OfficialSnervHfrError(f"conv weight must be OIHW, got {weight.shape}")
        if self.padding < 0:
            raise OfficialSnervHfrError("conv padding must be non-negative")
        if self.stride < 1:
            raise OfficialSnervHfrError("conv stride must be >= 1")
        if bias is not None and bias.shape != (weight.shape[0],):
            raise OfficialSnervHfrError(
                f"conv bias shape {bias.shape} does not match out channels {weight.shape[0]}"
            )

    @property
    def in_channels(self) -> int:
        return int(self.weight.shape[1])

    @property
    def out_channels(self) -> int:
        return int(self.weight.shape[0])

    def forward(self, x: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            x,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
        )

    def forward_mlx(
        self,
        x: Any,
        *,
        accumulation_mode: str = "fixed_fp32",
    ) -> Any:
        return conv2d_nchw_mlx(
            x,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
            accumulation_mode=accumulation_mode,
        )


@dataclass(frozen=True)
class OfficialHfrConvBlock:
    """Official SNeRV ``ConvBlock`` used by LH/HL/HH HFR heads."""

    conv1: OfficialConv2dNchw
    conv2: OfficialConv2dNchw
    act: str = "leaky01"

    def __post_init__(self) -> None:
        if self.conv1.weight.shape[2:] != (1, 1):
            raise OfficialSnervHfrError("official HFR conv1 kernel must be 1x1")
        if self.conv1.padding != 0 or self.conv1.stride != 1:
            raise OfficialSnervHfrError("official HFR conv1 must be 1x1 stride=1 padding=0")
        if self.conv2.weight.shape[2:] != (3, 3):
            raise OfficialSnervHfrError("official HFR conv2 kernel must be 3x3")
        if self.conv2.padding != 1 or self.conv2.stride != 1:
            raise OfficialSnervHfrError("official HFR conv2 must be 3x3 stride=1 padding=1")
        if self.conv2.in_channels != self.conv1.out_channels:
            raise OfficialSnervHfrError("official HFR conv2 input channels must match conv1 output")
        if self.conv2.out_channels != 3:
            raise OfficialSnervHfrError("official HFR head must output 3 RGB-detail channels")
        if self.act != "leaky01":
            raise OfficialSnervHfrError("official SNeRV HFR ConvBlock uses act='leaky01'")

    def forward(self, pyr_out: np.ndarray) -> np.ndarray:
        hidden = leaky_relu01(self.conv1.forward(pyr_out))
        return self.conv2.forward(hidden)

    def forward_mlx(
        self,
        pyr_out: Any,
        *,
        accumulation_mode: str = "fixed_fp32",
    ) -> Any:
        hidden = leaky_relu01_mlx(
            self.conv1.forward_mlx(pyr_out, accumulation_mode=accumulation_mode)
        )
        return self.conv2.forward_mlx(hidden, accumulation_mode=accumulation_mode)


@dataclass(frozen=True)
class OfficialHfrHeadsOutput:
    """Official three-subband HFR output layout."""

    lh: np.ndarray
    hl: np.ndarray
    hh: np.ndarray
    yh_out: np.ndarray

    def as_jsonable(self) -> dict[str, object]:
        return {
            "schema": "official_snerv_hfr_heads_output.v1",
            "lh_shape": list(self.lh.shape),
            "hl_shape": list(self.hl.shape),
            "hh_shape": list(self.hh.shape),
            "yh_out_shape": list(self.yh_out.shape),
            "torch_stack_dim": 2,
            **FALSE_AUTHORITY,
        }


@dataclass(frozen=True)
class OfficialHfrHeads:
    """Official SNeRV HFR: learned LH/HL/HH ConvBlock heads from ``pyr_out``."""

    lh_head: OfficialHfrConvBlock
    hl_head: OfficialHfrConvBlock
    hh_head: OfficialHfrConvBlock

    def __post_init__(self) -> None:
        in_channels = {
            self.lh_head.conv1.in_channels,
            self.hl_head.conv1.in_channels,
            self.hh_head.conv1.in_channels,
        }
        if len(in_channels) != 1:
            raise OfficialSnervHfrError("all official HFR heads must consume the same pyr_out channels")

    @property
    def in_channels(self) -> int:
        return int(self.lh_head.conv1.in_channels)

    def forward(self, pyr_out: np.ndarray) -> OfficialHfrHeadsOutput:
        x = _ensure_nchw(pyr_out)
        if x.shape[1] != self.in_channels:
            raise OfficialSnervHfrError(
                f"pyr_out channels {x.shape[1]} do not match HFR in_channels {self.in_channels}"
            )
        lh = self.lh_head.forward(x)
        hl = self.hl_head.forward(x)
        hh = self.hh_head.forward(x)
        yh_out = np.stack([lh, hl, hh], axis=2)
        return OfficialHfrHeadsOutput(lh=lh, hl=hl, hh=hh, yh_out=yh_out)

    def forward_mlx(
        self,
        pyr_out: Any,
        *,
        accumulation_mode: str = "fixed_fp32",
    ) -> OfficialHfrHeadsOutput:
        import mlx.core as mx

        x_shape = _ensure_nchw_shape(pyr_out)
        if int(x_shape[1]) != self.in_channels:
            raise OfficialSnervHfrError(
                f"pyr_out channels {x_shape[1]} do not match HFR in_channels {self.in_channels}"
            )
        lh = self.lh_head.forward_mlx(pyr_out, accumulation_mode=accumulation_mode)
        hl = self.hl_head.forward_mlx(pyr_out, accumulation_mode=accumulation_mode)
        hh = self.hh_head.forward_mlx(pyr_out, accumulation_mode=accumulation_mode)
        yh_out = mx.stack([lh, hl, hh], axis=2)
        return OfficialHfrHeadsOutput(lh=lh, hl=hl, hh=hh, yh_out=yh_out)


def conv2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """NumPy reference for PyTorch ``F.conv2d`` with NCHW input and OIHW weights."""

    x64 = _ensure_nchw(x)
    w64 = np.asarray(weight, dtype=np.float64)
    if w64.ndim != 4:
        raise OfficialSnervHfrError(f"weight must be OIHW, got {w64.shape}")
    if x64.shape[1] != w64.shape[1]:
        raise OfficialSnervHfrError(
            f"input channels {x64.shape[1]} do not match weight channels {w64.shape[1]}"
        )
    pad = int(padding)
    step = int(stride)
    if pad < 0:
        raise OfficialSnervHfrError("padding must be non-negative")
    if step < 1:
        raise OfficialSnervHfrError("stride must be >= 1")
    kh, kw = int(w64.shape[2]), int(w64.shape[3])
    if x64.shape[2] + 2 * pad < kh or x64.shape[3] + 2 * pad < kw:
        raise OfficialSnervHfrError("kernel is larger than padded input")
    padded = np.pad(
        x64,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kh, kw),
        axis=(2, 3),
    )
    windows = windows[:, :, ::step, ::step, :, :]
    out = np.einsum("nchwkl,ockl->nohw", windows, w64, optimize=True)
    if bias is not None:
        b64 = np.asarray(bias, dtype=np.float64)
        if b64.shape != (w64.shape[0],):
            raise OfficialSnervHfrError(
                f"bias shape {b64.shape} does not match out channels {w64.shape[0]}"
            )
        out = out + b64[None, :, None, None]
    return out.astype(np.float64, copy=False)


def conv2d_nchw_mlx(
    x: Any,
    weight: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    padding: int = 0,
    stride: int = 1,
    accumulation_mode: str = "fixed_fp32",
) -> Any:
    """MLX implementation of PyTorch-style NCHW/OIHW Conv2d.

    The default ``fixed_fp32`` path delegates to the canonical MLX scorer
    reference conv so reproduction favors deterministic fixed-order arithmetic.
    ``optimized`` remains available for MLX training throughput.  The returned
    tensor stays NCHW so PyTorch/NumPy/MLX call sites share one official HFR
    layout contract.
    """

    import mlx.core as mx

    x_shape = _ensure_nchw_shape(x)
    w64 = np.asarray(weight, dtype=np.float64)
    if w64.ndim != 4:
        raise OfficialSnervHfrError(f"weight must be OIHW, got {w64.shape}")
    if int(x_shape[1]) != int(w64.shape[1]):
        raise OfficialSnervHfrError(
            f"input channels {x_shape[1]} do not match weight channels {w64.shape[1]}"
        )
    b64 = None if bias is None else np.asarray(bias, dtype=np.float64)
    if b64 is not None and b64.shape != (w64.shape[0],):
        raise OfficialSnervHfrError(
            f"bias shape {b64.shape} does not match out channels {w64.shape[0]}"
        )
    x_mx = mx.array(x)
    weight_ohwi = mx.array(np.transpose(w64, (0, 2, 3, 1)))
    x_nhwc = mx.transpose(x_mx, (0, 2, 3, 1))
    if str(accumulation_mode) == "optimized":
        out_nhwc = mx.conv2d(
            x_nhwc,
            weight_ohwi,
            stride=(int(stride), int(stride)),
            padding=(int(padding), int(padding)),
        )
    else:
        from tac.local_acceleration.mlx_scorer_adapters import (
            mlx_reference_conv2d_nhwc,
        )

        out_nhwc = mlx_reference_conv2d_nhwc(
            x_nhwc,
            weight_ohwi,
            stride=(int(stride), int(stride)),
            padding=(int(padding), int(padding)),
            accumulation_mode=str(accumulation_mode),
        )
    if b64 is not None:
        out_nhwc = out_nhwc + mx.reshape(mx.array(b64), (1, 1, 1, int(b64.shape[0])))
    return mx.transpose(out_nhwc, (0, 3, 1, 2))


def leaky_relu01(x: np.ndarray) -> np.ndarray:
    """Official ``act='leaky01'`` activation."""

    x64 = np.asarray(x, dtype=np.float64)
    return np.where(x64 >= 0.0, x64, 0.1 * x64).astype(np.float64, copy=False)


def leaky_relu01_mlx(x: Any) -> Any:
    """MLX official ``act='leaky01'`` activation."""

    import mlx.core as mx

    return mx.where(x >= 0.0, x, 0.1 * x)


def _ensure_nchw(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    _ensure_nchw_shape(arr)
    return arr


def _ensure_nchw_shape(x: Any) -> tuple[int, int, int, int]:
    shape = tuple(int(v) for v in getattr(x, "shape", ()))
    if len(shape) != 4:
        raise OfficialSnervHfrError(f"expected NCHW tensor, got {shape}")
    return shape


__all__ = [
    "FALSE_AUTHORITY",
    "OFFICIAL_SNERV_HFR_SOURCE_CONTRACT",
    "OFFICIAL_SNERV_HFR_SOURCE_SHA",
    "SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF",
    "OfficialConv2dNchw",
    "OfficialHfrConvBlock",
    "OfficialHfrHeads",
    "OfficialHfrHeadsOutput",
    "OfficialSnervHfrError",
    "conv2d_nchw",
    "conv2d_nchw_mlx",
    "leaky_relu01",
    "leaky_relu01_mlx",
]
