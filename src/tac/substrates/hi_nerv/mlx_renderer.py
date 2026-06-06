# SPDX-License-Identifier: MIT
"""MLX-native HiNeRV renderer bridge.

This is the first real HiNeRV adapter surface for the MLX score-aware harness:
an MLX module that mirrors :mod:`tac.substrates.hi_nerv.architecture`, exposes
the canonical ``call_b2chw_255`` forward convention, and exports PyTorch-layout
state_dict tensors consumable by the existing HIV1 archive grammar.

Online reference anchors checked before this port:

* official HiNeRV repository ``hmkx/HiNeRV`` at HEAD
  ``fdb92ec22492246f800621dfd454f6a5c62ab75b``;
* official SNeRV repository ``qwertja/SNeRV`` at HEAD
  ``0844a08f9591eea9625f8b961ed91d08030e06d1``;
* official HNeRV repository ``haochen-rye/HNeRV`` at HEAD
  ``4872129c8d004a25477e0c1ffbbff4ba71943ad5``.

No third-party source is vendored here. The implementation follows this repo's
local PyTorch HiNeRV grammar so archive/export/runtime contracts remain ours.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tac.substrates.hi_nerv.architecture import HinervConfig

try:  # pragma: no cover - exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
except Exception as exc:  # pragma: no cover - non-Apple CI import guard.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    tree_unflatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "hi_nerv_mlx_renderer_v1"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"
_MIN_POSITIVE_FP16_SCALE = 5.960464477539063e-08
HI_NERV_DECODER_FAKE_QUANT_ACTION_BITS: tuple[int, ...] = (
    0,
    2,
    4,
    6,
    7,
    8,
    16,
    32,
)


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError(
            "MLX is not available on this host; the HiNeRV MLX renderer "
            "requires Apple Silicon with the mlx package installed. Original "
            f"import error: {_MLX_IMPORT_ERROR!r}"
        )


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc

    return pixel_shuffle_2x_nhwc(x)


def _bilinear_resize_nhwc(x: Any, target_h: int, target_w: int) -> Any:
    _require_mlx()
    src_h, src_w = int(x.shape[1]), int(x.shape[2])
    if src_h == target_h and src_w == target_w:
        return x
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
        bilinear_resize_nhwc,
    )

    if src_h * 2 == target_h and src_w * 2 == target_w:
        return bilinear_resize2x_align_corners_false_nhwc(x)
    return bilinear_resize_nhwc(
        x,
        target_h=int(target_h),
        target_w=int(target_w),
        align_corners=False,
    )


def _siren_uniform_bound(fan_in: int, w: float) -> float:
    return math.sqrt(6.0 / max(int(fan_in), 1)) / max(float(w), 1.0)


def _fake_quant_symmetric_ste(values: Any, *, bits: int) -> Any:
    """Archive-aligned symmetric fake quantization with STE.

    Decoder archive codecs use signed symmetric integer values with fp16 scales:
    per-output-channel (axis 0) for matrix/conv tensors and per-tensor for
    vectors.  This forward proxy mirrors that geometry so score-aware training
    sees the receiver surface it is being pushed toward, while the hard archive
    byte oracle still decides promotion.
    """

    _require_mlx()
    levels = max(1, (1 << (int(bits) - 1)) - 1)
    abs_values = mx.abs(values)  # type: ignore[union-attr]
    if len(values.shape) >= 2 and int(values.shape[0]) > 1:
        reduce_axes = tuple(range(1, len(values.shape)))
        abs_max = mx.max(abs_values, axis=reduce_axes, keepdims=True)  # type: ignore[union-attr]
    else:
        abs_max = mx.max(abs_values)  # type: ignore[union-attr]
    raw_scale = abs_max / float(levels)
    scale32 = mx.where(  # type: ignore[union-attr]
        abs_max > 0.0,
        mx.maximum(raw_scale, _MIN_POSITIVE_FP16_SCALE),  # type: ignore[union-attr]
        1.0,
    )
    scale = mx.stop_gradient(scale32.astype(mx.float16).astype(mx.float32))  # type: ignore[union-attr]
    q = mx.round(values / scale)  # type: ignore[union-attr]
    q = mx.clip(q, -float(levels), float(levels))  # type: ignore[union-attr]
    dequant = q * scale
    return values + mx.stop_gradient(dequant - values)  # type: ignore[union-attr]


def _apply_fake_quant_bits(values: Any, *, bits: int | None) -> Any:
    _require_mlx()
    if bits is None or int(bits) >= 32:
        return values
    if int(bits) == 0:
        return values + mx.stop_gradient(mx.zeros_like(values) - values)  # type: ignore[union-attr]
    return _fake_quant_symmetric_ste(values, bits=int(bits))


def _resolve_fake_quant_bits(
    *,
    name: str | None,
    fake_quant_bits: int | None,
    fake_quant_bits_by_name: Mapping[str, int] | None,
) -> int | None:
    if name and fake_quant_bits_by_name and name in fake_quant_bits_by_name:
        bits = int(fake_quant_bits_by_name[name])
        return None if bits >= 32 else bits
    return fake_quant_bits


def _linear_with_params(
    layer: Any,
    x: Any,
    *,
    fake_quant_bits: int | None,
    fake_quant_bits_by_name: Mapping[str, int] | None = None,
    weight_name: str | None = None,
    bias_name: str | None = None,
) -> Any:
    _require_mlx()
    weight = layer.weight
    bias = layer.bias
    weight = _apply_fake_quant_bits(
        weight,
        bits=_resolve_fake_quant_bits(
            name=weight_name,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
        ),
    )
    bias = _apply_fake_quant_bits(
        bias,
        bits=_resolve_fake_quant_bits(
            name=bias_name,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
        ),
    )
    return x @ mx.transpose(weight) + bias  # type: ignore[union-attr]


def _conv2d_with_params(
    layer: Any,
    x: Any,
    *,
    fake_quant_bits: int | None,
    fake_quant_bits_by_name: Mapping[str, int] | None = None,
    weight_name: str | None = None,
    bias_name: str | None = None,
) -> Any:
    _require_mlx()
    weight = layer.weight
    bias = layer.bias
    weight = _apply_fake_quant_bits(
        weight,
        bits=_resolve_fake_quant_bits(
            name=weight_name,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
        ),
    )
    bias = _apply_fake_quant_bits(
        bias,
        bits=_resolve_fake_quant_bits(
            name=bias_name,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
        ),
    )
    padding = getattr(layer, "padding", 0)
    stride = getattr(layer, "stride", 1)
    dilation = getattr(layer, "dilation", 1)
    groups = getattr(layer, "groups", 1)
    return mx.conv2d(
        x,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ) + bias  # type: ignore[union-attr]


def _layer_norm_with_params(
    layer: Any,
    x: Any,
    *,
    fake_quant_bits: int | None,
    fake_quant_bits_by_name: Mapping[str, int] | None = None,
    weight_name: str | None = None,
    bias_name: str | None = None,
) -> Any:
    _require_mlx()
    weight = _apply_fake_quant_bits(
        layer.weight,
        bits=_resolve_fake_quant_bits(
            name=weight_name,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
        ),
    )
    bias = _apply_fake_quant_bits(
        layer.bias,
        bits=_resolve_fake_quant_bits(
            name=bias_name,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
        ),
    )
    eps = float(getattr(layer, "eps", 1.0e-5))
    mean = mx.mean(x, axis=-1, keepdims=True)  # type: ignore[union-attr]
    centered = x - mean
    var = mx.mean(centered * centered, axis=-1, keepdims=True)  # type: ignore[union-attr]
    return centered * mx.rsqrt(var + eps) * weight + bias  # type: ignore[union-attr]


def trilinear_upsample_mlx(
    grid: Any,
    pair_indices: Any,
    *,
    num_pairs: int,
    target_h: int,
    target_w: int,
    local_scale: int,
) -> Any:
    """MLX mirror of the receiver-visible HiNeRV temporal-local grid sampler."""

    _require_mlx()
    if len(grid.shape) != 4:
        raise ValueError(f"grid must be (T,S,S,C), got {tuple(grid.shape)}")
    time_bins, scale_h, scale_w, channels = (int(v) for v in grid.shape)
    if scale_h != int(local_scale) or scale_w != int(local_scale):
        raise ValueError(
            f"grid local scale {(scale_h, scale_w)} != {int(local_scale)}"
        )
    if time_bins <= 0 or channels <= 0:
        raise ValueError("grid must have positive temporal bins and channels")

    denom = max(int(num_pairs) - 1, 1)
    t = pair_indices.astype(mx.float32) * (float(time_bins - 1) / float(denom))  # type: ignore[union-attr]
    t0 = mx.floor(t).astype(mx.int32)  # type: ignore[union-attr]
    t0 = mx.clip(t0, 0, time_bins - 1)  # type: ignore[union-attr]
    t1 = mx.clip(t0 + 1, 0, time_bins - 1)  # type: ignore[union-attr]
    alpha = (t - t0.astype(mx.float32)).reshape((-1, 1, 1, 1))  # type: ignore[union-attr]
    temporal = mx.take(grid, t0, axis=0) * (1.0 - alpha) + mx.take(grid, t1, axis=0) * alpha  # type: ignore[union-attr]

    ys = mx.arange(int(target_h), dtype=mx.int32) % int(local_scale)  # type: ignore[union-attr]
    xs = mx.arange(int(target_w), dtype=mx.int32) % int(local_scale)  # type: ignore[union-attr]
    flat_local = (ys[:, None] * int(local_scale) + xs[None, :]).reshape((-1,))
    flat = temporal.reshape((int(pair_indices.shape[0]), int(local_scale) ** 2, channels))
    sampled = flat[:, flat_local, :]
    return sampled.reshape((int(pair_indices.shape[0]), int(target_h), int(target_w), channels))


class HierarchicalFeatureGridMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX temporal-local feature grid, exported as PyTorch receiver tensors."""

    def __init__(
        self,
        *,
        num_pairs: int,
        local_scale: int,
        base_channels: int,
        levels: int,
        out_channels: int,
        reduction: int,
    ) -> None:
        _require_mlx()
        super().__init__()
        if levels <= 0:
            raise ValueError("local_grid_levels must be positive")
        if local_scale <= 0:
            raise ValueError("local_scale must be positive")
        self.num_pairs = int(num_pairs)
        self.local_scale = int(local_scale)
        self.levels = int(levels)
        self.base_channels = int(base_channels)
        self.level_channels: list[int] = []
        grids: list[Any] = []
        for level in range(self.levels):
            time_bins = max(2, math.ceil(self.num_pairs / float(2**level)))
            channels = max(1, (self.base_channels * (2**level)) // max(1, reduction))
            self.level_channels.append(int(channels))
            grids.append(
                mx.random.normal(  # type: ignore[union-attr]
                    shape=(time_bins, self.local_scale, self.local_scale, channels)
                )
                * 0.02
            )
        self.grids = grids
        self.proj: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=sum(self.level_channels),
            out_channels=int(out_channels),
            kernel_size=1,
        )

    def __call__(
        self,
        pair_indices: Any,
        *,
        spatial_shape: tuple[int, int],
        fake_quant_bits: int | None = None,
        fake_quant_bits_by_name: Mapping[str, int] | None = None,
        name_prefix: str = "feature_grids",
    ) -> Any:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        sampled = [
            trilinear_upsample_mlx(
                _apply_fake_quant_bits(
                    grid,
                    bits=_resolve_fake_quant_bits(
                        name=f"{name_prefix}.grids.{level}",
                        fake_quant_bits=fake_quant_bits,
                        fake_quant_bits_by_name=fake_quant_bits_by_name,
                    ),
                ),
                pair_indices,
                num_pairs=self.num_pairs,
                target_h=h,
                target_w=w,
                local_scale=self.local_scale,
            )
            for level, grid in enumerate(self.grids)
        ]
        enc = mx.concatenate(sampled, axis=-1)  # type: ignore[union-attr]
        return _conv2d_with_params(
            self.proj,
            enc,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name=f"{name_prefix}.proj.weight",
            bias_name=f"{name_prefix}.proj.bias",
        )


class ConvNeXtBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX ConvNeXt-style depthwise block matching PyTorch receiver layout."""

    def __init__(self, channels: int, *, mlp_ratio: int, kernel_size: int) -> None:
        _require_mlx()
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("convnext_kernel_size must be positive and odd")
        hidden = max(channels, channels * max(1, int(mlp_ratio)))
        self.channels = channels
        self.dwconv: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm: Any = nn.LayerNorm(channels)  # type: ignore[union-attr]
        self.pwconv1: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=channels,
            out_channels=hidden,
            kernel_size=1,
        )
        self.pwconv2: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=hidden,
            out_channels=channels,
            kernel_size=1,
        )
        self.gamma = mx.full((channels, 1, 1), 1.0e-3)  # type: ignore[union-attr]
        self.act: Any = nn.GELU()  # type: ignore[union-attr]

    def __call__(
        self,
        x: Any,
        *,
        fake_quant_bits: int | None = None,
        fake_quant_bits_by_name: Mapping[str, int] | None = None,
        name_prefix: str = "convnext_blocks",
    ) -> Any:
        residual = x
        y = _conv2d_with_params(
            self.dwconv,
            x,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name=f"{name_prefix}.dwconv.weight",
            bias_name=f"{name_prefix}.dwconv.bias",
        )
        y = _layer_norm_with_params(
            self.norm,
            y,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name=f"{name_prefix}.norm.weight",
            bias_name=f"{name_prefix}.norm.bias",
        )
        y = self.act(
            _conv2d_with_params(
                self.pwconv1,
                y,
                fake_quant_bits=fake_quant_bits,
                fake_quant_bits_by_name=fake_quant_bits_by_name,
                weight_name=f"{name_prefix}.pwconv1.weight",
                bias_name=f"{name_prefix}.pwconv1.bias",
            )
        )
        y = _conv2d_with_params(
            self.pwconv2,
            y,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name=f"{name_prefix}.pwconv2.weight",
            bias_name=f"{name_prefix}.pwconv2.bias",
        )
        gamma = _apply_fake_quant_bits(
            self.gamma,
            bits=_resolve_fake_quant_bits(
                name=f"{name_prefix}.gamma",
                fake_quant_bits=fake_quant_bits,
                fake_quant_bits_by_name=fake_quant_bits_by_name,
            ),
        )
        gamma_nhwc = mx.transpose(gamma, (1, 2, 0)).reshape((1, 1, 1, self.channels))  # type: ignore[union-attr]
        return residual + gamma_nhwc * y


class _UpBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX Conv2d -> sin -> PixelShuffle(2), matching PyTorch ``_UpBlock``."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.w = float(sin_freq)
        self.conv: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=int(in_ch),
            out_channels=int(out_ch) * 4,
            kernel_size=3,
            padding=1,
        )

    def __call__(
        self,
        x: Any,
        *,
        fake_quant_bits: int | None = None,
        fake_quant_bits_by_name: Mapping[str, int] | None = None,
        name_prefix: str = "blocks",
    ) -> Any:
        conv = _conv2d_with_params(
            self.conv,
            x,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name=f"{name_prefix}.conv.weight",
            bias_name=f"{name_prefix}.conv.bias",
        )
        return _pixel_shuffle_2x_nhwc(mx.sin(self.w * conv))  # type: ignore[union-attr]


class _LatentInjectorMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Project per-pair latent into a spatial additive NHWC tensor."""

    def __init__(self, latent_dim: int, channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.channels = int(channels)
        self.proj: Any = nn.Linear(int(latent_dim), int(channels))  # type: ignore[union-attr]

    def __call__(
        self,
        latent: Any,
        spatial_shape: tuple[int, int],
        *,
        fake_quant_bits: int | None = None,
        fake_quant_bits_by_name: Mapping[str, int] | None = None,
        name_prefix: str = "latent_injector.proj",
    ) -> Any:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        v = _linear_with_params(
            self.proj,
            latent,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name=f"{name_prefix}.weight",
            bias_name=f"{name_prefix}.bias",
        )
        v = mx.reshape(v, (-1, 1, 1, self.channels))  # type: ignore[union-attr]
        return mx.broadcast_to(v, (int(v.shape[0]), h, w, self.channels))  # type: ignore[union-attr]


class HinervSubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native mirror of :class:`HinervSubstrate`.

    Forward returns ``(B, 2, 3, H, W)`` in ``[0, 255]`` for the shared MLX
    score-aware harness.
    """

    def __init__(self, cfg: HinervConfig) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        if not 0 <= int(cfg.mid_injection_block_index) < int(cfg.num_upsample_blocks):
            raise ValueError("mid_injection_block_index out of range")
        if not 0 <= int(cfg.fine_injection_block_index) < int(cfg.num_upsample_blocks):
            raise ValueError("fine_injection_block_index out of range")
        if int(cfg.fine_injection_block_index) <= int(cfg.mid_injection_block_index):
            raise ValueError(
                "fine_injection_block_index must be > mid_injection_block_index"
            )
        mx.random.seed(int(getattr(cfg, "init_seed", 0)))  # type: ignore[union-attr]

        self.latents_coarse = mx.random.normal(  # type: ignore[union-attr]
            shape=(int(cfg.num_pairs), int(cfg.latent_dim_coarse))
        ) * 0.02
        self.latents_mid = mx.random.normal(  # type: ignore[union-attr]
            shape=(int(cfg.num_pairs), int(cfg.latent_dim_mid))
        ) * 0.02
        self.latents_fine = mx.random.normal(  # type: ignore[union-attr]
            shape=(int(cfg.num_pairs), int(cfg.latent_dim_fine))
        ) * 0.02

        self.latent_embed: Any = nn.Linear(  # type: ignore[union-attr]
            int(cfg.latent_dim_coarse),
            int(cfg.embed_dim) * int(cfg.initial_grid_h) * int(cfg.initial_grid_w),
        )
        channels = [int(cfg.embed_dim), *[int(value) for value in cfg.decoder_channels]]
        if len(channels) <= int(cfg.num_upsample_blocks):
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        self.blocks: list[Any] = [
            _UpBlockMLX(channels[i], channels[i + 1], float(cfg.sin_frequency))
            for i in range(int(cfg.num_upsample_blocks))
        ]
        self.feature_grids: list[Any] = []
        self.convnext_blocks: list[Any] = []
        for i in range(int(cfg.num_upsample_blocks)):
            out_ch = channels[i + 1]
            if bool(cfg.use_hierarchical_feature_grid):
                self.feature_grids.append(
                    HierarchicalFeatureGridMLX(
                        num_pairs=int(cfg.num_pairs),
                        local_scale=2,
                        base_channels=int(cfg.local_grid_channels),
                        levels=int(cfg.local_grid_levels),
                        out_channels=out_ch,
                        reduction=max(1, i + 1),
                    )
                )
            else:
                self.feature_grids.append(None)
            if bool(cfg.use_convnext_blocks):
                self.convnext_blocks.append(
                    ConvNeXtBlockMLX(
                        out_ch,
                        mlp_ratio=int(cfg.convnext_mlp_ratio),
                        kernel_size=int(cfg.convnext_kernel_size),
                    )
                )
            else:
                self.convnext_blocks.append(None)
        self.mid_injector = _LatentInjectorMLX(
            int(cfg.latent_dim_mid),
            channels[int(cfg.mid_injection_block_index) + 1],
        )
        self.fine_injector = _LatentInjectorMLX(
            int(cfg.latent_dim_fine),
            channels[int(cfg.fine_injection_block_index) + 1],
        )
        final_ch = channels[int(cfg.num_upsample_blocks)]
        self.head_rgb_0: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=3, padding=1
        )
        self.head_rgb_1: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=3, padding=1
        )
        self.decoder_fake_quant_forward_enabled = False
        self.decoder_fake_quant_forward_configured_enabled = False
        self.decoder_fake_quant_forward_stage_controlled = False
        self.decoder_fake_quant_forward_stage_qat_active = True
        self.decoder_fake_quant_forward_last_stage: dict[str, Any] = {}
        self.decoder_fake_quant_bits: int | None = 8
        self.decoder_fake_quant_bits_by_name: dict[str, int] = {}
        self._siren_init()

    def configure_decoder_fake_quant_forward(
        self,
        *,
        enabled: bool,
        quant_bits: int | None = 8,
        per_tensor_bits: Mapping[str, int] | None = None,
        stage_controlled: bool = False,
    ) -> None:
        """Enable decoder-weight fake-quant forward QAT for training.

        This affects forward computation only; archive/export still measures
        real packet bytes independently.  ``stage_controlled`` keeps the
        configured quantizer geometry resident while allowing PR95/canonical
        curriculum stages to activate it only during QAT phases.
        """

        bits = None if quant_bits is None else int(quant_bits)
        if bits is not None and (bits < 1 or bits > 16):
            raise ValueError(f"quant_bits must be in [1, 16]; got {quant_bits}")
        normalized: dict[str, int] = {}
        for name, value in dict(per_tensor_bits or {}).items():
            tensor_bits = int(value)
            if tensor_bits not in set(HI_NERV_DECODER_FAKE_QUANT_ACTION_BITS):
                raise ValueError(
                    "per_tensor_bits values must be one of "
                    f"{list(HI_NERV_DECODER_FAKE_QUANT_ACTION_BITS)}; "
                    f"got {value!r} for {name!r}"
                )
            normalized[str(name)] = tensor_bits
        self.decoder_fake_quant_forward_configured_enabled = bool(enabled)
        self.decoder_fake_quant_forward_stage_controlled = bool(stage_controlled)
        self.decoder_fake_quant_forward_stage_qat_active = not bool(stage_controlled)
        self.decoder_fake_quant_bits = bits
        self.decoder_fake_quant_bits_by_name = normalized
        self.decoder_fake_quant_forward_enabled = bool(enabled) and (
            not bool(stage_controlled)
        )

    def _set_decoder_fake_quant_stage_active(
        self,
        *,
        active: bool,
        stage_name: str,
        stage_epoch: int | None = None,
        stage_index: int | None = None,
        source: str,
    ) -> dict[str, Any]:
        active_bool = bool(active)
        self.decoder_fake_quant_forward_stage_qat_active = active_bool
        self.decoder_fake_quant_forward_enabled = (
            bool(self.decoder_fake_quant_forward_configured_enabled)
            and (active_bool or not bool(self.decoder_fake_quant_forward_stage_controlled))
        )
        self.decoder_fake_quant_forward_last_stage = {
            "schema": "hi_nerv_decoder_fake_quant_stage_control.v1",
            "source": str(source),
            "stage_name": str(stage_name),
            "stage_epoch": None if stage_epoch is None else int(stage_epoch),
            "stage_index": None if stage_index is None else int(stage_index),
            "stage_qat_active": active_bool,
            "stage_controlled": bool(self.decoder_fake_quant_forward_stage_controlled),
            "configured_enabled": bool(
                self.decoder_fake_quant_forward_configured_enabled
            ),
            "forward_active": bool(self.decoder_fake_quant_forward_enabled),
            "global_quant_bits": (
                None
                if self.decoder_fake_quant_bits is None
                else int(self.decoder_fake_quant_bits)
            ),
            "per_tensor_group_count": len(self.decoder_fake_quant_bits_by_name),
        }
        return dict(self.decoder_fake_quant_forward_last_stage)

    def notify_curriculum_stage(self, global_epoch: int, stage: Any) -> None:
        """Activate/deactivate configured fake quant from canonical stages."""

        if not bool(self.decoder_fake_quant_forward_stage_controlled):
            return
        self._set_decoder_fake_quant_stage_active(
            active=bool(getattr(stage, "enable_qat", False)),
            stage_name=str(getattr(stage, "name", "")),
            stage_epoch=int(global_epoch),
            source="canonical_curriculum_stage",
        )

    def notify_pr95_stage_verdict(self, global_epoch: int, verdict: Any) -> None:
        """Activate/deactivate configured fake quant from PR95 stage verdicts."""

        if not bool(self.decoder_fake_quant_forward_stage_controlled):
            return
        self._set_decoder_fake_quant_stage_active(
            active=bool(getattr(verdict, "qat_active", False)),
            stage_name=str(getattr(verdict, "descriptor_id", "")),
            stage_epoch=int(global_epoch),
            stage_index=int(getattr(verdict, "stage_index", 0) or 0),
            source="pr95_faithful_stage_verdict",
        )

    def configure_decoder_fake_quant_forward_from_waterfill_plan(
        self,
        decoder_weight_waterfill_plan: Mapping[str, Any],
        *,
        fallback_quant_bits: int | None = None,
    ) -> dict[str, Any]:
        """Bind shared decoder waterfill selections into train-time QAT."""

        from tac.substrates.hi_nerv.bitstream import (
            build_decoder_waterfill_fake_quant_forward_plan,
        )

        report = build_decoder_waterfill_fake_quant_forward_plan(
            decoder_weight_waterfill_plan
        )
        actuation_blockers = [
            str(blocker) for blocker in report.get("actuation_blockers") or []
        ]
        if actuation_blockers:
            raise ValueError(
                "decoder_weight_waterfill_plan is not safe for train-time "
                f"fake quantization: {actuation_blockers}"
            )
        per_tensor_bits = {
            str(name): int(bits)
            for name, bits in dict(report.get("per_tensor_bits") or {}).items()
        }
        enabled = bool(per_tensor_bits) or fallback_quant_bits is not None
        self.configure_decoder_fake_quant_forward(
            enabled=enabled,
            quant_bits=fallback_quant_bits,
            per_tensor_bits=per_tensor_bits,
            stage_controlled=False,
        )
        return {
            **report,
            "configured": bool(enabled),
            "fallback_quant_bits": (
                None if fallback_quant_bits is None else int(fallback_quant_bits)
            ),
            "configured_global_quant_bits": (
                None if self.decoder_fake_quant_bits is None else int(self.decoder_fake_quant_bits)
            ),
            "configured_per_tensor_bits": dict(self.decoder_fake_quant_bits_by_name),
        }

    def _fake_quant_bits(self) -> int | None:
        return (
            int(self.decoder_fake_quant_bits)
            if bool(self.decoder_fake_quant_forward_enabled)
            and self.decoder_fake_quant_bits is not None
            else None
        )

    def _fake_quant_bits_by_name(self) -> dict[str, int]:
        if not bool(self.decoder_fake_quant_forward_enabled):
            return {}
        return dict(self.decoder_fake_quant_bits_by_name)

    def _siren_init(self) -> None:
        w = float(self.cfg.sin_frequency)
        bound = _siren_uniform_bound(int(self.cfg.latent_dim_coarse), w)
        self.latent_embed.update({
            "weight": mx.random.uniform(  # type: ignore[union-attr]
                low=-bound,
                high=bound,
                shape=self.latent_embed.weight.shape,
            ),
            "bias": mx.zeros_like(self.latent_embed.bias),  # type: ignore[union-attr]
        })
        for block in self.blocks:
            conv = block.conv
            kh, kw = int(conv.weight.shape[1]), int(conv.weight.shape[2])
            fan_in = int(conv.weight.shape[3]) * kh * kw
            bound = _siren_uniform_bound(fan_in, w)
            conv.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound,
                    high=bound,
                    shape=conv.weight.shape,
                ),
                "bias": mx.zeros_like(conv.bias),  # type: ignore[union-attr]
            })
        for injector in (self.mid_injector, self.fine_injector):
            bound = _siren_uniform_bound(injector.latent_dim, w)
            injector.proj.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound,
                    high=bound,
                    shape=injector.proj.weight.shape,
                ),
                "bias": mx.zeros_like(injector.proj.bias),  # type: ignore[union-attr]
            })
        for head in (self.head_rgb_0, self.head_rgb_1):
            kh, kw = int(head.weight.shape[1]), int(head.weight.shape[2])
            fan_in = int(head.weight.shape[3]) * kh * kw
            bound = _siren_uniform_bound(fan_in, w)
            head.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound,
                    high=bound,
                    shape=head.weight.shape,
                ),
                "bias": mx.zeros_like(head.bias),  # type: ignore[union-attr]
            })

    def initialize_output_head_bias_from_targets(
        self,
        target_rgb_0: Any,
        target_rgb_1: Any,
        *,
        epsilon: float = 1.0 / 1024.0,
    ) -> dict[str, Any]:
        """Initialize sigmoid RGB-head biases to the target per-channel means.

        HiNeRV's sigmoid heads start at ``sigmoid(0) * 255 = 127.5`` when the
        bias is zero.  The contest video's scorer-resolution frames are much
        darker, so a zero-bias launch spends early optimization just moving the
        output into the right value domain.  This deterministic compression-time
        initialization solves the constant-color optimum in closed form while
        charging the result as ordinary archive decoder bias bytes.
        """

        def _bias_for_target(target_rgb: Any) -> tuple[Any, Any]:
            target = mx.array(target_rgb).astype(mx.float32)  # type: ignore[union-attr]
            if target.ndim != 4 or int(target.shape[-1]) != 3:
                raise ValueError(
                    "target RGB tensors must be NHWC with 3 channels; got "
                    f"shape={tuple(int(v) for v in target.shape)}"
                )
            eps = float(epsilon)
            if not math.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
                raise ValueError(f"epsilon must be finite in (0, 0.5); got {epsilon}")
            mean = mx.mean(mx.clip(target, eps, 1.0 - eps), axis=(0, 1, 2))  # type: ignore[union-attr]
            bias = mx.log(mean / (1.0 - mean))  # type: ignore[union-attr]
            return mean, bias

        mean_0, bias_0 = _bias_for_target(target_rgb_0)
        mean_1, bias_1 = _bias_for_target(target_rgb_1)
        self.head_rgb_0.update({"bias": bias_0.astype(self.head_rgb_0.bias.dtype)})
        self.head_rgb_1.update({"bias": bias_1.astype(self.head_rgb_1.bias.dtype)})
        mx.eval(self.head_rgb_0.bias, self.head_rgb_1.bias)  # type: ignore[union-attr]
        return {
            "schema": "hi_nerv_output_head_target_bias_init.v1",
            "enabled": True,
            "epsilon": float(epsilon),
            "target_rgb_0_mean": [
                float(v) for v in np.asarray(mean_0, dtype=np.float32).reshape(-1)
            ],
            "target_rgb_1_mean": [
                float(v) for v in np.asarray(mean_1, dtype=np.float32).reshape(-1)
            ],
            "head_rgb_0_bias": [
                float(v) for v in np.asarray(self.head_rgb_0.bias, dtype=np.float32).reshape(-1)
            ],
            "head_rgb_1_bias": [
                float(v) for v in np.asarray(self.head_rgb_1.bias, dtype=np.float32).reshape(-1)
            ],
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": [
                "head_rgb_0.bias",
                "head_rgb_1.bias",
            ],
        }

    def initialize_output_head_contrast_from_targets(
        self,
        target_rgb_0: Any,
        target_rgb_1: Any,
        *,
        pair_indices: Any,
        min_output_std: float = 1.0e-4,
        max_gain: float = 32.0,
    ) -> dict[str, Any]:
        """Scale sigmoid RGB-head weights to match target contrast at launch.

        The bias initializer solves the constant-color optimum.  Compact
        HiNeRV smokes then exposed the next basin: the scorer-resolution output
        starts with the right mean but only a few percent of the target RGB
        variance, so SegNet sees a one-class flat image.  This method applies
        the small-signal correction to the archive-charged RGB head weights:
        ``weight *= std(target) / std(output)`` per output channel, clipped to a
        bounded gain.  No sidecar is introduced; the changed tensors are the
        ordinary decoder weights inside the archive.
        """

        eps = float(min_output_std)
        gain_cap = float(max_gain)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                "min_output_std must be finite and positive; "
                f"got {min_output_std!r}"
            )
        if not math.isfinite(gain_cap) or gain_cap < 1.0:
            raise ValueError(f"max_gain must be finite and >= 1; got {max_gain!r}")

        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim != 1 or int(idx.shape[0]) <= 0:
            raise ValueError("pair_indices must be a non-empty rank-1 tensor")

        target0 = mx.array(target_rgb_0).astype(mx.float32)  # type: ignore[union-attr]
        target1 = mx.array(target_rgb_1).astype(mx.float32)  # type: ignore[union-attr]
        for name, target in (("target_rgb_0", target0), ("target_rgb_1", target1)):
            if target.ndim != 4 or int(target.shape[-1]) != 3:
                raise ValueError(
                    f"{name} must be NHWC with 3 channels; got "
                    f"shape={tuple(int(v) for v in target.shape)}"
                )
            if int(target.shape[0]) != int(idx.shape[0]):
                raise ValueError(
                    f"{name} batch {int(target.shape[0])} must match "
                    f"pair_indices length {int(idx.shape[0])}"
                )

        pair01 = self(idx) / 255.0
        cand0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))  # type: ignore[union-attr]
        cand1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))  # type: ignore[union-attr]

        def _channel_std(x: Any) -> Any:
            mean = mx.mean(x, axis=(0, 1, 2), keepdims=True)  # type: ignore[union-attr]
            var = mx.mean((x - mean) ** 2, axis=(0, 1, 2))  # type: ignore[union-attr]
            return mx.sqrt(mx.maximum(var, 0.0))  # type: ignore[union-attr]

        def _gain(target: Any, cand: Any) -> tuple[Any, Any, Any]:
            target_std = _channel_std(target)
            cand_std = _channel_std(cand)
            raw = target_std / mx.maximum(cand_std, eps)  # type: ignore[union-attr]
            clipped = mx.clip(raw, 1.0 / gain_cap, gain_cap)  # type: ignore[union-attr]
            return target_std, cand_std, clipped

        target0_std, before0_std, gain0 = _gain(target0, cand0)
        target1_std, before1_std, gain1 = _gain(target1, cand1)
        self.head_rgb_0.update(
            {
                "weight": self.head_rgb_0.weight
                * mx.reshape(gain0.astype(self.head_rgb_0.weight.dtype), (3, 1, 1, 1))
            }
        )
        self.head_rgb_1.update(
            {
                "weight": self.head_rgb_1.weight
                * mx.reshape(gain1.astype(self.head_rgb_1.weight.dtype), (3, 1, 1, 1))
            }
        )
        pair_after01 = self(idx) / 255.0
        after0 = mx.transpose(pair_after01[:, 0], (0, 2, 3, 1))  # type: ignore[union-attr]
        after1 = mx.transpose(pair_after01[:, 1], (0, 2, 3, 1))  # type: ignore[union-attr]
        after0_std = _channel_std(after0)
        after1_std = _channel_std(after1)
        mx.eval(
            self.head_rgb_0.weight,
            self.head_rgb_1.weight,
            target0_std,
            target1_std,
            before0_std,
            before1_std,
            after0_std,
            after1_std,
            gain0,
            gain1,
        )

        def _list(values: Any) -> list[float]:
            return [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)]

        target0_std_values = _list(target0_std)
        target1_std_values = _list(target1_std)
        before0_std_values = _list(before0_std)
        before1_std_values = _list(before1_std)
        after0_std_values = _list(after0_std)
        after1_std_values = _list(after1_std)
        gain0_values = _list(gain0)
        gain1_values = _list(gain1)

        def _mean(values: list[float]) -> float:
            return float(np.mean(np.asarray(values, dtype=np.float64)))

        def _lift(before_values: list[float], after_values: list[float]) -> float:
            return _mean(after_values) / max(_mean(before_values), 1.0e-12)

        def _target_capture(
            target_values: list[float],
            after_values: list[float],
        ) -> float:
            return _mean(after_values) / max(_mean(target_values), eps)

        lift0 = _lift(before0_std_values, after0_std_values)
        lift1 = _lift(before1_std_values, after1_std_values)
        target_capture0 = _target_capture(target0_std_values, after0_std_values)
        target_capture1 = _target_capture(target1_std_values, after1_std_values)
        target0_nonflat = _mean(target0_std_values) > eps
        target1_nonflat = _mean(target1_std_values) > eps
        blockers: list[str] = []
        if target0_nonflat and lift0 <= 1.01:
            blockers.append("hinerv_output_head_contrast_init_frame0_no_std_lift")
        if target1_nonflat and lift1 <= 1.01:
            blockers.append("hinerv_output_head_contrast_init_frame1_no_std_lift")
        clipped0 = sum(1 for value in gain0_values if abs(value - gain_cap) <= 1.0e-6)
        clipped1 = sum(1 for value in gain1_values if abs(value - gain_cap) <= 1.0e-6)

        return {
            "schema": "hi_nerv_output_head_target_contrast_init.v1",
            "enabled": True,
            "method": "per_channel_sigmoid_head_small_signal_std_match",
            "pair_count": int(idx.shape[0]),
            "min_output_std": eps,
            "max_gain": gain_cap,
            "target_rgb_0_std": target0_std_values,
            "target_rgb_1_std": target1_std_values,
            "output_rgb_0_std_before": before0_std_values,
            "output_rgb_1_std_before": before1_std_values,
            "output_rgb_0_std_after": after0_std_values,
            "output_rgb_1_std_after": after1_std_values,
            "output_rgb_0_std_lift_ratio": lift0,
            "output_rgb_1_std_lift_ratio": lift1,
            "output_rgb_0_target_std_capture_ratio": target_capture0,
            "output_rgb_1_target_std_capture_ratio": target_capture1,
            "head_rgb_0_weight_gain": gain0_values,
            "head_rgb_1_weight_gain": gain1_values,
            "head_rgb_0_gain_clipped_channel_count": clipped0,
            "head_rgb_1_gain_clipped_channel_count": clipped1,
            "contrast_lift_passed": not blockers,
            "blockers": blockers,
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": [
                "head_rgb_0.weight",
                "head_rgb_1.weight",
            ],
            "human_visual_fidelity_objective": False,
        }

    def fit_scorer_domain_bootstrap_from_targets(
        self,
        target_rgb_0: Any,
        target_rgb_1: Any,
        *,
        pair_indices: Any,
        target_segnet_argmax_1: Any | None = None,
        target_region_bootstrap_weight: float = 0.25,
        scorer_teacher: Any | None = None,
        segnet_margin_bootstrap_weight: float = 0.0,
        segnet_margin_bootstrap_floor: float = 0.25,
        segnet_hard_birth_bootstrap_weight: float = 0.0,
        segnet_hard_birth_bootstrap_min_ratio_floor: float = 0.02,
        steps: int = 8,
        learning_rate: float = 2.0e-3,
        rgb_weight: float = 1.0,
        yuv6_weight: float = 0.5,
        temporal_delta_weight: float = 0.25,
        contrast_floor_weight: float = 0.5,
        rgb_std_min_ratio: float = 0.75,
        yuv6_temporal_std_min_ratio: float = 0.5,
        weight_decay: float = 0.0,
        grad_clip_max_norm: float | None = 1.0,
    ) -> dict[str, Any]:
        """Run a bounded scorer-domain prefit on archive-charged parameters.

        Mean/std output-head calibration can still land in a SegNet one-class
        basin.  This bootstrap mutates the real decoder/latent tensors before
        the normal score-aware trainer begins, using the exact scorer-domain
        surfaces available without invoking the heavy scorer: SegNet's last
        frame RGB geometry and PoseNet's two-frame PR95/YUV6 pair plus temporal
        delta.  The result is ordinary model state, so export/runtime byte
        accounting stays unchanged apart from the charged tensor values.
        """

        _require_mlx()
        import mlx.optimizers as mlx_optim

        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

        step_count = int(steps)
        if step_count <= 0:
            return {
                "schema": "hi_nerv_scorer_domain_bootstrap.v1",
                "enabled": False,
                "reason": "steps_not_positive",
                "steps": step_count,
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [],
                "human_visual_fidelity_objective": False,
            }
        lr = float(learning_rate)
        weights = {
            "rgb_weight": float(rgb_weight),
            "yuv6_weight": float(yuv6_weight),
            "temporal_delta_weight": float(temporal_delta_weight),
            "contrast_floor_weight": float(contrast_floor_weight),
            "target_region_bootstrap_weight": float(target_region_bootstrap_weight),
            "segnet_margin_bootstrap_weight": float(segnet_margin_bootstrap_weight),
            "segnet_hard_birth_bootstrap_weight": float(
                segnet_hard_birth_bootstrap_weight
            ),
        }
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"learning_rate must be finite and positive; got {learning_rate}")
        for name, value in weights.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        if max(weights.values()) <= 0.0:
            raise ValueError("at least one scorer-domain bootstrap loss weight must be > 0")
        rgb_std_floor = float(rgb_std_min_ratio)
        temporal_std_floor = float(yuv6_temporal_std_min_ratio)
        for name, value in (
            ("rgb_std_min_ratio", rgb_std_floor),
            ("yuv6_temporal_std_min_ratio", temporal_std_floor),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        segnet_margin_floor = float(segnet_margin_bootstrap_floor)
        if not math.isfinite(segnet_margin_floor) or segnet_margin_floor < 0.0:
            raise ValueError(
                "segnet_margin_bootstrap_floor must be finite and non-negative; "
                f"got {segnet_margin_bootstrap_floor}"
            )
        segnet_hard_birth_floor = float(segnet_hard_birth_bootstrap_min_ratio_floor)
        if (
            not math.isfinite(segnet_hard_birth_floor)
            or segnet_hard_birth_floor < 0.0
            or segnet_hard_birth_floor > 1.0
        ):
            raise ValueError(
                "segnet_hard_birth_bootstrap_min_ratio_floor must be finite "
                f"in [0, 1]; got {segnet_hard_birth_bootstrap_min_ratio_floor}"
            )
        wd = float(weight_decay)
        if not math.isfinite(wd) or wd < 0.0:
            raise ValueError(f"weight_decay must be finite and non-negative; got {weight_decay}")
        clip = None if grad_clip_max_norm is None else float(grad_clip_max_norm)
        if clip is not None and (not math.isfinite(clip) or clip <= 0.0):
            raise ValueError(
                "grad_clip_max_norm must be None or finite and positive; "
                f"got {grad_clip_max_norm}"
            )

        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim != 1 or int(idx.shape[0]) <= 0:
            raise ValueError("pair_indices must be a non-empty rank-1 tensor")
        target0 = mx.array(target_rgb_0).astype(mx.float32)  # type: ignore[union-attr]
        target1 = mx.array(target_rgb_1).astype(mx.float32)  # type: ignore[union-attr]
        for name, target in (("target_rgb_0", target0), ("target_rgb_1", target1)):
            if target.ndim != 4 or int(target.shape[-1]) != 3:
                raise ValueError(
                    f"{name} must be NHWC with 3 channels; got "
                    f"shape={tuple(int(v) for v in target.shape)}"
                )
            if int(target.shape[0]) != int(idx.shape[0]):
                raise ValueError(
                    f"{name} batch {int(target.shape[0])} must match "
                    f"pair_indices length {int(idx.shape[0])}"
                )

        ref_yuv0 = mx.stop_gradient(rgb_to_yuv6_mlx(target0 * 255.0) / 255.0)  # type: ignore[union-attr]
        ref_yuv1 = mx.stop_gradient(rgb_to_yuv6_mlx(target1 * 255.0) / 255.0)  # type: ignore[union-attr]
        ref_temporal = ref_yuv1 - ref_yuv0

        def _target_region_weight_map() -> tuple[Any, dict[str, Any]]:
            """Return a mean-one frame-1 region weight map for bootstrap waterfill."""

            if target_segnet_argmax_1 is not None:
                labels = mx.array(target_segnet_argmax_1, dtype=mx.int32)  # type: ignore[union-attr]
                if labels.ndim == 4 and int(labels.shape[-1]) == 1:
                    labels = labels[..., 0]
                if labels.ndim != 3:
                    raise ValueError(
                        "target_segnet_argmax_1 must have shape BHW or BHW1; "
                        f"got {tuple(int(v) for v in labels.shape)}"
                    )
                if tuple(int(v) for v in labels.shape) != tuple(
                    int(v) for v in target1.shape[:3]
                ):
                    raise ValueError(
                        "target_segnet_argmax_1 shape must match target_rgb_1 BHW; "
                        f"argmax={tuple(int(v) for v in labels.shape)} "
                        f"target={tuple(int(v) for v in target1.shape[:3])}"
                    )
                class_count = int(np.asarray(mx.max(labels), dtype=np.int32).item()) + 1
                if class_count <= 0:
                    class_count = 1
                weight = mx.zeros_like(labels.astype(mx.float32))  # type: ignore[union-attr]
                class_rows: list[dict[str, Any]] = []
                eps = mx.array(1.0e-6, dtype=mx.float32)
                for class_index in range(class_count):
                    mask = (labels == class_index).astype(mx.float32)
                    fraction = mx.mean(mask)  # type: ignore[union-attr]
                    active = (fraction > 0.0).astype(mx.float32)
                    class_weight = active / mx.sqrt(mx.maximum(fraction, eps))  # type: ignore[union-attr]
                    weight = weight + mask * class_weight
                    mx.eval(fraction, class_weight)  # type: ignore[union-attr]
                    class_rows.append({
                        "class_index": int(class_index),
                        "target_fraction": float(fraction.item()),
                        "inverse_sqrt_fraction_weight": float(class_weight.item()),
                    })
                mean_weight = mx.mean(weight)  # type: ignore[union-attr]
                normalized = weight / mx.maximum(mean_weight, eps)  # type: ignore[union-attr]
                mx.eval(normalized, mean_weight)  # type: ignore[union-attr]
                return normalized[..., None], {
                    "source": "exact_segnet_target_argmax_frame1",
                    "class_count": int(class_count),
                    "class_rows": class_rows,
                    "mean_before_normalization": float(mean_weight.item()),
                }

            channel_mean = mx.mean(target1, axis=(1, 2), keepdims=True)  # type: ignore[union-attr]
            saliency = mx.mean(mx.abs(target1 - channel_mean), axis=-1, keepdims=True)  # type: ignore[union-attr]
            saliency_mean = mx.mean(saliency)  # type: ignore[union-attr]
            normalized_saliency = saliency / mx.maximum(  # type: ignore[union-attr]
                saliency_mean,
                mx.array(1.0e-6, dtype=mx.float32),
            )
            weight = mx.clip(  # type: ignore[union-attr]
                mx.array(0.5, dtype=mx.float32)
                + mx.array(0.5, dtype=mx.float32) * normalized_saliency,
                mx.array(0.25, dtype=mx.float32),
                mx.array(4.0, dtype=mx.float32),
            )
            mean_weight = mx.mean(weight)  # type: ignore[union-attr]
            normalized = weight / mx.maximum(  # type: ignore[union-attr]
                mean_weight,
                mx.array(1.0e-6, dtype=mx.float32),
            )
            mx.eval(normalized, mean_weight, saliency_mean)  # type: ignore[union-attr]
            return normalized, {
                "source": "rgb_frame1_saliency_fallback",
                "mean_before_normalization": float(mean_weight.item()),
                "saliency_mean": float(saliency_mean.item()),
            }

        target_region_weight_map, target_region_metadata = _target_region_weight_map()
        archive_charged_bootstrap_tensors = [
            "latents_coarse",
            "latents_mid",
            "latents_fine",
            "latent_embed.*",
            "blocks.*",
            "feature_grids.*",
            "convnext_blocks.*",
            "mid_injector.*",
            "fine_injector.*",
            "head_rgb_0.*",
            "head_rgb_1.*",
        ]
        live_segnet_scoped_bootstrap_tensors = [
            "latents_fine",
            "feature_grids.*",
            "fine_injector.*",
            "head_rgb_1.*",
        ]
        segnet_margin_live_fn = None
        segnet_margin_target_labels = None
        segnet_live_bootstrap_requested = bool(
            weights["segnet_margin_bootstrap_weight"] > 0.0
            or weights["segnet_hard_birth_bootstrap_weight"] > 0.0
        )
        if segnet_live_bootstrap_requested:
            archive_charged_bootstrap_tensors = live_segnet_scoped_bootstrap_tensors
        bootstrap_update_scope = (
            "live_segnet_scoped_late_feature_grid_fine_latent_head_rgb_1"
            if segnet_live_bootstrap_requested
            else "full_archive_charged_scorer_domain_prefit"
        )

        def _flat_param_name(raw_name: Any) -> str:
            if isinstance(raw_name, (tuple, list)):
                return ".".join(str(part) for part in raw_name)
            return str(raw_name)

        def _bootstrap_update_name_allowed(raw_name: Any) -> bool:
            if not segnet_live_bootstrap_requested:
                return True
            name = _flat_param_name(raw_name)
            return (
                name == "latents_fine"
                or name.startswith("latents_fine.")
                or name.startswith("feature_grids.")
                or name.startswith("fine_injector.")
                or name.startswith("head_rgb_1.")
            )
        segnet_margin_metadata: dict[str, Any] = {
            "schema": "hi_nerv_scorer_domain_bootstrap_live_segnet_margin.v1",
            "enabled": False,
            "weight": float(weights["segnet_margin_bootstrap_weight"]),
            "margin_floor": float(segnet_margin_floor),
            "reason": "segnet_margin_bootstrap_weight_not_positive",
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": [],
            "human_visual_fidelity_objective": False,
        }
        segnet_hard_birth_metadata: dict[str, Any] = {
            "schema": "hi_nerv_scorer_domain_bootstrap_live_segnet_hard_birth.v1",
            "enabled": False,
            "weight": float(weights["segnet_hard_birth_bootstrap_weight"]),
            "min_ratio_floor": float(segnet_hard_birth_floor),
            "reason": "segnet_hard_birth_bootstrap_weight_not_positive",
            "worst_loss_selection": "score_weighted_unsolved_argmax_mass",
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": [],
            "human_visual_fidelity_objective": False,
        }
        if segnet_live_bootstrap_requested:
            blockers: list[str] = []
            live_fn = getattr(scorer_teacher, "teacher_logits_for_frames_nhwc01", None)
            if not callable(live_fn):
                blockers.append("hi_nerv_bootstrap_live_segnet_candidate_logits_missing")
            labels = None
            if target_segnet_argmax_1 is not None:
                labels = mx.array(target_segnet_argmax_1, dtype=mx.int32)  # type: ignore[union-attr]
                if labels.ndim == 4 and int(labels.shape[-1]) == 1:
                    labels = labels[..., 0]
            else:
                local_teacher_indices = mx.arange(  # type: ignore[union-attr]
                    int(target1.shape[0]),
                    dtype=mx.int32,
                )
                argmax_fn = getattr(scorer_teacher, "teacher_argmax_for_indices", None)
                logits_fn = getattr(scorer_teacher, "teacher_logits_for_indices", None)
                if callable(argmax_fn):
                    labels = argmax_fn(local_teacher_indices)
                elif callable(logits_fn):
                    target_logits = logits_fn(local_teacher_indices)
                    labels = mx.argmax(target_logits, axis=-1)  # type: ignore[union-attr]
                else:
                    blockers.append("hi_nerv_bootstrap_target_segnet_argmax_missing")
            if labels is not None:
                if labels.ndim == 4 and int(labels.shape[-1]) == 1:
                    labels = labels[..., 0]
                if labels.ndim != 3:
                    raise ValueError(
                        "target_segnet_argmax_1 must have shape BHW or BHW1; "
                        f"got {tuple(int(v) for v in labels.shape)}"
                    )
                if tuple(int(v) for v in labels.shape) != tuple(
                    int(v) for v in target1.shape[:3]
                ):
                    raise ValueError(
                        "target_segnet_argmax_1 shape must match target_rgb_1 BHW; "
                        f"argmax={tuple(int(v) for v in labels.shape)} "
                        f"target={tuple(int(v) for v in target1.shape[:3])}"
                    )
                segnet_margin_target_labels = mx.stop_gradient(labels)
                mx.eval(segnet_margin_target_labels)  # type: ignore[union-attr]
            if blockers:
                raise ValueError(
                    "a live SegNet bootstrap weight was positive, but the live "
                    f"SegNet bootstrap actuator cannot run: {blockers}"
                )
            segnet_margin_live_fn = live_fn
            segnet_margin_metadata = {
                "schema": "hi_nerv_scorer_domain_bootstrap_live_segnet_margin.v1",
                "enabled": bool(weights["segnet_margin_bootstrap_weight"] > 0.0),
                "weight": float(weights["segnet_margin_bootstrap_weight"]),
                "margin_floor": float(segnet_margin_floor),
                "source": "live_mlx_segnet_candidate_logits_against_frame1_target_argmax",
                "target_index_semantics": "local_bootstrap_batch_indices",
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": archive_charged_bootstrap_tensors,
                "human_visual_fidelity_objective": False,
            }
            segnet_hard_birth_metadata = {
                "schema": "hi_nerv_scorer_domain_bootstrap_live_segnet_hard_birth.v1",
                "enabled": bool(weights["segnet_hard_birth_bootstrap_weight"] > 0.0),
                "weight": float(weights["segnet_hard_birth_bootstrap_weight"]),
                "min_ratio_floor": float(segnet_hard_birth_floor),
                "source": "live_mlx_segnet_candidate_logits_worst_target_class_birth",
                "worst_loss_selection": "score_weighted_unsolved_argmax_mass",
                "target_index_semantics": "local_bootstrap_batch_indices",
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": archive_charged_bootstrap_tensors,
                "human_visual_fidelity_objective": False,
            }

        def _predict_pair01(model_obj: Any) -> tuple[Any, Any]:
            pair01 = model_obj(idx) / 255.0
            pred0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))  # type: ignore[union-attr]
            pred1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))  # type: ignore[union-attr]
            return pred0, pred1

        def _segnet_margin_bootstrap_tensors(pred1: Any) -> dict[str, Any]:
            if (
                segnet_margin_live_fn is None
                or segnet_margin_target_labels is None
                or (
                    weights["segnet_margin_bootstrap_weight"] <= 0.0
                    and weights["segnet_hard_birth_bootstrap_weight"] <= 0.0
                )
            ):
                zero = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
                one = mx.array(1.0, dtype=mx.float32)  # type: ignore[union-attr]
                return {
                    "segnet_margin_bootstrap_loss": zero,
                    "segnet_margin_bootstrap_argmax_disagreement": zero,
                    "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass": zero,
                    "segnet_margin_bootstrap_score_weighted_worst_unsolved_argmax_mass": zero,
                    "segnet_margin_bootstrap_candidate_target_class_min_ratio": one,
                    "segnet_margin_bootstrap_worst_class_index": mx.array(  # type: ignore[union-attr]
                        -1.0,
                        dtype=mx.float32,
                    ),
                    "segnet_hard_birth_bootstrap_loss": zero,
                    "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass": zero,
                    "segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass": zero,
                    "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio": one,
                    "segnet_hard_birth_bootstrap_worst_class_index": mx.array(  # type: ignore[union-attr]
                        -1.0,
                        dtype=mx.float32,
                    ),
                }

            candidate_logits = segnet_margin_live_fn(pred1)
            if candidate_logits.ndim != 4:
                raise ValueError(
                    "live SegNet bootstrap candidate logits must be BHWC; got "
                    f"{tuple(int(v) for v in candidate_logits.shape)}"
                )
            if tuple(int(v) for v in candidate_logits.shape[:3]) != tuple(
                int(v) for v in segnet_margin_target_labels.shape
            ):
                raise ValueError(
                    "live SegNet bootstrap logits BHW must match target labels; "
                    f"logits={tuple(int(v) for v in candidate_logits.shape[:3])} "
                    f"labels={tuple(int(v) for v in segnet_margin_target_labels.shape)}"
                )
            class_count = int(candidate_logits.shape[-1])
            if class_count <= 0:
                raise ValueError("live SegNet bootstrap logits must have at least one class")
            max_target_class = int(
                np.asarray(mx.max(segnet_margin_target_labels), dtype=np.int32).item()
            )
            if max_target_class >= class_count:
                raise ValueError(
                    "target SegNet argmax contains a class outside live logits: "
                    f"max_target_class={max_target_class} class_count={class_count}"
                )
            pred_class = mx.argmax(candidate_logits, axis=-1)  # type: ignore[union-attr]
            eps = mx.array(1.0e-6, dtype=mx.float32)  # type: ignore[union-attr]
            total_loss = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            active_count = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            total_unsolved = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            worst_unsolved = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            worst_class = mx.array(-1.0, dtype=mx.float32)  # type: ignore[union-attr]
            min_region_ratio = mx.array(1.0, dtype=mx.float32)  # type: ignore[union-attr]
            logits = candidate_logits - mx.max(  # type: ignore[union-attr]
                candidate_logits,
                axis=-1,
                keepdims=True,
            )
            exp_logits = mx.exp(logits)  # type: ignore[union-attr]
            probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)  # type: ignore[union-attr]
            hard_birth_total_loss = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            hard_birth_active_count = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            hard_birth_worst_loss = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            hard_birth_worst_unsolved = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            hard_birth_worst_class = mx.array(-1.0, dtype=mx.float32)  # type: ignore[union-attr]
            hard_birth_worst_loss_class = mx.array(-1.0, dtype=mx.float32)  # type: ignore[union-attr]
            hard_birth_min_ratio = mx.array(1.0, dtype=mx.float32)  # type: ignore[union-attr]
            metrics: dict[str, Any] = {}
            for class_index in range(class_count):
                target_mask = (segnet_margin_target_labels == class_index).astype(
                    mx.float32
                )
                target_fraction = mx.mean(target_mask)  # type: ignore[union-attr]
                active = (target_fraction > 0.0).astype(mx.float32)
                class_pred_mask = (pred_class == class_index).astype(mx.float32)
                hard_fraction = mx.mean(class_pred_mask)  # type: ignore[union-attr]
                correct_pixels = mx.sum(class_pred_mask * target_mask)  # type: ignore[union-attr]
                target_pixels = mx.sum(target_mask)  # type: ignore[union-attr]
                region_ratio = correct_pixels / mx.maximum(target_pixels, eps)  # type: ignore[union-attr]
                support_ratio = hard_fraction / mx.maximum(target_fraction, eps)  # type: ignore[union-attr]
                class_logit = candidate_logits[..., class_index]
                if class_count == 1:
                    impostor_logit = mx.zeros_like(class_logit)  # type: ignore[union-attr]
                elif class_index == 0:
                    impostor_logit = mx.max(  # type: ignore[union-attr]
                        candidate_logits[..., 1:],
                        axis=-1,
                    )
                elif class_index == class_count - 1:
                    impostor_logit = mx.max(  # type: ignore[union-attr]
                        candidate_logits[..., :class_index],
                        axis=-1,
                    )
                else:
                    impostor_logit = mx.max(  # type: ignore[union-attr]
                        mx.concatenate(
                            [
                                candidate_logits[..., :class_index],
                                candidate_logits[..., class_index + 1 :],
                            ],
                            axis=-1,
                        ),
                        axis=-1,
                    )
                margin = mx.maximum(  # type: ignore[union-attr]
                    0.0,
                    impostor_logit - class_logit + segnet_margin_floor,
                )
                crossing_loss = mx.sum(  # type: ignore[union-attr]
                    margin * margin * target_mask
                ) / mx.maximum(target_pixels, eps)
                score_weighted_unsolved = (
                    mx.array(100.0, dtype=mx.float32)
                    * target_fraction
                    * mx.maximum(0.0, 1.0 - region_ratio)  # type: ignore[union-attr]
                )
                support_deficit = mx.maximum(0.0, 1.0 - support_ratio)  # type: ignore[union-attr]
                region_deficit = mx.maximum(0.0, 1.0 - region_ratio)  # type: ignore[union-attr]
                boost = mx.stop_gradient(  # type: ignore[union-attr]
                    active
                    * (
                        1.0
                        + mx.minimum(32.0, score_weighted_unsolved)
                        + 16.0 * support_deficit
                        + 16.0 * region_deficit
                    )
                )
                class_loss = boost * crossing_loss
                hard_birth_ratio = mx.minimum(  # type: ignore[union-attr]
                    support_ratio,
                    region_ratio,
                )
                hard_birth_deficit = mx.maximum(  # type: ignore[union-attr]
                    0.0,
                    segnet_hard_birth_floor - hard_birth_ratio,
                )
                hard_birth_active = active * (
                    hard_birth_deficit > 0.0
                ).astype(mx.float32)
                class_prob = probs[..., class_index]
                candidate_soft_fraction = mx.mean(class_prob)  # type: ignore[union-attr]
                target_prob_mean = mx.sum(target_mask * class_prob) / mx.maximum(  # type: ignore[union-attr]
                    target_pixels,
                    eps,
                )
                masked_margin = mx.where(  # type: ignore[union-attr]
                    target_mask > 0.0,
                    margin,
                    mx.array(1.0e30, dtype=margin.dtype),
                )
                frontier_margin = mx.stop_gradient(mx.min(masked_margin))  # type: ignore[union-attr]
                frontier_margin = mx.where(  # type: ignore[union-attr]
                    active > 0.0,
                    frontier_margin,
                    mx.array(0.0, dtype=mx.float32),
                )
                shifted_margin = mx.maximum(  # type: ignore[union-attr]
                    margin - frontier_margin,
                    mx.array(0.0, dtype=mx.float32),
                )
                seed_temperature = mx.minimum(  # type: ignore[union-attr]
                    mx.array(2.0, dtype=mx.float32),
                    mx.maximum(
                        mx.array(0.25, dtype=mx.float32),
                        mx.sqrt(mx.maximum(target_fraction, eps)),  # type: ignore[union-attr]
                    ),
                )
                seed_weight = target_mask * mx.exp(  # type: ignore[union-attr]
                    -mx.stop_gradient(shifted_margin) / seed_temperature
                )
                seed_weight_mass = mx.sum(seed_weight)  # type: ignore[union-attr]
                seed_weight_normalized = seed_weight / mx.maximum(  # type: ignore[union-attr]
                    seed_weight_mass,
                    eps,
                )
                seed_target_prob_mean = mx.sum(seed_weight_normalized * class_prob)  # type: ignore[union-attr]
                seed_crossing_loss = mx.sum(  # type: ignore[union-attr]
                    seed_weight_normalized * margin * margin
                )
                soft_mass_floor = mx.minimum(  # type: ignore[union-attr]
                    target_fraction,
                    mx.maximum(
                        mx.array(1.0e-3, dtype=mx.float32),
                        segnet_hard_birth_floor * target_fraction,
                    ),
                )
                soft_mass_log_ratio = mx.maximum(  # type: ignore[union-attr]
                    mx.log((soft_mass_floor + eps) / mx.maximum(candidate_soft_fraction, eps)),
                    0.0,
                )
                target_prob_floor = mx.minimum(  # type: ignore[union-attr]
                    mx.array(0.85, dtype=mx.float32),
                    mx.maximum(
                        mx.array(0.55, dtype=mx.float32),
                        mx.array(0.35, dtype=mx.float32)
                        + mx.array(4.0, dtype=mx.float32) * segnet_hard_birth_floor,
                    ),
                )
                seed_prob_floor = mx.minimum(  # type: ignore[union-attr]
                    mx.array(0.92, dtype=mx.float32),
                    target_prob_floor + mx.array(0.10, dtype=mx.float32),
                )
                target_prob_deficit = mx.maximum(  # type: ignore[union-attr]
                    target_prob_floor - target_prob_mean,
                    0.0,
                )
                seed_prob_deficit = mx.maximum(  # type: ignore[union-attr]
                    seed_prob_floor - seed_target_prob_mean,
                    0.0,
                )
                hard_birth_boost = mx.stop_gradient(  # type: ignore[union-attr]
                    hard_birth_active
                    * (
                        1.0
                        + mx.minimum(64.0, score_weighted_unsolved)
                        + 64.0 * hard_birth_deficit
                        + 16.0
                        / mx.sqrt(
                            mx.maximum(
                                target_fraction,
                                mx.array(1.0e-4, dtype=mx.float32),
                            )
                        )
                    )
                )
                hard_birth_loss_raw = (
                    8.0 * soft_mass_log_ratio * soft_mass_log_ratio
                    + 16.0 * target_prob_deficit * target_prob_deficit
                    + 24.0 * seed_prob_deficit * seed_prob_deficit
                    + (
                        1.0
                        + mx.minimum(32.0, mx.stop_gradient(score_weighted_unsolved))
                    )
                    * crossing_loss
                    + 8.0 * seed_crossing_loss
                )
                hard_birth_loss = hard_birth_boost * hard_birth_loss_raw
                hard_birth_total_loss = hard_birth_total_loss + hard_birth_loss
                hard_birth_active_count = hard_birth_active_count + hard_birth_active
                hard_birth_better_unsolved = (
                    score_weighted_unsolved > hard_birth_worst_unsolved
                )
                hard_birth_worst_loss = mx.where(  # type: ignore[union-attr]
                    hard_birth_better_unsolved,
                    hard_birth_loss,
                    hard_birth_worst_loss,
                )
                hard_birth_worst_loss_class = mx.where(  # type: ignore[union-attr]
                    hard_birth_better_unsolved,
                    mx.array(float(class_index), dtype=mx.float32),
                    hard_birth_worst_loss_class,
                )
                hard_birth_worst_unsolved = mx.where(  # type: ignore[union-attr]
                    hard_birth_better_unsolved,
                    score_weighted_unsolved,
                    hard_birth_worst_unsolved,
                )
                hard_birth_worst_class = mx.where(  # type: ignore[union-attr]
                    hard_birth_better_unsolved,
                    mx.array(float(class_index), dtype=mx.float32),
                    hard_birth_worst_class,
                )
                hard_birth_min_ratio = mx.minimum(  # type: ignore[union-attr]
                    hard_birth_min_ratio,
                    active * hard_birth_ratio + (1.0 - active),
                )
                total_loss = total_loss + class_loss
                active_count = active_count + active
                total_unsolved = total_unsolved + active * score_weighted_unsolved
                better_worst = score_weighted_unsolved > worst_unsolved
                worst_unsolved = mx.where(  # type: ignore[union-attr]
                    better_worst,
                    score_weighted_unsolved,
                    worst_unsolved,
                )
                worst_class = mx.where(  # type: ignore[union-attr]
                    better_worst,
                    mx.array(float(class_index), dtype=mx.float32),
                    worst_class,
                )
                min_region_ratio = mx.minimum(  # type: ignore[union-attr]
                    min_region_ratio,
                    active * region_ratio + (1.0 - active),
                )
                prefix = f"segnet_margin_bootstrap_class_{class_index}"
                metrics[f"{prefix}_target_fraction"] = target_fraction
                metrics[f"{prefix}_candidate_hard_fraction"] = hard_fraction
                metrics[f"{prefix}_candidate_support_ratio"] = support_ratio
                metrics[f"{prefix}_target_region_correct_ratio"] = region_ratio
                metrics[f"{prefix}_crossing_loss"] = crossing_loss
                metrics[f"{prefix}_score_weighted_unsolved_argmax_mass"] = (
                    score_weighted_unsolved
                )
                birth_prefix = f"segnet_hard_birth_bootstrap_class_{class_index}"
                metrics[f"{birth_prefix}_loss"] = hard_birth_loss_raw
                metrics[f"{birth_prefix}_active"] = hard_birth_active
                metrics[f"{birth_prefix}_target_fraction"] = target_fraction
                metrics[f"{birth_prefix}_candidate_hard_fraction"] = hard_fraction
                metrics[f"{birth_prefix}_candidate_soft_fraction"] = (
                    candidate_soft_fraction
                )
                metrics[f"{birth_prefix}_support_ratio"] = support_ratio
                metrics[f"{birth_prefix}_target_region_correct_ratio"] = region_ratio
                metrics[f"{birth_prefix}_birth_ratio"] = hard_birth_ratio
                metrics[f"{birth_prefix}_birth_deficit"] = hard_birth_deficit
                metrics[f"{birth_prefix}_target_prob_mean"] = target_prob_mean
                metrics[f"{birth_prefix}_target_prob_floor"] = target_prob_floor
                metrics[f"{birth_prefix}_target_prob_deficit"] = target_prob_deficit
                metrics[f"{birth_prefix}_seed_target_prob_mean"] = (
                    seed_target_prob_mean
                )
                metrics[f"{birth_prefix}_seed_prob_floor"] = seed_prob_floor
                metrics[f"{birth_prefix}_seed_prob_deficit"] = seed_prob_deficit
                metrics[f"{birth_prefix}_soft_mass_floor"] = soft_mass_floor
                metrics[f"{birth_prefix}_soft_mass_log_ratio"] = soft_mass_log_ratio
                metrics[f"{birth_prefix}_frontier_margin"] = frontier_margin
                metrics[f"{birth_prefix}_seed_crossing_loss"] = seed_crossing_loss
                metrics[f"{birth_prefix}_score_weighted_unsolved_argmax_mass"] = (
                    score_weighted_unsolved
                )
            metrics.update({
                "segnet_margin_bootstrap_loss": total_loss
                / mx.maximum(active_count, eps),
                "segnet_margin_bootstrap_argmax_disagreement": mx.mean(  # type: ignore[union-attr]
                    (pred_class != segnet_margin_target_labels).astype(mx.float32)
                ),
                "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass": (
                    total_unsolved
                ),
                "segnet_margin_bootstrap_score_weighted_worst_unsolved_argmax_mass": (
                    worst_unsolved
                ),
                "segnet_margin_bootstrap_candidate_target_class_min_ratio": (
                    min_region_ratio
                ),
                "segnet_margin_bootstrap_worst_class_index": worst_class,
                "segnet_hard_birth_bootstrap_loss": (
                    hard_birth_total_loss / mx.maximum(hard_birth_active_count, eps)
                    + hard_birth_worst_loss
                ),
                "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass": (
                    total_unsolved
                ),
                "segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass": (
                    hard_birth_worst_unsolved
                ),
                "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio": (
                    hard_birth_min_ratio
                ),
                "segnet_hard_birth_bootstrap_worst_class_index": hard_birth_worst_class,
                "segnet_hard_birth_bootstrap_worst_loss_class_index": (
                    hard_birth_worst_loss_class
                ),
                "segnet_hard_birth_bootstrap_active_class_count": (
                    hard_birth_active_count
                ),
            })
            return metrics

        def _metric_tensors(model_obj: Any) -> dict[str, Any]:
            pred0, pred1 = _predict_pair01(model_obj)
            rgb0 = mx.mean((pred0 - target0) * (pred0 - target0))  # type: ignore[union-attr]
            rgb1 = mx.mean((pred1 - target1) * (pred1 - target1))  # type: ignore[union-attr]
            target_region_rgb1 = mx.mean(  # type: ignore[union-attr]
                target_region_weight_map * (pred1 - target1) * (pred1 - target1)
            )
            pred_yuv0 = rgb_to_yuv6_mlx(pred0 * 255.0) / 255.0
            pred_yuv1 = rgb_to_yuv6_mlx(pred1 * 255.0) / 255.0
            yuv0 = mx.mean((pred_yuv0 - ref_yuv0) * (pred_yuv0 - ref_yuv0))  # type: ignore[union-attr]
            yuv1 = mx.mean((pred_yuv1 - ref_yuv1) * (pred_yuv1 - ref_yuv1))  # type: ignore[union-attr]
            pred_temporal = pred_yuv1 - pred_yuv0
            temporal = mx.mean(  # type: ignore[union-attr]
                (pred_temporal - ref_temporal) * (pred_temporal - ref_temporal)
            )
            metrics = {
                "rgb_pair_mse": 0.5 * (rgb0 + rgb1),
                "rgb_frame0_mse": rgb0,
                "rgb_frame1_mse": rgb1,
                "target_region_rgb_frame1_mse": target_region_rgb1,
                "yuv6_pair_mse": 0.5 * (yuv0 + yuv1),
                "yuv6_temporal_delta_mse": temporal,
                "output_rgb_std": 0.5 * (mx.std(pred0) + mx.std(pred1)),  # type: ignore[union-attr]
                "target_rgb_std": 0.5 * (mx.std(target0) + mx.std(target1)),  # type: ignore[union-attr]
                "output_yuv6_temporal_delta_std": mx.std(pred_temporal),  # type: ignore[union-attr]
                "target_yuv6_temporal_delta_std": mx.std(ref_temporal),  # type: ignore[union-attr]
            }
            metrics.update(_segnet_margin_bootstrap_tensors(pred1))
            return metrics

        def _contrast_floor_loss(metrics: Mapping[str, Any]) -> Any:
            eps = 1.0e-6
            rgb_target = metrics["target_rgb_std"]
            temporal_target = metrics["target_yuv6_temporal_delta_std"]
            rgb_deficit = mx.maximum(  # type: ignore[union-attr]
                0.0,
                rgb_std_floor * rgb_target - metrics["output_rgb_std"],
            ) / (rgb_target + eps)
            temporal_deficit = mx.maximum(  # type: ignore[union-attr]
                0.0,
                temporal_std_floor * temporal_target
                - metrics["output_yuv6_temporal_delta_std"],
            ) / (temporal_target + eps)
            return rgb_deficit * rgb_deficit + temporal_deficit * temporal_deficit

        def _loss_fn(model_obj: Any) -> Any:
            metrics = _metric_tensors(model_obj)
            return (
                weights["rgb_weight"] * metrics["rgb_pair_mse"]
                + weights["yuv6_weight"] * metrics["yuv6_pair_mse"]
                + weights["temporal_delta_weight"] * metrics["yuv6_temporal_delta_mse"]
                + weights["contrast_floor_weight"] * _contrast_floor_loss(metrics)
                + weights["target_region_bootstrap_weight"]
                * metrics["target_region_rgb_frame1_mse"]
                + weights["segnet_margin_bootstrap_weight"]
                * metrics["segnet_margin_bootstrap_loss"]
                + weights["segnet_hard_birth_bootstrap_weight"]
                * metrics["segnet_hard_birth_bootstrap_loss"]
            )

        def _scalar_metrics(model_obj: Any) -> dict[str, float]:
            metrics = _metric_tensors(model_obj)
            contrast_floor = _contrast_floor_loss(metrics)
            mx.eval(list(metrics.values()))  # type: ignore[union-attr]
            mx.eval(contrast_floor)  # type: ignore[union-attr]
            out = {
                name: float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
                for name, value in metrics.items()
            }
            out["contrast_floor_loss"] = float(contrast_floor.item())
            out["output_rgb_std_ratio"] = float(
                out["output_rgb_std"] / max(out["target_rgb_std"], 1.0e-6)
            )
            out["output_yuv6_temporal_delta_std_ratio"] = float(
                out["output_yuv6_temporal_delta_std"]
                / max(out["target_yuv6_temporal_delta_std"], 1.0e-6)
            )
            return out

        before = _scalar_metrics(self)
        loss_and_grad_fn = nn.value_and_grad(self, _loss_fn)  # type: ignore[union-attr]
        if tree_unflatten is None:
            raise RuntimeError("MLX tree_unflatten unavailable despite successful MLX import")

        def _snapshot_parameters() -> list[tuple[Any, Any]]:
            return [
                (raw_name, None if leaf is None else mx.array(leaf))  # type: ignore[union-attr]
                for raw_name, leaf in tree_flatten(self.parameters())  # type: ignore[operator]
            ]

        def _restore_parameters(snapshot: list[tuple[Any, Any]]) -> None:
            self.update(
                tree_unflatten(
                    [
                        (raw_name, None if leaf is None else mx.array(leaf))  # type: ignore[union-attr]
                        for raw_name, leaf in snapshot
                    ]
                )
            )
            mx.eval(self.parameters())  # type: ignore[union-attr]

        def _loss_scalar(model_obj: Any) -> float:
            value = _loss_fn(model_obj)
            mx.eval(value)  # type: ignore[union-attr]
            return float(value.item())

        def _contrast_floor_scalar(model_obj: Any) -> float:
            value = _contrast_floor_loss(_metric_tensors(model_obj))
            mx.eval(value)  # type: ignore[union-attr]
            return float(value.item())

        def _segnet_bootstrap_score_debt_scalar(model_obj: Any) -> float:
            if not segnet_live_bootstrap_requested:
                return 0.0
            value = _metric_tensors(model_obj)[
                "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass"
            ]
            mx.eval(value)  # type: ignore[union-attr]
            return float(value.item())

        def _apply_gradient_step(
            *,
            base_snapshot: list[tuple[Any, Any]],
            grads_tree: Any,
            step_lr: float,
        ) -> tuple[int, list[str], int]:
            grad_by_name = dict(tree_flatten(grads_tree))  # type: ignore[operator]
            updated: list[tuple[Any, Any]] = []
            applied = 0
            applied_names: list[str] = []
            scoped_out = 0
            for raw_name, leaf in base_snapshot:
                grad = grad_by_name.get(raw_name)
                if leaf is None or grad is None:
                    updated.append((raw_name, leaf))
                    continue
                if not _bootstrap_update_name_allowed(raw_name):
                    updated.append((raw_name, leaf))
                    scoped_out += 1
                    continue
                param = mx.array(leaf)  # type: ignore[union-attr]
                update = param - float(step_lr) * (grad + wd * param)
                updated.append((raw_name, update))
                applied += 1
                applied_names.append(_flat_param_name(raw_name))
            self.update(tree_unflatten(updated))
            mx.eval(self.parameters())  # type: ignore[union-attr]
            return applied, applied_names, scoped_out

        current_loss = _loss_scalar(self)
        current_contrast_floor = _contrast_floor_scalar(self)
        current_segnet_score_debt = _segnet_bootstrap_score_debt_scalar(self)
        preserve_contrast_floor = bool(weights["contrast_floor_weight"] > 0.0)
        preserve_segnet_score_debt = bool(segnet_live_bootstrap_requested)
        loss_history: list[float] = [current_loss]
        grad_norm_history: list[float] = []
        clipped_count = 0
        accepted_step_count = 0
        rejected_step_count = 0
        contrast_floor_rejected_step_count = 0
        segnet_score_debt_rejected_step_count = 0
        backtracking_attempt_count = 0
        bootstrap_scoped_out_gradient_tensor_count = 0
        accepted_bootstrap_update_tensor_names: set[str] = set()
        min_accepted_step_lr = lr
        max_backtracking_attempts = 8
        for _step in range(step_count):
            base_snapshot = _snapshot_parameters()
            loss, grads = loss_and_grad_fn(self)
            mx.eval(loss)  # type: ignore[union-attr]
            if clip is not None:
                grads, total_norm = mlx_optim.clip_grad_norm(grads, clip)
                mx.eval(total_norm)  # type: ignore[union-attr]
                grad_norm = float(total_norm.item())
                clipped_count += int(grad_norm > clip)
            else:
                grad_norm = float("nan")
            grad_norm_history.append(grad_norm)
            accepted = False
            step_lr = lr
            applied_tensor_count = 0
            applied_tensor_names: list[str] = []
            for _attempt in range(max_backtracking_attempts):
                backtracking_attempt_count += 1
                (
                    applied_tensor_count,
                    applied_tensor_names,
                    scoped_out_tensor_count,
                ) = _apply_gradient_step(
                    base_snapshot=base_snapshot,
                    grads_tree=grads,
                    step_lr=step_lr,
                )
                bootstrap_scoped_out_gradient_tensor_count += int(
                    scoped_out_tensor_count
                )
                candidate_loss = _loss_scalar(self)
                candidate_contrast_floor = _contrast_floor_scalar(self)
                candidate_segnet_score_debt = _segnet_bootstrap_score_debt_scalar(self)
                contrast_floor_ok = (
                    not preserve_contrast_floor
                    or candidate_contrast_floor <= current_contrast_floor + 1.0e-9
                )
                segnet_score_debt_ok = (
                    not preserve_segnet_score_debt
                    or candidate_segnet_score_debt
                    <= current_segnet_score_debt + 1.0e-6
                )
                if (
                    candidate_loss <= current_loss + 1.0e-12
                    and contrast_floor_ok
                    and segnet_score_debt_ok
                ):
                    current_loss = candidate_loss
                    current_contrast_floor = candidate_contrast_floor
                    current_segnet_score_debt = candidate_segnet_score_debt
                    min_accepted_step_lr = min(min_accepted_step_lr, step_lr)
                    loss_history.append(candidate_loss)
                    accepted = True
                    accepted_step_count += 1
                    accepted_bootstrap_update_tensor_names.update(applied_tensor_names)
                    break
                if not contrast_floor_ok:
                    contrast_floor_rejected_step_count += 1
                if not segnet_score_debt_ok:
                    segnet_score_debt_rejected_step_count += 1
                _restore_parameters(base_snapshot)
                step_lr *= 0.5
            if not accepted:
                _restore_parameters(base_snapshot)
                rejected_step_count += 1
                loss_history.append(current_loss)
                if applied_tensor_count == 0:
                    break
        after = _scalar_metrics(self)

        def _improvement(key: str) -> float:
            return float(before[key] - after[key])

        return {
            "schema": "hi_nerv_scorer_domain_bootstrap.v1",
            "enabled": True,
            "method": "bounded_archive_charged_backtracking_rgb_yuv6_temporal_prefit",
            "steps": step_count,
            "learning_rate": lr,
            "weight_decay": wd,
            "grad_clip_max_norm": clip,
            "rgb_std_min_ratio": rgb_std_floor,
            "yuv6_temporal_std_min_ratio": temporal_std_floor,
            "grad_clip_clipped_step_count": int(clipped_count),
            "bootstrap_update_scope": bootstrap_update_scope,
            "bootstrap_update_allowlist_patterns": list(
                archive_charged_bootstrap_tensors
            ),
            "bootstrap_update_applied_tensor_count": len(
                accepted_bootstrap_update_tensor_names
            ),
            "bootstrap_update_applied_tensor_names": sorted(
                accepted_bootstrap_update_tensor_names
            ),
            "bootstrap_scoped_out_gradient_tensor_count": int(
                bootstrap_scoped_out_gradient_tensor_count
            ),
            "accepted_step_count": int(accepted_step_count),
            "rejected_step_count": int(rejected_step_count),
            "contrast_floor_preserving_acceptance": preserve_contrast_floor,
            "contrast_floor_rejected_step_count": int(
                contrast_floor_rejected_step_count
            ),
            "segnet_score_debt_preserving_acceptance": preserve_segnet_score_debt,
            "segnet_score_debt_rejected_step_count": int(
                segnet_score_debt_rejected_step_count
            ),
            "backtracking_attempt_count": int(backtracking_attempt_count),
            "max_backtracking_attempts_per_step": int(max_backtracking_attempts),
            "min_accepted_step_learning_rate": float(min_accepted_step_lr),
            "pair_count": int(idx.shape[0]),
            **weights,
            "loss_history_first": loss_history[0] if loss_history else None,
            "loss_history_last": loss_history[-1] if loss_history else None,
            "grad_norm_first": grad_norm_history[0] if grad_norm_history else None,
            "grad_norm_last": grad_norm_history[-1] if grad_norm_history else None,
            "metrics_before": before,
            "metrics_after": after,
            "target_region_bootstrap": {
                "schema": "hi_nerv_scorer_domain_bootstrap_target_region_waterfill.v1",
                "enabled": bool(weights["target_region_bootstrap_weight"] > 0.0),
                "weight": float(weights["target_region_bootstrap_weight"]),
                "map_source": str(target_region_metadata.get("source")),
                "metadata": target_region_metadata,
            },
            "segnet_margin_bootstrap": segnet_margin_metadata,
            "segnet_hard_birth_bootstrap": segnet_hard_birth_metadata,
            "rgb_pair_mse_delta": _improvement("rgb_pair_mse"),
            "rgb_frame1_mse_delta": _improvement("rgb_frame1_mse"),
            "target_region_rgb_frame1_mse_delta": _improvement(
                "target_region_rgb_frame1_mse"
            ),
            "segnet_margin_bootstrap_loss_delta": _improvement(
                "segnet_margin_bootstrap_loss"
            ),
            "segnet_margin_bootstrap_argmax_disagreement_delta": _improvement(
                "segnet_margin_bootstrap_argmax_disagreement"
            ),
            "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass_delta": (
                _improvement(
                    "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass"
                )
            ),
            "segnet_margin_bootstrap_candidate_target_class_min_ratio_delta": float(
                after["segnet_margin_bootstrap_candidate_target_class_min_ratio"]
                - before["segnet_margin_bootstrap_candidate_target_class_min_ratio"]
            ),
            "segnet_hard_birth_bootstrap_loss_delta": _improvement(
                "segnet_hard_birth_bootstrap_loss"
            ),
            "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass_delta": (
                _improvement(
                    "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass"
                )
            ),
            "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio_delta": float(
                after["segnet_hard_birth_bootstrap_candidate_target_class_min_ratio"]
                - before["segnet_hard_birth_bootstrap_candidate_target_class_min_ratio"]
            ),
            "yuv6_pair_mse_delta": _improvement("yuv6_pair_mse"),
            "yuv6_temporal_delta_mse_delta": _improvement("yuv6_temporal_delta_mse"),
            "contrast_floor_loss_delta": _improvement("contrast_floor_loss"),
            "output_rgb_std_ratio_delta": float(
                after["output_rgb_std_ratio"] - before["output_rgb_std_ratio"]
            ),
            "output_yuv6_temporal_delta_std_ratio_delta": float(
                after["output_yuv6_temporal_delta_std_ratio"]
                - before["output_yuv6_temporal_delta_std_ratio"]
            ),
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": archive_charged_bootstrap_tensors,
            "target_surface": "segnet_last_frame_rgb_and_posenet_pr95_yuv6_pair",
            "human_visual_fidelity_objective": False,
        }

    def __call__(self, pair_indices: Any) -> Any:
        z_c = mx.take(self.latents_coarse, pair_indices, axis=0)  # type: ignore[union-attr]
        z_m = mx.take(self.latents_mid, pair_indices, axis=0)  # type: ignore[union-attr]
        z_f = mx.take(self.latents_fine, pair_indices, axis=0)  # type: ignore[union-attr]

        fake_quant_bits = self._fake_quant_bits()
        fake_quant_bits_by_name = self._fake_quant_bits_by_name()
        h = _linear_with_params(
            self.latent_embed,
            z_c,
            fake_quant_bits=fake_quant_bits,
            fake_quant_bits_by_name=fake_quant_bits_by_name,
            weight_name="latent_embed.weight",
            bias_name="latent_embed.bias",
        )
        h = mx.reshape(  # type: ignore[union-attr]
            h,
            (
                -1,
                int(self.cfg.embed_dim),
                int(self.cfg.initial_grid_h),
                int(self.cfg.initial_grid_w),
            ),
        )
        h = mx.transpose(h, (0, 2, 3, 1))  # type: ignore[union-attr]
        for i, block in enumerate(self.blocks):
            h = block(
                h,
                fake_quant_bits=fake_quant_bits,
                fake_quant_bits_by_name=fake_quant_bits_by_name,
                name_prefix=f"blocks.{i}",
            )
            if bool(self.cfg.use_hierarchical_feature_grid):
                h = h + self.feature_grids[i](
                    pair_indices,
                    spatial_shape=(int(h.shape[1]), int(h.shape[2])),
                    fake_quant_bits=fake_quant_bits,
                    fake_quant_bits_by_name=fake_quant_bits_by_name,
                    name_prefix=f"feature_grids.{i}",
                )
            if bool(self.cfg.use_convnext_blocks):
                h = self.convnext_blocks[i](
                    h,
                    fake_quant_bits=fake_quant_bits,
                    fake_quant_bits_by_name=fake_quant_bits_by_name,
                    name_prefix=f"convnext_blocks.{i}",
                )
            if i == int(self.cfg.mid_injection_block_index):
                h = h + self.mid_injector(
                    z_m,
                    (int(h.shape[1]), int(h.shape[2])),
                    fake_quant_bits=fake_quant_bits,
                    fake_quant_bits_by_name=fake_quant_bits_by_name,
                    name_prefix="mid_injector.proj",
                )
            if i == int(self.cfg.fine_injection_block_index):
                h = h + self.fine_injector(
                    z_f,
                    (int(h.shape[1]), int(h.shape[2])),
                    fake_quant_bits=fake_quant_bits,
                    fake_quant_bits_by_name=fake_quant_bits_by_name,
                    name_prefix="fine_injector.proj",
                )

        h = _bilinear_resize_nhwc(
            h,
            int(self.cfg.output_height),
            int(self.cfg.output_width),
        )
        rgb_0 = mx.sigmoid(
            _conv2d_with_params(
                self.head_rgb_0,
                h,
                fake_quant_bits=fake_quant_bits,
                fake_quant_bits_by_name=fake_quant_bits_by_name,
                weight_name="head_rgb_0.weight",
                bias_name="head_rgb_0.bias",
            )
        ) * 255.0  # type: ignore[union-attr]
        rgb_1 = mx.sigmoid(
            _conv2d_with_params(
                self.head_rgb_1,
                h,
                fake_quant_bits=fake_quant_bits,
                fake_quant_bits_by_name=fake_quant_bits_by_name,
                weight_name="head_rgb_1.weight",
                bias_name="head_rgb_1.bias",
            )
        ) * 255.0  # type: ignore[union-attr]
        pair_nhwc = mx.stack([rgb_0, rgb_1], axis=1)  # type: ignore[union-attr]
        return mx.transpose(pair_nhwc, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    def reconstruct_pair(self, pair_indices: Any) -> tuple[Any, Any]:
        """Return ``(rgb_0, rgb_1)`` NCHW in ``[0, 1]`` for bridge tests."""

        pair = self(pair_indices) / 255.0
        return pair[:, 0], pair[:, 1]

    def num_parameters(self) -> int:
        _require_mlx()
        total = 0
        for _name, arr in tree_flatten(self.parameters()):  # type: ignore[operator]
            total += int(np.prod(arr.shape))
        return total

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export tensors using the PyTorch HiNeRV state_dict layout."""

        _require_mlx()
        out: dict[str, np.ndarray] = {
            "latents_coarse": np.asarray(self.latents_coarse, dtype=np.float32).copy(),
            "latents_mid": np.asarray(self.latents_mid, dtype=np.float32).copy(),
            "latents_fine": np.asarray(self.latents_fine, dtype=np.float32).copy(),
            "latent_embed.weight": np.asarray(
                self.latent_embed.weight, dtype=np.float32
            ).copy(),
            "latent_embed.bias": np.asarray(
                self.latent_embed.bias, dtype=np.float32
            ).copy(),
        }
        for i, block in enumerate(self.blocks):
            conv = block.conv
            out[f"blocks.{i}.conv.weight"] = np.transpose(
                np.asarray(conv.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"blocks.{i}.conv.bias"] = np.asarray(
                conv.bias, dtype=np.float32
            ).copy()
        if bool(self.cfg.use_hierarchical_feature_grid):
            for i, grid_module in enumerate(self.feature_grids):
                for level, grid in enumerate(grid_module.grids):
                    out[f"feature_grids.{i}.grids.{level}"] = np.asarray(
                        grid,
                        dtype=np.float32,
                    ).copy()
                proj = grid_module.proj
                out[f"feature_grids.{i}.proj.weight"] = np.transpose(
                    np.asarray(proj.weight, dtype=np.float32),
                    (0, 3, 1, 2),
                ).copy()
                out[f"feature_grids.{i}.proj.bias"] = np.asarray(
                    proj.bias,
                    dtype=np.float32,
                ).copy()
        if bool(self.cfg.use_convnext_blocks):
            for i, block in enumerate(self.convnext_blocks):
                out[f"convnext_blocks.{i}.dwconv.weight"] = np.transpose(
                    np.asarray(block.dwconv.weight, dtype=np.float32),
                    (0, 3, 1, 2),
                ).copy()
                out[f"convnext_blocks.{i}.dwconv.bias"] = np.asarray(
                    block.dwconv.bias,
                    dtype=np.float32,
                ).copy()
                out[f"convnext_blocks.{i}.norm.weight"] = np.asarray(
                    block.norm.weight,
                    dtype=np.float32,
                ).copy()
                out[f"convnext_blocks.{i}.norm.bias"] = np.asarray(
                    block.norm.bias,
                    dtype=np.float32,
                ).copy()
                out[f"convnext_blocks.{i}.pwconv1.weight"] = np.transpose(
                    np.asarray(block.pwconv1.weight, dtype=np.float32),
                    (0, 3, 1, 2),
                ).copy()
                out[f"convnext_blocks.{i}.pwconv1.bias"] = np.asarray(
                    block.pwconv1.bias,
                    dtype=np.float32,
                ).copy()
                out[f"convnext_blocks.{i}.pwconv2.weight"] = np.transpose(
                    np.asarray(block.pwconv2.weight, dtype=np.float32),
                    (0, 3, 1, 2),
                ).copy()
                out[f"convnext_blocks.{i}.pwconv2.bias"] = np.asarray(
                    block.pwconv2.bias,
                    dtype=np.float32,
                ).copy()
                out[f"convnext_blocks.{i}.gamma"] = np.asarray(
                    block.gamma,
                    dtype=np.float32,
                ).copy()
        for name, injector in (
            ("mid_injector.proj", self.mid_injector),
            ("fine_injector.proj", self.fine_injector),
        ):
            out[f"{name}.weight"] = np.asarray(
                injector.proj.weight, dtype=np.float32
            ).copy()
            out[f"{name}.bias"] = np.asarray(
                injector.proj.bias, dtype=np.float32
            ).copy()
        for head_name in ("head_rgb_0", "head_rgb_1"):
            head = getattr(self, head_name)
            out[f"{head_name}.weight"] = np.transpose(
                np.asarray(head.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"{head_name}.bias"] = np.asarray(
                head.bias, dtype=np.float32
            ).copy()
        return out


__all__ = [
    "MLX_EVIDENCE_GRADE",
    "SCHEMA_VERSION",
    "ConvNeXtBlockMLX",
    "HierarchicalFeatureGridMLX",
    "HinervSubstrateMLX",
    "trilinear_upsample_mlx",
]
