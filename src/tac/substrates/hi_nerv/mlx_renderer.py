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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tac.substrates.hi_nerv.architecture import HinervConfig

try:  # pragma: no cover - exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - non-Apple CI import guard.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "hi_nerv_mlx_renderer_v1"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"
_MIN_POSITIVE_FP16_SCALE = 5.960464477539063e-08


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


def _linear_with_params(layer: Any, x: Any, *, fake_quant_bits: int | None) -> Any:
    _require_mlx()
    weight = layer.weight
    bias = layer.bias
    if fake_quant_bits is not None:
        weight = _fake_quant_symmetric_ste(weight, bits=int(fake_quant_bits))
        bias = _fake_quant_symmetric_ste(bias, bits=int(fake_quant_bits))
    return x @ mx.transpose(weight) + bias  # type: ignore[union-attr]


def _conv2d_with_params(layer: Any, x: Any, *, fake_quant_bits: int | None) -> Any:
    _require_mlx()
    weight = layer.weight
    bias = layer.bias
    if fake_quant_bits is not None:
        weight = _fake_quant_symmetric_ste(weight, bits=int(fake_quant_bits))
        bias = _fake_quant_symmetric_ste(bias, bits=int(fake_quant_bits))
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
    ) -> Any:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        sampled = [
            trilinear_upsample_mlx(
                grid,
                pair_indices,
                num_pairs=self.num_pairs,
                target_h=h,
                target_w=w,
                local_scale=self.local_scale,
            )
            for grid in self.grids
        ]
        enc = mx.concatenate(sampled, axis=-1)  # type: ignore[union-attr]
        return _conv2d_with_params(
            self.proj,
            enc,
            fake_quant_bits=fake_quant_bits,
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

    def __call__(self, x: Any, *, fake_quant_bits: int | None = None) -> Any:
        residual = x
        y = _conv2d_with_params(self.dwconv, x, fake_quant_bits=fake_quant_bits)
        y = self.norm(y)
        y = self.act(
            _conv2d_with_params(self.pwconv1, y, fake_quant_bits=fake_quant_bits)
        )
        y = _conv2d_with_params(self.pwconv2, y, fake_quant_bits=fake_quant_bits)
        gamma = self.gamma
        if fake_quant_bits is not None:
            gamma = _fake_quant_symmetric_ste(gamma, bits=int(fake_quant_bits))
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

    def __call__(self, x: Any, *, fake_quant_bits: int | None = None) -> Any:
        conv = _conv2d_with_params(self.conv, x, fake_quant_bits=fake_quant_bits)
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
    ) -> Any:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        v = _linear_with_params(self.proj, latent, fake_quant_bits=fake_quant_bits)
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
        self.decoder_fake_quant_bits = 8
        self._siren_init()

    def configure_decoder_fake_quant_forward(
        self,
        *,
        enabled: bool,
        quant_bits: int = 8,
    ) -> None:
        """Enable decoder-weight fake-quant forward QAT for training.

        This affects forward computation only; archive/export still measures
        real packet bytes independently.
        """

        bits = int(quant_bits)
        if bits < 1 or bits > 16:
            raise ValueError(f"quant_bits must be in [1, 16]; got {quant_bits}")
        self.decoder_fake_quant_forward_enabled = bool(enabled)
        self.decoder_fake_quant_bits = bits

    def _fake_quant_bits(self) -> int | None:
        return (
            int(self.decoder_fake_quant_bits)
            if bool(self.decoder_fake_quant_forward_enabled)
            else None
        )

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

    def __call__(self, pair_indices: Any) -> Any:
        z_c = mx.take(self.latents_coarse, pair_indices, axis=0)  # type: ignore[union-attr]
        z_m = mx.take(self.latents_mid, pair_indices, axis=0)  # type: ignore[union-attr]
        z_f = mx.take(self.latents_fine, pair_indices, axis=0)  # type: ignore[union-attr]

        fake_quant_bits = self._fake_quant_bits()
        h = _linear_with_params(
            self.latent_embed,
            z_c,
            fake_quant_bits=fake_quant_bits,
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
            h = block(h, fake_quant_bits=fake_quant_bits)
            if bool(self.cfg.use_hierarchical_feature_grid):
                h = h + self.feature_grids[i](
                    pair_indices,
                    spatial_shape=(int(h.shape[1]), int(h.shape[2])),
                    fake_quant_bits=fake_quant_bits,
                )
            if bool(self.cfg.use_convnext_blocks):
                h = self.convnext_blocks[i](h, fake_quant_bits=fake_quant_bits)
            if i == int(self.cfg.mid_injection_block_index):
                h = h + self.mid_injector(
                    z_m,
                    (int(h.shape[1]), int(h.shape[2])),
                    fake_quant_bits=fake_quant_bits,
                )
            if i == int(self.cfg.fine_injection_block_index):
                h = h + self.fine_injector(
                    z_f,
                    (int(h.shape[1]), int(h.shape[2])),
                    fake_quant_bits=fake_quant_bits,
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
            )
        ) * 255.0  # type: ignore[union-attr]
        rgb_1 = mx.sigmoid(
            _conv2d_with_params(
                self.head_rgb_1,
                h,
                fake_quant_bits=fake_quant_bits,
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
