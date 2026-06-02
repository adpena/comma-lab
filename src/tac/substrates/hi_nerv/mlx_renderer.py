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

    def __call__(self, x: Any) -> Any:
        return _pixel_shuffle_2x_nhwc(mx.sin(self.w * self.conv(x)))  # type: ignore[union-attr]


class _LatentInjectorMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Project per-pair latent into a spatial additive NHWC tensor."""

    def __init__(self, latent_dim: int, channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.channels = int(channels)
        self.proj: Any = nn.Linear(int(latent_dim), int(channels))  # type: ignore[union-attr]

    def __call__(self, latent: Any, spatial_shape: tuple[int, int]) -> Any:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        v = self.proj(latent)
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
        self._siren_init()

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

        h = self.latent_embed(z_c)
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
            h = block(h)
            if i == int(self.cfg.mid_injection_block_index):
                h = h + self.mid_injector(z_m, (int(h.shape[1]), int(h.shape[2])))
            if i == int(self.cfg.fine_injection_block_index):
                h = h + self.fine_injector(z_f, (int(h.shape[1]), int(h.shape[2])))

        h = _bilinear_resize_nhwc(
            h,
            int(self.cfg.output_height),
            int(self.cfg.output_width),
        )
        rgb_0 = mx.sigmoid(self.head_rgb_0(h)) * 255.0  # type: ignore[union-attr]
        rgb_1 = mx.sigmoid(self.head_rgb_1(h)) * 255.0  # type: ignore[union-attr]
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
    "HinervSubstrateMLX",
]
