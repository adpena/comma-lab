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

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
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


def _receiver_uint8_roundtrip_ste_nhwc01(rgb: Any) -> Any:
    """Clamp/round receiver RGB while preserving first-order gradients.

    The contest evaluator consumes inflated byte images, not continuous MLX
    tensors.  HiNeRV hard-birth losses therefore need to see this receiver
    surface during training, otherwise the optimizer can spend steps in a
    subquantum basin that changes float RGB but not the scored uint8 image.
    """

    _require_mlx()
    clamped = mx.clip(rgb, 0.0, 1.0)  # type: ignore[union-attr]
    receiver = mx.round(clamped * 255.0) / 255.0  # type: ignore[union-attr]
    return rgb + mx.stop_gradient(receiver - rgb)  # type: ignore[union-attr]


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
    return (
        mx.conv2d(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        + bias
    )  # type: ignore[union-attr]


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
        raise ValueError(f"grid local scale {(scale_h, scale_w)} != {int(local_scale)}")
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


def _write_pair_local_smoke_artifact(
    payload: Mapping[str, Any],
    *,
    artifact_dir: str | Path | None,
    pair_index: int,
    adapter_sha256: str,
) -> dict[str, Any] | None:
    if artifact_dir is None:
        return None
    out_dir = Path(artifact_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_schema = "hinerv_pair_local_actuator_smoke_artifact.v1"
    artifact_payload = {
        "schema": artifact_schema,
        "family": "hi_nerv",
        "payload": dict(payload),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    data = (
        json.dumps(
            artifact_payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = out_dir / (f"hinerv_pair_local_actuator_smoke_pair{int(pair_index):06d}_{str(adapter_sha256)[:12]}.json")
    path.write_bytes(data)
    return {
        "schema": "hinerv_pair_local_actuator_smoke_artifact_record.v1",
        "artifact_schema": artifact_schema,
        "path": path.as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_hard_region_miner_input_bundle(
    *,
    output_dir: str | Path | None,
    target_labels_bhw: np.ndarray,
    candidate_argmax_bhw: np.ndarray,
    target_margin_bhw: np.ndarray,
    pair_indices: np.ndarray,
) -> dict[str, Any]:
    schema = "hi_nerv_hard_region_miner_input_bundle.v2"
    if output_dir is None:
        return {
            "schema": schema,
            "written": False,
            "reason": "hard_region_miner_output_dir_not_requested",
        }
    out = Path(output_dir).expanduser().resolve(strict=False)
    posix = out.as_posix()
    if posix.startswith(("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(
            "refusing to write hard-region miner inputs to transient tmp tier: "
            f"{out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "hi_nerv_hard_region_miner_inputs.npz"
    manifest_path = out / "hi_nerv_hard_region_miner_inputs_manifest.json"
    labels = np.asarray(target_labels_bhw, dtype=np.int32)
    argmax = np.asarray(candidate_argmax_bhw, dtype=np.int32)
    margin = np.asarray(target_margin_bhw, dtype=np.float32)
    pairs = np.asarray(pair_indices, dtype=np.int64)
    if labels.ndim != 3 or argmax.shape != labels.shape or margin.shape != labels.shape:
        raise ValueError(
            "hard-region miner compact bundle requires matching BHW labels, "
            f"argmax, and target margin; got labels={labels.shape} argmax={argmax.shape} "
            f"margin={margin.shape}"
        )
    np.savez_compressed(
        bundle_path,
        target_labels_bhw=labels,
        candidate_argmax_bhw=argmax,
        target_margin_bhw=margin,
        pair_indices=pairs,
    )
    bundle_bytes = bundle_path.read_bytes()
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    plan_path = out / "hi_nerv_hard_region_plan.json"
    plan_blockers: list[str] = []
    plan_region_count = 0
    try:
        from tac.analysis.hinerv_hard_region_miner import (
            build_hard_region_mining_plan,
            mine_hard_regions_from_margin_map,
        )

        regions = mine_hard_regions_from_margin_map(
            labels,
            argmax,
            margin,
            top_k=8,
            min_region_pixels=1,
        )
        plan = build_hard_region_mining_plan(
            regions,
            source=bundle_path.as_posix(),
            top_k=8,
        )
        plan_region_count = int(plan.get("region_count") or 0)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        plan_blockers.append(
            f"hard_region_mining_plan_write_failed:{type(exc).__name__}:{exc}"
        )
    command = (
        "uv run python tools/mine_hinerv_hard_regions.py "
        f"--target-labels {bundle_path.as_posix()}::target_labels_bhw "
        f"--candidate-argmax {bundle_path.as_posix()}::candidate_argmax_bhw "
        f"--target-margin {bundle_path.as_posix()}::target_margin_bhw "
        f"--output {(out / 'hi_nerv_hard_region_plan.json').as_posix()}"
    )
    manifest = {
        "schema": schema,
        "written": True,
        "bundle_path": bundle_path.as_posix(),
        "bundle_sha256": bundle_sha256,
        "bundle_bytes": int(bundle_path.stat().st_size),
        "manifest_path": manifest_path.as_posix(),
        "keys": {
            "target_labels_bhw": list(labels.shape),
            "candidate_argmax_bhw": list(argmax.shape),
            "target_margin_bhw": list(margin.shape),
            "pair_indices": list(pairs.shape),
        },
        "max_rank_persisted_array": 3,
        "compact_sufficient_statistic": "target_margin_bhw=max_other_logit-target_class_logit",
        "full_logits_persisted": False,
        "pair_indices": [int(value) for value in pairs.reshape(-1).tolist()],
        "hard_region_plan_path": plan_path.as_posix(),
        "hard_region_plan_region_count": int(plan_region_count),
        "mine_command": command,
        "blockers": plan_blockers,
        "producer": "hinerv_target_region_birth",
        "consumer": "tools/mine_hinerv_hard_regions.py",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _target_region_portfolio_coverage_row(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from tac.analysis.hinerv_hard_region_miner import (
        OUTCOME_BIRTH_ACCEPTED,
        OUTCOME_BIRTH_REJECTED,
        build_representative_coverage_row,
    )

    outcomes: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        region_key = attempt.get("region_key")
        key_parts: tuple[int, ...] = ()
        if isinstance(region_key, Sequence) and not isinstance(region_key, (str, bytes)):
            key_parts = tuple(int(value) for value in region_key)
        region = {
            "batch_index": key_parts[0] if len(key_parts) >= 1 else int(attempt.get("batch_index") or 0),
            "class_index": int(attempt.get("class_index") or 0),
            "region_label": key_parts[2] if len(key_parts) >= 3 else int(attempt.get("region_label") or 1),
            "region_pixel_count": int(attempt.get("region_pixel_count") or 0),
        }
        accepted = bool(attempt.get("accepted")) and int(attempt.get("wrong_to_target_count") or 0) > 0
        if accepted:
            outcomes.append(
                {
                    "region": region,
                    "outcome": OUTCOME_BIRTH_ACCEPTED,
                    "first_failing_surface": None,
                }
            )
        else:
            blockers = [str(value) for value in attempt.get("blockers") or [] if str(value)]
            outcomes.append(
                {
                    "region": region,
                    "outcome": OUTCOME_BIRTH_REJECTED,
                    "first_failing_surface": blockers[0] if blockers else "birth_not_accepted",
                }
            )
    if not outcomes:
        return build_representative_coverage_row([])
    return build_representative_coverage_row(outcomes)


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
            raise ValueError("fine_injection_block_index must be > mid_injection_block_index")
        mx.random.seed(int(getattr(cfg, "init_seed", 0)))  # type: ignore[union-attr]

        self.latents_coarse = (
            mx.random.normal(  # type: ignore[union-attr]
                shape=(int(cfg.num_pairs), int(cfg.latent_dim_coarse))
            )
            * 0.02
        )
        self.latents_mid = (
            mx.random.normal(  # type: ignore[union-attr]
                shape=(int(cfg.num_pairs), int(cfg.latent_dim_mid))
            )
            * 0.02
        )
        self.latents_fine = (
            mx.random.normal(  # type: ignore[union-attr]
                shape=(int(cfg.num_pairs), int(cfg.latent_dim_fine))
            )
            * 0.02
        )

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
        self.decoder_fake_quant_forward_enabled = bool(enabled) and (not bool(stage_controlled))

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
        self.decoder_fake_quant_forward_enabled = bool(self.decoder_fake_quant_forward_configured_enabled) and (
            active_bool or not bool(self.decoder_fake_quant_forward_stage_controlled)
        )
        self.decoder_fake_quant_forward_last_stage = {
            "schema": "hi_nerv_decoder_fake_quant_stage_control.v1",
            "source": str(source),
            "stage_name": str(stage_name),
            "stage_epoch": None if stage_epoch is None else int(stage_epoch),
            "stage_index": None if stage_index is None else int(stage_index),
            "stage_qat_active": active_bool,
            "stage_controlled": bool(self.decoder_fake_quant_forward_stage_controlled),
            "configured_enabled": bool(self.decoder_fake_quant_forward_configured_enabled),
            "forward_active": bool(self.decoder_fake_quant_forward_enabled),
            "global_quant_bits": (None if self.decoder_fake_quant_bits is None else int(self.decoder_fake_quant_bits)),
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

        report = build_decoder_waterfill_fake_quant_forward_plan(decoder_weight_waterfill_plan)
        actuation_blockers = [str(blocker) for blocker in report.get("actuation_blockers") or []]
        if actuation_blockers:
            raise ValueError(
                f"decoder_weight_waterfill_plan is not safe for train-time fake quantization: {actuation_blockers}"
            )
        per_tensor_bits = {str(name): int(bits) for name, bits in dict(report.get("per_tensor_bits") or {}).items()}
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
            "fallback_quant_bits": (None if fallback_quant_bits is None else int(fallback_quant_bits)),
            "configured_global_quant_bits": (
                None if self.decoder_fake_quant_bits is None else int(self.decoder_fake_quant_bits)
            ),
            "configured_per_tensor_bits": dict(self.decoder_fake_quant_bits_by_name),
        }

    def _fake_quant_bits(self) -> int | None:
        return (
            int(self.decoder_fake_quant_bits)
            if bool(self.decoder_fake_quant_forward_enabled) and self.decoder_fake_quant_bits is not None
            else None
        )

    def _fake_quant_bits_by_name(self) -> dict[str, int]:
        if not bool(self.decoder_fake_quant_forward_enabled):
            return {}
        return dict(self.decoder_fake_quant_bits_by_name)

    def _siren_init(self) -> None:
        w = float(self.cfg.sin_frequency)
        bound = _siren_uniform_bound(int(self.cfg.latent_dim_coarse), w)
        self.latent_embed.update(
            {
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound,
                    high=bound,
                    shape=self.latent_embed.weight.shape,
                ),
                "bias": mx.zeros_like(self.latent_embed.bias),  # type: ignore[union-attr]
            }
        )
        for block in self.blocks:
            conv = block.conv
            kh, kw = int(conv.weight.shape[1]), int(conv.weight.shape[2])
            fan_in = int(conv.weight.shape[3]) * kh * kw
            bound = _siren_uniform_bound(fan_in, w)
            conv.update(
                {
                    "weight": mx.random.uniform(  # type: ignore[union-attr]
                        low=-bound,
                        high=bound,
                        shape=conv.weight.shape,
                    ),
                    "bias": mx.zeros_like(conv.bias),  # type: ignore[union-attr]
                }
            )
        for injector in (self.mid_injector, self.fine_injector):
            bound = _siren_uniform_bound(injector.latent_dim, w)
            injector.proj.update(
                {
                    "weight": mx.random.uniform(  # type: ignore[union-attr]
                        low=-bound,
                        high=bound,
                        shape=injector.proj.weight.shape,
                    ),
                    "bias": mx.zeros_like(injector.proj.bias),  # type: ignore[union-attr]
                }
            )
        for head in (self.head_rgb_0, self.head_rgb_1):
            kh, kw = int(head.weight.shape[1]), int(head.weight.shape[2])
            fan_in = int(head.weight.shape[3]) * kh * kw
            bound = _siren_uniform_bound(fan_in, w)
            head.update(
                {
                    "weight": mx.random.uniform(  # type: ignore[union-attr]
                        low=-bound,
                        high=bound,
                        shape=head.weight.shape,
                    ),
                    "bias": mx.zeros_like(head.bias),  # type: ignore[union-attr]
                }
            )

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
                    f"target RGB tensors must be NHWC with 3 channels; got shape={tuple(int(v) for v in target.shape)}"
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
            "target_rgb_0_mean": [float(v) for v in np.asarray(mean_0, dtype=np.float32).reshape(-1)],
            "target_rgb_1_mean": [float(v) for v in np.asarray(mean_1, dtype=np.float32).reshape(-1)],
            "head_rgb_0_bias": [float(v) for v in np.asarray(self.head_rgb_0.bias, dtype=np.float32).reshape(-1)],
            "head_rgb_1_bias": [float(v) for v in np.asarray(self.head_rgb_1.bias, dtype=np.float32).reshape(-1)],
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
            raise ValueError(f"min_output_std must be finite and positive; got {min_output_std!r}")
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
                    f"{name} must be NHWC with 3 channels; got shape={tuple(int(v) for v in target.shape)}"
                )
            if int(target.shape[0]) != int(idx.shape[0]):
                raise ValueError(
                    f"{name} batch {int(target.shape[0])} must match pair_indices length {int(idx.shape[0])}"
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
            {"weight": self.head_rgb_0.weight * mx.reshape(gain0.astype(self.head_rgb_0.weight.dtype), (3, 1, 1, 1))}
        )
        self.head_rgb_1.update(
            {"weight": self.head_rgb_1.weight * mx.reshape(gain1.astype(self.head_rgb_1.weight.dtype), (3, 1, 1, 1))}
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

    def build_pair_local_actuator_smoke_from_targets(
        self,
        target_rgb_0: Any,
        target_rgb_1: Any,
        *,
        pair_indices: Any,
        learning_rate: float = 1.0e-2,
        artifact_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Probe true per-pair latent controllability."""

        _require_mlx()
        if tree_flatten is None:
            raise RuntimeError("MLX tree_flatten unavailable despite successful MLX import")
        if tree_unflatten is None:
            raise RuntimeError("MLX tree_unflatten unavailable despite successful MLX import")
        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim != 1:
            raise ValueError(f"pair_indices must be 1-D; got shape {tuple(idx.shape)}")
        pair_count = int(idx.shape[0])
        if pair_count <= 0:
            raise ValueError("pair_indices must contain at least one pair")
        target0 = mx.array(target_rgb_0, dtype=mx.float32)  # type: ignore[union-attr]
        target1 = mx.array(target_rgb_1, dtype=mx.float32)  # type: ignore[union-attr]
        if target0.ndim != 4 or target1.ndim != 4 or int(target0.shape[-1]) != 3 or int(target1.shape[-1]) != 3:
            raise ValueError("target_rgb_0 and target_rgb_1 must be NHWC RGB tensors")
        if int(target0.shape[0]) != pair_count or int(target1.shape[0]) != pair_count:
            raise ValueError("target RGB batch dimension must match pair_indices")
        target_index = int(np.asarray(idx[0], dtype=np.int32).reshape(-1)[0])
        if target_index < 0 or target_index >= int(self.latents_fine.shape[0]):
            raise ValueError(f"target pair index out of range: {target_index}")
        non_target_index = next(
            (int(value) for value in range(int(self.latents_fine.shape[0])) if int(value) != target_index),
            None,
        )
        step_lr = float(learning_rate)
        if not math.isfinite(step_lr) or step_lr <= 0.0:
            raise ValueError(f"learning_rate must be finite and positive; got {learning_rate}")
        target_pair = mx.array([target_index], dtype=mx.int32)  # type: ignore[union-attr]
        target0_one = target0[:1]
        target1_one = target1[:1]

        def _predict_nhwc01(pair_ids: Any) -> tuple[Any, Any]:
            pair01 = self(pair_ids) / 255.0
            return (
                mx.transpose(pair01[:, 0], (0, 2, 3, 1)),  # type: ignore[union-attr]
                mx.transpose(pair01[:, 1], (0, 2, 3, 1)),  # type: ignore[union-attr]
            )

        before_target0, before_target1 = _predict_nhwc01(target_pair)
        before_target0 = mx.stop_gradient(mx.array(before_target0))  # type: ignore[union-attr]
        before_target1 = mx.stop_gradient(mx.array(before_target1))  # type: ignore[union-attr]
        mx.eval(before_target0, before_target1)  # type: ignore[union-attr]
        if non_target_index is None:
            before_non_target0 = before_non_target1 = None
            non_target_pair = None
        else:
            non_target_pair = mx.array([non_target_index], dtype=mx.int32)  # type: ignore[union-attr]
            before_non_target0, before_non_target1 = _predict_nhwc01(non_target_pair)
            before_non_target0 = mx.stop_gradient(mx.array(before_non_target0))  # type: ignore[union-attr]
            before_non_target1 = mx.stop_gradient(mx.array(before_non_target1))  # type: ignore[union-attr]
            mx.eval(before_non_target0, before_non_target1)  # type: ignore[union-attr]

        def _loss_fn(model_obj: Any) -> Any:
            pair01 = model_obj(target_pair) / 255.0
            pred0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))  # type: ignore[union-attr]
            pred1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))  # type: ignore[union-attr]
            return 0.5 * (
                mx.mean((pred0 - target0_one) * (pred0 - target0_one))  # type: ignore[union-attr]
                + mx.mean((pred1 - target1_one) * (pred1 - target1_one))  # type: ignore[union-attr]
            )

        def _latent_row_digest(row: np.ndarray) -> str:
            row = np.ascontiguousarray(row)
            digest = hashlib.sha256()
            digest.update(b"latents_fine")
            digest.update(np.asarray([target_index], dtype=np.int64).tobytes())
            digest.update(str(row.dtype).encode("utf-8"))
            digest.update(np.asarray(row.shape, dtype=np.int64).tobytes())
            digest.update(row.tobytes())
            return digest.hexdigest()

        def _uint8_change_stats(
            before0: Any,
            before1: Any,
            after0: Any,
            after1: Any,
        ) -> dict[str, Any]:
            before_np = np.concatenate(
                [
                    np.asarray(before0, dtype=np.float32).reshape(-1),
                    np.asarray(before1, dtype=np.float32).reshape(-1),
                ]
            )
            after_np = np.concatenate(
                [
                    np.asarray(after0, dtype=np.float32).reshape(-1),
                    np.asarray(after1, dtype=np.float32).reshape(-1),
                ]
            )
            before_u8 = np.rint(np.clip(before_np * 255.0, 0.0, 255.0)).astype(np.uint8)
            after_u8 = np.rint(np.clip(after_np * 255.0, 0.0, 255.0)).astype(np.uint8)
            changed = before_u8 != after_u8
            abs_delta = np.abs(after_u8.astype(np.int16) - before_u8.astype(np.int16))
            return {
                "changed_count": int(np.count_nonzero(changed)),
                "changed_fraction": (float(np.count_nonzero(changed)) / float(changed.size) if changed.size else 0.0),
                "delta_abs_max_uint8": int(abs_delta.max()) if abs_delta.size else 0,
            }

        original_latents_fine = mx.array(self.latents_fine)  # type: ignore[union-attr]
        loss_and_grad_fn = nn.value_and_grad(self, _loss_fn)  # type: ignore[union-attr]
        loss, grads = loss_and_grad_fn(self)
        mx.eval(loss)  # type: ignore[union-attr]
        grad_fine = None
        for raw_name, grad in tree_flatten(grads):  # type: ignore[operator]
            name = ".".join(str(part) for part in raw_name) if isinstance(raw_name, (tuple, list)) else str(raw_name)
            if name == "latents_fine":
                grad_fine = grad
                break
        if grad_fine is None:
            return {
                "schema": "hinerv_pair_local_actuator_smoke.v1",
                "family": "hi_nerv",
                "execution_attempted": True,
                "execution_completed": False,
                "blockers": ["hinerv_pair_local_latents_fine_gradient_missing"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        grad_np = np.asarray(grad_fine, dtype=np.float32)
        original_np = np.asarray(original_latents_fine, dtype=np.float32)
        original_row = np.ascontiguousarray(original_np[target_index])
        original_row_sha256 = _latent_row_digest(original_row)
        grad_row = np.ascontiguousarray(grad_np[target_index])
        grad_norm = float(np.linalg.norm(grad_row))
        updated_np = np.array(original_np, copy=True)
        updated_np[target_index] = updated_np[target_index] - step_lr * grad_row
        updated_row = np.ascontiguousarray(updated_np[target_index])
        adapter_sha256 = _latent_row_digest(updated_row)
        adapter_bytes = int(updated_row.nbytes)
        try:
            self.update({"latents_fine": mx.array(updated_np)})  # type: ignore[union-attr]
            mx.eval(self.parameters())  # type: ignore[union-attr]
            after_target0, after_target1 = _predict_nhwc01(target_pair)
            selected_delta_l2_tensor = mx.sqrt(  # type: ignore[union-attr]
                0.5
                * (
                    mx.mean((after_target0 - before_target0) ** 2)  # type: ignore[union-attr]
                    + mx.mean((after_target1 - before_target1) ** 2)  # type: ignore[union-attr]
                )
            )
            selected_delta_max_abs_tensor = mx.max(  # type: ignore[union-attr]
                mx.concatenate(  # type: ignore[union-attr]
                    [
                        mx.reshape(mx.abs(after_target0 - before_target0), (-1,)),  # type: ignore[union-attr]
                        mx.reshape(mx.abs(after_target1 - before_target1), (-1,)),  # type: ignore[union-attr]
                    ]
                )
            )
            if non_target_pair is None:
                non_target_delta_l2_tensor = mx.array(0.0, dtype=mx.float32)  # type: ignore[union-attr]
            else:
                after_non_target0, after_non_target1 = _predict_nhwc01(non_target_pair)
                non_target_delta_l2_tensor = mx.sqrt(  # type: ignore[union-attr]
                    0.5
                    * (
                        mx.mean((after_non_target0 - before_non_target0) ** 2)  # type: ignore[operator,union-attr]
                        + mx.mean((after_non_target1 - before_non_target1) ** 2)  # type: ignore[operator,union-attr]
                    )
                )
            mx.eval(
                selected_delta_l2_tensor,
                selected_delta_max_abs_tensor,
                non_target_delta_l2_tensor,
            )  # type: ignore[union-attr]
            selected_delta_l2 = float(selected_delta_l2_tensor.item())
            selected_delta_max_abs = float(selected_delta_max_abs_tensor.item())
            non_target_delta_l2 = float(non_target_delta_l2_tensor.item())
            target_uint8_stats = _uint8_change_stats(
                before_target0,
                before_target1,
                after_target0,
                after_target1,
            )
            if non_target_pair is None:
                non_target_uint8_stats = {
                    "changed_count": 0,
                    "changed_fraction": 0.0,
                    "delta_abs_max_uint8": 0,
                }
            else:
                non_target_uint8_stats = _uint8_change_stats(
                    before_non_target0,
                    before_non_target1,
                    after_non_target0,
                    after_non_target1,
                )
        finally:
            self.update({"latents_fine": original_latents_fine})
            mx.eval(self.parameters())  # type: ignore[union-attr]
        restored_np = np.asarray(self.latents_fine, dtype=np.float32)
        restored_row_sha256 = _latent_row_digest(restored_np[target_index])
        state_restored = restored_row_sha256 == original_row_sha256
        output_delta_l2_per_byte = selected_delta_l2 / float(adapter_bytes) if adapter_bytes > 0 else 0.0
        receiver_uint8_half_step_normalized = 0.5 / 255.0
        selected_delta_max_abs_uint8 = selected_delta_max_abs * 255.0
        receiver_uint8_crossing_potential = selected_delta_max_abs >= receiver_uint8_half_step_normalized
        blockers: list[str] = []
        if not math.isfinite(float(loss.item())):
            blockers.append("hinerv_pair_local_loss_not_finite")
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            blockers.append("hinerv_pair_local_grad_norm_not_positive")
        if not math.isfinite(selected_delta_l2) or selected_delta_l2 <= 0.0:
            blockers.append("hinerv_pair_local_output_delta_not_positive")
        if not receiver_uint8_crossing_potential:
            blockers.append("hinerv_pair_local_output_delta_below_uint8_half_step")
        receiver_uint8_changed = int(target_uint8_stats["changed_count"]) > 0
        if not receiver_uint8_changed:
            blockers.append("hinerv_pair_local_receiver_uint8_unchanged")
        non_target_uint8_unchanged = int(non_target_uint8_stats["changed_count"]) == 0
        if not non_target_uint8_unchanged:
            blockers.append("hinerv_pair_local_non_target_uint8_delta_detected")
        pair_locality_verified = non_target_delta_l2 <= 1.0e-12
        if not pair_locality_verified:
            blockers.append("hinerv_pair_local_non_target_pair_delta_detected")
        if not state_restored:
            blockers.append("hinerv_pair_local_state_restore_failed")
        section_output_rows = [
            {
                "section": "pair_local_latents_fine",
                "bytes": adapter_bytes,
                "grad_norm": grad_norm,
                "output_delta_l2": selected_delta_l2,
                "output_delta_l2_per_byte": output_delta_l2_per_byte,
                "value_semantics": "receiver_output_l2_per_byte_not_score_value",
                "score_value_per_byte_measured": False,
                "score_claim": False,
            }
        ]
        summary = {
            "schema": "pr95_scorer_atom_actuator_execution_evidence.v1",
            "family": "hi_nerv",
            "pair_local_smoke_schema": "hinerv_pair_local_actuator_smoke.v1",
            "actuator_kind": "pair_local_latent_row",
            "actuator_tensor_name": "latents_fine",
            "updated_tensor_names": ["latents_fine"],
            "state_mutation_scope": "latents_fine_row_only",
            "runtime_sidecar_bytes": 0,
            "pair_local_adapter_bytes": adapter_bytes,
            "pair_local_adapter_sha256": adapter_sha256,
            "pair_local_grad_norm": grad_norm,
            "pair_local_grad_norm_by_group": {"latents_fine": grad_norm},
            "pair_local_output_delta_l2": selected_delta_l2,
            "pair_local_output_delta_max_abs": selected_delta_max_abs,
            "pair_local_output_delta_max_abs_uint8": selected_delta_max_abs_uint8,
            "receiver_uint8_half_step_normalized": receiver_uint8_half_step_normalized,
            "receiver_uint8_crossing_potential": receiver_uint8_crossing_potential,
            "receiver_uint8_changed": receiver_uint8_changed,
            "receiver_uint8_changed_count": int(target_uint8_stats["changed_count"]),
            "receiver_uint8_changed_fraction": float(target_uint8_stats["changed_fraction"]),
            "receiver_uint8_delta_abs_max": int(target_uint8_stats["delta_abs_max_uint8"]),
            "non_target_pair_receiver_uint8_changed_count": int(non_target_uint8_stats["changed_count"]),
            "non_target_pair_receiver_uint8_delta_abs_max": int(non_target_uint8_stats["delta_abs_max_uint8"]),
            "pair_locality_verified": pair_locality_verified,
            "non_target_pair_output_delta_l2_max": non_target_delta_l2,
            "state_restored_after_smoke": state_restored,
            "pair_local_latents_fine_original_row_sha256": original_row_sha256,
            "pair_local_latents_fine_restored_row_sha256": restored_row_sha256,
            "section_output_delta_per_byte_rows": section_output_rows,
            "section_value_per_byte_rows": [],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        payload: dict[str, Any] = {
            "schema": "hinerv_pair_local_actuator_smoke.v1",
            "family": "hi_nerv",
            "axis_tag": MLX_EVIDENCE_GRADE,
            "evidence_grade": "macOS-MLX-research-signal",
            "execution_attempted": True,
            "execution_completed": not blockers,
            "pair_indices": [target_index],
            "pair_index_semantics": "source_non_overlapping_seq2_pair_index",
            "actuator": {
                "kind": "pair_local_latent_row",
                "tensor_name": "latents_fine",
                "row_indices": [target_index],
                "adapter_bytes": adapter_bytes,
                "adapter_sha256": adapter_sha256,
                "archive_charged_decoder_tensors": ["latents_fine"],
                "runtime_sidecar_bytes": 0,
            },
            "gradient": {
                "value_and_grad_checked": True,
                "loss_finite": math.isfinite(float(loss.item())),
                "grad_finite": math.isfinite(grad_norm),
                "pair_local_grad_norm": grad_norm,
                "pair_local_grad_norm_by_group": {"latents_fine": grad_norm},
                "updated_tensor_names": ["latents_fine"],
            },
            "output_delta": {
                "pair_local_output_delta_l2": selected_delta_l2,
                "pair_local_output_delta_max_abs": selected_delta_max_abs,
                "pair_local_output_delta_max_abs_uint8": selected_delta_max_abs_uint8,
                "receiver_uint8_half_step_normalized": receiver_uint8_half_step_normalized,
                "receiver_uint8_crossing_potential": receiver_uint8_crossing_potential,
                "receiver_uint8_changed": receiver_uint8_changed,
                "receiver_uint8_changed_count": int(target_uint8_stats["changed_count"]),
                "receiver_uint8_changed_fraction": float(target_uint8_stats["changed_fraction"]),
                "receiver_uint8_delta_abs_max": int(target_uint8_stats["delta_abs_max_uint8"]),
                "non_target_pair_receiver_uint8_changed_count": int(non_target_uint8_stats["changed_count"]),
                "non_target_pair_receiver_uint8_delta_abs_max": int(non_target_uint8_stats["delta_abs_max_uint8"]),
                "non_target_pair_output_delta_l2_max": non_target_delta_l2,
                "pair_locality_verified": pair_locality_verified,
            },
            "state_restore": {
                "checked_tensor_name": "latents_fine",
                "checked_row_indices": [target_index],
                "state_restored_after_smoke": state_restored,
                "original_row_sha256": original_row_sha256,
                "mutated_row_sha256": adapter_sha256,
                "restored_row_sha256": restored_row_sha256,
            },
            "pair_local_adapter_bytes": adapter_bytes,
            "pair_local_adapter_sha256": adapter_sha256,
            "pair_local_grad_norm": grad_norm,
            "pair_local_output_delta_l2": selected_delta_l2,
            "section_output_delta_per_byte_rows": section_output_rows,
            "section_value_per_byte_rows": [],
            "blockers": blockers,
            "summary_for_pr95_guard": summary if not blockers else None,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        artifact_record = _write_pair_local_smoke_artifact(
            payload,
            artifact_dir=artifact_dir,
            pair_index=target_index,
            adapter_sha256=adapter_sha256,
        )
        if artifact_record:
            payload["pair_local_smoke_artifact"] = artifact_record
            if isinstance(payload.get("summary_for_pr95_guard"), dict):
                payload["summary_for_pr95_guard"].update(
                    {
                        "pair_local_smoke_artifact_schema": artifact_record["artifact_schema"],
                        "pair_local_smoke_artifact_path": artifact_record["path"],
                        "pair_local_smoke_artifact_sha256": artifact_record["sha256"],
                        "pair_local_smoke_artifact_bytes": artifact_record["bytes"],
                    }
                )
        return payload

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
        pair_local_smoke_artifact_dir: str | Path | None = None,
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
            "segnet_hard_birth_bootstrap_weight": float(segnet_hard_birth_bootstrap_weight),
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
                f"segnet_margin_bootstrap_floor must be finite and non-negative; got {segnet_margin_bootstrap_floor}"
            )
        segnet_hard_birth_floor = float(segnet_hard_birth_bootstrap_min_ratio_floor)
        if not math.isfinite(segnet_hard_birth_floor) or segnet_hard_birth_floor < 0.0 or segnet_hard_birth_floor > 1.0:
            raise ValueError(
                "segnet_hard_birth_bootstrap_min_ratio_floor must be finite "
                f"in [0, 1]; got {segnet_hard_birth_bootstrap_min_ratio_floor}"
            )
        wd = float(weight_decay)
        if not math.isfinite(wd) or wd < 0.0:
            raise ValueError(f"weight_decay must be finite and non-negative; got {weight_decay}")
        clip = None if grad_clip_max_norm is None else float(grad_clip_max_norm)
        if clip is not None and (not math.isfinite(clip) or clip <= 0.0):
            raise ValueError(f"grad_clip_max_norm must be None or finite and positive; got {grad_clip_max_norm}")

        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim != 1 or int(idx.shape[0]) <= 0:
            raise ValueError("pair_indices must be a non-empty rank-1 tensor")
        target0 = mx.array(target_rgb_0).astype(mx.float32)  # type: ignore[union-attr]
        target1 = mx.array(target_rgb_1).astype(mx.float32)  # type: ignore[union-attr]
        for name, target in (("target_rgb_0", target0), ("target_rgb_1", target1)):
            if target.ndim != 4 or int(target.shape[-1]) != 3:
                raise ValueError(
                    f"{name} must be NHWC with 3 channels; got shape={tuple(int(v) for v in target.shape)}"
                )
            if int(target.shape[0]) != int(idx.shape[0]):
                raise ValueError(
                    f"{name} batch {int(target.shape[0])} must match pair_indices length {int(idx.shape[0])}"
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
                        f"target_segnet_argmax_1 must have shape BHW or BHW1; got {tuple(int(v) for v in labels.shape)}"
                    )
                if tuple(int(v) for v in labels.shape) != tuple(int(v) for v in target1.shape[:3]):
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
                    class_rows.append(
                        {
                            "class_index": int(class_index),
                            "target_fraction": float(fraction.item()),
                            "inverse_sqrt_fraction_weight": float(class_weight.item()),
                        }
                    )
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
                mx.array(0.5, dtype=mx.float32) + mx.array(0.5, dtype=mx.float32) * normalized_saliency,
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
            weights["segnet_margin_bootstrap_weight"] > 0.0 or weights["segnet_hard_birth_bootstrap_weight"] > 0.0
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
                        f"target_segnet_argmax_1 must have shape BHW or BHW1; got {tuple(int(v) for v in labels.shape)}"
                    )
                if tuple(int(v) for v in labels.shape) != tuple(int(v) for v in target1.shape[:3]):
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
                "source": (
                    "receiver_uint8_roundtrip_ste_live_mlx_segnet_candidate_logits_against_frame1_target_argmax"
                ),
                "receiver_surface": "clamp_round_uint8_rgb_ste_nhwc01",
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
                "source": ("receiver_uint8_roundtrip_ste_live_mlx_segnet_candidate_logits_worst_target_class_birth"),
                "receiver_surface": "clamp_round_uint8_rgb_ste_nhwc01",
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

            receiver_pred1 = _receiver_uint8_roundtrip_ste_nhwc01(pred1)
            candidate_logits = segnet_margin_live_fn(receiver_pred1)
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
            max_target_class = int(np.asarray(mx.max(segnet_margin_target_labels), dtype=np.int32).item())
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
                target_mask = (segnet_margin_target_labels == class_index).astype(mx.float32)
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
                    mx.array(100.0, dtype=mx.float32) * target_fraction * mx.maximum(0.0, 1.0 - region_ratio)  # type: ignore[union-attr]
                )
                support_deficit = mx.maximum(0.0, 1.0 - support_ratio)  # type: ignore[union-attr]
                region_deficit = mx.maximum(0.0, 1.0 - region_ratio)  # type: ignore[union-attr]
                boost = mx.stop_gradient(  # type: ignore[union-attr]
                    active
                    * (1.0 + mx.minimum(32.0, score_weighted_unsolved) + 16.0 * support_deficit + 16.0 * region_deficit)
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
                hard_birth_active = active * (hard_birth_deficit > 0.0).astype(mx.float32)
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
                        mx.array(0.35, dtype=mx.float32) + mx.array(4.0, dtype=mx.float32) * segnet_hard_birth_floor,
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
                    + (1.0 + mx.minimum(32.0, mx.stop_gradient(score_weighted_unsolved))) * crossing_loss
                    + 8.0 * seed_crossing_loss
                )
                hard_birth_loss = hard_birth_boost * hard_birth_loss_raw
                hard_birth_total_loss = hard_birth_total_loss + hard_birth_loss
                hard_birth_active_count = hard_birth_active_count + hard_birth_active
                hard_birth_better_unsolved = score_weighted_unsolved > hard_birth_worst_unsolved
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
                metrics[f"{prefix}_score_weighted_unsolved_argmax_mass"] = score_weighted_unsolved
                birth_prefix = f"segnet_hard_birth_bootstrap_class_{class_index}"
                metrics[f"{birth_prefix}_loss"] = hard_birth_loss_raw
                metrics[f"{birth_prefix}_active"] = hard_birth_active
                metrics[f"{birth_prefix}_target_fraction"] = target_fraction
                metrics[f"{birth_prefix}_candidate_hard_fraction"] = hard_fraction
                metrics[f"{birth_prefix}_candidate_soft_fraction"] = candidate_soft_fraction
                metrics[f"{birth_prefix}_support_ratio"] = support_ratio
                metrics[f"{birth_prefix}_target_region_correct_ratio"] = region_ratio
                metrics[f"{birth_prefix}_birth_ratio"] = hard_birth_ratio
                metrics[f"{birth_prefix}_birth_deficit"] = hard_birth_deficit
                metrics[f"{birth_prefix}_target_prob_mean"] = target_prob_mean
                metrics[f"{birth_prefix}_target_prob_floor"] = target_prob_floor
                metrics[f"{birth_prefix}_target_prob_deficit"] = target_prob_deficit
                metrics[f"{birth_prefix}_seed_target_prob_mean"] = seed_target_prob_mean
                metrics[f"{birth_prefix}_seed_prob_floor"] = seed_prob_floor
                metrics[f"{birth_prefix}_seed_prob_deficit"] = seed_prob_deficit
                metrics[f"{birth_prefix}_soft_mass_floor"] = soft_mass_floor
                metrics[f"{birth_prefix}_soft_mass_log_ratio"] = soft_mass_log_ratio
                metrics[f"{birth_prefix}_frontier_margin"] = frontier_margin
                metrics[f"{birth_prefix}_seed_crossing_loss"] = seed_crossing_loss
                metrics[f"{birth_prefix}_score_weighted_unsolved_argmax_mass"] = score_weighted_unsolved
            metrics.update(
                {
                    "segnet_margin_bootstrap_loss": total_loss / mx.maximum(active_count, eps),
                    "segnet_margin_bootstrap_argmax_disagreement": mx.mean(  # type: ignore[union-attr]
                        (pred_class != segnet_margin_target_labels).astype(mx.float32)
                    ),
                    "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass": (total_unsolved),
                    "segnet_margin_bootstrap_score_weighted_worst_unsolved_argmax_mass": (worst_unsolved),
                    "segnet_margin_bootstrap_candidate_target_class_min_ratio": (min_region_ratio),
                    "segnet_margin_bootstrap_worst_class_index": worst_class,
                    "segnet_hard_birth_bootstrap_loss": (
                        hard_birth_total_loss / mx.maximum(hard_birth_active_count, eps) + hard_birth_worst_loss
                    ),
                    "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass": (total_unsolved),
                    "segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass": (
                        hard_birth_worst_unsolved
                    ),
                    "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio": (hard_birth_min_ratio),
                    "segnet_hard_birth_bootstrap_worst_class_index": hard_birth_worst_class,
                    "segnet_hard_birth_bootstrap_worst_loss_class_index": (hard_birth_worst_loss_class),
                    "segnet_hard_birth_bootstrap_active_class_count": (hard_birth_active_count),
                }
            )
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
                temporal_std_floor * temporal_target - metrics["output_yuv6_temporal_delta_std"],
            ) / (temporal_target + eps)
            return rgb_deficit * rgb_deficit + temporal_deficit * temporal_deficit

        def _loss_fn(model_obj: Any) -> Any:
            metrics = _metric_tensors(model_obj)
            return (
                weights["rgb_weight"] * metrics["rgb_pair_mse"]
                + weights["yuv6_weight"] * metrics["yuv6_pair_mse"]
                + weights["temporal_delta_weight"] * metrics["yuv6_temporal_delta_mse"]
                + weights["contrast_floor_weight"] * _contrast_floor_loss(metrics)
                + weights["target_region_bootstrap_weight"] * metrics["target_region_rgb_frame1_mse"]
                + weights["segnet_margin_bootstrap_weight"] * metrics["segnet_margin_bootstrap_loss"]
                + weights["segnet_hard_birth_bootstrap_weight"] * metrics["segnet_hard_birth_bootstrap_loss"]
            )

        def _scalar_metrics(model_obj: Any) -> dict[str, float]:
            metrics = _metric_tensors(model_obj)
            contrast_floor = _contrast_floor_loss(metrics)
            mx.eval(list(metrics.values()))  # type: ignore[union-attr]
            mx.eval(contrast_floor)  # type: ignore[union-attr]
            out = {name: float(np.asarray(value, dtype=np.float32).reshape(-1)[0]) for name, value in metrics.items()}
            out["contrast_floor_loss"] = float(contrast_floor.item())
            out["output_rgb_std_ratio"] = float(out["output_rgb_std"] / max(out["target_rgb_std"], 1.0e-6))
            out["output_yuv6_temporal_delta_std_ratio"] = float(
                out["output_yuv6_temporal_delta_std"] / max(out["target_yuv6_temporal_delta_std"], 1.0e-6)
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

        def _segnet_bootstrap_debt_scalars(model_obj: Any) -> tuple[float, float, float]:
            if not segnet_live_bootstrap_requested:
                return 0.0, 0.0, 1.0
            metrics = _metric_tensors(model_obj)
            if weights["segnet_hard_birth_bootstrap_weight"] > 0.0:
                total = metrics["segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass"]
                worst = metrics["segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass"]
                min_ratio = metrics["segnet_hard_birth_bootstrap_candidate_target_class_min_ratio"]
            else:
                total = metrics["segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass"]
                worst = metrics["segnet_margin_bootstrap_score_weighted_worst_unsolved_argmax_mass"]
                min_ratio = metrics["segnet_margin_bootstrap_candidate_target_class_min_ratio"]
            mx.eval(total, worst, min_ratio)  # type: ignore[union-attr]
            return float(total.item()), float(worst.item()), float(min_ratio.item())

        def _frame1_delta_max_abs_from_base(base_pred1: Any) -> float:
            _, candidate_pred1 = _predict_pair01(self)
            delta = mx.max(mx.abs(candidate_pred1 - base_pred1))  # type: ignore[union-attr]
            mx.eval(delta)  # type: ignore[union-attr]
            return float(delta.item())

        def _frame1_receiver_uint8_stats_from_base(
            base_pred1: Any,
        ) -> dict[str, float]:
            _, candidate_pred1 = _predict_pair01(self)
            base_u8 = mx.round(mx.clip(base_pred1, 0.0, 1.0) * 255.0)  # type: ignore[union-attr]
            candidate_u8 = mx.round(  # type: ignore[union-attr]
                mx.clip(candidate_pred1, 0.0, 1.0) * 255.0
            )
            delta_u8 = mx.abs(candidate_u8 - base_u8)  # type: ignore[union-attr]
            changed = (delta_u8 > 0.0).astype(mx.float32)
            changed_count = mx.sum(changed)  # type: ignore[union-attr]
            changed_fraction = mx.mean(changed)  # type: ignore[union-attr]
            max_abs_uint8 = mx.max(delta_u8)  # type: ignore[union-attr]
            float_delta = mx.max(mx.abs(candidate_pred1 - base_pred1))  # type: ignore[union-attr]
            mx.eval(changed_count, changed_fraction, max_abs_uint8, float_delta)  # type: ignore[union-attr]
            return {
                "changed_count": float(changed_count.item()),
                "changed_fraction": float(changed_fraction.item()),
                "max_abs_uint8": float(max_abs_uint8.item()),
                "float_delta_abs_max": float(float_delta.item()),
            }

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
        (
            current_segnet_score_debt,
            current_segnet_worst_debt,
            current_segnet_min_ratio,
        ) = _segnet_bootstrap_debt_scalars(self)
        preserve_contrast_floor = bool(weights["contrast_floor_weight"] > 0.0)
        preserve_segnet_score_debt = bool(segnet_live_bootstrap_requested)
        receiver_quantum_acceptance_enabled = bool(
            segnet_live_bootstrap_requested and weights["segnet_hard_birth_bootstrap_weight"] > 0.0
        )
        receiver_uint8_half_step_normalized = 0.5 / 255.0
        loss_history: list[float] = [current_loss]
        grad_norm_history: list[float] = []
        clipped_count = 0
        accepted_step_count = 0
        rejected_step_count = 0
        contrast_floor_rejected_step_count = 0
        segnet_score_debt_rejected_step_count = 0
        receiver_quantum_rejected_step_count = 0
        receiver_quantum_crossing_accepted_step_count = 0
        receiver_quantum_attempt_count = 0
        receiver_quantum_growth_attempt_count = 0
        receiver_quantum_shrink_attempt_count = 0
        hard_birth_argmax_progress_accepted_step_count = 0
        hard_birth_argmax_progress_rejected_step_count = 0
        hard_birth_worst_improved_total_spill_rejected_step_count = 0
        segnet_worst_debt_rejected_step_count = 0
        max_candidate_segnet_total_debt_reduction = 0.0
        max_candidate_segnet_worst_debt_reduction = 0.0
        max_candidate_segnet_min_ratio_increase = 0.0
        max_candidate_segnet_total_debt_spill_given_worst_improvement = 0.0
        max_accepted_segnet_total_debt_reduction = 0.0
        max_accepted_segnet_worst_debt_reduction = 0.0
        max_accepted_segnet_min_ratio_increase = 0.0
        max_candidate_frame1_delta_abs = 0.0
        max_candidate_frame1_delta_abs_uint8 = 0.0
        max_candidate_frame1_receiver_uint8_changed_count = 0.0
        max_candidate_frame1_receiver_uint8_changed_fraction = 0.0
        max_candidate_frame1_receiver_uint8_delta_abs = 0.0
        max_accepted_frame1_delta_abs = 0.0
        max_accepted_frame1_delta_abs_uint8 = 0.0
        max_accepted_frame1_receiver_uint8_changed_count = 0.0
        max_accepted_frame1_receiver_uint8_changed_fraction = 0.0
        max_accepted_frame1_receiver_uint8_delta_abs = 0.0
        backtracking_attempt_count = 0
        bootstrap_scoped_out_gradient_tensor_count = 0
        accepted_bootstrap_update_tensor_names: set[str] = set()
        min_accepted_step_lr = lr
        max_backtracking_attempts = 8
        max_receiver_quantum_growth_attempts = 20
        for _step in range(step_count):
            base_snapshot = _snapshot_parameters()
            _, base_pred1 = _predict_pair01(self)
            base_pred1 = mx.stop_gradient(mx.array(base_pred1))  # type: ignore[union-attr]
            mx.eval(base_pred1)  # type: ignore[union-attr]
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
            if receiver_quantum_acceptance_enabled:
                shrink_trials = [lr / (2.0**attempt) for attempt in range(max_backtracking_attempts, 0, -1)]
                growth_trials = [lr * (2.0**attempt) for attempt in range(max_receiver_quantum_growth_attempts)]
                step_lr_schedule = shrink_trials + growth_trials
            else:
                step_lr_schedule = [lr / (2.0**attempt) for attempt in range(max_backtracking_attempts)]
            for _attempt_index, step_lr in enumerate(step_lr_schedule):
                backtracking_attempt_count += 1
                if receiver_quantum_acceptance_enabled:
                    receiver_quantum_attempt_count += 1
                    if step_lr > lr:
                        receiver_quantum_growth_attempt_count += 1
                    elif step_lr < lr:
                        receiver_quantum_shrink_attempt_count += 1
                (
                    applied_tensor_count,
                    applied_tensor_names,
                    scoped_out_tensor_count,
                ) = _apply_gradient_step(
                    base_snapshot=base_snapshot,
                    grads_tree=grads,
                    step_lr=step_lr,
                )
                bootstrap_scoped_out_gradient_tensor_count += int(scoped_out_tensor_count)
                candidate_loss = _loss_scalar(self)
                candidate_contrast_floor = _contrast_floor_scalar(self)
                (
                    candidate_segnet_score_debt,
                    candidate_segnet_worst_debt,
                    candidate_segnet_min_ratio,
                ) = _segnet_bootstrap_debt_scalars(self)
                candidate_total_debt_reduction = current_segnet_score_debt - candidate_segnet_score_debt
                candidate_worst_debt_reduction = current_segnet_worst_debt - candidate_segnet_worst_debt
                candidate_min_ratio_increase = candidate_segnet_min_ratio - current_segnet_min_ratio
                max_candidate_segnet_total_debt_reduction = max(
                    max_candidate_segnet_total_debt_reduction,
                    candidate_total_debt_reduction,
                )
                max_candidate_segnet_worst_debt_reduction = max(
                    max_candidate_segnet_worst_debt_reduction,
                    candidate_worst_debt_reduction,
                )
                max_candidate_segnet_min_ratio_increase = max(
                    max_candidate_segnet_min_ratio_increase,
                    candidate_min_ratio_increase,
                )
                if candidate_worst_debt_reduction > 1.0e-6:
                    max_candidate_segnet_total_debt_spill_given_worst_improvement = max(
                        max_candidate_segnet_total_debt_spill_given_worst_improvement,
                        -candidate_total_debt_reduction,
                    )
                candidate_receiver_uint8_stats = _frame1_receiver_uint8_stats_from_base(base_pred1)
                candidate_frame1_delta_abs = float(candidate_receiver_uint8_stats["float_delta_abs_max"])
                candidate_frame1_delta_abs_uint8 = candidate_frame1_delta_abs * 255.0
                candidate_frame1_receiver_uint8_changed_count = float(candidate_receiver_uint8_stats["changed_count"])
                candidate_frame1_receiver_uint8_changed_fraction = float(
                    candidate_receiver_uint8_stats["changed_fraction"]
                )
                candidate_frame1_receiver_uint8_delta_abs = float(candidate_receiver_uint8_stats["max_abs_uint8"])
                max_candidate_frame1_delta_abs = max(
                    max_candidate_frame1_delta_abs,
                    candidate_frame1_delta_abs,
                )
                max_candidate_frame1_delta_abs_uint8 = max(
                    max_candidate_frame1_delta_abs_uint8,
                    candidate_frame1_delta_abs_uint8,
                )
                max_candidate_frame1_receiver_uint8_changed_count = max(
                    max_candidate_frame1_receiver_uint8_changed_count,
                    candidate_frame1_receiver_uint8_changed_count,
                )
                max_candidate_frame1_receiver_uint8_changed_fraction = max(
                    max_candidate_frame1_receiver_uint8_changed_fraction,
                    candidate_frame1_receiver_uint8_changed_fraction,
                )
                max_candidate_frame1_receiver_uint8_delta_abs = max(
                    max_candidate_frame1_receiver_uint8_delta_abs,
                    candidate_frame1_receiver_uint8_delta_abs,
                )
                contrast_floor_ok = (
                    not preserve_contrast_floor or candidate_contrast_floor <= current_contrast_floor + 1.0e-9
                )
                segnet_score_debt_ok = (
                    not preserve_segnet_score_debt or candidate_segnet_score_debt <= current_segnet_score_debt + 1.0e-6
                )
                segnet_worst_debt_ok = (
                    not preserve_segnet_score_debt or candidate_segnet_worst_debt <= current_segnet_worst_debt + 1.0e-6
                )
                hard_birth_argmax_progress_ok = (
                    not receiver_quantum_acceptance_enabled
                    or candidate_total_debt_reduction > 1.0e-6
                    or candidate_worst_debt_reduction > 1.0e-6
                    or candidate_min_ratio_increase > 1.0e-6
                )
                receiver_quantum_ok = (
                    not receiver_quantum_acceptance_enabled or candidate_frame1_receiver_uint8_changed_count > 0.0
                )
                loss_ok = candidate_loss <= current_loss + 1.0e-12 or (
                    receiver_quantum_acceptance_enabled and hard_birth_argmax_progress_ok
                )
                if (
                    loss_ok
                    and contrast_floor_ok
                    and segnet_score_debt_ok
                    and segnet_worst_debt_ok
                    and hard_birth_argmax_progress_ok
                    and receiver_quantum_ok
                ):
                    current_loss = candidate_loss
                    current_contrast_floor = candidate_contrast_floor
                    current_segnet_score_debt = candidate_segnet_score_debt
                    current_segnet_worst_debt = candidate_segnet_worst_debt
                    current_segnet_min_ratio = candidate_segnet_min_ratio
                    min_accepted_step_lr = min(min_accepted_step_lr, step_lr)
                    max_accepted_segnet_total_debt_reduction = max(
                        max_accepted_segnet_total_debt_reduction,
                        candidate_total_debt_reduction,
                    )
                    max_accepted_segnet_worst_debt_reduction = max(
                        max_accepted_segnet_worst_debt_reduction,
                        candidate_worst_debt_reduction,
                    )
                    max_accepted_segnet_min_ratio_increase = max(
                        max_accepted_segnet_min_ratio_increase,
                        candidate_min_ratio_increase,
                    )
                    max_accepted_frame1_delta_abs = max(
                        max_accepted_frame1_delta_abs,
                        candidate_frame1_delta_abs,
                    )
                    max_accepted_frame1_delta_abs_uint8 = max(
                        max_accepted_frame1_delta_abs_uint8,
                        candidate_frame1_delta_abs_uint8,
                    )
                    max_accepted_frame1_receiver_uint8_changed_count = max(
                        max_accepted_frame1_receiver_uint8_changed_count,
                        candidate_frame1_receiver_uint8_changed_count,
                    )
                    max_accepted_frame1_receiver_uint8_changed_fraction = max(
                        max_accepted_frame1_receiver_uint8_changed_fraction,
                        candidate_frame1_receiver_uint8_changed_fraction,
                    )
                    max_accepted_frame1_receiver_uint8_delta_abs = max(
                        max_accepted_frame1_receiver_uint8_delta_abs,
                        candidate_frame1_receiver_uint8_delta_abs,
                    )
                    loss_history.append(candidate_loss)
                    accepted = True
                    accepted_step_count += 1
                    hard_birth_argmax_progress_accepted_step_count += int(
                        receiver_quantum_acceptance_enabled and hard_birth_argmax_progress_ok
                    )
                    receiver_quantum_crossing_accepted_step_count += int(
                        receiver_quantum_acceptance_enabled and candidate_frame1_receiver_uint8_changed_count > 0.0
                    )
                    accepted_bootstrap_update_tensor_names.update(applied_tensor_names)
                    break
                if not contrast_floor_ok:
                    contrast_floor_rejected_step_count += 1
                if not segnet_score_debt_ok:
                    segnet_score_debt_rejected_step_count += 1
                    if candidate_worst_debt_reduction > 1.0e-6:
                        hard_birth_worst_improved_total_spill_rejected_step_count += 1
                if not segnet_worst_debt_ok:
                    segnet_worst_debt_rejected_step_count += 1
                if not hard_birth_argmax_progress_ok:
                    hard_birth_argmax_progress_rejected_step_count += 1
                if not receiver_quantum_ok:
                    receiver_quantum_rejected_step_count += 1
                _restore_parameters(base_snapshot)
            if not accepted:
                _restore_parameters(base_snapshot)
                rejected_step_count += 1
                loss_history.append(current_loss)
                if applied_tensor_count == 0:
                    break
        after = _scalar_metrics(self)
        pair_local_smoke = self.build_pair_local_actuator_smoke_from_targets(
            target0,
            target1,
            pair_indices=idx,
            learning_rate=min(lr, 1.0e-2),
            artifact_dir=pair_local_smoke_artifact_dir,
        )
        pr95_actuator_execution_evidence = (
            pair_local_smoke.get("summary_for_pr95_guard") if isinstance(pair_local_smoke, Mapping) else None
        )

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
            "bootstrap_update_allowlist_patterns": list(archive_charged_bootstrap_tensors),
            "bootstrap_update_applied_tensor_count": len(accepted_bootstrap_update_tensor_names),
            "bootstrap_update_applied_tensor_names": sorted(accepted_bootstrap_update_tensor_names),
            "bootstrap_scoped_out_gradient_tensor_count": int(bootstrap_scoped_out_gradient_tensor_count),
            "hinerv_pair_local_actuator_smoke": pair_local_smoke,
            "pr95_scorer_atom_actuator_execution_evidence": (pr95_actuator_execution_evidence),
            "accepted_step_count": int(accepted_step_count),
            "rejected_step_count": int(rejected_step_count),
            "contrast_floor_preserving_acceptance": preserve_contrast_floor,
            "contrast_floor_rejected_step_count": int(contrast_floor_rejected_step_count),
            "segnet_score_debt_preserving_acceptance": preserve_segnet_score_debt,
            "segnet_score_debt_rejected_step_count": int(segnet_score_debt_rejected_step_count),
            "segnet_worst_debt_rejected_step_count": int(segnet_worst_debt_rejected_step_count),
            "hard_birth_argmax_progress_accepted_step_count": int(hard_birth_argmax_progress_accepted_step_count),
            "hard_birth_argmax_progress_rejected_step_count": int(hard_birth_argmax_progress_rejected_step_count),
            "hard_birth_worst_improved_total_spill_rejected_step_count": int(
                hard_birth_worst_improved_total_spill_rejected_step_count
            ),
            "max_candidate_segnet_total_debt_reduction": float(max_candidate_segnet_total_debt_reduction),
            "max_candidate_segnet_worst_debt_reduction": float(max_candidate_segnet_worst_debt_reduction),
            "max_candidate_segnet_min_ratio_increase": float(max_candidate_segnet_min_ratio_increase),
            "max_candidate_segnet_total_debt_spill_given_worst_improvement": float(
                max_candidate_segnet_total_debt_spill_given_worst_improvement
            ),
            "max_accepted_segnet_total_debt_reduction": float(max_accepted_segnet_total_debt_reduction),
            "max_accepted_segnet_worst_debt_reduction": float(max_accepted_segnet_worst_debt_reduction),
            "max_accepted_segnet_min_ratio_increase": float(max_accepted_segnet_min_ratio_increase),
            "receiver_quantum_acceptance_enabled": receiver_quantum_acceptance_enabled,
            "receiver_uint8_half_step_normalized": receiver_uint8_half_step_normalized,
            "receiver_quantum_surface": "clamp_round_uint8_rgb_frame1",
            "receiver_quantum_attempt_count": int(receiver_quantum_attempt_count),
            "receiver_quantum_growth_attempt_count": int(receiver_quantum_growth_attempt_count),
            "receiver_quantum_shrink_attempt_count": int(receiver_quantum_shrink_attempt_count),
            "receiver_quantum_rejected_step_count": int(receiver_quantum_rejected_step_count),
            "receiver_quantum_crossing_accepted_step_count": int(receiver_quantum_crossing_accepted_step_count),
            "max_candidate_frame1_delta_abs": float(max_candidate_frame1_delta_abs),
            "max_candidate_frame1_delta_abs_uint8": float(max_candidate_frame1_delta_abs_uint8),
            "max_candidate_frame1_receiver_uint8_changed_count": float(
                max_candidate_frame1_receiver_uint8_changed_count
            ),
            "max_candidate_frame1_receiver_uint8_changed_fraction": float(
                max_candidate_frame1_receiver_uint8_changed_fraction
            ),
            "max_candidate_frame1_receiver_uint8_delta_abs": float(max_candidate_frame1_receiver_uint8_delta_abs),
            "max_accepted_frame1_delta_abs": float(max_accepted_frame1_delta_abs),
            "max_accepted_frame1_delta_abs_uint8": float(max_accepted_frame1_delta_abs_uint8),
            "max_accepted_frame1_receiver_uint8_changed_count": float(max_accepted_frame1_receiver_uint8_changed_count),
            "max_accepted_frame1_receiver_uint8_changed_fraction": float(
                max_accepted_frame1_receiver_uint8_changed_fraction
            ),
            "max_accepted_frame1_receiver_uint8_delta_abs": float(max_accepted_frame1_receiver_uint8_delta_abs),
            "backtracking_attempt_count": int(backtracking_attempt_count),
            "max_backtracking_attempts_per_step": int(max_backtracking_attempts),
            "max_receiver_quantum_growth_attempts_per_step": int(max_receiver_quantum_growth_attempts),
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
            "target_region_rgb_frame1_mse_delta": _improvement("target_region_rgb_frame1_mse"),
            "segnet_margin_bootstrap_loss_delta": _improvement("segnet_margin_bootstrap_loss"),
            "segnet_margin_bootstrap_argmax_disagreement_delta": _improvement(
                "segnet_margin_bootstrap_argmax_disagreement"
            ),
            "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass_delta": (
                _improvement("segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass")
            ),
            "segnet_margin_bootstrap_candidate_target_class_min_ratio_delta": float(
                after["segnet_margin_bootstrap_candidate_target_class_min_ratio"]
                - before["segnet_margin_bootstrap_candidate_target_class_min_ratio"]
            ),
            "segnet_hard_birth_bootstrap_loss_delta": _improvement("segnet_hard_birth_bootstrap_loss"),
            "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass_delta": (
                _improvement("segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass")
            ),
            "segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass_delta": (
                _improvement("segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass")
            ),
            "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio_delta": float(
                after["segnet_hard_birth_bootstrap_candidate_target_class_min_ratio"]
                - before["segnet_hard_birth_bootstrap_candidate_target_class_min_ratio"]
            ),
            "yuv6_pair_mse_delta": _improvement("yuv6_pair_mse"),
            "yuv6_temporal_delta_mse_delta": _improvement("yuv6_temporal_delta_mse"),
            "contrast_floor_loss_delta": _improvement("contrast_floor_loss"),
            "output_rgb_std_ratio_delta": float(after["output_rgb_std_ratio"] - before["output_rgb_std_ratio"]),
            "output_yuv6_temporal_delta_std_ratio_delta": float(
                after["output_yuv6_temporal_delta_std_ratio"] - before["output_yuv6_temporal_delta_std_ratio"]
            ),
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": archive_charged_bootstrap_tensors,
            "target_surface": "segnet_last_frame_rgb_and_posenet_pr95_yuv6_pair",
            "human_visual_fidelity_objective": False,
        }

    def fit_target_region_birth_from_segnet(
        self,
        *,
        scorer_teacher: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
        pair_indices: Any,
        target_segnet_argmax_1: Any | None = None,
        pose_teacher: Any | None = None,
        require_pose_trust: bool = False,
        max_steps: int = 64,
        learning_rate: float = 5.0e-4,
        target_min_region_ratio: float = 0.02,
        margin_tau: float = 0.3,
        margin_floor: float = 1.0,
        prob_floor: float = 0.55,
        lambda_prob_floor: float = 16.0,
        lambda_seed: float = 8.0,
        lambda_support_preserve: float = 64.0,
        lambda_outside_argmax_preserve: float = 16.0,
        lambda_already_won_hard_preserve: float = 0.0,
        lambda_already_won_rgb_preserve: float = 0.0,
        already_won_margin_floor: float = 1.0,
        lambda_pose_trust_preserve: float = 0.0,
        lambda_pose_target: float = 0.0,
        seed_temperature: float = 0.5,
        min_margin_mean_drop: float = 1.0e-6,
        max_pose_output_delta_l2: float = 0.05,
        exact_score_epsilon_batch_local: float = 1.0e-6,
        catastrophic_pose_relative_cap: float = 0.25,
        catastrophic_pose_score_regression_cap: float = 0.5,
        catastrophic_pose_output_l2_hard_cap: float | None = None,
        grad_clip_max_norm: float | None = None,
        min_region_pixels: int = 1,
        target_geometry_mode: str = "frontier_band",
        target_geometry_frontier_dilation: int = 1,
        emit_four_arm_ablation: bool = True,
        four_arm_ablation_compensation_steps: int = 8,
        target_region_portfolio_max_regions: int = 4,
        target_region_forced_key: Sequence[int] | None = None,
        require_fakequant_survival: bool = False,
        fakequant_survival_bits: int = 4,
        hard_region_miner_output_dir: str | Path | None = None,
        _target_region_portfolio_excluded: Sequence[Sequence[int]] = (),
        _target_region_portfolio_attempts: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Run a scoped hard-birth prefit on the worst SegNet target region.

        Class-aggregate losses can improve while ``target_min_ratio`` stays
        ``0.0`` because no connected component of the target class ever wins
        the receiver argmax (the subquantum / soft-mass-only failure mode).
        This actuator selects the single worst-debt connected region in exact
        contest score units, then drives a frontier-margin crossing inside
        that region only, updating only the late archive-charged tensors with
        local spatial leverage (``latents_fine`` / ``feature_grids.*`` /
        ``fine_injector.*`` / ``head_rgb_1.*``).  Steps are admitted only when
        the receiver uint8 image moves inside the region AND the region's
        hard ratio or median frontier margin improves AND the exact nonlinear
        Seg/Pose nonrate score improves without catastrophic pose blow-up.  The
        old raw pose-output cap is reported as counterfactual telemetry, not as
        the primary admission surface. The result is ordinary model state: no
        sidecar bytes, byte accounting unchanged.
        """

        _require_mlx()
        from scipy import ndimage as scipy_ndimage

        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx
        from tac.substrates.hi_nerv.target_region_birth import (
            allowed_birth_update_name,
            allowed_pose_compensation_update_name,
            birth_action_id,
            build_target_region_birth_receipt,
            find_target_region_debts,
            pose_trusted_birth_admission_decision,
            region_argmax_transition_counts,
            region_margin_stats,
            select_worst_target_region_with_mask,
        )

        step_count = int(max_steps)
        if step_count <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        lr = float(learning_rate)
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"learning_rate must be finite and positive; got {learning_rate}")
        for name, value in (
            ("target_min_region_ratio", float(target_min_region_ratio)),
            ("margin_tau", float(margin_tau)),
            ("margin_floor", float(margin_floor)),
            ("prob_floor", float(prob_floor)),
            ("lambda_prob_floor", float(lambda_prob_floor)),
            ("lambda_seed", float(lambda_seed)),
            ("lambda_support_preserve", float(lambda_support_preserve)),
            ("lambda_outside_argmax_preserve", float(lambda_outside_argmax_preserve)),
            ("lambda_already_won_hard_preserve", float(lambda_already_won_hard_preserve)),
            ("lambda_already_won_rgb_preserve", float(lambda_already_won_rgb_preserve)),
            ("already_won_margin_floor", float(already_won_margin_floor)),
            ("lambda_pose_trust_preserve", float(lambda_pose_trust_preserve)),
            ("lambda_pose_target", float(lambda_pose_target)),
            ("seed_temperature", float(seed_temperature)),
            ("min_margin_mean_drop", float(min_margin_mean_drop)),
            ("max_pose_output_delta_l2", float(max_pose_output_delta_l2)),
            ("exact_score_epsilon_batch_local", float(exact_score_epsilon_batch_local)),
            ("catastrophic_pose_relative_cap", float(catastrophic_pose_relative_cap)),
            ("catastrophic_pose_score_regression_cap", float(catastrophic_pose_score_regression_cap)),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        if catastrophic_pose_output_l2_hard_cap is not None and (
            not math.isfinite(float(catastrophic_pose_output_l2_hard_cap))
            or float(catastrophic_pose_output_l2_hard_cap) < 0.0
        ):
            raise ValueError(
                "catastrophic_pose_output_l2_hard_cap must be None or finite "
                f"and non-negative; got {catastrophic_pose_output_l2_hard_cap}"
            )
        if not 0.0 <= float(target_min_region_ratio) <= 1.0:
            raise ValueError(f"target_min_region_ratio must be in [0, 1]; got {target_min_region_ratio}")
        if float(margin_tau) <= 0.0 or float(seed_temperature) <= 0.0:
            raise ValueError("margin_tau and seed_temperature must be positive")
        geometry_mode = str(target_geometry_mode)
        valid_geometry_modes = {"frontier_band", "largest_unsolved_component", "full_tail"}
        if geometry_mode not in valid_geometry_modes:
            raise ValueError(
                "target_geometry_mode must be one of "
                f"{sorted(valid_geometry_modes)}; got {target_geometry_mode!r}"
            )
        frontier_dilation = int(target_geometry_frontier_dilation)
        if frontier_dilation < 1:
            raise ValueError(
                "target_geometry_frontier_dilation must be >= 1; "
                f"got {target_geometry_frontier_dilation}"
            )
        ablation_compensation_steps = int(four_arm_ablation_compensation_steps)
        if ablation_compensation_steps < 1:
            raise ValueError(
                "four_arm_ablation_compensation_steps must be >= 1; "
                f"got {four_arm_ablation_compensation_steps}"
            )
        portfolio_max_regions = int(target_region_portfolio_max_regions)
        if portfolio_max_regions < 1:
            raise ValueError(
                "target_region_portfolio_max_regions must be >= 1; "
                f"got {target_region_portfolio_max_regions}"
            )
        portfolio_excluded_keys: tuple[tuple[int, int, int], ...] = tuple(
            (int(key[0]), int(key[1]), int(key[2]))
            for key in _target_region_portfolio_excluded
        )
        portfolio_prior_attempts = [dict(record) for record in _target_region_portfolio_attempts]
        forced_region_key: tuple[int, int, int] | None = None
        if target_region_forced_key is not None:
            forced_parts = tuple(int(part) for part in target_region_forced_key)
            if len(forced_parts) != 3:
                raise ValueError(
                    "target_region_forced_key must contain exactly batch,class,region; "
                    f"got {target_region_forced_key!r}"
                )
            forced_region_key = (forced_parts[0], forced_parts[1], forced_parts[2])
        fakequant_bits = int(fakequant_survival_bits)
        if fakequant_bits < 1 or fakequant_bits > 16:
            raise ValueError(f"fakequant_survival_bits must be in [1, 16]; got {fakequant_survival_bits}")
        clip = None if grad_clip_max_norm is None else float(grad_clip_max_norm)
        if clip is not None and (not math.isfinite(clip) or clip <= 0.0):
            raise ValueError(f"grad_clip_max_norm must be None or finite and positive; got {grad_clip_max_norm}")

        import mlx.optimizers as mlx_optim

        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim != 1 or int(idx.shape[0]) <= 0:
            raise ValueError("pair_indices must be a non-empty rank-1 tensor")
        target0 = mx.array(target_rgb_0).astype(mx.float32)  # type: ignore[union-attr]
        target1 = mx.array(target_rgb_1).astype(mx.float32)  # type: ignore[union-attr]
        for name, target in (("target_rgb_0", target0), ("target_rgb_1", target1)):
            if target.ndim != 4 or int(target.shape[-1]) != 3:
                raise ValueError(
                    f"{name} must be NHWC with 3 channels; got shape={tuple(int(v) for v in target.shape)}"
                )
            if int(target.shape[0]) != int(idx.shape[0]):
                raise ValueError(
                    f"{name} batch {int(target.shape[0])} must match pair_indices length {int(idx.shape[0])}"
                )

        live_fn = getattr(scorer_teacher, "teacher_logits_for_frames_nhwc01", None)
        if not callable(live_fn):
            raise ValueError(
                "scorer_teacher must expose teacher_logits_for_frames_nhwc01 for the target-region birth actuator"
            )
        if target_segnet_argmax_1 is not None:
            labels = mx.array(target_segnet_argmax_1, dtype=mx.int32)  # type: ignore[union-attr]
        else:
            local_indices = mx.arange(int(target1.shape[0]), dtype=mx.int32)  # type: ignore[union-attr]
            argmax_fn = getattr(scorer_teacher, "teacher_argmax_for_indices", None)
            logits_fn = getattr(scorer_teacher, "teacher_logits_for_indices", None)
            if callable(argmax_fn):
                labels = argmax_fn(local_indices)
            elif callable(logits_fn):
                labels = mx.argmax(logits_fn(local_indices), axis=-1)  # type: ignore[union-attr]
            else:
                raise ValueError(
                    "target_segnet_argmax_1 missing and scorer_teacher exposes "
                    "neither teacher_argmax_for_indices nor teacher_logits_for_indices"
                )
        if labels.ndim == 4 and int(labels.shape[-1]) == 1:
            labels = labels[..., 0]
        if labels.ndim != 3 or tuple(int(v) for v in labels.shape) != tuple(int(v) for v in target1.shape[:3]):
            raise ValueError(
                "target SegNet argmax must be BHW matching target_rgb_1; got "
                f"argmax={tuple(int(v) for v in labels.shape)} "
                f"target={tuple(int(v) for v in target1.shape[:3])}"
            )
        labels = mx.stop_gradient(labels)  # type: ignore[union-attr]
        mx.eval(labels)  # type: ignore[union-attr]

        pose_fn = getattr(pose_teacher, "teacher_pose_for_yuv6_pair_nhwc", None) if pose_teacher is not None else None
        pose_available = callable(pose_fn)
        # Official PoseNet preprocess interpolates each frame to (384, 512)
        # BEFORE rgb_to_yuv6; this actuator applies no resize, so the
        # concat-YUV6-pair surface is contest-faithful only when frames are
        # already at contest size (then upstream's interpolate is a no-op,
        # mirroring build_mlx_posenet_pair_teacher's hard requirement).
        pose_input_contest_resolution = (
            int(target1.shape[1]),
            int(target1.shape[2]),
        ) == (384, 512)
        if require_pose_trust and pose_available and not pose_input_contest_resolution:
            return {
                "schema": "hi_nerv_target_region_birth.v1",
                "enabled": True,
                "accepted": False,
                "reason": "pose_trust_requires_contest_resolution_frames",
                "pose_input_height": int(target1.shape[1]),
                "pose_input_width": int(target1.shape[2]),
                "accepted_step_count": 0,
                "rejected_step_count": 0,
                "blockers": ["hinerv_target_region_birth_pose_surface_non_contest_resolution"],
                "runtime_sidecar_bytes": 0,
                "human_visual_fidelity_objective": False,
            }
        if require_pose_trust and not pose_available:
            # L3+ stages must not admit pose-blind updates. Fail closed before
            # any work: no steps, no state movement, loud blocker.
            return {
                "schema": "hi_nerv_target_region_birth.v1",
                "enabled": True,
                "accepted": False,
                "reason": "pose_trust_required_but_teacher_missing",
                "accepted_step_count": 0,
                "rejected_step_count": 0,
                "blockers": ["hinerv_target_region_birth_pose_trust_required_but_teacher_missing"],
                "runtime_sidecar_bytes": 0,
                "human_visual_fidelity_objective": False,
            }

        def _predict_pair01(model_obj: Any) -> tuple[Any, Any]:
            pair01 = model_obj(idx) / 255.0
            pred0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))  # type: ignore[union-attr]
            pred1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))  # type: ignore[union-attr]
            return pred0, pred1

        def _receiver_uint8_int(pred: Any) -> np.ndarray:
            receiver = mx.round(mx.clip(pred, 0.0, 1.0) * 255.0)  # type: ignore[union-attr]
            mx.eval(receiver)  # type: ignore[union-attr]
            return np.asarray(receiver, dtype=np.int16)

        def _candidate_logits_np(pred1: Any) -> np.ndarray:
            logits = live_fn(_receiver_uint8_roundtrip_ste_nhwc01(pred1))
            mx.eval(logits)  # type: ignore[union-attr]
            out = np.asarray(logits, dtype=np.float32)
            if out.ndim != 4:
                raise ValueError(f"live SegNet candidate logits must be BHWC; got {out.shape}")
            return out

        def _pose_output_np(pred0: Any, pred1: Any) -> np.ndarray | None:
            if not pose_available:
                return None
            # Mirrors the shared posenet_yuv6_pair_nhwc255 surface: per-frame
            # receiver-quantized RGB->YUV6 at byte scale, concatenated to 12
            # channels NHWC.  Pose and SegNet must see the same uint8 receiver
            # surface; otherwise hard-birth can optimize sub-byte pose fiction.
            receiver0 = _receiver_uint8_roundtrip_ste_nhwc01(pred0)
            receiver1 = _receiver_uint8_roundtrip_ste_nhwc01(pred1)
            yuv6_pair = mx.concatenate(  # type: ignore[union-attr]
                [rgb_to_yuv6_mlx(receiver0 * 255.0), rgb_to_yuv6_mlx(receiver1 * 255.0)],
                axis=-1,
            )
            pose = pose_fn(yuv6_pair)
            mx.eval(pose)  # type: ignore[union-attr]
            return np.asarray(pose, dtype=np.float32)

        base_pred0, base_pred1 = _predict_pair01(self)
        base_pred0 = mx.stop_gradient(mx.array(base_pred0))  # type: ignore[union-attr]
        base_pred1 = mx.stop_gradient(mx.array(base_pred1))  # type: ignore[union-attr]
        mx.eval(base_pred0, base_pred1)  # type: ignore[union-attr]
        initial_logits_np = _candidate_logits_np(base_pred1)
        class_count = int(initial_logits_np.shape[-1])
        labels_np = np.asarray(labels, dtype=np.int64)
        if int(labels_np.max()) >= class_count:
            raise ValueError(
                "target SegNet argmax contains a class outside live logits: "
                f"max_target_class={int(labels_np.max())} class_count={class_count}"
            )
        initial_argmax_np = np.argmax(initial_logits_np, axis=-1)
        target_logits_np = np.take_along_axis(
            initial_logits_np,
            labels_np[..., None],
            axis=-1,
        )[..., 0]
        masked_logits_np = np.array(initial_logits_np, copy=True)
        np.put_along_axis(masked_logits_np, labels_np[..., None], -np.inf, axis=-1)
        target_margin_np = np.max(masked_logits_np, axis=-1) - target_logits_np
        hard_region_miner_inputs = _write_hard_region_miner_input_bundle(
            output_dir=hard_region_miner_output_dir,
            target_labels_bhw=labels_np,
            candidate_argmax_bhw=initial_argmax_np,
            target_margin_bhw=target_margin_np,
            pair_indices=np.asarray(pair_indices, dtype=np.int64),
        )
        worst, region_mask_np = select_worst_target_region_with_mask(
            labels_np,
            initial_argmax_np,
            min_region_pixels=int(min_region_pixels),
            excluded_region_keys=portfolio_excluded_keys,
            forced_region_key=forced_region_key,
        )
        if worst.region_unsolved_pixel_count == 0:
            return {
                "schema": "hi_nerv_target_region_birth.v1",
                "enabled": False,
                "reason": "no_unsolved_target_region",
                "worst_region": worst.as_dict(),
                "hard_region_miner_inputs": hard_region_miner_inputs,
                "accepted_step_count": 0,
                "rejected_step_count": 0,
                "target_region_portfolio": {
                    "schema": "hi_nerv_target_region_birth_portfolio.v1",
                    "max_regions": int(portfolio_max_regions),
                    "forced_region_key": (
                        None if forced_region_key is None else list(forced_region_key)
                    ),
                    "attempt_count": len(portfolio_prior_attempts),
                    "attempts": list(portfolio_prior_attempts),
                    "exhausted": True,
                },
                "blockers": [],
                "runtime_sidecar_bytes": 0,
                "human_visual_fidelity_objective": False,
            }
        birth_class = int(worst.class_index)
        total_scored_pixels = float(worst.total_scored_pixels)
        before_stats = region_margin_stats(initial_logits_np, region_mask_np, birth_class)
        initial_uint8 = _receiver_uint8_int(base_pred1)
        initial_pose = _pose_output_np(base_pred0, base_pred1)
        region_bool_np = region_mask_np > 0.0
        unsolved_tail_bool_np = region_bool_np & (initial_argmax_np != birth_class)
        already_won_bool_np = region_bool_np & (initial_argmax_np == birth_class)
        unsolved_tail_pixel_count = int(np.count_nonzero(unsolved_tail_bool_np))
        already_won_pixel_count = int(np.count_nonzero(already_won_bool_np))
        if unsolved_tail_pixel_count != int(worst.region_unsolved_pixel_count):
            raise RuntimeError(
                "target-region unsolved-tail reconstruction drifted from the priced row; "
                f"tail={unsolved_tail_pixel_count} row={worst.region_unsolved_pixel_count}"
            )
        unsolved_tail_mask_np = unsolved_tail_bool_np.astype(np.float32)
        already_won_mask_np = already_won_bool_np.astype(np.float32)
        already_won_mask = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(already_won_mask_np, dtype=mx.float32)
        )
        already_won_mask_4 = mx.expand_dims(already_won_mask, axis=-1)  # type: ignore[union-attr]
        unsolved_tail_pixels = float(unsolved_tail_pixel_count)
        already_won_pixels = float(already_won_pixel_count)

        connectivity_4 = np.array(
            [[False, True, False], [True, True, True], [False, True, False]],
            dtype=bool,
        )

        def _largest_component_mask(mask_bhw: np.ndarray) -> np.ndarray:
            mask_bool = np.asarray(mask_bhw, dtype=bool)
            out = np.zeros(mask_bool.shape, dtype=bool)
            for batch_index in range(mask_bool.shape[0]):
                labeled, component_count = scipy_ndimage.label(
                    mask_bool[batch_index],
                    structure=connectivity_4,
                )
                if int(component_count) <= 0:
                    continue
                counts = np.bincount(labeled.reshape(-1))
                if counts.shape[0] <= 1:
                    continue
                component_label = int(np.argmax(counts[1:]) + 1)
                out[batch_index] = labeled == component_label
            return out

        def _dilate_per_frame(mask_bhw: np.ndarray) -> np.ndarray:
            mask_bool = np.asarray(mask_bhw, dtype=bool)
            out = np.zeros(mask_bool.shape, dtype=bool)
            for batch_index in range(mask_bool.shape[0]):
                out[batch_index] = scipy_ndimage.binary_dilation(
                    mask_bool[batch_index],
                    structure=connectivity_4,
                    iterations=int(frontier_dilation),
                )
            return out

        def _select_actuator_tail_mask(argmax_np: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            """Return the current differentiable target geometry inside the priced parent region."""

            current_unsolved = region_bool_np & (np.asarray(argmax_np) != birth_class)
            current_won = region_bool_np & (np.asarray(argmax_np) == birth_class)
            fallback = None
            selected = current_unsolved
            if geometry_mode == "frontier_band":
                if np.any(current_won):
                    frontier = _dilate_per_frame(current_won) & current_unsolved
                    if np.any(frontier):
                        selected = frontier
                    else:
                        fallback = "frontier_band_empty_largest_unsolved_component"
                        selected = _largest_component_mask(current_unsolved)
                else:
                    fallback = "no_already_won_support_largest_unsolved_component"
                    selected = _largest_component_mask(current_unsolved)
            elif geometry_mode == "largest_unsolved_component":
                selected = _largest_component_mask(current_unsolved)
            elif geometry_mode == "full_tail":
                selected = current_unsolved
            if not np.any(selected) and np.any(current_unsolved):
                fallback = fallback or "selected_empty_full_tail"
                selected = current_unsolved
            selected = selected.astype(np.float32)
            return selected, {
                "mode": geometry_mode,
                "fallback": fallback,
                "frontier_dilation": int(frontier_dilation),
                "active_tail_pixel_count": int(np.count_nonzero(selected)),
                "current_unsolved_tail_pixel_count": int(np.count_nonzero(current_unsolved)),
                "current_already_won_region_pixel_count": int(np.count_nonzero(current_won)),
            }

        active_tail_mask_np, initial_active_tail_geometry = _select_actuator_tail_mask(initial_argmax_np)
        if int(np.count_nonzero(active_tail_mask_np)) <= 0:
            raise RuntimeError("target-region birth selected an empty active unsolved-tail geometry")
        active_tail_mask = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(active_tail_mask_np, dtype=mx.float32)
        )
        active_tail_pixels = float(np.count_nonzero(active_tail_mask_np))
        active_tail_geometry = dict(initial_active_tail_geometry)
        active_tail_geometry_history: list[dict[str, Any]] = [
            {"step_index": -1, "reason": "initial", **dict(initial_active_tail_geometry)}
        ]
        active_tail_refresh_count = 0

        def _refresh_active_tail_mask(argmax_np: np.ndarray, *, step_index: int, reason: str) -> bool:
            nonlocal active_tail_mask_np, active_tail_mask, active_tail_pixels
            nonlocal active_tail_geometry, active_tail_refresh_count
            refreshed_np, refreshed_geometry = _select_actuator_tail_mask(argmax_np)
            refreshed_pixels = int(np.count_nonzero(refreshed_np))
            active_tail_geometry = dict(refreshed_geometry)
            active_tail_refresh_count += 1
            active_tail_geometry_history.append(
                {"step_index": int(step_index), "reason": str(reason), **dict(refreshed_geometry)}
            )
            if refreshed_pixels <= 0:
                active_tail_pixels = 0.0
                return False
            active_tail_mask_np = refreshed_np
            active_tail_mask = mx.stop_gradient(  # type: ignore[union-attr]
                mx.array(active_tail_mask_np, dtype=mx.float32)
            )
            active_tail_pixels = float(refreshed_pixels)
            return True
        before_unsolved_tail_stats = region_margin_stats(
            initial_logits_np,
            unsolved_tail_mask_np,
            birth_class,
        )
        before_already_won_stats = (
            region_margin_stats(initial_logits_np, already_won_mask_np, birth_class)
            if already_won_pixel_count > 0
            else None
        )
        target_pose_np: np.ndarray | None = None
        target_pose_reference: Any | None = None
        if pose_available:
            pose_target_fn = getattr(pose_teacher, "teacher_pose_for_indices", None)
            if callable(pose_target_fn):
                target_pose = pose_target_fn(
                    mx.arange(int(target1.shape[0]), dtype=mx.int32)  # type: ignore[union-attr]
                )
            else:
                # Target pose through the SAME live surface, from the actual
                # target frames — keeps candidate/target pose comparable.
                target_pose = pose_fn(
                    mx.concatenate(  # type: ignore[union-attr]
                        [
                            rgb_to_yuv6_mlx(_receiver_uint8_roundtrip_ste_nhwc01(target0) * 255.0),
                            rgb_to_yuv6_mlx(_receiver_uint8_roundtrip_ste_nhwc01(target1) * 255.0),
                        ],
                        axis=-1,
                    )
                )
            mx.eval(target_pose)  # type: ignore[union-attr]
            target_pose_np = np.asarray(target_pose, dtype=np.float32)
            target_pose_reference = mx.stop_gradient(mx.array(target_pose, dtype=mx.float32))  # type: ignore[union-attr]

        def _d_seg_batch(argmax_np: np.ndarray) -> float:
            return float(np.mean(argmax_np != labels_np))

        def _d_pose_batch(pose: np.ndarray | None) -> float | None:
            if pose is None or target_pose_np is None:
                return None
            return float(np.mean((pose - target_pose_np) ** 2))

        def _nonrate_score(d_seg: float, d_pose: float | None) -> float | None:
            if d_pose is None:
                return None
            return 100.0 * d_seg + math.sqrt(10.0 * d_pose)

        initial_d_seg = _d_seg_batch(initial_argmax_np)
        initial_d_pose = _d_pose_batch(initial_pose)
        initial_nonrate = _nonrate_score(initial_d_seg, initial_d_pose)

        def _target_birth_admission_decision(
            *,
            new_d_seg: float,
            new_d_pose: float | None,
            pose_output_l2_delta: float | None,
        ) -> dict[str, Any]:
            if initial_d_pose is not None and new_d_pose is not None:
                return pose_trusted_birth_admission_decision(
                    old_d_seg=initial_d_seg,
                    new_d_seg=float(new_d_seg),
                    old_d_pose=initial_d_pose,
                    new_d_pose=new_d_pose,
                    pose_output_l2_delta=pose_output_l2_delta,
                    raw_pose_cap_l2=float(max_pose_output_delta_l2),
                    exact_score_epsilon=float(exact_score_epsilon_batch_local),
                    catastrophic_relative_cap=float(catastrophic_pose_relative_cap),
                    catastrophic_pose_score_regression_cap=float(catastrophic_pose_score_regression_cap),
                    catastrophic_pose_output_l2_hard_cap=catastrophic_pose_output_l2_hard_cap,
                )
            if require_pose_trust:
                return pose_trusted_birth_admission_decision(
                    old_d_seg=initial_d_seg,
                    new_d_seg=float(new_d_seg),
                    old_d_pose=initial_d_pose,
                    new_d_pose=new_d_pose,
                    pose_output_l2_delta=pose_output_l2_delta,
                    raw_pose_cap_l2=float(max_pose_output_delta_l2),
                    exact_score_epsilon=float(exact_score_epsilon_batch_local),
                    catastrophic_relative_cap=float(catastrophic_pose_relative_cap),
                    catastrophic_pose_score_regression_cap=float(catastrophic_pose_score_regression_cap),
                    catastrophic_pose_output_l2_hard_cap=catastrophic_pose_output_l2_hard_cap,
                )
            seg_score_delta = 100.0 * (float(new_d_seg) - float(initial_d_seg))
            return {
                "schema": "hi_nerv_pose_trusted_birth_admission_decision.v1",
                "old_d_seg": float(initial_d_seg),
                "new_d_seg": float(new_d_seg),
                "old_d_pose": None,
                "new_d_pose": None if new_d_pose is None else float(new_d_pose),
                "seg_score_delta": float(seg_score_delta),
                "pose_score_delta": None,
                "exact_delta_score_nonrate": float(seg_score_delta),
                "delta_score_nonrate": float(seg_score_delta),
                "pose_output_l2_delta": None if pose_output_l2_delta is None else float(pose_output_l2_delta),
                "raw_pose_cap_l2": float(max_pose_output_delta_l2),
                "raw_cap_decision": "not_applicable",
                "raw_pose_cap_result": "not_applicable",
                "exact_score_epsilon": float(exact_score_epsilon_batch_local),
                "exact_score_decision": "not_applicable",
                "catastrophic_relative_cap": float(catastrophic_pose_relative_cap),
                "catastrophic_pose_score_regression_cap": float(catastrophic_pose_score_regression_cap),
                "catastrophic_pose_output_l2_hard_cap": (
                    None
                    if catastrophic_pose_output_l2_hard_cap is None
                    else float(catastrophic_pose_output_l2_hard_cap)
                ),
                "catastrophic_guard_decision": "satisfied",
                "catastrophic_guard_reasons": [],
                "accepted": False,
                "would_accept_exact_score_if_raw_cap_disabled": False,
                "would_accept_without_catastrophic_guard": False,
                "rejected_by_raw_pose_cap": False,
                "would_reject_under_raw_pose_cap": False,
                "rejected_by_exact_delta_score": False,
                "rejected_by_catastrophic_pose_guard": False,
                "rejection_source": None,
                "raw_cap_is_counterfactual_only": True,
                "pose_trust_evidence_missing_but_not_required": True,
            }

        def _impostor_logit(logits: Any) -> Any:
            if class_count == 1:
                return mx.zeros_like(logits[..., 0])  # type: ignore[union-attr]
            if birth_class == 0:
                return mx.max(logits[..., 1:], axis=-1)  # type: ignore[union-attr]
            if birth_class == class_count - 1:
                return mx.max(logits[..., :birth_class], axis=-1)  # type: ignore[union-attr]
            return mx.max(  # type: ignore[union-attr]
                mx.concatenate(
                    [logits[..., :birth_class], logits[..., birth_class + 1 :]],
                    axis=-1,
                ),
                axis=-1,
            )

        eps = 1.0e-6
        initial_impostor_np = np.array(initial_logits_np, copy=True)
        initial_impostor_np[..., birth_class] = -np.inf
        initial_margin_reference_np = initial_impostor_np.max(axis=-1) - initial_logits_np[..., birth_class]
        initial_margin_reference = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(initial_margin_reference_np, dtype=mx.float32)
        )
        initial_argmax_one_hot_np = np.eye(class_count, dtype=np.float32)[initial_argmax_np]
        initial_argmax_one_hot = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(initial_argmax_one_hot_np, dtype=mx.float32)
        )
        target_label_one_hot_np = np.eye(class_count, dtype=np.float32)[labels_np]
        target_label_one_hot = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(target_label_one_hot_np, dtype=mx.float32)
        )
        initial_target_margin_reference = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(target_margin_np, dtype=mx.float32)
        )
        initial_argmax_logit_np = np.sum(initial_logits_np * initial_argmax_one_hot_np, axis=-1)
        initial_argmax_impostor_np = np.array(initial_logits_np, copy=True)
        np.put_along_axis(
            initial_argmax_impostor_np,
            initial_argmax_np[..., None],
            -np.inf,
            axis=-1,
        )
        initial_argmax_margin_reference = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(initial_argmax_impostor_np.max(axis=-1) - initial_argmax_logit_np, dtype=mx.float32)
        )
        outside_region_mask_np = (~region_bool_np).astype(np.float32)
        outside_region_mask = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(outside_region_mask_np, dtype=mx.float32)
        )
        outside_region_pixels = float(np.count_nonzero(outside_region_mask_np))
        global_already_won_bool_np = (initial_argmax_np == labels_np) & (~region_bool_np)
        global_already_won_mask_np = global_already_won_bool_np.astype(np.float32)
        global_already_won_mask = mx.stop_gradient(  # type: ignore[union-attr]
            mx.array(global_already_won_mask_np, dtype=mx.float32)
        )
        global_already_won_pixels = float(np.count_nonzero(global_already_won_bool_np))
        global_already_won_preserve_active = bool(
            global_already_won_pixels > 0.0
            and (bool(require_pose_trust) or bool(require_fakequant_survival))
        )
        initial_pose_reference = (
            mx.stop_gradient(mx.array(initial_pose, dtype=mx.float32))  # type: ignore[union-attr]
            if initial_pose is not None
            else None
        )

        def _global_target_transition_counts(argmax_np: np.ndarray) -> dict[str, int]:
            after_correct = np.asarray(argmax_np) == labels_np
            before_correct = initial_argmax_np == labels_np
            outside = ~region_bool_np
            return {
                "outside_correct_to_wrong_count": int(np.count_nonzero(outside & before_correct & ~after_correct)),
                "outside_wrong_to_correct_count": int(np.count_nonzero(outside & ~before_correct & after_correct)),
                "global_correct_to_wrong_count": int(np.count_nonzero(before_correct & ~after_correct)),
                "global_wrong_to_correct_count": int(np.count_nonzero(~before_correct & after_correct)),
                "global_net_correct_delta": int(np.count_nonzero(~before_correct & after_correct))
                - int(np.count_nonzero(before_correct & ~after_correct)),
            }

        def _loss_fn(model_obj: Any) -> Any:
            pred0, pred1 = _predict_pair01(model_obj)
            receiver = _receiver_uint8_roundtrip_ste_nhwc01(pred1)
            logits = live_fn(receiver)
            class_logit = logits[..., birth_class]
            impostor = _impostor_logit(logits)
            margin_raw = impostor - class_logit
            crossing = mx.logaddexp(  # type: ignore[union-attr]
                (margin_raw + float(margin_floor)) / float(margin_tau),
                0.0,
            )
            loss_margin = mx.sum(crossing * crossing * active_tail_mask) / active_tail_pixels  # type: ignore[union-attr]
            shifted = logits - mx.max(logits, axis=-1, keepdims=True)  # type: ignore[union-attr]
            exp_logits = mx.exp(shifted)  # type: ignore[union-attr]
            probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)  # type: ignore[union-attr]
            prob_deficit = mx.maximum(  # type: ignore[union-attr]
                float(prob_floor) - probs[..., birth_class],
                0.0,
            )
            loss_prob = mx.sum(prob_deficit * prob_deficit * active_tail_mask) / active_tail_pixels  # type: ignore[union-attr]
            margin_detached = mx.stop_gradient(margin_raw)  # type: ignore[union-attr]
            masked_margin = mx.where(  # type: ignore[union-attr]
                active_tail_mask > 0.0,
                margin_detached,
                mx.array(1.0e30, dtype=margin_detached.dtype),
            )
            margin_min = mx.min(masked_margin)  # type: ignore[union-attr]
            seed_weight = active_tail_mask * mx.exp(  # type: ignore[union-attr]
                -(margin_detached - margin_min) / float(seed_temperature)
            )
            seed_weight = seed_weight / mx.maximum(mx.sum(seed_weight), eps)  # type: ignore[union-attr]
            loss_seed = mx.sum(seed_weight * crossing * crossing)  # type: ignore[union-attr]
            loss_preserve = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            loss_already_won_hard_preserve = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            loss_already_won_rgb_preserve = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            loss_global_already_won_preserve = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            if already_won_pixel_count > 0:
                # Preserve already-won target support as a trust-region
                # constraint, not only after it has crossed the argmax boundary.
                # The v9 smoke exposed a bad local minimum where a step won a
                # few unsolved-tail pixels by eroding and then destroying most
                # already-correct pixels. Penalize margin erosion against the
                # initial receiver-surface margin so the optimizer feels that
                # spill before it becomes a hard argmax loss.
                preserve_violation = mx.maximum(  # type: ignore[union-attr]
                    margin_raw - initial_margin_reference,
                    0.0,
                )
                preserve_weight = max(1.0, already_won_pixels / unsolved_tail_pixels)
                loss_preserve = (  # type: ignore[assignment]
                    float(lambda_support_preserve)
                    * float(preserve_weight)
                    * mx.sum(preserve_violation * preserve_violation * already_won_mask)
                    / already_won_pixels
                )
                hard_preserve = mx.logaddexp(  # type: ignore[union-attr]
                    (margin_raw + float(already_won_margin_floor)) / float(margin_tau),
                    0.0,
                )
                loss_already_won_hard_preserve = (  # type: ignore[assignment]
                    float(lambda_already_won_hard_preserve)
                    * float(preserve_weight)
                    * mx.sum(hard_preserve * hard_preserve * already_won_mask)
                    / already_won_pixels
                )
                rgb_preserve_delta = (pred1 - base_pred1) * already_won_mask_4
                loss_already_won_rgb_preserve = (  # type: ignore[assignment]
                    float(lambda_already_won_rgb_preserve)
                    * mx.sum(rgb_preserve_delta * rgb_preserve_delta)
                    / max(already_won_pixels * 3.0, 1.0)
                )
            if global_already_won_preserve_active and float(lambda_support_preserve) > 0.0:
                target_label_logit = mx.sum(logits * target_label_one_hot, axis=-1)  # type: ignore[union-attr]
                target_competitor_logits = mx.where(  # type: ignore[union-attr]
                    target_label_one_hot > 0.0,
                    mx.array(-1.0e30, dtype=logits.dtype),
                    logits,
                )
                target_impostor = mx.max(target_competitor_logits, axis=-1)  # type: ignore[union-attr]
                target_margin = target_impostor - target_label_logit
                global_margin_erosion = mx.maximum(  # type: ignore[union-attr]
                    target_margin - initial_target_margin_reference,
                    0.0,
                )
                loss_global_already_won_preserve = (  # type: ignore[assignment]
                    float(lambda_support_preserve)
                    * mx.sum(global_margin_erosion * global_margin_erosion * global_already_won_mask)
                    / global_already_won_pixels
                )
            loss_outside_preserve = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            if outside_region_pixels > 0.0 and float(lambda_outside_argmax_preserve) > 0.0:
                initial_winner_logit = mx.sum(logits * initial_argmax_one_hot, axis=-1)  # type: ignore[union-attr]
                competitor_logits = mx.where(  # type: ignore[union-attr]
                    initial_argmax_one_hot > 0.0,
                    mx.array(-1.0e30, dtype=logits.dtype),
                    logits,
                )
                initial_winner_impostor = mx.max(competitor_logits, axis=-1)  # type: ignore[union-attr]
                outside_margin_erosion = mx.maximum(  # type: ignore[union-attr]
                    (initial_winner_impostor - initial_winner_logit) - initial_argmax_margin_reference,
                    0.0,
                )
                loss_outside_preserve = (  # type: ignore[assignment]
                    float(lambda_outside_argmax_preserve)
                    * mx.sum(outside_margin_erosion * outside_margin_erosion * outside_region_mask)
                    / outside_region_pixels
                )
            loss_pose_preserve = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            loss_pose_target = mx.sum(margin_raw * 0.0)  # type: ignore[union-attr]
            if initial_pose_reference is not None and (
                float(lambda_pose_trust_preserve) > 0.0 or float(lambda_pose_target) > 0.0
            ):
                receiver0 = _receiver_uint8_roundtrip_ste_nhwc01(pred0)
                receiver1 = _receiver_uint8_roundtrip_ste_nhwc01(pred1)
                pose_live = pose_fn(  # type: ignore[misc]
                    mx.concatenate(  # type: ignore[union-attr]
                        [
                            rgb_to_yuv6_mlx(receiver0 * 255.0),
                            rgb_to_yuv6_mlx(receiver1 * 255.0),
                        ],
                        axis=-1,
                    )
                )
                if float(lambda_pose_trust_preserve) > 0.0:
                    pose_drift_sq = mx.sum(  # type: ignore[union-attr]
                        (pose_live - initial_pose_reference) * (pose_live - initial_pose_reference),
                        axis=-1,
                    )
                    pose_cap_sq = max(float(max_pose_output_delta_l2) ** 2, 1.0e-12)
                    pose_trust_violation = mx.maximum(  # type: ignore[union-attr]
                        pose_drift_sq / pose_cap_sq - 1.0,
                        0.0,
                    )
                    loss_pose_preserve = (  # type: ignore[assignment]
                        float(lambda_pose_trust_preserve)
                        * mx.mean(pose_trust_violation * pose_trust_violation)
                    )
                if target_pose_reference is not None and float(lambda_pose_target) > 0.0:
                    pose_target_mse = mx.mean(  # type: ignore[union-attr]
                        (pose_live - target_pose_reference) * (pose_live - target_pose_reference)
                    )
                    pose_target_scale = max(float(initial_d_pose or 0.0), 1.0e-6)
                    loss_pose_target = (  # type: ignore[assignment]
                        float(lambda_pose_target) * pose_target_mse / pose_target_scale
                    )
            pred_class = mx.argmax(logits, axis=-1)  # type: ignore[union-attr]
            unsolved = mx.stop_gradient(  # type: ignore[union-attr]
                mx.sum(active_tail_mask * (pred_class != birth_class).astype(mx.float32))
            )
            debt_weight = 1.0 + 100.0 * unsolved / total_scored_pixels
            return debt_weight * (
                loss_margin
                + float(lambda_prob_floor) * loss_prob
                + float(lambda_seed) * loss_seed
                + loss_preserve
                + loss_already_won_hard_preserve
                + loss_already_won_rgb_preserve
                + loss_global_already_won_preserve
                + loss_outside_preserve
                + loss_pose_preserve
                + loss_pose_target
            )

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

        def _flat_param_name(raw_name: Any) -> str:
            if isinstance(raw_name, (tuple, list)):
                return ".".join(str(part) for part in raw_name)
            return str(raw_name)

        def _group_for_name(name: str) -> str:
            if name == "latents_fine" or name.startswith("latents_fine."):
                return "latents_fine"
            for group in ("feature_grids", "fine_injector", "head_rgb_1"):
                if name.startswith(f"{group}."):
                    return group
            # head_rgb_0 is the frame0 *compensation* scope, NOT a birth scope.
            # SegNet reads frame1 only, so frame0 head moves carry zero seg
            # leverage; they exist solely to absorb a pose-cap/joint loss that a
            # frame1 birth step would otherwise have to backtrack away from. It
            # is reported as its own group so the out-of-scope frozen-bit proof
            # (which must stay bit-identical) never folds head_rgb_0 into it.
            if name == "head_rgb_0" or name.startswith("head_rgb_0."):
                return "compensation_head_rgb_0"
            return "out_of_scope"

        def _parameter_group_sha256(snapshot: list[tuple[Any, Any]]) -> dict[str, str]:
            """Hash every parameter group so 'scoped' is a receipt, not a claim."""

            import hashlib

            digests: dict[str, Any] = {}
            for raw_name, leaf in sorted(
                snapshot,
                key=lambda item: _flat_param_name(item[0]),
            ):
                if leaf is None:
                    continue
                group = _group_for_name(_flat_param_name(raw_name))
                digest = digests.setdefault(group, hashlib.sha256())
                digest.update(_flat_param_name(raw_name).encode("utf-8"))
                digest.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
            return {group: digest.hexdigest() for group, digest in digests.items()}

        def _apply_scoped_step(
            base_snapshot: list[tuple[Any, Any]],
            grads_tree: Any,
            step_lr: float,
            *,
            step_direction: float = -1.0,
        ) -> tuple[list[str], dict[str, float], dict[str, float]]:
            grad_by_name = dict(tree_flatten(grads_tree))  # type: ignore[operator]
            updated: list[tuple[Any, Any]] = []
            applied_names: list[str] = []
            grad_sq_by_group: dict[str, float] = {}
            update_sq_by_group: dict[str, float] = {}
            for raw_name, leaf in base_snapshot:
                grad = grad_by_name.get(raw_name)
                flat = _flat_param_name(raw_name)
                if leaf is None or grad is None or not allowed_birth_update_name(flat):
                    updated.append((raw_name, leaf))
                    continue
                grad_sq = mx.sum(grad * grad)  # type: ignore[union-attr]
                mx.eval(grad_sq)  # type: ignore[union-attr]
                grad_sq_value = float(grad_sq.item())
                if grad_sq_value <= 0.0:
                    updated.append((raw_name, leaf))
                    continue
                param = mx.array(leaf)  # type: ignore[union-attr]
                step = float(step_lr) * grad
                update = param + float(step_direction) * step
                updated.append((raw_name, update))
                applied_names.append(flat)
                group = _group_for_name(flat)
                grad_sq_by_group[group] = grad_sq_by_group.get(group, 0.0) + grad_sq_value
                step_sq = mx.sum(step * step)  # type: ignore[union-attr]
                mx.eval(step_sq)  # type: ignore[union-attr]
                update_sq_by_group[group] = update_sq_by_group.get(group, 0.0) + float(step_sq.item())
            self.update(tree_unflatten(updated))
            mx.eval(self.parameters())  # type: ignore[union-attr]
            return (
                applied_names,
                {name: math.sqrt(value) for name, value in grad_sq_by_group.items()},
                {name: math.sqrt(value) for name, value in update_sq_by_group.items()},
            )

        def _region_candidate_state() -> dict[str, Any]:
            pred0, pred1 = _predict_pair01(self)
            pred0 = mx.stop_gradient(mx.array(pred0))  # type: ignore[union-attr]
            pred1 = mx.stop_gradient(mx.array(pred1))  # type: ignore[union-attr]
            mx.eval(pred0, pred1)  # type: ignore[union-attr]
            logits_np = _candidate_logits_np(pred1)
            stats = region_margin_stats(logits_np, region_mask_np, birth_class)
            unsolved_tail_stats = region_margin_stats(
                logits_np,
                unsolved_tail_mask_np,
                birth_class,
            )
            already_won_stats = (
                region_margin_stats(logits_np, already_won_mask_np, birth_class)
                if already_won_pixel_count > 0
                else None
            )
            uint8 = _receiver_uint8_int(pred1)
            pose = _pose_output_np(pred0, pred1)
            float_delta = mx.max(mx.abs(pred1 - base_pred1))  # type: ignore[union-attr]
            mx.eval(float_delta)  # type: ignore[union-attr]
            argmax_np = np.argmax(logits_np, axis=-1)
            region_transitions = region_argmax_transition_counts(
                initial_argmax_np,
                argmax_np,
                region_bool_np,
                birth_class,
            )
            unsolved_tail_transitions = region_argmax_transition_counts(
                initial_argmax_np,
                argmax_np,
                unsolved_tail_bool_np,
                birth_class,
            )
            global_target_transitions = _global_target_transition_counts(argmax_np)
            already_won_lost_count = int(np.count_nonzero(already_won_bool_np & (argmax_np != birth_class)))
            return {
                "stats": stats,
                "unsolved_tail_stats": unsolved_tail_stats,
                "already_won_stats": already_won_stats,
                "region_argmax_transitions": region_transitions,
                "unsolved_tail_argmax_transitions": unsolved_tail_transitions,
                "global_target_transitions": global_target_transitions,
                "already_won_lost_count": already_won_lost_count,
                "uint8": uint8,
                "pose": pose,
                "logits_np": logits_np,
                "argmax_np": argmax_np,
                "d_seg_batch": _d_seg_batch(argmax_np),
                "d_pose_batch": _d_pose_batch(pose),
                "float_rgb_delta_linf": float(float_delta.item()),
            }

        def _fakequant_config_snapshot() -> dict[str, Any]:
            return {
                "configured_enabled": bool(
                    getattr(self, "decoder_fake_quant_forward_configured_enabled", False)
                ),
                "stage_controlled": bool(
                    getattr(self, "decoder_fake_quant_forward_stage_controlled", False)
                ),
                "stage_qat_active": bool(
                    getattr(self, "decoder_fake_quant_forward_stage_qat_active", True)
                ),
                "bits": getattr(self, "decoder_fake_quant_bits", None),
                "bits_by_name": dict(getattr(self, "decoder_fake_quant_bits_by_name", {}) or {}),
                "forward_enabled": bool(getattr(self, "decoder_fake_quant_forward_enabled", False)),
            }

        def _restore_fakequant_config(snapshot: Mapping[str, Any]) -> None:
            self.configure_decoder_fake_quant_forward(
                enabled=bool(snapshot["configured_enabled"]),
                quant_bits=snapshot["bits"],
                per_tensor_bits=dict(snapshot["bits_by_name"]),
                stage_controlled=bool(snapshot["stage_controlled"]),
            )
            self.decoder_fake_quant_forward_stage_qat_active = bool(snapshot["stage_qat_active"])
            self.decoder_fake_quant_forward_enabled = bool(snapshot["forward_enabled"])

        def _fakequant_survival_for_candidate() -> dict[str, Any]:
            snapshot = _fakequant_config_snapshot()
            try:
                self.configure_decoder_fake_quant_forward(
                    enabled=True,
                    quant_bits=fakequant_bits,
                    per_tensor_bits=None,
                    stage_controlled=False,
                )
                fq_candidate = _region_candidate_state()
            finally:
                _restore_fakequant_config(snapshot)
            fq_pose_delta = _pose_delta_l2(fq_candidate["pose"])
            fq_decision = _target_birth_admission_decision(
                new_d_seg=float(fq_candidate["d_seg_batch"]),
                new_d_pose=fq_candidate["d_pose_batch"],
                pose_output_l2_delta=fq_pose_delta,
            )
            fq_transitions = dict(fq_candidate["region_argmax_transitions"])
            fq_nonrate = _nonrate_score(
                fq_candidate["d_seg_batch"],
                fq_candidate["d_pose_batch"],
            )
            fq_delta_nonrate = (
                None
                if fq_nonrate is None or initial_nonrate is None
                else float(fq_nonrate - initial_nonrate)
            )
            fq_support_spill = bool(
                int(fq_transitions["target_to_wrong_count"]) > 0
                or int(fq_transitions["net_target_support_delta"]) <= 0
            )
            fq_exact_ok = fq_decision["exact_score_decision"] == "accepted"
            fq_catastrophic_ok = fq_decision["catastrophic_guard_decision"] == "satisfied"
            survived = bool(
                not fq_support_spill
                and int(fq_transitions["wrong_to_target_count"]) > 0
                and fq_exact_ok
                and fq_catastrophic_ok
            )
            return {
                "schema": "hi_nerv_target_region_birth_fakequant_admission.v1",
                "required": bool(require_fakequant_survival),
                "fakequant_bits": int(fakequant_bits),
                "survived": survived,
                "wrong_to_target_count": int(fq_transitions["wrong_to_target_count"]),
                "target_to_wrong_count": int(fq_transitions["target_to_wrong_count"]),
                "wrong_to_wrong_count": int(fq_transitions["wrong_to_wrong_count"]),
                "net_target_support_delta": int(fq_transitions["net_target_support_delta"]),
                "support_spill": bool(fq_support_spill),
                "d_seg_batch": float(fq_candidate["d_seg_batch"]),
                "d_pose_batch": (
                    None
                    if fq_candidate["d_pose_batch"] is None
                    else float(fq_candidate["d_pose_batch"])
                ),
                "delta_score_nonrate": fq_delta_nonrate,
                "exact_score_decision": fq_decision["exact_score_decision"],
                "catastrophic_guard_decision": fq_decision["catastrophic_guard_decision"],
                "catastrophic_guard_reasons": list(fq_decision["catastrophic_guard_reasons"]),
                "pose_output_delta_l2": (
                    None if fq_pose_delta is None else float(fq_pose_delta)
                ),
            }

        def _uint8_changed_in_region(before_u8: np.ndarray, after_u8: np.ndarray) -> int:
            changed = np.any(before_u8 != after_u8, axis=-1)
            return int(np.count_nonzero(changed & region_bool_np))

        def _pose_delta_l2(pose: np.ndarray | None) -> float | None:
            if pose is None or initial_pose is None:
                return None
            per_item = np.sqrt(np.sum((pose - initial_pose) ** 2, axis=-1))
            return float(per_item.max())

        # ---- Frame0 pose-compensation composite operator (pose-only) --------
        # When a receiver-visible frame1 birth step holds region progress but
        # loses the pose cap or the exact joint gate, attempt ONE frame0-only
        # compensation BEFORE backtracking the learning rate. The compensation
        # minimizes the pose distance to target by updating ``head_rgb_0.*``
        # ONLY. SegNet reads frame1 only, so the seg term is structurally
        # untouched; an explicit assertion proves the frame1 receiver uint8 did
        # not move during compensation. The composite (frame1 birth + frame0
        # compensation) is admitted only when the exact nonrate score strictly
        # improves vs best AND the pose cap is satisfied.
        compensation_step_count = 8
        compensation_lr = lr

        def _pose_compensation_loss_fn(model_obj: Any) -> Any:
            pred0, pred1 = _predict_pair01(model_obj)
            # Freeze frame1: only the frame0 head may absorb pose error. This
            # keeps the gradient (and therefore the receiver-visible motion)
            # confined to head_rgb_0; frame1's birth state must not move.
            pred1 = mx.stop_gradient(pred1)  # type: ignore[union-attr]
            yuv6_pair = mx.concatenate(  # type: ignore[union-attr]
                [rgb_to_yuv6_mlx(pred0 * 255.0), rgb_to_yuv6_mlx(pred1 * 255.0)],
                axis=-1,
            )
            pose = pose_fn(yuv6_pair)
            # The acceptance cap is movement from the *initial live pose*, not
            # distance to the source video's target pose.  The v9 smoke showed
            # target-pose compensation can improve a proxy while still
            # violating the trust cap.  Compensate on the exact guard surface.
            trust_target = (
                initial_pose_reference
                if initial_pose_reference is not None
                else mx.array(target_pose_np)  # type: ignore[union-attr]
            )
            return mx.mean((pose - trust_target) ** 2)  # type: ignore[union-attr]

        def _pose_target_frame0_loss_fn(model_obj: Any) -> Any:
            pred0, pred1 = _predict_pair01(model_obj)
            pred1 = mx.stop_gradient(pred1)  # type: ignore[union-attr]
            yuv6_pair = mx.concatenate(  # type: ignore[union-attr]
                [rgb_to_yuv6_mlx(pred0 * 255.0), rgb_to_yuv6_mlx(pred1 * 255.0)],
                axis=-1,
            )
            pose = pose_fn(yuv6_pair)
            return mx.mean((pose - target_pose_reference) * (pose - target_pose_reference))  # type: ignore[operator, union-attr]

        def _apply_compensation_step(
            base_snapshot: list[tuple[Any, Any]],
            grads_tree: Any,
            step_lr: float,
        ) -> list[str]:
            grad_by_name = dict(tree_flatten(grads_tree))  # type: ignore[operator]
            updated: list[tuple[Any, Any]] = []
            applied: list[str] = []
            for raw_name, leaf in base_snapshot:
                grad = grad_by_name.get(raw_name)
                flat = _flat_param_name(raw_name)
                # Compensation is pose-only and restricted to the frame0 RGB
                # head via the canonical allow-list. It is deliberately NOT the
                # birth allow-list: head_rgb_0 must never enter the seg-side
                # birth scope.
                if leaf is None or grad is None or not allowed_pose_compensation_update_name(flat):
                    updated.append((raw_name, leaf))
                    continue
                grad_sq = mx.sum(grad * grad)  # type: ignore[union-attr]
                mx.eval(grad_sq)  # type: ignore[union-attr]
                if float(grad_sq.item()) <= 0.0:
                    updated.append((raw_name, leaf))
                    continue
                update = mx.array(leaf) - float(step_lr) * grad  # type: ignore[union-attr]
                updated.append((raw_name, update))
                applied.append(flat)
            self.update(tree_unflatten(updated))
            mx.eval(self.parameters())  # type: ignore[union-attr]
            return applied

        def _attempt_frame0_compensation(
            pre_compensation_snapshot: list[tuple[Any, Any]],
            frame1_uint8_before: np.ndarray,
            best_nonrate_value: float | None,
        ) -> dict[str, Any]:
            """Run a frame0-only pose compensation and price the composite.

            Returns a record with ``accepted`` plus composite telemetry. On
            rejection the model state is restored to ``pre_compensation_snapshot``
            (the frame1 birth state) so the caller's existing backtracking path
            sees an unchanged world.
            """

            applied_compensation_names: set[str] = set()
            for _comp_step in range(compensation_step_count):
                comp_snapshot = _snapshot_parameters()
                comp_loss, comp_grads = comp_loss_and_grad_fn(self)
                mx.eval(comp_loss)  # type: ignore[union-attr]
                if not math.isfinite(float(comp_loss.item())):
                    _restore_parameters(comp_snapshot)
                    break
                applied = _apply_compensation_step(comp_snapshot, comp_grads, compensation_lr)
                if not applied:
                    break
                applied_compensation_names.update(applied)
            composite = _region_candidate_state()
            # Structural seg-safety proof: frame0 compensation MUST NOT move the
            # frame1 receiver uint8. SegNet reads frame1 only, so a moved frame1
            # would mean the seg term changed — a contract violation, not a tune.
            frame1_uint8_after = composite["uint8"]
            frame1_uint8_unchanged = bool(np.array_equal(frame1_uint8_before, frame1_uint8_after))
            if not frame1_uint8_unchanged:
                _restore_parameters(pre_compensation_snapshot)
                mx.eval(self.parameters())  # type: ignore[union-attr]
                raise RuntimeError(
                    "frame0 pose compensation moved the frame1 receiver uint8; "
                    "head_rgb_0 must not affect frame1 (SegNet reads frame1 only)"
                )
            composite_pose_delta = _pose_delta_l2(composite["pose"])
            composite_nonrate = _nonrate_score(
                composite["d_seg_batch"],
                composite["d_pose_batch"],
            )
            composite_decision = _target_birth_admission_decision(
                new_d_seg=float(composite["d_seg_batch"]),
                new_d_pose=composite["d_pose_batch"],
                pose_output_l2_delta=composite_pose_delta,
            )
            composite_delta_score_nonrate = (
                None
                if composite_nonrate is None or initial_nonrate is None
                else float(composite_nonrate - initial_nonrate)
            )
            raw_cap_ok = composite_decision["raw_cap_decision"] == "satisfied"
            exact_score_ok = composite_decision["exact_score_decision"] == "accepted"
            catastrophic_guard_ok = composite_decision["catastrophic_guard_decision"] == "satisfied"
            joint_improved = (
                composite_nonrate is not None
                and best_nonrate_value is not None
                and composite_nonrate < best_nonrate_value - float(exact_score_epsilon_batch_local)
            )
            accepted = bool(
                joint_improved
                and exact_score_ok
                and catastrophic_guard_ok
                and applied_compensation_names
            )
            record: dict[str, Any] = {
                "attempted": True,
                "frame": 0,
                "accepted": accepted,
                "frame1_receiver_uint8_unchanged": frame1_uint8_unchanged,
                "compensation_updated_parameter_names": sorted(applied_compensation_names),
                "composite_delta_score_nonrate": composite_delta_score_nonrate,
                "composite_new_nonrate_score": (None if composite_nonrate is None else float(composite_nonrate)),
                "composite_pose_output_delta_l2": (
                    None if composite_pose_delta is None else float(composite_pose_delta)
                ),
                "composite_pose_cap_satisfied": bool(raw_cap_ok),
                "composite_raw_cap_decision": composite_decision["raw_cap_decision"],
                "composite_exact_score_decision": composite_decision["exact_score_decision"],
                "composite_catastrophic_guard_decision": composite_decision["catastrophic_guard_decision"],
                "would_accept_exact_score_if_raw_cap_disabled": bool(
                    composite_decision["would_accept_exact_score_if_raw_cap_disabled"]
                ),
                "would_accept_without_catastrophic_guard": bool(
                    composite_decision["would_accept_without_catastrophic_guard"]
                ),
                "rejection_source": composite_decision["rejection_source"],
                "admission_decision": {
                    key: value
                    for key, value in composite_decision.items()
                    if key not in {"old_d_seg", "new_d_seg", "old_d_pose", "new_d_pose"}
                },
                "composite_d_seg_batch": float(composite["d_seg_batch"]),
                "composite_d_pose_batch": (
                    None if composite["d_pose_batch"] is None else float(composite["d_pose_batch"])
                ),
                "composite_stats": dict(composite["stats"]),
                "composite_state": composite,
            }
            if not accepted:
                _restore_parameters(pre_compensation_snapshot)
                mx.eval(self.parameters())  # type: ignore[union-attr]
            return record

        loss_and_grad_fn = nn.value_and_grad(self, _loss_fn)  # type: ignore[union-attr]
        comp_loss_and_grad_fn = (
            nn.value_and_grad(self, _pose_compensation_loss_fn)  # type: ignore[union-attr]
            if pose_available and target_pose_np is not None
            else None
        )
        pose_target_loss_and_grad_fn = (
            nn.value_and_grad(self, _pose_target_frame0_loss_fn)  # type: ignore[union-attr]
            if pose_available and target_pose_np is not None
            else None
        )
        birth_gradient_surface = (
            "fakequant_forward_ste"
            if bool(require_fakequant_survival)
            else "live_forward"
        )
        birth_gradient_surface_eval_count = 0

        def _value_and_grad_on_birth_surface(model_obj: Any) -> tuple[Any, Any]:
            nonlocal birth_gradient_surface_eval_count
            if not bool(require_fakequant_survival):
                birth_gradient_surface_eval_count += 1
                return loss_and_grad_fn(model_obj)
            snapshot = _fakequant_config_snapshot()
            try:
                self.configure_decoder_fake_quant_forward(
                    enabled=True,
                    quant_bits=fakequant_bits,
                    per_tensor_bits=None,
                    stage_controlled=False,
                )
                loss_value, grads_tree = loss_and_grad_fn(model_obj)
                grad_leaves = [
                    leaf
                    for _name, leaf in tree_flatten(grads_tree)  # type: ignore[operator]
                    if leaf is not None
                ]
                mx.eval(loss_value, *grad_leaves)  # type: ignore[union-attr]
                birth_gradient_surface_eval_count += 1
                return loss_value, grads_tree
            finally:
                _restore_fakequant_config(snapshot)

        initial_snapshot = _snapshot_parameters()
        parameter_group_sha256_before = _parameter_group_sha256(initial_snapshot)
        blockers: list[str] = []
        if not pose_available:
            blockers.append("hinerv_target_region_birth_pose_trust_telemetry_missing")
        accepted_step_count = 0
        rejected_step_count = 0
        subquantum_rejected_step_count = 0
        pose_guard_rejected_step_count = 0
        joint_score_rejected_step_count = 0
        no_progress_rejected_step_count = 0
        fakequant_survival_rejected_step_count = 0
        best_nonrate = initial_nonrate
        receiver_quantum_growth_attempt_count = 0
        backtracking_attempt_count = 0
        loss_history: list[float] = []
        accepted_update_names: set[str] = set()
        last_grad_norm_by_group: dict[str, float] = {}
        last_update_norm_by_group: dict[str, float] = {}
        max_accepted_pose_delta_l2 = 0.0
        best_unsolved_tail_stats = dict(before_unsolved_tail_stats)
        candidate_attempt_count = 0
        region_progress_candidate_count = 0
        spill_rejected_candidate_count = 0
        pose_cap_satisfied_candidate_count = 0
        joint_score_improved_candidate_count = 0
        pose_rejected_candidate_count = 0
        joint_rejected_candidate_count = 0
        no_progress_candidate_count = 0
        fakequant_survival_candidate_count = 0
        fakequant_survival_rejected_candidate_count = 0
        argmax_birth_quantum_growth_attempt_count = 0
        max_candidate_receiver_uint8_changed_pixels_region = 0
        max_candidate_argmax_flipped_pixels_region = 0
        max_candidate_region_hard_won_delta = 0
        max_candidate_region_hard_ratio_delta = 0.0
        max_candidate_already_won_lost_count = 0
        max_candidate_outside_correct_to_wrong_count = 0
        max_candidate_global_correct_to_wrong_count = 0
        max_candidate_margin_mean_improvement = 0.0
        max_candidate_margin_p50_improvement = 0.0
        max_candidate_nonrate_improvement = 0.0
        max_candidate_pose_delta_l2: float | None = None
        min_candidate_pose_delta_l2: float | None = None
        max_pose_rejected_receiver_uint8_changed_pixels_region = 0
        max_pose_rejected_region_hard_won_delta = 0
        max_pose_rejected_margin_mean_improvement = 0.0
        min_pose_rejected_pose_delta_l2: float | None = None
        # Frame0 composite-compensation telemetry (pose-only). These remain
        # at their no-op defaults on the no-pose-teacher path so that path is
        # behaviorally byte-identical to before this operator existed.
        pose_compensation_attempted = False
        pose_compensation_accepted = False
        composite_accepted_count = 0
        composite_attempt_count = 0
        composite_updated_parameter_names: set[str] = set()
        composite_records: list[dict[str, Any]] = []
        last_composite_delta_score_nonrate: float | None = None
        current_lr = lr
        consecutive_rejected_steps = 0
        max_quantum_growth_attempts = 20
        max_argmax_birth_quantum_growth_attempts = max_quantum_growth_attempts
        max_backtracking_attempts = 8
        min_lr = lr * (0.5**max_backtracking_attempts)
        candidate_attempt_records: list[dict[str, Any]] = []
        post_acceptance_terminal_events: list[str] = []
        accepted_receipt_snapshot: list[tuple[Any, Any]] | None = None
        accepted_birth_only_snapshot: list[tuple[Any, Any]] | None = None
        accepted_birth_only_state: dict[str, Any] | None = None
        accepted_birth_only_updated_names: list[str] = []
        accepted_birth_only_decision: dict[str, Any] | None = None
        accepted_composite_state: dict[str, Any] | None = None
        accepted_composite_updated_names: list[str] = []
        accepted_composite_decision: dict[str, Any] | None = None
        target_geometry_exhausted = False

        def _gate_with_optional_compensation(
            *,
            region_progress: bool,
            candidate: dict[str, Any],
        ) -> tuple[dict[str, Any], list[tuple[Any, Any]]] | None:
            """Try a frame0 compensation when a birth step lost pose/joint.

            Returns the admissible composite ``_region_candidate_state`` dict
            plus the pre-compensation frame1-birth snapshot when the composite
            strictly improves the exact nonrate score AND satisfies the pose
            cap; otherwise restores the frame1 birth world and returns
            ``None`` so the caller falls back to the existing backtrack/reject
            path.
            Eligibility requires region progress and an available pose teacher;
            on the no-pose-teacher path it is a no-op (returns ``None``), which
            keeps that path behaviorally byte-identical.
            """

            nonlocal pose_compensation_attempted, pose_compensation_accepted
            nonlocal composite_accepted_count, composite_attempt_count
            nonlocal last_composite_delta_score_nonrate
            if not (region_progress and comp_loss_and_grad_fn is not None):
                return None
            pose_compensation_attempted = True
            composite_attempt_count += 1
            # The frame1 birth step is currently live (applied and receiver-
            # visible). Snapshot it so a rejected composite restores exactly
            # the frame1 birth world.
            frame1_birth_snapshot = _snapshot_parameters()
            record = _attempt_frame0_compensation(
                frame1_birth_snapshot,
                candidate["uint8"],
                best_nonrate,
            )
            composite_records.append(record)
            if not record["accepted"]:
                # _attempt_frame0_compensation already restored the frame1
                # birth state on reject; the caller restores to pre-step next.
                return None
            pose_compensation_accepted = True
            composite_accepted_count += 1
            composite_updated_parameter_names.update(record["compensation_updated_parameter_names"])
            last_composite_delta_score_nonrate = record["composite_delta_score_nonrate"]
            return record["composite_state"], frame1_birth_snapshot

        for _step in range(step_count):
            base_snapshot = _snapshot_parameters()
            _, step_base_pred1 = _predict_pair01(self)
            step_base_uint8 = _receiver_uint8_int(step_base_pred1)
            loss, grads = _value_and_grad_on_birth_surface(self)
            mx.eval(loss)  # type: ignore[union-attr]
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                if accepted_receipt_snapshot is not None:
                    _restore_parameters(accepted_receipt_snapshot)
                    post_acceptance_terminal_events.append(
                        "hinerv_target_region_birth_post_acceptance_loss_not_finite_restored_best_accepted_state"
                    )
                else:
                    _restore_parameters(initial_snapshot)
                    blockers.append("hinerv_target_region_birth_loss_not_finite")
                break
            loss_history.append(loss_value)
            if clip is not None:
                grads, _total_norm = mlx_optim.clip_grad_norm(grads, clip)
            step_accepted = False
            attempt_lr = current_lr
            quantum_growth_attempts = 0
            pose_violation_seen = False
            while True:
                applied_names, grad_norms, update_norms = _apply_scoped_step(
                    base_snapshot,
                    grads,
                    attempt_lr,
                )
                if not applied_names:
                    _restore_parameters(base_snapshot)
                    blockers.append("hinerv_target_region_birth_no_scoped_gradient_signal")
                    break
                candidate = _region_candidate_state()
                changed_in_region = _uint8_changed_in_region(
                    step_base_uint8,
                    candidate["uint8"],
                )
                if changed_in_region <= 0:
                    _restore_parameters(base_snapshot)
                    if quantum_growth_attempts < max_quantum_growth_attempts:
                        quantum_growth_attempts += 1
                        receiver_quantum_growth_attempt_count += 1
                        attempt_lr *= 2.0
                        continue
                    if pose_violation_seen:
                        # The growth/shrink churn was caused by the pose cap:
                        # every receiver-visible variant breached it.
                        pose_guard_rejected_step_count += 1
                    else:
                        subquantum_rejected_step_count += 1
                    rejected_step_count += 1
                    break
                candidate_update_orientation = "gradient_descent"
                bidirectional_probe: dict[str, Any] = {
                    "attempted": False,
                    "selected_orientation": candidate_update_orientation,
                }
                descent_transitions_probe = region_argmax_transition_counts(
                    initial_argmax_np,
                    candidate["argmax_np"],
                    region_bool_np,
                    birth_class,
                )
                descent_nonrate_probe = _nonrate_score(
                    candidate["d_seg_batch"],
                    candidate["d_pose_batch"],
                )
                descent_delta_probe = (
                    None
                    if descent_nonrate_probe is None or initial_nonrate is None
                    else float(descent_nonrate_probe - initial_nonrate)
                )
                # Real scorer smokes can expose sign/pathology that synthetic
                # teachers hide: a mathematically valid autograd direction can
                # cross uint8 quanta but churn wrong->wrong classes and worsen
                # the exact score.  Probe the mirrored signed step against the
                # SAME receiver/evaluate.py surfaces before concluding the
                # region is impossible.
                if int(descent_transitions_probe["wrong_to_target_count"]) <= 0 and (
                    descent_delta_probe is None or descent_delta_probe >= 0.0
                ):
                    _restore_parameters(base_snapshot)
                    mirror_applied_names, mirror_grad_norms, mirror_update_norms = _apply_scoped_step(
                        base_snapshot,
                        grads,
                        attempt_lr,
                        step_direction=1.0,
                    )
                    mirror_candidate: dict[str, Any] | None = None
                    mirror_changed_in_region = 0
                    mirror_transitions_probe: dict[str, int] | None = None
                    mirror_delta_probe: float | None = None
                    if mirror_applied_names:
                        mirror_candidate = _region_candidate_state()
                        mirror_changed_in_region = _uint8_changed_in_region(
                            step_base_uint8,
                            mirror_candidate["uint8"],
                        )
                        if mirror_changed_in_region > 0:
                            mirror_transitions_probe = region_argmax_transition_counts(
                                initial_argmax_np,
                                mirror_candidate["argmax_np"],
                                region_bool_np,
                                birth_class,
                            )
                            mirror_nonrate_probe = _nonrate_score(
                                mirror_candidate["d_seg_batch"],
                                mirror_candidate["d_pose_batch"],
                            )
                            mirror_delta_probe = (
                                None
                                if mirror_nonrate_probe is None or initial_nonrate is None
                                else float(mirror_nonrate_probe - initial_nonrate)
                            )
                    descent_delta_sort = float("inf") if descent_delta_probe is None else float(descent_delta_probe)
                    mirror_delta_sort = float("inf") if mirror_delta_probe is None else float(mirror_delta_probe)
                    mirror_wrong_to_target = (
                        0
                        if mirror_transitions_probe is None
                        else int(mirror_transitions_probe["wrong_to_target_count"])
                    )
                    descent_wrong_to_target = int(descent_transitions_probe["wrong_to_target_count"])
                    choose_mirror = bool(
                        mirror_candidate is not None
                        and mirror_changed_in_region > 0
                        and (
                            mirror_wrong_to_target > descent_wrong_to_target
                            or (
                                mirror_wrong_to_target == descent_wrong_to_target
                                and mirror_delta_sort < descent_delta_sort
                            )
                        )
                    )
                    bidirectional_probe = {
                        "attempted": True,
                        "descent_wrong_to_target_count": int(descent_wrong_to_target),
                        "descent_exact_delta_score_nonrate": descent_delta_probe,
                        "mirror_receiver_uint8_changed_pixels_region": int(
                            mirror_changed_in_region
                        ),
                        "mirror_wrong_to_target_count": int(mirror_wrong_to_target),
                        "mirror_exact_delta_score_nonrate": mirror_delta_probe,
                        "selected_orientation": (
                            "gradient_ascent_mirror" if choose_mirror else "gradient_descent"
                        ),
                    }
                    if choose_mirror and mirror_candidate is not None:
                        candidate = mirror_candidate
                        changed_in_region = mirror_changed_in_region
                        applied_names = mirror_applied_names
                        grad_norms = mirror_grad_norms
                        update_norms = mirror_update_norms
                        candidate_update_orientation = "gradient_ascent_mirror"
                    else:
                        # Restore the original descent candidate for the
                        # existing backtracking/rejection path.
                        _restore_parameters(base_snapshot)
                        applied_names, grad_norms, update_norms = _apply_scoped_step(
                            base_snapshot,
                            grads,
                            attempt_lr,
                            step_direction=-1.0,
                        )
                        candidate = _region_candidate_state()
                        changed_in_region = _uint8_changed_in_region(
                            step_base_uint8,
                            candidate["uint8"],
                        )
                hard_birth_probe_transitions = region_argmax_transition_counts(
                    initial_argmax_np,
                    candidate["argmax_np"],
                    region_bool_np,
                    birth_class,
                )
                if (
                    int(hard_birth_probe_transitions["wrong_to_target_count"]) <= 0
                    and quantum_growth_attempts < max_argmax_birth_quantum_growth_attempts
                ):
                    _restore_parameters(base_snapshot)
                    quantum_growth_attempts += 1
                    receiver_quantum_growth_attempt_count += 1
                    argmax_birth_quantum_growth_attempt_count += 1
                    attempt_lr *= 2.0
                    continue
                # Region progress is computed up-front (it is needed both by
                # the final acceptance rule and by the frame0-compensation
                # eligibility test at the pose/joint gates below). The frame1
                # birth step is receiver-visible at this point; the only open
                # questions are pose cap, joint score, and region progress.
                tail_stats = candidate["unsolved_tail_stats"]
                region_transitions = candidate["region_argmax_transitions"]
                tail_hard_won_delta = float(
                    tail_stats["region_hard_won_pixels"]
                    - best_unsolved_tail_stats["region_hard_won_pixels"]
                )
                tail_hard_ratio_delta = float(
                    tail_stats["region_hard_ratio"]
                    - best_unsolved_tail_stats["region_hard_ratio"]
                )
                tail_margin_mean_improvement = float(
                    best_unsolved_tail_stats["margin_mean"]
                    - tail_stats["margin_mean"]
                )
                tail_margin_p50_improvement = float(
                    best_unsolved_tail_stats["margin_p50"]
                    - tail_stats["margin_p50"]
                )
                net_target_support_delta = int(region_transitions["net_target_support_delta"])
                already_won_lost_count = int(candidate["already_won_lost_count"])
                global_transitions = dict(candidate["global_target_transitions"])
                outside_correct_to_wrong_count = int(
                    global_transitions["outside_correct_to_wrong_count"]
                )
                global_correct_to_wrong_count = int(
                    global_transitions["global_correct_to_wrong_count"]
                )
                support_spill = bool(
                    already_won_lost_count > 0
                    or (global_already_won_preserve_active and outside_correct_to_wrong_count > 0)
                    or net_target_support_delta < 0
                )
                ratio_progress = bool(net_target_support_delta > 0 and not support_spill)
                # Mean margin registers single-pixel receiver motion; the
                # median stays pure telemetry because one flipped pixel cannot
                # move it, which would re-open the quantum-growth/backtrack
                # ping-pong this acceptance rule exists to terminate.
                margin_progress = bool(
                    tail_margin_mean_improvement > float(min_margin_mean_drop)
                    and not support_spill
                )
                region_progress = bool(ratio_progress or margin_progress)
                pose_delta = _pose_delta_l2(candidate["pose"])
                candidate_nonrate = _nonrate_score(
                    candidate["d_seg_batch"],
                    candidate["d_pose_batch"],
                )

                candidate_decision = _target_birth_admission_decision(
                    new_d_seg=float(candidate["d_seg_batch"]),
                    new_d_pose=candidate["d_pose_batch"],
                    pose_output_l2_delta=pose_delta,
                )
                pose_missing_research_smoke = bool(
                    candidate_decision.get("pose_trust_evidence_missing_but_not_required")
                )
                raw_cap_rejects = candidate_decision["raw_cap_decision"] not in {
                    "satisfied",
                    "not_applicable",
                }
                exact_score_rejects = (
                    candidate_decision["exact_score_decision"] != "accepted"
                    and not pose_missing_research_smoke
                )
                catastrophic_rejects = candidate_decision["catastrophic_guard_decision"] != "satisfied"
                joint_rejects = (
                    candidate_nonrate is not None
                    and best_nonrate is not None
                    and candidate_nonrate >= best_nonrate - float(exact_score_epsilon_batch_local)
                )
                fakequant_admission: dict[str, Any] = {
                    "schema": "hi_nerv_target_region_birth_fakequant_admission.v1",
                    "required": bool(require_fakequant_survival),
                    "fakequant_bits": int(fakequant_bits),
                    "survived": None,
                    "reason": (
                        "not_required"
                        if not bool(require_fakequant_survival)
                        else "live_candidate_not_admissible"
                    ),
                }
                fakequant_rejects = False
                if (
                    bool(require_fakequant_survival)
                    and region_progress
                    and not catastrophic_rejects
                    and not exact_score_rejects
                    and not joint_rejects
                ):
                    fakequant_survival_candidate_count += 1
                    fakequant_admission = _fakequant_survival_for_candidate()
                    fakequant_rejects = bool(fakequant_admission.get("survived") is not True)
                    if fakequant_rejects:
                        fakequant_survival_rejected_candidate_count += 1
                candidate_attempt_count += 1
                candidate_region_hard_won_delta = int(tail_hard_won_delta)
                candidate_region_hard_ratio_delta = float(tail_hard_ratio_delta)
                candidate_margin_mean_improvement = float(tail_margin_mean_improvement)
                candidate_margin_p50_improvement = float(tail_margin_p50_improvement)
                candidate_argmax_flipped_region = int(
                    np.count_nonzero((candidate["argmax_np"] != initial_argmax_np) & region_bool_np)
                )
                max_candidate_receiver_uint8_changed_pixels_region = max(
                    max_candidate_receiver_uint8_changed_pixels_region,
                    int(changed_in_region),
                )
                max_candidate_argmax_flipped_pixels_region = max(
                    max_candidate_argmax_flipped_pixels_region,
                    candidate_argmax_flipped_region,
                )
                max_candidate_region_hard_won_delta = max(
                    max_candidate_region_hard_won_delta,
                    candidate_region_hard_won_delta,
                )
                max_candidate_region_hard_ratio_delta = max(
                    max_candidate_region_hard_ratio_delta,
                    candidate_region_hard_ratio_delta,
                )
                max_candidate_already_won_lost_count = max(
                    max_candidate_already_won_lost_count,
                    int(already_won_lost_count),
                )
                max_candidate_outside_correct_to_wrong_count = max(
                    max_candidate_outside_correct_to_wrong_count,
                    int(outside_correct_to_wrong_count),
                )
                max_candidate_global_correct_to_wrong_count = max(
                    max_candidate_global_correct_to_wrong_count,
                    int(global_correct_to_wrong_count),
                )
                max_candidate_margin_mean_improvement = max(
                    max_candidate_margin_mean_improvement,
                    candidate_margin_mean_improvement,
                )
                max_candidate_margin_p50_improvement = max(
                    max_candidate_margin_p50_improvement,
                    candidate_margin_p50_improvement,
                )
                if pose_delta is not None:
                    pose_delta_float = float(pose_delta)
                    max_candidate_pose_delta_l2 = (
                        pose_delta_float
                        if max_candidate_pose_delta_l2 is None
                        else max(max_candidate_pose_delta_l2, pose_delta_float)
                    )
                    min_candidate_pose_delta_l2 = (
                        pose_delta_float
                        if min_candidate_pose_delta_l2 is None
                        else min(min_candidate_pose_delta_l2, pose_delta_float)
                    )
                if candidate_nonrate is not None and best_nonrate is not None:
                    max_candidate_nonrate_improvement = max(
                        max_candidate_nonrate_improvement,
                        float(best_nonrate - candidate_nonrate),
                    )
                if region_progress:
                    region_progress_candidate_count += 1
                else:
                    no_progress_candidate_count += 1
                    if support_spill:
                        spill_rejected_candidate_count += 1
                if not raw_cap_rejects:
                    pose_cap_satisfied_candidate_count += 1
                if candidate_nonrate is not None and best_nonrate is not None and candidate_nonrate < best_nonrate:
                    joint_score_improved_candidate_count += 1
                if raw_cap_rejects:
                    pose_rejected_candidate_count += 1
                    max_pose_rejected_receiver_uint8_changed_pixels_region = max(
                        max_pose_rejected_receiver_uint8_changed_pixels_region,
                        int(changed_in_region),
                    )
                    max_pose_rejected_region_hard_won_delta = max(
                        max_pose_rejected_region_hard_won_delta,
                        candidate_region_hard_won_delta,
                    )
                    max_pose_rejected_margin_mean_improvement = max(
                        max_pose_rejected_margin_mean_improvement,
                        candidate_margin_mean_improvement,
                    )
                    if pose_delta is not None:
                        pose_delta_float = float(pose_delta)
                        min_pose_rejected_pose_delta_l2 = (
                            pose_delta_float
                            if min_pose_rejected_pose_delta_l2 is None
                            else min(min_pose_rejected_pose_delta_l2, pose_delta_float)
                        )
                if joint_rejects:
                    joint_rejected_candidate_count += 1
                candidate_attempt_record: dict[str, Any] = {
                    "step_index": int(len(loss_history) - 1),
                    "attempt_learning_rate": float(attempt_lr),
                    "target_geometry_mode": geometry_mode,
                    "active_tail_pixel_count_before_attempt": int(active_tail_pixels),
                    "active_tail_geometry": dict(active_tail_geometry),
                    "receiver_uint8_changed_pixels_region": int(changed_in_region),
                    "receiver_uint8_changed_pixels_active_tail": int(
                        np.count_nonzero(np.any(step_base_uint8 != candidate["uint8"], axis=-1) & (active_tail_mask_np > 0.0))
                    ),
                    "receiver_uint8_changed_pixels_unsolved_tail": int(
                        np.count_nonzero(np.any(step_base_uint8 != candidate["uint8"], axis=-1) & unsolved_tail_bool_np)
                    ),
                    "argmax_changed_count_region": int(candidate_argmax_flipped_region),
                    "region_wrong_to_target_count": int(region_transitions["wrong_to_target_count"]),
                    "region_target_to_wrong_count": int(region_transitions["target_to_wrong_count"]),
                    "region_net_target_support_delta": int(net_target_support_delta),
                    "already_won_lost_count": int(already_won_lost_count),
                    "outside_correct_to_wrong_count": int(outside_correct_to_wrong_count),
                    "outside_wrong_to_correct_count": int(
                        global_transitions["outside_wrong_to_correct_count"]
                    ),
                    "global_correct_to_wrong_count": int(global_correct_to_wrong_count),
                    "global_wrong_to_correct_count": int(
                        global_transitions["global_wrong_to_correct_count"]
                    ),
                    "global_net_correct_delta": int(
                        global_transitions["global_net_correct_delta"]
                    ),
                    "unsolved_tail_hard_won_delta": float(tail_hard_won_delta),
                    "unsolved_tail_hard_ratio_delta": float(tail_hard_ratio_delta),
                    "unsolved_tail_margin_mean_improvement": float(tail_margin_mean_improvement),
                    "unsolved_tail_margin_p50_improvement": float(tail_margin_p50_improvement),
                    "support_spill": bool(support_spill),
                    "region_progress": bool(region_progress),
                    "pose_output_delta_l2": (None if pose_delta is None else float(pose_delta)),
                    "pose_cap_satisfied": bool(not raw_cap_rejects),
                    "raw_cap_decision": candidate_decision["raw_cap_decision"],
                    "raw_pose_cap_result": candidate_decision["raw_pose_cap_result"],
                    "exact_score_decision": candidate_decision["exact_score_decision"],
                    "catastrophic_guard_decision": candidate_decision["catastrophic_guard_decision"],
                    "catastrophic_guard_reasons": list(candidate_decision["catastrophic_guard_reasons"]),
                    "would_accept_exact_score_if_raw_cap_disabled": bool(
                        candidate_decision["would_accept_exact_score_if_raw_cap_disabled"]
                    ),
                    "would_accept_without_catastrophic_guard": bool(
                        candidate_decision["would_accept_without_catastrophic_guard"]
                    ),
                    "rejected_by_raw_pose_cap": bool(candidate_decision["rejected_by_raw_pose_cap"]),
                    "rejected_by_exact_delta_score": bool(candidate_decision["rejected_by_exact_delta_score"]),
                    "rejected_by_catastrophic_pose_guard": bool(
                        candidate_decision["rejected_by_catastrophic_pose_guard"]
                    ),
                    "rejection_source": candidate_decision["rejection_source"],
                    "seg_score_delta": candidate_decision["seg_score_delta"],
                    "pose_score_delta": candidate_decision["pose_score_delta"],
                    "exact_delta_score_nonrate": candidate_decision["exact_delta_score_nonrate"],
                    "joint_score_improved": bool(not joint_rejects and not exact_score_rejects),
                    "candidate_nonrate_score": (
                        None if candidate_nonrate is None else float(candidate_nonrate)
                    ),
                    "best_nonrate_score_before_attempt": (
                        None if best_nonrate is None else float(best_nonrate)
                    ),
                    "fakequant_admission": fakequant_admission,
                    "fakequant_survival_required": bool(require_fakequant_survival),
                    "birth_gradient_surface": birth_gradient_surface,
                    "fakequant_survived": fakequant_admission.get("survived"),
                    "argmax_birth_quantum_growth_attempts_before_attempt": int(
                        quantum_growth_attempts
                    ),
                    "decision": "pending",
                    "update_orientation": candidate_update_orientation,
                    "bidirectional_probe": bidirectional_probe,
                }

                if fakequant_rejects:
                    _restore_parameters(base_snapshot)
                    if attempt_lr * 0.5 >= min_lr:
                        backtracking_attempt_count += 1
                        candidate_attempt_record["decision"] = "backtrack_fakequant_survival"
                        candidate_attempt_records.append(candidate_attempt_record)
                        attempt_lr *= 0.5
                        continue
                    fakequant_survival_rejected_step_count += 1
                    rejected_step_count += 1
                    candidate_attempt_record["decision"] = "rejected_fakequant_survival"
                    candidate_attempt_records.append(candidate_attempt_record)
                    break

                if catastrophic_rejects or exact_score_rejects or joint_rejects:
                    # The frame1 birth step is receiver-visible but loses the
                    # exact score gate, catastrophic guard, or monotone best
                    # gate. BEFORE backtracking the learning rate, attempt ONE
                    # frame0-only compensation iff region progress holds and a
                    # pose teacher is available. Raw cap violation alone is now
                    # counterfactual telemetry and must not suppress exact wins.
                    pose_violation_seen = True
                    composite_accepted = _gate_with_optional_compensation(
                        region_progress=region_progress,
                        candidate=candidate,
                    )
                    if composite_accepted is not None:
                        composite_state, frame1_birth_snapshot = composite_accepted
                        accepted_birth_only_snapshot = frame1_birth_snapshot
                        accepted_birth_only_state = candidate
                        accepted_birth_only_updated_names = sorted(applied_names)
                        accepted_birth_only_decision = dict(candidate_decision)
                        step_accepted = True
                        accepted_step_count += 1
                        accepted_update_names.update(applied_names)
                        last_grad_norm_by_group = grad_norms
                        last_update_norm_by_group = update_norms
                        best_unsolved_tail_stats = dict(composite_state["unsolved_tail_stats"])
                        target_geometry_exhausted = not _refresh_active_tail_mask(
                            composite_state["argmax_np"],
                            step_index=int(len(loss_history) - 1),
                            reason="accepted_with_frame0_pose_compensation",
                        )
                        composite_nonrate = _nonrate_score(
                            composite_state["d_seg_batch"],
                            composite_state["d_pose_batch"],
                        )
                        if composite_nonrate is not None:
                            best_nonrate = composite_nonrate
                        composite_pose_delta = _pose_delta_l2(composite_state["pose"])
                        if composite_pose_delta is not None:
                            max_accepted_pose_delta_l2 = max(max_accepted_pose_delta_l2, composite_pose_delta)
                        current_lr = attempt_lr
                        accepted_composite_state = composite_state
                        accepted_composite_updated_names = sorted(
                            [*applied_names, *composite_updated_parameter_names]
                        )
                        accepted_composite_decision = dict(
                            (composite_records[-1].get("admission_decision") or {})
                            if composite_records
                            else {}
                        )
                        accepted_receipt_snapshot = _snapshot_parameters()
                        candidate_attempt_record["decision"] = "accepted_with_frame0_pose_compensation"
                        candidate_attempt_records.append(candidate_attempt_record)
                        break
                    # No admissible composite: restore the frame1 birth step and
                    # fall back to the existing backtrack/reject disposition.
                    _restore_parameters(base_snapshot)
                    if attempt_lr * 0.5 >= min_lr:
                        backtracking_attempt_count += 1
                        candidate_attempt_record["decision"] = (
                            "backtrack_catastrophic_pose_guard"
                            if catastrophic_rejects
                            else "backtrack_exact_delta_score"
                            if exact_score_rejects
                            else "backtrack_joint_score"
                        )
                        candidate_attempt_records.append(candidate_attempt_record)
                        attempt_lr *= 0.5
                        continue
                    if catastrophic_rejects:
                        pose_guard_rejected_step_count += 1
                        candidate_attempt_record["decision"] = "rejected_catastrophic_pose_guard"
                    elif exact_score_rejects:
                        joint_score_rejected_step_count += 1
                        candidate_attempt_record["decision"] = "rejected_exact_delta_score"
                    else:
                        joint_score_rejected_step_count += 1
                        candidate_attempt_record["decision"] = "rejected_joint_score"
                    candidate_attempt_records.append(candidate_attempt_record)
                    rejected_step_count += 1
                    break
                if not region_progress:
                    _restore_parameters(base_snapshot)
                    if attempt_lr * 0.5 >= min_lr:
                        backtracking_attempt_count += 1
                        candidate_attempt_record["decision"] = "backtrack_no_unsolved_tail_progress"
                        candidate_attempt_records.append(candidate_attempt_record)
                        attempt_lr *= 0.5
                        continue
                    no_progress_rejected_step_count += 1
                    rejected_step_count += 1
                    candidate_attempt_record["decision"] = "rejected_no_unsolved_tail_progress"
                    candidate_attempt_records.append(candidate_attempt_record)
                    break
                step_accepted = True
                accepted_step_count += 1
                accepted_update_names.update(applied_names)
                accepted_birth_only_snapshot = _snapshot_parameters()
                accepted_receipt_snapshot = accepted_birth_only_snapshot
                accepted_birth_only_state = candidate
                accepted_birth_only_updated_names = sorted(applied_names)
                accepted_birth_only_decision = dict(candidate_decision)
                last_grad_norm_by_group = grad_norms
                last_update_norm_by_group = update_norms
                best_unsolved_tail_stats = dict(tail_stats)
                if candidate_nonrate is not None:
                    best_nonrate = candidate_nonrate
                if pose_delta is not None:
                    max_accepted_pose_delta_l2 = max(
                        max_accepted_pose_delta_l2,
                        pose_delta,
                    )
                target_geometry_exhausted = not _refresh_active_tail_mask(
                    candidate["argmax_np"],
                    step_index=int(len(loss_history) - 1),
                    reason="accepted",
                )
                current_lr = attempt_lr
                candidate_attempt_record["decision"] = "accepted"
                candidate_attempt_records.append(candidate_attempt_record)
                break
            if (
                "hinerv_target_region_birth_no_scoped_gradient_signal" in blockers
                or "hinerv_target_region_birth_loss_not_finite" in blockers
            ):
                if accepted_receipt_snapshot is not None:
                    if "hinerv_target_region_birth_no_scoped_gradient_signal" in blockers:
                        blockers = [
                            blocker
                            for blocker in blockers
                            if blocker != "hinerv_target_region_birth_no_scoped_gradient_signal"
                        ]
                        post_acceptance_terminal_events.append(
                            "hinerv_target_region_birth_post_acceptance_no_scoped_gradient_signal_restored_best_accepted_state"
                        )
                    if "hinerv_target_region_birth_loss_not_finite" in blockers:
                        blockers = [
                            blocker
                            for blocker in blockers
                            if blocker != "hinerv_target_region_birth_loss_not_finite"
                        ]
                        post_acceptance_terminal_events.append(
                            "hinerv_target_region_birth_post_acceptance_loss_not_finite_restored_best_accepted_state"
                        )
                    _restore_parameters(accepted_receipt_snapshot)
                break
            if step_accepted:
                consecutive_rejected_steps = 0
                if target_geometry_exhausted:
                    break
            else:
                consecutive_rejected_steps += 1
                # One chaotic step must not end a long fit; three fully
                # rejected steps in a row means the servo is genuinely stuck.
                if consecutive_rejected_steps >= 3:
                    break

        if accepted_step_count == 0:
            _restore_parameters(initial_snapshot)
            blockers.append("hinerv_target_region_birth_no_accepted_step")
        elif accepted_receipt_snapshot is not None:
            # The optimizer may continue probing after an accepted receiver
            # crossing and then hit a terminal condition (non-finite loss,
            # exhausted scoped signal, etc.).  Long-run gating cares about the
            # best accepted receiver surface, not a later failed probe.  Keep
            # those terminal events as telemetry, but emit receipts from the
            # accepted state so live -> uint8 -> scorer evidence is not erased.
            _restore_parameters(accepted_receipt_snapshot)
        else:
            blockers.append("hinerv_target_region_birth_accepted_snapshot_missing")

        final_model_snapshot = _snapshot_parameters()

        parameter_group_sha256_after = _parameter_group_sha256(_snapshot_parameters())
        out_of_scope_bit_frozen_verified = parameter_group_sha256_before.get(
            "out_of_scope"
        ) == parameter_group_sha256_after.get("out_of_scope")
        if not out_of_scope_bit_frozen_verified:
            # Fail loudly: a scoped actuator that moved out-of-scope state is
            # a contract violation, not a tunable.
            blockers.append("hinerv_target_region_birth_out_of_scope_state_mutated")
        final = _region_candidate_state()
        after_stats = final["stats"]
        after_unsolved_tail_stats = final["unsolved_tail_stats"]
        after_already_won_stats = final["already_won_stats"]
        final_argmax_np = final["argmax_np"]
        argmax_flipped_region = int(np.count_nonzero((final_argmax_np != initial_argmax_np) & region_bool_np))
        argmax_transitions = region_argmax_transition_counts(
            initial_argmax_np,
            final_argmax_np,
            region_bool_np,
            birth_class,
        )
        final_nonrate = _nonrate_score(final["d_seg_batch"], final["d_pose_batch"])
        exact_nonrate_payload: dict[str, Any] = {
            "authority": "batch_local_live_mlx",
            # Local pair-batch units never masquerade as full-video score
            # units: a 1-pair region debt drop is servo evidence, not a
            # promotion delta. Full-equivalent estimates, when computed, use
            # the separate *_full_equivalent_estimate keys.
            "normalization_scope": "batch_local",
            "old_d_seg_batch": float(initial_d_seg),
            "new_d_seg_batch": float(final["d_seg_batch"]),
            "old_d_pose_batch": (None if initial_d_pose is None else float(initial_d_pose)),
            "new_d_pose_batch": (None if final["d_pose_batch"] is None else float(final["d_pose_batch"])),
            "old_nonrate_score": (None if initial_nonrate is None else float(initial_nonrate)),
            "new_nonrate_score": (None if final_nonrate is None else float(final_nonrate)),
            "delta_score_nonrate": (
                None if initial_nonrate is None or final_nonrate is None else float(final_nonrate - initial_nonrate)
            ),
            "pose_term_available": bool(target_pose_np is not None),
        }
        uint8_changed_total = _uint8_changed_in_region(initial_uint8, final["uint8"])
        uint8_delta_abs_max = float(
            np.abs(final["uint8"].astype(np.int32) - initial_uint8.astype(np.int32))[region_bool_np].max()
            if region_bool_np.any()
            else 0.0
        )
        final_pose_delta = _pose_delta_l2(final["pose"])
        if accepted_step_count > 0 and after_stats["region_hard_ratio"] < float(target_min_region_ratio):
            blockers.append("hinerv_target_region_birth_min_ratio_floor_not_reached")
        pose_guard_payload: dict[str, Any] = {
            "available": bool(pose_available),
            "input_convention": "concat_yuv6_pair_nhwc255_frame0_then_frame1",
            "pose_input_height": int(target1.shape[1]),
            "pose_input_width": int(target1.shape[2]),
            "pose_input_contest_resolution": bool(pose_input_contest_resolution),
            "max_pose_output_delta_l2": float(max_pose_output_delta_l2),
            "raw_pose_cap_l2": float(max_pose_output_delta_l2),
            "raw_pose_cap_is_counterfactual_only": True,
            "exact_score_epsilon_batch_local": float(exact_score_epsilon_batch_local),
            "catastrophic_pose_relative_cap": float(catastrophic_pose_relative_cap),
            "catastrophic_pose_score_regression_cap": float(catastrophic_pose_score_regression_cap),
            "catastrophic_pose_output_l2_hard_cap": (
                None
                if catastrophic_pose_output_l2_hard_cap is None
                else float(catastrophic_pose_output_l2_hard_cap)
            ),
            "max_accepted_pose_output_delta_l2": float(max_accepted_pose_delta_l2),
            "final_pose_output_delta_l2": (None if final_pose_delta is None else float(final_pose_delta)),
            "pose_guard_rejected_step_count": int(pose_guard_rejected_step_count),
        }
        candidate_frontier_telemetry: dict[str, Any] = {
            "schema": "hi_nerv_target_region_birth_candidate_frontier_telemetry.v1",
            "authority": "batch_local_live_mlx_false_authority",
            "candidate_attempt_count": int(candidate_attempt_count),
            "region_progress_candidate_count": int(region_progress_candidate_count),
            "spill_rejected_candidate_count": int(spill_rejected_candidate_count),
            "pose_cap_satisfied_candidate_count": int(pose_cap_satisfied_candidate_count),
            "joint_score_improved_candidate_count": int(joint_score_improved_candidate_count),
            "pose_rejected_candidate_count": int(pose_rejected_candidate_count),
            "joint_rejected_candidate_count": int(joint_rejected_candidate_count),
            "no_progress_candidate_count": int(no_progress_candidate_count),
            "fakequant_survival_required": bool(require_fakequant_survival),
            "birth_gradient_surface": birth_gradient_surface,
            "birth_gradient_surface_eval_count": int(birth_gradient_surface_eval_count),
            "fakequant_survival_bits": int(fakequant_bits),
            "fakequant_survival_candidate_count": int(fakequant_survival_candidate_count),
            "fakequant_survival_rejected_candidate_count": int(
                fakequant_survival_rejected_candidate_count
            ),
            "argmax_birth_quantum_growth_attempt_count": int(
                argmax_birth_quantum_growth_attempt_count
            ),
            "max_argmax_birth_quantum_growth_attempts_per_step": int(
                max_argmax_birth_quantum_growth_attempts
            ),
            "max_candidate_receiver_uint8_changed_pixels_region": int(
                max_candidate_receiver_uint8_changed_pixels_region
            ),
            "max_candidate_argmax_flipped_pixels_region": int(max_candidate_argmax_flipped_pixels_region),
            "max_candidate_region_hard_won_delta": int(max_candidate_region_hard_won_delta),
            "max_candidate_region_hard_ratio_delta": float(max_candidate_region_hard_ratio_delta),
            "max_candidate_already_won_lost_count": int(max_candidate_already_won_lost_count),
            "max_candidate_outside_correct_to_wrong_count": int(
                max_candidate_outside_correct_to_wrong_count
            ),
            "max_candidate_global_correct_to_wrong_count": int(
                max_candidate_global_correct_to_wrong_count
            ),
            "max_candidate_margin_mean_improvement": float(max_candidate_margin_mean_improvement),
            "max_candidate_margin_p50_improvement": float(max_candidate_margin_p50_improvement),
            "progress_reference": "initial_unsolved_tail_with_already_won_preservation",
            "trust_region_controls": {
                "lambda_support_preserve": float(lambda_support_preserve),
                "lambda_outside_argmax_preserve": float(lambda_outside_argmax_preserve),
                "lambda_already_won_hard_preserve": float(lambda_already_won_hard_preserve),
                "lambda_already_won_rgb_preserve": float(lambda_already_won_rgb_preserve),
                "already_won_margin_floor": float(already_won_margin_floor),
                "lambda_pose_trust_preserve": float(lambda_pose_trust_preserve),
                "lambda_pose_target": float(lambda_pose_target),
                "outside_region_pixel_count": int(outside_region_pixels),
                "global_already_won_outside_region_pixel_count": int(
                    global_already_won_pixels
                ),
                "global_already_won_preserve_active": bool(
                    global_already_won_preserve_active
                ),
                "pose_trust_cap_l2": float(max_pose_output_delta_l2),
            },
            "target_geometry": {
                "mode": geometry_mode,
                "frontier_dilation": int(frontier_dilation),
                "initial_active_tail_pixel_count": int(
                    initial_active_tail_geometry["active_tail_pixel_count"]
                ),
                "final_active_tail_pixel_count": int(active_tail_geometry["active_tail_pixel_count"]),
                "active_tail_refresh_count": int(active_tail_refresh_count),
                "history": active_tail_geometry_history,
                "parent_region_pixel_count": int(worst.region_pixel_count),
                "parent_unsolved_tail_pixel_count": int(unsolved_tail_pixel_count),
                "parent_already_won_pixel_count": int(already_won_pixel_count),
            },
            "unsolved_tail_pixel_count": int(unsolved_tail_pixel_count),
            "already_won_region_pixel_count": int(already_won_pixel_count),
            "before_unsolved_tail_margin_stats": dict(before_unsolved_tail_stats),
            "after_unsolved_tail_margin_stats": dict(after_unsolved_tail_stats),
            "before_already_won_margin_stats": (
                dict(before_already_won_stats)
                if before_already_won_stats is not None
                else None
            ),
            "after_already_won_margin_stats": (
                dict(after_already_won_stats)
                if after_already_won_stats is not None
                else None
            ),
            "final_already_won_lost_count": int(final["already_won_lost_count"]),
            "max_candidate_nonrate_improvement": float(max_candidate_nonrate_improvement),
            "max_candidate_pose_output_delta_l2": (
                None if max_candidate_pose_delta_l2 is None else float(max_candidate_pose_delta_l2)
            ),
            "min_candidate_pose_output_delta_l2": (
                None if min_candidate_pose_delta_l2 is None else float(min_candidate_pose_delta_l2)
            ),
            "max_pose_rejected_receiver_uint8_changed_pixels_region": int(
                max_pose_rejected_receiver_uint8_changed_pixels_region
            ),
            "max_pose_rejected_region_hard_won_delta": int(max_pose_rejected_region_hard_won_delta),
            "max_pose_rejected_margin_mean_improvement": float(max_pose_rejected_margin_mean_improvement),
            "min_pose_rejected_pose_output_delta_l2": (
                None if min_pose_rejected_pose_delta_l2 is None else float(min_pose_rejected_pose_delta_l2)
            ),
            "attempts": candidate_attempt_records,
            "post_acceptance_terminal_events": list(post_acceptance_terminal_events),
            "reference_region_stats": "current_best_live_mlx",
            "human_visual_fidelity_objective": False,
        }
        # Frame0 composite-compensation payload (batch-local authority). The
        # compensated scope (``head_rgb_0.*``) is recorded SEPARATELY from the
        # birth ``updated_parameter_names`` so the receipt's birth-scope check
        # never sees it: compensation is a pose-only frame0 surface, not a
        # seg-side birth scope, and admitting it must never relax the birth
        # allow-list. The per-attempt records carry the JSON-clean summary only
        # (the live MLX/np ``composite_state`` is dropped here).
        compensation_attempt_summaries = [
            {
                "accepted": bool(record["accepted"]),
                "frame1_receiver_uint8_unchanged": bool(record["frame1_receiver_uint8_unchanged"]),
                "compensation_updated_parameter_names": list(record["compensation_updated_parameter_names"]),
                "composite_delta_score_nonrate": record["composite_delta_score_nonrate"],
                "composite_new_nonrate_score": record["composite_new_nonrate_score"],
                "composite_pose_output_delta_l2": record["composite_pose_output_delta_l2"],
                "composite_pose_cap_satisfied": bool(record["composite_pose_cap_satisfied"]),
                "composite_raw_cap_decision": record.get("composite_raw_cap_decision"),
                "composite_exact_score_decision": record.get("composite_exact_score_decision"),
                "composite_catastrophic_guard_decision": record.get(
                    "composite_catastrophic_guard_decision"
                ),
                "would_accept_exact_score_if_raw_cap_disabled": bool(
                    record.get("would_accept_exact_score_if_raw_cap_disabled")
                ),
                "would_accept_without_catastrophic_guard": bool(
                    record.get("would_accept_without_catastrophic_guard")
                ),
                "rejection_source": record.get("rejection_source"),
                "admission_decision": dict(record.get("admission_decision") or {}),
                "composite_d_seg_batch": record["composite_d_seg_batch"],
                "composite_d_pose_batch": record["composite_d_pose_batch"],
            }
            for record in composite_records
        ]
        pose_compensation_payload: dict[str, Any] | None = None
        if pose_compensation_attempted:
            compensation_names_sorted = sorted(composite_updated_parameter_names)
            # Defensive scope proof: every compensated name is the frame0 head
            # ONLY, and NONE of them may be birth-scoped (that would mean a
            # frame0 edit leaked into the seg-side birth allow-list).
            escaped = [
                name
                for name in compensation_names_sorted
                if not allowed_pose_compensation_update_name(name) or allowed_birth_update_name(name)
            ]
            if escaped:
                raise RuntimeError(
                    "frame0 compensation scope escaped head_rgb_0 or collided with the "
                    f"birth allow-list: {escaped}"
                )
            pose_compensation_payload = {
                "authority": "batch_local_live_mlx",
                "normalization_scope": "batch_local",
                "pose_compensation_attempted": True,
                "pose_compensation_frame": 0,
                "composite_accepted": bool(composite_accepted_count > 0),
                "composite_attempt_count": int(composite_attempt_count),
                "composite_accepted_count": int(composite_accepted_count),
                "composite_delta_score_nonrate": last_composite_delta_score_nonrate,
                # head_rgb_0 lives in a SEPARATE compensation scope record; it
                # is deliberately NOT added to updated_parameter_names nor to
                # ALLOWED_BIRTH_UPDATE_*.
                "compensation_updated_parameter_names": compensation_names_sorted,
                "compensation_scope": "head_rgb_0",
                "frame1_receiver_uint8_unchanged_by_compensation": bool(
                    all(record["frame1_receiver_uint8_unchanged"] for record in composite_records)
                ),
                "attempts": compensation_attempt_summaries,
                "human_visual_fidelity_objective": False,
            }
        trained_groups_for_action_id = {_group_for_name(name) for name in accepted_update_names}
        if pose_compensation_payload is not None and pose_compensation_payload["composite_accepted"]:
            trained_groups_for_action_id.add("compensation_head_rgb_0")
        action_identity = birth_action_id(
            debt=worst,
            initial_group_sha256=parameter_group_sha256_before,
            trained_groups=sorted(trained_groups_for_action_id),
        )
        region_mask_bool_flat = np.ascontiguousarray(region_bool_np.reshape(-1).astype(np.uint8))
        region_mask_packbits_payload = {
            "schema": "hi_nerv_target_region_birth_mask_packbits.v1",
            "encoding": "numpy.packbits:uint8:bitorder_big",
            "shape": [int(v) for v in region_bool_np.shape],
            "true_count": int(np.count_nonzero(region_bool_np)),
            "data_b64": base64.b64encode(
                np.packbits(region_mask_bool_flat, bitorder="big").tobytes()
            ).decode("ascii"),
        }
        worst_region_payload = dict(worst.as_dict())
        worst_region_payload["region_mask_bhw_packbits"] = region_mask_packbits_payload

        def _snapshot_delta_names(
            before_snapshot: list[tuple[Any, Any]],
            after_snapshot: list[tuple[Any, Any]],
            predicate: Any,
        ) -> list[str]:
            before_by_name = {_flat_param_name(raw_name): leaf for raw_name, leaf in before_snapshot}
            out: list[str] = []
            for raw_name, leaf in after_snapshot:
                flat = _flat_param_name(raw_name)
                if not predicate(flat):
                    continue
                before_leaf = before_by_name.get(flat)
                if before_leaf is None or leaf is None:
                    continue
                if not np.array_equal(np.asarray(before_leaf), np.asarray(leaf)):
                    out.append(flat)
            return sorted(out)

        def _compose_disjoint_snapshots(
            birth_snapshot: list[tuple[Any, Any]],
            compensation_snapshot: list[tuple[Any, Any]],
        ) -> list[tuple[Any, Any]]:
            birth_by_name = {_flat_param_name(raw_name): leaf for raw_name, leaf in birth_snapshot}
            compensation_by_name = {_flat_param_name(raw_name): leaf for raw_name, leaf in compensation_snapshot}
            composed: list[tuple[Any, Any]] = []
            for raw_name, base_leaf in initial_snapshot:
                flat = _flat_param_name(raw_name)
                if allowed_pose_compensation_update_name(flat):
                    composed.append((raw_name, compensation_by_name.get(flat, base_leaf)))
                elif allowed_birth_update_name(flat):
                    composed.append((raw_name, birth_by_name.get(flat, base_leaf)))
                else:
                    composed.append((raw_name, base_leaf))
            return composed

        def _region_debt_units(argmax_np: np.ndarray) -> float:
            unsolved = int(np.count_nonzero(region_bool_np & (np.asarray(argmax_np) != birth_class)))
            return float(100.0 * unsolved / total_scored_pixels)

        def _dedupe_arm_blockers(values: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for value in values:
                if value and value not in seen:
                    seen.add(value)
                    out.append(value)
            return out

        def _ablation_arm_row(
            *,
            arm: str,
            action_kind: str,
            state: dict[str, Any] | None,
            updated_names: list[str],
            admission_decision: Mapping[str, Any] | None,
            decision: str,
            blockers_for_arm: list[str] | tuple[str, ...] = (),
            interaction_or_commutator: float | None = None,
        ) -> dict[str, Any]:
            row_blockers = [str(value) for value in blockers_for_arm]
            exact_nonrate: dict[str, Any] = {
                "authority": "batch_local_live_mlx",
                "normalization_scope": "batch_local",
                "old_d_seg_batch": float(initial_d_seg),
                "new_d_seg_batch": None,
                "old_d_pose_batch": None if initial_d_pose is None else float(initial_d_pose),
                "new_d_pose_batch": None,
                "old_nonrate_score": None if initial_nonrate is None else float(initial_nonrate),
                "new_nonrate_score": None,
                "delta_score_nonrate": None,
                "pose_term_available": bool(target_pose_np is not None),
            }
            transitions = {
                "argmax_changed_count_region": 0,
                "wrong_to_target_count": 0,
                "target_to_wrong_count": 0,
                "wrong_to_wrong_count": 0,
                "target_hard_won_count": 0,
                "target_hard_lost_count": 0,
                "net_target_support_delta": 0,
            }
            uint8_changed = 0
            uint8_delta_abs_max_arm = 0.0
            float_rgb_delta_linf = 0.0
            argmax_changed = 0
            pose_delta_l2 = None
            if state is None:
                row_blockers.append("hinerv_four_arm_state_missing")
            else:
                state_nonrate = _nonrate_score(state["d_seg_batch"], state["d_pose_batch"])
                exact_nonrate.update(
                    {
                        "new_d_seg_batch": float(state["d_seg_batch"]),
                        "new_d_pose_batch": (
                            None if state["d_pose_batch"] is None else float(state["d_pose_batch"])
                        ),
                        "new_nonrate_score": None if state_nonrate is None else float(state_nonrate),
                        "delta_score_nonrate": (
                            None
                            if initial_nonrate is None or state_nonrate is None
                            else float(state_nonrate - initial_nonrate)
                        ),
                    }
                )
                transitions = region_argmax_transition_counts(
                    initial_argmax_np,
                    state["argmax_np"],
                    region_bool_np,
                    birth_class,
                )
                uint8_changed = _uint8_changed_in_region(initial_uint8, state["uint8"])
                uint8_delta_abs_max_arm = float(
                    np.abs(state["uint8"].astype(np.int32) - initial_uint8.astype(np.int32))[region_bool_np].max()
                    if region_bool_np.any()
                    else 0.0
                )
                float_rgb_delta_linf = float(state["float_rgb_delta_linf"])
                argmax_changed = int(
                    np.count_nonzero((state["argmax_np"] != initial_argmax_np) & region_bool_np)
                )
                pose_delta_l2 = _pose_delta_l2(state["pose"])
            decision_payload = dict(admission_decision or {})
            if not decision_payload and state is not None:
                decision_payload = _target_birth_admission_decision(
                    new_d_seg=float(state["d_seg_batch"]),
                    new_d_pose=state["d_pose_batch"],
                    pose_output_l2_delta=pose_delta_l2,
                )
            exact_decision = str(decision_payload.get("exact_score_decision") or "not_applicable")
            catastrophic_decision = str(
                decision_payload.get("catastrophic_guard_decision") or "not_applicable"
            )
            accepted = bool(
                decision in {"accepted", "accepted_with_frame0_pose_compensation"}
                or (
                    exact_decision == "accepted"
                    and catastrophic_decision == "satisfied"
                    and not row_blockers
                )
            )
            old_region_debt = float(worst.score_debt_units)
            new_region_debt = old_region_debt if state is None else _region_debt_units(state["argmax_np"])
            return {
                "schema": "hi_nerv_target_region_birth_four_arm.v1",
                "action_id": action_identity,
                "surface": "live_mlx",
                "authority": "batch_local_live_mlx",
                "normalization_scope": "batch_local",
                "family": "hinerv",
                "producer": "hinerv_target_region_birth",
                "action_kind": action_kind,
                "arm": str(arm),
                "ablation_arm": str(arm),
                "decision": str(decision),
                "accepted": bool(accepted),
                "blockers": _dedupe_arm_blockers(row_blockers),
                "pair_index": int(worst.batch_index),
                "worst_region": worst.as_dict(),
                "updated_parameter_names": sorted(str(name) for name in updated_names),
                "payload_sections": sorted(str(name) for name in updated_names),
                "trained_groups": sorted(
                    {
                        _group_for_name(str(name))
                        for name in updated_names
                        if _group_for_name(str(name)) != "out_of_scope"
                    }
                ),
                "exact_nonrate": exact_nonrate,
                "admission_decision": {
                    **decision_payload,
                    "arm": str(arm),
                    "action_kind": action_kind,
                },
                "argmax_transitions": transitions,
                **transitions,
                "old_region_debt": old_region_debt,
                "new_region_debt": float(new_region_debt),
                "receiver_surface_uint8_changed_pixels": int(uint8_changed),
                "receiver_uint8_changed_pixels_region": int(uint8_changed),
                "receiver_uint8_delta_abs_max": float(uint8_delta_abs_max_arm),
                "receiver_surface_uint8_delta_abs_max": float(uint8_delta_abs_max_arm),
                "receiver_float_rgb_delta_linf": float(float_rgb_delta_linf),
                "argmax_changed_count_region": int(argmax_changed),
                "argmax_flipped_pixels_region": int(argmax_changed),
                "pose_output_l2_delta": None if pose_delta_l2 is None else float(pose_delta_l2),
                "raw_cap_decision": decision_payload.get("raw_cap_decision"),
                "catastrophic_guard_decision": decision_payload.get("catastrophic_guard_decision"),
                "would_accept_exact_score_if_raw_cap_disabled": decision_payload.get(
                    "would_accept_exact_score_if_raw_cap_disabled"
                ),
                "would_accept_without_catastrophic_guard": decision_payload.get(
                    "would_accept_without_catastrophic_guard"
                ),
                "seg_score_delta": decision_payload.get("seg_score_delta"),
                "pose_score_delta": decision_payload.get("pose_score_delta"),
                "rejection_source": decision_payload.get("rejection_source"),
                "interaction_or_commutator": interaction_or_commutator,
                "restore_state_pass": True,
                "promotion_eligible": False,
                "human_visual_fidelity_objective": False,
            }

        def _measure_frame0_pose_target_arm() -> tuple[list[tuple[Any, Any]] | None, dict[str, Any] | None, list[str], list[str]]:
            if pose_target_loss_and_grad_fn is None:
                return None, None, [], ["hinerv_four_arm_pose_teacher_missing"]
            _restore_parameters(initial_snapshot)
            updated_names: set[str] = set()
            for _ in range(ablation_compensation_steps):
                arm_snapshot = _snapshot_parameters()
                arm_loss, arm_grads = pose_target_loss_and_grad_fn(self)
                mx.eval(arm_loss)  # type: ignore[union-attr]
                if not math.isfinite(float(arm_loss.item())):
                    _restore_parameters(arm_snapshot)
                    return None, None, sorted(updated_names), ["hinerv_four_arm_frame0_pose_loss_not_finite"]
                applied = _apply_compensation_step(arm_snapshot, arm_grads, compensation_lr)
                if not applied:
                    break
                updated_names.update(applied)
            measured_snapshot = _snapshot_parameters()
            measured_state = _region_candidate_state()
            return measured_snapshot, measured_state, sorted(updated_names), []

        four_arm_ablation_payload: dict[str, Any] | None = None
        if emit_four_arm_ablation:
            restore_target_snapshot = final_model_snapshot
            arm_rows: list[dict[str, Any]] = []
            try:
                arm_a_state = accepted_birth_only_state
                arm_a_decision = accepted_birth_only_decision
                if arm_a_state is None and accepted_step_count > 0:
                    arm_a_state = final
                    arm_a_decision = _target_birth_admission_decision(
                        new_d_seg=float(final["d_seg_batch"]),
                        new_d_pose=final["d_pose_batch"],
                        pose_output_l2_delta=final_pose_delta,
                    )
                arm_rows.append(
                    _ablation_arm_row(
                        arm="A",
                        action_kind="birth_only",
                        state=arm_a_state,
                        updated_names=accepted_birth_only_updated_names or sorted(accepted_update_names),
                        admission_decision=arm_a_decision,
                        decision="accepted" if arm_a_state is not None and accepted_step_count > 0 else "blocked",
                        blockers_for_arm=(
                            []
                            if arm_a_state is not None and accepted_step_count > 0
                            else ["hinerv_four_arm_birth_only_missing"]
                        ),
                    )
                )

                arm_b_snapshot, arm_b_state, arm_b_names, arm_b_blockers = _measure_frame0_pose_target_arm()
                arm_b_decision = (
                    None
                    if arm_b_state is None
                    else _target_birth_admission_decision(
                        new_d_seg=float(arm_b_state["d_seg_batch"]),
                        new_d_pose=arm_b_state["d_pose_batch"],
                        pose_output_l2_delta=_pose_delta_l2(arm_b_state["pose"]),
                    )
                )
                arm_rows.append(
                    _ablation_arm_row(
                        arm="B",
                        action_kind="frame0_pose_target_only",
                        state=arm_b_state,
                        updated_names=arm_b_names,
                        admission_decision=arm_b_decision,
                        decision="measured" if arm_b_state is not None else "blocked",
                        blockers_for_arm=arm_b_blockers,
                    )
                )

                arm_c_state: dict[str, Any] | None = None
                arm_c_names: list[str] = []
                arm_c_blockers: list[str] = []
                if accepted_birth_only_snapshot is None:
                    arm_c_blockers.append("hinerv_four_arm_birth_snapshot_missing")
                if arm_b_snapshot is None:
                    arm_c_blockers.append("hinerv_four_arm_frame0_pose_snapshot_missing")
                if not arm_c_blockers:
                    _restore_parameters(
                        _compose_disjoint_snapshots(
                            accepted_birth_only_snapshot,  # type: ignore[arg-type]
                            arm_b_snapshot,  # type: ignore[arg-type]
                        )
                    )
                    arm_c_state = _region_candidate_state()
                    arm_c_names = sorted(
                        [
                            *_snapshot_delta_names(
                                initial_snapshot,
                                accepted_birth_only_snapshot,  # type: ignore[arg-type]
                                allowed_birth_update_name,
                            ),
                            *_snapshot_delta_names(
                                initial_snapshot,
                                arm_b_snapshot,  # type: ignore[arg-type]
                                allowed_pose_compensation_update_name,
                            ),
                        ]
                    )
                arm_c_decision = (
                    None
                    if arm_c_state is None
                    else _target_birth_admission_decision(
                        new_d_seg=float(arm_c_state["d_seg_batch"]),
                        new_d_pose=arm_c_state["d_pose_batch"],
                        pose_output_l2_delta=_pose_delta_l2(arm_c_state["pose"]),
                    )
                )
                delta_a = arm_rows[0]["exact_nonrate"].get("delta_score_nonrate")
                delta_b = arm_rows[1]["exact_nonrate"].get("delta_score_nonrate")
                delta_c = (
                    None
                    if arm_c_state is None or initial_nonrate is None or _nonrate_score(arm_c_state["d_seg_batch"], arm_c_state["d_pose_batch"]) is None
                    else float(_nonrate_score(arm_c_state["d_seg_batch"], arm_c_state["d_pose_batch"]) - initial_nonrate)  # type: ignore[operator]
                )
                comm_c = (
                    None
                    if not all(isinstance(value, (int, float)) for value in (delta_a, delta_b, delta_c))
                    else float(delta_c - float(delta_a) - float(delta_b))  # type: ignore[operator]
                )
                arm_rows.append(
                    _ablation_arm_row(
                        arm="C",
                        action_kind="independent_birth_plus_frame0_pose",
                        state=arm_c_state,
                        updated_names=arm_c_names,
                        admission_decision=arm_c_decision,
                        decision="measured" if arm_c_state is not None else "blocked",
                        blockers_for_arm=arm_c_blockers,
                        interaction_or_commutator=comm_c,
                    )
                )

                arm_d_state = accepted_composite_state
                arm_d_names = accepted_composite_updated_names
                arm_d_decision = accepted_composite_decision
                arm_d_blockers: list[str] = []
                if arm_d_state is None and accepted_birth_only_snapshot is not None and comp_loss_and_grad_fn is not None:
                    _restore_parameters(accepted_birth_only_snapshot)
                    d_record = _attempt_frame0_compensation(
                        accepted_birth_only_snapshot,
                        accepted_birth_only_state["uint8"] if accepted_birth_only_state is not None else final["uint8"],
                        initial_nonrate,
                    )
                    arm_d_state = d_record.get("composite_state")
                    arm_d_names = sorted(
                        [
                            *_snapshot_delta_names(
                                initial_snapshot,
                                accepted_birth_only_snapshot,
                                allowed_birth_update_name,
                            ),
                            *[str(name) for name in d_record.get("compensation_updated_parameter_names") or []],
                        ]
                    )
                    arm_d_decision = dict(d_record.get("admission_decision") or {})
                    if not d_record.get("accepted"):
                        arm_d_blockers.append("hinerv_four_arm_joint_line_search_not_admitted")
                elif arm_d_state is None:
                    arm_d_blockers.append("hinerv_four_arm_joint_line_search_not_measured")
                delta_d = (
                    None
                    if arm_d_state is None or initial_nonrate is None or _nonrate_score(arm_d_state["d_seg_batch"], arm_d_state["d_pose_batch"]) is None
                    else float(_nonrate_score(arm_d_state["d_seg_batch"], arm_d_state["d_pose_batch"]) - initial_nonrate)  # type: ignore[operator]
                )
                comm_d = (
                    None
                    if not all(isinstance(value, (int, float)) for value in (delta_a, delta_b, delta_d))
                    else float(delta_d - float(delta_a) - float(delta_b))  # type: ignore[operator]
                )
                arm_rows.append(
                    _ablation_arm_row(
                        arm="D",
                        action_kind="joint_line_search_composite",
                        state=arm_d_state,
                        updated_names=arm_d_names,
                        admission_decision=arm_d_decision,
                        decision=(
                            "accepted_with_frame0_pose_compensation"
                            if arm_d_state is not None and not arm_d_blockers
                            else "blocked"
                        ),
                        blockers_for_arm=arm_d_blockers,
                        interaction_or_commutator=comm_d,
                    )
                )
                four_arm_ablation_payload = {
                    "schema": "hi_nerv_target_region_birth_four_arm_ablation.v1",
                    "action_id": action_identity,
                    "authority": "batch_local_live_mlx",
                    "normalization_scope": "batch_local",
                    "producer": "hinerv_target_region_birth",
                    "arm_count": len(arm_rows),
                    "required_arms": ["A", "B", "C", "D"],
                    "measured_arms": [str(row.get("arm")) for row in arm_rows if not row.get("blockers")],
                    "blocked_arms": [
                        str(row.get("arm"))
                        for row in arm_rows
                        if row.get("blockers")
                    ],
                    "arms": arm_rows,
                    "commutators": [
                        {
                            "basis": "C_minus_A_minus_B",
                            "value": arm_rows[2].get("interaction_or_commutator"),
                        },
                        {
                            "basis": "D_minus_A_minus_B",
                            "value": arm_rows[3].get("interaction_or_commutator"),
                        },
                    ],
                    "restore_state_pass": True,
                    "promotion_eligible": False,
                    "human_visual_fidelity_objective": False,
                }
            finally:
                _restore_parameters(restore_target_snapshot)

        receipt = build_target_region_birth_receipt(
            debt=worst,
            before_margin_stats=before_stats,
            after_margin_stats=after_stats,
            receiver_uint8_changed_pixels_region=uint8_changed_total,
            receiver_uint8_delta_abs_max=uint8_delta_abs_max,
            receiver_float_rgb_delta_linf=float(final["float_rgb_delta_linf"]),
            argmax_flipped_pixels_region=argmax_flipped_region,
            accepted_step_count=accepted_step_count,
            rejected_step_count=rejected_step_count,
            blockers=blockers,
            grad_norm_by_group=last_grad_norm_by_group,
            update_norm_by_group=last_update_norm_by_group,
            updated_parameter_names=sorted(accepted_update_names),
            pose_guard=pose_guard_payload,
            runtime_sidecar_bytes=0,
            argmax_transitions=argmax_transitions,
            exact_nonrate=exact_nonrate_payload,
            candidate_frontier_telemetry=candidate_frontier_telemetry,
            pose_compensation=pose_compensation_payload,
            action_id=action_identity,
            surface="live_mlx",
        )
        receipt["trained_groups"] = sorted(trained_groups_for_action_id)
        receipt["worst_region"] = dict(worst_region_payload)
        archive_charged_decoder_tensors = [
            "latents_fine",
            "feature_grids.*",
            "fine_injector.*",
            "head_rgb_1.*",
        ]
        if pose_compensation_payload is not None and pose_compensation_payload["composite_accepted"]:
            archive_charged_decoder_tensors.append("head_rgb_0.*")
        payload = {
            "schema": "hi_nerv_target_region_birth.v1",
            "enabled": True,
            "action_id": action_identity,
            "accepted": bool(accepted_step_count > 0),
            "birth_class_index": birth_class,
            "target_region_forced_key": (
                None if forced_region_key is None else list(forced_region_key)
            ),
            "hard_region_miner_inputs": hard_region_miner_inputs,
            "worst_region": worst_region_payload,
            "before_region_margin_stats": dict(before_stats),
            "after_region_margin_stats": dict(after_stats),
            "before_unsolved_tail_margin_stats": dict(before_unsolved_tail_stats),
            "after_unsolved_tail_margin_stats": dict(after_unsolved_tail_stats),
            "unsolved_tail_pixel_count": int(unsolved_tail_pixel_count),
            "already_won_region_pixel_count": int(already_won_pixel_count),
            "before_region_hard_ratio": float(before_stats["region_hard_ratio"]),
            "after_region_hard_ratio": float(after_stats["region_hard_ratio"]),
            "before_unsolved_tail_hard_ratio": float(before_unsolved_tail_stats["region_hard_ratio"]),
            "after_unsolved_tail_hard_ratio": float(after_unsolved_tail_stats["region_hard_ratio"]),
            "target_geometry_mode": geometry_mode,
            "target_geometry_frontier_dilation": int(frontier_dilation),
            "initial_active_tail_pixel_count": int(initial_active_tail_geometry["active_tail_pixel_count"]),
            "final_active_tail_pixel_count": int(active_tail_geometry["active_tail_pixel_count"]),
            "active_tail_refresh_count": int(active_tail_refresh_count),
            "active_tail_geometry_history": active_tail_geometry_history,
            "target_min_region_ratio": float(target_min_region_ratio),
            "target_min_region_ratio_reached": bool(after_stats["region_hard_ratio"] >= float(target_min_region_ratio)),
            "accepted_step_count": int(accepted_step_count),
            "rejected_step_count": int(rejected_step_count),
            "subquantum_rejected_step_count": int(subquantum_rejected_step_count),
            "pose_guard_rejected_step_count": int(pose_guard_rejected_step_count),
            "joint_score_rejected_step_count": int(joint_score_rejected_step_count),
            "no_progress_rejected_step_count": int(no_progress_rejected_step_count),
            "fakequant_survival_required": bool(require_fakequant_survival),
            "birth_gradient_surface": birth_gradient_surface,
            "birth_gradient_surface_eval_count": int(birth_gradient_surface_eval_count),
            "fakequant_survival_bits": int(fakequant_bits),
            "fakequant_survival_rejected_step_count": int(
                fakequant_survival_rejected_step_count
            ),
            "candidate_frontier_telemetry": candidate_frontier_telemetry,
            "post_acceptance_terminal_events": list(post_acceptance_terminal_events),
            "argmax_transitions": argmax_transitions,
            "exact_nonrate": exact_nonrate_payload,
            # Frame0 composite-compensation summary (batch-local). Defaults are
            # the no-op no-pose-teacher path: attempted=False, frame=None.
            "pose_compensation_attempted": bool(pose_compensation_attempted),
            "pose_compensation_frame": (0 if pose_compensation_attempted else None),
            "composite_attempt_count": int(composite_attempt_count),
            "composite_accepted_count": int(composite_accepted_count),
            "composite_accepted": bool(composite_accepted_count > 0),
            "composite_delta_score_nonrate": last_composite_delta_score_nonrate,
            "compensation_updated_parameter_names": sorted(composite_updated_parameter_names),
            "pose_compensation": pose_compensation_payload,
            "four_arm_ablation": four_arm_ablation_payload,
            "receiver_quantum_growth_attempt_count": int(receiver_quantum_growth_attempt_count),
            "argmax_birth_quantum_growth_attempt_count": int(
                argmax_birth_quantum_growth_attempt_count
            ),
            "max_argmax_birth_quantum_growth_attempts_per_step": int(
                max_argmax_birth_quantum_growth_attempts
            ),
            "backtracking_attempt_count": int(backtracking_attempt_count),
            "loss_history_first": (loss_history[0] if loss_history else None),
            "loss_history_last": (loss_history[-1] if loss_history else None),
            "final_learning_rate": float(current_lr),
            "updated_parameter_names": sorted(accepted_update_names),
            "trained_groups": sorted(trained_groups_for_action_id),
            "grad_norm_by_group": dict(last_grad_norm_by_group),
            "update_norm_by_group": dict(last_update_norm_by_group),
            "parameter_group_sha256_before": parameter_group_sha256_before,
            "parameter_group_sha256_after": parameter_group_sha256_after,
            "out_of_scope_bit_frozen_verified": bool(out_of_scope_bit_frozen_verified),
            "pose_guard": pose_guard_payload,
            "receipt": receipt,
            "blockers": list(blockers),
            "archive_charged_decoder_tensors": archive_charged_decoder_tensors,
            "runtime_sidecar_bytes": 0,
            "receiver_surface": "clamp_round_uint8_rgb_ste_nhwc01",
            "target_surface": "segnet_last_frame_rgb_argmax_worst_connected_region",
            "human_visual_fidelity_objective": False,
        }
        portfolio_current_attempt = {
            "schema": "hi_nerv_target_region_birth_portfolio_attempt.v1",
            "attempt_index": len(portfolio_prior_attempts),
            "region_key": [int(worst.batch_index), int(worst.class_index), int(worst.region_label)],
            "class_index": int(worst.class_index),
            "region_pixel_count": int(worst.region_pixel_count),
            "region_unsolved_pixel_count": int(worst.region_unsolved_pixel_count),
            "score_debt_units": float(worst.score_debt_units),
            "accepted": bool(payload["accepted"]),
            "accepted_step_count": int(accepted_step_count),
            "rejected_step_count": int(rejected_step_count),
            "wrong_to_target_count": int(argmax_transitions["wrong_to_target_count"]),
            "target_to_wrong_count": int(argmax_transitions["target_to_wrong_count"]),
            "wrong_to_wrong_count": int(argmax_transitions["wrong_to_wrong_count"]),
            "net_target_support_delta": int(argmax_transitions["net_target_support_delta"]),
            "delta_score_nonrate": exact_nonrate_payload.get("delta_score_nonrate"),
            "blockers": list(blockers),
        }
        portfolio_attempts = [*portfolio_prior_attempts, portfolio_current_attempt]
        representative_region_coverage = _target_region_portfolio_coverage_row(
            portfolio_attempts
        )
        payload["target_region_portfolio"] = {
            "schema": "hi_nerv_target_region_birth_portfolio.v1",
            "max_regions": int(portfolio_max_regions),
            "forced_region_key": (
                None if forced_region_key is None else list(forced_region_key)
            ),
            "attempt_count": len(portfolio_attempts),
            "attempts": portfolio_attempts,
            "exhausted": bool(
                payload["accepted"] or len(portfolio_attempts) >= int(portfolio_max_regions)
            ),
            "selection_policy": (
                "forced_region_key_no_retry"
                if forced_region_key is not None
                else "score_debt_desc_then_next_positive_region_after_exact_admission_failure"
            ),
            "admission_authority": "exact_nonlinear_batch_local_seg_pose_nonrate",
            "representative_region_coverage": representative_region_coverage,
        }
        payload["representative_region_coverage"] = representative_region_coverage
        excluded_next = (
            *portfolio_excluded_keys,
            (int(worst.batch_index), int(worst.class_index), int(worst.region_label)),
        )
        retry_positive_regions = [
            row
            for row in find_target_region_debts(
                labels_np,
                initial_argmax_np,
                min_region_pixels=int(min_region_pixels),
            )
            if row.region_unsolved_pixel_count > 0
            and (row.batch_index, row.class_index, row.region_label) not in excluded_next
        ]
        remaining_positive_region_count = len(retry_positive_regions)
        portfolio_budget_exhausted = bool(
            not payload["accepted"]
            and len(portfolio_attempts) >= int(portfolio_max_regions)
            and remaining_positive_region_count > 0
        )
        portfolio_search_exhausted = bool(not payload["accepted"] and remaining_positive_region_count == 0)
        payload["target_region_portfolio"]["remaining_positive_region_count"] = remaining_positive_region_count
        payload["target_region_portfolio"]["budget_exhausted"] = portfolio_budget_exhausted
        payload["target_region_portfolio"]["search_exhausted"] = portfolio_search_exhausted
        payload["target_region_portfolio"]["exhausted"] = bool(
            payload["accepted"] or portfolio_search_exhausted
        )
        if portfolio_budget_exhausted:
            budget_blocker = (
                "hinerv_target_region_birth_portfolio_budget_exhausted_with_remaining_positive_regions"
            )
            if budget_blocker not in payload["blockers"]:
                payload["blockers"].append(budget_blocker)
            receipt_blockers = payload.get("receipt", {}).get("blockers")
            if isinstance(receipt_blockers, list) and budget_blocker not in receipt_blockers:
                receipt_blockers.append(budget_blocker)
            payload["target_region_portfolio"]["budget_blocker"] = budget_blocker
        if (
            not payload["accepted"]
            and forced_region_key is None
            and len(portfolio_attempts) < int(portfolio_max_regions)
            and retry_positive_regions
        ):
            excluded_next = (
                *portfolio_excluded_keys,
                (int(worst.batch_index), int(worst.class_index), int(worst.region_label)),
            )
            return self.fit_target_region_birth_from_segnet(
                scorer_teacher=scorer_teacher,
                target_rgb_0=target_rgb_0,
                target_rgb_1=target_rgb_1,
                pair_indices=pair_indices,
                target_segnet_argmax_1=target_segnet_argmax_1,
                pose_teacher=pose_teacher,
                require_pose_trust=require_pose_trust,
                max_steps=max_steps,
                learning_rate=learning_rate,
                target_min_region_ratio=target_min_region_ratio,
                margin_tau=margin_tau,
                margin_floor=margin_floor,
                prob_floor=prob_floor,
                lambda_prob_floor=lambda_prob_floor,
                lambda_seed=lambda_seed,
                lambda_support_preserve=lambda_support_preserve,
                lambda_outside_argmax_preserve=lambda_outside_argmax_preserve,
                lambda_already_won_hard_preserve=lambda_already_won_hard_preserve,
                lambda_already_won_rgb_preserve=lambda_already_won_rgb_preserve,
                already_won_margin_floor=already_won_margin_floor,
                lambda_pose_trust_preserve=lambda_pose_trust_preserve,
                lambda_pose_target=lambda_pose_target,
                seed_temperature=seed_temperature,
                min_margin_mean_drop=min_margin_mean_drop,
                max_pose_output_delta_l2=max_pose_output_delta_l2,
                exact_score_epsilon_batch_local=exact_score_epsilon_batch_local,
                catastrophic_pose_relative_cap=catastrophic_pose_relative_cap,
                catastrophic_pose_score_regression_cap=catastrophic_pose_score_regression_cap,
                catastrophic_pose_output_l2_hard_cap=catastrophic_pose_output_l2_hard_cap,
                grad_clip_max_norm=grad_clip_max_norm,
                min_region_pixels=min_region_pixels,
                target_geometry_mode=target_geometry_mode,
                target_geometry_frontier_dilation=target_geometry_frontier_dilation,
                emit_four_arm_ablation=emit_four_arm_ablation,
                four_arm_ablation_compensation_steps=four_arm_ablation_compensation_steps,
                target_region_portfolio_max_regions=target_region_portfolio_max_regions,
                target_region_forced_key=target_region_forced_key,
                require_fakequant_survival=require_fakequant_survival,
                fakequant_survival_bits=fakequant_survival_bits,
                _target_region_portfolio_excluded=excluded_next,
                _target_region_portfolio_attempts=tuple(portfolio_attempts),
            )
        return payload

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
        rgb_0 = (
            mx.sigmoid(
                _conv2d_with_params(
                    self.head_rgb_0,
                    h,
                    fake_quant_bits=fake_quant_bits,
                    fake_quant_bits_by_name=fake_quant_bits_by_name,
                    weight_name="head_rgb_0.weight",
                    bias_name="head_rgb_0.bias",
                )
            )
            * 255.0
        )  # type: ignore[union-attr]
        rgb_1 = (
            mx.sigmoid(
                _conv2d_with_params(
                    self.head_rgb_1,
                    h,
                    fake_quant_bits=fake_quant_bits,
                    fake_quant_bits_by_name=fake_quant_bits_by_name,
                    weight_name="head_rgb_1.weight",
                    bias_name="head_rgb_1.bias",
                )
            )
            * 255.0
        )  # type: ignore[union-attr]
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
            "latent_embed.weight": np.asarray(self.latent_embed.weight, dtype=np.float32).copy(),
            "latent_embed.bias": np.asarray(self.latent_embed.bias, dtype=np.float32).copy(),
        }
        for i, block in enumerate(self.blocks):
            conv = block.conv
            out[f"blocks.{i}.conv.weight"] = np.transpose(
                np.asarray(conv.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"blocks.{i}.conv.bias"] = np.asarray(conv.bias, dtype=np.float32).copy()
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
            out[f"{name}.weight"] = np.asarray(injector.proj.weight, dtype=np.float32).copy()
            out[f"{name}.bias"] = np.asarray(injector.proj.bias, dtype=np.float32).copy()
        for head_name in ("head_rgb_0", "head_rgb_1"):
            head = getattr(self, head_name)
            out[f"{head_name}.weight"] = np.transpose(np.asarray(head.weight, dtype=np.float32), (0, 3, 1, 2)).copy()
            out[f"{head_name}.bias"] = np.asarray(head.bias, dtype=np.float32).copy()
        return out


__all__ = [
    "MLX_EVIDENCE_GRADE",
    "SCHEMA_VERSION",
    "ConvNeXtBlockMLX",
    "HierarchicalFeatureGridMLX",
    "HinervSubstrateMLX",
    "trilinear_upsample_mlx",
]
