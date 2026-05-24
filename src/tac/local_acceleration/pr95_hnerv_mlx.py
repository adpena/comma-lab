# SPDX-License-Identifier: MIT
"""Native MLX reproduction primitives for the public PR95 HNeRV lane.

This module is deliberately narrow: it ports the PR95 decoder topology and
stage-8 optimizer partition into MLX so local Apple Silicon timing and parity
work can become queueable evidence.  It does not claim score authority; archive
smokes emitted from this lane remain byte-closed training artifacts until a
runtime consumes them and exact CPU/CUDA auth eval anchors them.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised in environments with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    tree_unflatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


LANE_ID = "lane_pr95_hnerv_mlx_reproduction"
SMOKE_MANIFEST_SCHEMA = "pr95_hnerv_mlx_timing_smoke_manifest_v1"
SMOKE_ARCHIVE_SCHEMA = "pr95_hnerv_mlx_byte_closed_smoke_archive_v1"

PR95_STAGE_MODULES: dict[int, str] = {
    1: "stage1_v328_ce",
    5: "stage5_c1a_l7",
    8: "stage8_muon_finetune",
}

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "dispatch_packet_ready": False,
}

EXACT_READINESS_REFUSAL_BLOCKERS: tuple[str, ...] = (
    "pr95_hnerv_mlx_timing_smoke_is_local_training_signal_not_score",
    "synthetic_targets_do_not_establish_contest_quality",
    "byte_closed_smoke_archive_not_consumed_by_pr95_runtime",
    "runtime_consumption_proof_missing",
    "receiver_proof_missing",
    "requires_pytorch_export_forward_parity_on_source_checkpoint",
    "requires_byte_closed_contest_archive_export",
    "requires_exact_cpu_cuda_auth_eval_before_score_claim",
)


class Pr95HNeRVMlxError(RuntimeError):
    """Raised when the PR95 MLX lane cannot execute faithfully."""


def require_mlx() -> None:
    """Fail clearly when imported on a machine without MLX."""

    if mx is None or nn is None or tree_flatten is None or tree_unflatten is None:
        raise Pr95HNeRVMlxError(
            "MLX is required for PR95/HNeRV local reproduction"
        ) from _MLX_IMPORT_ERROR


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _param_count_from_tree(tree: Any) -> int:
    require_mlx()
    count = 0
    for _name, value in tree_flatten(tree):  # type: ignore[misc]
        shape = getattr(value, "shape", ())
        if shape:
            count += math.prod(int(dim) for dim in shape)
    return count


def _mlx_array_from_any(value: Any) -> Any:
    require_mlx()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return mx.array(np.asarray(value, dtype=np.float32))  # type: ignore[union-attr]


def _torch_conv_to_mlx(value: Any) -> Any:
    arr = np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value,
        dtype=np.float32,
    )
    if arr.ndim != 4:
        raise ValueError(f"expected torch conv weight with 4 dims, got {arr.shape}")
    return _mlx_array_from_any(np.transpose(arr, (0, 2, 3, 1)))


def _mlx_conv_to_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 4:
        raise ValueError(f"expected MLX conv weight with 4 dims, got {arr.shape}")
    return np.transpose(arr, (0, 3, 1, 2))


def pixel_shuffle_2x_nhwc(x: Any, *, upscale_factor: int = 2) -> Any:
    """PixelShuffle for NHWC tensors using native MLX reshape/transpose ops."""

    require_mlx()
    if upscale_factor != 2:
        raise ValueError("PR95 uses only 2x pixel shuffle")
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC tensor, got shape {x.shape}")
    batch, height, width, channels = (int(dim) for dim in x.shape)
    block = upscale_factor * upscale_factor
    if channels % block:
        raise ValueError(
            f"channels must be divisible by {block} for 2x pixel shuffle; got {channels}"
        )
    out_channels = channels // block
    y = mx.reshape(  # type: ignore[union-attr]
        x,
        (batch, height, width, upscale_factor, upscale_factor, out_channels),
    )
    y = mx.transpose(y, (0, 1, 3, 2, 4, 5))  # type: ignore[union-attr]
    return mx.reshape(  # type: ignore[union-attr]
        y,
        (batch, height * upscale_factor, width * upscale_factor, out_channels),
    )


def bilinear_resize2x_align_corners_false_nhwc(x: Any) -> Any:
    """2x bilinear resize for NHWC tensors matching PyTorch align_corners=False."""

    require_mlx()
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC tensor, got shape {x.shape}")

    left = mx.concatenate([x[:, :, :1, :], x[:, :, :-1, :]], axis=2)  # type: ignore[union-attr]
    right = mx.concatenate([x[:, :, 1:, :], x[:, :, -1:, :]], axis=2)  # type: ignore[union-attr]
    even_w = x * 0.75 + left * 0.25
    odd_w = x * 0.75 + right * 0.25
    width_up = mx.reshape(  # type: ignore[union-attr]
        mx.stack([even_w, odd_w], axis=3),  # type: ignore[union-attr]
        (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]) * 2, int(x.shape[3])),
    )

    top = mx.concatenate([width_up[:, :1, :, :], width_up[:, :-1, :, :]], axis=1)  # type: ignore[union-attr]
    bottom = mx.concatenate([width_up[:, 1:, :, :], width_up[:, -1:, :, :]], axis=1)  # type: ignore[union-attr]
    even_h = width_up * 0.75 + top * 0.25
    odd_h = width_up * 0.75 + bottom * 0.25
    return mx.reshape(  # type: ignore[union-attr]
        mx.stack([even_h, odd_h], axis=2),  # type: ignore[union-attr]
        (
            int(width_up.shape[0]),
            int(width_up.shape[1]) * 2,
            int(width_up.shape[2]),
            int(width_up.shape[3]),
        ),
    )


class _HNeRVUpsampleBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_channels: int, out_channels: int) -> None:
        require_mlx()
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)  # type: ignore[union-attr]
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1)  # type: ignore[union-attr]
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: Any) -> Any:
        identity = bilinear_resize2x_align_corners_false_nhwc(x)
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        decoded = pixel_shuffle_2x_nhwc(self.conv(x))
        return mx.sin(decoded + identity)  # type: ignore[union-attr]


class HNeRVDecoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """PR95 HNeRV decoder in native MLX/NHWC form.

    The public PR95 PyTorch model returns ``(B, 2, 3, 384, 512)``.  This MLX
    port keeps MLX's NHWC kernels internally and returns that same N2CHW layout
    by default for export/parity tests.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 28,
        base_channels: int = 36,
        eval_size: tuple[int, int] = (384, 512),
        output_layout: str = "n2chw",
    ) -> None:
        require_mlx()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.eval_size = tuple(int(dim) for dim in eval_size)
        self.base_h = 6
        self.base_w = 8
        self.output_layout = output_layout
        if self.eval_size != (self.base_h * 64, self.base_w * 64):
            raise ValueError(
                "PR95 decoder topology fixes eval_size to "
                f"{(self.base_h * 64, self.base_w * 64)}, got {self.eval_size}"
            )
        if output_layout not in {"n2chw", "n2hwc"}:
            raise ValueError("output_layout must be 'n2chw' or 'n2hwc'")
        channels = [
            self.base_channels,
            self.base_channels,
            self.base_channels,
            int(self.base_channels * 0.75),
            int(self.base_channels * 0.58),
            int(self.base_channels * 0.5),
            int(self.base_channels * 0.5),
        ]
        if min(channels) < 1:
            raise ValueError("base_channels too small for PR95 channel taper")
        self.channels = channels

        self.stem = nn.Linear(  # type: ignore[union-attr]
            self.latent_dim,
            channels[0] * self.base_h * self.base_w,
        )
        self.blocks = [
            _HNeRVUpsampleBlockMLX(channels[i], channels[i + 1]) for i in range(6)
        ]
        final_ch = channels[-1]
        self.refine0 = nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2)  # type: ignore[union-attr]
        self.refine1 = nn.Conv2d(final_ch // 2, final_ch, 3, padding=1)  # type: ignore[union-attr]
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)  # type: ignore[union-attr]
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)  # type: ignore[union-attr]

    def features_nhwc(self, z: Any) -> Any:
        batch = int(z.shape[0])
        x = self.stem(z)
        x = mx.reshape(  # type: ignore[union-attr]
            x,
            (batch, self.channels[0], self.base_h, self.base_w),
        )
        x = mx.transpose(x, (0, 2, 3, 1))  # type: ignore[union-attr]
        x = mx.sin(x)  # type: ignore[union-attr]
        for block in self.blocks:
            x = block(x)
        refined = self.refine1(self.refine0(x))
        return x + 0.1 * mx.sin(refined)  # type: ignore[union-attr]

    def decode_pair_nhwc(self, z: Any) -> Any:
        x = self.features_nhwc(z)
        f0 = mx.sigmoid(self.rgb_0(x)) * 255.0  # type: ignore[union-attr]
        f1 = mx.sigmoid(self.rgb_1(x)) * 255.0  # type: ignore[union-attr]
        return mx.stack([f0, f1], axis=1)  # type: ignore[union-attr]

    def __call__(self, z: Any) -> Any:
        pair = self.decode_pair_nhwc(z)
        if self.output_layout == "n2hwc":
            return pair
        return mx.transpose(pair, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "schema": "pr95_hnerv_mlx_architecture_v1",
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "eval_size": list(self.eval_size),
            "base_grid": [self.base_h, self.base_w],
            "channels": list(self.channels),
            "upsample_blocks": 6,
            "internal_layout": "NHWC",
            "default_output_layout": self.output_layout,
            "decoder_param_count": _param_count_from_tree(self.parameters()),
            "source_pr": 95,
            "source_architecture": "submissions/hnerv_muon/src/model.py::HNeRVDecoder",
        }


class HNeRVSyntheticTrainingBundleMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Decoder plus trainable per-pair latents for timing-only MLX smokes."""

    def __init__(
        self,
        *,
        latent_count: int,
        latent_dim: int = 28,
        base_channels: int = 36,
        seed: int = 0,
        output_layout: str = "n2chw",
    ) -> None:
        require_mlx()
        super().__init__()
        key = mx.random.key(seed)  # type: ignore[union-attr]
        self.latents = mx.random.normal((latent_count, latent_dim), key=key) * 0.1  # type: ignore[union-attr]
        self.decoder = HNeRVDecoderMLX(
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_layout=output_layout,
        )

    def __call__(self, indices: Any) -> Any:
        return self.decoder(mx.take(self.latents, indices, axis=0))  # type: ignore[union-attr]


def load_pytorch_state_dict_into_mlx(
    model: HNeRVDecoderMLX,
    state_dict: dict[str, Any],
) -> None:
    """Load a PR95 PyTorch ``state_dict`` into an MLX decoder."""

    require_mlx()
    flat = dict(tree_flatten(model.parameters()))  # type: ignore[misc]

    def set_param(path: str, value: Any) -> None:
        if path not in flat:
            raise KeyError(f"MLX model has no parameter path {path!r}")
        flat[path] = value

    set_param("stem.weight", _mlx_array_from_any(state_dict["stem.weight"]))
    set_param("stem.bias", _mlx_array_from_any(state_dict["stem.bias"]))
    for index in range(6):
        set_param(
            f"blocks.{index}.conv.weight",
            _torch_conv_to_mlx(state_dict[f"blocks.{index}.weight"]),
        )
        set_param(
            f"blocks.{index}.conv.bias",
            _mlx_array_from_any(state_dict[f"blocks.{index}.bias"]),
        )
        skip_weight = f"skips.{index}.weight"
        if skip_weight in state_dict:
            set_param(
                f"blocks.{index}.skip_conv.weight",
                _torch_conv_to_mlx(state_dict[skip_weight]),
            )
            set_param(
                f"blocks.{index}.skip_conv.bias",
                _mlx_array_from_any(state_dict[f"skips.{index}.bias"]),
            )

    set_param("refine0.weight", _torch_conv_to_mlx(state_dict["refine.0.weight"]))
    set_param("refine0.bias", _mlx_array_from_any(state_dict["refine.0.bias"]))
    set_param("refine1.weight", _torch_conv_to_mlx(state_dict["refine.1.weight"]))
    set_param("refine1.bias", _mlx_array_from_any(state_dict["refine.1.bias"]))
    for head in ("rgb_0", "rgb_1"):
        set_param(f"{head}.weight", _torch_conv_to_mlx(state_dict[f"{head}.weight"]))
        set_param(f"{head}.bias", _mlx_array_from_any(state_dict[f"{head}.bias"]))

    model.update(tree_unflatten(list(flat.items())))  # type: ignore[misc]


def pytorch_state_dict_from_mlx(
    model: HNeRVDecoderMLX,
    *,
    as_torch: bool = False,
) -> dict[str, Any]:
    """Export MLX decoder parameters using the public PR95 PyTorch names."""

    require_mlx()
    flat = dict(tree_flatten(model.parameters()))  # type: ignore[misc]
    exported: dict[str, np.ndarray] = {
        "stem.weight": np.asarray(flat["stem.weight"]),
        "stem.bias": np.asarray(flat["stem.bias"]),
    }
    for index in range(6):
        exported[f"blocks.{index}.weight"] = _mlx_conv_to_numpy(
            flat[f"blocks.{index}.conv.weight"]
        )
        exported[f"blocks.{index}.bias"] = np.asarray(flat[f"blocks.{index}.conv.bias"])
        skip_weight = f"blocks.{index}.skip_conv.weight"
        if skip_weight in flat:
            exported[f"skips.{index}.weight"] = _mlx_conv_to_numpy(flat[skip_weight])
            exported[f"skips.{index}.bias"] = np.asarray(
                flat[f"blocks.{index}.skip_conv.bias"]
            )
    exported["refine.0.weight"] = _mlx_conv_to_numpy(flat["refine0.weight"])
    exported["refine.0.bias"] = np.asarray(flat["refine0.bias"])
    exported["refine.1.weight"] = _mlx_conv_to_numpy(flat["refine1.weight"])
    exported["refine.1.bias"] = np.asarray(flat["refine1.bias"])
    for head in ("rgb_0", "rgb_1"):
        exported[f"{head}.weight"] = _mlx_conv_to_numpy(flat[f"{head}.weight"])
        exported[f"{head}.bias"] = np.asarray(flat[f"{head}.bias"])
    if not as_torch:
        return exported
    import torch

    return {name: torch.from_numpy(value.copy()) for name, value in exported.items()}


def partition_pr95_mlx_parameter_names(params: Any) -> dict[str, list[str]]:
    """Return the source-faithful PR95 stage-8 Muon/AdamW parameter split."""

    require_mlx()
    muon: list[str] = []
    adamw: list[str] = []
    for name, value in tree_flatten(params):  # type: ignore[misc]
        low = name.lower()
        ndim = len(getattr(value, "shape", ()))
        if (
            ndim >= 2
            and name.endswith("weight")
            and "stem" not in low
            and "rgb_" not in low
            and "latents" not in low
        ):
            muon.append(name)
        else:
            adamw.append(name)
    return {"muon": sorted(muon), "adamw": sorted(adamw)}


def zeropower_via_newtonschulz5_mlx(
    gradient: Any,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    cast_float32_to_bfloat16: bool = True,
) -> Any:
    """PR95/Keller-Jordan Newton-Schulz orthogonalization in MLX."""

    require_mlx()
    if len(gradient.shape) != 2:
        raise ValueError(f"Newton-Schulz expects 2D input, got {gradient.shape}")
    original_dtype = gradient.dtype
    x = gradient.astype(mx.bfloat16) if cast_float32_to_bfloat16 else gradient  # type: ignore[union-attr]
    transposed = int(x.shape[-2]) > int(x.shape[-1])
    if transposed:
        x = x.T
    x = x / (mx.linalg.norm(x, keepdims=True) + eps)  # type: ignore[union-attr]
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.astype(original_dtype)


@dataclass(frozen=True)
class Pr95MlxOptimizerConfig:
    use_muon: bool
    adamw_lr: float = 3e-5
    latent_lr_mult: float = 10.0
    muon_lr: float = 2e-4
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    grad_clip_muon: float | None = 1.0
    cast_muon_float32_to_bfloat16: bool = True


@dataclass
class Pr95MlxOptimizerState:
    step: int = 0
    muon_buffers: dict[str, Any] = field(default_factory=dict)
    adamw_m: dict[str, Any] = field(default_factory=dict)
    adamw_v: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pr95MlxStageSmokeConfig:
    stage_index: int
    stage_module: str
    optimizer: Pr95MlxOptimizerConfig
    synthetic_loss: str = "normalized_rgb_pair_mse"


def stage_smoke_config(stage_index: int) -> Pr95MlxStageSmokeConfig:
    """Return a faithful optimizer switch for PR95 stages 1, 5, and 8."""

    if stage_index not in PR95_STAGE_MODULES:
        raise ValueError("supported PR95 MLX timing stages are 1, 5, and 8")
    if stage_index == 8:
        optimizer = Pr95MlxOptimizerConfig(
            use_muon=True,
            adamw_lr=1e-5,
            muon_lr=2e-4,
            muon_weight_decay=5e-4,
            latent_lr_mult=10.0,
            grad_clip=1.0,
            grad_clip_muon=1.0,
        )
    else:
        optimizer = Pr95MlxOptimizerConfig(
            use_muon=False,
            adamw_lr=3e-5,
            latent_lr_mult=10.0,
            grad_clip=1.0,
            grad_clip_muon=None,
        )
    return Pr95MlxStageSmokeConfig(
        stage_index=stage_index,
        stage_module=PR95_STAGE_MODULES[stage_index],
        optimizer=optimizer,
    )


def _clip_flat_gradients(
    gradients: dict[str, Any],
    names: list[str],
    *,
    max_norm: float | None,
) -> None:
    require_mlx()
    if max_norm is None or max_norm <= 0 or not names:
        return
    norm_sq = None
    for name in names:
        grad = gradients.get(name)
        if grad is None:
            continue
        term = mx.sum(grad * grad)  # type: ignore[union-attr]
        norm_sq = term if norm_sq is None else norm_sq + term
    if norm_sq is None:
        return
    norm = mx.sqrt(norm_sq)  # type: ignore[union-attr]
    scale = mx.minimum(mx.array(1.0), mx.array(float(max_norm)) / (norm + 1e-6))  # type: ignore[union-attr]
    for name in names:
        if name in gradients and gradients[name] is not None:
            gradients[name] = gradients[name] * scale


def apply_pr95_mlx_optimizer_step(
    module: Any,
    gradients: Any,
    state: Pr95MlxOptimizerState,
    config: Pr95MlxOptimizerConfig,
) -> dict[str, Any]:
    """Apply one source-faithful PR95 optimizer step to an MLX module."""

    require_mlx()
    params_flat = dict(tree_flatten(module.parameters()))  # type: ignore[misc]
    grads_flat = dict(tree_flatten(gradients))  # type: ignore[misc]
    split = partition_pr95_mlx_parameter_names(module.parameters())
    muon_names = split["muon"] if config.use_muon else []
    adamw_names = list(split["adamw"] + ([] if config.use_muon else split["muon"]))
    _clip_flat_gradients(grads_flat, adamw_names, max_norm=config.grad_clip)
    _clip_flat_gradients(grads_flat, muon_names, max_norm=config.grad_clip_muon)

    state.step += 1
    beta1, beta2 = config.adamw_betas
    updated: dict[str, Any] = {}
    for name, param in params_flat.items():
        grad = grads_flat.get(name)
        if grad is None:
            updated[name] = param
            continue
        if name in muon_names:
            base = (
                param * (1.0 - config.muon_lr * config.muon_weight_decay)
                if config.muon_weight_decay
                else param
            )
            buf = state.muon_buffers.get(name)
            if buf is None:
                buf = mx.zeros_like(grad)  # type: ignore[union-attr]
            buf = buf * config.muon_momentum + grad
            state.muon_buffers[name] = buf
            update = (
                grad + buf * config.muon_momentum
                if config.muon_nesterov
                else buf
            )
            original_shape = update.shape
            if len(update.shape) == 4:
                rows = int(update.shape[0])
                cols = math.prod(int(dim) for dim in update.shape[1:])
                update_2d = mx.reshape(update, (rows, cols))  # type: ignore[union-attr]
                update_2d = zeropower_via_newtonschulz5_mlx(
                    update_2d,
                    steps=config.muon_ns_steps,
                    cast_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
                )
                scale = max(1.0, math.sqrt(rows / cols))
                update = mx.reshape(update_2d * scale, original_shape)  # type: ignore[union-attr]
            elif len(update.shape) == 2:
                rows = int(update.shape[0])
                cols = int(update.shape[1])
                update = zeropower_via_newtonschulz5_mlx(
                    update,
                    steps=config.muon_ns_steps,
                    cast_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
                )
                update = update * max(1.0, math.sqrt(rows / cols))
            updated[name] = base - config.muon_lr * update
            continue

        lr = (
            config.adamw_lr * config.latent_lr_mult
            if "latents" in name.lower()
            else config.adamw_lr
        )
        base = (
            param * (1.0 - lr * config.adamw_weight_decay)
            if config.adamw_weight_decay
            else param
        )
        m = state.adamw_m.get(name)
        v = state.adamw_v.get(name)
        if m is None:
            m = mx.zeros_like(grad)  # type: ignore[union-attr]
        if v is None:
            v = mx.zeros_like(grad)  # type: ignore[union-attr]
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        state.adamw_m[name] = m
        state.adamw_v[name] = v
        bias_corrected_lr = lr * math.sqrt(1.0 - beta2**state.step) / (
            1.0 - beta1**state.step
        )
        updated[name] = base - bias_corrected_lr * m / (mx.sqrt(v) + config.adamw_eps)  # type: ignore[union-attr]

    module.update(tree_unflatten(list(updated.items())))  # type: ignore[misc]
    return {
        "schema": "pr95_hnerv_mlx_optimizer_step_summary_v1",
        "step": state.step,
        "use_muon": config.use_muon,
        "muon_tensor_count": len(muon_names),
        "adamw_tensor_count": len(adamw_names),
        "muon_parameter_names": muon_names,
        "adamw_parameter_names": sorted(adamw_names),
    }


def run_pr95_mlx_synthetic_timing_smoke(
    *,
    stage_index: int,
    steps: int,
    batch_size: int,
    synthetic_pairs: int,
    seed: int,
    base_channels: int = 36,
    latent_dim: int = 28,
) -> dict[str, Any]:
    """Run a local MLX timing smoke against synthetic targets."""

    require_mlx()
    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if synthetic_pairs < batch_size:
        raise ValueError("synthetic_pairs must be >= batch_size")

    stage = stage_smoke_config(stage_index)
    mx.random.seed(seed)  # type: ignore[union-attr]
    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=synthetic_pairs,
        latent_dim=latent_dim,
        base_channels=base_channels,
        seed=seed,
        output_layout="n2chw",
    )
    target_key = mx.random.key(seed + 1)  # type: ignore[union-attr]
    target = mx.random.uniform(  # type: ignore[union-attr]
        0,
        255,
        shape=(synthetic_pairs, 2, 3, 384, 512),
        key=target_key,
    )
    optimizer_state = Pr95MlxOptimizerState()

    def loss_fn(model: Any, indices: Any) -> Any:
        pred = model(indices)
        selected = mx.take(target, indices, axis=0)  # type: ignore[union-attr]
        residual = (pred - selected) / 255.0
        return mx.mean(residual * residual)  # type: ignore[union-attr]

    loss_and_grad = nn.value_and_grad(bundle, loss_fn)  # type: ignore[union-attr]
    last_loss = None
    step_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(steps):
        start = (step * batch_size) % synthetic_pairs
        raw_indices = [(start + offset) % synthetic_pairs for offset in range(batch_size)]
        indices = mx.array(raw_indices, dtype=mx.uint32)  # type: ignore[union-attr]
        loss, grads = loss_and_grad(bundle, indices)
        step_summary = apply_pr95_mlx_optimizer_step(
            bundle,
            grads,
            optimizer_state,
            stage.optimizer,
        )
        mx.eval(loss, bundle.parameters())  # type: ignore[union-attr]
        last_loss = float(loss)
        step_summaries.append(step_summary)
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / steps

    split = partition_pr95_mlx_parameter_names(bundle.parameters())
    profile_id = f"pr95_hnerv_mlx_stage{stage_index}_seed{seed}_steps{steps}"
    runtime_profile = {
        "schema": "trainer_runtime_profile_observation.v1",
        "profile_id": profile_id,
        "candidate_id": profile_id,
        "lane_id": LANE_ID,
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "training_backend": "mlx",
        "device": "mlx",
        "hardware_substrate": f"{platform.system()}_{platform.machine()}_mlx",
        "seed": seed,
        "stage_id": stage.stage_module,
        "stage_index": stage.stage_index,
        "seconds_per_step": seconds_per_step,
        "examples_per_second": batch_size / seconds_per_step if seconds_per_step else None,
        "state_bytes": _param_count_from_tree(bundle.parameters()) * 4,
        "kernel_fusion_strategy_id": "native_mlx_pr95_hnerv_decoder_muon_adamw_v1",
        "operator_mix": {
            "conv2d": 15,
            "pixel_shuffle_2x": 6,
            "bilinear_resize2x": 6,
            "linear": 1,
            "newton_schulz5": len(split["muon"]) if stage.optimizer.use_muon else 0,
        },
        "packet_compiler_bridge": {
            "packet_compiler_target_declared": True,
            "archive_export_schema": SMOKE_ARCHIVE_SCHEMA,
            "archive_export_tool": "tools/run_pr95_mlx_timing_smoke.py",
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_present": False,
            "blockers": [
                "runtime_consumption_proof_missing",
                "byte_closed_contest_archive_export_missing",
            ],
        },
        "local_cloud_substitution": {
            "intended_to_replace_cloud_gpu_training": True,
            "cloud_gpu_reference": "PR95 HNeRV Muon stage timing and reproduction lane",
        },
        **FALSE_AUTHORITY,
    }
    manifest = {
        "schema": SMOKE_MANIFEST_SCHEMA,
        "candidate_id": profile_id,
        "lane_id": LANE_ID,
        "generated_utc": datetime.now(UTC).isoformat(),
        "stage_index": stage.stage_index,
        "stage_module": stage.stage_module,
        "steps": steps,
        "batch_size": batch_size,
        "synthetic_pairs": synthetic_pairs,
        "seed": seed,
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "training_backend": "mlx",
        "evidence_grade": "[macOS-MLX research-signal]",
        "timing": {
            "elapsed_seconds": elapsed,
            "seconds_per_step": seconds_per_step,
            "examples_per_second": runtime_profile["examples_per_second"],
        },
        "last_loss": last_loss,
        "architecture": bundle.decoder.architecture_manifest(),
        "optimizer_recipe": {
            "schema": "pr95_hnerv_mlx_optimizer_recipe_v1",
            "stage_uses_muon": stage.optimizer.use_muon,
            "muon_lr": stage.optimizer.muon_lr if stage.optimizer.use_muon else None,
            "adamw_lr": stage.optimizer.adamw_lr,
            "latent_lr_mult": stage.optimizer.latent_lr_mult,
            "muon_weight_decay": (
                stage.optimizer.muon_weight_decay if stage.optimizer.use_muon else None
            ),
            "muon_partition": split["muon"],
            "adamw_partition": split["adamw"],
            "source_stage8_partition_contract": (
                "Muon for hidden 2D+ non-stem/non-rgb weights; AdamW for latents, "
                "stem, RGB heads, biases, and 1D/scalar parameters."
            ),
        },
        "runtime_profile": runtime_profile,
        "step_summaries": step_summaries[-3:],
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": list(EXACT_READINESS_REFUSAL_BLOCKERS),
        },
        "pytorch_export_parity": {
            "schema": "pr95_hnerv_pytorch_export_parity_status_v1",
            "state_dict_names_match_public_pr95": True,
            "conv_layout_conversion": "MLX O,H,W,I <-> PyTorch O,I,H,W",
            "forward_parity_on_source_checkpoint": False,
            "blocker": "requires source checkpoint load and paired PyTorch/MLX forward smoke",
        },
        **FALSE_AUTHORITY,
    }
    return manifest


def write_pr95_mlx_byte_closed_smoke_archive(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Write a deterministic byte-closed smoke archive for queue plumbing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SMOKE_ARCHIVE_SCHEMA,
        "candidate_id": manifest["candidate_id"],
        "lane_id": manifest["lane_id"],
        "generated_utc": manifest["generated_utc"],
        "stage_index": manifest["stage_index"],
        "stage_module": manifest["stage_module"],
        "runtime_profile": manifest["runtime_profile"],
        "architecture": manifest["architecture"],
        "exact_readiness_refusal": manifest["exact_readiness_refusal"],
        **FALSE_AUTHORITY,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload_path = output_dir / "0.bin"
    payload_path.write_bytes(payload_bytes)
    archive_path = output_dir / "archive.zip"
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.extra = b""
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.comment = b""
        zf.writestr(info, payload_bytes)
    return {
        "schema": "byte_closed_archive_smoke_summary.v1",
        "path": archive_path.as_posix(),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "member": "0.bin",
        "member_bytes": len(payload_bytes),
        "member_sha256": _sha256_bytes(payload_bytes),
        "compression_method": "stored",
        "runtime_consumption_proof_present": False,
        "receiver_proof_present": False,
        **FALSE_AUTHORITY,
    }


__all__ = [
    "EXACT_READINESS_REFUSAL_BLOCKERS",
    "FALSE_AUTHORITY",
    "LANE_ID",
    "PR95_STAGE_MODULES",
    "SMOKE_ARCHIVE_SCHEMA",
    "SMOKE_MANIFEST_SCHEMA",
    "HNeRVDecoderMLX",
    "HNeRVSyntheticTrainingBundleMLX",
    "Pr95HNeRVMlxError",
    "Pr95MlxOptimizerConfig",
    "Pr95MlxOptimizerState",
    "apply_pr95_mlx_optimizer_step",
    "bilinear_resize2x_align_corners_false_nhwc",
    "load_pytorch_state_dict_into_mlx",
    "partition_pr95_mlx_parameter_names",
    "pixel_shuffle_2x_nhwc",
    "pytorch_state_dict_from_mlx",
    "require_mlx",
    "run_pr95_mlx_synthetic_timing_smoke",
    "stage_smoke_config",
    "write_pr95_mlx_byte_closed_smoke_archive",
    "zeropower_via_newtonschulz5_mlx",
]
