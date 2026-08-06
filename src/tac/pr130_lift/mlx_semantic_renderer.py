"""MLX port of PR130's semantic-token renderer training leg.

borrowed_substrate_accounting:
  source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
  source_head: 2f94596bb0136d342254022a5c9584756eae0468
  source_paths:
    - code/semantic_renderer_oracle.py sha256=2bf3a6a8621334723fec1c3e596665d1f049a55f311c5348dd5a4c588873f25b
    - code/train_semantic_full.py sha256=2d7a3575e422dc2b5823b97e52101ad5632e5fa04a98e0af8a83d85b7c2176b8
    - code/train_semantic_quantized.py sha256=4bcaf8a5c581c1e5eb057ea0ef760f269e1eabfdd2fca926bcbf61f4163a248d
    - code/evaluate_semantic_quantization.py sha256=5bbd2136174bfa2c99219d73f45103d4293f60e3f4eced5ee188e38053923962
  theirs: architecture, curriculum phases, uint8-STE exact-R shape, QAT math,
    cosine schedule, selected-boundary checkpoint policy.
  ours: MLX/Metal substrate adaptation, explicit torch-state import mapping,
    checkpoint/resume wrapper, and fail-closed local device reporting.

No score authority lives here.  Any d_seg verdict must still come from the
frozen scorer path named in the caller's receipt and, for promotion, from
``upstream/evaluate.py`` on exact archive bytes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
N_TOTAL_PAIRS = 600


class MlxUnavailableError(RuntimeError):
    """Raised when MLX cannot be imported or cannot execute on this host."""


@dataclass(frozen=True)
class MlxSemanticConfig:
    """PR130 semantic renderer configuration for the retained QAT12k -> tail6k leg."""

    width: int = 96
    blocks: int = 4
    frame_dim: int = 8
    num_pairs: int = N_TOTAL_PAIRS
    num_tokens: int = 5
    phase_y: int = 1
    phase_x: int = 1
    temporal_radius: int = 0
    bits: int = 4
    ce_fraction: float = 0.0
    softplus_fraction: float = -999.0
    lr: float = 2.0e-7
    steps: int = 6_000

    @classmethod
    def from_pr130_checkpoint_config(cls, payload: Mapping[str, Any]) -> "MlxSemanticConfig":
        """Build from PR130's saved ``checkpoint["config"]`` dictionary."""

        return cls(
            width=int(payload.get("width", cls.width)),
            blocks=int(payload.get("blocks", cls.blocks)),
            frame_dim=int(payload.get("frame_dim", cls.frame_dim)),
            phase_y=int(payload.get("phase_y", cls.phase_y)),
            phase_x=int(payload.get("phase_x", cls.phase_x)),
            temporal_radius=int(payload.get("temporal_radius", cls.temporal_radius)),
        )

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def require_mlx(*, device: str = "cpu") -> tuple[Any, Any, Any]:
    """Import MLX lazily and select the requested device before any layer build."""

    try:
        import mlx.core as mx

        target = str(device).lower()
        if target in {"cpu", "mlx-cpu"}:
            mx.set_default_device(mx.Device(mx.cpu, 0))
        elif target in {"gpu", "metal", "mlx-gpu"}:
            mx.set_default_device(mx.Device(mx.gpu, 0))
        else:
            raise ValueError(f"unknown MLX device {device!r}; use cpu or gpu")
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as exc:  # pragma: no cover - depends on host MLX runtime
        raise MlxUnavailableError(str(exc)) from exc
    return mx, nn, optim


def mlx_device_probe(*, device: str = "cpu") -> dict[str, Any]:
    """Return a machine-readable MLX availability probe without claiming parity."""

    try:
        mx, _nn, _optim = require_mlx(device=device)
        x = mx.array(np.asarray([1.0, 2.0], dtype=np.float32))
        y = mx.sum(x)
        mx.eval(y)
        return {
            "status": "available",
            "device_request": device,
            "sum_probe": float(y),
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "device_request": device,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _torch_conv_to_mlx(mx: Any, value: Any) -> Any:
    """PyTorch OIHW -> MLX OHWI."""

    arr = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"expected conv weight rank 4, got {arr.shape}")
    return mx.array(np.transpose(arr, (0, 2, 3, 1)))


def _torch_array(mx: Any, value: Any, *, dtype: Any | None = None) -> Any:
    arr = np.asarray(value.detach().cpu().numpy())
    if dtype is not None:
        arr = arr.astype(dtype)
    return mx.array(arr)


def build_mlx_semantic_renderer_class(mx: Any, nn: Any) -> Any:
    """Construct the MLX class after MLX imports succeed."""

    class MlxTokenBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, width: int, frame_dim: int, dilation: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(
                width, width, 3, padding=dilation, dilation=dilation, groups=width
            )
            self.pw = nn.Conv2d(width, width, 1)
            self.norm = nn.GroupNorm(max(1, width // 8), width, pytorch_compatible=True)
            self.film = nn.Linear(frame_dim, 2 * width)
            self.film.weight = mx.zeros_like(self.film.weight)
            self.film.bias = mx.zeros_like(self.film.bias)

        def __call__(self, x: Any, frame: Any) -> Any:
            y = self.norm(self.pw(self.dw(x)))
            film = self.film(frame)
            scale, shift = mx.split(film, 2, axis=1)
            y = y * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
            return x + nn.gelu(y)

    class MlxSemanticTokenRenderer(nn.Module):  # type: ignore[misc]
        def __init__(self, config: MlxSemanticConfig):
            super().__init__()
            self.config = config
            self.width = int(config.width)
            self.num_tokens = int(config.num_tokens)
            self.phase_y = int(config.phase_y)
            self.phase_x = int(config.phase_x)
            self.temporal_radius = int(config.temporal_radius)
            self.token_embed = nn.Embedding(config.num_tokens, config.width)
            self.frame_embed = nn.Embedding(config.num_pairs, config.frame_dim)
            phase_channels = (config.phase_y if config.phase_y > 1 else 0) + (
                config.phase_x if config.phase_x > 1 else 0
            )
            temporal_channels = 2 * config.temporal_radius * config.num_tokens
            self.coord_mix = nn.Conv2d(
                config.width + 4 + phase_channels + temporal_channels,
                config.width,
                1,
            )
            dilations = [
                1,
                1,
                *[min(2 ** (index - 1), 4) for index in range(2, config.blocks)],
            ]
            self.blocks = [
                MlxTokenBlock(config.width, config.frame_dim, dilation=dilations[index])
                for index in range(config.blocks)
            ]
            self.head = nn.Conv2d(config.width, 3, 3, padding=1)

        def coordinates(self, batch: int, h: int, w: int, dtype: Any) -> Any:
            yy = mx.linspace(-1.0, 1.0, h, dtype=dtype)
            xx = mx.linspace(-1.0, 1.0, w, dtype=dtype)
            grid_y = mx.broadcast_to(yy[None, :, None], (batch, h, w))
            grid_x = mx.broadcast_to(xx[None, None, :], (batch, h, w))
            channels = [grid_x, grid_y, grid_x * grid_x, grid_y * grid_y]
            if self.phase_y > 1:
                rows = mx.arange(h) % self.phase_y
                for phase in range(self.phase_y):
                    ch = mx.broadcast_to((rows == phase)[None, :, None], (batch, h, w))
                    channels.append(ch.astype(dtype))
            if self.phase_x > 1:
                cols = mx.arange(w) % self.phase_x
                for phase in range(self.phase_x):
                    ch = mx.broadcast_to((cols == phase)[None, None, :], (batch, h, w))
                    channels.append(ch.astype(dtype))
            return mx.stack(channels, axis=-1).astype(dtype)

        def __call__(self, tokens: Any, pair_idx: Any) -> Any:
            temporal = []
            if self.temporal_radius:
                expected = 2 * self.temporal_radius + 1
                if len(tokens.shape) != 4 or int(tokens.shape[1]) != expected:
                    raise ValueError(
                        f"temporal renderer expects [B,{expected},H,W] tokens"
                    )
                center = tokens[:, self.temporal_radius]
                for offset in range(expected):
                    if offset == self.temporal_radius:
                        continue
                    temporal.append(
                        mx.one_hot(tokens[:, offset].astype(mx.int32), self.num_tokens)
                    )
            else:
                if len(tokens.shape) != 3:
                    raise ValueError("renderer expects [B,H,W] tokens")
                center = tokens
            x = self.token_embed(center.astype(mx.int32))
            features = [x, self.coordinates(x.shape[0], x.shape[1], x.shape[2], x.dtype)]
            features.extend(item.astype(x.dtype) for item in temporal)
            x = self.coord_mix(mx.concatenate(features, axis=-1))
            frame = self.frame_embed(pair_idx.astype(mx.int32))
            for block in self.blocks:
                x = block(x, frame)
            return mx.sigmoid(self.head(nn.gelu(x))) * 255.0

    return MlxSemanticTokenRenderer


def make_mlx_renderer(config: MlxSemanticConfig, *, device: str = "cpu") -> Any:
    """Instantiate the PR130 renderer in MLX."""

    mx, nn, _optim = require_mlx(device=device)
    return build_mlx_semantic_renderer_class(mx, nn)(config)


def ste_uint8_mlx(mx: Any, x: Any) -> Any:
    clipped = mx.clip(x, 0.0, 255.0)
    return clipped + mx.stop_gradient(mx.round(clipped) - clipped)


def fake_quantize_mlx(mx: Any, value: Any, *, bits: int, embedding: bool) -> Any:
    """PR130 QAT fake quantization in MLX."""

    source = value.astype(mx.float32)
    if len(source.shape) < 2:
        rounded = source.astype(mx.float16).astype(mx.float32)
        return source + mx.stop_gradient(rounded - source)
    limit = (1 << (bits - 1)) - 1
    reduce_axes = tuple(range(len(source.shape) - 1)) if embedding else tuple(range(1, len(source.shape)))
    scale = mx.max(mx.abs(source), axis=reduce_axes, keepdims=True)
    scale = mx.maximum(scale, 1e-8) / limit
    scale = scale.astype(mx.float16).astype(mx.float32)
    normalized = mx.clip(source / scale, -limit, limit)
    codes = normalized + mx.stop_gradient(mx.round(normalized) - normalized)
    return codes * scale


def fake_quantize_parameter_tree(
    mx: Any,
    tree_flatten: Any,
    tree_unflatten: Any,
    params: Mapping[str, Any],
    *,
    bits: int,
) -> Mapping[str, Any]:
    """Apply PR130's name-aware QAT transform to an MLX parameter tree."""

    quantized = []
    for name, value in tree_flatten(params):
        if hasattr(value, "shape"):
            quantized.append(
                (
                    name,
                    fake_quantize_mlx(
                        mx,
                        value,
                        bits=bits,
                        embedding=str(name).endswith("embed.weight"),
                    ),
                )
            )
        else:
            quantized.append((name, value))
    return tree_unflatten(quantized)


def target_margin_mlx(mx: Any, logits_nchw: Any, target_hw: Any) -> Any:
    c = int(logits_nchw.shape[1])
    target = target_hw.astype(mx.int32)
    onehot = mx.stack(
        [(target == k).astype(logits_nchw.dtype) for k in range(c)], axis=1
    )
    target_logit = mx.sum(logits_nchw * onehot, axis=1, keepdims=True)
    other = logits_nchw + onehot * -1.0e9
    return target_logit - mx.max(other, axis=1, keepdims=True)


def curriculum_loss_mlx(
    mx: Any,
    logits_nchw: Any,
    target_hw: Any,
    *,
    step: int,
    total_steps: int,
    ce_fraction: float,
    softplus_fraction: float,
) -> tuple[Any, str]:
    """MLX twin of PR130's CE -> softplus-margin -> expected-flip loss."""

    progress = step / max(total_steps - 1, 1)
    target = target_hw.astype(mx.int32)
    if progress < ce_fraction:
        temp = 1.0 * (0.08 ** (progress / ce_fraction))
        log_probs = logits_nchw / temp - mx.logsumexp(
            logits_nchw / temp, axis=1, keepdims=True
        )
        c = int(logits_nchw.shape[1])
        onehot = mx.stack(
            [(target == k).astype(logits_nchw.dtype) for k in range(c)], axis=1
        )
        return -mx.mean(mx.sum(log_probs * onehot, axis=1)), "ce"
    margin = target_margin_mlx(mx, logits_nchw, target)
    if progress < softplus_fraction:
        tau = 0.20
        return mx.mean(mx.logaddexp(mx.zeros_like(margin), -margin / tau) * tau), "softplus_margin"
    tail = (progress - softplus_fraction) / max(1.0 - softplus_fraction, 1e-6)
    tau = 0.15 - 0.10 * tail
    return mx.mean(mx.sigmoid(-margin / tau)), "expected_flip"


def load_torch_state_dict_into_mlx(model: Any, state_dict: Mapping[str, Any], *, device: str = "cpu") -> None:
    """Load PR130 torch weights into the MLX renderer with explicit layout maps."""

    mx, _nn, _optim = require_mlx(device=device)
    model.token_embed.weight = _torch_array(mx, state_dict["token_embed.weight"], dtype=np.float32)
    model.frame_embed.weight = _torch_array(mx, state_dict["frame_embed.weight"], dtype=np.float32)
    model.coord_mix.weight = _torch_conv_to_mlx(mx, state_dict["coord_mix.weight"])
    model.coord_mix.bias = _torch_array(mx, state_dict["coord_mix.bias"], dtype=np.float32)
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}."
        block.dw.weight = _torch_conv_to_mlx(mx, state_dict[prefix + "dw.weight"])
        block.dw.bias = _torch_array(mx, state_dict[prefix + "dw.bias"], dtype=np.float32)
        block.pw.weight = _torch_conv_to_mlx(mx, state_dict[prefix + "pw.weight"])
        block.pw.bias = _torch_array(mx, state_dict[prefix + "pw.bias"], dtype=np.float32)
        block.norm.weight = _torch_array(mx, state_dict[prefix + "norm.weight"], dtype=np.float32)
        block.norm.bias = _torch_array(mx, state_dict[prefix + "norm.bias"], dtype=np.float32)
        block.film.weight = _torch_array(mx, state_dict[prefix + "film.weight"], dtype=np.float32)
        block.film.bias = _torch_array(mx, state_dict[prefix + "film.bias"], dtype=np.float32)
    model.head.weight = _torch_conv_to_mlx(mx, state_dict["head.weight"])
    model.head.bias = _torch_array(mx, state_dict["head.bias"], dtype=np.float32)


def save_stage_checkpoint_npz(
    path: str | Path,
    *,
    model: Any,
    config: MlxSemanticConfig,
    step: int,
    history: list[dict[str, Any]],
    optimizer_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a stage-encoded, byte-loadable MLX checkpoint atomically."""

    from mlx.utils import tree_flatten

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta::config_json": np.frombuffer(
            json.dumps(config.asdict(), sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        ),
        "meta::step": np.asarray([int(step)], dtype=np.int64),
        "meta::history_json": np.frombuffer(
            json.dumps(history, sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    if extra:
        payload["meta::extra_json"] = np.frombuffer(
            json.dumps(dict(extra), sort_keys=True, default=str).encode("utf-8"),
            dtype=np.uint8,
        )
    for name, array in tree_flatten(model.parameters()):
        payload[f"param::{name}"] = np.asarray(array)
    if optimizer_state is not None:
        for name, array in tree_flatten(optimizer_state):
            if hasattr(array, "shape"):
                payload[f"opt::{name}"] = np.asarray(array)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(out)
    return out


def load_stage_checkpoint_npz(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    mx: Any | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_stage_checkpoint_npz`."""

    if mx is None:
        mx, _nn, _optim = require_mlx(device="cpu")
    from mlx.utils import tree_unflatten

    payload = np.load(Path(path), allow_pickle=False)
    model_weights = [
        (key.removeprefix("param::"), mx.array(payload[key]))
        for key in payload.files
        if key.startswith("param::")
    ]
    if model_weights:
        model.load_weights(model_weights, strict=False)
    if optimizer is not None:
        opt_weights = [
            (key.removeprefix("opt::"), mx.array(payload[key]))
            for key in payload.files
            if key.startswith("opt::")
        ]
        if opt_weights:
            optimizer.state = tree_unflatten(opt_weights)
    history = json.loads(bytes(payload["meta::history_json"]).decode("utf-8"))
    config = json.loads(bytes(payload["meta::config_json"]).decode("utf-8"))
    extra = (
        json.loads(bytes(payload["meta::extra_json"]).decode("utf-8"))
        if "meta::extra_json" in payload.files
        else {}
    )
    return {
        "step": int(payload["meta::step"][0]),
        "config": config,
        "history": history,
        "extra": extra,
        "param_count": len(model_weights),
        "optimizer_state_count": len([k for k in payload.files if k.startswith("opt::")]),
    }
