"""Quantization utilities for task-aware codec post-filters.

Supports:
  - Per-tensor symmetric int8 (legacy, backward compatible)
  - Per-channel symmetric int8 (better fidelity, same size)
  - FakeQuant STE for QAT training
  - LSQ (Learned Step Size) quantization
  - Save/load with metadata and variant tags
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn

# ── Fake Quantization (for training) ─────────────────────────────────────


class FakeQuantSTE(torch.autograd.Function):
    """Straight-through estimator for symmetric per-tensor int8 quantization.

    Forward: round-to-nearest int8, scale back to float.
    Backward: pass gradient through unchanged (STE), zero where saturated.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if w.ndim >= 2:
                # Per-channel: scale per output channel (dim 0)
                # .contiguous() needed for MPS — parametrized weights can
                # have incompatible strides from nn.utils.parametrize
                w_c = w.contiguous()
                flat = w_c.detach().reshape(w_c.shape[0], -1)
                scale = flat.abs().amax(dim=1) / 127.0
                scale = scale.clamp(min=1e-10)
                scale_view = scale.reshape(-1, *([1] * (w_c.ndim - 1)))
                q = (w_c / scale_view).round().clamp(-128.0, 127.0)
                saturated = ((w_c / scale_view).abs() > 127.5).contiguous()
                ctx.save_for_backward(saturated)
                return (q * scale_view).contiguous()
            else:
                # Per-tensor for 1D (bias)
                scale = w.detach().abs().max() / 127.0
                if scale.item() < 1e-10:
                    ctx.save_for_backward(torch.zeros_like(w, dtype=torch.bool))
                    return w
                q = (w / scale).round().clamp(-128.0, 127.0)
                saturated = (w / scale).abs() > 127.5
                ctx.save_for_backward(saturated)
                return q * scale

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (saturated,) = ctx.saved_tensors
        return grad_out * (~saturated).to(grad_out.dtype)


def fake_quant(t: torch.Tensor) -> torch.Tensor:
    """Apply fake int8 quantization with STE."""
    return FakeQuantSTE.apply(t)


# ── Uint8 Activation STE ───────────────────────────────────────────────


class Uint8STE(torch.autograd.Function):
    """Straight-through estimator for exact uint8 round-trip with saturation-aware
    gradient blocking.

    Migrated from experiments/train_postfilter_uint8ste.py.
    Note: Addresses compliance audit gap — ensures training matches exact
    deployment uint8 round-trip behavior.

    Forward: x -> clamp(round(x), 0, 255) — exactly what inflate writes to raw frames
    Backward: identity inside [0, 255], zero outside (saturation-aware gradient blocking)

    Usage:
        y = model(x)
        y_uint8 = Uint8STE.apply(y)  # scorer now sees deployment-exact tensor
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        saturated = (x < 0.0) | (x > 255.0)
        ctx.save_for_backward(saturated)
        return x.detach().clamp(0.0, 255.0).round()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (saturated,) = ctx.saved_tensors
        return grad_out * (~saturated).to(grad_out.dtype)


def uint8_ste(x: torch.Tensor) -> torch.Tensor:
    """Apply uint8 round-trip STE. Convenience wrapper around Uint8STE.apply."""
    return Uint8STE.apply(x)


# ── LSQ (Learned Step Size Quantization) ────────────────────────────


class LSQScale(nn.Module):
    """Learned step size for symmetric int8 quantization.

    Instead of computing scale = abs(w).max() / 127 at each forward pass,
    LSQ makes the scale a trainable parameter. The gradient of the rounding
    operation flows through via STE, but the scale itself gets a proper
    gradient scaled by 1/sqrt(numel) for stability.

    Usage:
        lsq = LSQScale.from_tensor(weight)
        quantized_weight = lsq(weight)  # differentiable
    """

    def __init__(self, init_scale: float, numel: int):
        super().__init__()
        self.step = nn.Parameter(torch.tensor(init_scale))
        self.grad_scale = 1.0 / (numel * 127) ** 0.5

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> LSQScale:
        init = t.detach().abs().max().item() / 127.0
        return cls(max(init, 1e-8), t.numel())

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        step = self.step * self.grad_scale + self.step.detach() * (1 - self.grad_scale)
        q = (w / step).round().clamp(-128, 127)
        return w + (q * step - w).detach()  # STE: forward uses quantized, backward uses w


def apply_lsq(model: nn.Module) -> dict[str, LSQScale]:
    """Attach LSQ scales to all Conv2d layers via forward pre-hooks.

    Each Conv2d gets an LSQScale that quantizes its weights before every
    forward pass. The scale is a trainable parameter (add to optimizer
    with higher lr). Hooks ensure LSQ is applied during training.

    Returns a dict of LSQScale modules keyed by module name.
    """
    scales = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            lsq = LSQScale.from_tensor(module.weight)
            scales[name] = lsq

            # Pre-hook: save float weights, replace with LSQ-quantized
            def _pre_hook(mod, inputs, *, _lsq=lsq):
                mod._weight_float = mod.weight.data.clone()
                mod.weight.data = _lsq(mod.weight).data

            # Post-hook: restore float weights so optimizer sees originals
            def _post_hook(mod, inputs, output):
                if hasattr(mod, "_weight_float"):
                    mod.weight.data.copy_(mod._weight_float)
                    del mod._weight_float
                return output

            hooks.append(module.register_forward_pre_hook(_pre_hook))
            hooks.append(module.register_forward_hook(_post_hook))

    model._lsq_hooks = hooks
    return scales


# ── QAT Wrapper ─────────────────────────────────────────────────────


class QATPostFilter(nn.Module):
    """Wraps any PostFilter with FakeQuant STE on weights during training.

    Forward pass quantizes weights to int8 (simulated) before each conv.
    Backward pass uses straight-through estimator. Biases are NOT quantized
    (matching the fp32_bias deployment path).

    NOTE: This is an experimental wrapper, NOT used by the standard Trainer
    pipeline. Trainer uses quantize_state_dict() at eval time instead.
    The mid-forward weight replacement pattern is fragile — exceptions
    between replacement and restore leave the model in a corrupt state.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporarily replace conv weights with fake-quantized versions.
        # Wrapped in try/finally to restore weights on exception — without this,
        # the model is left in a corrupt state if the forward pass throws.
        originals = {}
        for name, module in self.base.named_modules():
            if isinstance(module, nn.Conv2d):
                originals[name] = module.weight
                module.weight = nn.Parameter(fake_quant(module.weight))

        try:
            out = self.base(x)
        finally:
            # Restore originals (so optimizer sees real weights)
            for name, module in self.base.named_modules():
                if isinstance(module, nn.Conv2d) and name in originals:
                    module.weight = originals[name]

        return out


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    per_channel: bool = False,
) -> dict[str, torch.Tensor]:
    """Simulate int8 quantization on a state dict (for eval or checkpoint selection).

    Returns a new state dict with quantize-then-dequantize applied to all
    floating-point tensors. This is what the model will look like after
    save_int8 + load_int8 round-trip.
    """
    result = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            result[name] = tensor.clone()
            continue
        p = tensor.detach().float()
        if per_channel and p.ndim >= 2:
            C = p.shape[0]
            flat = p.reshape(C, -1)
            scales = flat.abs().max(dim=1).values / 127.0
            scales = torch.where(scales < 1e-10, torch.ones_like(scales), scales)
            scales_bc = scales.view(C, *([1] * (p.ndim - 1)))
            q = (p / scales_bc).round().clamp(-128, 127)
            result[name] = q * scales_bc
        else:
            scale = p.abs().max() / 127.0
            if scale.item() < 1e-10:
                result[name] = p.clone()
            else:
                q = (p / scale).round().clamp(-128, 127)
                result[name] = q * scale
    return result


# ── Save / Load ──────────────────────────────────────────────────────────


def save_int8(
    model: nn.Module,
    path: str | os.PathLike,
    *,
    meta: dict[str, Any] | None = None,
    per_channel: bool = True,
    fp32_bias: bool = False,
) -> int:
    """Save model weights in int8 quantized format.

    Args:
        model: the post-filter module
        path: output .pt file path
        meta: optional metadata dict (variant, hidden, kernel, alpha, ...)
        per_channel: use per-channel scales (better fidelity, ~same size).
            Default True to match FakeQuantSTE which always uses per-channel.
        fp32_bias: keep bias tensors in fp32 (avoids quantization on small tensors)

    Returns:
        File size in bytes.
    """
    state: dict[str, Any] = {}

    for name, param in model.state_dict().items():
        p = param.detach().cpu().float()

        # Optionally keep biases in full precision
        if fp32_bias and name.endswith(".bias"):
            state[name] = p
            continue

        if per_channel and p.ndim >= 2:
            # Per-channel: one scale per output channel
            C = p.shape[0]
            flat = p.reshape(C, -1)
            scales = flat.abs().max(dim=1).values / 127.0
            scales = torch.where(scales < 1e-10, torch.ones_like(scales), scales)
            scales_bc = scales.view(C, *([1] * (p.ndim - 1)))
            quantized = (p / scales_bc).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scales
        else:
            # Per-tensor: one global scale
            scale = p.abs().max() / 127.0
            if scale.item() < 1e-10:
                scale = torch.tensor(1.0)
            quantized = (p / scale).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scale

    if meta is not None:
        state["__meta__"] = dict(meta)

    torch.save(state, path)
    return os.path.getsize(path)


def load_int8(
    path: str | os.PathLike,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """Load int8-quantized weights into a model.

    Supports per-tensor (scalar scale), per-channel (vector scale),
    and fp32 bias formats. Backward compatible with all legacy formats.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    float_state: dict[str, torch.Tensor] = {}
    seen: set[str] = set()

    for raw_key in state:
        if raw_key == "__meta__" or raw_key == "__entropy_info__":
            continue
        if raw_key.endswith((".q", ".s", ".log_scale")):
            base = raw_key.rsplit(".", 1)[0]
            if base in seen:
                continue
            seen.add(base)
            q = state[base + ".q"].float()
            s = state[base + ".s"]
            is_log = (base + ".log_scale") in state and state[base + ".log_scale"].item()

            if is_log:
                # Reverse log-scale mapping from entropy_optimized_quantize
                log127 = math.log(127.0)
                q_norm = q / 127.0
                q_abs = q_norm.abs() * log127
                p_norm = torch.sign(q) * torch.expm1(q_abs) / 126.0
                if s.ndim == 0:
                    float_state[base] = p_norm * s
                else:
                    shape = [s.shape[0]] + [1] * (q.ndim - 1)
                    float_state[base] = p_norm * s.view(*shape)
            else:
                if s.ndim == 0:
                    float_state[base] = q * s
                else:
                    shape = [s.shape[0]] + [1] * (q.ndim - 1)
                    float_state[base] = q * s.view(*shape)
        else:
            float_state[raw_key] = state[raw_key].float()
            seen.add(raw_key)

    model.load_state_dict(float_state)
    return model.eval().to(device)




def save_int8_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    path: str | os.PathLike,
    meta: dict[str, Any] | None = None,
) -> int:
    """Save int8 from a bare state dict (no model needed).

    Useful for checkpoint averaging workflows that produce a state dict
    without a model instance.
    """
    state: dict[str, Any] = {}
    for name, param in state_dict.items():
        p = param.detach().cpu().float()
        scale = p.abs().max() / 127.0
        if scale.item() < 1e-10:
            scale = torch.tensor(1.0)
        state[name + ".q"] = (p / scale).round().clamp(-128, 127).to(torch.int8)
        state[name + ".s"] = scale
    if meta is not None:
        state["__meta__"] = dict(meta)
    torch.save(state, path)
    return os.path.getsize(path)


def prune_and_compress_int8(
    state_dict: dict[str, torch.Tensor],
    prune_ratio: float = 0.3,
    per_channel: bool = True,
) -> dict[str, Any]:
    """Prune smallest weights, then quantize to int8 with entropy-friendly encoding.

    Archive size optimization trick: pruning sets small weights to zero,
    which compresses extremely well under ZIP/zlib entropy coding (the
    archive.zip format used by the contest). Combined with int8 quantization,
    this can reduce model size by 30-50% vs vanilla int8.

    Pipeline:
        1. Global magnitude pruning: zero out the smallest `prune_ratio` fraction
           of all weight values (biases excluded).
        2. Per-channel int8 quantization on the pruned weights.
        3. Return the quantized state dict (caller saves with torch.save or save_int8).

    The zeros in the pruned weights become exact 0 int8 values, which have
    very high entropy coding efficiency (long runs of zeros compress to bits).

    Args:
        state_dict: float state dict (e.g., from model.state_dict() or EMA)
        prune_ratio: fraction of weight values to zero (default 0.3 = 30%)
        per_channel: use per-channel int8 scales (default True)

    Returns:
        Packed state dict ready for torch.save (same format as save_int8).
    """
    # Step 1: Collect all weight magnitudes for global threshold
    all_magnitudes = []
    weight_keys = []
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            continue
        if name.endswith(".bias"):
            continue  # don't prune biases
        all_magnitudes.append(tensor.detach().abs().reshape(-1))
        weight_keys.append(name)

    if not all_magnitudes:
        # No prunable weights, fall back to plain quantization
        return _pack_int8(state_dict, per_channel)

    all_mag = torch.cat(all_magnitudes)
    threshold = torch.quantile(all_mag.float(), prune_ratio).item()

    # Step 2: Prune + quantize
    pruned_sd: dict[str, torch.Tensor] = {}
    total_zeros = 0
    total_params = 0
    for name, tensor in state_dict.items():
        p = tensor.detach().cpu().float()
        if name in weight_keys:
            mask = p.abs() > threshold
            p = p * mask.float()
            total_zeros += (~mask).sum().item()
            total_params += mask.numel()
        pruned_sd[name] = p

    # Step 3: Pack as int8
    packed = _pack_int8(pruned_sd, per_channel)
    sparsity = total_zeros / max(total_params, 1) * 100
    packed["__prune_info__"] = {
        "prune_ratio": prune_ratio,
        "threshold": threshold,
        "sparsity_pct": round(sparsity, 1),
    }
    return packed


def _pack_int8(
    state_dict: dict[str, torch.Tensor],
    per_channel: bool = True,
) -> dict[str, Any]:
    """Pack a float state dict into int8 format (same as save_int8 internals)."""
    state: dict[str, Any] = {}
    for name, param in state_dict.items():
        p = param.detach().cpu().float()
        if not torch.is_floating_point(param):
            state[name] = param.clone()
            continue
        if name.endswith(".bias"):
            state[name] = p
            continue
        if per_channel and p.ndim >= 2:
            C = p.shape[0]
            flat = p.reshape(C, -1)
            scales = flat.abs().max(dim=1).values / 127.0
            scales = torch.where(scales < 1e-10, torch.ones_like(scales), scales)
            scales_bc = scales.view(C, *([1] * (p.ndim - 1)))
            quantized = (p / scales_bc).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scales
        else:
            scale = p.abs().max() / 127.0
            if scale.item() < 1e-10:
                scale = torch.tensor(1.0)
            quantized = (p / scale).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scale
    return state


def save_pruned_int8(
    model: nn.Module,
    path: str | os.PathLike,
    *,
    prune_ratio: float = 0.3,
    meta: dict[str, Any] | None = None,
) -> int:
    """Prune + int8 quantize + save. Returns file size in bytes."""
    packed = prune_and_compress_int8(model.state_dict(), prune_ratio=prune_ratio)
    if meta is not None:
        packed["__meta__"] = dict(meta)
    torch.save(packed, path)
    return os.path.getsize(path)


def get_meta(path: str | os.PathLike) -> dict[str, Any]:
    """Read metadata from an int8 weight file without loading all weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    return dict(state.get("__meta__", {}))


# ── Technique 11: SPZ Entropy-Reducing Quantization (Niantic) ─────────


def _layer_sensitivity(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Estimate per-layer sensitivity to quantization noise.

    For each weight tensor, measures the ratio of weight magnitude to
    weight range. Layers with higher sensitivity need more bits.

    Returns dict mapping weight key to sensitivity score in [0, 1].
    """
    sensitivities = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor) or name.endswith(".bias"):
            continue
        p = tensor.detach().float()
        if p.numel() < 2:
            sensitivities[name] = 1.0
            continue
        # Sensitivity = coefficient of variation (higher = more sensitive)
        std = p.std().item()
        mean_abs = p.abs().mean().item()
        if mean_abs < 1e-10:
            sensitivities[name] = 0.0
        else:
            sensitivities[name] = min(std / mean_abs, 1.0)
    return sensitivities


def entropy_optimized_quantize(
    state_dict: dict[str, torch.Tensor],
    dead_zone_ratio: float = 0.1,
    per_channel: bool = True,
) -> dict[str, Any]:
    """Technique 11: SPZ entropy-reducing quantization.

    Two-stage process inspired by Niantic's SPZ codec:
    1. Dead zone: zero out weights below a small threshold (increases zeros,
       reduces entropy under ZIP compression)
    2. Log-scale quantization: use non-uniform quantization bins that are
       denser near zero (where most weights live) and sparser at extremes.
       This reduces the entropy of the quantized representation.
    3. Different effective bit widths per layer based on sensitivity analysis.

    The result is packed as int8 but with entropy-friendly weight distribution
    that compresses much better under ZIP (the archive format).

    Args:
        state_dict: float state dict
        dead_zone_ratio: fraction of weight range to zero (dead zone around 0)
        per_channel: use per-channel scales

    Returns:
        Packed state dict ready for torch.save (same format as save_int8).
    """
    result: dict[str, Any] = {}
    total_zeros = 0
    total_params = 0

    for name, tensor in state_dict.items():
        p = tensor.detach().cpu().float()

        if not torch.is_floating_point(tensor):
            result[name] = tensor.clone()
            continue

        # Keep biases in fp32
        if name.endswith(".bias"):
            result[name] = p
            continue

        # Stage 1: Dead zone — zero out small weights
        abs_max = p.abs().max().item()
        if abs_max < 1e-10:
            result[name] = p
            continue

        dead_zone_threshold = abs_max * dead_zone_ratio
        dead_mask = p.abs() < dead_zone_threshold
        p = p * (~dead_mask).float()

        total_zeros += dead_mask.sum().item()
        total_params += p.numel()

        # Stage 2: Log-scale quantization
        # Map weights through sign-preserving log scale for non-uniform bins
        # log1p(|w|/scale) concentrates bins near zero
        if per_channel and p.ndim >= 2:
            C = p.shape[0]
            flat = p.reshape(C, -1)
            scales = flat.abs().max(dim=1).values
            scales = torch.where(scales < 1e-10, torch.ones_like(scales), scales)
            scales_bc = scales.view(C, *([1] * (p.ndim - 1)))

            # Normalize to [-1, 1]
            p_norm = p / scales_bc
            # Log-scale mapping: sign * log1p(|x| * 126) / log(127)
            # This gives denser quantization bins near zero
            log127 = math.log(127.0)
            p_log = torch.sign(p_norm) * torch.log1p(p_norm.abs() * 126.0) / log127
            # Quantize the log-scale representation
            q = (p_log * 127.0).round().clamp(-128, 127).to(torch.int8)

            result[name + ".q"] = q
            result[name + ".s"] = scales
            result[name + ".log_scale"] = torch.tensor(True)  # flag for decoder
        else:
            scale = p.abs().max() / 127.0
            if scale.item() < 1e-10:
                scale = torch.tensor(1.0)
            # Standard uniform quantization for small tensors
            q = (p / scale).round().clamp(-128, 127).to(torch.int8)
            result[name + ".q"] = q
            result[name + ".s"] = scale

    sparsity = total_zeros / max(total_params, 1) * 100
    result["__entropy_info__"] = {
        "dead_zone_ratio": dead_zone_ratio,
        "sparsity_pct": round(sparsity, 1),
        "per_channel": per_channel,
        "method": "spz_entropy_optimized",
    }
    return result


def save_entropy_optimized_int8(
    model: nn.Module,
    path: str | os.PathLike,
    *,
    dead_zone_ratio: float = 0.1,
    meta: dict[str, Any] | None = None,
) -> int:
    """Technique 11: Entropy-optimized int8 save.

    Applies dead-zone + log-scale quantization for better ZIP compression.

    Args:
        model: the post-filter module
        path: output .pt file path
        dead_zone_ratio: fraction of weight range to zero
        meta: optional metadata

    Returns:
        File size in bytes.
    """
    packed = entropy_optimized_quantize(model.state_dict(), dead_zone_ratio=dead_zone_ratio)
    if meta is not None:
        packed["__meta__"] = dict(meta)
    torch.save(packed, path)
    return os.path.getsize(path)


def load_entropy_optimized_int8(
    path: str | os.PathLike,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """Load entropy-optimized int8 weights (with log-scale support).

    Thin wrapper around load_int8, which already handles the `.log_scale`
    flag stored by `entropy_optimized_quantize`. Kept for backward
    compatibility and API clarity.
    """
    return load_int8(path, model, device=device)


# ── Production PostFilter Loader ───────────────────────────────────────


DEFAULT_POSTFILTER_META: dict[str, int | str] = {
    "variant": "residual",
    "hidden": 16,
    "kernel": 3,
}


def normalize_postfilter_meta(meta: object | None) -> dict[str, int | str]:
    """Normalize checkpoint metadata to canonical form.

    Handles missing or partial metadata from legacy checkpoints.
    Returns a dict with at least 'variant', 'hidden', 'kernel'.
    """
    normalized = dict(DEFAULT_POSTFILTER_META)
    if isinstance(meta, dict):
        if "variant" in meta:
            normalized["variant"] = str(meta["variant"])
        if "hidden" in meta:
            normalized["hidden"] = int(meta["hidden"])
        if "kernel" in meta:
            normalized["kernel"] = int(meta["kernel"])
    return normalized


# ── Trick 23: Scorer-Aware uint8 Quantization Noise Shaping ─────────────


def noise_shaped_round(
    x: torch.Tensor,
    scorer_gradient: torch.Tensor,
    diffuse_error: bool = True,
) -> torch.Tensor:
    """Round float pixels to uint8 biased by scorer gradient direction.

    Instead of uniform rounding (round-to-nearest), evaluates both floor
    and ceil for each sub-pixel and picks whichever moves the scorer loss
    downward. The scorer gradient tells us d(loss)/d(pixel): if negative,
    increasing the pixel reduces loss (choose ceil); if positive,
    decreasing the pixel reduces loss (choose floor).

    This is 1-bit optimization per pixel per channel — zero model cost,
    pure numerical precision gain. The steganalysis analogy: LSB embedding
    with a distortion function defined by the scorer, not by HVS.

    Optional Floyd-Steinberg error diffusion propagates the rounding error
    to neighboring pixels, shaping quantization noise into the scorer's
    insensitive directions.

    Args:
        x: (B, C, H, W) or (B, H, W, C) float tensor in [0, 255].
            The pre-rounded frame pixels (postfilter output).
        scorer_gradient: same shape as x. Per-element d(loss)/d(pixel).
            Obtained by backpropagating scorer loss through the frame.
            Sign is what matters: negative = increasing pixel helps.
        diffuse_error: if True, apply Floyd-Steinberg error diffusion
            to spread quantization noise spatially. Slower but reduces
            structured artifacts.

    Returns:
        uint8-valued float tensor (same shape as x), clamped to [0, 255].
        Values are exact integers suitable for direct uint8 cast.
    """
    # Ensure inputs match
    assert x.shape == scorer_gradient.shape, (
        f"Shape mismatch: x={x.shape}, gradient={scorer_gradient.shape}"
    )

    x_clamped = x.detach().clamp(0.0, 255.0)

    # Gradient-directed rounding: negative gradient means "increase pixel helps"
    # so we ceil; positive gradient means "decrease pixel helps" so we floor.
    x_floor = x_clamped.floor()
    x_ceil = (x_clamped + 1.0).floor().clamp(max=255.0)  # safe ceil

    # Where gradient is negative, ceil is better (reduces loss by increasing pixel)
    # Where gradient is positive, floor is better (reduces loss by decreasing pixel)
    # Where gradient is zero, use standard round-to-nearest
    use_ceil = scorer_gradient < 0
    use_floor = scorer_gradient > 0
    # Default to nearest for zero gradient
    nearest = x_clamped.round().clamp(0.0, 255.0)

    result = torch.where(use_ceil, x_ceil, torch.where(use_floor, x_floor, nearest))

    if not diffuse_error:
        return result.clamp(0.0, 255.0)

    # Floyd-Steinberg error diffusion variant
    # Process each image in the batch sequentially (diffusion is spatial)
    output = result.clone()
    quant_error = x_clamped - output  # signed rounding error

    # FS kernel: right=7/16, below-left=3/16, below=5/16, below-right=1/16
    # We diffuse the error to neighbors, then re-quantize with gradient bias
    is_hwc = x.ndim == 4 and x.shape[-1] <= 4 and x.shape[1] > 4
    if is_hwc:
        # (B, H, W, C) layout
        B, H, W, C = x.shape
        for b in range(B):
            err = quant_error[b]  # (H, W, C)
            out = output[b]
            grad = scorer_gradient[b]
            for i in range(H):
                for j in range(W):
                    if j + 1 < W:
                        correction = err[i, j] * (7.0 / 16.0)
                        pixel = out[i, j + 1] + correction
                        g = grad[i, j + 1]
                        out[i, j + 1] = torch.where(g < 0, pixel.ceil(), torch.where(g > 0, pixel.floor(), pixel.round())).clamp(0, 255)
                        err[i, j + 1] = pixel - out[i, j + 1]
                    if i + 1 < H:
                        if j > 0:
                            out[i + 1, j - 1] = (out[i + 1, j - 1] + err[i, j] * (3.0 / 16.0)).clamp(0, 255)
                        out[i + 1, j] = (out[i + 1, j] + err[i, j] * (5.0 / 16.0)).clamp(0, 255)
                        if j + 1 < W:
                            out[i + 1, j + 1] = (out[i + 1, j + 1] + err[i, j] * (1.0 / 16.0)).clamp(0, 255)
            output[b] = out.round().clamp(0, 255)
    else:
        # (B, C, H, W) layout — process per channel
        B, C, H, W = x.shape
        for b in range(B):
            for c in range(C):
                err = quant_error[b, c]  # (H, W)
                out = output[b, c]
                grad = scorer_gradient[b, c]
                for i in range(H):
                    for j in range(W):
                        if j + 1 < W:
                            correction = err[i, j] * (7.0 / 16.0)
                            pixel = out[i, j + 1] + correction
                            g = grad[i, j + 1]
                            out[i, j + 1] = torch.where(g < 0, pixel.ceil(), torch.where(g > 0, pixel.floor(), pixel.round())).clamp(0, 255)
                            err[i, j + 1] = pixel - out[i, j + 1]
                        if i + 1 < H:
                            if j > 0:
                                out[i + 1, j - 1] = (out[i + 1, j - 1] + err[i, j] * (3.0 / 16.0)).clamp(0, 255)
                            out[i + 1, j] = (out[i + 1, j] + err[i, j] * (5.0 / 16.0)).clamp(0, 255)
                            if j + 1 < W:
                                out[i + 1, j + 1] = (out[i + 1, j + 1] + err[i, j] * (1.0 / 16.0)).clamp(0, 255)
                output[b, c] = out.round().clamp(0, 255)

    return output


def noise_shaped_round_fast(
    x: torch.Tensor,
    scorer_gradient: torch.Tensor,
) -> torch.Tensor:
    """Fast gradient-directed rounding without error diffusion.

    Vectorized version of noise_shaped_round with diffuse_error=False.
    Use this in the inner training loop where speed matters. The full
    Floyd-Steinberg variant is for final frame export only.

    Args:
        x: (B, C, H, W) float tensor in [0, 255].
        scorer_gradient: same shape as x, d(loss)/d(pixel).

    Returns:
        uint8-valued float tensor, clamped to [0, 255].
    """
    x_clamped = x.detach().clamp(0.0, 255.0)
    # Gradient sign determines rounding direction
    # negative gradient -> ceil (increase pixel helps)
    # positive gradient -> floor (decrease pixel helps)
    # zero gradient -> nearest
    directed = torch.where(
        scorer_gradient < 0,
        x_clamped.ceil(),
        torch.where(scorer_gradient > 0, x_clamped.floor(), x_clamped.round()),
    )
    return directed.clamp(0.0, 255.0)


def load_postfilter_int8(
    path: str | os.PathLike,
    device: str = "cpu",
) -> nn.Module:
    """Load int8-quantized post-filter weights with auto-detected architecture.

    This is the production loader used by inflate_postfilter.py. Unlike
    load_int8() which requires a pre-built model, this function reads the
    checkpoint metadata (``__meta__``) to determine the architecture variant,
    builds the model, and loads weights.

    Supports three on-disk formats (backward compatible):
      * ``key.q`` int8 + scalar ``key.s`` -> legacy per-tensor symmetric
      * ``key.q`` int8 + vector ``key.s`` (shape [C]) -> per-channel symmetric
        broadcasted across the first weight dimension
      * ``key`` float tensor (no .q/.s suffix) -> uncompressed fp32 fallback,
        used when biases are stored in full precision to keep a tiny tensor
        from losing fidelity for the sake of a few bytes.

    Auto-detection heuristic: if metadata claims a simple variant
    (standard/residual) but the checkpoint actually has 4 conv layers with
    12-channel input (PixelUnshuffle), override to pixelshuffle_dilated.
    This handles legacy checkpoints saved before __meta__ was updated to
    reflect architecture changes. Limitation: cannot distinguish pixelshuffle
    from pixelshuffle_dilated without checking dilation.

    Args:
        path: path to the int8 checkpoint (.pt file)
        device: target device ("cpu", "cuda", "mps")

    Returns:
        The post-filter model in eval mode on the requested device.

    Raises:
        ValueError: if weight keys don't match the detected architecture.
    """
    from .architectures import build_postfilter

    state = torch.load(path, map_location=device, weights_only=True)
    float_state: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for raw_key in state.keys():
        if raw_key == "__meta__" or raw_key == "__entropy_info__":
            continue
        if raw_key.endswith((".q", ".s", ".log_scale")):
            base = raw_key.rsplit(".", 1)[0]
            if base in seen:
                continue
            seen.add(base)
            q = state[base + ".q"].float()
            s = state[base + ".s"]
            is_log = (base + ".log_scale") in state and state[base + ".log_scale"].item()

            if is_log:
                # Reverse log-scale mapping from entropy_optimized_quantize
                log127 = math.log(127.0)
                q_norm = q / 127.0
                q_abs = q_norm.abs() * log127
                p_norm = torch.sign(q) * torch.expm1(q_abs) / 126.0
                if s.ndim == 0:
                    float_state[base] = p_norm * s
                else:
                    shape = [s.shape[0]] + [1] * (q.ndim - 1)
                    float_state[base] = p_norm * s.view(*shape)
            else:
                if s.ndim == 0:
                    float_state[base] = q * s
                else:
                    # per-channel: reshape s to (C, 1, 1, ...) matching q's rank
                    shape = [s.shape[0]] + [1] * (q.ndim - 1)
                    float_state[base] = q * s.view(*shape)
        else:
            # uncompressed fp32 tensor (e.g., bias stored in full precision)
            # Skip non-tensor metadata (dicts like __meta__, config_fingerprint)
            val = state[raw_key]
            if isinstance(val, torch.Tensor):
                float_state[raw_key] = val.float()
            seen.add(raw_key)
    meta = normalize_postfilter_meta(state.get("__meta__"))
    # Auto-detection heuristic: if metadata claims a simple variant (standard/residual)
    # but the checkpoint actually has 4 conv layers with 12-channel input (PixelUnshuffle),
    # override to pixelshuffle_dilated. This handles legacy checkpoints saved before
    # __meta__ was updated to reflect architecture changes. Limitation: cannot
    # distinguish pixelshuffle from pixelshuffle_dilated without checking dilation.
    if (
        meta.get("variant") in {"standard", "residual", "saliency_weighted", "segaware"}
        and "conv4.weight" in float_state
        and "conv1.weight" in float_state
        and float_state["conv1.weight"].ndim == 4
        and int(float_state["conv1.weight"].shape[1]) == 12
    ):
        meta["variant"] = "pixelshuffle_dilated"
    model = build_postfilter(
        variant=str(meta["variant"]),
        hidden=int(meta["hidden"]),
        kernel=int(meta["kernel"]),
    )
    # Cross-check: weight keys must match exactly. A mismatch here means the
    # architecture in tac.architectures has diverged from what the checkpoint expects.
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(float_state.keys())
    if model_keys != ckpt_keys:
        raise ValueError(
            f"Weight key mismatch between model and checkpoint. "
            f"Missing in ckpt: {model_keys - ckpt_keys}, "
            f"Extra in ckpt: {ckpt_keys - model_keys}. "
            f"Meta: {meta}"
        )
    model.load_state_dict(float_state)
    return model.eval().to(device)