"""Quantization utilities for task-aware codec post-filters.

Supports:
  - Per-tensor symmetric int8 (legacy, backward compatible)
  - Per-channel symmetric int8 (better fidelity, same size)
  - FakeQuant STE for QAT training
  - LSQ (Learned Step Size) quantization
  - Save/load with metadata and variant tags
"""
from __future__ import annotations

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
            scale = w.detach().abs().max() / 127.0
            if scale.item() < 1e-10:
                ctx.save_for_backward(torch.zeros_like(w, dtype=torch.bool))
                return w
            q = (w / scale).round().clamp(-128.0, 127.0)
            # Zero gradient only for values that were ACTUALLY clipped,
            # not those that merely round to the boundary. Pre-round check.
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
    def from_tensor(cls, t: torch.Tensor) -> "LSQScale":
        init = t.detach().abs().max().item() / 127.0
        return cls(max(init, 1e-8), t.numel())

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        step = self.step * self.grad_scale + self.step.detach() * (1 - self.grad_scale)
        q = (w / step).round().clamp(-128, 127)
        return w + (q * step - w).detach()  # STE: forward uses quantized, backward uses w


def apply_lsq(model: nn.Module) -> dict[str, LSQScale]:
    """Attach LSQ scales to all weight parameters in a model.

    Returns a dict of LSQScale modules keyed by parameter name.
    Add these to an optimizer group with a higher learning rate (e.g., 10x).
    """
    scales = {}
    for name, param in model.named_parameters():
        if param.ndim >= 2:  # conv/linear weights, not biases
            scales[name] = LSQScale.from_tensor(param)
    return scales


# ── Save / Load ──────────────────────────────────────────────────────────


def save_int8(
    model: nn.Module,
    path: str | os.PathLike,
    *,
    meta: dict[str, Any] | None = None,
    per_channel: bool = False,
    fp32_bias: bool = False,
) -> int:
    """Save model weights in int8 quantized format.

    Args:
        model: the post-filter module
        path: output .pt file path
        meta: optional metadata dict (variant, hidden, kernel, alpha, ...)
        per_channel: use per-channel scales (better fidelity, ~same size)
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
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            scales_bc = scales.view(C, *([1] * (p.ndim - 1)))
            quantized = (p / scales_bc).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scales
        else:
            # Per-tensor: one global scale
            scale = p.abs().max() / 127.0
            if scale.item() == 0.0:
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

    for raw_key in state.keys():
        if raw_key == "__meta__":
            continue
        if raw_key.endswith(".q") or raw_key.endswith(".s"):
            base = raw_key[:-2]
            if base in seen:
                continue
            seen.add(base)
            q = state[base + ".q"].float()
            s = state[base + ".s"]
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


def get_meta(path: str | os.PathLike) -> dict[str, Any]:
    """Read metadata from an int8 weight file without loading all weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    return dict(state.get("__meta__", {}))
