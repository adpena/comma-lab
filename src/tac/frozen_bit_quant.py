# SPDX-License-Identifier: MIT
"""Lane Ω FrozenBitFakeQuant — per-element bit-depth STE fake-quantization.

Given a weight tensor W and a SAME-SHAPE uint8 tensor `bits` specifying the
bit-depth for each element, produce a quantized W with each element rounded
to the nearest representable value in its assigned bit-depth. Per-output-
channel scale (max(|w|) over each output channel) gives proper dynamic-range
coverage without needing a global scale.

Backward = STE (straight-through estimator): gradient w.r.t. weight = upstream
gradient. Bits tensor is FROZEN (uint8, no_grad) — Lane Ω's bit allocation is
decided BEFORE QAT (Phase 2 water-fill) and the QAT loop only fine-tunes the
underlying float weight to minimize loss given the fixed per-weight bit-depth.

Why this design (vs Lane S's LearnableBitDepth):
    * Lane S learns per-CHANNEL bit-depth via Lagrangian rate penalty.
    * Lane Ω freezes per-WEIGHT bit-depth from a Hessian profile — much
      finer resolution but much smaller search space (the gradient signal
      for "how many bits should THIS individual weight get" is too noisy
      to learn directly; instead we measure it once via Fisher importance,
      then freeze).

Per-output-channel scaling (not per-tensor):
    * Per-tensor max scaling lets a single outlier kill the precision of
      every other weight in the layer.
    * Per-output-channel scaling matches what Lane S / SCv1 / FP4A all do
      and what Quantizr / OBQ literature recommends.

Symmetric round-to-nearest:
    * For b bits, representable values are {-(2^(b-1)-1), …, 0, …, 2^(b-1)-1}
      multiplied by scale / (2^(b-1)-1). For b=1 we use {-scale, +scale}.
    * Quantization is symmetric → no zero-point storage needed.

CLAUDE.md compliance:
    * Pure PyTorch. CUDA-compatible. No MPS fallback.
    * Tested: gradient flow (STE), 8-bit near-identity, 1-bit clustering,
      per-channel scale correctness, finite outputs.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

__all__ = [
    "FrozenBitFakeQuant",
    "fake_quantize_per_weight",
    "FrozenBitConv2d",
    "wrap_layer_with_frozen_bits",
    "compute_per_channel_scale",
]


def compute_per_channel_scale(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel max(|w|) scale.

    For a weight tensor of shape (C_out, C_in, kH, kW) (Conv2d) or (C_out, C_in)
    (Linear), returns a 1D tensor of length C_out where scale[i] = max
    absolute value of weight[i].

    A floor of 1e-8 prevents zero-scale (would make the bin width 0 and
    saturate every quantized weight to ±something / 0).
    """
    if weight.dim() < 1:
        raise ValueError(f"weight must be ≥1D, got shape {tuple(weight.shape)}")
    out_dim = weight.shape[0]
    flat = weight.reshape(out_dim, -1)
    scale = flat.abs().max(dim=1).values
    return scale.clamp(min=1e-8)


class FrozenBitFakeQuant(torch.autograd.Function):
    """Per-element bit-depth fake-quantization with STE backward.

    Forward: For each element w, compute q = round(w / step) * step where
        step = scale_per_channel / (2^(bits-1) - 1)
    For bits=1 use the {-scale, +scale} sign-quantizer (no representable 0).

    Backward: STE — pass upstream gradient through unchanged for the weight,
    return None for scale and bits (frozen).
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        scale: torch.Tensor,  # (C_out,)
        bits: torch.Tensor,  # same shape as weight, uint8, in [1, 8]
    ) -> torch.Tensor:
        if weight.dim() < 1:
            raise ValueError(f"weight must be ≥1D, got {tuple(weight.shape)}")
        if bits.shape != weight.shape:
            raise ValueError(
                f"bits shape {tuple(bits.shape)} must match weight shape "
                f"{tuple(weight.shape)}"
            )
        if scale.shape != (weight.shape[0],):
            raise ValueError(
                f"scale shape {tuple(scale.shape)} must equal "
                f"({weight.shape[0]},) for per-output-channel scale"
            )

        # Broadcast scale to weight shape: (C_out, 1, 1, …)
        scale_b = scale.view(-1, *([1] * (weight.dim() - 1))).to(weight.dtype)

        bits_f = bits.to(weight.dtype)
        # Number of one-sided levels: for b ≥ 2 → 2^(b-1) - 1 (e.g. b=8 → 127)
        # For b == 1 we encode it as the sign quantizer (levels = 1).
        levels = (2.0 ** (bits_f - 1.0) - 1.0).clamp(min=1.0)

        # step has the same broadcastable shape as weight via scale_b.
        step = scale_b / levels  # (C_out, 1, 1, …) or (C_out, 1)

        # Symmetric round-to-nearest, then clip to [-scale, scale].
        # For bits ≥ 2 (levels ≥ 1): standard quantization → values in
        # {-levels, …, +levels} × step.
        # For bits == 1: we override to SIGN-ONLY (±scale) so the QAT path
        # matches the OMG1 export packer (which encodes 1-bit as sign).
        # Without this override, the QAT-time quantizer would also produce
        # 0 for |w| < scale/2, but the export packer always emits ±scale
        # → load-time weight ≠ QAT-time weight. STE gradient is unaffected.
        q_general = torch.round(weight / step) * step
        q_general = q_general.clamp(min=-scale_b, max=scale_b)
        # Sign quantizer for 1-bit elements: q = sign(w) × scale.
        q_sign = torch.where(
            weight >= 0, scale_b.expand_as(weight), -scale_b.expand_as(weight),
        )
        one_bit_mask = (bits.to(weight.dtype) == 1.0)
        q = torch.where(one_bit_mask, q_sign, q_general)

        # Sanity: keep dtype matching weight for downstream conv/linear.
        return q.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D401
        # STE: gradient w.r.t. weight is the upstream gradient. Scale and bits
        # are frozen (no gradient).
        return grad_output, None, None


def fake_quantize_per_weight(
    weight: torch.Tensor,
    bits: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convenience entrypoint that auto-computes per-output-channel scale.

    Args:
        weight: float tensor of any shape.
        bits: uint8 tensor of same shape, values in [1, 8].
        scale: optional per-output-channel scale; if None, computed from weight.

    Returns:
        Same-shape float tensor with each element quantized to its bit-depth.
    """
    if scale is None:
        scale = compute_per_channel_scale(weight)
    return FrozenBitFakeQuant.apply(weight, scale, bits)


# ── Layer wrappers (used during QAT to splice fake-quant into forward) ──


class FrozenBitConv2d(nn.Module):
    """Conv2d that fake-quantizes its weight per-element on every forward pass.

    Holds the underlying Conv2d's weight as a learnable parameter; bits is a
    frozen buffer (uint8, no grad). On every forward pass:
        1. Recompute per-output-channel scale from the current weight (so the
           scale tracks training drift, exactly like SC's fake-quant).
        2. Apply FrozenBitFakeQuant(weight, scale, bits) to get q_weight.
        3. Run F.conv2d with q_weight (and the original bias).

    This means QAT sees the rounded weight in forward but the STE backward
    updates the underlying float weight. After training, calling
    `q_weight = fake_quantize_per_weight(self.weight, self.bits)` gives the
    deterministic weights to write into the Ωv1 export.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        # bits buffer: defaults to all-8 (full precision) until install_bits().
        self.register_buffer(
            "bits",
            torch.full(
                tuple(self.conv.weight.shape), fill_value=8, dtype=torch.uint8,
            ),
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def install_bits(self, bits: torch.Tensor) -> None:
        """Replace the bits buffer (must match weight shape)."""
        if bits.shape != self.conv.weight.shape:
            raise ValueError(
                f"bits shape {tuple(bits.shape)} must match weight shape "
                f"{tuple(self.conv.weight.shape)}"
            )
        if bits.dtype != torch.uint8:
            bits = bits.to(torch.uint8)
        self.bits = bits.to(self.conv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = compute_per_channel_scale(self.conv.weight)
        q_w = FrozenBitFakeQuant.apply(self.conv.weight, scale, self.bits)
        return torch.nn.functional.conv2d(
            x,
            q_w,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def wrap_layer_with_frozen_bits(
    parent: nn.Module,
    child_name: str,
    bits: torch.Tensor,
) -> FrozenBitConv2d:
    """Replace a child Conv2d on `parent` with a FrozenBitConv2d that
    fake-quantizes its weight to the given per-element bit-depths.

    Copies the Conv2d's existing weights/bias into the new wrapper so the
    QAT loop starts from the trained (Lane A) weights, not random init.

    Returns the new FrozenBitConv2d (already setattr'd on parent).
    """
    old = getattr(parent, child_name)
    if not isinstance(old, nn.Conv2d):
        raise TypeError(
            f"wrap_layer_with_frozen_bits expects nn.Conv2d at "
            f"{child_name!r}, got {type(old).__name__}"
        )
    if bits.shape != old.weight.shape:
        raise ValueError(
            f"bits shape {tuple(bits.shape)} must match weight shape "
            f"{tuple(old.weight.shape)} for child {child_name!r}"
        )
    kernel_size = old.kernel_size[0] if isinstance(old.kernel_size, tuple) else old.kernel_size
    stride = old.stride[0] if isinstance(old.stride, tuple) else old.stride
    padding = old.padding[0] if isinstance(old.padding, tuple) else old.padding
    dilation = old.dilation[0] if isinstance(old.dilation, tuple) else old.dilation

    wrapper = FrozenBitConv2d(
        in_channels=old.in_channels,
        out_channels=old.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        wrapper.conv.weight.copy_(old.weight)
        if old.bias is not None:
            wrapper.conv.bias.copy_(old.bias)
    wrapper.install_bits(bits)
    setattr(parent, child_name, wrapper)
    return wrapper


def iter_layer_pairs(
    model: nn.Module,
) -> Iterable[tuple[str, nn.Module, str, nn.Module]]:
    """Yield (full_name, parent, child_name, child_module) for every nn.Module
    so callers can locate where to splice FrozenBitConv2d. Used by Lane Ω
    Phase 3 QAT setup.

    Status (2026-05-12, W/I/A I-4 decision): RESEARCH-ONLY utility helper.
    The bit-allocator side of the unified-Lagrangian solver does not yet
    consume this iterator (per integration_gap_audit_20260512.md I-4). The
    helper is preserved per CLAUDE.md "KILL/FALSIFIED is LAST RESORT" with
    the following reactivation criterion:

    Reactivation requires a Lane Ω Phase 3 QAT consumer that takes the
    iterator and produces per-layer FrozenBitConv2d splice decisions wired
    into the autopilot's bit-allocator hook (CLAUDE.md unified-Lagrangian
    wire-in hook 3). Until then the helper stays here as a typed iterator
    contract for that future consumer.

    Wire-in declaration per CLAUDE.md Catalog #125 (subagent
    coherence-by-default): hook 3 (bit-allocator) — DEFERRED-pending-consumer.
    """
    parents: dict[str, nn.Module] = {"": model}
    for name, mod in model.named_modules():
        parents[name] = mod
    for name, mod in model.named_modules():
        if name == "":
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = parents.get(parent_name)
        else:
            parent = model
            child_name = name
        if parent is None:
            continue
        yield name, parent, child_name, mod
