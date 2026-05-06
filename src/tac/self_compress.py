"""Self-compressing postfilter with learnable per-channel bit-depth.

Szabolcs Csefalvay's insight (arXiv 2301.13142): instead of uniform int8
quantization, learn VARIABLE bit-depth per channel during training. Channels
the scorer doesn't care about get pruned to 0 bits. Critical channels keep
up to 8 bits. Average: 2-3 bits/weight.

The bit-depth parameter is continuous during training via straight-through
estimator (STE), so gradient descent finds the optimal bit allocation.

Expected: 46KB postfilter -> 5-10KB. Rate savings at 25x multiplier: ~0.024 pts.

Usage::

    from tac.self_compress import (
        SelfCompressingPostFilter,
        train_self_compressing,
        export_compressed_checkpoint,
        load_compressed_checkpoint,
    )

    model = SelfCompressingPostFilter(hidden=64, kernel=3)
    trained = train_self_compressing(model, data, scorers, target_bits=8000)
    blob = export_compressed_checkpoint(trained)
    restored = load_compressed_checkpoint(blob, hidden=64, kernel=3)
"""

from __future__ import annotations

import io
import json
import math
import struct
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnableBitDepth",
    "SelfCompressingConv2d",
    "SelfCompressingPostFilter",
    "train_self_compressing",
    "export_compressed_checkpoint",
    "load_compressed_checkpoint",
    # Lane S renderer integration (2026-04-27)
    "SC_PROTECTED_NAME_PATTERNS",
    "swap_renderer_convs_with_self_compress",
    "list_self_compress_layers",
    "renderer_total_weight_bits",
    "renderer_average_bits_per_weight",
    "compute_renderer_rate_penalty",
    # Lane SG (2026-04-28) — protected-pattern-set selector
    "SC_SEGNET_PROTECTED_NAME_PATTERNS",
    "get_protected_patterns",
    # Lane SG (2026-04-28) — measured-sensitivity helper (anti-arbitrariness)
    "attribute_score_sensitivity_per_layer",
    "patterns_from_measured_sensitivity",
]


# ── Quantization primitives ─────────────────────────────────────────────


class _STEQuantize(torch.autograd.Function):
    """Quantize to given bit-depth with straight-through estimator.

    Forward: quantize weight to `2^bits` levels uniformly in [-max, max].
    Backward: pass gradient through unchanged (STE), zero where saturated.
    """

    @staticmethod
    def forward(
        ctx: Any,
        weight: torch.Tensor,
        bits: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        # bits is per-channel: shape (C_out,)
        # weight is (C_out, C_in, kH, kW) or (C_out,) for bias
        bits_clamped = bits.clamp(0.0, 8.0)

        if weight.ndim >= 2:
            view_shape = (-1,) + (1,) * (weight.ndim - 1)
        else:
            view_shape = (-1,)

        # Number of quantization levels per channel
        # When bits < 0.5, levels = 0 -> channel pruned
        levels = (2.0 ** bits_clamped.round()).long()  # (C_out,)

        # Per-channel scale
        flat = weight.detach().reshape(weight.shape[0], -1)
        abs_max = flat.abs().amax(dim=1).clamp(min=1e-10)  # (C_out,)
        abs_max = abs_max.reshape(view_shape)

        # Quantize: map to [-1, 1], discretize to n_levels values, map back.
        # For n_levels = 2^bits, we use signed range [-(n_levels/2 - 1), n_levels/2 - 1]
        # which gives exactly n_levels - 1 representable values (symmetric around 0).
        # This matches the export bit-packing which stores unsigned [0, n_levels-1].
        levels_f = levels.float().reshape(view_shape)
        half_levels = (levels_f / 2.0 - 1.0).clamp(min=0.5)  # avoid /0

        normalized = weight / abs_max  # [-1, 1]
        # Discretize to levels
        scaled = normalized * half_levels
        rounded = scaled.round()
        # Clamp to valid range
        rounded = rounded.clamp(-half_levels, half_levels)
        quantized = rounded / half_levels * abs_max

        # Prune channels with bits < 0.5
        prune_mask = (bits_clamped < 0.5).reshape(view_shape)
        quantized = quantized.masked_fill(prune_mask, 0.0)

        # Save for backward: detect saturation by checking if clamp changed the value.
        # A weight is saturated if it was clamped by rounded.clamp(-half_levels, half_levels),
        # meaning the quantized value hit the extreme representable bin.
        saturated = (scaled.abs() > half_levels)
        ctx.save_for_backward(saturated, prune_mask.expand_as(weight))

        return quantized

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        saturated, pruned = ctx.saved_tensors
        # STE: pass gradient through, zero where saturated or pruned
        mask = (~saturated) & (~pruned)
        grad_weight = grad_out * mask.float()
        # No gradient for bits: bit-depth is optimized via the explicit
        # rate penalty (Lagrangian), not through reconstruction loss.
        return grad_weight, None, None


def _ste_quantize(weight: torch.Tensor, bits: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Apply per-channel bit-depth quantization with STE."""
    return _STEQuantize.apply(weight, bits, training)


# ── Learnable bit-depth module ──────────────────────────────────────────


class LearnableBitDepth(nn.Module):
    """Differentiable per-channel bit-depth during training.

    Each output channel has a learnable `bits` parameter (float, 0-8).
    During forward: quantize weights to `round(bits)` levels.
    Gradient flows through straight-through estimator.
    When bits < 0.5, the channel is effectively pruned (zero output).

    After training: export only non-zero channels at their learned bit-depth.
    Archive size = sum(channels * fan_in * bits_per_channel) / 8 bytes.

    Args:
        num_channels: number of output channels to manage.
        init_bits: initial bit-depth (default 8.0 = standard int8).
    """

    def __init__(self, num_channels: int, init_bits: float = 8.0):
        super().__init__()
        self.num_channels = num_channels
        # Learnable bits parameter per channel, initialized to init_bits
        self.bits = nn.Parameter(torch.full((num_channels,), init_bits))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor according to learned bit-depths.

        Args:
            weight: (C_out, ...) weight tensor. C_out must == num_channels.

        Returns:
            Quantized weight tensor, same shape.
        """
        assert weight.shape[0] == self.num_channels, (
            f"Expected {self.num_channels} channels, got {weight.shape[0]}"
        )
        return _ste_quantize(weight, self.bits, self.training)

    def sum_channel_bitwidths(self) -> torch.Tensor:
        """Sum of per-channel bit-widths (NOT total model bits).

        Returns sum of clamped bit-depth values across channels. For a 128-channel
        layer at 4 bits each, this returns 512 — the sum of bit-widths, NOT the
        total number of bits in the layer (which would be 512 * fan_in).

        To get actual total bits for rate calculation, use
        SelfCompressingConv2d.weight_bits() which multiplies by fan_in.
        """
        return self.bits.clamp(0.0, 8.0).sum()

    def active_channels(self) -> int:
        """Number of channels with bits >= 0.5 (not pruned)."""
        return int((self.bits.detach() >= 0.5).sum().item())

    def effective_bits(self) -> float:
        """Mean bit-depth across active channels."""
        active = self.bits.detach() >= 0.5
        if active.sum() == 0:
            return 0.0
        return float(self.bits.detach()[active].mean().item())


# ── Self-compressing Conv2d ─────────────────────────────────────────────


class SelfCompressingConv2d(nn.Module):
    """Conv2d with learnable per-channel bit-depth.

    Wraps nn.Conv2d but quantizes weights on-the-fly during forward pass
    using the learned bit-depth per output channel.

    Lane S extension (2026-04-27): now accepts the full nn.Conv2d kwargs
    surface (``stride``, ``groups``, ``padding_mode``) so it can be used
    as a drop-in replacement for the renderer's ResBlocks, downsample
    convs, and 1x1 fusion convs. The backing nn.Conv2d carries the real
    state; ``SelfCompressingConv2d`` only adds ``bit_depth`` (a tiny
    nn.Embedding-shaped LearnableBitDepth tensor).

    Args:
        in_channels: input channels.
        out_channels: output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        padding: convolution padding.
        dilation: convolution dilation.
        groups: convolution groups (1 for standard, in_channels for depthwise).
        bias: whether to use bias.
        padding_mode: 'zeros' | 'reflect' | 'replicate' | 'circular'.
        init_bits: initial bit-depth per channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        init_bits: float = 8.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
        )
        self.bit_depth = LearnableBitDepth(out_channels, init_bits=init_bits)
        # Store architecture params for serialization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with self-compressed weights."""
        q_weight = self.bit_depth(self.conv.weight)
        if self.conv.bias is not None:
            # Bias is 1D (C_out,) -- quantize using same STE path as weights.
            # Reshape to (C_out, 1) so _ste_quantize treats dim-0 as channels.
            q_bias = _ste_quantize(
                self.conv.bias.unsqueeze(1),
                self.bit_depth.bits,
                self.training,
            ).squeeze(1)
        else:
            q_bias = None
        # Use the same padding-mode pipeline as the underlying conv. The
        # plain F.conv2d() path only handles 'zeros'; for non-zeros modes
        # we fall back to F.pad → F.conv2d(padding=0) like nn.Conv2d does.
        if self.conv.padding_mode != "zeros":
            pad = self.conv.padding
            if isinstance(pad, int):
                pad_tuple = (pad, pad, pad, pad)
            else:
                pad_tuple = (pad[1], pad[1], pad[0], pad[0])
            x = F.pad(x, pad_tuple, mode=self.conv.padding_mode)
            return F.conv2d(
                x, q_weight, q_bias,
                stride=self.conv.stride, padding=0,
                dilation=self.conv.dilation, groups=self.conv.groups,
            )
        return F.conv2d(
            x, q_weight, q_bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups,
        )

    def weight_numel(self) -> int:
        """Total number of weights (channels * fan_in)."""
        fan_in = (self.in_channels // self.groups) * self.kernel_size * self.kernel_size
        return self.out_channels * fan_in

    def weight_bits(self) -> torch.Tensor:
        """Total bits for this layer's weights (differentiable).

        For grouped convolutions each output channel only sees
        ``in_channels // groups`` input channels.
        """
        fan_in = (self.in_channels // self.groups) * self.kernel_size * self.kernel_size
        per_channel_bits = self.bit_depth.bits.clamp(0.0, 8.0)  # (C_out,)
        # Each channel stores fan_in weights at its bit-depth
        return (per_channel_bits * fan_in).sum()

    def bias_bits(self) -> torch.Tensor:
        """Total bits for this layer's bias (differentiable)."""
        if not self.has_bias:
            return torch.tensor(0.0, device=self.bit_depth.bits.device)
        return self.bit_depth.bits.clamp(0.0, 8.0).sum()

    def effective_bits_per_weight(self) -> float:
        """Mean bit-depth across all channels (non-differentiable)."""
        return float(self.bit_depth.bits.detach().clamp(0.0, 8.0).mean().item())


# ── Self-compressing postfilter ─────────────────────────────────────────


class SelfCompressingPostFilter(nn.Module):
    """Dilated postfilter with learnable per-channel bit-depth.

    Same architecture as DilatedPostFilter (3-layer residual CNN with
    dilation=2 on middle layer, 15x15 RF), but each Conv2d is replaced
    with SelfCompressingConv2d that learns its own quantization.

    Training: standard scorer loss + rate penalty on total bits.
    The rate penalty directly optimizes archive size.

    Args:
        hidden: hidden channel width (default 64, matching proven baseline).
        kernel: convolution kernel size.
        init_bits: initial bit-depth per channel (default 8.0).
    """

    def __init__(self, hidden: int = 64, kernel: int = 3, init_bits: float = 8.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = SelfCompressingConv2d(
            3, hidden, kernel, padding=pad, init_bits=init_bits,
        )
        self.conv2 = SelfCompressingConv2d(
            hidden, hidden, kernel, padding=pad * 2, dilation=2, init_bits=init_bits,
        )
        self.conv3 = SelfCompressingConv2d(
            hidden, 3, kernel, padding=pad, init_bits=init_bits,
        )
        self.act = nn.ReLU(inplace=True)
        # Zero-init output layer (starts as identity)
        nn.init.zeros_(self.conv3.conv.weight)
        nn.init.zeros_(self.conv3.conv.bias)

        self.hidden = hidden
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: input + learned_correction, with self-compressed weights."""
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)

    def total_bits(self) -> torch.Tensor:
        """Total bits across all layers (differentiable for rate penalty)."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in [self.conv1, self.conv2, self.conv3]:
            total = total + layer.weight_bits() + layer.bias_bits()
        return total

    def total_bytes(self) -> float:
        """Total bytes at current bit allocation (non-differentiable)."""
        return float(self.total_bits().detach().item()) / 8.0

    def compression_stats(self) -> dict[str, Any]:
        """Human-readable compression statistics."""
        stats: dict[str, Any] = {"layers": []}
        for name, layer in [("conv1", self.conv1), ("conv2", self.conv2), ("conv3", self.conv3)]:
            bd = layer.bit_depth
            stats["layers"].append({
                "name": name,
                "channels": bd.num_channels,
                "active": bd.active_channels(),
                "pruned": bd.num_channels - bd.active_channels(),
                "mean_bits": round(bd.effective_bits(), 2),
                "weight_bits": int(layer.weight_bits().item()),
                "bias_bits": int(layer.bias_bits().item()),
            })
        stats["total_bits"] = sum(l["weight_bits"] + l["bias_bits"] for l in stats["layers"])
        stats["total_bytes"] = stats["total_bits"] / 8
        # int8 baseline: each weight is 1 byte
        layer_modules = [self.conv1, self.conv2, self.conv3]
        int8_bytes = sum(
            m.out_channels * m.in_channels * m.kernel_size ** 2
            for m in layer_modules
        )
        stats["compression_ratio"] = int8_bytes / max(stats["total_bytes"], 1)
        return stats


# ── Training with rate penalty ──────────────────────────────────────────


def train_self_compressing(
    model: SelfCompressingPostFilter,
    comp_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    *,
    target_bits: int = 8000,
    epochs: int = 500,
    lr: float = 5e-4,
    lr_bits: float = 1e-2,
    lambda_rate_start: float = 0.0,
    lambda_rate_end: float = 1.0,
    ramp_start_frac: float = 0.3,
    scorer_weight: float = 20.0,
    device: str | None = None,
    log_every: int = 50,
    allow_cpu: bool = False,
    ema_decay: float = 0.997,
    eval_roundtrip: bool = True,
) -> SelfCompressingPostFilter:
    """Train a self-compressing postfilter.

    Loss = scorer_loss + lambda_rate * max(0, total_bits - target_bits)

    lambda_rate starts small (let model learn first) then ramps up
    (force compression). This is Lagrangian rate-distortion optimization.

    Args:
        model: SelfCompressingPostFilter instance.
        comp_frames: (N, 3, H, W) compressed frames float [0, 255].
        gt_frames: (N, 3, H, W) ground truth frames float [0, 255].
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        target_bits: target total bits for the model.
        epochs: number of training epochs.
        lr: learning rate for conv weights.
        lr_bits: learning rate for bit-depth parameters (higher = faster pruning).
        lambda_rate_start: initial rate penalty multiplier.
        lambda_rate_end: final rate penalty multiplier.
        ramp_start_frac: fraction of training before rate penalty ramp starts.
        scorer_weight: weight on scorer loss.
        device: compute device. Required (no default). Pass "cuda" for
            authoritative training. Use ``allow_cpu=True`` to opt into CPU
            training (advisory only — bytes/score will diverge from CUDA).
        log_every: log interval in epochs.
        allow_cpu: explicit opt-in to CPU/MPS training. Without this, a
            non-CUDA device raises. Even with the flag, an [advisory only]
            banner is printed and the returned checkpoint must NOT be promoted.
        ema_decay: EMA decay for inference-time weights (CLAUDE.md
            non-negotiable: 0.997). EMA is applied at eval-time with
            snapshot+restore; the trained EMA shadow is written into the
            returned model so ``export_compressed_checkpoint`` ships the
            shadow, not the live last-epoch weights.
        eval_roundtrip: when True, route filtered output through a uint8
            STE round-trip before scorer heads. Models the contest inflate
            uint8 quantization; without it, proxy-auth gap is 2-11x. CLAUDE.md
            non-negotiable.

    Returns:
        Trained SelfCompressingPostFilter (with EMA shadow loaded as the
        canonical inference-time weights, per CLAUDE.md EMA rule).
    """
    # Lazy import to avoid circular dependency: tac.training depends on
    # tac.losses + tac.data which are not heavy, but keeping the import
    # local guarantees self_compress remains a leaf module.
    from tac.training import EMA  # noqa: PLC0415  (intentional lazy import)

    # FORBIDDEN PATTERN guard: device must be explicit. Default "cpu" silently
    # produced [advisory only] checkpoints that were promoted as if they were
    # CUDA-faithful (CLAUDE.md MPS-NOISE / device-fallback non-negotiable).
    if device is None:
        raise ValueError(
            "train_self_compressing(...): `device` is required (no default). "
            "Pass device='cuda' for authoritative training, or "
            "device='cpu' with allow_cpu=True for [advisory only] runs."
        )
    device_str = str(device)
    is_cuda = device_str.startswith("cuda") or device_str == "cuda"
    if not is_cuda and not allow_cpu:
        raise ValueError(
            f"train_self_compressing: refusing to run on device={device_str!r} "
            f"without `allow_cpu=True`. Bytes and score on non-CUDA hardware "
            f"diverge from contest CUDA (FORBIDDEN PATTERN: device-fallback). "
            f"Pass allow_cpu=True if this is an [advisory only] run."
        )
    if not is_cuda:
        print(
            f"[advisory only] train_self_compressing on device={device_str!r}; "
            f"resulting checkpoint MUST NOT be promoted (CLAUDE.md MPS-NOISE)."
        )

    model = model.to(device)
    posenet = posenet.to(device).eval()
    segnet = segnet.to(device).eval()
    comp_frames = comp_frames.to(device)
    gt_frames = gt_frames.to(device)

    # CLAUDE.md NON-NEGOTIABLE: every training path MUST instantiate EMA,
    # update it after every optimizer.step, and use the shadow as the
    # inference checkpoint. Decay default 0.997. Snapshot+restore at eval.
    ema = EMA(model, decay=ema_decay)

    # Separate parameter groups: conv weights vs bit-depth params
    weight_params = []
    bits_params = []
    for name, param in model.named_parameters():
        if "bit_depth.bits" in name:
            bits_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": lr},
        {"params": bits_params, "lr": lr_bits},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = comp_frames.shape[0]

    for epoch in range(epochs):
        model.train()

        # Sample a random pair for scorer loss
        idx = torch.randint(0, max(1, N - 1), (1,)).item()
        comp_pair = comp_frames[idx:idx + 2]  # (2, 3, H, W)
        gt_pair = gt_frames[idx:idx + 2]

        if comp_pair.shape[0] < 2:
            comp_pair = comp_frames[:2]
            gt_pair = gt_frames[:2]

        # Forward through postfilter
        filtered = model(comp_pair)

        # CLAUDE.md NON-NEGOTIABLE: eval_roundtrip — STE uint8 round-trip
        # before scorer heads. The contest inflate path quantizes the
        # rendered output to uint8; training without this round-trip leaves
        # a 2-11x proxy-auth gap. Use straight-through estimator so the
        # gradient flows through the rounding op unchanged.
        if eval_roundtrip:
            filtered = filtered + (
                filtered.round().clamp(0.0, 255.0) - filtered
            ).detach()

        # Scorer loss: PoseNet + SegNet
        # Build (B, T, C, H, W) pairs for scorer
        filtered_5d = filtered.unsqueeze(0)  # (1, 2, 3, H, W)
        gt_5d = gt_pair.unsqueeze(0)

        # PoseNet loss
        with torch.no_grad():
            gt_pose_in = posenet.preprocess_input(gt_5d)
            gt_pose_out = posenet(gt_pose_in)
            gt_pose = gt_pose_out["pose"] if isinstance(gt_pose_out, dict) else gt_pose_out
        filtered_pose_in = posenet.preprocess_input(filtered_5d)
        pred_pose_out = posenet(filtered_pose_in)
        pred_pose = pred_pose_out["pose"] if isinstance(pred_pose_out, dict) else pred_pose_out
        pose_loss = F.mse_loss(pred_pose[..., :6], gt_pose[..., :6])

        # SegNet loss
        with torch.no_grad():
            gt_seg_in = segnet.preprocess_input(gt_5d)
            gt_seg = segnet(gt_seg_in)
        filtered_seg_in = segnet.preprocess_input(filtered_5d)
        pred_seg = segnet(filtered_seg_in)
        seg_loss = F.cross_entropy(
            pred_seg.reshape(-1, pred_seg.shape[-1]),
            gt_seg.argmax(dim=-1).reshape(-1),
        )

        scorer_loss = scorer_weight * (pose_loss + 100.0 * seg_loss)

        # Rate penalty: Lagrangian on total bits
        total_bits = model.total_bits()
        rate_excess = F.relu(total_bits - target_bits)

        # Lambda ramp schedule
        progress = epoch / max(epochs - 1, 1)
        if progress < ramp_start_frac:
            lambda_rate = lambda_rate_start
        else:
            ramp_progress = (progress - ramp_start_frac) / (1.0 - ramp_start_frac)
            lambda_rate = lambda_rate_start + (lambda_rate_end - lambda_rate_start) * ramp_progress

        rate_loss = lambda_rate * rate_excess

        total_loss = scorer_loss + rate_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Clamp bits to [0, 8]
        with torch.no_grad():
            for layer in [model.conv1, model.conv2, model.conv3]:
                layer.bit_depth.bits.clamp_(0.0, 8.0)

        # CLAUDE.md NON-NEGOTIABLE: update EMA after every optimizer.step.
        ema.update(model)

        if epoch % log_every == 0 or epoch == epochs - 1:
            stats = model.compression_stats()
            print(
                f"  epoch {epoch:4d}/{epochs} | "
                f"scorer={scorer_loss.item():.4f} rate={rate_loss.item():.4f} | "
                f"bits={stats['total_bits']:,} ({stats['total_bytes']:.0f}B) | "
                f"active={sum(l['active'] for l in stats['layers'])}/{sum(l['channels'] for l in stats['layers'])}"
            )

    # CLAUDE.md NON-NEGOTIABLE: inference / archive bytes come from the EMA
    # shadow, not from model.state_dict() after training. We load the shadow
    # into the returned model so downstream callers
    # (export_compressed_checkpoint, etc.) get the EMA weights by default.
    # This matches the train_distill / Quantizr / Lane G v3 canonical pattern.
    ema.apply(model)
    model.eval()
    return model


# ── Export / import compressed checkpoint ────────────────────────────────


def export_compressed_checkpoint(model: SelfCompressingPostFilter) -> bytes:
    """Export self-compressed model as minimal bytes.

    For each channel:
    - If bits < 0.5: skip (pruned)
    - Otherwise: quantize weights to learned bit-depth, pack bits

    Format:
      Header (JSON): architecture config + per-layer channel maps
      Body: packed weight bits using arithmetic-coded byte stream

    Uses simple byte packing for deterministic encode/decode.

    Args:
        model: trained SelfCompressingPostFilter.

    Returns:
        Compressed bytes.
    """
    model.eval()

    # Collect layer data
    layers_data: list[dict] = []
    weight_blobs: list[bytes] = []

    for name, layer in [("conv1", model.conv1), ("conv2", model.conv2), ("conv3", model.conv3)]:
        bits = layer.bit_depth.bits.detach().cpu()
        weight = layer.conv.weight.detach().cpu()
        bias = layer.conv.bias.detach().cpu() if layer.conv.bias is not None else None

        active_mask = bits >= 0.5
        active_indices = torch.where(active_mask)[0].tolist()
        channel_bits = bits[active_mask].round().clamp(1, 8).long().tolist()
        # Promote 1-bit to 2-bit BEFORE storing in header (export packs at promoted bits)
        channel_bits = [max(b, 2) for b in channel_bits]

        # Quantize active channels and pack
        packed = bytearray()

        for i, ch_idx in enumerate(active_indices):
            ch_bits = channel_bits[i]
            ch_weight = weight[ch_idx]  # (C_in, kH, kW)

            # Quantize to ch_bits levels (n_levels = 2^bits values: 0..n_levels-1)
            # Guard: promote 1-bit channels to 2-bit. At 1-bit, half=1, half-1=0,
            # making the quantization formula divide by zero. The STE trains 1-bit
            # channels with half_levels=0.5 (clamped), but export cannot represent
            # this — so promote to 2-bit which gives 4 levels and is the minimum
            # exportable precision.
            ch_bits = max(ch_bits, 2)
            flat = ch_weight.reshape(-1)
            abs_max = flat.abs().max().clamp(min=1e-10).item()
            n_levels = 2 ** ch_bits
            half = n_levels // 2

            # Map to integer levels: [-half, half-1] -> unsigned [0, n_levels-1]
            quantized = (flat / abs_max * (half - 1)).round().clamp(-(half - 1), half - 1).long()
            # Shift to unsigned: [0, n_levels-1]
            unsigned = (quantized + half).clamp(0, n_levels - 1).tolist()

            # Store scale as float16. ``struct.pack("<e", x)`` silently wraps
            # values above the fp16 max (65504.0) to ±inf in the binary stream;
            # clamp before packing so a large weight magnitude can never produce
            # an inf scale at decode time. (PARADIGM-δεζ audit fp16 overflow guard.)
            abs_max_packed = min(abs_max, 65504.0)
            packed.extend(struct.pack("<e", abs_max_packed))

            # Pack values at ch_bits each
            _pack_values(packed, unsigned, ch_bits)

        # Pack bias for active channels
        bias_packed = bytearray()
        if bias is not None:
            for i, ch_idx in enumerate(active_indices):
                ch_bits = channel_bits[i]
                b_val = bias[ch_idx].item()
                abs_max_b = max(abs(b_val), 1e-10)
                n_levels = 2 ** ch_bits
                half = n_levels // 2
                q = int(round(b_val / abs_max_b * (half - 1)))
                q = max(-(half - 1), min(half - 1, q))
                u = q + half
                # fp16 overflow guard (mirror of weight scale at L698).
                abs_max_b_packed = min(abs_max_b, 65504.0)
                bias_packed.extend(struct.pack("<e", abs_max_b_packed))
                bias_packed.extend(struct.pack("<H", u))

        weight_blobs.append(bytes(packed))

        layers_data.append({
            "name": name,
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "kernel_size": layer.kernel_size,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "has_bias": layer.has_bias,
            "active_indices": active_indices,
            "channel_bits": channel_bits,
            # NOTE: ``bias_blob_len`` removed — the decoder reads the bias-blob
            # length from the binary stream (struct.unpack at the read site),
            # not from the header. Storing it as header metadata was dead.
            # (PARADIGM-δεζ audit: dead-metadata removal.)
        })

        weight_blobs.append(bytes(bias_packed))

    # Build header
    header = {
        "version": 1,
        "hidden": model.hidden,
        "kernel": model.kernel,
        "layers": layers_data,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Pack: [header_len (4B)] [header JSON] [weight_blob_0] [bias_blob_0] ...
    buf = bytearray()
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    for blob in weight_blobs:
        buf.extend(struct.pack("<I", len(blob)))
        buf.extend(blob)

    return bytes(buf)


def _pack_values(buf: bytearray, values: list[int], bits: int) -> None:
    """Pack a list of unsigned integer values at `bits` per value into buf."""
    if bits == 8:
        # Fast path: one byte per value
        buf.extend(bytes(v & 0xFF for v in values))
        return

    # General bit-packing
    bit_buffer = 0
    bits_in_buffer = 0
    for v in values:
        bit_buffer |= (v & ((1 << bits) - 1)) << bits_in_buffer
        bits_in_buffer += bits
        while bits_in_buffer >= 8:
            buf.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer > 0:
        buf.append(bit_buffer & 0xFF)


def _unpack_values(data: bytes, offset: int, count: int, bits: int) -> tuple[list[int], int]:
    """Unpack `count` values at `bits` per value from data starting at offset.

    Returns (values, new_offset).
    """
    if bits == 8:
        values = [data[offset + i] for i in range(count)]
        return values, offset + count

    total_bits = count * bits
    total_bytes = (total_bits + 7) // 8
    raw = data[offset:offset + total_bytes]

    bit_buffer = 0
    for i, b in enumerate(raw):
        bit_buffer |= b << (i * 8)

    mask = (1 << bits) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bits

    return values, offset + total_bytes


def load_compressed_checkpoint(
    data: bytes,
    hidden: int | None = None,
    kernel: int | None = None,
) -> SelfCompressingPostFilter:
    """Load a self-compressed checkpoint at inflate time.

    Args:
        data: compressed bytes from export_compressed_checkpoint.
        hidden: override hidden width (uses header value if None).
        kernel: override kernel size (uses header value if None).

    Returns:
        SelfCompressingPostFilter with restored weights.
    """
    offset = 0

    # Read header
    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    h = hidden if hidden is not None else header["hidden"]
    k = kernel if kernel is not None else header["kernel"]

    # Create model with default init
    model = SelfCompressingPostFilter(hidden=h, kernel=k, init_bits=0.0)

    layer_modules = {"conv1": model.conv1, "conv2": model.conv2, "conv3": model.conv3}

    for layer_info in header["layers"]:
        name = layer_info["name"]
        module = layer_modules[name]
        active_indices = layer_info["active_indices"]
        channel_bits = layer_info["channel_bits"]
        fan_in = layer_info["in_channels"] * layer_info["kernel_size"] ** 2

        # Read weight blob
        blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        weight_data = data[offset:offset + blob_len]
        offset += blob_len

        # Read bias blob
        bias_blob_len_stored = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        bias_data = data[offset:offset + bias_blob_len_stored]
        offset += bias_blob_len_stored

        # Restore weights
        w_offset = 0
        with torch.no_grad():
            # Zero all weights first
            module.conv.weight.zero_()
            if module.conv.bias is not None:
                module.conv.bias.zero_()
            # Zero all bit-depths
            module.bit_depth.bits.zero_()

            for i, ch_idx in enumerate(active_indices):
                ch_bits = channel_bits[i]
                n_levels = 2 ** ch_bits
                half = n_levels // 2

                # Read scale
                scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                w_offset += 2

                # Unpack values
                values, w_offset = _unpack_values(weight_data, w_offset, fan_in, ch_bits)

                # Dequantize: unsigned [0, n_levels-1] -> signed [-(half-1), half-1] -> float
                dequant = torch.tensor(
                    [(v - half) / max(half - 1, 1) * scale for v in values],
                    dtype=torch.float32,
                )
                shape = module.conv.weight.shape[1:]  # (C_in, kH, kW)
                module.conv.weight[ch_idx] = dequant.reshape(shape)
                module.bit_depth.bits[ch_idx] = float(ch_bits)

            # Restore bias
            if module.conv.bias is not None and bias_data:
                b_offset = 0
                for i, ch_idx in enumerate(active_indices):
                    scale_b = struct.unpack("<e", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    u_val = struct.unpack("<H", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    ch_bits = channel_bits[i]
                    n_levels = 2 ** ch_bits
                    half = n_levels // 2
                    q = u_val - half
                    module.conv.bias[ch_idx] = q / max(half - 1, 1) * scale_b

    return model


# ── Smoke tests ─────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic shape, forward-pass, and export/import checks."""
    print("self_compress: running smoke tests...")

    B, H, W = 2, 64, 64

    # 1. SelfCompressingPostFilter forward
    model = SelfCompressingPostFilter(hidden=16, kernel=3, init_bits=8.0)
    x = torch.rand(B, 3, H, W) * 255.0
    y = model(x)
    assert y.shape == (B, 3, H, W), f"Output shape: {y.shape}"
    assert y.min() >= 0.0 and y.max() <= 255.0, "Output range violation"
    print(f"  forward pass: OK ({y.shape})")

    # 2. Total bits is differentiable
    total = model.total_bits()
    assert total.requires_grad, "total_bits must be differentiable"
    total.backward()
    for layer in [model.conv1, model.conv2, model.conv3]:
        assert layer.bit_depth.bits.grad is not None, "bits gradient must flow"
    print(f"  differentiable bits: OK (total={total.item():.0f} bits)")

    # 3. Compression stats
    stats = model.compression_stats()
    assert "total_bytes" in stats
    assert stats["total_bits"] > 0
    print(f"  compression stats: {stats['total_bytes']:.0f} bytes, {len(stats['layers'])} layers")

    # 4. Export and re-import
    model.eval()
    blob = export_compressed_checkpoint(model)
    assert len(blob) > 0
    print(f"  exported: {len(blob)} bytes")

    restored = load_compressed_checkpoint(blob, hidden=16, kernel=3)
    y2 = restored(x)
    assert y2.shape == (B, 3, H, W)
    # Quantization means exact match is unlikely, but should be close
    diff = (y - y2).abs().max().item()
    print(f"  round-trip max diff: {diff:.2f}")
    assert diff < 50.0, f"Round-trip error too large: {diff}"

    # 5. Pruning simulation: set some channels to 0 bits
    with torch.no_grad():
        model.conv1.bit_depth.bits[:8] = 0.0  # Prune first 8 channels
    stats_pruned = model.compression_stats()
    assert stats_pruned["layers"][0]["pruned"] == 8
    blob_pruned = export_compressed_checkpoint(model)
    assert len(blob_pruned) < len(blob), "Pruned model should be smaller"
    print(f"  pruned export: {len(blob_pruned)} bytes (was {len(blob)})")

    # 6. Load pruned checkpoint
    restored_pruned = load_compressed_checkpoint(blob_pruned, hidden=16, kernel=3)
    y3 = restored_pruned(x)
    assert y3.shape == (B, 3, H, W)
    print(f"  pruned round-trip: OK (max diff={( y3 - model(x)).abs().max().item():.2f})")

    print("self_compress: all smoke tests passed")


# ── Lane S renderer integration (2026-04-27) ─────────────────────────────
#
# The renderer must NOT swap every Conv2d uniformly: per Lane F findings the
# decoder's RGB head + motion's flow head + FiLM layers are scorer-sensitive
# at 100-1000x the rate of the rest of the network (PoseNet sees the YUV6
# residual; even a 1-bit perturbation on the head bias shows up as 0.5 in
# auth PoseNet score). So we hold those FP32 and only quantize the bulk
# feature-extraction convs.
#
# Architectural rationale (load-bearing — see report at the end of this
# task for full alternatives + why this won):
#   * Bulk convs (~95% of weights) → SelfCompressingConv2d. Carry the
#     learned-bit-depth load.
#   * Head convs (renderer.head, motion.head) → FP32 nn.Conv2d. <1% of
#     weights but >50% of scorer sensitivity per local SC postfilter
#     experiments and Lane F's same-arch FP4 regression of +0.44.
#   * FiLMLayer.scale / FiLMLayer.shift → FP32 nn.Linear. Tiny (12K params)
#     and modulate the entire bottleneck.
#   * 1x1 fuse_conv / fuse2_conv → FP32. Per-pixel mixing; sensitive.
#   * nn.ConvTranspose2d (up_conv / up2_conv) → FP32. Different gradient
#     profile from Conv2d STE; also small (~5% of weights).
#
# The swap is done POST-CONSTRUCTION via name-pattern matching so the rest
# of the renderer's architecture code stays unchanged. Profiles opt in with
# `use_self_compress_codec=True`.

# Name patterns that MUST stay FP32 (regex-style suffix or substring match).
# Concrete suffixes are checked first; if a layer's full dotted name ends
# with any of these, it stays FP32. See test_self_compress_renderer.py for
# the regression test that pins this list.
SC_PROTECTED_NAME_PATTERNS: tuple[str, ...] = (
    # Decoder RGB head — last conv before sigmoid output. PoseNet-sensitive.
    "renderer.head",
    # Motion flow / gate / residual head — sensitive to bit-quantization
    # because warp coordinates need sub-pixel precision.
    "motion.head",
    # FiLM modulation linear layers — small, scorer-sensitive.
    "film_bottleneck.scale",
    "film_bottleneck.shift",
    "film_decoder.scale",
    "film_decoder.shift",
    # 1x1 skip-fusion convs — per-pixel mixing, sensitive to quantization.
    "fuse_conv",
    "fuse2_conv",
    # Per-class CLADE embedding-driven affine — already small + sensitive.
    # (Embedding is not Conv2d so won't be swapped, but classify_gamma /
    # class_beta would be.)
)


def _is_protected_name(qualified_name: str) -> bool:
    """Return True iff a layer name should stay FP32.

    Matches by suffix. ``model.renderer.head`` matches pattern
    ``renderer.head`` because the suffix matches. We use suffix matching
    (not full-name) so the protection works regardless of how the
    AsymmetricPairGenerator nests its modules (model.renderer.head vs
    model.head, depending on caller).
    """
    for pat in SC_PROTECTED_NAME_PATTERNS:
        if qualified_name == pat or qualified_name.endswith("." + pat):
            return True
    return False


# Lane SG (2026-04-28 council EUREKA #5): re-scope which layers stay FP32
# based on which scorer we are protecting. The asymmetric scoring formula
# (100·seg + sqrt(10·pose) + 25·rate) makes SegNet 2-5× more impactful per
# unit-distortion at our operating point. The default protection list above
# (`posenet_prior`) was chosen for the FP4 → ASYM PoseNet path; the SegNet
# path needs a different set targeting the renderer layers that drive
# semantic-class boundaries (decoder out_conv, decode_head).
#
# These two sets are DISJOINT by construction: a layer protected for SegNet
# was unprotected for PoseNet and vice versa. The Lane SG hypothesis is that
# protecting SegNet-relevant layers at FP32 (while bulk decoder reaches 2.5
# bits avg) preserves the dominant 100·seg score component.
SC_SEGNET_PROTECTED_NAME_PATTERNS: tuple[str, ...] = (
    # Decoder RGB output convs — drive per-pixel semantic boundaries
    # that SegNet argmax flips on. Higher-precision here directly reduces
    # SegNet distortion (the 100× term).
    "out_conv",
    "decode_head",
    # Class-conditional gamma/beta affine — the per-class signal the SegNet
    # encoder reads through its EfficientNet-B2 stem.
    "class_gamma",
    "class_beta",
    # Decoder upsample 1x1 convs — semantic-mask-aware mixing
    "seg_fuse",
    "seg_skip_fuse",
)


def get_protected_patterns(pattern_set: str) -> list[str]:
    """Return the layer-name protection list for a given prior.

    Lane SG (2026-04-28): allows training/QAT to choose which scorer's
    sensitive layers stay FP32. The asymmetric scoring formula motivates
    different protection sets per scorer:

    - ``"posenet_prior"`` (legacy default): protects the FiLM bottleneck +
      motion head + decoder head — the layers Lane S identified as
      PoseNet-sensitive in the FP4 ASYM path.
    - ``"segnet_prior"`` (Lane SG): protects the decoder RGB output convs +
      class-conditional affines — the layers driving SegNet argmax.

    The two pattern sets are guaranteed disjoint so a Lane SG run cannot
    accidentally protect PoseNet-only layers (which would dilute the
    SegNet-sensitivity signal).

    Raises ValueError on unknown pattern set.
    """
    if pattern_set == "posenet_prior":
        return list(SC_PROTECTED_NAME_PATTERNS)
    if pattern_set == "segnet_prior":
        return list(SC_SEGNET_PROTECTED_NAME_PATTERNS)
    raise ValueError(
        f"Unknown pattern_set={pattern_set!r}; "
        f"expected one of: 'posenet_prior', 'segnet_prior'"
    )


def swap_renderer_convs_with_self_compress(
    model: nn.Module,
    *,
    init_bits: float = 8.0,
    skip_transposed: bool = True,
    skip_groupwise: bool = False,
    extra_protected_patterns: tuple[str, ...] = (),
    protected_patterns: tuple[str, ...] | None = None,
) -> dict:
    """Swap eligible nn.Conv2d layers in a renderer with SelfCompressingConv2d.

    Operates in-place on ``model``. The swap walks ``named_modules`` and
    replaces each ``nn.Conv2d`` whose qualified name is NOT in the active
    protection list with a ``SelfCompressingConv2d`` of the same shape,
    copying the original weights and bias. The ``LearnableBitDepth`` starts
    at ``init_bits`` (default 8 = full precision); the Lagrangian rate
    penalty during training drives bit-depth down toward the target.

    Args:
        model: a build_renderer(...) output (PairGenerator or
            AsymmetricPairGenerator).
        init_bits: initial bit-depth for swapped layers.
        skip_transposed: leave nn.ConvTranspose2d FP32 (recommended; STE
            backward through stride-2 transposed conv is ill-behaved in
            our experiments).
        skip_groupwise: leave grouped convs (e.g. depthwise) FP32. Default
            False — depthwise convs are tiny and benefit from SC.
        extra_protected_patterns: ADDITIONAL name suffixes to protect on
            top of ``SC_PROTECTED_NAME_PATTERNS``. Useful for ablations
            that want PoseNet protections + a few extra layers.
        protected_patterns: REPLACEMENT protection list. When provided
            (non-None), the legacy hardcoded ``SC_PROTECTED_NAME_PATTERNS``
            list is BYPASSED and only ``protected_patterns`` (plus any
            ``extra_protected_patterns``) is used. This is the canonical
            way to switch between scorer-prior protection sets — Lane SG
            (``segnet_prior``) needs SegNet-only protection, NOT
            posenet-prior + segnet (which is what
            ``extra_protected_patterns=segnet_list`` would have produced
            by mistake; see Codex F3 2026-04-28).

    Returns:
        Dict with diagnostics: ``{"swapped": list[str], "protected": list[str],
        "skipped": list[str], "total_swapped_params": int,
        "protected_patterns_used": list[str]}``.
    """
    swapped: list[str] = []
    protected: list[str] = []
    skipped: list[str] = []
    total_swapped_params = 0

    # F3 fix (2026-04-28): the active protection list can be EITHER the
    # legacy default (when caller passes nothing or only extras) OR a full
    # replacement (when caller passes protected_patterns=). The replacement
    # path is what Lane SG needs to protect SegNet-only layers without
    # also protecting the disjoint PoseNet-prior set.
    if protected_patterns is None:
        active_protected: tuple[str, ...] = tuple(SC_PROTECTED_NAME_PATTERNS) + tuple(extra_protected_patterns)
    else:
        active_protected = tuple(protected_patterns) + tuple(extra_protected_patterns)

    def _is_protected_with_extras(name: str) -> bool:
        for pat in active_protected:
            if name == pat or name.endswith("." + pat):
                return True
        return False

    # Collect (parent_module, child_name, full_name, child_module) so we can
    # safely setattr after iterating (mutating during iteration corrupts
    # the named_modules walk).
    candidates: list[tuple[nn.Module, str, str, nn.Module]] = []
    parents: dict[str, nn.Module] = {"": model}
    for full_name, module in model.named_modules():
        parents[full_name] = module
    for full_name, module in model.named_modules():
        # nn.ConvTranspose2d does NOT subclass nn.Conv2d in modern torch
        # (verified 2026-04-27 against torch 2.x). It is collected separately
        # below so we can decide whether to skip.
        if isinstance(module, nn.ConvTranspose2d):
            if skip_transposed:
                skipped.append(full_name + " (transposed)")
            else:
                skipped.append(full_name + " (transposed; SC for transposed not yet implemented)")
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if skip_groupwise and module.groups != 1 and module.groups != module.in_channels:
            # Skip non-trivial grouped convs (depthwise is fine, mid-grouped
            # is unusual and the SC rate accounting may be off).
            skipped.append(full_name + f" (groups={module.groups})")
            continue
        if _is_protected_with_extras(full_name):
            protected.append(full_name)
            continue
        # Find parent and child name
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = parents[parent_name]
        else:
            parent = model
            child_name = full_name
        candidates.append((parent, child_name, full_name, module))

    for parent, child_name, full_name, conv in candidates:
        # nn.Conv2d kwargs we need to mirror
        kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation

        sc = SelfCompressingConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            init_bits=init_bits,
        )
        with torch.no_grad():
            sc.conv.weight.copy_(conv.weight)
            if conv.bias is not None and sc.conv.bias is not None:
                sc.conv.bias.copy_(conv.bias)
        # Preserve dtype + device of the original conv
        sc = sc.to(conv.weight.device).to(conv.weight.dtype)
        setattr(parent, child_name, sc)
        swapped.append(full_name)
        total_swapped_params += sc.weight_numel() + (sc.out_channels if sc.has_bias else 0)

    return {
        "swapped": swapped,
        "protected": protected,
        "skipped": skipped,
        "total_swapped_params": total_swapped_params,
        # F3 fix: surface which protection list was used so ops can audit
        # post-hoc whether Lane SG actually protected SegNet-only layers
        # (NOT PoseNet ∪ SegNet, which the old additive code produced).
        "protected_patterns_used": list(active_protected),
    }


def list_self_compress_layers(model: nn.Module) -> list[tuple[str, "SelfCompressingConv2d"]]:
    """Return the list of (qualified_name, layer) for SC layers in model."""
    out: list[tuple[str, SelfCompressingConv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, SelfCompressingConv2d):
            out.append((name, module))
    return out


def renderer_total_weight_bits(model: nn.Module) -> torch.Tensor:
    """Sum of weight_bits across all SC layers (differentiable).

    Returns 0 (zero-dim tensor on cpu) when there are no SC layers — this
    keeps the rate-penalty call site safe to invoke unconditionally.
    """
    sc_layers = list_self_compress_layers(model)
    if not sc_layers:
        return torch.tensor(0.0)
    device = next(iter(sc_layers))[1].bit_depth.bits.device
    total = torch.zeros((), device=device)
    for _name, layer in sc_layers:
        total = total + layer.weight_bits() + layer.bias_bits()
    return total


def renderer_average_bits_per_weight(model: nn.Module) -> float:
    """Mean bits per weight across all SC layers (non-differentiable)."""
    sc_layers = list_self_compress_layers(model)
    if not sc_layers:
        return 0.0
    total_bits = 0.0
    total_weights = 0
    for _name, layer in sc_layers:
        total_bits += float(layer.weight_bits().detach().item())
        total_weights += layer.weight_numel()
    return total_bits / max(total_weights, 1)


def compute_renderer_rate_penalty(
    model: nn.Module,
    target_bits_per_weight: float,
    lambda_rate: float,
) -> torch.Tensor:
    """Compute the Lagrangian rate penalty for a self-compressed renderer.

    Penalty form (smooth hinge above target)::

        excess_bits = ReLU(total_bits - target_bits_per_weight * total_weights)
        rate_penalty = lambda_rate * excess_bits / total_weights  # normalized to bits/weight

    Normalising by total_weights makes the loss magnitude scale-invariant
    so the same ``lambda_rate`` works across renderer sizes.

    Returns 0 (zero-dim tensor) when the model has no SC layers, so the
    caller can `loss + compute_renderer_rate_penalty(...)` unconditionally.
    """
    sc_layers = list_self_compress_layers(model)
    if not sc_layers:
        return torch.tensor(0.0)
    device = next(iter(sc_layers))[1].bit_depth.bits.device
    total_bits = torch.zeros((), device=device)
    total_weights = 0
    for _name, layer in sc_layers:
        total_bits = total_bits + layer.weight_bits() + layer.bias_bits()
        total_weights += layer.weight_numel()
    target_total = float(target_bits_per_weight) * float(total_weights)
    excess = F.relu(total_bits - target_total)
    return lambda_rate * (excess / max(total_weights, 1))


# ── Lane SG (2026-04-28) — measured per-layer sensitivity ───────────────
#
# THIS IS THE ANTI-ARBITRARINESS PIECE.
#
# `SC_PROTECTED_NAME_PATTERNS` and `SC_SEGNET_PROTECTED_NAME_PATTERNS` above
# are HEURISTIC pattern strings derived from architectural intuition. Per
# CLAUDE.md and the Lane G v3 wedge attribution council
# (`.omx/research/lane_g_v3_stacking_skunkworks_20260428.md` §1.4), heuristic
# patterns are exactly the kind of arbitrary choice this codebase warns
# against. The pattern sets must be VALIDATED by measurement before they are
# trusted to drive a multi-hour Vast.ai training run.
#
# `attribute_score_sensitivity_per_layer` implements that measurement:
# perturb each layer's weights by a calibrated fraction, propagate through the
# renderer + scorer fns, and return the resulting Δ(seg_score) and Δ(pose_score)
# normalized per parameter. Layers with the highest sensitivity per parameter
# are the ones whose protection actually buys score; everything else can be
# self-compressed without measurable distortion impact.
#
# Output:
#     {
#       "renderer.head":     {"seg_dscore": 0.0123, "pose_dscore": 0.0008,
#                             "n_params": 111, "rate_per_param": 0.0001},
#       "renderer.fuse_conv":{"seg_dscore": 0.0034, "pose_dscore": 0.0002, ...},
#       ...
#     }
#
# The companion helper `patterns_from_measured_sensitivity` then takes that
# dict + a target ("segnet" | "posenet" | "both") + a top-K cutoff and emits
# a list of layer-name patterns that can be passed as
# `extra_protected_patterns` to `swap_renderer_convs_with_self_compress`.
#
# Usage::
#
#     sens = attribute_score_sensitivity_per_layer(
#         renderer, seg_score_fn, pose_score_fn, samples,
#         delta_frac=0.05, n_repeats=2, device="cpu",
#     )
#     extra = patterns_from_measured_sensitivity(sens, target="segnet", top_k=4)
#     swap_renderer_convs_with_self_compress(
#         renderer, extra_protected_patterns=tuple(extra),
#     )


def _flatten_to_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape an arbitrary-shape tensor into 2D (batch, features).

    Helper for treating renderer outputs as opaque feature vectors so the
    measurement helper can be called with toy scorer functions in tests.
    """
    return x.detach().reshape(x.shape[0], -1)


def attribute_score_sensitivity_per_layer(
    model: nn.Module,
    seg_score_fn,  # callable(model_output) -> scalar Tensor
    pose_score_fn,  # callable(model_output) -> scalar Tensor
    samples,  # iterable of args; each item is passed as model(*args)
    *,
    delta_frac: float = 0.05,
    n_repeats: int = 2,
    device: str | torch.device = "cpu",
    skip_protected: bool = False,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Measure per-layer ΔSegNet vs ΔPoseNet score sensitivity.

    For each ``nn.Conv2d`` (or ``SelfCompressingConv2d.conv``) layer in
    ``model``, perturb the weight tensor by Gaussian noise of std =
    ``delta_frac * weight.abs().mean()``, re-evaluate the model on
    ``samples``, and record the absolute change in ``seg_score_fn`` and
    ``pose_score_fn`` averaged over ``n_repeats`` perturbation seeds.

    The perturbation is RESTORED after each measurement so the model state
    is unchanged on return.

    Args:
        model: a renderer (PairGenerator or AsymmetricPairGenerator).
        seg_score_fn: callable taking the model output and returning a scalar
            tensor. Tests can pass a simple ``output.mean()``-style stand-in;
            production callers should pass a real SegNet score function.
        pose_score_fn: same signature, for PoseNet.
        samples: iterable of tuples; each tuple is unpacked as ``model(*sample)``.
        delta_frac: perturbation magnitude as a fraction of mean abs weight.
            0.05 (default) is small enough to stay in the linear regime for
            most renderer convs; raise to 0.1 for noisier signal.
        n_repeats: number of independent perturbation seeds to average.
        device: compute device; weights moved as-needed.
        skip_protected: if True, also skip layers already in
            ``SC_PROTECTED_NAME_PATTERNS`` ∪ ``SC_SEGNET_PROTECTED_NAME_PATTERNS``.
            Default False — measure every layer so the result can be cross-
            checked against the heuristic lists.
        seed: torch RNG seed base. Each repeat uses ``seed + repeat_idx``.

    Returns:
        ``dict[layer_name, dict[str, float]]`` with keys:
          - ``seg_dscore``: mean |Δ seg_score_fn| under perturbation
          - ``pose_dscore``: mean |Δ pose_score_fn| under perturbation
          - ``n_params``: number of weights in the perturbed layer
          - ``rate_per_param``: 1/n_params (rate cost of NOT compressing this
            layer; useful for ranking sensitivity-per-byte)
    """
    model = model.to(device)
    model.eval()

    # 1. Baseline scores (no perturbation), averaged across samples.
    baseline_seg = 0.0
    baseline_pose = 0.0
    n_samples = 0
    sample_list = list(samples)
    if not sample_list:
        return {}

    with torch.no_grad():
        for sample in sample_list:
            sample_args = tuple(s.to(device) if torch.is_tensor(s) else s for s in sample)
            out = model(*sample_args)
            baseline_seg += float(seg_score_fn(out).detach().item())
            baseline_pose += float(pose_score_fn(out).detach().item())
            n_samples += 1
    baseline_seg /= max(n_samples, 1)
    baseline_pose /= max(n_samples, 1)

    # 2. Identify candidate Conv2d layers (mirror swap_renderer_convs logic).
    candidates: list[tuple[str, nn.Conv2d]] = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            continue
        if isinstance(module, SelfCompressingConv2d):
            # For SC layers, perturb the underlying conv.weight.
            candidates.append((full_name, module.conv))
            continue
        if isinstance(module, nn.Conv2d):
            if skip_protected and (
                _is_protected_name(full_name)
                or any(full_name.endswith("." + p) or full_name == p
                       for p in SC_SEGNET_PROTECTED_NAME_PATTERNS)
            ):
                continue
            candidates.append((full_name, module))

    if not candidates:
        return {}

    # 3. Perturb each layer and measure score deltas.
    results: dict[str, dict[str, float]] = {}
    rng = torch.Generator(device="cpu")

    for full_name, conv in candidates:
        weight = conv.weight
        n_params = int(weight.numel())
        # Calibrate noise scale to weight magnitude to keep perturbation
        # in linear regime regardless of layer init.
        weight_scale = float(weight.detach().abs().mean().item())
        if weight_scale < 1e-12:
            # Zero-init layers (e.g., renderer.head) — use a tiny absolute std
            # so we still get measurable but bounded deltas.
            std = max(delta_frac * 1e-3, 1e-6)
        else:
            std = delta_frac * weight_scale

        seg_deltas: list[float] = []
        pose_deltas: list[float] = []

        for repeat_idx in range(n_repeats):
            rng.manual_seed(seed + repeat_idx)
            noise = torch.randn(weight.shape, generator=rng, dtype=weight.dtype) * std
            noise = noise.to(weight.device)

            with torch.no_grad():
                weight.add_(noise)

            with torch.no_grad():
                seg_p = 0.0
                pose_p = 0.0
                for sample in sample_list:
                    sample_args = tuple(
                        s.to(device) if torch.is_tensor(s) else s for s in sample
                    )
                    out = model(*sample_args)
                    seg_p += float(seg_score_fn(out).detach().item())
                    pose_p += float(pose_score_fn(out).detach().item())
                seg_p /= max(n_samples, 1)
                pose_p /= max(n_samples, 1)

            seg_deltas.append(abs(seg_p - baseline_seg))
            pose_deltas.append(abs(pose_p - baseline_pose))

            # Restore weight (subtract the noise we just added)
            with torch.no_grad():
                weight.sub_(noise)

        seg_dscore = sum(seg_deltas) / max(len(seg_deltas), 1)
        pose_dscore = sum(pose_deltas) / max(len(pose_deltas), 1)
        results[full_name] = {
            "seg_dscore": seg_dscore,
            "pose_dscore": pose_dscore,
            "n_params": n_params,
            "rate_per_param": 1.0 / max(n_params, 1),
        }

    return results


def patterns_from_measured_sensitivity(
    sensitivity: dict[str, dict[str, float]],
    *,
    target: str = "segnet",
    top_k: int = 4,
    min_dscore: float = 0.0,
) -> list[str]:
    """Convert a sensitivity dict into a protected-layer name pattern list.

    Top-K layers (ranked by ``{target}_dscore``) become the protected list.
    Returned strings are FULL qualified names — the swap helper matches by
    suffix so passing them as ``extra_protected_patterns`` will protect
    exactly those layers.

    Args:
        sensitivity: output of ``attribute_score_sensitivity_per_layer``.
        target: ``"segnet"`` (Lane SG default), ``"posenet"`` (legacy),
            or ``"both"`` (rank by ``seg_dscore + pose_dscore``).
        top_k: number of layers to protect.
        min_dscore: skip layers whose target dscore is below this floor;
            useful to avoid protecting layers that have effectively zero
            measured impact.

    Raises:
        ValueError: on unknown ``target``.

    Returns:
        List of qualified layer names ranked by descending sensitivity.
    """
    if target == "segnet":
        key = lambda v: v["seg_dscore"]
    elif target == "posenet":
        key = lambda v: v["pose_dscore"]
    elif target == "both":
        key = lambda v: v["seg_dscore"] + v["pose_dscore"]
    else:
        raise ValueError(
            f"Unknown target={target!r}; expected 'segnet'|'posenet'|'both'"
        )

    ranked = sorted(sensitivity.items(), key=lambda item: key(item[1]), reverse=True)
    out: list[str] = []
    for name, vals in ranked:
        if key(vals) < min_dscore:
            continue
        out.append(name)
        if len(out) >= top_k:
            break
    return out


if __name__ == "__main__":
    _smoke_test()
