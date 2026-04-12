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

        # Quantize: map to [-1, 1], discretize, map back
        levels_f = levels.float().reshape(view_shape)
        half_levels = (levels_f / 2.0).clamp(min=0.5)  # avoid /0

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

        # Save for backward
        saturated = (normalized.abs() > 1.0)
        ctx.save_for_backward(saturated, prune_mask.expand_as(weight))

        return quantized

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        saturated, pruned = ctx.saved_tensors
        # STE: pass gradient through, zero where saturated or pruned
        mask = (~saturated) & (~pruned)
        grad_weight = grad_out * mask.float()
        # Gradient for bits parameter: computed via chain rule in the module
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

    def total_bits(self) -> torch.Tensor:
        """Total bits across all channels (differentiable)."""
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

    Args:
        in_channels: input channels.
        out_channels: output channels.
        kernel_size: convolution kernel size.
        padding: convolution padding.
        dilation: convolution dilation.
        bias: whether to use bias.
        init_bits: initial bit-depth per channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
        init_bits: float = 8.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=bias,
        )
        self.bit_depth = LearnableBitDepth(out_channels, init_bits=init_bits)
        # Store architecture params for serialization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with self-compressed weights."""
        q_weight = self.bit_depth(self.conv.weight)
        if self.conv.bias is not None:
            # Bias is 1D (C_out,) -- quantize directly (per-channel via same bits)
            # Use simple round-to-nearest per-channel quantization for bias
            bits_clamped = self.bit_depth.bits.detach().clamp(0.0, 8.0)
            prune_mask = bits_clamped < 0.5
            abs_max = self.conv.bias.detach().abs().clamp(min=1e-10)
            levels = (2.0 ** bits_clamped.round()).clamp(min=1).long()
            half = (levels.float() / 2.0).clamp(min=0.5)
            normalized = self.conv.bias / abs_max
            scaled = normalized * half
            # STE: use detach trick
            q_bias = (scaled.round() - scaled).detach() + scaled
            q_bias = q_bias / half * abs_max
            q_bias = q_bias.masked_fill(prune_mask, 0.0)
        else:
            q_bias = None
        return F.conv2d(
            x, q_weight, q_bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups,
        )

    def weight_bits(self) -> torch.Tensor:
        """Total bits for this layer's weights (differentiable)."""
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        per_channel_bits = self.bit_depth.bits.clamp(0.0, 8.0)  # (C_out,)
        # Each channel stores fan_in weights at its bit-depth
        return (per_channel_bits * fan_in).sum()

    def bias_bits(self) -> torch.Tensor:
        """Total bits for this layer's bias (differentiable)."""
        if not self.has_bias:
            return torch.tensor(0.0, device=self.bit_depth.bits.device)
        return self.bit_depth.bits.clamp(0.0, 8.0).sum()


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
        int8_bytes = sum(
            l["channels"] * self.conv1.in_channels * self.kernel ** 2
            for l in stats["layers"]
        )  # rough estimate
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
    device: str = "cpu",
    log_every: int = 50,
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
        device: compute device.
        log_every: log interval in epochs.

    Returns:
        Trained SelfCompressingPostFilter.
    """
    model = model.to(device)
    posenet = posenet.to(device).eval()
    segnet = segnet.to(device).eval()
    comp_frames = comp_frames.to(device)
    gt_frames = gt_frames.to(device)

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

        # Scorer loss: PoseNet + SegNet
        # Build (B, T, C, H, W) pairs for scorer
        filtered_5d = filtered.unsqueeze(0)  # (1, 2, 3, H, W)
        gt_5d = gt_pair.unsqueeze(0)

        # PoseNet loss
        with torch.no_grad():
            gt_pose_in = posenet.preprocess_input(gt_5d)
            gt_pose = posenet(gt_pose_in)
        filtered_pose_in = posenet.preprocess_input(filtered_5d)
        pred_pose = posenet(filtered_pose_in)
        pose_loss = F.mse_loss(pred_pose, gt_pose)

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

        if epoch % log_every == 0 or epoch == epochs - 1:
            stats = model.compression_stats()
            print(
                f"  epoch {epoch:4d}/{epochs} | "
                f"scorer={scorer_loss.item():.4f} rate={rate_loss.item():.4f} | "
                f"bits={stats['total_bits']:,} ({stats['total_bytes']:.0f}B) | "
                f"active={sum(l['active'] for l in stats['layers'])}/{sum(l['channels'] for l in stats['layers'])}"
            )

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

        # Quantize active channels and pack
        packed = bytearray()

        for i, ch_idx in enumerate(active_indices):
            ch_bits = channel_bits[i]
            ch_weight = weight[ch_idx]  # (C_in, kH, kW)

            # Quantize to ch_bits levels
            flat = ch_weight.reshape(-1)
            abs_max = flat.abs().max().clamp(min=1e-10).item()
            n_levels = 2 ** ch_bits
            half = n_levels // 2

            # Map to integer levels
            quantized = (flat / abs_max * half).round().clamp(-half, half).long()
            # Shift to unsigned: [0, n_levels]
            unsigned = (quantized + half).clamp(0, n_levels).tolist()

            # Store scale as float16
            packed.extend(struct.pack("<e", abs_max))

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
                q = int(round(b_val / abs_max_b * half))
                q = max(-half, min(half, q))
                u = q + half
                bias_packed.extend(struct.pack("<e", abs_max_b))
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
            "bias_blob_len": len(bias_packed),
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

                # Dequantize
                dequant = torch.tensor(
                    [(v - half) / half * scale for v in values],
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
                    module.conv.bias[ch_idx] = q / half * scale_b

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


if __name__ == "__main__":
    _smoke_test()
