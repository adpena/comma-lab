"""FP4 quantization with codebook for extreme weight compression.

The competitor (mask2mask, score 0.60) uses a clever 4-bit scheme:
    - 8-value codebook: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    - Each weight is mapped to nearest codebook entry after per-block scaling
    - Block size 32: one fp16 scale per 32 weights
    - 4 bits per weight → ~2x smaller than int8

This achieves ~195KB for a 300K-param model (vs ~390KB for int8).

Functions:
    - quantize_fp4: compress a state dict to packed FP4 format
    - dequantize_fp4: decompress packed FP4 back to float state dict
    - FakeQuantFP4: STE-based fake quantization for QAT training
    - save_fp4 / load_fp4: disk serialization

The codebook is asymmetric (positive only, with a sign bit stored separately)
because neural network weights are roughly symmetric around zero. The actual
storage is:
    - 3-bit codebook index (8 values)
    - 1-bit sign
    = 4 bits total per weight
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

# Default codebook: 8 positive values, used with separate sign bit
# These are the magnitudes; actual weight = sign * scale * codebook[index]
DEFAULT_CODEBOOK = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

DEFAULT_BLOCK_SIZE = 32


# ── Core quantization ───────────────────────────────────────────────────


def _quantize_block(
    block: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 1D block of weights to FP4.

    Args:
        block: (block_size,) float tensor
        codebook: (8,) positive codebook values

    Returns:
        (indices, signs, scale) where:
            indices: (block_size,) uint8 in [0, 7] (3-bit codebook index)
            signs: (block_size,) int8 in {-1, +1}
            scale: scalar float (per-block scale factor)
    """
    # Extract signs and magnitudes
    signs = block.sign()
    signs[signs == 0] = 1.0  # map zero to positive
    magnitudes = block.abs()

    # Compute block scale: maps max magnitude to max codebook value
    max_mag = magnitudes.max()
    max_cb = codebook[-1]
    scale = max_mag / max_cb if max_mag > 1e-10 else torch.tensor(1.0, device=block.device)

    # Normalize magnitudes to codebook range
    normalized = magnitudes / scale

    # Find nearest codebook entry for each weight
    # (N, 1) vs (1, 8) → (N, 8) distances
    dists = (normalized.unsqueeze(1) - codebook.unsqueeze(0).to(block.device)).abs()
    indices = dists.argmin(dim=1).to(torch.uint8)

    return indices, signs.to(torch.int8), scale


def _dequantize_block(
    indices: torch.Tensor,
    signs: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a block from FP4 representation.

    Args:
        indices: (block_size,) uint8 codebook indices
        signs: (block_size,) int8 signs
        scale: scalar scale factor
        codebook: (8,) codebook values

    Returns:
        (block_size,) float tensor of reconstructed weights
    """
    cb = codebook.to(scale.device)
    values = cb[indices.long()]
    return values * signs.float() * scale


def _pack_indices_signs(
    indices: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Pack 3-bit index + 1-bit sign into 4-bit nibbles, two per byte.

    Layout per nibble: [sign_bit | index[2] | index[1] | index[0]]
    Two nibbles packed per uint8: high nibble = even index, low nibble = odd index.

    Args:
        indices: (N,) uint8 in [0, 7]
        signs: (N,) int8 in {-1, +1}

    Returns:
        (ceil(N/2),) uint8 packed tensor
    """
    # sign bit: 0 for positive, 1 for negative
    sign_bits = (signs < 0).to(torch.uint8)
    nibbles = (sign_bits << 3) | indices  # 4-bit value per weight

    # Pack two nibbles per byte
    n = nibbles.shape[0]
    if n % 2 != 0:
        nibbles = torch.cat([nibbles, torch.zeros(1, dtype=torch.uint8, device=nibbles.device)])
    nibbles = nibbles.reshape(-1, 2)
    packed = (nibbles[:, 0] << 4) | nibbles[:, 1]
    return packed


def _unpack_indices_signs(
    packed: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack 4-bit nibbles back to indices and signs.

    Args:
        packed: (ceil(count/2),) uint8 packed tensor
        count: original number of weights

    Returns:
        (indices, signs) each (count,) tensors
    """
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    nibbles = torch.stack([high, low], dim=1).reshape(-1)[:count]

    indices = (nibbles & 0x07).to(torch.uint8)
    sign_bits = (nibbles >> 3) & 0x01
    signs = torch.where(sign_bits == 0, torch.tensor(1, dtype=torch.int8),
                        torch.tensor(-1, dtype=torch.int8))
    return indices, signs


# ── State dict level ────────────────────────────────────────────────────


def quantize_fp4(
    state_dict: dict[str, torch.Tensor],
    codebook: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    """Quantize a model state dict to FP4 packed format.

    Each parameter is split into blocks of `block_size` weights. Each block
    gets a per-block scale, and each weight gets a 4-bit representation
    (3-bit codebook index + 1-bit sign).

    Args:
        state_dict: model state dict with float tensors
        codebook: 8-value codebook (default: [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        block_size: weights per scale factor

    Returns:
        Packed dict with keys:
            "{name}.packed": uint8 packed indices+signs
            "{name}.scales": float16 per-block scales
            "{name}.shape": original tensor shape
            "__codebook__": the codebook used
            "__block_size__": block size
    """
    if codebook is None:
        codebook = DEFAULT_CODEBOOK.clone()

    packed_state: dict[str, Any] = {
        "__codebook__": codebook,
        "__block_size__": block_size,
    }

    for name, param in state_dict.items():
        if not torch.is_floating_point(param):
            packed_state[name] = param.clone()
            continue

        p = param.detach().cpu().float().reshape(-1)
        n = p.shape[0]

        # Pad to multiple of block_size
        pad_len = (block_size - n % block_size) % block_size
        if pad_len > 0:
            p = torch.cat([p, torch.zeros(pad_len)])

        # Quantize each block
        all_packed = []
        all_scales = []
        for start in range(0, p.shape[0], block_size):
            block = p[start : start + block_size]
            indices, signs, scale = _quantize_block(block, codebook)
            packed = _pack_indices_signs(indices, signs)
            all_packed.append(packed)
            all_scales.append(scale)

        packed_state[f"{name}.packed"] = torch.cat(all_packed)
        packed_state[f"{name}.scales"] = torch.tensor(all_scales, dtype=torch.float16)
        packed_state[f"{name}.shape"] = list(param.shape)
        packed_state[f"{name}.numel"] = n  # original count before padding

    return packed_state


def dequantize_fp4(
    packed_state: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Decompress FP4 packed state dict back to float tensors.

    Args:
        packed_state: output from quantize_fp4()

    Returns:
        Standard state dict with float32 tensors
    """
    codebook = packed_state["__codebook__"]
    block_size = packed_state["__block_size__"]

    state_dict: dict[str, torch.Tensor] = {}
    seen: set[str] = set()

    for key in packed_state:
        if key.startswith("__") or key.endswith((".packed", ".scales", ".shape", ".numel")):
            continue
        # Non-quantized tensor (e.g., integer metadata)
        if not isinstance(packed_state[key], torch.Tensor) or not key.endswith((".packed", ".scales", ".shape", ".numel")):
            state_dict[key] = packed_state[key]

    for key in packed_state:
        if not key.endswith(".packed"):
            continue
        name = key[:-7]  # strip ".packed"
        if name in seen:
            continue
        seen.add(name)

        packed = packed_state[f"{name}.packed"]
        scales = packed_state[f"{name}.scales"].float()
        shape = packed_state[f"{name}.shape"]
        numel = packed_state[f"{name}.numel"]

        # Pad numel to block boundary
        padded_numel = numel + (block_size - numel % block_size) % block_size

        # Unpack all blocks
        indices, signs = _unpack_indices_signs(packed, padded_numel)

        # Dequantize block by block
        all_values = []
        for i, start in enumerate(range(0, padded_numel, block_size)):
            end = start + block_size
            block_values = _dequantize_block(
                indices[start:end], signs[start:end], scales[i], codebook,
            )
            all_values.append(block_values)

        flat = torch.cat(all_values)[:numel]
        state_dict[name] = flat.reshape(shape)

    return state_dict


# ── Fake quantization for QAT ──────────────────────────────────────────


class FakeQuantFP4(torch.autograd.Function):
    """Straight-through estimator for FP4 quantization.

    Forward: quantize to FP4 codebook, then dequantize (round-trip).
    Backward: pass gradient through unchanged (STE).

    This trains the model to be robust to FP4 quantization noise,
    similar to how FakeQuantSTE trains for int8 robustness.
    """

    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        codebook: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        original_shape = w.shape
        flat = w.detach().reshape(-1)
        n = flat.shape[0]

        # Pad to block boundary
        pad_len = (block_size - n % block_size) % block_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=w.device)])

        # Vectorized block-wise quantization: reshape to (num_blocks, block_size)
        blocks = flat.reshape(-1, block_size)  # (B, block_size)
        signs = blocks.sign()
        signs[signs == 0] = 1.0
        magnitudes = blocks.abs()

        # Per-block scales: (B,)
        max_mag = magnitudes.amax(dim=1)
        max_cb = codebook[-1]
        scales = max_mag / max_cb
        scales = scales.clamp(min=1e-10)

        # Normalize magnitudes to codebook range: (B, block_size)
        normalized = magnitudes / scales.unsqueeze(1)

        # Find nearest codebook entry via broadcasting: (B, block_size, 1) vs (1, 1, 8)
        cb = codebook.to(w.device)
        dists = (normalized.unsqueeze(2) - cb.reshape(1, 1, -1)).abs()
        indices = dists.argmin(dim=2)  # (B, block_size)

        # Dequantize: reconstruct from codebook values
        values = cb[indices]  # (B, block_size)
        result = (values * signs * scales.unsqueeze(1)).reshape(-1)

        return result[:n].reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # STE: pass gradient through unchanged
        return grad_out, None, None


def fake_quant_fp4(
    t: torch.Tensor,
    codebook: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Apply fake FP4 quantization with STE for training.

    Args:
        t: weight tensor to quantize
        codebook: 8-value codebook (default used if None)
        block_size: weights per scale group

    Returns:
        Fake-quantized tensor (same shape, quantization noise injected)
    """
    if codebook is None:
        codebook = DEFAULT_CODEBOOK.to(t.device)
    return FakeQuantFP4.apply(t, codebook, block_size)


# ── QAT Wrapper ─────────────────────────────────────────────────────────


class _FP4ForwardHook:
    """Forward pre-hook that applies FP4 fake quantization via STE.

    Uses the standard STE pattern: compute fake-quantized weight as a
    *function* of the original weight (not by replacing it). This keeps
    the computational graph connected so gradients flow to the original
    parameters and the optimizer's references remain valid.
    """

    def __init__(self, codebook: torch.Tensor, block_size: int):
        self.codebook = codebook
        self.block_size = block_size

    def __call__(self, module: nn.Module, inputs):
        # Apply FP4 STE: w_q = w + (quantize(w) - w).detach()
        # Forward uses quantized weights, backward flows through w
        w = module.weight
        w_q = fake_quant_fp4(w, self.codebook.to(w.device), self.block_size)
        module._fp4_orig_weight = w.data
        module.weight.data = w_q.data


class _FP4BackwardHook:
    """Forward hook (post) that restores original weights after forward pass.

    This ensures the optimizer always sees the original FP32 weights, while
    the forward pass uses FP4-quantized weights.
    """

    def __call__(self, module: nn.Module, inputs, output):
        if hasattr(module, '_fp4_orig_weight'):
            module.weight.data = module._fp4_orig_weight
            del module._fp4_orig_weight


class QATRendererFP4(nn.Module):
    """Wraps a PairGenerator with FP4 fake quantization during training.

    Uses forward pre/post hooks on Conv2d, ConvTranspose2d, and Embedding
    layers to inject fake FP4 quantization noise. The hooks modify
    weight.data in-place (not the Parameter object), so the optimizer's
    parameter references remain valid and gradients flow correctly via STE.

    Biases are left in full precision (negligible size contribution).
    """

    def __init__(
        self,
        base_model: nn.Module,
        codebook: torch.Tensor | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        super().__init__()
        self.base = base_model
        self.codebook = codebook if codebook is not None else DEFAULT_CODEBOOK.clone()
        self.block_size = block_size
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Attach FP4 STE hooks to all quantizable layers."""
        pre_hook = _FP4ForwardHook(self.codebook, self.block_size)
        post_hook = _FP4BackwardHook()
        for module in self.base.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)):
                h1 = module.register_forward_pre_hook(pre_hook)
                h2 = module.register_forward_hook(post_hook)
                self._hooks.extend([h1, h2])

    def remove_hooks(self):
        """Remove all FP4 hooks (for eval/export)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)


# ── Save / Load ─────────────────────────────────────────────────────────


def save_fp4(
    model: nn.Module,
    path: str | os.PathLike,
    *,
    meta: dict[str, Any] | None = None,
    codebook: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> int:
    """Save model weights in FP4 packed format.

    Expected size for a 300K-param model: ~195KB (vs ~390KB for int8).

    Args:
        model: the renderer/pair_generator module
        path: output .pt file path
        meta: optional metadata dict
        codebook: quantization codebook
        block_size: weights per scale group

    Returns:
        File size in bytes
    """
    packed = quantize_fp4(model.state_dict(), codebook, block_size)
    if meta is not None:
        packed["__meta__"] = dict(meta)
    torch.save(packed, path)
    size = os.path.getsize(path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[fp4] Saved {param_count:,} params to {path} ({size:,} bytes, "
          f"{size / param_count * 8:.2f} bits/param)")
    return size


def load_fp4(
    path: str | os.PathLike,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """Load FP4-packed weights into a model.

    Args:
        path: .pt file with packed FP4 weights
        model: target model (must match architecture)
        device: target device

    Returns:
        Model with loaded weights, in eval mode
    """
    packed = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = dequantize_fp4(packed)
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def get_fp4_meta(path: str | os.PathLike) -> dict[str, Any]:
    """Read metadata from an FP4 weight file without loading all weights."""
    packed = torch.load(path, map_location="cpu", weights_only=False)
    return dict(packed.get("__meta__", {}))
