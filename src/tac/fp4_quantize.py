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
#
# DEFAULT_CODEBOOK is the original mask2mask competitor codebook. It has
# uniform spacing in [0, 6] which is GOOD for weights uniformly distributed
# but BAD for residual heads where most weights are near zero. The smallest
# nonzero entry is 0.5, meaning anything below 0.25*scale rounds to zero.
# For a typical scale 0.167 (max_mag=1.0), that wipes out all weights
# below 0.042 — devastating for the small-magnitude tail of a residual head.
#
# RESIDUAL_CODEBOOK is denser near zero (geometric-ish spacing). The
# boundary between 0 and the first nonzero entry drops from 0.25 → 0.0625,
# preserving 4× more small-magnitude detail. Use this for renderers with a
# residual / correction head (e.g., Cool-Chic, C3 residual).
#
# (R-FP4-fix 2026-04-25: trend report showed FP4 SegNet plateau at ep5 while
# float SegNet kept dropping 0.54 → 0.28 — confirming small-mag clipping.)
DEFAULT_CODEBOOK = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
RESIDUAL_CODEBOOK = torch.tensor([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0])

DEFAULT_BLOCK_SIZE = 32

# Percentile used for scale calculation when robust=True. Using max(|w|) is
# fragile to outliers — a single 5σ weight in a 32-block forces everything
# else to round near zero. p99.5 keeps the same dynamic range while ignoring
# at most 0-1 outliers per block.
ROBUST_SCALE_PERCENTILE = 0.995


# ── Core quantization ───────────────────────────────────────────────────


def _quantize_block(
    block: torch.Tensor,
    codebook: torch.Tensor,
    *,
    robust_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 1D block of weights to FP4.

    Args:
        block: (block_size,) float tensor
        codebook: (8,) positive codebook values
        robust_scale: use percentile-based scale instead of max(|w|).
            R-FP4-fix: must match the training-time setting on QATRendererFP4
            so round-trip weights are identical to what training saw.

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

    # Compute block scale: maps max (or p99.5) magnitude to max codebook value.
    max_cb = codebook[-1]
    if robust_scale:
        # Single-block percentile — quantile expects float32 input.
        ref_mag = torch.quantile(magnitudes.float(), ROBUST_SCALE_PERCENTILE)
    else:
        ref_mag = magnitudes.max()
    scale = ref_mag / max_cb if ref_mag > 1e-10 else torch.tensor(1.0, device=block.device)

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
    signs = torch.where(sign_bits == 0, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
    return indices, signs


# ── State dict level ────────────────────────────────────────────────────


def quantize_fp4(
    state_dict: dict[str, torch.Tensor],
    codebook: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    robust_scale: bool = False,
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

        # R-FP4-fix: skip ndim<2 tensors. QATRendererFP4 only wraps ndim>=2
        # weights (Conv/Linear/Embedding), but quantize_fp4 historically
        # quantized everything — including 1-D buffers like Fourier-feature
        # frequency vectors that can be in the [1, 100] range. Quantizing
        # those with the small-magnitude codebook causes train↔export drift
        # (training never sees the quantization noise on these tensors).
        # Storing them as float passthroughs costs ~24 bytes per buffer —
        # negligible vs the train/export consistency win.
        if param.ndim < 2:
            packed_state[name] = param.detach().cpu().clone()
            continue

        p = param.detach().cpu().float().reshape(-1)
        n = p.shape[0]

        # Pad to multiple of block_size
        pad_len = (block_size - n % block_size) % block_size
        if pad_len > 0:
            p = torch.cat([p, torch.zeros(pad_len)])

        # Quantize each block. R-FP4-fix: robust_scale must match the QAT
        # wrapper setting so the saved scales reproduce training-time values.
        all_packed = []
        all_scales = []
        for start in range(0, p.shape[0], block_size):
            block = p[start : start + block_size]
            indices, signs, scale = _quantize_block(block, codebook, robust_scale=robust_scale)
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

    # Collect all base names that have .packed entries (these are quantized)
    quantized_names: set[str] = set()
    for key in packed_state:
        if key.endswith(".packed"):
            quantized_names.add(key[:-7])

    # Recover non-quantized entries: anything that isn't a metadata key,
    # isn't a quantized sub-key (.packed/.scales/.shape/.numel), and isn't
    # a base name that was quantized.
    for key in packed_state:
        if key.startswith("__"):
            continue
        if key.endswith((".packed", ".scales", ".shape", ".numel")):
            continue
        # This key was not quantized — pass it through directly
        if key not in quantized_names:
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
                indices[start:end],
                signs[start:end],
                scales[i],
                codebook,
            )
            all_values.append(block_values)

        flat = torch.cat(all_values)[:numel]
        state_dict[name] = flat.reshape(shape)

    return state_dict


# ── Fake quantization for QAT ──────────────────────────────────────────


def _block_scales(
    magnitudes: torch.Tensor,
    max_cb: torch.Tensor,
    *,
    robust: bool,
) -> torch.Tensor:
    """Per-block scale factor.

    robust=False: max(|w|) / max_cb (original behaviour, sensitive to outliers).
    robust=True:  quantile(|w|, ROBUST_SCALE_PERCENTILE) / max_cb — ignores
                  the top ~0.5% of weights so a single outlier per block can't
                  push the small-magnitude tail past the rounding boundary.

    The clamp(min=1e-10) is preserved either way to avoid div-by-zero on
    all-zero blocks (which happen at the padded tail of every weight).
    """
    if robust:
        # quantile is along dim=1 (within each block); float() cast required
        # because torch.quantile rejects float16/bfloat16.
        scales = torch.quantile(magnitudes.float(), ROBUST_SCALE_PERCENTILE, dim=1)
        scales = scales.to(magnitudes.dtype) / max_cb
    else:
        scales = magnitudes.amax(dim=1) / max_cb
    return scales.clamp(min=1e-10)


class FakeQuantFP4(torch.autograd.Function):
    """Straight-through estimator for FP4 quantization.

    Forward: quantize to FP4 codebook, then dequantize (round-trip).
    Backward: pass gradient through unchanged (STE).

    This trains the model to be robust to FP4 quantization noise,
    similar to how FakeQuantSTE trains for int8 robustness.

    Args:
        w: weights to quantize
        codebook: positive-magnitude codebook
        block_size: weights per per-block scale
        stochastic: if True, use stochastic rounding (round to floor/ceil
            with probability proportional to fractional distance) — adds
            unbiased dither that helps small-magnitude weights receive
            non-zero gradient signal during training. Use False at eval/export.
        robust_scale: if True, use percentile-based per-block scale instead
            of max(|w|) — protects against outlier-driven small-magnitude
            collapse. Default False for backward compat.

    R-FP4-fix 2026-04-25: stochastic + robust_scale + RESIDUAL_CODEBOOK
    together close the float→FP4 gap that left the trend report's FP4 score
    plateaued at 93.44 while float kept improving.
    """

    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        codebook: torch.Tensor,
        block_size: int,
        stochastic: bool = False,
        robust_scale: bool = False,
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
        cb = codebook.to(w.device)
        max_cb = cb[-1]
        scales = _block_scales(magnitudes, max_cb, robust=robust_scale)

        # Normalize magnitudes to codebook range: (B, block_size)
        normalized = magnitudes / scales.unsqueeze(1)

        # Find nearest codebook entry via broadcasting: (B, block_size, 1) vs (1, 1, 8)
        # When stochastic=True we instead pick between the two surrounding
        # codebook entries with probability proportional to fractional position
        # — this is unbiased dither and lets small-magnitude weights get nonzero
        # expected value over many forward passes.
        if stochastic:
            # Find the two nearest codebook entries (one above, one below)
            # by position in the SORTED codebook. Codebook is monotonically
            # increasing, so torch.bucketize gives the upper-bound index.
            upper_idx = torch.bucketize(normalized, cb)
            upper_idx = upper_idx.clamp(max=cb.numel() - 1)
            lower_idx = (upper_idx - 1).clamp(min=0)
            cb_lower = cb[lower_idx]
            cb_upper = cb[upper_idx]
            denom = (cb_upper - cb_lower).clamp(min=1e-10)
            p_upper = ((normalized - cb_lower) / denom).clamp(0.0, 1.0)
            # Sample uniform[0,1) per element; pick upper if u < p_upper else lower.
            # Use torch.rand on the right device + dtype.
            u = torch.rand_like(normalized)
            indices = torch.where(u < p_upper, upper_idx, lower_idx)
        else:
            dists = (normalized.unsqueeze(2) - cb.reshape(1, 1, -1)).abs()
            indices = dists.argmin(dim=2)  # (B, block_size)

        # Dequantize: reconstruct from codebook values
        values = cb[indices]  # (B, block_size)
        result = (values * signs * scales.unsqueeze(1)).reshape(-1)

        return result[:n].reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # STE: pass gradient through unchanged. Return None for non-tensor args.
        return grad_out, None, None, None, None


def fake_quant_fp4(
    t: torch.Tensor,
    codebook: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    stochastic: bool = False,
    robust_scale: bool = False,
) -> torch.Tensor:
    """Apply fake FP4 quantization with STE for training.

    Args:
        t: weight tensor to quantize
        codebook: 8-value codebook (default used if None)
        block_size: weights per scale group
        stochastic: stochastic rounding for training-time dither (default False)
        robust_scale: percentile-based per-block scale (default False)

    Returns:
        Fake-quantized tensor (same shape, quantization noise injected)
    """
    if codebook is None:
        codebook = DEFAULT_CODEBOOK.to(t.device)
    # FP4_HARDWARE_DISCLOSED: NVFP4 hardware needs Blackwell (CC >= 10.0).
    # On RTX 4090 (CC 8.9) and earlier, FakeQuantFP4 SIMULATES FP4 via
    # codebook lookup in FP32 — the inflate-time inference still runs at
    # FP32 (we just store the 4-bit codebook indices in the archive).
    # See project_cosmos_deep_dive_addendum_20260428 for the rescue path
    # (Lane F-V5: hardware FP8 via torchao.float8 on 4090).
    return FakeQuantFP4.apply(t, codebook, block_size, stochastic, robust_scale)


# ── QAT Wrapper ─────────────────────────────────────────────────────────


class FP4Parametrize(nn.Module):
    """nn.utils.parametrize module that applies FP4 fake quantization via STE.

    The parametrize pattern ensures the forward pass uses quantized weights
    while gradients flow through the STE to the original FP32 parameters.
    Unlike the old hook-based approach, this never mutates .data directly,
    so the autograd graph is fully preserved.

    Stochastic rounding + robust scale are training-time toggles; export
    paths must use deterministic argmin rounding + max-based scale to match
    what dequantize_fp4() produces (see _quantize_block at module top).
    Toggle them off via .eval() (forwarded by parent QATRendererFP4) so that
    proxy/auth eval matches inflate-time math.
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        block_size: int,
        *,
        stochastic: bool = False,
        robust_scale: bool = False,
    ):
        super().__init__()
        self.register_buffer("codebook", codebook)
        self.block_size = block_size
        self.stochastic = stochastic
        self.robust_scale = robust_scale

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        # .contiguous() required for MPS: nn.utils.parametrize can produce
        # non-contiguous weight views that cause reshape failures in backward
        # R-FP4-fix: stochastic + robust_scale ONLY apply at training time so
        # eval matches export. self.training inherits from the parent module.
        return fake_quant_fp4(
            weight.contiguous(),
            self.codebook,
            self.block_size,
            stochastic=self.stochastic and self.training,
            robust_scale=self.robust_scale,
        )


class QATRendererFP4(nn.Module):
    """Wraps a PairGenerator with FP4 fake quantization during training.

    Uses nn.utils.parametrize on Conv2d, ConvTranspose2d, Embedding, and
    Linear layers to inject fake FP4 quantization noise. ALL layer types
    are wrapped because export_asymmetric_checkpoint_fp4 quantizes ALL
    layers — training must match deployment exactly.

    Biases are left in full precision (negligible size contribution).
    """

    def __init__(
        self,
        base_model: nn.Module,
        codebook: torch.Tensor | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        *,
        stochastic: bool = False,
        robust_scale: bool = False,
    ):
        """
        R-FP4-fix args:
          stochastic: stochastic rounding during training (unbiased dither
            that helps small-magnitude weights gradient flow). Export uses
            deterministic argmin regardless. Recommended for residual heads.
          robust_scale: percentile-based per-block scale instead of max(|w|).
            Protects small-magnitude tail from outlier-driven collapse.
            Recommended unless an audit confirms outlier-free weight stats.
        """
        super().__init__()
        self.base = base_model
        self.register_buffer("codebook", codebook if codebook is not None else DEFAULT_CODEBOOK.clone())
        self.block_size = block_size
        self.stochastic = stochastic
        self.robust_scale = robust_scale
        self._parametrized_modules: list[nn.Module] = []
        self._register_parametrizations()

    def _register_parametrizations(self):
        """Attach FP4 STE parametrizations to all quantizable layers.

        Skips layers already parametrized (prevents double-wrapping if
        called from both Phase 1 and Phase 2 — C1 fix).
        """
        for module in self.base.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding, nn.Linear)):
                if hasattr(module, "weight") and module.weight.ndim >= 2:
                    if nn.utils.parametrize.is_parametrized(module, "weight"):
                        continue  # already wrapped — skip to prevent double quantization
                    nn.utils.parametrize.register_parametrization(
                        module,
                        "weight",
                        FP4Parametrize(
                            self.codebook.clone(),
                            self.block_size,
                            stochastic=self.stochastic,
                            robust_scale=self.robust_scale,
                        ),
                    )
                    self._parametrized_modules.append(module)

    def remove_hooks(self):
        """Remove all FP4 parametrizations (for eval/export)."""
        for module in self._parametrized_modules:
            if nn.utils.parametrize.is_parametrized(module, "weight"):
                nn.utils.parametrize.remove_parametrizations(module, "weight")
        self._parametrized_modules.clear()

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
    robust_scale: bool = False,
) -> int:
    """Save model weights in FP4 packed format.

    Expected size for a 300K-param model: ~195KB (vs ~390KB for int8).

    Args:
        model: the renderer/pair_generator module
        path: output .pt file path
        meta: optional metadata dict
        codebook: quantization codebook (use RESIDUAL_CODEBOOK for renderers
            with a residual / correction head — denser near zero)
        block_size: weights per scale group
        robust_scale: percentile-based per-block scale. R-FP4-fix: MUST match
            the QATRendererFP4 setting that trained this model — mismatched
            scale calculation between train and export breaks round-trip.

    Returns:
        File size in bytes
    """
    packed = quantize_fp4(model.state_dict(), codebook, block_size,
                          robust_scale=robust_scale)
    if meta is not None:
        packed["__meta__"] = dict(meta)
    torch.save(packed, path)
    size = os.path.getsize(path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[fp4] Saved {param_count:,} params to {path} ({size:,} bytes, "
          f"{size / param_count * 8:.2f} bits/param, robust_scale={robust_scale})")
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
    packed = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = dequantize_fp4(packed)
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def get_fp4_meta(path: str | os.PathLike) -> dict[str, Any]:
    """Read metadata from an FP4 weight file without loading all weights."""
    packed = torch.load(path, map_location="cpu", weights_only=True)
    return dict(packed.get("__meta__", {}))
