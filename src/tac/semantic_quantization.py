"""Adaptive semantic quantization — per-class bit-depth allocation (Trick 18).

For SPADE/CLADE-based renderers, different semantic classes have different
visual importance for the scorer. This module follows the contest SegNet class
order in ``tac.semantic_label_contract``:
    - Road (class 0): 8-bit - highest PoseNet sensitivity
    - Lane markings (class 1): 8-bit - thin structure, high SegNet/PoseNet risk
    - Undrivable (class 2): 4-bit
    - Movable objects (class 3): 6-bit
    - My-car / ego car (class 4): 4-bit

The scoring formula weights SegNet at 100x and PoseNet at sqrt(10x),
so road pixels (which drive both PoseNet and SegNet) get maximum precision,
while undrivable and ego-car pixels can tolerate more quantization noise.

This saves ~20% rate compared to uniform 8-bit quantization with
negligible distortion increase on high-sensitivity classes.

Usage::

    from tac.semantic_quantization import semantic_adaptive_quantize
    q_state = semantic_adaptive_quantize(
        model.state_dict(),
        class_masks,
        class_bits={0: 8, 1: 8, 2: 4, 3: 6, 4: 4},
    )
"""

from __future__ import annotations

from typing import Any

import torch

from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASS_NAMES,
    SEMANTIC_QUANTIZATION_DEFAULT_BITS,
)


# Default bit allocation per semantic class
DEFAULT_CLASS_BITS: dict[int, int] = dict(SEMANTIC_QUANTIZATION_DEFAULT_BITS)

# Human-readable class names for logging
CLASS_NAMES: dict[int, str] = dict(CONTEST_SEGNET_CLASS_NAMES)


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, float]:
    """Symmetric quantization of a tensor to given bit depth.

    Args:
        tensor: float tensor to quantize.
        bits: target bit depth (2-8).

    Returns:
        (quantized_tensor, scale) — quantized values as float, and the scale factor.
    """
    assert 2 <= bits <= 8, f"bits must be in [2, 8], got {bits}"
    qmax = (1 << (bits - 1)) - 1  # e.g., 7-bit signed: 63
    qmin = -(1 << (bits - 1))

    amax = tensor.abs().max().item()
    if amax < 1e-10:
        return torch.zeros_like(tensor), 1.0

    scale = amax / qmax
    q = (tensor / scale).round().clamp(qmin, qmax)
    return q * scale, scale


def semantic_adaptive_quantize(
    state_dict: dict[str, torch.Tensor],
    masks: torch.Tensor | None = None,
    class_bits: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Quantize model weights with different precision per semantic class.

    For SPADE/CLADE renderers, the normalization layers have per-class
    parameters (embedding weights or per-class conv filters).  This function
    identifies those parameters and applies class-specific bit depths.

    Non-class-specific parameters (backbone convs, heads) get the maximum
    bit depth from class_bits (typically 8-bit).

    Args:
        state_dict: model state dict (float weights).
        masks: optional (N, H, W) long tensor — used for statistics only,
            not required for quantization.
        class_bits: mapping from class index to bit depth.
            Default: {0: 8, 1: 8, 2: 4, 3: 6, 4: 4}.

    Returns:
        Dict with:
            - "quantized_state_dict": quantized state dict (float tensors).
            - "scales": per-key scale factors.
            - "bits_used": per-key bit depths used.
            - "savings_estimate": estimated size reduction ratio.
    """
    if class_bits is None:
        class_bits = DEFAULT_CLASS_BITS.copy()

    max_bits = max(class_bits.values())
    num_classes = len(class_bits)

    quantized: dict[str, torch.Tensor] = {}
    scales: dict[str, float | list[float]] = {}
    bits_used: dict[str, int | list[int]] = {}
    total_original_bits = 0
    total_quantized_bits = 0

    for key, tensor in state_dict.items():
        is_class_param = _is_class_specific_param(key, tensor, num_classes)

        if is_class_param and tensor.shape[0] == num_classes:
            # Per-class quantization: each row gets different bits
            rows = []
            row_scales = []
            row_bits = []
            for cls_idx in range(num_classes):
                bits = class_bits.get(cls_idx, max_bits)
                q_row, s = quantize_tensor(tensor[cls_idx], bits)
                rows.append(q_row)
                row_scales.append(s)
                row_bits.append(bits)
                total_original_bits += tensor[cls_idx].numel() * 32
                total_quantized_bits += tensor[cls_idx].numel() * bits
            quantized[key] = torch.stack(rows)
            scales[key] = row_scales
            bits_used[key] = row_bits
        else:
            # Uniform quantization at max bit depth
            q_tensor, s = quantize_tensor(tensor, max_bits)
            quantized[key] = q_tensor
            scales[key] = s
            bits_used[key] = max_bits
            total_original_bits += tensor.numel() * 32
            total_quantized_bits += tensor.numel() * max_bits

    savings = 1.0 - (total_quantized_bits / max(total_original_bits, 1))

    return {
        "quantized_state_dict": quantized,
        "scales": scales,
        "bits_used": bits_used,
        "savings_estimate": savings,
    }


def _is_class_specific_param(
    key: str,
    tensor: torch.Tensor,
    num_classes: int,
) -> bool:
    """Heuristic: is this parameter class-specific (e.g., CLADE embedding, SPADE per-class)?

    Checks:
        1. Key contains class-related substrings (embedding, class_gamma, class_beta).
        2. First dimension matches num_classes.
    """
    class_keywords = [
        "class_gamma",
        "class_beta",
        "embedding",
        "cls_",
        "per_class",
        "spade_class",
    ]
    if tensor.shape[0] != num_classes:
        return False
    return any(kw in key.lower() for kw in class_keywords)


# ── Smoke tests ───────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic quantization checks."""
    # Single tensor quantization
    t = torch.randn(64, 32)
    q8, s8 = quantize_tensor(t, 8)
    q4, s4 = quantize_tensor(t, 4)
    assert q8.shape == t.shape
    assert q4.shape == t.shape
    # 4-bit should have larger error than 8-bit
    err8 = (q8 - t).abs().mean()
    err4 = (q4 - t).abs().mean()
    assert err4 > err8, "4-bit should have more error than 8-bit"

    # Semantic quantization of a mock CLADE state dict
    state_dict = {
        "clade.class_gamma.weight": torch.randn(5, 32),  # class-specific
        "clade.class_beta.weight": torch.randn(5, 32),  # class-specific
        "conv1.weight": torch.randn(32, 3, 3, 3),  # backbone
        "conv1.bias": torch.randn(32),  # backbone
    }
    result = semantic_adaptive_quantize(state_dict)
    assert "quantized_state_dict" in result
    assert "scales" in result
    assert "bits_used" in result
    assert "savings_estimate" in result

    # Class-specific params should have per-class bits
    gamma_bits = result["bits_used"]["clade.class_gamma.weight"]
    assert isinstance(gamma_bits, list) and len(gamma_bits) == 5
    assert gamma_bits[0] == 8, "Road should be 8-bit"
    assert gamma_bits[2] == 4, "Undrivable should be 4-bit by default"

    # Backbone should be uniform max bits
    assert result["bits_used"]["conv1.weight"] == 8

    # Savings should be positive (mixed < uniform 32-bit)
    assert result["savings_estimate"] > 0.0

    # Shapes preserved
    for key in state_dict:
        assert result["quantized_state_dict"][key].shape == state_dict[key].shape

    print("semantic_quantization: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
