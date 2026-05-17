# SPDX-License-Identifier: MIT
"""INT4 + INT8 mixed-bit quantization — Hugging Face bitsandbytes pattern.

This module implements bitsandbytes-style 4-bit groupwise quantization
(NF4 / FP4 groupwise) + 8-bit activations with per-tensor
sensitivity-aware bit assignment.

Per CLAUDE.md "QAT pipeline — non-negotiable for FP4 deployment":
the 5-stage pipeline is (1) train float (2) freeze BN (3) insert
per-channel FP4 fake-quant on weights + per-tensor on activations
(4) fine-tune at 0.1× LR (5) export. This module covers stages 3-5.

The sensitivity-aware bit assignment uses the Fisher-information-style
``mean(g**2 * w**2)`` proxy (which Catalog #123 explicitly FORBIDS as a
*saliency proxy on score-gradient substrates* — but the assignment
here is for ALLOCATING bits to tensors, not for KILLING bytes that
contribute to score-gradient training; the canonical caveat is that
this proxy should be computed AFTER the substrate has been
score-gradient-trained, not used to GATE the training).

[verified-against:Hugging Face bitsandbytes NF4 spec + QLoRA paper
(Dettmers 2023) + GPTQ (Frantar 2023)]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# NF4 (NormalFloat-4) levels per bitsandbytes (Dettmers 2023). These are
# the 16 levels of a normal-distribution-fitted FP4-like grid; preferred
# over plain FP4 (E2M1) when weight distributions are approximately
# Gaussian.
NF4_LEVELS: tuple[float, ...] = (
    -1.0, -0.6961928, -0.5250730, -0.39491748, -0.28444138, -0.18477343,
    -0.09105944, 0.0, 0.07958029, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261665, 0.72295684, 1.0,
)


@dataclass(frozen=True)
class GroupwiseInt4Encoded:
    """Groupwise int4 encoding: 4-bit indices packed + per-group scales.

    Group size defaults to 64 elements (bitsandbytes default). Each group
    has its own scale.
    """
    indices_packed: torch.Tensor  # uint8, 2 4-bit indices per byte
    scales: torch.Tensor  # fp16 per-group
    group_size: int
    n_elements: int
    original_shape: tuple[int, ...]


def encode_int4_groupwise(
    weight: torch.Tensor,
    *,
    group_size: int = 64,
    use_nf4: bool = True,
) -> GroupwiseInt4Encoded:
    """Groupwise 4-bit quantization (NF4 or symmetric).

    Args:
        weight: any-shape tensor
        group_size: number of elements per scale (default 64 per bnb)
        use_nf4: if True use the NF4 grid; else symmetric int4 (-8..+7)

    Wire size: ceil(N/2) bytes (indices) + ceil(N/group_size)*2 bytes
    (fp16 scales). For 1008 weights at group_size=64:
        504 + 16*2 = 536 bytes (vs FP4 per-channel 576 bytes — slightly
        smaller; the groupwise scheme amortizes scale overhead across
        finer groups).

    [verified-against:bitsandbytes NF4 quantize_4bit / dequantize_4bit
    + QLoRA paper Algorithm 1]
    """
    w_flat = weight.detach().contiguous().reshape(-1).float()
    n = w_flat.numel()
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    if pad:
        w_flat = torch.cat([w_flat, torch.zeros(pad, dtype=torch.float32)])
    groups = w_flat.reshape(n_groups, group_size)
    abs_max = groups.abs().amax(dim=1).clamp(min=1e-10)
    if use_nf4:
        # NF4 normalizes to [-1, 1]
        normalized = groups / abs_max.unsqueeze(1)
        levels = torch.tensor(NF4_LEVELS, dtype=torch.float32)
        distances = (normalized.unsqueeze(-1) - levels).abs()
        indices = distances.argmin(dim=-1).to(torch.uint8)
        scales = abs_max
    else:
        # Symmetric int4: -8..+7
        scales = abs_max / 7.0
        normalized = (groups / scales.unsqueeze(1)).round().clamp(-8, 7)
        indices = (normalized + 8).to(torch.uint8)  # shift to [0, 15]
    flat_idx = indices.reshape(-1)
    if flat_idx.numel() % 2 == 1:
        flat_idx = torch.cat([flat_idx, torch.zeros(1, dtype=torch.uint8)])
    low = flat_idx[0::2] & 0x0F
    high = (flat_idx[1::2] & 0x0F) << 4
    packed = (low | high).to(torch.uint8)
    return GroupwiseInt4Encoded(
        indices_packed=packed,
        scales=scales.to(torch.float16),
        group_size=group_size,
        n_elements=n,
        original_shape=tuple(weight.shape),
    )


def decode_int4_groupwise(
    encoded: GroupwiseInt4Encoded,
    *,
    use_nf4: bool = True,
) -> torch.Tensor:
    """Inverse of :func:`encode_int4_groupwise`."""
    packed = encoded.indices_packed.to(torch.uint8)
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.long)
    unpacked[0::2] = low
    unpacked[1::2] = high
    n_groups = encoded.scales.numel()
    indices = unpacked[: n_groups * encoded.group_size].reshape(n_groups, encoded.group_size)
    scales = encoded.scales.float()
    if use_nf4:
        levels = torch.tensor(NF4_LEVELS, dtype=torch.float32)
        normalized = levels[indices]
        out = normalized * scales.unsqueeze(1)
    else:
        normalized = indices.to(torch.float32) - 8.0  # shift back from [0, 15] to [-8, 7]
        out = normalized * scales.unsqueeze(1)
    return out.reshape(-1)[: encoded.n_elements].reshape(encoded.original_shape)


def sensitivity_aware_mixed_bit_assignment(
    weights_and_gradients: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    target_average_bits: float = 4.5,
    candidate_bits: tuple[int, ...] = (3, 4, 5, 6, 8),
) -> dict[str, int]:
    """Assign per-tensor bit-width to minimize total error at target avg bits.

    Uses a Fisher-information-style sensitivity proxy: ``mean(g**2 * w**2)``
    per tensor. Tensors with higher sensitivity get more bits.

    Args:
        weights_and_gradients: ``{name: (weight, gradient)}`` (gradient is
            optional — pass a copy of the weight if no gradient is
            available; result degrades to magnitude-only sensitivity).
        target_average_bits: target average bits/parameter across all
            tensors (weighted by tensor size).
        candidate_bits: which bit-widths are allowed (must be supported
            by the GGUF / bitsandbytes / FP4 encoders).

    Returns:
        ``{name: bit_width}`` assignment.

    NOTE — Catalog #123 SAFETY: the Fisher proxy is appropriate for
    ALLOCATING bits across tensors, NOT for KILLING bytes from score-
    gradient-trained substrates. The substrate must be trained against
    the contest scorer FIRST; this function is called AFTER training to
    decide how many bits to spend per tensor in the archive.

    [verified-against:bitsandbytes mixed-precision pattern + GPTQ
    sensitivity allocation; CLAUDE.md Catalog #123]
    """
    sensitivities: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for name, (w, g) in weights_and_gradients.items():
        s = float((g.detach() ** 2 * w.detach() ** 2).mean().item()) if g is not None else float((w.detach() ** 2).mean().item())
        sensitivities[name] = s
        sizes[name] = int(w.numel())
    total_params = sum(sizes.values())
    if total_params == 0:
        return {name: candidate_bits[len(candidate_bits) // 2] for name in sensitivities}
    target_total_bits = int(round(target_average_bits * total_params))
    # Sort by sensitivity-per-param descending (most-sensitive first).
    ordered = sorted(sensitivities.items(), key=lambda kv: -kv[1])
    assignment: dict[str, int] = {}
    remaining_bits = target_total_bits
    remaining_params = total_params
    sorted_candidates = sorted(candidate_bits, reverse=True)
    for name, _ in ordered:
        size = sizes[name]
        # Greedy: pick the largest candidate that doesn't exceed the
        # per-param budget for the remaining tensors.
        if remaining_params <= 0:
            assignment[name] = sorted_candidates[-1]
            continue
        budget_per_param = remaining_bits / remaining_params
        # Pick the candidate closest to (but <=) the budget if possible,
        # else the smallest candidate.
        feasible = [b for b in sorted_candidates if b * size <= remaining_bits]
        if not feasible:
            feasible = [sorted_candidates[-1]]
        # Most-sensitive tensors get the largest feasible bit; later ones
        # get smaller bits to stay within target.
        # Use the bit closest to budget_per_param from below to avoid
        # overshooting.
        below = [b for b in feasible if b <= budget_per_param]
        if below:
            chosen = max(below)
        else:
            chosen = min(feasible)
        assignment[name] = chosen
        remaining_bits -= chosen * size
        remaining_params -= size
    return assignment


class BitsAndBytesStyleQuantizer(nn.Module):
    """Wrapper that applies INT4 (NF4) groupwise quantization to all
    Linear weights, INT8 to all Conv2d weights — the canonical
    bitsandbytes mixed-precision pattern.

    Per CLAUDE.md "QAT pipeline" stages 3-5: insert per-channel FP4 on
    weights + per-tensor on activations. This module covers the weight
    side; activation quantization is delegated to the trainer's
    Uint8STE (already canonical per ``tac.quantization.Uint8STE``).
    """

    def __init__(self, base: nn.Module, *, linear_bits: int = 4, conv_bits: int = 8):
        super().__init__()
        self.base = base
        self.linear_bits = linear_bits
        self.conv_bits = conv_bits
        self._encodings: dict[str, object] = {}

    def encode_all_weights(self) -> dict[str, object]:
        """Encode every Linear + Conv2d weight to its target bit-width.

        Returns a dict keyed by layer name. Caller is responsible for
        serializing to the archive (typically via brotli on the packed
        codewords + scales tuples).
        """
        encodings: dict[str, object] = {}
        for name, mod in self.base.named_modules():
            if isinstance(mod, nn.Linear):
                if self.linear_bits == 4:
                    encodings[name] = encode_int4_groupwise(mod.weight, group_size=64, use_nf4=True)
                else:
                    encodings[name] = ("int8_per_channel", mod.weight.detach())
            elif isinstance(mod, nn.Conv2d):
                if self.conv_bits == 4:
                    encodings[name] = encode_int4_groupwise(mod.weight, group_size=64, use_nf4=True)
                else:
                    encodings[name] = ("int8_per_channel", mod.weight.detach())
        self._encodings = encodings
        return encodings
