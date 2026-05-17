# SPDX-License-Identifier: MIT
"""Magnitude-pruning + quantization composition — academic + SparseGPT pattern.

The canonical pattern (per SparseGPT 2023, the Lottery Ticket Hypothesis,
and NVIDIA's 2:4 structured sparsity for sparse tensor cores):

1. **Prune** the smallest-magnitude weights to a target sparsity (50% =
   2:4 structured for NVIDIA hardware; 75% = unstructured).
2. **Quantize** the remaining weights to a target bit-width.

Composing pruning with quantization yields multiplicative bit-budget
savings:
    int4 + 50% sparse ≈ 2 bits/param effective
    int8 + 75% sparse ≈ 2 bits/param effective

Per CLAUDE.md "Bit-level deconstruction and entropy discipline":
the sparsity + quantization composition is the canonical first-pass
archive-byte-budget primitive. PR101's per-tensor byte-map encoding
already exploits implicit sparsity via the byte-map; this module makes
the composition explicit.

[verified-against:SparseGPT paper (Frantar 2023) + NVIDIA 2:4
structured sparsity white paper + Lottery Ticket Hypothesis (Frankle
2019)]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseQuantEncoded:
    """Sparse + quantized weight blob."""
    nonzero_positions: torch.Tensor  # uint32 positions of non-zero weights
    nonzero_values_encoded: object  # FP4 / NF4 / INT4 encoded values
    n_elements: int
    original_shape: tuple[int, ...]
    sparsity: float


def magnitude_prune_then_quantize(
    weight: torch.Tensor,
    *,
    sparsity: float = 0.5,
    quant_bits: int = 4,
    structured_2_4: bool = False,
) -> SparseQuantEncoded:
    """Prune + quantize a weight tensor.

    Args:
        weight: input tensor
        sparsity: fraction of weights to prune (0.5 = 50% pruned)
        quant_bits: bit-width for retained weights (4 or 8)
        structured_2_4: if True use 2:4 structured sparsity (every group
            of 4 consecutive weights keeps the 2 largest by magnitude).
            Required for NVIDIA Ampere+ sparse tensor cores.

    Wire size estimate:
        - structured_2_4 + 4-bit: 2 bits/param + ~1 bit/param position mask
          = ~3 bits/param effective (vs FP4's 4 bits/param)
        - unstructured 75% + 4-bit: 1 bit/param + uint32 positions for
          25% retained = ~9 bits/param for retained * 0.25 + ~position
          overhead

    For NVIDIA Ampere+ deployment, prefer structured_2_4. For pure
    archive byte savings, prefer unstructured at higher sparsity.

    [verified-against:SparseGPT + NVIDIA 2:4 white paper]
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1); got {sparsity}")
    if quant_bits not in (4, 8):
        raise ValueError(f"quant_bits must be 4 or 8; got {quant_bits}")
    flat = weight.detach().contiguous().reshape(-1).float()
    n = flat.numel()
    if structured_2_4:
        # Every group of 4 elements: keep the 2 largest by magnitude.
        n_groups = (n + 3) // 4
        pad = n_groups * 4 - n
        if pad:
            flat_padded = torch.cat([flat, torch.zeros(pad, dtype=torch.float32)])
        else:
            flat_padded = flat
        groups = flat_padded.reshape(n_groups, 4)
        # Per-group: keep the 2 largest by abs
        abs_groups = groups.abs()
        # Sort indices descending; keep top 2
        _, top_idx = abs_groups.topk(2, dim=1)
        mask = torch.zeros_like(groups, dtype=torch.bool)
        for j in range(2):
            mask.scatter_(1, top_idx[:, j : j + 1], True)
        pruned_flat = (groups * mask).reshape(-1)[:n]
    else:
        # Unstructured magnitude pruning
        n_keep = int(n * (1 - sparsity))
        n_keep = max(n_keep, 1)
        threshold = flat.abs().sort(descending=True).values[n_keep - 1].item()
        mask = flat.abs() >= threshold
        pruned_flat = flat * mask
    # Find non-zero positions + values
    nonzero_mask = pruned_flat != 0
    positions = nonzero_mask.nonzero().squeeze(-1).to(torch.int32)
    values = pruned_flat[nonzero_mask]
    # Quantize the retained values
    if quant_bits == 4:
        from tac.quantization_wave.int4_int8_mixed_bit import encode_int4_groupwise
        encoded_values = encode_int4_groupwise(values, group_size=min(64, max(values.numel(), 1)), use_nf4=True)
    else:  # 8-bit
        scale = values.abs().max() / 127.0
        scale = max(scale.item(), 1e-10)
        q = (values / scale).round().clamp(-128, 127).to(torch.int8)
        encoded_values = {"int8_per_tensor": (q, scale)}
    return SparseQuantEncoded(
        nonzero_positions=positions,
        nonzero_values_encoded=encoded_values,
        n_elements=n,
        original_shape=tuple(weight.shape),
        sparsity=float(1.0 - len(positions) / max(n, 1)),
    )


class SparseQuantComposition:
    """Composer that applies prune + quantize to all Linear/Conv2d layers.

    Per the T4 SYMPOSIUM verdict's Carmack voice ("entropy-coding density
    is the winning move"), pure pruning without entropy coding does NOT
    minimize archive bytes — the position-mask overhead can exceed the
    bit savings from pruning. The canonical pattern is:

        prune → quantize → entropy-code the position mask

    The entropy coding is handled by ``entropy_coding_archive_primitives``;
    this module produces the input to that pipeline.
    """

    def __init__(
        self,
        *,
        sparsity: float = 0.5,
        quant_bits: int = 4,
        structured_2_4: bool = False,
    ):
        self.sparsity = sparsity
        self.quant_bits = quant_bits
        self.structured_2_4 = structured_2_4
        self.encodings: dict[str, SparseQuantEncoded] = {}

    def encode_all(self, named_weights: dict[str, torch.Tensor]) -> dict[str, SparseQuantEncoded]:
        for name, w in named_weights.items():
            self.encodings[name] = magnitude_prune_then_quantize(
                w,
                sparsity=self.sparsity,
                quant_bits=self.quant_bits,
                structured_2_4=self.structured_2_4,
            )
        return self.encodings
