# SPDX-License-Identifier: MIT
"""Z3-G1 scorer-class-conditional gating architecture.

The G1 substitution is a 1KB-budget cooperative-receiver move: REPLACE the
Ballé hyperprior MLP (the per-pair scale predictor `h_a` + `h_s`) with a
LEARNABLE per-class sigma TABLE indexed by SegNet's per-pair dominant class.

Key insight (Wunderkind G1):

    Z3 v2 hyperprior:  y_p (28-dim) -> h_a -> w_hat_p (8-dim) -> h_s -> sigma_p
                       requires ~1KB MLP weights + 600 * 8 = 4800B w_hat
    G1 gating:         class_p (1-dim, in [0,4]) -> sigma_table[class_p] -> sigma_p
                       requires 5 * 28 * 1B = 140B int8 table + 600 * 1B = 600B
                       class indices

For SegNet's 5-class output on driving scenes (road / car / building / sky /
foreground), the class label is HIGHLY correlated with the per-pair latent
residual statistics (sky pairs have low variance, foreground pairs have high
variance). The class-conditional sigma table is the BARE MINIMUM hyperprior:
no learned MLP, no continuous side-info, just a 5x28 lookup table.

Compute breakdown:

    - sigma_table: nn.Parameter(5, 28) float32 = 560B at fp32, ~140B int8
    - per-pair class index: precomputed once per training run via
      `g1_per_pair_dominant_class_from_segnet_argmax(segnet_argmax_per_pair)`
      where input is the SegNet argmax tensor (B, H, W) per GT frame and
      output is the per-pair mode int in [0, 4].

NO score claim. NO promotion. NO exact-eval dispatch from this module.

Bolt-on LOC budget: <= 350 LOC per HNeRV parity discipline L7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse A1 substrate constants (single source of truth).
from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    _round_ste,
    conditional_gaussian_rate_bits,
    factorized_uniform_rate_bits,
)

# SegNet outputs 5-class logits per pixel (upstream/modules.py: smp.Unet
# 'tu-efficientnet_b2', classes=5). The 5 classes are road/car/building/sky/
# foreground per the comma2k19 segmentation labeling.
G1_NUM_SCORER_CLASSES = 5


@dataclass(frozen=True)
class Z3G1Config:
    """Static design-time parameters for Z3-G1 scorer-class gating.

    All fields required-keyword (no silent defaults beyond explicit ones).
    """

    num_scorer_classes: int = G1_NUM_SCORER_CLASSES
    """Number of SegNet classes (5 per upstream/modules.py)."""

    int8_sigma_scale: float = 16.0
    """Scale factor for int8 quantization of sigma_table:
    sigma_int8 = round(sigma_real * 127 / int8_sigma_scale), reconstructed as
    sigma_real = sigma_int8 * int8_sigma_scale / 127. Default chosen so
    sigma in [0, 16] maps to int8 in [0, 127]."""

    min_sigma: float = 1e-3
    """Lower clamp for sigma outputs to avoid div-by-zero in AC coder."""

    max_sigma: float = 16.0
    """Upper clamp for sigma outputs (codable range bound)."""

    quantization_step: float = 1.0
    """Quantization step Δ for the residual coder (must match Z3 v2 grammar)."""

    factorized_half_range: float = 16.0
    """Half-range for the class-index factorized prior (small uniform; 5
    classes occupy < 3 bits per pair). The class index itself is shipped
    as a single int8 per pair, not coded under this prior; this field is
    retained for grammar compatibility with Z3 v2."""

    init_sigma: float = 2.0
    """Initial sigma value for the table (uniform across classes/dims).
    The training loop learns class-specific sigma values from data."""


class Z3G1ScorerClassGatingHead(nn.Module):
    """Tiny per-class sigma TABLE indexed by SegNet's dominant class.

    Inputs:
        class_indices: (N_pairs,) long tensor in [0, num_scorer_classes)
            giving the per-pair dominant SegNet class.

    Outputs:
        sigma: (N_pairs, A1_LATENT_DIM) per-dim scale > 0 used by the
            arithmetic coder's conditional Gaussian prior.

    The "head" is a single 5x28 nn.Parameter; the forward pass is a single
    F.embedding lookup. Total param count = 5 * 28 = 140 params.
    """

    def __init__(self, config: Z3G1Config | None = None) -> None:
        super().__init__()
        cfg = config or Z3G1Config()
        if cfg.num_scorer_classes <= 0 or cfg.num_scorer_classes > 255:
            raise ValueError(
                "num_scorer_classes must be in [1, 255]; got "
                f"{cfg.num_scorer_classes}"
            )
        self.config = cfg
        # Initialize: each class starts with the same flat sigma; training
        # learns class-specific values. Use softplus-pre-image so the
        # forward-pass softplus + clamp produces ``init_sigma`` initially.
        init_pre_softplus = math.log(math.expm1(cfg.init_sigma))
        self.sigma_logits = nn.Parameter(
            torch.full(
                (cfg.num_scorer_classes, A1_LATENT_DIM),
                init_pre_softplus,
                dtype=torch.float32,
            )
        )

    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Lookup per-class sigma for each pair.

        Args:
            class_indices: (N_pairs,) long tensor in
                [0, num_scorer_classes).

        Returns:
            sigma: (N_pairs, A1_LATENT_DIM) per-dim scale > 0.
        """
        if class_indices.dim() != 1:
            raise ValueError(
                f"class_indices must be 1D; got shape {tuple(class_indices.shape)}"
            )
        if class_indices.dtype not in (torch.int64, torch.int32, torch.uint8):
            raise ValueError(
                f"class_indices must be integer dtype; got {class_indices.dtype}"
            )
        # Bounds check (no silent clamp; explicit ValueError instead).
        cmax = int(class_indices.max().item()) if class_indices.numel() > 0 else 0
        cmin = int(class_indices.min().item()) if class_indices.numel() > 0 else 0
        if cmin < 0 or cmax >= self.config.num_scorer_classes:
            raise ValueError(
                f"class_indices must be in [0, {self.config.num_scorer_classes}); "
                f"got range [{cmin}, {cmax}]"
            )
        # Embedding lookup: sigma_logits has shape (C, D); class_indices
        # has shape (N,); result has shape (N, D).
        sigma_logits_per_pair = F.embedding(
            class_indices.to(torch.long), self.sigma_logits
        )
        sigma = F.softplus(sigma_logits_per_pair) + self.config.min_sigma
        sigma = sigma.clamp(
            min=self.config.min_sigma, max=self.config.max_sigma
        )
        return sigma

    def quantize_sigma_table_int8(self) -> tuple[torch.Tensor, float]:
        """Quantize the learned sigma table to int8 for archive shipping.

        Returns ``(sigma_int8, scale_int8)`` where
        ``sigma_real ~ sigma_int8.float() * scale_int8 / 127``.
        """
        with torch.no_grad():
            # Apply forward-pass transform to get real sigma values.
            sigma_real = F.softplus(self.sigma_logits) + self.config.min_sigma
            sigma_real = sigma_real.clamp(
                min=self.config.min_sigma, max=self.config.max_sigma
            )
            scale = float(self.config.int8_sigma_scale)
            sigma_int8 = (sigma_real * 127.0 / scale).round().clamp(
                min=0.0, max=127.0
            ).to(torch.int8)
        return sigma_int8, scale


def dequantize_sigma_table_int8(
    sigma_int8: torch.Tensor, scale_int8: float, *, min_sigma: float = 1e-3
) -> torch.Tensor:
    """Reconstruct sigma table from int8 bytes shipped in archive.

    Args:
        sigma_int8: (C, D) int8 tensor.
        scale_int8: positive float (the value matched at encode-time).
        min_sigma: lower clamp to avoid div-by-zero in AC coder.
    """
    if scale_int8 <= 0:
        raise ValueError(f"scale_int8 must be positive; got {scale_int8}")
    sigma_real = sigma_int8.to(torch.float32) * scale_int8 / 127.0
    return sigma_real.clamp(min=min_sigma)


def g1_per_pair_dominant_class_from_segnet_argmax(
    segnet_argmax_per_pair: torch.Tensor,
    *,
    num_classes: int = G1_NUM_SCORER_CLASSES,
) -> torch.Tensor:
    """Reduce a per-pixel SegNet argmax map to a per-pair dominant class.

    Args:
        segnet_argmax_per_pair: ``(N_pairs, H, W)`` int tensor in
            [0, num_classes) giving SegNet's argmax label per pixel for
            each pair (e.g., the GT frame 0 or frame 1 of each pair).
        num_classes: Number of SegNet classes (default 5).

    Returns:
        ``(N_pairs,)`` long tensor with the mode (most frequent) class per
        pair. Ties broken by lowest class id.
    """
    if segnet_argmax_per_pair.dim() != 3:
        raise ValueError(
            f"segnet_argmax_per_pair must be (N, H, W); got "
            f"{tuple(segnet_argmax_per_pair.shape)}"
        )
    n_pairs = segnet_argmax_per_pair.shape[0]
    flat = segnet_argmax_per_pair.reshape(n_pairs, -1).to(torch.long)
    # Per-pair histogram via one-hot sum.
    one_hot = F.one_hot(flat, num_classes=num_classes)  # (N, H*W, C)
    histogram = one_hot.sum(dim=1)  # (N, C)
    # argmax returns lowest index on ties (deterministic).
    return histogram.argmax(dim=1)


def g1_total_rate_bits(
    residual: torch.Tensor,
    sigma: torch.Tensor,
    class_indices: torch.Tensor,
    *,
    num_classes: int = G1_NUM_SCORER_CLASSES,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
) -> torch.Tensor:
    """Total G1 rate in BITS per sample.

    R_total = R_residual + R_class_index
            = -log p_y(residual | sigma_table[class]) - log p_z(class)

    where p_z is a factorized uniform prior over [0, num_classes-1] (which
    is small enough — log2(5) ≈ 2.32 bits/pair).

    Returns (N,) total bits per sample.
    """
    rate_residual = conditional_gaussian_rate_bits(
        residual, sigma, quantization_step=quantization_step
    )
    # Each pair pays log2(num_classes) bits for its class index. Use a
    # factorized uniform prior since we ship raw class indices (no AC).
    n = class_indices.shape[0]
    rate_class = torch.full(
        (n,),
        math.log2(float(num_classes)),
        device=residual.device,
        dtype=residual.dtype,
    )
    return rate_residual + rate_class


__all__ = [
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "G1_NUM_SCORER_CLASSES",
    "Z3G1Config",
    "Z3G1ScorerClassGatingHead",
    "_round_ste",
    "conditional_gaussian_rate_bits",
    "dequantize_sigma_table_int8",
    "factorized_uniform_rate_bits",
    "g1_per_pair_dominant_class_from_segnet_argmax",
    "g1_total_rate_bits",
]
