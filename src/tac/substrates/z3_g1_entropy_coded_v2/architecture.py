# SPDX-License-Identifier: MIT
"""Z3-G1 entropy-coded v2 architecture.

Per `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md`:
v2 introduces a NEW magic + grammar (`Z3G2`) that REPLACES the empty Z3HV2
slots with TWO entropy-coded streams shipped at the wire-byte level. The
training-side architecture remains a 5x28 sigma table + per-pair class
index lookup (same as v1) but the EXPORT path now actually ships those
bytes via Brotli (sigma table and residual bytes) plus constriction-Huffman
coding (class indices).

Key insight (v2 wire grammar):

    v1 (Z3HV2 production-safe direct-residual):
        - hyperprior_weights_int8 = b""  ← empty (F1 codex finding)
        - w_hat_int8 = b""               ← empty (F1 codex finding)
        - residual_int8 = ~1200B         ← only Z3HV2 residual ships
        => ZERO distinguishing G1 bytes ship vs Z3 v2

    v2 (Z3G2 entropy-coded):
        - sigma_table_blob = brotli(140 int8 sigma)   ~300B
        - class_prior_cdf_blob = 5*uint16 = 10B fixed
        - class_index_blob = constriction-Huffman encoded ~200-400B
        - residual_blob = brotli(600*28 int8 residual)
        => 510-710B of distinguishing G1 bytes SHIP

The training-time architecture is identical to v1's
``Z3G1ScorerClassGatingHead`` but renamed
``Z3G2EntropyCodedScorerClassGatingHead`` for clarity that it's the
producer for the v2 wire grammar (and to avoid module-import collision
when sister tests import both v1 and v2).

LOC budget: <= 350 LOC per HNeRV parity discipline L7 (bolt-on).
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

# SegNet outputs 5-class logits per pixel. Same as v1.
G1_NUM_SCORER_CLASSES = 5


@dataclass(frozen=True)
class Z3G1EntropyCodedV2Config:
    """Static design-time parameters for Z3-G1 entropy-coded v2.

    Mirrors v1's ``Z3G1Config`` plus an explicit entropy-coder precision
    field reserved for future class-index stream variants.
    """

    num_scorer_classes: int = G1_NUM_SCORER_CLASSES
    """Number of SegNet classes (5 per upstream/modules.py)."""

    int8_sigma_scale: float = 16.0
    """Scale factor for int8 quantization of sigma_table:
    sigma_int8 = round(sigma_real * 127 / int8_sigma_scale), reconstructed as
    sigma_real = sigma_int8 * int8_sigma_scale / 127."""

    min_sigma: float = 1e-3
    """Lower clamp for sigma outputs to avoid div-by-zero in residual scaling."""

    max_sigma: float = 16.0
    """Upper clamp for sigma outputs (codable range bound)."""

    quantization_step: float = 1.0
    """Quantization step Δ for the residual coder (must match Z3HV2 grammar)."""

    init_sigma: float = 2.0
    """Initial sigma value for the table (uniform across classes/dims).
    Training learns class-specific sigma values from data."""

    ac_precision_bits: int = 24
    """Reserved precision field for future range/ANS class-index coders.
    The current implementation uses constriction-Huffman and records measured
    byte length instead of claiming arithmetic precision."""


class Z3G2EntropyCodedScorerClassGatingHead(nn.Module):
    """Tiny per-class sigma TABLE indexed by SegNet's dominant class.

    Identical training-time architecture to v1's
    ``Z3G1ScorerClassGatingHead`` (5x28 nn.Parameter; F.embedding lookup
    forward; total 140 params) but renamed to avoid collision when
    importing both v1 and v2 in the same module (e.g. cross-version tests).

    The export path is what's NEW in v2: see ``archive.py`` for the v2
    Z3G2 wire grammar that ACTUALLY ships the int8 sigma table + class
    indices via Brotli + constriction-Huffman coding.

    Inputs:
        class_indices: (N_pairs,) long tensor in [0, num_scorer_classes)
            giving the per-pair dominant SegNet class.

    Outputs:
        sigma: (N_pairs, A1_LATENT_DIM) per-dim scale > 0.
    """

    def __init__(self, config: Z3G1EntropyCodedV2Config | None = None) -> None:
        super().__init__()
        cfg = config or Z3G1EntropyCodedV2Config()
        if cfg.num_scorer_classes <= 0 or cfg.num_scorer_classes > 255:
            raise ValueError(
                "num_scorer_classes must be in [1, 255]; got "
                f"{cfg.num_scorer_classes}"
            )
        if cfg.ac_precision_bits < 8 or cfg.ac_precision_bits > 32:
            raise ValueError(
                f"ac_precision_bits must be in [8, 32]; got {cfg.ac_precision_bits}"
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
        """Lookup per-class sigma for each pair."""
        if class_indices.dim() != 1:
            raise ValueError(
                f"class_indices must be 1D; got shape {tuple(class_indices.shape)}"
            )
        if class_indices.dtype not in (torch.int64, torch.int32, torch.uint8):
            raise ValueError(
                f"class_indices must be integer dtype; got {class_indices.dtype}"
            )
        cmax = int(class_indices.max().item()) if class_indices.numel() > 0 else 0
        cmin = int(class_indices.min().item()) if class_indices.numel() > 0 else 0
        if cmin < 0 or cmax >= self.config.num_scorer_classes:
            raise ValueError(
                f"class_indices must be in [0, {self.config.num_scorer_classes}); "
                f"got range [{cmin}, {cmax}]"
            )
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
    """Reconstruct sigma table from int8 bytes shipped in archive."""
    if scale_int8 <= 0:
        raise ValueError(f"scale_int8 must be positive; got {scale_int8}")
    sigma_real = sigma_int8.to(torch.float32) * scale_int8 / 127.0
    return sigma_real.clamp(min=min_sigma)


def g1_v2_per_pair_dominant_class_from_segnet_argmax(
    segnet_argmax_per_pair: torch.Tensor,
    *,
    num_classes: int = G1_NUM_SCORER_CLASSES,
) -> torch.Tensor:
    """Reduce a per-pixel SegNet argmax map to a per-pair dominant class.

    Identical to v1's helper; renamed for v2-namespace clarity.
    """
    if segnet_argmax_per_pair.dim() != 3:
        raise ValueError(
            f"segnet_argmax_per_pair must be (N, H, W); got "
            f"{tuple(segnet_argmax_per_pair.shape)}"
        )
    n_pairs = segnet_argmax_per_pair.shape[0]
    flat = segnet_argmax_per_pair.reshape(n_pairs, -1).to(torch.long)
    one_hot = F.one_hot(flat, num_classes=num_classes)
    histogram = one_hot.sum(dim=1)
    return histogram.argmax(dim=1)


def compute_class_prior_cdf(
    class_indices: torch.Tensor,
    *,
    num_classes: int = G1_NUM_SCORER_CLASSES,
    smoothing: int = 1,
) -> torch.Tensor:
    """Compute per-class frequency counts for use as the entropy-coder prior.

    Args:
        class_indices: (N,) long tensor of class indices in [0, num_classes).
        num_classes: Number of distinct classes.
        smoothing: Add-one smoothing constant to avoid zero-frequency classes
            (which would produce invalid zero-probability symbols under
            entropy coding). Default 1 (Laplace smoothing).

    Returns:
        (num_classes,) int64 tensor of frequency counts (uint16-compatible
        range; the encoder packs these as uint16). Shipping the frequency
        counts (10B) is cheaper than shipping the normalized CDF, and the
        decoder normalizes back to a probability distribution.
    """
    if class_indices.dim() != 1:
        raise ValueError(
            f"class_indices must be 1D; got shape {tuple(class_indices.shape)}"
        )
    if smoothing < 0:
        raise ValueError(f"smoothing must be >= 0; got {smoothing}")
    counts = torch.zeros(num_classes, dtype=torch.int64)
    if class_indices.numel() > 0:
        cmax = int(class_indices.max().item())
        cmin = int(class_indices.min().item())
        if cmin < 0 or cmax >= num_classes:
            raise ValueError(
                f"class_indices must be in [0, {num_classes}); "
                f"got range [{cmin}, {cmax}]"
            )
        # bincount returns counts indexed by value.
        bc = torch.bincount(class_indices.to(torch.long), minlength=num_classes)
        counts = bc.to(torch.int64)
    counts = counts + int(smoothing)
    if counts.max().item() > 0xFFFF:
        raise ValueError(
            f"class frequency counts overflow uint16: max={counts.max().item()}"
        )
    return counts


def g1_v2_total_rate_bits(
    residual: torch.Tensor,
    sigma: torch.Tensor,
    class_indices: torch.Tensor,
    class_prior_counts: torch.Tensor,
    *,
    num_classes: int = G1_NUM_SCORER_CLASSES,
    quantization_step: float = 1.0,
) -> torch.Tensor:
    """Total v2 rate in BITS per sample.

    R_total = R_residual + R_class_index
            = -log p_y(residual | sigma_table[class]) - log p_z(class | prior_cdf)

    where p_z is the per-class empirical prior CDF (NOT factorized uniform);
    this is the v2 improvement over v1's hard-coded log2(num_classes) rate.

    Returns (N,) total bits per sample.
    """
    rate_residual = conditional_gaussian_rate_bits(
        residual, sigma, quantization_step=quantization_step
    )
    # Normalize counts to probabilities; clamp to avoid log(0).
    counts_f = class_prior_counts.to(residual.dtype).clamp(min=1.0)
    probs = counts_f / counts_f.sum()
    # Per-pair class rate: -log2(p[class]).
    class_log_probs = torch.log2(probs.clamp(min=1e-10))
    rate_class = -class_log_probs[class_indices.to(torch.long)]
    return rate_residual + rate_class


__all__ = [
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "G1_NUM_SCORER_CLASSES",
    "Z3G1EntropyCodedV2Config",
    "Z3G2EntropyCodedScorerClassGatingHead",
    "_round_ste",
    "compute_class_prior_cdf",
    "conditional_gaussian_rate_bits",
    "dequantize_sigma_table_int8",
    "factorized_uniform_rate_bits",
    "g1_v2_per_pair_dominant_class_from_segnet_argmax",
    "g1_v2_total_rate_bits",
]
