# SPDX-License-Identifier: MIT
"""Z3 Ballé hyperprior bolt-on architecture.

Per zen-floor band v2 council (`feedback_zen_floor_band_v2_post_z1_ablation_20260514.md`)
+ campaign roadmap (`feedback_long_term_multi_year_campaigns_landed_20260514.md`) +
operator decision Z3 ranking, this is the across-class staircase Step 1: cheapest
$2 validation that Ballé-2018 scale-hyperprior side-info reduces bytes on the
frozen A1 base. A1's latent_blob is currently arithmetic/LZMA-coded with a
FACTORIZED prior + delta + dim-reorder; the hyperprior reads each pair's local
latent statistics and emits a smaller AC-coded representation under a
conditional Gaussian prior.

Architecture (Ballé 2018 ICLR scale-hyperprior, tiny per-pair variant):

    For each pair p in range(600):
        y_p = A1_latents[p, :]   # frozen 28-dim latent code from A1's
                                  # decoded latent_blob
           |
           v
        Hyper-analysis h_a:        y_p -> w_p (8-dim hyper-latent)
           Linear(28 -> 16) -> GDN -> Linear(16 -> 8)
           |
           v
        Quantize w_p to w_hat_p   (round + STE for training)
           |
           v
        Hyper-synthesis h_s:       w_hat_p -> σ_p (28-dim per-dim scale > 0)
           Linear(8 -> 16) -> IGDN -> Linear(16 -> 28) -> softplus
           |
           v
        Conditional density p_y(y_p | σ_p) = N(0, σ_p²)  (used by AC coder)

Per Ballé 2018:

    rate R = E[-log p_z(w_hat)] + E[-log p_y(y_hat | σ(w_hat))]

where p_z is a factorized prior over the hyper-latent w_hat (small 8-dim
side-info) and p_y is a conditional Gaussian whose per-dim scale comes
from h_s(w_hat). The side-info bytes amortise when |y stream| > |w stream|.

A1 substrate context (per CLAUDE.md "HNeRV / leaderboard-implementation parity
discipline" + Z1 ablation `feedback_z1_mdl_ablation_landed_20260514.md`):

    A1 measured scorer-conditional MDL density = 99.29% — the encoder is
    saturated within the HNeRV-family class. Hyperprior is an ACROSS-CLASS
    move (Ballé 2018 hyperprior is a recognized class-shift literature
    anchor per Catalog #219 cathedral_autopilot ranker v2). Predicted
    ΔS = −0.005 to −0.010 [prediction; not yet empirically verified]
    via byte savings on latent_blob (~5-15% of 15,387 B) with zero
    distortion change. Strict-honest: at 99.29% density the realistic
    delta is at the low end of this band; smoke is the canonical test.

NO score claim. NO promotion. NO exact-eval dispatch from this module.
Tagged `research_only=true` until smoke + full-run empirical anchor land.

Param count target: ≤ 1500 params total (16+8 + 16+28 = ~750 params for
the two linear layers + GDN bias; tiny by design — the hyperprior must
amortize, not dominate). The encoder/decoder weights are quantized to
int8 and brotli-compressed for the sidecar.

Bolt-on LOC budget: ≤ 350 LOC per HNeRV parity discipline L7
(`bolt_on_loc_budget=350`). Substrate-engineering exemption is NOT used;
this IS a bolt-on (A1 weights frozen).

CLAUDE.md compliance:
- No silent device defaults (callers pass device explicitly)
- No scorer loading inside this module (score-aware loss in sibling)
- No /tmp paths
- Reviewable in 30 seconds per L12 (single-file architecture)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# A1 substrate constants (frozen; from submissions/a1/src/codec.py).
A1_LATENT_DIM = 28
A1_N_PAIRS = 600
A1_BASE_CHANNELS = 36
A1_EVAL_H = 384
A1_EVAL_W = 512
A1_CAMERA_H = 874
A1_CAMERA_W = 1164


@dataclass(frozen=True)
class Z3HyperpriorConfig:
    """Static design-time parameters for Z3 hyperprior bolt-on.

    All fields required-keyword (no silent defaults beyond explicit ones).
    Council-calibrated for byte savings vs sidecar overhead amortization.
    """

    hyper_latent_dim: int = 8
    """Hyper-latent w_p dimensionality (per-pair side-info). 8 is the
    Ballé 2018 canonical small variant. Must be << A1_LATENT_DIM=28."""

    hyper_hidden_dim: int = 16
    """Hidden width inside h_a and h_s (linear MLPs)."""

    int8_scale_clip: float = 7.0
    """Clip range for int8-quantized hyperprior weights (q_real = real /
    scale; scale chosen per layer)."""

    min_sigma: float = 1e-3
    """Lower clamp for σ_p outputs to avoid div-by-zero in AC coder."""

    max_sigma: float = 16.0
    """Upper clamp for σ_p outputs (codable range bound)."""

    quantization_step: float = 1.0
    """Quantization step Δ for both y_hat and w_hat (Ballé 2018 default 1)."""


class _LinearGDN(nn.Module):
    """Linear → GDN (or IGDN) block per Ballé 2018.

    GDN: y = x / sqrt(beta + sum(gamma * x²)). Inverse GDN swaps to
    y = x * sqrt(beta + sum(gamma * x²)). For Z3's tiny per-pair MLPs
    we use a *channel-wise* simplification (no spatial convolution) since
    each pair carries a single latent code y_p of shape (latent_dim,).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        inverse: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.beta = nn.Parameter(torch.full((out_dim,), 1.0))
        self.gamma = nn.Parameter(torch.full((out_dim, out_dim), 0.1))
        self.inverse = inverse
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim)
        h = self.linear(x)  # (B, out_dim)
        # gamma_pos = gamma² (always positive, per Ballé 2018 reparam).
        gamma_pos = self.gamma ** 2
        beta_pos = self.beta ** 2 + self.eps
        # norm[b, c] = beta_c + sum_d gamma_pos[c, d] * h[b, d]²
        norm = beta_pos.unsqueeze(0) + (h ** 2) @ gamma_pos.T
        denom = torch.sqrt(norm)
        if self.inverse:
            return h * denom
        return h / denom


class Z3HyperpriorMLP(nn.Module):
    """Ballé 2018 scale-hyperprior tiny MLP variant for A1 per-pair latents.

    Inputs:
        y: (N_pairs, A1_LATENT_DIM=28) frozen A1 latents (decoded via
           A1's existing codec).

    Outputs:
        sigma: (N_pairs, A1_LATENT_DIM=28) per-dim scale > 0 used by the
           AC coder's conditional Gaussian prior.
        w_hat: (N_pairs, hyper_latent_dim=8) quantized hyper-latent
           (the side-info ACTUALLY shipped in the Z3HP1 sidecar).
    """

    def __init__(self, config: Z3HyperpriorConfig | None = None) -> None:
        super().__init__()
        cfg = config or Z3HyperpriorConfig()
        self.config = cfg
        # Hyper-analysis: y (28) -> w (8) via 28 -> 16 -> 8.
        self.h_a_1 = _LinearGDN(A1_LATENT_DIM, cfg.hyper_hidden_dim, inverse=False)
        self.h_a_2 = nn.Linear(cfg.hyper_hidden_dim, cfg.hyper_latent_dim)
        # Hyper-synthesis: w_hat (8) -> sigma (28) via 8 -> 16 -> 28.
        self.h_s_1 = _LinearGDN(cfg.hyper_latent_dim, cfg.hyper_hidden_dim, inverse=True)
        self.h_s_2 = nn.Linear(cfg.hyper_hidden_dim, A1_LATENT_DIM)

    def forward(
        self,
        y: torch.Tensor,
        *,
        quantize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (sigma, w_hat).

        Args:
            y: (N, A1_LATENT_DIM) frozen A1 latents.
            quantize: If True, w_hat is round-with-STE; else identity
                (used during early training warmup so gradients flow).
        """
        if y.dim() != 2 or y.shape[-1] != A1_LATENT_DIM:
            raise ValueError(
                f"y must be (N, {A1_LATENT_DIM}); got {tuple(y.shape)}"
            )
        # Hyper-analysis.
        h1 = self.h_a_1(y)
        w = self.h_a_2(h1)  # (N, hyper_latent_dim)
        # Quantize via round + STE.
        if quantize:
            w_hat = _round_ste(w / self.config.quantization_step) * self.config.quantization_step
        else:
            w_hat = w
        # Hyper-synthesis.
        h2 = self.h_s_1(w_hat)
        sigma_logits = self.h_s_2(h2)
        # Softplus + clamp to (min_sigma, max_sigma).
        sigma = F.softplus(sigma_logits) + self.config.min_sigma
        sigma = sigma.clamp(min=self.config.min_sigma, max=self.config.max_sigma)
        return sigma, w_hat


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through-estimator (STE) gradient.

    Forward: round(x). Backward: identity (gradient passes through
    unchanged). Per Ballé 2018 default training-time quantization
    surrogate.
    """
    return (x.round() - x).detach() + x


def conditional_gaussian_rate_bits(
    y: torch.Tensor,
    sigma: torch.Tensor,
    *,
    quantization_step: float = 1.0,
    log2_eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-sample rate in BITS under conditional Gaussian prior.

    rate(y | σ) = -log2 [ Φ((y_quant + Δ/2) / σ) - Φ((y_quant - Δ/2) / σ) ]

    where Φ is the standard normal CDF. Returns the SUMMED bits over all
    dimensions per sample (i.e., output shape (N,)).

    Per Ballé 2018 + MacKay (Information Theory book § 6.7): this is the
    canonical ideal-AC-coder rate estimate under a continuous prior with
    uniform-noise dequantization. Real AC coders (e.g., constriction's
    `RangeEncoder` over a Gaussian-quantizer model) achieve this rate
    within ~0.5 bits / pair overhead.

    Args:
        y: (N, D) sample values (assumed already at the quantized grid OR
            will be auto-quantized to step Δ).
        sigma: (N, D) per-dim scale > 0.
        quantization_step: Δ, the quantization step (Ballé 2018 default 1).
    """
    if y.shape != sigma.shape:
        raise ValueError(
            f"y {tuple(y.shape)} and sigma {tuple(sigma.shape)} must match"
        )
    # Quantize y to the integer grid scaled by Δ.
    y_q = (y / quantization_step).round() * quantization_step
    # CDF differences.
    upper = (y_q + 0.5 * quantization_step) / sigma
    lower = (y_q - 0.5 * quantization_step) / sigma
    # Numerically stable: clamp the prob to >= log2_eps.
    cdf_diff = 0.5 * (torch.erf(upper / math.sqrt(2.0)) - torch.erf(lower / math.sqrt(2.0)))
    cdf_diff = cdf_diff.clamp(min=log2_eps)
    bits_per_dim = -torch.log2(cdf_diff)
    return bits_per_dim.sum(dim=-1)  # (N,)


def factorized_uniform_rate_bits(
    w: torch.Tensor,
    *,
    quantization_step: float = 1.0,
    half_range: float = 16.0,
) -> torch.Tensor:
    """Estimate per-sample rate in BITS for the hyper-latent w under a
    factorized uniform prior over [-half_range, +half_range].

    This is the simplest factorized prior (Ballé 2018 baseline). The
    real Z3HP1 implementation may upgrade to a learned cumulative
    factorized prior (Ballé 2018 Eq. 7); the uniform variant gives an
    UPPER BOUND on rate that is sufficient for the smoke validation.

    rate(w) = D * log2(2 * half_range / Δ)

    Returns (N,) summed bits.
    """
    levels = 2.0 * half_range / quantization_step
    bits_per_dim = math.log2(levels)
    return torch.full(
        (w.shape[0],), bits_per_dim * w.shape[-1], device=w.device, dtype=w.dtype
    )


def total_balle_rate_bits(
    y: torch.Tensor,
    sigma: torch.Tensor,
    w_hat: torch.Tensor,
    *,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
) -> torch.Tensor:
    """Total Ballé 2018 hyperprior rate in BITS per sample.

    R_total = R_y + R_w
            = -log p_y(y_hat | σ) - log p_z(w_hat)

    Returns (N,) total bits per sample.
    """
    rate_y = conditional_gaussian_rate_bits(
        y, sigma, quantization_step=quantization_step
    )
    rate_w = factorized_uniform_rate_bits(
        w_hat,
        quantization_step=quantization_step,
        half_range=factorized_half_range,
    )
    return rate_y + rate_w


__all__ = [
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "Z3HyperpriorConfig",
    "Z3HyperpriorMLP",
    "conditional_gaussian_rate_bits",
    "factorized_uniform_rate_bits",
    "total_balle_rate_bits",
]
