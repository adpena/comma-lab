# SPDX-License-Identifier: MIT
"""Score-aware Ballé R+λD loss for the Z3 hyperprior bolt-on.

Z3 is a BOLT-ON over the FROZEN A1 base: the A1 renderer + posenet/segnet
score components are computed on the reconstructed pair, and the
hyperprior adds a RATE term:

    L_z3(θ_hp, A1_latents)
        = α_rate * R_balle(y, σ, w_hat) / N_pairs
        + β_seg  * seg_dist(A1_pair_reconstructed_from_latents, GT_pair)
        + γ_pose * sqrt(pose_dist(A1_pair_reconstructed_from_latents, GT_pair))

where:

- ``R_balle = rate_y + rate_w`` is the conditional-Gaussian-plus-factorized
  Ballé 2018 total rate in bits per sample (architecture.py).
- The A1 pair reconstruction uses the FROZEN A1 decoder applied to the
  quantized residual latents (the Z3 contribution is the RATE; the
  distortion comes from the frozen A1 decoder).
- α/β/γ are the contest weights (`CONTEST_RATE_WEIGHT=25`,
  `CONTEST_SEG_WEIGHT=100`, `CONTEST_POSE_SQRT_WEIGHT=sqrt(10)`).

Per Catalog #164 + canonical `tac.substrates.score_aware_common.score_pair_components`,
the scorer pathway is routed through `preprocess_input` BEFORE forward.
The differentiable YUV6 patch (Catalog #187) is applied at trainer init,
NOT here.

NO score claim. The smoke trainer initially trains rate-only against a
fixed A1 latent prior; the full trainer composes with the A1 decoder via
the trainer-side glue (cannot be fully self-contained here because
A1 codec.py lives in `submissions/a1/src/`).

LOC budget: ≤ 350 LOC per HNeRV parity discipline L7 (Z3 is a bolt-on;
substrate-engineering exemption NOT used).
"""
from __future__ import annotations

import math

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
    score_pair_components_dispatch,
)
from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_LATENT_DIM,
    Z3HyperpriorMLP,
    total_balle_rate_bits,
)


def balle_rate_term_bits_per_sample(
    *,
    hyperprior: Z3HyperpriorMLP,
    a1_latents: torch.Tensor,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Ballé 2018 R = R_y + R_w rate term (in BITS) for a batch
    of A1 latents.

    Args:
        hyperprior: Z3 hyperprior MLP (encoder + decoder + quantizer).
        a1_latents: ``(N, A1_LATENT_DIM)`` frozen A1 latents (one per pair).
        quantization_step: Δ for both y_hat and w_hat (Ballé default 1).
        factorized_half_range: clip range for the uniform factorized prior
            on w_hat (Ballé default 16).

    Returns:
        ``(total_bits_per_sample, sigma, w_hat)`` — total_bits_per_sample
        is shape (N,); the trainer SUMS or AVERAGES over the batch to get
        the rate Lagrangian.
    """
    if a1_latents.dim() != 2 or a1_latents.shape[-1] != A1_LATENT_DIM:
        raise ValueError(
            f"a1_latents must be (N, {A1_LATENT_DIM}); got {tuple(a1_latents.shape)}"
        )
    sigma, w_hat = hyperprior(a1_latents, quantize=True)
    bits_per_sample = total_balle_rate_bits(
        a1_latents,
        sigma,
        w_hat,
        quantization_step=quantization_step,
        factorized_half_range=factorized_half_range,
    )
    return bits_per_sample, sigma, w_hat


def z3_lagrangian(
    *,
    hyperprior: Z3HyperpriorMLP,
    a1_latents: torch.Tensor,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
    a1_pair_pred_rt: tuple[torch.Tensor, torch.Tensor] | None,
    gt_pair: tuple[torch.Tensor, torch.Tensor] | None,
    alpha_rate: float = CONTEST_RATE_WEIGHT,
    beta_seg: float = CONTEST_SEG_WEIGHT,
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
    n_pairs_normalizer: int = 600,
    contest_normalizer_bytes: float = 37_545_489.0,
    gt_pose_batch: torch.Tensor | None = None,
    gt_seg_batch: torch.Tensor | None = None,
    gt_seg_already_probs: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Full score-aware Ballé R+λD Lagrangian for Z3.

    When ``a1_pair_pred_rt`` and ``gt_pair`` are None, only the rate term
    is computed (rate-only training mode — useful for initial smoke pass
    before composing with the frozen A1 decoder).

    Returns a dict with keys: ``rate_bits_total``, ``rate_lagrangian``,
    ``seg_dist`` (or None), ``pose_dist`` (or None), ``total_loss``.

    The contest score normalization mirrors the upstream evaluator:

        score = β * seg_dist + γ * sqrt(pose_dist) + α * total_bytes / N

    where ``N = contest_normalizer_bytes`` and ``total_bytes`` is converted
    from bits via the trainer-side full-archive byte accounting (not
    computed here; the trainer adds the base A1 archive size + the Z3
    sidecar overhead to the bits-from-this-loss before scoring).
    """
    bits_per_sample, sigma, w_hat = balle_rate_term_bits_per_sample(
        hyperprior=hyperprior,
        a1_latents=a1_latents,
        quantization_step=quantization_step,
        factorized_half_range=factorized_half_range,
    )
    rate_bits_total = bits_per_sample.sum()  # scalar
    # Convert bits → bytes / N_contest for the rate Lagrangian.
    rate_bytes = rate_bits_total / 8.0
    rate_lagrangian = alpha_rate * rate_bytes / contest_normalizer_bytes

    seg_dist: torch.Tensor | None = None
    pose_dist: torch.Tensor | None = None
    cache_args_present = (
        gt_pose_batch is not None
        or gt_seg_batch is not None
        or gt_seg_already_probs is not None
    )
    if a1_pair_pred_rt is not None and (gt_pair is not None or cache_args_present):
        rgb_0_rt, rgb_1_rt = a1_pair_pred_rt
        gt_rgb_0: torch.Tensor | None
        gt_rgb_1: torch.Tensor | None
        if gt_pair is None:
            gt_rgb_0 = None
            gt_rgb_1 = None
        else:
            gt_rgb_0, gt_rgb_1 = gt_pair
        seg_dist, pose_dist = score_pair_components_dispatch(
            seg_scorer=seg_scorer,
            pose_scorer=pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
            gt_pose_batch=gt_pose_batch,
            gt_seg_batch=gt_seg_batch,
            gt_seg_already_probs=gt_seg_already_probs,
        )

    if seg_dist is None or pose_dist is None:
        # Rate-only mode (early smoke).
        total_loss = rate_lagrangian
    else:
        total_loss = (
            rate_lagrangian
            + beta_seg * seg_dist
            + gamma_pose * pose_dist.clamp(min=0.0).sqrt()
        )

    return {
        "rate_bits_total": rate_bits_total,
        "rate_bytes": rate_bytes,
        "rate_lagrangian": rate_lagrangian,
        "seg_dist": seg_dist if seg_dist is not None else torch.tensor(0.0),
        "pose_dist": pose_dist if pose_dist is not None else torch.tensor(0.0),
        "total_loss": total_loss,
        "sigma_mean": sigma.mean().detach(),
        "w_hat_abs_max": w_hat.detach().abs().max(),
    }


def estimate_sidecar_overhead_bytes(
    *,
    hyperprior: Z3HyperpriorMLP,
    n_pairs: int = 600,
    hyper_dim: int = 8,
    avg_brotli_ratio: float = 0.6,
) -> int:
    """Estimate the Z3HP1 sidecar overhead in bytes.

    The sidecar = header (27B) + brotli(weights) + brotli(w_hat) +
    brotli(residual). At Ballé small-MLP scale (~700 params) brotli
    typically compresses int8 weights ~60%; w_hat is n_pairs * hyper_dim
    int8 bytes; residual is n_pairs * A1_LATENT_DIM int8 bytes.

    Per Ballé 2018 amortization principle: ship the sidecar only when
    bytes_saved (via reduced latent_blob via conditional Gaussian) >
    estimated sidecar overhead. The trainer uses this estimate to decide
    whether to include the sidecar in the final archive.
    """
    param_count = sum(p.numel() for p in hyperprior.parameters())
    weights_bytes = int(param_count * avg_brotli_ratio)
    w_hat_bytes = int(n_pairs * hyper_dim * avg_brotli_ratio)
    residual_bytes = int(n_pairs * A1_LATENT_DIM * avg_brotli_ratio)
    header_bytes = 27 + 2 + 4 + 4  # struct + len prefixes
    return header_bytes + weights_bytes + w_hat_bytes + residual_bytes


__all__ = [
    "balle_rate_term_bits_per_sample",
    "estimate_sidecar_overhead_bytes",
    "z3_lagrangian",
]
