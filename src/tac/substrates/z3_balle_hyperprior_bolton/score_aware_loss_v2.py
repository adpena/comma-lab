# SPDX-License-Identifier: MIT
"""Diagnostic Ballé R+λD loss for the Z3 v2 latent-replacement substrate.

Differs from sibling ``score_aware_loss.py`` in TWO ways:

1. The "y" in the conditional Gaussian rate term is the QUANTIZED RESIDUAL
   in A1's range (``(latents - latent_offset) / latent_scale`` rounded), NOT
   the raw latents. This matches the v2 wire format where the residual is
   what's actually shipped in bytes.

2. The frozen A1 decoder IS composed in the loop: the trainer hands
   ``(decoded_pair, gt_pair)`` to the canonical
   ``score_pair_components_dispatch`` per Catalog #164. The distortion
   gradient flows back through the decoder into the (frozen) latent
   reconstruction, so ``decoder.requires_grad_(False)`` is OK — only the
   hyperprior MLP weights update; the latents themselves are CONSTANT
   per-batch (read from the A1 archive); only their RATE-coding parameters
   move.

NO score claim. NO promotion. NO exact-eval dispatch from this module.
The current production packet is direct-residual Z3HV2; this loss is a
diagnostic prior until a real entropy-coded residual decoder consumes the
hyperprior side-info at inflate time.

Per Catalog #220 this module is not itself score authority. Operational
promotion requires the emitted Z3HV2 archive plus paired exact CPU/CUDA auth
eval; rate-only local training only shapes diagnostic residual statistics.

LOC budget: ≤ 350 LOC per HNeRV parity discipline L7 (Z3 is a bolt-on).
"""
from __future__ import annotations

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


def balle_rate_term_residual_bits_per_sample(
    *,
    hyperprior: Z3HyperpriorMLP,
    a1_latents: torch.Tensor,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual-domain Ballé 2018 rate (BITS) per sample for the v2 grammar.

    Args:
        hyperprior: Z3 hyperprior MLP (encoder + decoder + quantizer).
        a1_latents: ``(N, A1_LATENT_DIM)`` frozen A1 latents (one per pair).
        latent_offset: ``(A1_LATENT_DIM,)`` centered affine offset.
        latent_scale: ``(A1_LATENT_DIM,)`` per-dim scale (must be > eps).
        quantization_step: Δ for the conditional-Gaussian AC coder.
        factorized_half_range: half-range for the w_hat factorized prior.
        eps: numerical guard added to ``latent_scale`` to avoid div-by-zero.

    Returns:
        ``(bits_per_sample, sigma, w_hat)`` — bits_per_sample is shape (N,).
    """
    if a1_latents.dim() != 2 or a1_latents.shape[-1] != A1_LATENT_DIM:
        raise ValueError(
            f"a1_latents must be (N, {A1_LATENT_DIM}); got {tuple(a1_latents.shape)}"
        )
    if latent_offset.shape != (A1_LATENT_DIM,) or latent_scale.shape != (A1_LATENT_DIM,):
        raise ValueError(
            f"latent_offset and latent_scale must be ({A1_LATENT_DIM},); got "
            f"{tuple(latent_offset.shape)} and {tuple(latent_scale.shape)}"
        )
    safe_scale = latent_scale.clamp(min=eps)
    residual = (a1_latents - latent_offset.unsqueeze(0)) / safe_scale.unsqueeze(0)
    sigma, w_hat = hyperprior(residual, quantize=True)
    bits_per_sample = total_balle_rate_bits(
        residual,
        sigma,
        w_hat,
        quantization_step=quantization_step,
        factorized_half_range=factorized_half_range,
    )
    return bits_per_sample, sigma, w_hat


def reconstruct_v2_latents_for_decoder(
    *,
    a1_latents: torch.Tensor,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
    quantization_step: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reconstruct latents in A1's range after Z3 quantization (deterministic).

    The v2 grammar quantizes residual = (latents - offset) / scale, then ships
    the int8 residual. At inflate time we reconstruct
    latents = residual * scale + offset.
    During training, we must simulate the same quantization in a
    DIFFERENTIABLE way (round + STE) so the distortion path's gradient
    flows back to the per-dim affine + the (frozen) latents.

    Returns the reconstructed latents in fp32 with the same shape as
    ``a1_latents``.
    """
    safe_scale = latent_scale.clamp(min=eps)
    residual = (a1_latents - latent_offset.unsqueeze(0)) / safe_scale.unsqueeze(0)
    # Round + STE so gradient passes through.
    residual_q = (residual.round().detach() - residual).detach() + residual
    # Clamp to int8 range as in the runtime.
    residual_q = residual_q.clamp(min=-127.0, max=127.0)
    reconstructed = residual_q * safe_scale.unsqueeze(0) + latent_offset.unsqueeze(0)
    return reconstructed


def z3_v2_lagrangian(
    *,
    hyperprior: Z3HyperpriorMLP,
    a1_latents: torch.Tensor,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
    decoded_pair_rt: tuple[torch.Tensor, torch.Tensor] | None,
    gt_pair: tuple[torch.Tensor, torch.Tensor] | None,
    alpha_rate: float = CONTEST_RATE_WEIGHT,
    beta_seg: float = CONTEST_SEG_WEIGHT,
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
    contest_normalizer_bytes: float = 37_545_489.0,
    gt_pose_batch: torch.Tensor | None = None,
    gt_seg_batch: torch.Tensor | None = None,
    gt_seg_already_probs: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Full score-aware Ballé R+λD Lagrangian for Z3 v2.

    When ``decoded_pair_rt`` and ``gt_pair`` are None, only the rate term
    is computed (rate-only training mode — useful for initial smoke pass
    before composing with the frozen A1 decoder).

    Returns a dict with keys: ``rate_bits_total``, ``rate_lagrangian``,
    ``seg_dist`` (or zero tensor), ``pose_dist`` (or zero tensor),
    ``total_loss``, plus diagnostic ``sigma_mean`` and ``w_hat_abs_max``.
    """
    bits_per_sample, sigma, w_hat = balle_rate_term_residual_bits_per_sample(
        hyperprior=hyperprior,
        a1_latents=a1_latents,
        latent_offset=latent_offset,
        latent_scale=latent_scale,
        quantization_step=quantization_step,
        factorized_half_range=factorized_half_range,
    )
    rate_bits_total = bits_per_sample.sum()
    rate_bytes = rate_bits_total / 8.0
    rate_lagrangian = alpha_rate * rate_bytes / contest_normalizer_bytes

    seg_dist: torch.Tensor | None = None
    pose_dist: torch.Tensor | None = None
    cache_args_present = (
        gt_pose_batch is not None
        or gt_seg_batch is not None
        or gt_seg_already_probs is not None
    )
    if decoded_pair_rt is not None and (gt_pair is not None or cache_args_present):
        rgb_0_rt, rgb_1_rt = decoded_pair_rt
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


def estimate_z3v2_section_overhead_bytes(
    *,
    hyperprior: Z3HyperpriorMLP,
    n_pairs: int = 600,
    hyper_dim: int = 8,
    avg_brotli_ratio: float = 0.6,
) -> int:
    """Estimate the Z3HV2 section overhead in bytes.

    Section = header (31B) + length prefixes (10B) + brotli(weights) +
    brotli(w_hat) + brotli(residual) + per-dim affine (224B). Per Ballé
    2018 amortization principle: ship the section only when bytes_saved
    (vs A1's 15387 B latent_blob) > section_size.
    """
    param_count = sum(p.numel() for p in hyperprior.parameters())
    weights_bytes = int(param_count * avg_brotli_ratio)
    w_hat_bytes = int(n_pairs * hyper_dim * avg_brotli_ratio)
    residual_bytes = int(n_pairs * A1_LATENT_DIM * avg_brotli_ratio)
    header_bytes = 31 + 2 + 4 + 4
    affine_bytes = 4 * A1_LATENT_DIM * 2
    return header_bytes + weights_bytes + w_hat_bytes + residual_bytes + affine_bytes


__all__ = [
    "balle_rate_term_residual_bits_per_sample",
    "estimate_z3v2_section_overhead_bytes",
    "reconstruct_v2_latents_for_decoder",
    "z3_v2_lagrangian",
]
