# SPDX-License-Identifier: MIT
"""Diagnostic G1 R+λD loss for the Z3-G1 scorer-class-conditional substrate.

Mirrors `tac.substrates.z3_balle_hyperprior_bolton.score_aware_loss_v2` but
substitutes the per-pair Ballé hyperprior MLP with a per-class sigma TABLE
indexed by SegNet's per-pair dominant class.

NO score claim. NO promotion. NO exact-eval dispatch from this module.

LOC budget: <= 350 LOC per HNeRV parity discipline L7 (G1 is a bolt-on).
"""
from __future__ import annotations

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
    score_pair_components_dispatch,
)
from tac.substrates.z3_g1_scorer_softmax_hyperprior_gating.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G1ScorerClassGatingHead,
    g1_total_rate_bits,
)


def g1_residual_rate_bits_per_sample(
    *,
    gating_head: Z3G1ScorerClassGatingHead,
    a1_latents: torch.Tensor,
    class_indices: torch.Tensor,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
    quantization_step: float = 1.0,
    factorized_half_range: float = 16.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual-domain G1 rate (BITS) per sample for the Z3HV2 grammar.

    Args:
        gating_head: Z3-G1 scorer-class gating head.
        a1_latents: ``(N, A1_LATENT_DIM)`` frozen A1 latents (one per pair).
        class_indices: ``(N,)`` long tensor of per-pair dominant SegNet class.
        latent_offset: ``(A1_LATENT_DIM,)`` centered affine offset.
        latent_scale: ``(A1_LATENT_DIM,)`` per-dim scale (must be > eps).
        quantization_step: Δ for the conditional-Gaussian AC coder.
        factorized_half_range: half-range for the class-index factorized prior.
        eps: numerical guard added to ``latent_scale`` to avoid div-by-zero.

    Returns:
        ``(bits_per_sample, sigma)`` — bits_per_sample is shape (N,);
        sigma is shape (N, A1_LATENT_DIM).
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
    if class_indices.shape != (a1_latents.shape[0],):
        raise ValueError(
            f"class_indices shape {tuple(class_indices.shape)} must equal "
            f"({a1_latents.shape[0]},)"
        )
    safe_scale = latent_scale.clamp(min=eps)
    residual = (a1_latents - latent_offset.unsqueeze(0)) / safe_scale.unsqueeze(0)
    sigma = gating_head(class_indices)
    bits_per_sample = g1_total_rate_bits(
        residual,
        sigma,
        class_indices,
        num_classes=gating_head.config.num_scorer_classes,
        quantization_step=quantization_step,
        factorized_half_range=factorized_half_range,
    )
    return bits_per_sample, sigma


def z3_g1_lagrangian(
    *,
    gating_head: Z3G1ScorerClassGatingHead,
    a1_latents: torch.Tensor,
    class_indices: torch.Tensor,
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
    """Full score-aware G1 R+λD Lagrangian for Z3-G1.

    When ``decoded_pair_rt`` and ``gt_pair`` are None, only the rate term
    is computed (rate-only training mode — useful for initial smoke pass
    before composing with the frozen A1 decoder; consistent with the Z3 v2
    production path which is rate-only).

    Returns a dict with keys: ``rate_bits_total``, ``rate_lagrangian``,
    ``seg_dist`` (or zero tensor), ``pose_dist`` (or zero tensor),
    ``total_loss``, plus diagnostic ``sigma_mean``.
    """
    bits_per_sample, sigma = g1_residual_rate_bits_per_sample(
        gating_head=gating_head,
        a1_latents=a1_latents,
        class_indices=class_indices,
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
    }


def estimate_z3g1_section_overhead_bytes(
    *,
    gating_head: Z3G1ScorerClassGatingHead,
    n_pairs: int = A1_N_PAIRS,
    avg_brotli_ratio: float = 0.6,
) -> int:
    """Estimate the Z3HV2 (G1 variant) section overhead in bytes.

    Section = header (31B) + length prefixes (10B) +
              brotli(sigma_table_int8) +
              brotli(class_indices) +
              brotli(residual_int8) +
              per-dim affine (224B).

    Per Wunderkind G1 amortization: ship the section only when bytes_saved
    (vs A1's 15387 B latent_blob) > section_size.
    """
    sigma_table_bytes = int(
        gating_head.config.num_scorer_classes * A1_LATENT_DIM * avg_brotli_ratio
    )
    class_indices_bytes = int(n_pairs * 1 * avg_brotli_ratio)
    residual_bytes = int(n_pairs * A1_LATENT_DIM * avg_brotli_ratio)
    header_bytes = 31 + 2 + 4 + 4
    affine_bytes = 4 * A1_LATENT_DIM * 2
    return (
        header_bytes
        + sigma_table_bytes
        + class_indices_bytes
        + residual_bytes
        + affine_bytes
    )


__all__ = [
    "estimate_z3g1_section_overhead_bytes",
    "g1_residual_rate_bits_per_sample",
    "z3_g1_lagrangian",
]
