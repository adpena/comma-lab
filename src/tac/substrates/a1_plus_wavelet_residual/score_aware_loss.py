"""Score-aware Lagrangian for the A1 + wavelet residual composition.

L = alpha * B / N + beta * d_seg + gamma * sqrt(d_pose)

where B is the COMPOSITION archive byte count (A1 base bytes are constant
= ~178,162; the composition adds the wavelet sidecar bytes which are the
only DIFFERENTIABLE-via-proxy term — the int8 coefficient layout is fixed
at config time, so the rate term is constant during training and gradient
flows through seg + pose).

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian
(NOT weight-domain proxies like rel_err²).
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": apply_eval_roundtrip=True
is the only acceptable training mode.

This loss is the FREEZE-A1 variant (D3.C in the council taxonomy): A1's
decoder + latents are gradient-stopped; only the wavelet residual head's
parameters carry gradients.  This is cheaper than D3.B joint mode
(~$0.50 vs $4-5) and provides a clean attribution probe for whether the
wavelet basis carries pose-axis information at the A1 operating point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)


@dataclass(frozen=True)
class A1PlusWaveletResidualLossWeights:
    """The score-domain Lagrangian weights."""

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Optional operating-point pose tilt.

    The default is the contest formula (1.0).  META-COUNCIL 2026-05-13
    noted the pose-marginal is 2.71x SegNet at PR106 r2 operating point,
    but A1's operating point (0.193 contest-CPU) sits slightly above PR106
    r2's so the multiplier should be empirically calibrated, not assumed.
    Default 1.0 preserves the contest formula; experimental >1.0 is opt-in.
    """
    contest_normalizer: float = 37_545_489.0


class A1PlusWaveletResidualScoreAwareLoss(torch.nn.Module):
    """Lagrangian as a torch Module.

    The base A1 RGB comes pre-rendered (frozen decoder, no gradient through
    A1 weights).  The wavelet residual is reconstructed via single-level
    DB4 IDWT at the foveal patch and added to A1's predicted RGB at
    selected pair indices.  Only the residual's parameters carry the
    gradient — A1's decoder + latents are gradient-stopped.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: A1PlusWaveletResidualLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        rgb_0: torch.Tensor,
        rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        # Lazy import per the canonical pattern.
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        seg_term, pose_term = score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
        )
        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "A1PlusWaveletResidualLossWeights",
    "A1PlusWaveletResidualScoreAwareLoss",
]
