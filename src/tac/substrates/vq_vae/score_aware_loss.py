# SPDX-License-Identifier: MIT
"""vq_vae score-aware Lagrangian — L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose).

Per CLAUDE.md HNeRV parity discipline L6 + L8: the score-domain Lagrangian must
backprop through SegNet/PoseNet via the differentiable eval-roundtrip + the
patched yuv6 (PR #95/#106 monkey-patch contract). The rate term consumes:

1. A closed-form upper-bound proxy on the archive byte count (Ballé R(D) bound).
2. The VQ-VAE commitment loss (cfg.commitment_cost * |z_e - sg(z_q)|^2),
   which gradient-flows into the encoder.
3. (Optional) A codebook usage prior penalty to encourage codebook utilization.

The PR106 r2 operating point (pose_avg ~ 3.4e-5) flips marginal-value to 2.71x
pose-vs-seg; the loss exposes ``pose_weight_scale`` for the trainer to ramp up
when empirical pose_avg drops below 2.5e-4.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std reserved for forward-compat (current STE is deterministic)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)


@dataclass(frozen=True)
class ScoreAwareLossWeights:
    """The (alpha, beta, gamma) of the score-domain Lagrangian."""

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (alpha_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    """PoseNet term weight. Contest default is sqrt(10) times sqrt(d_pose)."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""

    commitment_weight: float = 0.25
    """Weight on the VQ-VAE commitment loss (per van den Oord 2017)."""


class VqVaeScoreAwareLoss(torch.nn.Module):
    """Lagrangian as a torch Module for vq_vae substrate.

    Mirror of sane_hnerv's loss with the addition of the VQ-VAE commitment-loss
    term consumed via the substrate's ``compute_commitment_loss`` method.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: ScoreAwareLossWeights,
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
        commitment_loss: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: rendered frame pair, ``(B, 3, H, W)``, in [0, 1].
            gt_rgb_0, gt_rgb_1: ground-truth frame pair (already roundtripped
                upstream by the trainer if needed).
            archive_bytes_proxy: scalar tensor — closed-form upper bound on the
                archive byte count.
            commitment_loss: scalar — VQ-VAE commitment loss from
                ``substrate.compute_commitment_loss(pair_indices)``.

        Returns:
            ``(loss_total, parts)`` for the trainer to log.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        del noise_std  # reserved for forward-compat
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        seg_term, pose_term = score_pair_components_dispatch(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
            gt_pose_batch=gt_pose_batch,
            gt_seg_batch=gt_seg_batch,
            gt_seg_already_probs=gt_seg_already_probs,
        )

        rate_term = self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + self.weights.commitment_weight * commitment_loss
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "commitment_term": (self.weights.commitment_weight * commitment_loss).detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
