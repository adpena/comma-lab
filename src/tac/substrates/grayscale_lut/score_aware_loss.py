# SPDX-License-Identifier: MIT
"""grayscale_lut score-aware Lagrangian — L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose).

Per CLAUDE.md HNeRV parity discipline L6 + L8: the score-domain Lagrangian must
backprop through SegNet/PoseNet via the differentiable eval-roundtrip + the
patched yuv6 (PR #95/#106 monkey-patch contract). Rate term combines (a) a
closed-form upper-bound proxy on the archive byte count (Ballé R(D) bound) plus
(b) an explicit total-variation-style regularizer on the analog grayscale field
(natural-video grayscale is smooth; smoothness is what brotli exploits at
archive time, so penalizing rough grayscale lowers the post-export bytes).

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
    score_pair_components,
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

    grayscale_tv_weight: float = 0.01
    """Total-variation regularizer on the analog grayscale field
    (smoother grayscale -> smaller brotli output)."""


class GrayscaleLutScoreAwareLoss(torch.nn.Module):
    """Lagrangian as a torch Module for grayscale_lut substrate.

    Mirror of sane_hnerv's loss with the addition of a TV regularizer on the
    grayscale field consumed via the substrate's ``grayscale`` parameter.
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
        grayscale_param: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: rendered frame pair, ``(B, 3, H, W)``, in [0, 1].
            gt_rgb_0, gt_rgb_1: ground-truth frame pair (already roundtripped
                upstream by the trainer if needed).
            archive_bytes_proxy: scalar tensor — closed-form upper bound on the
                archive byte count (Ballé R(D) bound).
            grayscale_param: substrate's grayscale parameter ``(N, 1, H/D, W/D)``
                for the TV regularizer.

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

        seg_term, pose_term = score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
        )

        rate_term = self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer

        # Grayscale TV regularizer (smoother grayscale compresses better with brotli)
        tv_x = (grayscale_param[..., 1:] - grayscale_param[..., :-1]).abs().mean()
        tv_y = (grayscale_param[..., 1:, :] - grayscale_param[..., :-1, :]).abs().mean()
        tv_term = self.weights.grayscale_tv_weight * (tv_x + tv_y)

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + tv_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "tv_term": tv_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
