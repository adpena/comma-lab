# SPDX-License-Identifier: MIT
"""boost_nerv score-aware Lagrangian — L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose).

L0 SKETCH; mirrors the ds_nerv reference. Trainer wiring through
``tac.differentiable_eval_roundtrip`` lands at L1 SCAFFOLD per the Phase 2
lifecycle. The boosting chain operates on the FINAL rgb_0/rgb_1 output;
the loss path does not need to be aware of the per-round residuals.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
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
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""
    contest_normalizer: float = 37_545_489.0


class BoostnervScoreAwareLoss(torch.nn.Module):
    """The Lagrangian as a torch Module.

    Trainer usage (L1 SCAFFOLD; not yet shipped at L0):

        # 1. patch yuv6 BEFORE scorer construction
        # tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()
        loss_fn = BoostnervScoreAwareLoss(seg_scorer, pose_scorer, weights=...)
        loss, parts = loss_fn(rgb_0, rgb_1, gt_0, gt_1, archive_bytes_proxy)
        loss.backward()
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
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        del noise_std
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

        rate_term = (
            self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer
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
