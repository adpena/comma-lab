# SPDX-License-Identifier: MIT
"""pact_nerv_distilled_scorer score-aware Lagrangian.

L = alpha*B/N + beta*d_seg + gamma*sqrt(10*d_pose) + delta*KL_distill_T2

The KL distillation term is the Hinton 1503.02531 §3 KL with T=2.0 between
distilled-scorer-surrogate logits and frozen-teacher (upstream SegNet +
PoseNet) logits. Stage 1 dispatch lands the full distillation wire-in; at
L0 SCAFFOLD this module declares the canonical loss surface.

CLAUDE.md compliance:
- eval_roundtrip=True MANDATORY DEFAULT per Catalog #6
- patches differentiable yuv6 BEFORE scorer construction per PR95/PR106
- No silent CUDA fallback
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
    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    contest_normalizer: float = 37_545_489.0
    delta_distill: float = 1.0
    """Weight on the Hinton KL-T=2.0 distillation term (Stage 1 default 1.0)."""


class PactNervDistilledScorerScoreAwareLoss(torch.nn.Module):
    """The Lagrangian as a torch Module (L0 SKETCH).

    Per Catalog #222 scorer-loader assignment-order: pose_scorer FIRST,
    seg_scorer SECOND. Per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE":
    apply_eval_roundtrip=True default; False raises ValueError.
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
                "'eval_roundtrip - non-negotiable' (Catalog #6 MANDATORY DEFAULT)"
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
