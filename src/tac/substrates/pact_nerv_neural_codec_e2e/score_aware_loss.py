# SPDX-License-Identifier: MIT
"""pact_nerv_neural_codec_e2e score-aware Lagrangian.

L = alpha*B/N + beta*d_seg + gamma*sqrt(10*d_pose) + delta*rate_bits

The rate_bits term is the Ballé 2018 §3 entropy bottleneck rate proxy
(differentiable Gaussian rate from per-latent hyperprior-emitted scales).
At L0 this is a simple Gaussian rate; Stage 1 dispatch replaces with the
full additive-uniform-noise-during-training + round-during-inference per
Ballé canonical.

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
    delta_rate_bits: float = 0.001
    """Ballé differentiable rate-loss weight. Tunes the rate/distortion knee
    of the trained codec; canonical default 0.001-0.01 per Ballé 2018."""


class PactNervNeuralCodecE2eScoreAwareLoss(torch.nn.Module):
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
        rate_bits: torch.Tensor,
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
            + self.weights.delta_rate_bits * rate_bits
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "rate_bits": rate_bits.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
