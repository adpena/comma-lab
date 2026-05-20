# SPDX-License-Identifier: MIT
"""pact_nerv_neural_codec_e2e_cross score-aware Lagrangian (canonical routing).

Extends the canonical score-aware Lagrangian per Catalog #6 + #164 + #226
with an optional gate-entropy regularization bonus (lambda_gate) to
encourage the hyperprior gate to make decisive per-pair branch
selections (CARGO-CULTED at L0; L1 will replace with information-
theoretic rate-distortion-gate Lagrangian per Shannon 1948 R(D)).
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
    lambda_gate: float = 0.0
    """Gate-entropy regularization weight (0.0 disables; L0 default)."""


class PactNervNeuralCodecE2ECrossScoreAwareLoss(torch.nn.Module):
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
        gate_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "Catalog #6 MANDATORY DEFAULT"
            )
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )
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
        gate_entropy_term = torch.zeros((), device=rgb_0.device)
        if gate_values is not None and self.weights.lambda_gate > 0.0:
            # Binary entropy of gate values (encourage decisive selection)
            g = gate_values.clamp(1e-6, 1.0 - 1e-6)
            entropy = -(g * torch.log(g) + (1.0 - g) * torch.log(1.0 - g))
            gate_entropy_term = self.weights.lambda_gate * entropy.mean()
        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + gate_entropy_term
        )
        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "gate_entropy_term": gate_entropy_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
