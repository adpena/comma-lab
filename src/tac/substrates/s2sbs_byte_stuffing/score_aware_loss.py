"""Score-domain Lagrangian for the S2SBS substrate.

Per CLAUDE.md "eval_roundtrip — non-negotiable" + Catalog #164:

* ``apply_eval_roundtrip_during_training`` is applied to rendered RGB
  before the scorer forward.
* The canonical ``score_pair_components`` routes through
  ``scorer.preprocess_input(pair_btchw)`` so 5D pair tensors land at the
  contest scorer contract.
* The HF byte channel imposes a structural rate cost; the loss adds an
  optional HF perturbation penalty to keep the substrate honest about
  the joint-safety threshold from the φ3 audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
    score_pair_components,
)


@dataclass(frozen=True)
class S2sbsLossWeights:
    alpha_rate: float = CONTEST_RATE_WEIGHT
    beta_seg: float = CONTEST_SEG_WEIGHT
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    hf_perturbation_penalty_weight: float = 0.0
    contest_normalizer: float = 37_545_489.0

    def __post_init__(self) -> None:
        for name in (
            "alpha_rate",
            "beta_seg",
            "gamma_pose",
            "pose_weight_scale",
            "hf_perturbation_penalty_weight",
            "contest_normalizer",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.contest_normalizer <= 0.0:
            raise ValueError("contest_normalizer must be positive")


class S2sbsScoreAwareLoss(torch.nn.Module):
    """Score-aware Lagrangian wrapping SegNet + PoseNet through the canonical kit."""

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: S2sbsLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights or S2sbsLossWeights()

    def forward(
        self,
        rgb_0: torch.Tensor,
        rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        *,
        hf_perturbation: torch.Tensor | None = None,
        apply_eval_roundtrip: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip - non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

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
        if hf_perturbation is None:
            hf_term = torch.zeros_like(rate_term)
        else:
            hf_term = self.weights.hf_perturbation_penalty_weight * hf_perturbation.mean()
        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + hf_term
        )
        return loss, {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "hf_perturbation_term": hf_term.detach(),
            "loss_total": loss.detach(),
        }


__all__ = ["S2sbsLossWeights", "S2sbsScoreAwareLoss"]
