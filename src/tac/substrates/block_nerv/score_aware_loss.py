"""block_nerv score-aware Lagrangian (L0 SKETCH).

Reuses the canonical sane_hnerv Lagrangian shape:

    L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)

The block-nerv ARCHITECTURE provides per-pair specialization via LoRA-style
residuals on the latent embedding; the LOSS itself has no new term (vs
sane_hnerv). The hypothesis is that the per-pair LoRA tables give the
optimizer enough flexibility to over-specialize on individual frame-pairs'
SegNet/PoseNet idiosyncrasies that a single shared decoder cannot capture.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std reserved for future variant (deterministic STE today)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths

L0 SKETCH disclaimer: full trainer wire-in deferred to L1 promotion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockScoreAwareLossWeights:
    """(alpha, beta, gamma) of the block_nerv score-domain Lagrangian.

    Defaults are the Phase 2 council pre-set; trainer CLI overrides at L1.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = 1.0
    pose_weight_scale: float = 1.0
    """Set to 2.71 at PR106-r2 operating point (pose_avg < 2.5e-4)."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class BlockNervScoreAwareLoss(torch.nn.Module):
    """The block_nerv Lagrangian as a torch Module.

    Identical shape to sane_hnerv's; included as a per-substrate module so
    the trainer CLI surface mirrors sane_hnerv's at L1 promotion time.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: BlockScoreAwareLossWeights,
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
        """Compute the score-domain Lagrangian on a pair of rendered frames."""
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        del noise_std  # reserved; STE deterministic at this revision
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        seg_out = self.seg_scorer(rgb_1_rt.unsqueeze(1))
        seg_gt = self.seg_scorer(gt_rgb_1.unsqueeze(1))
        seg_term = _seg_distortion_proxy(seg_out, seg_gt)

        pose_in = torch.cat([rgb_0_rt, rgb_1_rt], dim=1)
        pose_gt = torch.cat([gt_rgb_0, gt_rgb_1], dim=1)
        pose_out = self.pose_scorer(pose_in)
        pose_target = self.pose_scorer(pose_gt)
        pose_term = ((pose_out[:, :6] - pose_target[:, :6]) ** 2).mean()

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


def _seg_distortion_proxy(
    seg_logits_pred: torch.Tensor, seg_logits_gt: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy between predicted and gt seg logits."""
    log_p = torch.log_softmax(seg_logits_pred, dim=1)
    q = torch.softmax(seg_logits_gt, dim=1)
    return -(q * log_p).sum(dim=1).mean()
