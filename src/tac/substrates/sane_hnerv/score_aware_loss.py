"""sane_hnerv score-aware Lagrangian — L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose).

The Fields-medal grand council 2026-05-12 score-domain Lagrangian per
CLAUDE.md HNeRV parity discipline L6:

* `d_seg(theta)`: contest SegNet distortion, gradient-reachable via
  ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``
  + ``patch_upstream_yuv6_globally`` (PR #95/#106 monkey-patch contract).
* `d_pose(theta)`: contest PoseNet distortion via same wiring; the
  sqrt-transform is the contest scorer's pose contribution shape.
* `B(theta)`: post-export archive size in bytes; differentiability is via
  an arithmetic-coder proxy (not in this scaffold — anchored to a
  closed-form bound for now per Ballé 2018).

The PR106 r2 operating point (pose_avg ~ 3.4e-5) flips marginal-value to
2.71x pose-vs-seg; the loss exposes a `pose_weight_scale` kwarg the trainer
can ramp ↑ when the empirical pose_avg drops below 2.5e-4.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std=0.5 (Hotz STE fix)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Import lazily inside functions to avoid hot-path penalty when only the
# class is being type-checked.


@dataclass(frozen=True)
class ScoreAwareLossWeights:
    """The (alpha, beta, gamma) of the score-domain Lagrangian.

    Defaults per the Phase 2 council; tunable via the trainer CLI.
    """

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (alpha_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = 1.0
    """PoseNet term weight (multiplied by sqrt(d_pose))."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class SaneHnervScoreAwareLoss(torch.nn.Module):
    """The Lagrangian as a torch Module so trainers can ``loss.forward(...)``.

    Trainer usage::

        loss_fn = SaneHnervScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=ScoreAwareLossWeights(...),
        )
        # patch yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip
        # rule:
        # tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()
        ...
        loss, parts = loss_fn(rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy)
        loss.backward()

    `rgb_*` come from the substrate (after eval_roundtrip simulation); they
    are differentiable. `archive_bytes_proxy` is a scalar tensor for the
    rate term — usually a closed-form upper bound on the rate (Ballé R(D)
    bound), since post-export bytes are non-differentiable.
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian on a pair of rendered frames.

        Returns:
            ``(loss_total, parts)`` where parts is a dict with named
            components for logging (``rate_term``, ``seg_term``, ``pose_term``).
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        # Lazy import to keep this module's import-time cheap
        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0, noise_std=noise_std)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1, noise_std=noise_std)

        # Run SegNet on the LAST frame of the pair (rgb_1), per upstream contract
        seg_out = self.seg_scorer(rgb_1_rt.unsqueeze(1))  # (B, T=1, C, H, W) -> (B, 5, h, w)
        seg_gt = self.seg_scorer(gt_rgb_1.unsqueeze(1))
        # Cross-entropy proxy on argmax-disagreement: convert to soft via softmax KL
        seg_term = _seg_distortion_proxy(seg_out, seg_gt)

        # PoseNet on (rgb_0, rgb_1) pair
        pose_in = torch.cat([rgb_0_rt, rgb_1_rt], dim=1)  # (B, 6, H, W)
        pose_gt = torch.cat([gt_rgb_0, gt_rgb_1], dim=1)
        pose_out = self.pose_scorer(pose_in)
        pose_target = self.pose_scorer(pose_gt)
        pose_term = ((pose_out[:, :6] - pose_target[:, :6]) ** 2).mean()

        rate_term = self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer

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
    """Soft cross-entropy between predicted seg logits and gt seg logits.

    Direct argmax-disagreement isn't differentiable; soft KL on softmaxed
    logits is the canonical surrogate (T=2.0 ablation flag in trainer).
    """
    log_p = torch.log_softmax(seg_logits_pred, dim=1)
    q = torch.softmax(seg_logits_gt, dim=1)
    return -(q * log_p).sum(dim=1).mean()
