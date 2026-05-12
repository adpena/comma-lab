"""tc_nerv score-aware Lagrangian (L0 SKETCH).

Adds an explicit **temporal-consistency** regularizer to the sane_hnerv
score-domain Lagrangian:

    L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) + lambda_tc * mean(||delta_rgb||^2)

where ``delta_rgb`` is the per-batch difference between adjacent decoded
frames within the rendered pair (``rgb_1 - rgb_0``); the mean is over
``(batch, channel, height, width)``. The hypothesis: penalizing intra-pair
delta should reduce PoseNet's pose distortion at the PR106 r2 operating
point (pose_avg ~ 3.4e-5) faster than the un-regularized HNeRV.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std reserved for future variant (deterministic STE today)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths

L0 SKETCH disclaimer: this Lagrangian is the design declaration; full
trainer wire-in (mirror of train_substrate_sane_hnerv.py) is deferred to
L1 promotion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TCScoreAwareLossWeights:
    """(alpha, beta, gamma, lambda_tc) of the tc_nerv score-domain Lagrangian.

    Defaults are the Phase 2 council pre-set; trainer CLI overrides at L1.
    """

    alpha_rate: float = 25.0
    """Contest rate term (alpha * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet distortion weight."""

    gamma_pose: float = 1.0
    """PoseNet distortion weight (multiplied by sqrt(d_pose))."""

    pose_weight_scale: float = 1.0
    """Set to 2.71 at PR106-r2 operating point (pose_avg < 2.5e-4)."""

    lambda_tc: float = 0.5
    """Temporal-consistency regularizer weight. 0 disables.

    Council pre-set 0.5 — small enough not to dominate the score-domain
    terms, large enough for the gradient signal to bias adjacency. L1
    calibration may sweep over {0.0, 0.1, 0.5, 1.0, 2.0}.
    """

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class TCNervScoreAwareLoss(torch.nn.Module):
    """The tc_nerv Lagrangian as a torch Module.

    Trainer usage::

        loss_fn = TCNervScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=TCScoreAwareLossWeights(...),
        )
        # patch yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip
        # rule; e.g. tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()
        loss, parts = loss_fn(rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy)
        loss.backward()
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: TCScoreAwareLossWeights,
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
        """Compute the score-domain Lagrangian + temporal-consistency term.

        Returns ``(loss_total, parts)`` where parts is a dict with named
        components for logging (``rate_term``, ``seg_term``, ``pose_term``,
        ``tc_term``).
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        del noise_std  # reserved; STE is deterministic at this revision
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        # SegNet on LAST frame per upstream contract
        seg_out = self.seg_scorer(rgb_1_rt.unsqueeze(1))
        seg_gt = self.seg_scorer(gt_rgb_1.unsqueeze(1))
        seg_term = _seg_distortion_proxy(seg_out, seg_gt)

        # PoseNet on (rgb_0, rgb_1) pair
        pose_in = torch.cat([rgb_0_rt, rgb_1_rt], dim=1)
        pose_gt = torch.cat([gt_rgb_0, gt_rgb_1], dim=1)
        pose_out = self.pose_scorer(pose_in)
        pose_target = self.pose_scorer(pose_gt)
        pose_term = ((pose_out[:, :6] - pose_target[:, :6]) ** 2).mean()

        # Temporal-consistency regularizer
        delta = rgb_1 - rgb_0  # use pre-roundtrip rgb so the renderer gets clean grad
        tc_term = (delta ** 2).mean()

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
            + self.weights.lambda_tc * tc_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "tc_term": tc_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


def _seg_distortion_proxy(
    seg_logits_pred: torch.Tensor, seg_logits_gt: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy between predicted seg logits and gt seg logits."""
    log_p = torch.log_softmax(seg_logits_pred, dim=1)
    q = torch.softmax(seg_logits_gt, dim=1)
    return -(q * log_p).sum(dim=1).mean()
