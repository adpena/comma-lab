"""hybrid_renderer_residual score-aware Lagrangian (γ).

    L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ)) + λ_res·||c||_1

The Fields-medal grand council 2026-05-12 score-domain Lagrangian per
CLAUDE.md HNeRV parity discipline L6, extended for γ with the residual-basis
sparsity term:

* `d_seg(θ)`: contest SegNet distortion, gradient-reachable via
  ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``
  + ``patch_upstream_yuv6_globally`` (PR #95/#106 monkey-patch contract).
* `d_pose(θ)`: contest PoseNet distortion via the same wiring; the
  sqrt-transform is the contest scorer's pose-axis contribution shape.
* `B(θ)`: post-export archive size in bytes; differentiability is via
  an arithmetic-coder proxy (Ballé closed-form rate bound). The residual
  sparsity ALSO indirectly drives this term (smaller k -> smaller archive).
* `||c||_1`: L_1 norm of the active residual coefficients — the
  pose-axis attack vector at the PR106 r2 operating point. The trainer
  sweeps λ_res to find the sparsity threshold where the residual stops
  earning its bytes.

The PR106 r2 operating point (pose_avg ~ 3.4e-5) flips marginal-value to
2.71× pose-vs-seg; the loss exposes ``pose_weight_scale`` like α + β.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std=0.5 (Hotz STE fix)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HybridResidualScoreAwareLossWeights:
    """The (α, β, γ, λ_res) of the γ score-domain Lagrangian.

    Defaults per the Phase 2 council SKETCH; tunable via the follow-up
    subagent's trainer CLI.
    """

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (α_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = 1.0
    """PoseNet term weight (multiplied by sqrt(d_pose))."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    lambda_residual: float = 0.1
    """Weight on the residual sparsity term ||c||_1.

    Balances "drive sparsity up (smaller archive + cleaner pose-axis
    attack)" vs "let the renderer + residual together actually beat the
    base-renderer score." 0.1 is a conservative SKETCH start; the
    follow-up subagent should sweep [0.0, 1.0]."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class HybridRendererResidualScoreAwareLoss(torch.nn.Module):
    """The γ Lagrangian as a torch Module.

    Trainer usage (post-α-anchor, follow-up subagent)::

        # PATCH yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip
        # rule:
        # tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()

        loss_fn = HybridRendererResidualScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet,
            weights=HybridResidualScoreAwareLossWeights(...),
        )
        ...
        rgb_0, rgb_1, residual_l1 = model(pair_indices)
        loss, parts = loss_fn(
            rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy, residual_l1,
        )
        loss.backward()
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: HybridResidualScoreAwareLossWeights,
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
        residual_l1: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian for the γ substrate.

        Returns:
            ``(loss_total, parts)`` where parts is a dict with named
            components for logging (``rate_term``, ``seg_term``,
            ``pose_term``, ``residual_term``, ``loss_total``).
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        # Lazy import to keep this module's import-time cheap
        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        # ``noise_std`` is reserved for a future variant of the eval-roundtrip
        # (currently the canonical ``apply_eval_roundtrip_during_training``
        # uses ``Uint8STE`` deterministically — no additive noise). Kept as a
        # kwarg for forward-compat so the trainer's CLI surface remains stable.
        del noise_std  # unused at this revision; STE is deterministic
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        # SegNet on LAST frame of pair (rgb_1); upstream contract is (B, T=1, C, H, W).
        seg_out = self.seg_scorer(rgb_1_rt.unsqueeze(1))
        seg_gt = self.seg_scorer(gt_rgb_1.unsqueeze(1))
        seg_term = _seg_distortion_proxy(seg_out, seg_gt)

        # PoseNet on the (rgb_0, rgb_1) pair concatenated.
        pose_in = torch.cat([rgb_0_rt, rgb_1_rt], dim=1)
        pose_gt = torch.cat([gt_rgb_0, gt_rgb_1], dim=1)
        pose_out = self.pose_scorer(pose_in)
        pose_target = self.pose_scorer(pose_gt)
        pose_term = ((pose_out[:, :6] - pose_target[:, :6]) ** 2).mean()

        rate_term = (
            self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer
        )

        residual_term = self.weights.lambda_residual * residual_l1

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + residual_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "residual_term": residual_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


def _seg_distortion_proxy(
    seg_logits_pred: torch.Tensor, seg_logits_gt: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy between predicted seg logits and gt seg logits.

    Direct argmax-disagreement isn't differentiable; soft KL on softmaxed
    logits is the canonical surrogate (T=2.0 ablation flag is the trainer's
    follow-up).
    """
    log_p = torch.log_softmax(seg_logits_pred, dim=1)
    q = torch.softmax(seg_logits_gt, dim=1)
    return -(q * log_p).sum(dim=1).mean()
