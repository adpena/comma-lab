"""balle_renderer score-aware Lagrangian (β).

    L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ)) + λ_hp·R_hyperprior(θ)

The Fields-medal grand council 2026-05-12 score-domain Lagrangian per
CLAUDE.md HNeRV parity discipline L6, extended for β with the Ballé 2018
hyperprior rate term:

* `d_seg(θ)`: contest SegNet distortion, gradient-reachable via
  ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``
  + ``patch_upstream_yuv6_globally`` (PR #95/#106 monkey-patch contract).
* `d_pose(θ)`: contest PoseNet distortion via the same wiring; the
  sqrt-transform is the contest scorer's pose-axis contribution shape.
* `B(θ)`: post-export archive size in bytes; differentiability is via
  an arithmetic-coder proxy (Ballé closed-form rate bound +
  hyper-rate term, NOT post-export bytes which are non-differentiable).
* `R_hyperprior(θ) = E[-log p_z(w_hat)] + E[-log p_y(z_hat | σ(w_hat))]`
  comes directly from the substrate forward pass (architecture returns
  the rate_components dict). This is the *score-aware* version of the
  Ballé 2018 rate term — its gradient steers θ toward weight
  distributions that arithmetic-code well.

The PR106 r2 operating point (pose_avg ~ 3.4e-5) flips marginal-value to
2.71× pose-vs-seg; the loss exposes ``pose_weight_scale`` like α.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std=0.5 (Hotz STE fix)
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
class BalleScoreAwareLossWeights:
    """The (α, β, γ, λ_hp) of the β score-domain Lagrangian.

    Defaults per the Phase 2 council; tunable via the trainer CLI in the
    follow-up subagent.
    """

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (α_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    """PoseNet term weight. Contest default is sqrt(10) times sqrt(d_pose)."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    lambda_hyperprior: float = 0.5
    """Weight on the Ballé hyperprior rate term R_hyperprior.

    Balances "drive rate down" vs "let the scorer-axis terms dominate
    so the substrate doesn't collapse to a uniform distribution that
    codes well but renders nothing." 0.5 is a conservative start; the
    follow-up subagent should sweep [0.1, 1.0]."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class BalleRendererScoreAwareLoss(torch.nn.Module):
    """The β Lagrangian as a torch Module.

    Trainer usage (post-α-anchor, follow-up subagent)::

        # PATCH yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip
        # rule:
        # tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()

        loss_fn = BalleRendererScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet,
            weights=BalleScoreAwareLossWeights(...),
        )
        ...
        rgb_0, rgb_1, rate_components = model(pair_indices)
        loss, parts = loss_fn(
            rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy, rate_components,
        )
        loss.backward()

    ``rate_components`` is the dict returned by
    ``BalleRendererSubstrate.forward`` containing ``hyper_rate`` /
    ``main_rate`` / ``total_rate``. Its gradient is the score-aware steer
    on the substrate weights toward arithmetic-codeable distributions.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: BalleScoreAwareLossWeights,
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
        rate_components: dict[str, torch.Tensor],
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian for the β substrate.

        Returns:
            ``(loss_total, parts)`` where parts is a dict with named
            components for logging (``rate_term``, ``seg_term``,
            ``pose_term``, ``hyperprior_rate_term``, ``loss_total``).
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        for key in ("hyper_rate", "main_rate", "total_rate"):
            if key not in rate_components:
                raise ValueError(
                    f"rate_components missing required key {key!r}; "
                    "ensure substrate forward returned the dict"
                )

        # Lazy import to keep this module's import-time cheap
        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)
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

        rate_term = (
            self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer
        )

        hp_rate_term = rate_components["total_rate"]

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + self.weights.lambda_hyperprior * hp_rate_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "hyperprior_rate_term": hp_rate_term.detach(),
            "hyper_rate": rate_components["hyper_rate"].detach(),
            "main_rate": rate_components["main_rate"].detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
