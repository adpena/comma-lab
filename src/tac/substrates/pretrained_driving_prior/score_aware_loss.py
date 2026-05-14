# SPDX-License-Identifier: MIT
"""Score-aware Lagrangian for the pre-trained driving prior substrate.

    L = alpha * B / N + beta * d_seg + gamma * sqrt(d_pose) + delta_prior * L_prior

where:

* ``B`` is the archive byte count (codebook + renderer + residual + meta)
* ``N`` is the contest normalizer (37,545,489)
* ``d_seg``, ``d_pose`` are canonical scorer distortions via
  :func:`tac.substrates.score_aware_common.score_pair_components` (Catalog #164)
* ``L_prior`` is the codebook soft-prior penalty (see
  :class:`prior_application.DashcamPriorLoss`)

**Atick-Redlich cooperative-receiver hook:** ``score_pair_components`` runs
the eval-roundtrip-corrected RGB through the FIXED contest scorer (SegNet +
PoseNet). This is mathematically equivalent to maximizing ``MI(B; S(B))``
subject to the rate constraint.

**Predictive-coding interpretation:** the prior term encourages the renderer's
RGB to remain within the dashcam-distribution manifold; the per-pair residual
encodes the contest-specific delta. Rao-Ballard 1999 / Friston free-energy:
the prior + residual decomposition minimizes total description length.

**Operating-point awareness** (per CLAUDE.md "SegNet vs PoseNet importance
operating-point dependent"): at PR106 r2 (pose_avg ~3.4e-5) pose marginal
sensitivity is 2.71× SegNet's; at older 1.x scores SegNet dominates. The
``pose_weight_scale`` knob lets the trainer tilt toward pose-marginal at
saturated operating points without hardcoding the asymmetry.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian
(NOT weight-domain proxies like rel_err²).
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": ``apply_eval_roundtrip=True``
is the only acceptable training mode (the L1 NeRV/HNeRV trainer-parity
contract enforced by :mod:`tac.hnerv_training_parity_guard`).
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally
(this module is the score-domain loss only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)

if TYPE_CHECKING:
    from tac.substrates.pretrained_driving_prior.prior_application import (
        DashcamPriorLoss,
    )


@dataclass(frozen=True)
class DrivingPriorLossWeights:
    """Score-domain Lagrangian weights for the driving-prior substrate.

    Defaults match the contest formula (alpha=25, beta=100). ``delta_prior``
    controls the soft-prior strength; default 0.05 keeps it well below
    the score terms so the prior shapes (not dictates) the solution.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula. PR106 r2 has 2.71x
    pose marginal at saturated operating points."""
    delta_prior: float = 0.05
    """Weight on the codebook soft-prior penalty."""
    contest_normalizer: float = 37_545_489.0


class DrivingPriorScoreAwareLoss(torch.nn.Module):
    """Score-aware Lagrangian as a torch Module.

    Composition surface: ``score_pair_components`` (Catalog #164 canonical
    helper) handles SegNet/PoseNet preprocess + forward; the prior is
    added as an auxiliary term. The renderer + per-pair residual carry
    gradient; the codebook is FROZEN (registered as non-persistent buffers
    inside ``prior_loss``).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        prior_loss: DashcamPriorLoss,
        weights: DrivingPriorLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.prior_loss = prior_loss
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
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: predicted RGB pair tensors (B, 3, H, W) in [0, 1]
                or [0, 255] (canonical helper handles both).
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors with same range.
            archive_bytes_proxy: scalar tensor with the archive byte count.
            apply_eval_roundtrip: must be True per CLAUDE.md non-negotiable.
            noise_std: per-pixel additive noise during training (zero at eval).
            gt_pose_batch / gt_seg_batch / gt_seg_already_probs: optional F3
                GTScorerCache batch (per-pair-index lookup from
                :meth:`tac.training_optimization.GTScorerCache.lookup`). When
                ALL THREE are provided, the dispatch helper routes through
                ``score_pair_components_with_cache`` (~50%% scorer-compute
                savings); when ALL THREE are None, falls back to the
                GT-forward path. Partial cache args raise via the dispatcher.
                Per Council omnibus Decision 13 PROCEED Option C 2026-05-14:
                PDP substrate-side wire-in defaults OFF (substrate authors
                opt in via composition). The kwargs are accepted but the
                dispatcher's None-default preserves byte-faithful behavior.

        Returns:
            ``(loss, parts_dict)`` — loss is a scalar tensor with gradient;
            parts_dict has detached tensor values for logging.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")

        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        # F3 GTScorerCache routing per F3-BACKPORT-WAVE-V2 + Council omnibus
        # Decision 13 PROCEED Option C 2026-05-14. The dispatch helper picks
        # the cached path when ALL THREE cache kwargs are non-None, otherwise
        # falls back to the canonical GT-forward path used by Catalog #164.
        # The two paths are mathematically identical.
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
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))

        # Soft prior term (uses the eval-roundtripped predictions to stay
        # consistent with the score-domain Lagrangian — the codebook
        # projection is defined post-roundtrip).
        prior_rgb = (rgb_0_rt + rgb_1_rt) * 0.5
        # Normalize to [0, 1] if the upstream range is [0, 255] (canonical
        # helper accepts both ranges; prior projection expects [0, 1]).
        if prior_rgb.max() > 1.5:  # heuristic: in [0, 255] range
            prior_rgb = prior_rgb / 255.0
        prior_total, prior_parts = self.prior_loss(prior_rgb)

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + self.weights.delta_prior * prior_total
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "prior_total": prior_total.detach(),
            "loss_total": loss.detach(),
        }
        # Merge in per-prior-term values.
        for k, v in prior_parts.items():
            parts[k] = v
        return loss, parts


__all__ = [
    "DrivingPriorLossWeights",
    "DrivingPriorScoreAwareLoss",
]
