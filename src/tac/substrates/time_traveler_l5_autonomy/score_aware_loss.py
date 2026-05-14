# SPDX-License-Identifier: MIT
"""Score-aware Lagrangian for the Time-Traveler L5 Autonomy substrate.

L = alpha * B / N + beta * d_seg + gamma * sqrt(d_pose) + delta * H_pred

where:

* ``B`` is the archive byte count (rate term)
* ``N`` is the contest normalizer (37,545,489 frames-equivalent)
* ``d_seg``, ``d_pose`` are the canonical scorer distortions through the
  Atick-Redlich cooperative-receiver primitive (which itself delegates to
  the canonical ``score_pair_components`` pipeline per Catalog #164)
* ``H_pred`` is the Rao-Ballard predictive-coding-hierarchy auxiliary term:
  the L2 norm of the per-pair predictive-coding residual that the world
  model failed to predict.

The Atick-Redlich cooperative-receiver and Rao-Ballard predictive-coding
primitives now live in :mod:`tac.codec.cooperative_receiver` so any other
substrate (SIREN-pose, NeRV-residual, cross-paradigm composition cells, etc.)
can compose them orthogonally. This module is the IN-TREE consumer that
proved them out and continues to drive their default tuning.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain
Lagrangian (NOT weight-domain proxies like rel_err²).
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally
(this module is the score-domain loss only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.codec.cooperative_receiver import (
    AtickRedlichWeights,
    PredictiveCodingWeights,
    cooperative_receiver_loss,
    predictive_coding_residual_term,
)
from tac.substrates.score_aware_common import CONTEST_POSE_SQRT_WEIGHT


@dataclass(frozen=True)
class TimeTravelerLossWeights:
    """The score-domain Lagrangian weights.

    Defaults match the contest formula. ``delta_predict`` controls the
    auxiliary predictive-coding-hierarchy term; default 0.1 keeps it
    small relative to seg + pose so the substrate still primarily learns
    to satisfy the scorer.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula. PR106 r2 has 2.71x
    pose marginal, but that is an experiment knob not a default."""
    delta_predict: float = 0.1
    """Weight on the predictive-coding-hierarchy auxiliary term."""
    contest_normalizer: float = 37_545_489.0


class TimeTravelerScoreAwareLoss(torch.nn.Module):
    """Score-aware Lagrangian as a torch Module.

    The world-model renderer + foveation + dynamics + per-pair pose codes
    all carry gradient. The per-pair int8-quantized side info is treated
    as a STE-friendly residual (the float pre-quant tensor carries
    gradient; quantization happens at archive build time).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: TimeTravelerLossWeights,
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
        predictive_residual: torch.Tensor | None = None,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: predicted RGB pair tensors (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors (B, 3, H, W) in [0, 255].
            archive_bytes_proxy: scalar tensor with the archive byte count
                (carries no gradient — rate term is non-differentiable; the
                gradient flows through seg + pose + predictive only).
            predictive_residual: optional ``(B, ...)`` tensor of the
                per-pair predictive-coding residual. When provided, its L2
                norm is added to the loss weighted by ``delta_predict``.
                The world model is rewarded for predicting more (smaller
                residual = better world model).
            apply_eval_roundtrip: must be True per CLAUDE.md eval_roundtrip
                non-negotiable; ValueError on False.
            noise_std: per-pixel additive noise during training (CLAUDE.md
                ``noise_std`` discipline; eval-time forward zeroes it).

        Returns:
            ``(loss, parts_dict)`` — loss is scalar tensor with gradient;
            parts_dict has detached tensor values for logging.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")

        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        # Atick-Redlich cooperative-receiver primitive: scorer-conditional
        # loss against the FIXED known SegNet + PoseNet pair, with the
        # mandatory eval-roundtrip applied internally per CLAUDE.md.
        cooperative = cooperative_receiver_loss(
            rgb_0,
            rgb_1,
            gt_rgb_0,
            gt_rgb_1,
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            weights=AtickRedlichWeights(
                beta_seg=self.weights.beta_seg,
                gamma_pose=self.weights.gamma_pose,
                pose_weight_scale=self.weights.pose_weight_scale,
            ),
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        loss = rate_term + cooperative.cooperative_loss

        # Rao-Ballard predictive-coding-hierarchy primitive: penalize the
        # magnitude of the per-pair residual the world model failed to
        # predict.  A good world model makes the residual small.
        predict_term_value: torch.Tensor | float = 0.0
        if predictive_residual is not None and self.weights.delta_predict > 0.0:
            predictive = predictive_coding_residual_term(
                predictive_residual,
                weights=PredictiveCodingWeights(
                    delta_predict=self.weights.delta_predict,
                ),
            )
            loss = loss + predictive.scaled_term
            predict_term_value = predictive.unscaled_residual_l2.detach()
        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": cooperative.seg_term.detach(),
            "pose_term": cooperative.pose_term.detach(),
            "pose_sqrt": cooperative.pose_sqrt.detach(),
            "predict_term": (
                predict_term_value
                if isinstance(predict_term_value, torch.Tensor)
                else torch.tensor(float(predict_term_value))
            ),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "TimeTravelerLossWeights",
    "TimeTravelerScoreAwareLoss",
]
