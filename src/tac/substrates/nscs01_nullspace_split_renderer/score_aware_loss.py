# SPDX-License-Identifier: MIT
"""NSCS01 split-head score-aware Lagrangian.

L = alpha * B/N
  + beta_seg * d_seg(SegNet(frame_1_pred), SegNet(gt_frame_1))
  + gamma_pose * sqrt(d_pose(PoseNet(frame_0_pred, frame_1_pred),
                              PoseNet(gt_frame_0, gt_frame_1)))
  + lambda_pixel_0 * MSE(frame_0_pred, gt_frame_0)
  + lambda_pixel_1 * MSE(frame_1_pred, gt_frame_1)

This loss is INTENTIONALLY a fork of the canonical
``score_pair_components_dispatch`` per the design memo §4 — the canonical
helper does not expose split gradient routing per frame. NSCS01 still
honors Catalog #164 because BOTH scorers are still called via
``preprocess_input`` (the canonical low-level
``scorer_loss_terms_btchw`` from ``tac.losses.core`` does the
preprocess routing inside).

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian.
Per CLAUDE.md "eval_roundtrip — non-negotiable": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally
(this module is the score-domain loss only).

UNIQUE-vs-canonical decision (per standing directive
``feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md``):
this loss is FORK because the split-frame gradient routing is the entire
NSCS01 mechanism. The canonical helper at
``tac.substrates.score_aware_common.score_pair_components_dispatch`` would
homogenize the routing and erase the exploit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.losses import scorer_loss_terms_btchw
from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
    ScoreAwareScorerContractError,
    stage_frame_pair,
)

CONTEST_NORMALIZER: float = 37_545_489.0


_RGB_255_DOMAIN_EPS = 1e-3


def _validate_rgb_255_domain(name: str, tensor: torch.Tensor) -> None:
    """Fail closed when callers pass nonzero unit-domain RGB by mistake."""
    detached = tensor.detach()
    if not torch.isfinite(detached).all():
        raise ValueError(f"{name} must contain finite RGB values")
    min_value = float(detached.min().item())
    max_value = float(detached.max().item())
    if min_value < -_RGB_255_DOMAIN_EPS or max_value > 255.0 + _RGB_255_DOMAIN_EPS:
        raise ValueError(
            f"{name} must be in [0, 255]; got min={min_value} max={max_value}"
        )
    if max_value > 0.0 and max_value <= 1.0:
        raise ValueError(
            f"{name} appears to be unit-domain RGB; expected [0, 255] scorer input"
        )


@dataclass(frozen=True)
class NullspaceSplitLossWeights:
    """Lagrangian weights for the NSCS01 split-head loss.

    Defaults match the contest formula. ``lambda_pixel_0`` is intentionally
    smaller than ``lambda_pixel_1`` because frame_0 only needs to be "good
    enough for PoseNet" — pixel-MSE on frame_0 is the auxiliary anchor that
    keeps gradients flowing early in training, but the dominant frame_0
    signal is the pose term.
    """

    alpha_rate: float = CONTEST_RATE_WEIGHT
    beta_seg: float = CONTEST_SEG_WEIGHT
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula. At PR101 frontier the
    pose marginal is roughly 2.71x SegNet's (CLAUDE.md "SegNet vs PoseNet"
    section). Knob, not default."""
    lambda_pixel_0: float = 0.05
    """Auxiliary pixel-MSE weight on frame_0 (small; PoseNet drives the head)."""
    lambda_pixel_1: float = 0.20
    """Auxiliary pixel-MSE weight on frame_1 (larger; SegNet argmax stability)."""
    contest_normalizer: float = CONTEST_NORMALIZER


class NullspaceSplitScoreAwareLoss(torch.nn.Module):
    """NSCS01 split-head score-domain Lagrangian as a torch Module.

    Per the design memo §4: the loss FORKS from the canonical
    ``score_pair_components_dispatch`` because the SegNet preprocess slice
    means ``seg_term`` is mathematically independent of ``frame_0_pred``;
    the loss makes that explicit by:

    1. computing scorer terms via ``scorer_loss_terms_btchw`` (the same
       low-level function the canonical helper uses);
    2. adding split per-frame pixel-MSE auxiliary terms with different
       weights (frame_1 weight > frame_0 weight, expressing the asymmetric
       importance);
    3. NOT adding a frame_0-specific seg-style term (would be the bug).

    The split gradient routing happens through autograd: ``seg_term`` only
    depends on ``frame_1_pred`` (because SegNet sees only the last frame),
    so its backward writes ZERO gradient to ``frame_0_head`` parameters
    automatically.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: NullspaceSplitLossWeights,
    ) -> None:
        super().__init__()
        # Catalog #164 contract: both scorers must expose preprocess_input;
        # scorer_loss_terms_btchw will fail loud otherwise.
        if not callable(getattr(seg_scorer, "preprocess_input", None)):
            raise ScoreAwareScorerContractError(
                "seg_scorer must expose preprocess_input(pair_btchw)"
            )
        if not callable(getattr(pose_scorer, "preprocess_input", None)):
            raise ScoreAwareScorerContractError(
                "pose_scorer must expose preprocess_input(pair_btchw)"
            )
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        *,
        frame_0_pred: torch.Tensor,
        frame_1_pred: torch.Tensor,
        gt_frame_0: torch.Tensor,
        gt_frame_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the NSCS01 split-head score-domain Lagrangian.

        Args:
            frame_0_pred: predicted frame_0 ``(B, 3, H, W)`` in [0, 255]
                from ``NullspaceSplitRenderer.frame_0_head``.
            frame_1_pred: predicted frame_1 ``(B, 3, H, W)`` in [0, 255]
                from ``NullspaceSplitRenderer.frame_1_head``.
            gt_frame_0, gt_frame_1: ground-truth RGB pair tensors
                ``(B, 3, H, W)`` in [0, 255].
            archive_bytes_proxy: scalar tensor with the COMBINED archive byte
                count. Carries no gradient.
            apply_eval_roundtrip: must be True; ValueError on False.
            noise_std: per-pixel additive noise during training.

        Returns:
            ``(loss, parts_dict)``.

        Raises:
            ValueError: if eval_roundtrip is False, noise_std is negative,
                or any RGB tensor is outside [0, 255].
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        for name, tensor in (
            ("frame_0_pred", frame_0_pred),
            ("frame_1_pred", frame_1_pred),
            ("gt_frame_0", gt_frame_0),
            ("gt_frame_1", gt_frame_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        # Lazy import per the canonical pattern.
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        rgb_0 = frame_0_pred
        rgb_1 = frame_1_pred
        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        # Build the canonical (B, T=2, C=3, H, W) pair tensors and route
        # through the canonical low-level scorer helper. Catalog #164
        # honored: scorer_loss_terms_btchw calls each scorer's
        # preprocess_input internally.
        pair_pred = stage_frame_pair(rgb_0_rt, rgb_1_rt)
        pair_gt = stage_frame_pair(gt_frame_0, gt_frame_1)

        _rate_dummy, pose_term, seg_term = scorer_loss_terms_btchw(
            pair_pred,
            pair_gt,
            self.pose_scorer,
            self.seg_scorer,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))

        # Per-frame pixel-MSE auxiliaries (in [0, 255] domain → divide by
        # 255**2 to keep magnitude commensurate with the scorer terms).
        pixel_0_l2 = (frame_0_pred - gt_frame_0).pow(2).mean() / (255.0 ** 2)
        pixel_1_l2 = (frame_1_pred - gt_frame_1).pow(2).mean() / (255.0 ** 2)

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + self.weights.lambda_pixel_0 * pixel_0_l2
            + self.weights.lambda_pixel_1 * pixel_1_l2
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "pixel_0_l2": pixel_0_l2.detach(),
            "pixel_1_l2": pixel_1_l2.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "CONTEST_NORMALIZER",
    "NullspaceSplitLossWeights",
    "NullspaceSplitScoreAwareLoss",
]
