# SPDX-License-Identifier: MIT
"""YUCR score-aware loss — composes UNIWARD cost-map term with canonical scorer pair.

Per CLAUDE.md Catalog #164 every substrate score-aware loss MUST route
through :func:`tac.substrates.score_aware_common.score_pair_components`. The
YUCR loss adds a single new term: the cost-map-weighted L1 reconstruction
penalty that pulls the renderer toward placing residual error on
scorer-blind pixels.

The full loss is:

.. math::

    L_total = L_seg + sqrt(10) * L_pose + L_rate + lambda_yucr * mean(C * |x - x_hat|)

where ``C`` is the per-pixel cost map (high = scorer-blind = cheap to err).
The fourth term is the **inverse-steganalysis penalty** — it tells the
optimizer "if you must put error somewhere, put it where the scorer can't
see it."

Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE", the loss expects the
caller to have already applied
:func:`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`
to the renderer outputs before passing them in (consistent with sister
substrate trainers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_SEG_WEIGHT,
    score_pair_components,
)


@dataclass(frozen=True)
class YUCRLossWeights:
    """Loss-term weights for :class:`YUCRScoreAwareLoss`.

    Defaults match the contest formula. ``lambda_yucr`` controls the
    cost-map-weighted L1 penalty; values in ``[0.01, 0.5]`` are the typical
    operating range. Higher = stronger UNIWARD pull = noise concentrates
    in scorer-blind regions; too high suppresses sensitive-region accuracy.
    """

    seg_weight: float = CONTEST_SEG_WEIGHT
    pose_sqrt_weight: float = CONTEST_POSE_SQRT_WEIGHT
    rate_weight: float = 25.0
    lambda_yucr: float = 0.05
    eps: float = 1e-6

    def __post_init__(self) -> None:  # noqa: D401
        if self.seg_weight <= 0:
            raise ValueError(f"seg_weight must be > 0; got {self.seg_weight}")
        if self.pose_sqrt_weight <= 0:
            raise ValueError(
                f"pose_sqrt_weight must be > 0; got {self.pose_sqrt_weight}"
            )
        if self.rate_weight < 0:
            raise ValueError(f"rate_weight must be >= 0; got {self.rate_weight}")
        if self.lambda_yucr < 0 or self.lambda_yucr > 10.0:
            raise ValueError(
                f"lambda_yucr={self.lambda_yucr} out of safe range [0, 10]"
            )
        if self.eps <= 0 or self.eps > 1.0:
            raise ValueError(f"eps={self.eps} out of range (0, 1]")


class YUCRScoreAwareLoss(nn.Module):
    """Score-aware loss for YUCR substrate.

    Public contract:

    1. Caller applies eval-roundtrip to renderer outputs first.
    2. Caller passes ``rgb_0_rt``, ``rgb_1_rt``, ``gt_rgb_0``, ``gt_rgb_1``,
       ``cost_map``, and the rate term scalar.
    3. Loss returns a dict with ``total`` + per-term breakdown.

    The cost-map term is implemented as
    ``mean(C * (|rgb_0_rt - gt_rgb_0| + |rgb_1_rt - gt_rgb_1|))``. Cost map
    is broadcast across channels.

    Args:
        seg_scorer, pose_scorer: Upstream contest scorers (must expose
            ``preprocess_input`` per Catalog #164).
        weights: :class:`YUCRLossWeights` — defaults to contest formula.
    """

    def __init__(
        self,
        *,
        seg_scorer: nn.Module,
        pose_scorer: nn.Module,
        weights: YUCRLossWeights | None = None,
    ) -> None:
        super().__init__()
        if not callable(getattr(seg_scorer, "preprocess_input", None)):
            raise ValueError(
                "seg_scorer must expose preprocess_input(pair_btchw) per Catalog #164"
            )
        if not callable(getattr(pose_scorer, "preprocess_input", None)):
            raise ValueError(
                "pose_scorer must expose preprocess_input(pair_btchw) per Catalog #164"
            )
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights or YUCRLossWeights()

    def forward(
        self,
        *,
        rgb_0_rt: torch.Tensor,
        rgb_1_rt: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        cost_map: torch.Tensor,
        rate_term: torch.Tensor | float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the full YUCR score-aware loss.

        Args:
            rgb_0_rt, rgb_1_rt: Roundtripped reconstruction pair, ``(B, 3, H, W)``.
            gt_rgb_0, gt_rgb_1: Ground-truth pair.
            cost_map: ``(B, H, W)`` or ``(H, W)`` cost map (will be broadcast).
            rate_term: Scalar (or 0-D tensor) representing the archive-rate
                term in the contest formula.

        Returns:
            Dict with keys ``total``, ``seg``, ``pose``, ``rate``, ``yucr``.
        """
        if rgb_0_rt.shape != rgb_1_rt.shape:
            raise ValueError(
                f"rgb_0_rt and rgb_1_rt shapes differ: "
                f"{tuple(rgb_0_rt.shape)} vs {tuple(rgb_1_rt.shape)}"
            )
        if gt_rgb_0.shape != rgb_0_rt.shape:
            raise ValueError(
                f"gt_rgb_0 shape {tuple(gt_rgb_0.shape)} != rgb_0_rt shape "
                f"{tuple(rgb_0_rt.shape)}"
            )
        if cost_map.dim() == 2:
            cost_map = cost_map.unsqueeze(0).expand(rgb_0_rt.shape[0], -1, -1)
        if cost_map.dim() != 3:
            raise ValueError(
                f"cost_map must be 2D or 3D; got {tuple(cost_map.shape)}"
            )
        if cost_map.shape[-2:] != rgb_0_rt.shape[-2:]:
            cost_map = torch.nn.functional.interpolate(
                cost_map.unsqueeze(1).float(),
                size=rgb_0_rt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        seg_term, pose_term = score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
        )
        # Pose term enters via sqrt(pose_avg) per the contest formula.
        pose_sqrt = torch.sqrt(pose_term.clamp_min(self.weights.eps))
        rate_t = (
            rate_term
            if isinstance(rate_term, torch.Tensor)
            else torch.tensor(float(rate_term), device=rgb_0_rt.device)
        )

        # YUCR cost-map-weighted L1 — broadcast cost across channels.
        cost_bchw = cost_map.unsqueeze(1)  # (B, 1, H, W)
        l1_pair = (
            (rgb_0_rt - gt_rgb_0).abs() + (rgb_1_rt - gt_rgb_1).abs()
        )  # (B, 3, H, W)
        yucr_term = (cost_bchw * l1_pair).mean()

        total = (
            self.weights.seg_weight * seg_term
            + self.weights.pose_sqrt_weight * pose_sqrt
            + self.weights.rate_weight * rate_t
            + self.weights.lambda_yucr * yucr_term
        )

        return {
            "total": total,
            "seg": seg_term.detach(),
            "pose": pose_term.detach(),
            "rate": rate_t.detach() if isinstance(rate_t, torch.Tensor) else rate_t,
            "yucr": yucr_term.detach(),
        }


__all__ = ["YUCRLossWeights", "YUCRScoreAwareLoss"]
