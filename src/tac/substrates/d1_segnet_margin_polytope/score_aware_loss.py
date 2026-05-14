# SPDX-License-Identifier: MIT
"""D1 score-aware loss — margin-preserving hinge + canonical scorer pair.

Per CLAUDE.md Catalog #164 every substrate score-aware loss MUST route
through :func:`tac.substrates.score_aware_common.score_pair_components`.
The D1 loss adds ONE new term: the **margin-preserving hinge** that
penalizes pushing pixels below the safe margin threshold during training
so the argmax decision-frontier stays stable.

The full loss is:

.. math::

    L_\\text{total} = L_\\text{seg} + \\sqrt{10} \\cdot L_\\text{pose}
    + L_\\text{rate}
    + \\lambda_\\text{d1} \\cdot \\text{mean}(\\max(0,
    \\text{margin\\_threshold} - m(x, y)))

where ``m(x, y)`` is the per-pixel logit margin (top1 - top2 from the
SegNet output on the reconstructed frame-1 pair) and the hinge term
pulls the optimizer toward MAXIMIZING the safe-polytope volume in
reconstruction space (per the deep-math memo §3.6 polytope-interior
geometry).

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
    stage_frame_pair,
)


@dataclass(frozen=True)
class D1PolytopeLossWeights:
    """Loss-term weights for :class:`D1PolytopeScoreAwareLoss`.

    Defaults match the contest formula. ``lambda_d1`` controls the
    margin-preserving hinge penalty; values in ``[0.001, 0.5]`` are the
    typical operating range. Higher = stronger argmax-stability pull;
    too high suppresses generic reconstruction.

    ``margin_threshold`` defines the hinge cutoff — below this margin the
    penalty is active. Defaults to 0.1 which is a calibrated empirical
    midpoint (the SegNet 5-class logit-margin histogram on typical
    comma2k19 frames has the boundary-pixel mode at ~0.05 and the
    interior mode at ~1.0; 0.1 sits at the boundary-interior transition).
    """

    seg_weight: float = CONTEST_SEG_WEIGHT
    pose_sqrt_weight: float = CONTEST_POSE_SQRT_WEIGHT
    rate_weight: float = 25.0
    lambda_d1: float = 0.05
    margin_threshold: float = 0.1
    eps: float = 1e-6

    def __post_init__(self) -> None:  # noqa: D401
        if self.seg_weight <= 0:
            raise ValueError(f"seg_weight must be > 0; got {self.seg_weight}")
        if self.pose_sqrt_weight <= 0:
            raise ValueError(
                f"pose_sqrt_weight must be > 0; got {self.pose_sqrt_weight}"
            )
        if self.rate_weight < 0:
            raise ValueError(
                f"rate_weight must be >= 0; got {self.rate_weight}"
            )
        if self.lambda_d1 < 0 or self.lambda_d1 > 10.0:
            raise ValueError(
                f"lambda_d1={self.lambda_d1} out of safe range [0, 10]"
            )
        if self.margin_threshold < 0 or self.margin_threshold > 10.0:
            raise ValueError(
                f"margin_threshold={self.margin_threshold} out of "
                "range [0, 10]"
            )
        if self.eps <= 0 or self.eps > 1.0:
            raise ValueError(f"eps={self.eps} out of range (0, 1]")


class D1PolytopeScoreAwareLoss(nn.Module):
    """Score-aware loss for D1 substrate.

    Public contract:

    1. Caller applies eval-roundtrip to renderer outputs first.
    2. Caller passes ``rgb_0_rt``, ``rgb_1_rt``, ``gt_rgb_0``, ``gt_rgb_1``,
       and the rate term scalar.
    3. Loss returns a dict with ``total`` + per-term breakdown including
       the new ``d1_margin_hinge`` term.

    The margin-preserving hinge is computed by re-running the SegNet
    forward on the reconstructed frame-1 (via the canonical
    :func:`stage_frame_pair` + ``preprocess_input`` route per Catalog
    #164) and taking ``mean(relu(margin_threshold - margin))``. This
    pulls the optimizer toward LARGER per-pixel margins which in turn
    EXPAND the safe-polytope-interior the inflate-time noise allocator
    can exploit.

    Args:
        seg_scorer, pose_scorer: Upstream contest scorers (must expose
            ``preprocess_input`` per Catalog #164).
        weights: :class:`D1PolytopeLossWeights` — defaults to contest
            formula plus the margin-preserving hinge.
    """

    def __init__(
        self,
        *,
        seg_scorer: nn.Module,
        pose_scorer: nn.Module,
        weights: D1PolytopeLossWeights | None = None,
    ) -> None:
        super().__init__()
        if not callable(getattr(seg_scorer, "preprocess_input", None)):
            raise ValueError(
                "seg_scorer must expose preprocess_input(pair_btchw) per "
                "Catalog #164"
            )
        if not callable(getattr(pose_scorer, "preprocess_input", None)):
            raise ValueError(
                "pose_scorer must expose preprocess_input(pair_btchw) per "
                "Catalog #164"
            )
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights or D1PolytopeLossWeights()

    def _compute_margin_hinge(
        self,
        rgb_0_rt: torch.Tensor,
        rgb_1_rt: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-preserving hinge: penalize pixels below margin_threshold.

        Runs SegNet on the reconstructed pair (via canonical preprocess)
        and extracts the top1-minus-top2 logit margin. The hinge is
        ``mean(relu(margin_threshold - margin))`` — only active for
        boundary pixels with small margins.
        """
        pair_pred = stage_frame_pair(rgb_0_rt, rgb_1_rt)
        # Canonical SegNet preprocess: slices frame 1, resamples to (384, 512).
        seg_input = self.seg_scorer.preprocess_input(pair_pred)
        logits = self.seg_scorer(seg_input)
        if logits.dim() != 4 or logits.shape[1] < 2:
            raise ValueError(
                "SegNet logits shape unexpected (need (B, C>=2, H, W)); "
                f"got {tuple(logits.shape)}"
            )
        top2_values, _ = torch.topk(
            logits, k=2, dim=1, largest=True, sorted=True
        )
        margin = (top2_values[:, 0] - top2_values[:, 1]).clamp_min(0.0)
        hinge = torch.relu(self.weights.margin_threshold - margin)
        return hinge.mean()

    def forward(
        self,
        *,
        rgb_0_rt: torch.Tensor,
        rgb_1_rt: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        rate_term: torch.Tensor | float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the full D1 score-aware loss.

        Args:
            rgb_0_rt, rgb_1_rt: Roundtripped reconstruction pair,
                ``(B, 3, H, W)``.
            gt_rgb_0, gt_rgb_1: Ground-truth pair.
            rate_term: Scalar (or 0-D tensor) representing the
                archive-rate term in the contest formula.

        Returns:
            Dict with keys ``total``, ``seg``, ``pose``, ``rate``,
            ``d1_margin_hinge``.
        """
        if rgb_0_rt.shape != rgb_1_rt.shape:
            raise ValueError(
                f"rgb_0_rt and rgb_1_rt shapes differ: "
                f"{tuple(rgb_0_rt.shape)} vs {tuple(rgb_1_rt.shape)}"
            )
        if gt_rgb_0.shape != rgb_0_rt.shape:
            raise ValueError(
                f"gt_rgb_0 shape {tuple(gt_rgb_0.shape)} != rgb_0_rt "
                f"shape {tuple(rgb_0_rt.shape)}"
            )

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

        # D1 margin-preserving hinge — only active for boundary pixels.
        margin_hinge = self._compute_margin_hinge(rgb_0_rt, rgb_1_rt)

        total = (
            self.weights.seg_weight * seg_term
            + self.weights.pose_sqrt_weight * pose_sqrt
            + self.weights.rate_weight * rate_t
            + self.weights.lambda_d1 * margin_hinge
        )

        return {
            "total": total,
            "seg": seg_term.detach(),
            "pose": pose_term.detach(),
            "rate": (
                rate_t.detach() if isinstance(rate_t, torch.Tensor) else rate_t
            ),
            "d1_margin_hinge": margin_hinge.detach(),
        }


__all__ = ["D1PolytopeLossWeights", "D1PolytopeScoreAwareLoss"]
