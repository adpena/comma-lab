# SPDX-License-Identifier: MIT
"""pr101_lc_v2_clone score-aware Lagrangian — same form as sane_hnerv.

Per CLAUDE.md HNeRV parity discipline L6: the loss MUST be score-aware. The
form is::

    L(theta) = alpha * B(theta) / N + beta * d_seg(theta)
             + gamma * pose_weight_scale * sqrt(d_pose(theta))

where ``d_seg`` and ``d_pose`` are gradient-reachable via
``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``
+ the patched ``rgb_to_yuv6`` (PR #95/#106 monkey-patch contract).

The clone shares the loss form with sane_hnerv so forensic
apples-to-apples comparisons can use the same training Lagrangian; only the
substrate (architecture + archive grammar) differs.

CLAUDE.md compliance:
* eval_roundtrip=True (NON-NEGOTIABLE)
* No silent CUDA fallback (caller supplies device + scorers)
* No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)


@dataclass(frozen=True)
class Pr101LcV2ScoreAwareLossWeights:
    """The (alpha, beta, gamma) of the score-domain Lagrangian.

    Defaults match the Phase 2 council values; tunable via the trainer CLI.
    """

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (alpha_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    """PoseNet term weight. Contest default is sqrt(10) times sqrt(d_pose)."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py (37,545,489 = 600 * 2 * 3 * 874 * 1164 / scale).

    Matches sane_hnerv's value so the two substrates share Lagrangian scale.
    """


class Pr101LcV2CloneScoreAwareLoss(torch.nn.Module):
    """The Lagrangian as a torch Module so trainers can ``loss.forward(...)``.

    Trainer usage::

        loss_fn = Pr101LcV2CloneScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=Pr101LcV2ScoreAwareLossWeights(...),
        )
        # patch yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip
        # rule:
        # tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()
        ...
        loss, parts = loss_fn(
            rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy,
        )
        loss.backward()

    ``rgb_*`` come from the substrate (after eval-roundtrip simulation); they
    must be in [0, 1] (the substrate emits [0, 255]; the trainer divides by
    255.0 before calling this). ``archive_bytes_proxy`` is a scalar tensor for
    the rate term — usually a closed-form upper bound on the rate (Balle R(D)
    bound), since post-export bytes are non-differentiable.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: Pr101LcV2ScoreAwareLossWeights,
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
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian on a pair of rendered frames."""
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        # Lazy import to keep this module's import-time cheap.
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        del noise_std  # unused at this revision; STE is deterministic
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

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
