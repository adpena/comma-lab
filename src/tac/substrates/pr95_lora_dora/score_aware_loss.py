# SPDX-License-Identifier: MIT
"""Score-aware Lagrangian for the PR95 LoRA/DoRA substrate.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 6
the contest-domain Lagrangian is
``L = alpha_rate · B/N + beta_seg · d_seg + gamma_pose · sqrt(d_pose)``.

This module re-exports the canonical implementation from
``tac.substrates.sane_hnerv.score_aware_loss`` plus a thin
``Pr95LoRADoRAScoreAwareLoss`` subclass that defaults
``pose_weight_scale=2.71`` per the operating-point-aware rule (PR106 r2
pose_avg ~3.4e-5).

Per Catalog #164 the scorer forward goes through
``score_pair_components`` (canonical) which routes through
``score_aware_common`` and respects ``preprocess_input`` shape semantics.

Per CLAUDE.md "eval_roundtrip — non-negotiable" + Catalog #114 the loss
ALWAYS uses ``apply_eval_roundtrip=True`` (the parent class refuses False).
The noise_std=0.5 Hotz STE fix is preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.substrates.sane_hnerv.score_aware_loss import (
    SaneHnervScoreAwareLoss,
    ScoreAwareLossWeights,
)
from tac.substrates.score_aware_common import score_pair_components

if TYPE_CHECKING:
    import torch


class Pr95LoRADoRAScoreAwareLoss(SaneHnervScoreAwareLoss):
    """Score-aware Lagrangian tuned for PR95 LoRA/DoRA adaptation.

    Defaults are inherited; the LoRA training loop typically tunes
    `pose_weight_scale` upward (2.71 at PR106 r2 operating point per the
    CLAUDE.md "Operating-point-aware rule").

    No new fields; this subclass exists primarily for explicit lane
    attribution so Catalog #164 audits (scorer preprocess discipline) can
    track the substrate that owns the loss callsite.
    """

    def score_pair_components(
        self,
        *,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        rgb_0_rt: torch.Tensor,
        rgb_1_rt: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route PR95 LoRA/DoRA through the canonical scorer contract."""
        return score_pair_components(
            seg_scorer=seg_scorer,
            pose_scorer=pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
        )


__all__ = [
    "Pr95LoRADoRAScoreAwareLoss",
    "ScoreAwareLossWeights",
]
