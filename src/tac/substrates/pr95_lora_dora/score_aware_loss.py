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

from tac.substrates.sane_hnerv.score_aware_loss import (
    SaneHnervScoreAwareLoss,
    ScoreAwareLossWeights,
)


class Pr95LoRADoRAScoreAwareLoss(SaneHnervScoreAwareLoss):
    """Score-aware Lagrangian tuned for PR95 LoRA/DoRA adaptation.

    Defaults are inherited; the LoRA training loop typically tunes
    `pose_weight_scale` upward (2.71 at PR106 r2 operating point per the
    CLAUDE.md "Operating-point-aware rule").

    No new fields; this subclass exists primarily for explicit lane
    attribution so Catalog #164 audits (scorer preprocess discipline) can
    track the substrate that owns the loss callsite.
    """


__all__ = [
    "Pr95LoRADoRAScoreAwareLoss",
    "ScoreAwareLossWeights",
]
