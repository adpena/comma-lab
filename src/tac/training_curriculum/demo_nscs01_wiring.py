# SPDX-License-Identifier: MIT
"""Demo wiring: NSCS01 + multi-stage curriculum + pause-and-diagnose.

Demonstrates how the canonical pausing-exploits helpers compose with the
NSCS01 nullspace-split renderer (`experiments/train_substrate_nscs01_
nullspace_split_renderer.py`). This module is NOT invoked by the production
trainer; it is the operator-readable RECIPE that future BOLT-ON-on-A1 or
NSCS01-Phase-2 subagents copy when they need multi-stage curriculum + pause-
and-diagnose checkpoints.

Why a separate demo module (not direct trainer edits)?
──────────────────────────────────────────────────────
Per CLAUDE.md "Subagent coherence-by-default" + Catalog #230 bulk-rewrite
ownership: the NSCS01 trainer is a shared resource — sister subagents
(NSCS01-PHASE-2, FREEZING-WAVE, ORTHOGONAL-OPT-AUDIT) may be editing it
concurrently. A demo module shipped IN this package avoids edit collision
while still providing the operator-readable wiring template.

Future production wiring: once NSCS01 enters Phase 2 (per
`feedback_sextet_council_nscs01_phase_2_consensus_20260516.md` Revision #1
head0 probe + Revision #2 smoke), the demo recipe BECOMES the trainer's
production curriculum. The demo is the contract; the trainer's CLI
(``--enable-multi-stage-curriculum``) is the consumption surface.

NSCS01-specific recipe (per design memo + T4 SYMPOSIUM):
────────────────────────────────────────────────────────
NSCS01 exploits SegNet's last-frame-only nullspace structure (the SegNet
scorer slices ``x[:, -1, ...]`` so frame_0 is in the gradient nullspace).
The split-frame architecture has TWO heads: ``frame_0_head`` (gets only pose
+ pixel gradients) and ``frame_1_head`` (gets seg + pose + pixel gradients).

The canonical 3-stage curriculum for NSCS01 (Karpathy + Quantizr pattern):

1. **Stage "anchor"**: pixel-only loss for both heads; LR multiplier 1.0;
   optimizer reset; warmup 100 epochs. The head0 nullspace exploit is NOT
   activated yet — both heads see only pixel reconstruction signal.
2. **Stage "joint"**: pixel + scorer-aware split loss (the
   NullspaceSplitScoreAwareLoss from
   `tac.substrates.nscs01_nullspace_split_renderer.score_aware_loss`); LR
   multiplier 0.5; inherit_lr_reset; 600 epochs. The split-frame gradient
   routing is the canonical exploit.
3. **Stage "distill"**: pixel + scorer + KL-on-logits T=2.0 distillation
   from EMA shadow teacher (THIS substrate's own EMA at end of Stage 2;
   self-distillation); LR multiplier 0.1; inherit; 300 epochs.

Pause-and-diagnose hooks:
- After Stage 1 ("anchor" end): capture head0/head1 weight norms + EMA
  shadow snapshot. Used to verify warmup convergence before joint training.
- After Stage 2 ("joint" end): capture EMA shadow as teacher for Stage 3
  distillation. The teacher snapshot lives in the curriculum manifest.
- After Stage 3 ("distill" end): capture final EMA shadow + diagnostic
  scorer metric for archive packing decision.

Apples-to-apples evidence discipline:
- The demo curriculum config below is a `[derived]` design choice based on
  Quantizr 5-stage canonical pipeline + PR101 gold pattern. Whether 3
  stages vs 5 stages produces sub-A1 score on NSCS01 specifically is
  `[would-need-empirical]` and pending NSCS01 Phase 2 paid smoke.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tac.training_curriculum.multi_stage_curriculum import (
    CurriculumStage,
    StageScheduler,
)
from tac.training_curriculum.pause_and_diagnose import (
    DiagnosticCheckpoint,
    pause_and_capture,
)
from tac.training_curriculum.pause_to_swap_loss import (
    LossSwap,
    swap_loss_at_pause,
)


def build_nscs01_canonical_curriculum() -> StageScheduler:
    """Build the canonical 3-stage NSCS01 curriculum.

    Returns a :class:`StageScheduler` with 1000 total epochs split as:
    - "anchor": 100 epochs (warmup, pixel-only)
    - "joint": 600 epochs (split-frame scorer-aware Lagrangian)
    - "distill": 300 epochs (self-distillation from EMA shadow teacher)

    Returns:
        :class:`StageScheduler`.
    """
    return StageScheduler(
        (
            CurriculumStage(
                name="anchor",
                epochs=100,
                loss_key="pixel_only_both_heads",
                lr_multiplier=1.0,
                optimizer_state_policy="reset",
                notes=(
                    "Warmup pixel-only L1 loss on both heads; "
                    "split-frame exploit NOT yet activated. Catalog #229 PV: "
                    "verify head0+head1 reconstruct frames before scorer-aware."
                ),
            ),
            CurriculumStage(
                name="joint",
                epochs=600,
                loss_key="pixel_plus_split_scorer_lagrangian",
                lr_multiplier=0.5,
                optimizer_state_policy="inherit_lr_reset",
                notes=(
                    "Activate NullspaceSplitScoreAwareLoss: frame_0 head gets "
                    "pose+pixel only; frame_1 head gets seg+pose+pixel. "
                    "Catalog #220 operational mechanism: split-frame "
                    "gradient routing IS the exploit."
                ),
            ),
            CurriculumStage(
                name="distill",
                epochs=300,
                loss_key="kl_distill_T2_self_from_ema_shadow",
                lr_multiplier=0.1,
                optimizer_state_policy="inherit",
                notes=(
                    "KL-on-logits T=2.0 self-distillation from EMA shadow "
                    "teacher captured at end of Stage 2. Per Hinton verbatim "
                    "T4 SYMPOSIUM recommendation; canonical Quantizr 0.33 "
                    "[contest-CUDA] pattern."
                ),
            ),
        )
    )


def head0_grad_norm_metric_fn(
    model: nn.Module,
) -> float:
    """Diagnostic metric: L2 norm of head0 weights.

    Useful at Stage 1→2 transition to verify head0 has learned the warmup
    pixel reconstruction; if this is near zero, the warmup failed.

    NSCS01-specific: assumes model has ``.frame_0_head`` attribute (which
    is the canonical NSCS01 architecture per
    `tac.substrates.nscs01_nullspace_split_renderer.architecture`).
    """
    head0 = getattr(model, "frame_0_head", None)
    if head0 is None:
        return 0.0
    with torch.no_grad():
        sq = torch.tensor(0.0)
        for p in head0.parameters():
            sq = sq + (p.detach() ** 2).sum()
        return float(sq.sqrt())


def head1_grad_norm_metric_fn(
    model: nn.Module,
) -> float:
    """Sister of :func:`head0_grad_norm_metric_fn` for frame_1_head."""
    head1 = getattr(model, "frame_1_head", None)
    if head1 is None:
        return 0.0
    with torch.no_grad():
        sq = torch.tensor(0.0)
        for p in head1.parameters():
            sq = sq + (p.detach() ** 2).sum()
        return float(sq.sqrt())


def nscs01_diagnostic_metric_fns() -> dict[
    str,
    tuple[Callable[[nn.Module], float], str, str],
]:
    """Canonical NSCS01 diagnostic metric registry for pause_and_capture.

    Returns:
        Dict suitable for ``pause_and_capture(metric_fns=...)``.
    """
    return {
        "head0_l2_norm": (
            head0_grad_norm_metric_fn,
            "diagnostic",
            "L2 norm of frame_0_head weights; verifies pixel warmup converged",
        ),
        "head1_l2_norm": (
            head1_grad_norm_metric_fn,
            "diagnostic",
            "L2 norm of frame_1_head weights; verifies scorer-aware activation",
        ),
    }


@dataclass(frozen=True)
class NSCS01CurriculumRecipe:
    """Operator-readable recipe for NSCS01 multi-stage curriculum training.

    This is the OPERATOR INTERFACE; the trainer's CLI (``--enable-multi-
    stage-curriculum``) consumes a serialized form of this dataclass.
    """

    total_epochs: int = 1000
    pause_at_stage_transitions: bool = True
    pause_at_final_epoch: bool = True
    ema_shadow_as_teacher_for_distill: bool = True

    def __post_init__(self) -> None:
        if self.total_epochs != 1000:
            raise ValueError(
                f"NSCS01CurriculumRecipe.total_epochs must be 1000 for the "
                f"canonical 3-stage budget; got {self.total_epochs}. To use "
                "a non-canonical epoch budget, construct StageScheduler "
                "directly."
            )
