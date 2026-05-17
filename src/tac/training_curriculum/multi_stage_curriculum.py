# SPDX-License-Identifier: MIT
"""Multi-stage curriculum scheduler — Karpathy + Hugging Face PEFT pattern.

The canonical staged-curriculum pattern that PR101 gold + Quantizr 0.33 both
used: training is NOT a single monolithic loop with a weighted-sum loss; it's
N discrete stages with explicit transitions. Each stage has its own loss,
optimizer-state policy (reset / inherit / reset-LR-only), and epoch budget.

Quantizr 5-stage canonical pipeline:
1. Stage "anchor": pixel-loss only, full LR, N epochs warmup.
2. Stage "finetune": pixel + lightweight scorer-loss, LR/2.
3. Stage "joint": full scorer-loss + EMA + eval_roundtrip, LR/4.
4. Stage "QAT": quantization-aware training overlay, LR/10.
5. Stage "final": QAT + KL-on-logits T=2.0 distillation from EMA shadow.

Hugging Face PEFT pretrain → SFT → RLHF analog:
1. Stage "pretrain": base task loss on un-aligned data.
2. Stage "SFT": supervised fine-tune on aligned data.
3. Stage "RLHF": reward-model + preference-optimization stages.

Karpathy nanoGPT analog:
1. Stage "warmup": LR ramp + small batch.
2. Stage "main": full LR + full batch + main loss.
3. Stage "anneal": LR cosine-anneal to zero.

Chinchilla compute-optimal analog:
- Stage budget allocation should match the compute-budget proportions
  documented in Hoffmann et al 2022; default is uniform unless operator
  routes per-substrate.

`[literature-extrapolation]` claims:
- PR101 gold + Quantizr 0.33 [contest-CUDA] both used 5-stage pipelines per
  ``feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_
  20260509.md``. The literature claim is the empirical evidence that 5-stage
  > single-stage at the contest scorer.
- Karpathy's nanoGPT (2024 lectures) documents stage-based curriculum as the
  default for 100M+ param training; bolt-on at contest scale ~100k params is
  a literature-extrapolation.

`[derived]` claims:
- Stage transitions are O(1) and add no compute cost; the entire scheduler
  overhead is O(num_stages) integer arithmetic per epoch step.

Cargo-cult audit per assumption
───────────────────────────────
* "Sequential stages always beat monolithic weighted-sum loss" — HARD-EARNED
  for empirical PR101 + Quantizr anchors; CARGO-CULTED for substrates where
  the staged-loss decomposition has not been validated empirically. Each
  consuming substrate must justify its stage choice in its design memo.
* "Stage transitions should reset optimizer state" — CARGO-CULTED default;
  Adam momentum may benefit from inheritance. We expose 3 policies (reset /
  inherit / inherit-LR-reset) and let the substrate decide.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Stage budget arithmetic → ADOPT canonical (sum of integer epoch counts).
* Stage transition loss-swap → UNIQUE per substrate (each substrate supplies
  its own loss callable per stage); :mod:`tac.training_curriculum.
  pause_to_swap_loss` is the sister helper for the actual swap.
* Optimizer-state policy → DOCUMENTED FORK (3-option enum); literature
  evidence is contradictory; we expose the choice rather than picking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

OptimizerStatePolicy = Literal["reset", "inherit", "inherit_lr_reset"]


class CurriculumStageBudgetError(ValueError):
    """Raised when stage budgets are invalid (non-positive, mismatched total)."""


@dataclass(frozen=True)
class CurriculumStage:
    """One stage in a multi-stage curriculum.

    Args:
        name: Operator-readable stage name (e.g. ``"anchor"`` / ``"finetune"``
            / ``"joint"`` / ``"QAT"`` / ``"final"``).
        epochs: Integer epoch budget for this stage (>= 1).
        loss_key: Operator-readable loss identifier (e.g.
            ``"pixel_only"`` / ``"pixel_plus_scorer"`` / ``"kl_distill_T2"``).
            The actual loss callable is supplied at scheduler-construction
            time via a registry mapping; this keeps the dataclass JSON-
            serializable for design-memo audit per Catalog #305.
        lr_multiplier: Multiplier applied to base LR for this stage (e.g.
            ``1.0`` for warmup, ``0.5`` for finetune, ``0.1`` for QAT).
            Default ``1.0``.
        optimizer_state_policy: One of ``"reset"`` (fresh AdamW state),
            ``"inherit"`` (preserve momentum + Adam state), ``"inherit_lr_
            reset"`` (preserve momentum but reset LR scheduler). Default
            ``"inherit"`` per Karpathy nanoGPT default.
        notes: Operator-readable rationale; required (rejected if empty per
            CLAUDE.md "Comment-only contracts are FORBIDDEN").
    """

    name: str
    epochs: int
    loss_key: str
    lr_multiplier: float = 1.0
    optimizer_state_policy: OptimizerStatePolicy = "inherit"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise CurriculumStageBudgetError(
                "CurriculumStage.name must be non-empty"
            )
        if self.epochs < 1:
            raise CurriculumStageBudgetError(
                f"CurriculumStage(name={self.name!r}).epochs={self.epochs} "
                "must be >= 1"
            )
        if not self.loss_key or not self.loss_key.strip():
            raise CurriculumStageBudgetError(
                f"CurriculumStage(name={self.name!r}).loss_key must be non-empty"
            )
        if self.lr_multiplier <= 0:
            raise CurriculumStageBudgetError(
                f"CurriculumStage(name={self.name!r}).lr_multiplier="
                f"{self.lr_multiplier} must be > 0"
            )
        if self.optimizer_state_policy not in {
            "reset",
            "inherit",
            "inherit_lr_reset",
        }:
            raise CurriculumStageBudgetError(
                f"CurriculumStage(name={self.name!r}).optimizer_state_policy="
                f"{self.optimizer_state_policy!r} not in canonical set"
            )
        if not self.notes or not self.notes.strip():
            raise CurriculumStageBudgetError(
                f"CurriculumStage(name={self.name!r}).notes must be non-empty "
                "per CLAUDE.md 'Comment-only contracts are FORBIDDEN'"
            )


@dataclass(frozen=True)
class StageTransition:
    """Records one stage→stage transition at an epoch boundary.

    Args:
        from_stage_name: Stage we exited.
        to_stage_name: Stage we entered.
        epoch: Epoch index AT WHICH the transition occurred (i.e. epoch N
            BELONGS to `to_stage_name`; epoch N-1 belonged to
            `from_stage_name`).
        action_keys: Tuple of operator-readable action tokens applied at the
            transition (e.g. ``("optimizer_reset", "scheduler_reset",
            "loss_swapped")``).
    """

    from_stage_name: str
    to_stage_name: str
    epoch: int
    action_keys: tuple[str, ...]


class StageScheduler:
    """Karpathy + Quantizr 5-stage curriculum scheduler.

    Usage::

        from tac.training_curriculum import StageScheduler, CurriculumStage

        stages = (
            CurriculumStage(
                name="anchor",
                epochs=100,
                loss_key="pixel_only",
                lr_multiplier=1.0,
                optimizer_state_policy="reset",
                notes="warmup pixel-only L1 loss",
            ),
            CurriculumStage(
                name="joint",
                epochs=400,
                loss_key="pixel_plus_scorer",
                lr_multiplier=0.5,
                optimizer_state_policy="inherit_lr_reset",
                notes="add scorer term; reset LR for cosine anneal",
            ),
            CurriculumStage(
                name="distill",
                epochs=200,
                loss_key="kl_distill_T2",
                lr_multiplier=0.1,
                optimizer_state_policy="inherit",
                notes="KL-on-logits T=2.0 distillation from EMA shadow",
            ),
        )
        scheduler = StageScheduler(stages)
        assert scheduler.total_epochs == 700

        for epoch in range(scheduler.total_epochs):
            stage = scheduler.stage_for_epoch(epoch)
            loss_fn = loss_registry[stage.loss_key]
            current_lr = base_lr * stage.lr_multiplier
            # ... train epoch ...
            if scheduler.is_transition_epoch(epoch):
                transition = scheduler.transition_at_epoch(epoch)
                # ... apply transition actions ...

    Args:
        stages: Ordered tuple of :class:`CurriculumStage` (>= 1).

    Raises:
        :class:`CurriculumStageBudgetError` on validation failure.
    """

    def __init__(self, stages: tuple[CurriculumStage, ...]) -> None:
        if not stages:
            raise CurriculumStageBudgetError(
                "StageScheduler requires >= 1 stage"
            )
        names = [s.name for s in stages]
        if len(names) != len(set(names)):
            raise CurriculumStageBudgetError(
                f"StageScheduler stage names must be unique; got {names}"
            )
        self._stages = stages
        self._stage_starts: list[int] = []
        cumulative = 0
        for stage in stages:
            self._stage_starts.append(cumulative)
            cumulative += stage.epochs
        self._total_epochs = cumulative

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        return self._stages

    @property
    def total_epochs(self) -> int:
        return self._total_epochs

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        """Return the stage that owns ``epoch``.

        Raises:
            :class:`CurriculumStageBudgetError` if ``epoch`` is outside total
            epoch budget.
        """
        if epoch < 0 or epoch >= self._total_epochs:
            raise CurriculumStageBudgetError(
                f"epoch={epoch} outside total budget {self._total_epochs}"
            )
        for i, start in enumerate(self._stage_starts):
            stage = self._stages[i]
            if start <= epoch < start + stage.epochs:
                return stage
        # Unreachable per total_epochs invariant.
        raise CurriculumStageBudgetError(
            f"epoch={epoch} not found in any stage (invariant violation)"
        )

    def is_transition_epoch(self, epoch: int) -> bool:
        """True if ``epoch`` is the FIRST epoch of a non-first stage."""
        if epoch <= 0:
            return False
        # epoch is a transition iff it equals a stage start AND that stage
        # is not the first stage.
        return epoch in self._stage_starts[1:]

    def transition_at_epoch(self, epoch: int) -> StageTransition:
        """Return the :class:`StageTransition` at ``epoch``.

        Raises:
            :class:`CurriculumStageBudgetError` if ``epoch`` is not a
            transition epoch.
        """
        if not self.is_transition_epoch(epoch):
            raise CurriculumStageBudgetError(
                f"epoch={epoch} is not a transition epoch; "
                "use is_transition_epoch first"
            )
        to_stage_idx = self._stage_starts.index(epoch)
        from_stage = self._stages[to_stage_idx - 1]
        to_stage = self._stages[to_stage_idx]
        actions: list[str] = ["loss_swapped"]
        if to_stage.optimizer_state_policy == "reset":
            actions.append("optimizer_reset")
            actions.append("scheduler_reset")
        elif to_stage.optimizer_state_policy == "inherit_lr_reset":
            actions.append("scheduler_reset")
        # inherit: no extra actions beyond loss swap.
        return StageTransition(
            from_stage_name=from_stage.name,
            to_stage_name=to_stage.name,
            epoch=epoch,
            action_keys=tuple(actions),
        )

    def stage_index_for_epoch(self, epoch: int) -> int:
        """Return the integer index of the stage owning ``epoch``."""
        stage = self.stage_for_epoch(epoch)
        return self._stages.index(stage)
