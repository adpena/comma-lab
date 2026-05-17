# SPDX-License-Identifier: MIT
"""Pause-to-swap-loss — Karpathy + Hugging Face PEFT pattern.

The canonical "stop training, swap loss, resume training" primitive that
:class:`StageScheduler` calls under the hood. Exposed separately because some
substrates may want to swap loss at OPERATOR-decided points (e.g.
"plateau detected → swap to KL-distill") rather than at a fixed epoch
boundary.

Sister of :class:`tac.training_curriculum.multi_stage_curriculum.StageScheduler`
that handles the FIXED-EPOCH-BOUNDARY case; this module handles the
PLATEAU-DETECTED case + the EXPLICIT-PAUSE case where the operator wants to
inspect a checkpoint before deciding the next loss.

`[derived]` claims:
- Loss swap is O(1) memory; the old loss callable is replaced by reference.
- Optimizer state preservation policy mirrors :class:`StageScheduler`'s
  3-option enum (reset / inherit / inherit_lr_reset).

`[literature-extrapolation]` claims:
- HF PEFT pretrain → SFT → RLHF transitions are loss swaps where the optimizer
  is typically reset (RLHF uses a different optimizer entirely).
- Karpathy nanoGPT typically inherits optimizer momentum across loss swaps
  for warmup → main transitions; resets for main → anneal.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Loss callable signature → UNIQUE per substrate (no canonical loss
  signature; each substrate's loss has its own arg list).
* Optimizer state policy → DOCUMENTED FORK (same 3-option enum as
  :class:`StageScheduler`).
* Pause-trigger mechanism → DEFERRED (no canonical implementation;
  substrate decides via callback).

Cargo-cult audit per assumption
───────────────────────────────
* "Loss swap mid-training is always safe" — CARGO-CULTED; if the new loss
  has very different gradient magnitudes, the substrate may need gradient
  rescaling. We document this as a known-risk surface; we do NOT auto-
  rescale (would mask a real diagnostic).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

OptimizerStateAfterSwap = Literal["reset", "inherit", "inherit_lr_reset"]


class LossSwapError(RuntimeError):
    """Raised when loss-swap invariants are violated."""


@dataclass(frozen=True)
class LossSwap:
    """Record of a loss swap event for forensic audit.

    Args:
        epoch: Epoch at which swap occurred.
        old_loss_key: Operator-readable identifier of the old loss.
        new_loss_key: Operator-readable identifier of the new loss.
        optimizer_state_after: One of ``"reset"`` / ``"inherit"`` /
            ``"inherit_lr_reset"``.
        rationale: 1-line operator-readable rationale (rejected if empty per
            CLAUDE.md "Comment-only contracts are FORBIDDEN").
    """

    epoch: int
    old_loss_key: str
    new_loss_key: str
    optimizer_state_after: OptimizerStateAfterSwap
    rationale: str

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise LossSwapError(f"epoch={self.epoch} must be >= 0")
        if not self.old_loss_key or not self.old_loss_key.strip():
            raise LossSwapError("old_loss_key must be non-empty")
        if not self.new_loss_key or not self.new_loss_key.strip():
            raise LossSwapError("new_loss_key must be non-empty")
        if self.optimizer_state_after not in {
            "reset",
            "inherit",
            "inherit_lr_reset",
        }:
            raise LossSwapError(
                f"optimizer_state_after={self.optimizer_state_after!r} not in "
                "canonical set"
            )
        if not self.rationale or not self.rationale.strip():
            raise LossSwapError(
                "LossSwap.rationale must be non-empty per CLAUDE.md "
                "'Comment-only contracts are FORBIDDEN'"
            )


def swap_loss_at_pause(
    *,
    optimizer: torch.optim.Optimizer,
    new_loss_fn: Callable[..., torch.Tensor],
    optimizer_state_after: OptimizerStateAfterSwap = "inherit",
    optimizer_factory: Callable[[], torch.optim.Optimizer] | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    lr_scheduler_factory: (
        Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        | None
    ) = None,
) -> tuple[
    Callable[..., torch.Tensor],
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler | None,
]:
    """Swap the loss function + apply optimizer-state policy.

    Args:
        optimizer: Current optimizer (may be replaced per policy).
        new_loss_fn: New loss callable to return.
        optimizer_state_after: Optimizer-state policy.
        optimizer_factory: REQUIRED if ``optimizer_state_after == "reset"``;
            zero-arg callable that returns a fresh optimizer.
        lr_scheduler: Current LR scheduler (may be replaced per policy).
        lr_scheduler_factory: REQUIRED if ``optimizer_state_after`` is
            ``"reset"`` or ``"inherit_lr_reset"`` AND ``lr_scheduler`` is not
            None; takes the new (or same) optimizer and returns fresh
            scheduler.

    Returns:
        ``(new_loss_fn, new_optimizer, new_lr_scheduler)``.

    Raises:
        :class:`LossSwapError` on policy/factory validation.
    """
    if not callable(new_loss_fn):
        raise LossSwapError("new_loss_fn must be callable")

    if optimizer_state_after == "reset":
        if optimizer_factory is None:
            raise LossSwapError(
                "optimizer_state_after='reset' requires optimizer_factory"
            )
        new_optimizer = optimizer_factory()
    else:
        new_optimizer = optimizer

    if optimizer_state_after in {"reset", "inherit_lr_reset"}:
        if lr_scheduler is None:
            new_lr_scheduler = None
        else:
            if lr_scheduler_factory is None:
                raise LossSwapError(
                    f"optimizer_state_after={optimizer_state_after!r} with "
                    "non-None lr_scheduler requires lr_scheduler_factory"
                )
            new_lr_scheduler = lr_scheduler_factory(new_optimizer)
    else:
        new_lr_scheduler = lr_scheduler

    return new_loss_fn, new_optimizer, new_lr_scheduler
