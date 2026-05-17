# SPDX-License-Identifier: MIT
"""Early stopping with patience + resume from best checkpoint.

The Hugging Face Trainer pattern: track best validation metric across
training; if no improvement for ``patience`` epochs, EARLY STOP and load
the best checkpoint. Generalizes A1's "best EMA checkpoint" pattern already
present in :class:`tac.training.Trainer` (which saves best-int8-checkpoint
per epoch but does NOT stop early on plateau).

This module ADDS the EARLY-STOP-ON-PLATEAU primitive on top of the existing
best-checkpoint pattern; the resume-from-best primitive is also exposed
because the existing trainer's resume is tied to its full optimizer state
which is heavier than needed for plateau-detection workflows.

`[literature-extrapolation]` claims:
- HF Trainer ``EarlyStoppingCallback`` is the canonical reference; patience
  default is 1 epoch (very aggressive); typical Kaggle/research uses
  patience ∈ {10, 20, 50} for fine-tuning.
- Bishop "Pattern Recognition and Machine Learning" §5.5.2 documents early
  stopping as the canonical regularizer for neural networks.

`[derived]` claims:
- Patience-based stop adds zero compute beyond the existing per-epoch
  validation; the bookkeeping is integer arithmetic.

Cargo-cult audit per assumption
───────────────────────────────
* "Early stopping always improves generalization" — HARD-EARNED for tasks
  with held-out validation; CARGO-CULTED if the validation metric is the
  training-time proxy (e.g. distilled scorer surrogate). Per CLAUDE.md
  "Proxy scores are APPROXIMATIONS, not truth": the proxy-auth gap can be
  2-11x for PoseNet. Each consuming substrate should validate that its
  early-stop metric is well-correlated with the contest scorer before
  trusting the stop.
* "Patience = 10 is a sensible default" — CARGO-CULTED; HF Trainer default
  is 1; Kaggle community averages 20-50. Substrates with high-variance
  validation should use patience >= 50.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class EarlyStoppingState:
    """Mutable state tracking best metric + patience countdown.

    Args:
        best_metric: Best metric observed so far (initialized to ``+inf``
            for minimize, ``-inf`` for maximize).
        best_epoch: Epoch at which best was observed.
        epochs_since_best: How many epochs since the best.
        stopped: Whether early-stop has been triggered (True after
            ``epochs_since_best >= patience``).
        best_state_dict_path: Path to the persisted best state_dict.
    """

    best_metric: float
    best_epoch: int
    epochs_since_best: int
    stopped: bool
    best_state_dict_path: str


class EarlyStoppingTracker:
    """Patience-based early stopping tracker.

    Usage::

        tracker = EarlyStoppingTracker(
            patience=10,
            minimize=True,
            checkpoint_path=Path("experiments/results/.../best.pt"),
        )
        for epoch in range(N):
            train_one_epoch(model)
            val_metric = validate(model)
            tracker.update(model, val_metric, epoch)
            if tracker.state.stopped:
                print(f"early-stop at epoch {epoch}; best={tracker.state.best_metric}")
                break
        # at end: load best checkpoint
        tracker.load_best(model)

    Args:
        patience: How many epochs of no improvement before stopping.
        minimize: If True (default), lower metric is better.
        checkpoint_path: Where to persist the best state_dict.
        improvement_threshold: Minimum absolute delta to count as
            improvement. Default ``0.0`` (strict ``<`` or ``>``).
    """

    def __init__(
        self,
        *,
        patience: int,
        minimize: bool,
        checkpoint_path: Path,
        improvement_threshold: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience={patience} must be >= 1")
        if improvement_threshold < 0:
            raise ValueError(
                f"improvement_threshold={improvement_threshold} must be >= 0"
            )
        self.patience = patience
        self.minimize = minimize
        self.checkpoint_path = Path(checkpoint_path)
        self.improvement_threshold = improvement_threshold
        self.state = EarlyStoppingState(
            best_metric=float("inf") if minimize else float("-inf"),
            best_epoch=-1,
            epochs_since_best=0,
            stopped=False,
            best_state_dict_path=str(self.checkpoint_path),
        )

    def update(
        self,
        model: nn.Module,
        val_metric: float,
        epoch: int,
    ) -> bool:
        """Update tracker with the latest validation metric.

        Returns True if this epoch produced a NEW best (and the checkpoint
        was saved); False otherwise.
        """
        if self.minimize:
            is_better = val_metric < self.state.best_metric - self.improvement_threshold
        else:
            is_better = val_metric > self.state.best_metric + self.improvement_threshold

        if is_better:
            self.state.best_metric = val_metric
            self.state.best_epoch = epoch
            self.state.epochs_since_best = 0
            self._save_checkpoint(model)
            return True
        else:
            self.state.epochs_since_best += 1
            if self.state.epochs_since_best >= self.patience:
                self.state.stopped = True
            return False

    def _save_checkpoint(self, model: nn.Module) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: .tmp + replace per Catalog #128/#131/#245 discipline.
        tmp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        snapshot = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        torch.save(snapshot, tmp)
        tmp.replace(self.checkpoint_path)


class ResumeFromBestCheckpoint:
    """Load the best checkpoint from an :class:`EarlyStoppingTracker`."""

    def __call__(self, model: nn.Module, tracker: EarlyStoppingTracker) -> None:
        ckpt_path = Path(tracker.state.best_state_dict_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"best checkpoint not found at {ckpt_path}; "
                "tracker has not yet produced a new best"
            )
        # WEIGHTS_ONLY_FALSE_OK:trusted-local-checkpoint-from-this-process
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state)
