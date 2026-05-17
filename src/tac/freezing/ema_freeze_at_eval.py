# SPDX-License-Identifier: MIT
"""EMA snapshot/apply/restore helper for eval."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EMAEvalSnapshot:
    """Evidence for one EMA eval snapshot context."""

    tensor_count: int
    training_before: bool


@contextmanager
def ema_freeze_at_eval_snapshot_restore(
    model: nn.Module,
    ema_state_dict: Mapping[str, torch.Tensor],
):
    """Temporarily apply EMA weights for eval, then restore live weights."""

    live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    training_before = model.training
    model.load_state_dict(dict(ema_state_dict), strict=False)
    model.eval()
    try:
        yield EMAEvalSnapshot(
            tensor_count=len(ema_state_dict),
            training_before=training_before,
        )
    finally:
        model.load_state_dict(live_state, strict=False)
        model.train(training_before)
