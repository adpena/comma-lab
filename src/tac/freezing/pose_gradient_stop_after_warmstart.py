# SPDX-License-Identifier: MIT
"""Pose-gradient stop helper for post-warmstart training."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from tac.freezing.compress_time_scorer_freeze import freeze_module_parameters


@dataclass(frozen=True)
class GradientStopReport:
    """Evidence for a post-warmstart gradient-stop decision."""

    name: str
    current_epoch: int
    warmstart_epochs: int
    stopped: bool
    trainable_after: int


def apply_pose_gradient_stop_after_warmstart(
    module: nn.Module,
    *,
    current_epoch: int,
    warmstart_epochs: int,
    name: str = "pose_path",
) -> GradientStopReport:
    """Freeze ``module`` once ``current_epoch`` reaches ``warmstart_epochs``."""

    if current_epoch < 0 or warmstart_epochs < 0:
        raise ValueError("epochs must be non-negative")
    stopped = current_epoch >= warmstart_epochs
    if stopped:
        report = freeze_module_parameters(module, name=name)
        trainable_after = report.trainable_after
    else:
        trainable_after = sum(
            param.numel() for param in module.parameters() if param.requires_grad
        )
    return GradientStopReport(
        name=name,
        current_epoch=current_epoch,
        warmstart_epochs=warmstart_epochs,
        stopped=stopped,
        trainable_after=trainable_after,
    )
