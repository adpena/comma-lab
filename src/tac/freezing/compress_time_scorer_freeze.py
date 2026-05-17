# SPDX-License-Identifier: MIT
"""Compress-time scorer freezing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class FreezeReport:
    """Machine-readable evidence for a module freeze operation."""

    name: str
    parameter_count: int
    trainable_before: int
    trainable_after: int
    training_before: bool
    training_after: bool


class ScorerNotFrozenError(RuntimeError):
    """Raised when a scorer subtree is still trainable at compress time."""


def _count_trainable(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def freeze_module_parameters(module: nn.Module, *, name: str = "module") -> FreezeReport:
    """Set all parameters non-trainable and switch the module to eval mode."""

    trainable_before = _count_trainable(module)
    training_before = module.training
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return FreezeReport(
        name=name,
        parameter_count=sum(param.numel() for param in module.parameters()),
        trainable_before=trainable_before,
        trainable_after=_count_trainable(module),
        training_before=training_before,
        training_after=module.training,
    )


def ensure_compress_time_scorer_freeze(
    *scorers: nn.Module,
    names: tuple[str, ...] | None = None,
) -> tuple[FreezeReport, ...]:
    """Return reports when all scorer modules are frozen, otherwise raise."""

    if names is None:
        names = tuple(f"scorer_{idx}" for idx in range(len(scorers)))
    if len(names) != len(scorers):
        raise ValueError("names length must match scorers length")
    reports: list[FreezeReport] = []
    for name, scorer in zip(names, scorers, strict=True):
        trainable = _count_trainable(scorer)
        if trainable:
            raise ScorerNotFrozenError(f"{name} has {trainable} trainable parameters")
        if scorer.training:
            raise ScorerNotFrozenError(f"{name} is still in training mode")
        reports.append(
            FreezeReport(
                name=name,
                parameter_count=sum(param.numel() for param in scorer.parameters()),
                trainable_before=0,
                trainable_after=0,
                training_before=False,
                training_after=False,
            )
        )
    return tuple(reports)
