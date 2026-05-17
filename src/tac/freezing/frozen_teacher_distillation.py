# SPDX-License-Identifier: MIT
"""Frozen-teacher KL distillation helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.freezing.compress_time_scorer_freeze import freeze_module_parameters


@dataclass(frozen=True)
class FrozenTeacherDistillationConfig:
    """Configuration for Hinton-style frozen-teacher KL."""

    temperature: float = 2.0
    reduction: str = "batchmean"


@dataclass(frozen=True)
class FrozenTeacherDistillationReport:
    """Evidence for one frozen-teacher distillation computation."""

    temperature: float
    reduction: str
    teacher_trainable_parameters: int
    loss_value: float


def build_frozen_teacher_from_state_dict(
    factory: Callable[[], nn.Module],
    state_dict: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Instantiate, load, freeze, and eval a teacher module."""

    teacher = factory()
    teacher.load_state_dict(state_dict)
    if device is not None:
        teacher.to(device)
    freeze_module_parameters(teacher, name="frozen_teacher")
    return teacher


def frozen_teacher_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    config: FrozenTeacherDistillationConfig | None = None,
) -> tuple[torch.Tensor, FrozenTeacherDistillationReport]:
    """Return Hinton KL loss from frozen teacher logits to student logits."""

    cfg = config or FrozenTeacherDistillationConfig()
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive")
    log_p = F.log_softmax(student_logits / cfg.temperature, dim=-1)
    q = F.softmax(teacher_logits.detach() / cfg.temperature, dim=-1)
    loss = F.kl_div(log_p, q, reduction=cfg.reduction) * (cfg.temperature**2)
    report = FrozenTeacherDistillationReport(
        temperature=cfg.temperature,
        reduction=cfg.reduction,
        teacher_trainable_parameters=0,
        loss_value=float(loss.detach().cpu()),
    )
    return loss, report
