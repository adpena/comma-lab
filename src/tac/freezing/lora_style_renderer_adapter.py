# SPDX-License-Identifier: MIT
"""LoRA-style frozen-base adapter for linear renderer blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.freezing.compress_time_scorer_freeze import freeze_module_parameters


@dataclass(frozen=True)
class LoRAAdapterReport:
    """Byte/parameter evidence for a LoRA adapter."""

    in_features: int
    out_features: int
    rank: int
    adapter_parameters: int
    base_parameters: int
    alpha: float


class LoRARendererAdapter(nn.Module):
    """Frozen ``nn.Linear`` base plus trainable low-rank delta."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if rank > min(base.in_features, base.out_features):
            raise ValueError("rank must not exceed min(in_features, out_features)")
        self.base = base
        freeze_module_parameters(self.base, name="lora_base")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        down = F.linear(x, self.a)
        delta = F.linear(down, self.b) * (self.alpha / self.rank)
        return base + delta

    def report(self) -> LoRAAdapterReport:
        return LoRAAdapterReport(
            in_features=self.base.in_features,
            out_features=self.base.out_features,
            rank=self.rank,
            adapter_parameters=self.a.numel() + self.b.numel(),
            base_parameters=sum(param.numel() for param in self.base.parameters()),
            alpha=self.alpha,
        )
