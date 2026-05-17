# SPDX-License-Identifier: MIT
"""Magnitude-prune lottery-ticket extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LotteryTicketMask:
    """Boolean mask bundle for a lottery-ticket subnetwork."""

    masks: dict[str, torch.Tensor]
    keep_fraction: float
    total_parameters: int
    kept_parameters: int


def extract_lottery_ticket(
    model: nn.Module,
    *,
    keep_fraction: float,
) -> LotteryTicketMask:
    """Return magnitude masks that keep the largest ``keep_fraction`` weights."""

    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    named = [
        (name, param.detach().abs().flatten())
        for name, param in model.named_parameters()
        if param.numel() > 0
    ]
    if not named:
        return LotteryTicketMask({}, keep_fraction, 0, 0)
    all_abs = torch.cat([values for _, values in named])
    total = int(all_abs.numel())
    keep = max(1, int(round(total * keep_fraction)))
    if keep >= total:
        threshold = torch.min(all_abs)
    else:
        threshold = torch.topk(all_abs, keep, largest=True).values[-1]
    masks: dict[str, torch.Tensor] = {}
    kept = 0
    for name, param in model.named_parameters():
        mask = param.detach().abs() >= threshold
        masks[name] = mask
        kept += int(mask.sum().item())
    return LotteryTicketMask(masks, keep_fraction, total, kept)
