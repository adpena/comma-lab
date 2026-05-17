# SPDX-License-Identifier: MIT
"""State-dict stochastic weight averaging helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SWACheckpointReport:
    """Evidence for SWA checkpoint accumulation."""

    snapshot_count: int
    tensor_count: int


class SWACheckpointAverager:
    """Average model state_dict snapshots without requiring EMA ownership."""

    def __init__(self) -> None:
        self._avg: dict[str, torch.Tensor] = {}
        self._last_non_float: dict[str, torch.Tensor] = {}
        self.snapshot_count = 0

    def update(self, model: nn.Module) -> SWACheckpointReport:
        state = model.state_dict()
        self.snapshot_count += 1
        for key, value in state.items():
            detached = value.detach().clone()
            if not detached.is_floating_point():
                self._last_non_float[key] = detached
                continue
            if key not in self._avg:
                self._avg[key] = detached
            else:
                self._avg[key].add_((detached - self._avg[key]) / self.snapshot_count)
        return self.report()

    def averaged_state_dict(self) -> dict[str, torch.Tensor]:
        if not self._avg and not self._last_non_float:
            raise RuntimeError("no SWA snapshots recorded")
        return {
            **{key: value.clone() for key, value in self._avg.items()},
            **{key: value.clone() for key, value in self._last_non_float.items()},
        }

    def apply_to(self, model: nn.Module) -> SWACheckpointReport:
        model.load_state_dict(self.averaged_state_dict(), strict=False)
        return self.report()

    def report(self) -> SWACheckpointReport:
        return SWACheckpointReport(
            snapshot_count=self.snapshot_count,
            tensor_count=len(self._avg) + len(self._last_non_float),
        )
