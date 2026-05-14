# SPDX-License-Identifier: MIT
"""Typed config for the driving-prior world-model scaffold.

This package is an L0 research-only substrate scaffold. It intentionally
contains no trainer and no exact-eval authority; the config only fixes the
small deterministic receiver prior used by the archive/inflate contract tests.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LANE_ID = "lane_2032_driving_prior_world_model"


@dataclass(frozen=True)
class DrivingPriorWorldModelConfig:
    """Static dimensions for the placeholder receiver prior.

    The defaults are deliberately tiny so local roundtrip tests stay cheap.
    A future trained 2032 lane can supply contest dimensions while preserving
    the same archive grammar.
    """

    output_height: int = 16
    output_width: int = 24
    num_pairs: int = 4
    codebook_entries: int = 8
    residual_grid_height: int = 4
    residual_grid_width: int = 6
    lane_id: str = DEFAULT_LANE_ID

    def __post_init__(self) -> None:
        for name in (
            "output_height",
            "output_width",
            "num_pairs",
            "codebook_entries",
            "residual_grid_height",
            "residual_grid_width",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if value > 0xFFFF:
                raise ValueError(f"{name} must fit in uint16")
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
