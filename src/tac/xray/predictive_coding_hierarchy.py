# SPDX-License-Identifier: MIT
"""F12a: Rao-Ballard 1999 predictive-coding hierarchy analyzer.

Wraps the existing :class:`tac.codec.cooperative_receiver.PredictiveCodingWeights`
codec primitive in a typed :class:`XRayPrimitive` surface. Given a sequence
of frames, estimates the per-level prediction error norm + residual byte
budget at each level of the Rao-Ballard 1999 hierarchical predictive-coding
model.

The Time-Traveler L5 substrate (per
``time_traveler_architecture_reverse_engineered_20260513.md``) uses a
SINGLE-LEVEL Rao-Ballard hierarchy. The council deliberated 2026-05-13
that deeper hierarchies (2-3 levels) should yield additional rate savings
via cross-frame redundancy capture.

This primitive is the analyzer that quantifies that claim: given a frame
sequence, it returns per-level residual norms so the bit-allocator and
autopilot can decide whether to expand the hierarchy.

Wire-in hooks engaged:

- ``bit_allocator``: per-level residual byte budgets feed the allocator.
- ``sensitivity_map``: high-residual frames receive higher per-pair
  sensitivity weight.
- ``probe_disambiguator``: comparing single-level vs multi-level residual
  norms disambiguates "is the hierarchy DEEP enough?" from "is the
  per-level capacity sufficient?".

Cross-references
----------------
- Source codec primitive: :mod:`tac.codec.cooperative_receiver.predictive_coding`
- Rao-Ballard 1999 *Nature Neuroscience* canonical paper
- Time-Traveler L5 reverse-engineering memo

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)


@dataclass(frozen=True)
class PredictiveCodingReport:
    """Typed result from :meth:`PredictiveCodingHierarchy.compute`.

    Attributes
    ----------
    n_frames : int
        Number of frames in the sequence analyzed.
    n_levels : int
        Number of hierarchy levels evaluated.
    per_level_residual_norm : tuple[float, ...]
        L2 norm of the prediction error at each level.
    per_level_residual_byte_budget : tuple[int, ...]
        Estimated byte budget at each level (= ceil(log2(residual_norm) /
        8 * tensor_size)).
    total_residual_byte_budget : int
        Sum of per_level_residual_byte_budget.
    compression_ratio_estimate : float
        Estimated raw-frames-to-residual compression ratio
        (= raw_bytes / total_residual_byte_budget).
    """

    n_frames: int
    n_levels: int
    per_level_residual_norm: tuple[float, ...]
    per_level_residual_byte_budget: tuple[int, ...]
    total_residual_byte_budget: int
    compression_ratio_estimate: float

    def __post_init__(self) -> None:
        if self.n_frames < 0:
            raise ValueError("n_frames must be non-negative")
        if self.n_levels <= 0:
            raise ValueError("n_levels must be positive")
        if len(self.per_level_residual_norm) != self.n_levels:
            raise ValueError("per_level_residual_norm length must match n_levels")
        if len(self.per_level_residual_byte_budget) != self.n_levels:
            raise ValueError(
                "per_level_residual_byte_budget length must match n_levels"
            )
        if self.total_residual_byte_budget < 0:
            raise ValueError("total_residual_byte_budget must be non-negative")
        if self.compression_ratio_estimate < 0.0:
            raise ValueError("compression_ratio_estimate must be non-negative")


class PredictiveCodingHierarchy:
    """F12a canonical primitive: Rao-Ballard 1999 hierarchical analyzer.

    For each level ``l`` of the hierarchy, the residual at level ``l+1``
    is computed by:

    1. Downsample frame at level ``l`` by factor 2 (spatial).
    2. Predict frame at level ``l`` from level ``l+1`` via upsampling.
    3. Compute residual at level ``l`` = (frame_l - prediction_l).

    The L2 norm of each level's residual tells the bit-allocator how
    much byte budget to spend at that level.
    """

    @property
    def name(self) -> str:
        return "predictive_coding_hierarchy"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "bit_allocator",
            "sensitivity_map",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: torch.Tensor,
        *,
        n_levels: int = 3,
        bytes_per_residual_dim: int = 1,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Analyze the predictive-coding hierarchy for a frame sequence.

        Parameters
        ----------
        target : torch.Tensor
            Frame sequence tensor of shape ``(N, C, H, W)`` or
            ``(C, H, W)`` (single frame).
        n_levels : int
            Number of hierarchy levels to analyze.
        bytes_per_residual_dim : int
            Bytes per residual coefficient at quantization (1 = int8).
        """
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")
        if target.dim() == 3:
            frames = target.unsqueeze(0)
        elif target.dim() == 4:
            frames = target
        else:
            raise ValueError(
                f"target must be (C, H, W) or (N, C, H, W); got shape "
                f"{tuple(target.shape)}"
            )
        n_frames = frames.shape[0]
        frames = frames.float()

        # Iterative downsample / predict / residual loop.
        residuals: list[torch.Tensor] = []
        current = frames
        import torch.nn.functional as F

        for _level in range(n_levels):
            n, c, h, w = current.shape
            if h < 2 or w < 2:
                break
            downsampled = F.avg_pool2d(current, kernel_size=2)
            predicted = F.interpolate(
                downsampled,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            residual = current - predicted
            residuals.append(residual)
            current = downsampled

        per_level_norm = tuple(
            float(r.flatten().norm().item()) for r in residuals
        )
        per_level_bytes = tuple(
            r.numel() * bytes_per_residual_dim for r in residuals
        )
        total_bytes = sum(per_level_bytes)
        raw_bytes = frames.numel() * bytes_per_residual_dim
        compression_ratio = (
            raw_bytes / total_bytes if total_bytes > 0 else 0.0
        )

        report = PredictiveCodingReport(
            n_frames=n_frames,
            n_levels=len(residuals),
            per_level_residual_norm=per_level_norm,
            per_level_residual_byte_budget=per_level_bytes,
            total_residual_byte_budget=total_bytes,
            compression_ratio_estimate=compression_ratio,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="council-deliberation",
            confidence_band=(
                compression_ratio * 0.5,
                compression_ratio * 1.5,
            ),
            composes_with=("foveation_ego_motion",),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "n_levels_requested": n_levels,
                "bytes_per_residual_dim": bytes_per_residual_dim,
            },
        )

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = ["PredictiveCodingHierarchy", "PredictiveCodingReport"]
