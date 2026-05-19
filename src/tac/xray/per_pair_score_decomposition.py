# SPDX-License-Identifier: MIT
"""F9: Per-pair score decomposition primitive.

Per deep_math §4.1, the contest score decomposes as
``S = (1/N) * Sum_pairs (100*seg_i + sqrt(10*pose_i)) + 25*B/N``
— each of the 600 frame pairs contributes independently to the
distortion sum. Per-pair contribution is heterogeneous: a small minority
of pairs typically dominates >50% of the total distortion sum.

This primitive:

1. Given per-pair (seg_distortion, pose_distortion) tensors, computes
   the per-pair score contribution: ``c_i = 100*seg_i + sqrt(10*pose_i)``.
2. Returns the sorted contribution distribution + the top-K pair indices
   (the "high-leverage" pairs the autopilot should target).
3. Provides a priority vector consumable by the cathedral-autopilot
   ranker: pairs with higher c_i get higher dispatch priority.

Wire-in hooks engaged:

- ``cathedral_autopilot``: top-K pair indices feed the autopilot's
  per-pair candidate priority (selective bit-budget allocation).
- ``sensitivity_map``: per-pair contribution magnitude weights the
  pair-level sensitivity for substrate-side score-gradient training.
- ``bit_allocator``: bytes budgeted per pair can be scaled by c_i.

Cross-references
----------------
- Source: ``upstream/evaluate.py:92`` (contest formula)
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §4.1
- Codex finding: "per-pair selector is the #1 high-leverage primitive"
  (sister memory feedback_codex_3_findings_fix_landed_20260514.md)

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

# Contest formula coefficients (pinned from upstream/evaluate.py:92).
SEG_COEFF = 100.0
POSE_SQRT_COEFF = 10.0


@dataclass(frozen=True)
class PerPairScoreBreakdown:
    """Typed result from :meth:`PerPairScoreDecomposition.compute`.

    Attributes
    ----------
    n_pairs : int
        Number of pairs analyzed.
    total_distortion_sum : float
        Sum of c_i across all pairs (excludes rate term).
    mean_per_pair_contribution : float
        Mean c_i.
    max_per_pair_contribution : float
        Max c_i.
    top_k_pair_indices : tuple[int, ...]
        Pair indices sorted descending by c_i, capped at K (default 10).
    top_k_cumulative_fraction : tuple[float, ...]
        Cumulative fraction-of-total-distortion contributed by top 1, 2,
        ..., K pairs.
    per_pair_priority : tuple[float, ...]
        Full per-pair priority vector (= c_i / mean_c_i). Length = n_pairs.
    """

    n_pairs: int
    total_distortion_sum: float
    mean_per_pair_contribution: float
    max_per_pair_contribution: float
    top_k_pair_indices: tuple[int, ...]
    top_k_cumulative_fraction: tuple[float, ...]
    per_pair_priority: tuple[float, ...]
    master_gradient_pair_norm: tuple[float, ...] = ()
    master_gradient_reweighted_priority: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be non-negative")
        if self.total_distortion_sum < 0.0:
            raise ValueError("total_distortion_sum must be non-negative")
        for f in self.top_k_cumulative_fraction:
            if not (-1e-6 <= f <= 1.0 + 1e-6):
                raise ValueError(
                    f"cumulative fraction {f} must be in [0.0, 1.0]"
                )


class PerPairScoreDecomposition:
    """F9 canonical primitive: per-pair score-contribution decomposition."""

    @property
    def name(self) -> str:
        return "per_pair_score_decomposition"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "cathedral_autopilot",
            "sensitivity_map",
            "bit_allocator",
        )

    def compute(
        self,
        target: torch.Tensor,
        *,
        target_pose: torch.Tensor | None = None,
        top_k: int = 10,
        master_gradient_archive_sha256: str | None = None,
        master_gradient_anchor_path: Path | None = None,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Decompose total distortion into per-pair contributions.

        Parameters
        ----------
        target : torch.Tensor
            Either a single seg-distortion tensor of shape ``(N,)`` (in which
            case ``target_pose`` must also be provided), or a paired tensor
            of shape ``(N, 2)`` where columns are (seg, pose).
        target_pose : torch.Tensor | None
            Optional pose-distortion tensor of shape ``(N,)``.
        top_k : int
            Number of top-contribution pairs to surface (default 10).
        """
        if target.dim() == 2 and target.shape[1] == 2:
            seg = target[:, 0].float()
            pose = target[:, 1].float()
        elif target.dim() == 1:
            if target_pose is None:
                raise ValueError(
                    "target_pose must be provided when target is (N,)"
                )
            if target_pose.dim() != 1 or target_pose.shape != target.shape:
                raise ValueError(
                    f"target_pose must be (N,) matching target; got shape "
                    f"{tuple(target_pose.shape)}"
                )
            seg = target.float()
            pose = target_pose.float()
        else:
            raise ValueError(
                f"target must be (N,) or (N, 2); got shape "
                f"{tuple(target.shape)}"
            )

        n_pairs = seg.shape[0]
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > n_pairs:
            top_k = n_pairs

        # Per-pair contribution: 100*seg + sqrt(10*pose).
        per_pair = (
            SEG_COEFF * seg + torch.sqrt(POSE_SQRT_COEFF * pose.clamp(min=0.0))
        )
        total = float(per_pair.sum().item())
        mean = float(per_pair.mean().item()) if n_pairs > 0 else 0.0
        maxv = float(per_pair.max().item()) if n_pairs > 0 else 0.0

        # Sort descending for top-K.
        sorted_per_pair, sorted_indices = torch.sort(per_pair, descending=True)
        top_indices = tuple(int(i) for i in sorted_indices[:top_k].tolist())

        # Cumulative fraction over top 1..top_k.
        cum_sum = torch.cumsum(sorted_per_pair, dim=0)
        if total > 0:
            cum_frac = tuple(
                float(cum_sum[i].item()) / total for i in range(top_k)
            )
        else:
            cum_frac = tuple(0.0 for _ in range(top_k))

        # Priority vector = c_i / mean. Clip pathological 0-mean.
        priority_tensor = (
            per_pair / mean
            if mean > 0
            else torch.zeros((n_pairs,), dtype=per_pair.dtype)
        )
        priority = tuple(float(v) for v in priority_tensor.tolist())

        master_gradient_pair_norm: tuple[float, ...] = ()
        master_gradient_reweighted_priority: tuple[float, ...] = ()
        master_gradient_metadata: dict[str, Any] = {
            "master_gradient_consumed": False,
        }
        if master_gradient_archive_sha256 is not None:
            from tac.master_gradient_consumers import (
                load_per_pair_gradient_from_anchor,
            )

            gradient, anchor = load_per_pair_gradient_from_anchor(
                archive_sha256=master_gradient_archive_sha256,
                anchor_path=master_gradient_anchor_path,
            )
            gradient_tensor = torch.as_tensor(
                gradient,
                dtype=per_pair.dtype,
                device=priority_tensor.device,
            )
            if gradient_tensor.ndim != 3 or gradient_tensor.shape[1] != n_pairs:
                raise ValueError(
                    "per-pair master gradient must have shape "
                    f"(N_bytes, {n_pairs}, 3); got {tuple(gradient_tensor.shape)}"
                )
            pair_norm = torch.sqrt(
                torch.sum(gradient_tensor * gradient_tensor, dim=(0, 2))
            )
            norm_mean = float(pair_norm.mean().item()) if n_pairs > 0 else 0.0
            if norm_mean > 0.0:
                gradient_priority = pair_norm / norm_mean
                fused = priority_tensor * gradient_priority
                fused_mean = float(fused.mean().item())
                if fused_mean > 0.0:
                    fused = fused / fused_mean
            else:
                gradient_priority = torch.zeros_like(pair_norm)
                fused = torch.zeros_like(pair_norm)
            master_gradient_pair_norm = tuple(
                float(v) for v in pair_norm.tolist()
            )
            master_gradient_reweighted_priority = tuple(
                float(v) for v in fused.tolist()
            )
            master_gradient_metadata = {
                "master_gradient_consumed": True,
                "master_gradient_archive_sha256": master_gradient_archive_sha256,
                "master_gradient_measurement_axis": anchor.get("measurement_axis"),
                "master_gradient_measurement_hardware": anchor.get(
                    "measurement_hardware"
                ),
                "master_gradient_pair_norm_mean": norm_mean,
                "master_gradient_pair_norm_max": (
                    float(pair_norm.max().item()) if n_pairs > 0 else 0.0
                ),
            }

        breakdown = PerPairScoreBreakdown(
            n_pairs=n_pairs,
            total_distortion_sum=total,
            mean_per_pair_contribution=mean,
            max_per_pair_contribution=maxv,
            top_k_pair_indices=top_indices,
            top_k_cumulative_fraction=cum_frac,
            per_pair_priority=priority,
            master_gradient_pair_norm=master_gradient_pair_norm,
            master_gradient_reweighted_priority=master_gradient_reweighted_priority,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=breakdown,
            evidence_grade="mathematical-derivation",
            confidence_band=(
                max(0.0, mean - mean * 0.1),
                maxv,
            ),
            composes_with=("posenet_se3_lie_algebra", "segnet_margin_polytope"),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "top_k": top_k,
                "seg_coeff": SEG_COEFF,
                "pose_sqrt_coeff": POSE_SQRT_COEFF,
                **master_gradient_metadata,
            },
        )

    def identify_high_score_pairs(
        self,
        target: torch.Tensor,
        target_pose: torch.Tensor | None = None,
        *,
        percentile: float = 90.0,
    ) -> list[int]:
        """Return pair indices in the top ``percentile`` of contribution."""
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(
                f"percentile must be in [0.0, 100.0]; got {percentile}"
            )
        result = self.compute(target, target_pose=target_pose, top_k=10000)
        breakdown = result.primitive_value
        # Threshold at the percentile boundary.
        priorities = torch.tensor(breakdown.per_pair_priority)
        if priorities.numel() == 0:
            return []
        threshold = float(
            priorities.quantile(percentile / 100.0).item()
        )
        return [
            int(i)
            for i, p in enumerate(breakdown.per_pair_priority)
            if p >= threshold
        ]

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "POSE_SQRT_COEFF",
    "SEG_COEFF",
    "PerPairScoreBreakdown",
    "PerPairScoreDecomposition",
]
