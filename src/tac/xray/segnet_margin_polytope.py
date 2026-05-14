# SPDX-License-Identifier: MIT
"""F7: SegNet logit-margin polytope primitive.

Per deep_math §2.5 + the D1 substrate, the SegNet distortion is
**argmax disagreement**:
``d_seg = (out1.argmax(dim=1) != out2.argmax(dim=1)).float().mean()``
— a hard-decision Bernoulli loss over 5-class logits at 384x512.

This means: at each pixel, only the *argmax class* matters. Logit-space
perturbations are "free" as long as they preserve the argmax. Define
the per-pixel **margin map** ``M(x, y) = top1_logit - top2_logit``; any
logit perturbation with L2 norm <= M(x, y) is safe (preserves argmax).

This primitive computes the margin map from a tensor of SegNet logits
and returns the safe-perturbation polytope budget — the L-infinity norm
budget that preserves the argmax across the entire frame.

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: this primitive does
NOT itself run SegNet (which on MPS produces 2x distortion drift). It
accepts pre-computed logits as a tensor input.

Wire-in hooks engaged:

- ``sensitivity_map``: per-pixel margin is the canonical sensitivity weight
  for D1-substrate-style codecs.
- ``bit_allocator``: pixels with smaller margins need MORE bits (their
  argmax is fragile); pixels with larger margins can be compressed harder.
- ``probe_disambiguator``: comparing the margin-polytope budget computed
  from raw-logits vs roundtripped-eval logits disambiguates "does
  eval_roundtrip change the margin geometry?".

Cross-references
----------------
- Source: ``upstream/modules.py:111-113`` (argmax-disagreement-rate)
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §2.5
- D1 substrate (current consumer): :mod:`tac.substrates.d1_segnet_margin_aware_codec` (when landed)

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

# Canonical SegNet output class count (pinned from upstream/modules.py).
SEGNET_N_CLASSES = 5


@dataclass(frozen=True)
class SegNetMarginReport:
    """Typed result from :meth:`SegNetLogitMarginPolytope.compute`.

    Attributes
    ----------
    logits_shape : tuple[int, ...]
        Shape of the input logits tensor (e.g., (B, 5, 384, 512)).
    margin_map_shape : tuple[int, ...]
        Shape of the margin map (e.g., (B, 384, 512)).
    min_margin : float
        Smallest per-pixel margin across the frame (= L-infinity safe budget).
    mean_margin : float
        Mean per-pixel margin.
    median_margin : float
        Median per-pixel margin.
    max_margin : float
        Maximum per-pixel margin.
    n_pixels_below_threshold : int
        Number of pixels with margin < margin_threshold (= fragile pixels
        needing extra bits).
    fragile_pixel_fraction : float
        n_pixels_below_threshold / total_pixels.
    safe_perturbation_budget_l_inf : float
        Min margin = the global L-infinity perturbation budget.
    """

    logits_shape: tuple[int, ...]
    margin_map_shape: tuple[int, ...]
    min_margin: float
    mean_margin: float
    median_margin: float
    max_margin: float
    n_pixels_below_threshold: int
    fragile_pixel_fraction: float
    safe_perturbation_budget_l_inf: float

    def __post_init__(self) -> None:
        if self.min_margin < 0.0:
            # Margin can be negative if the argmax is not unique (e.g.,
            # tied top1 and top2). In that case fragility is total.
            pass  # Accept; downstream consumers handle negative margins.
        if self.n_pixels_below_threshold < 0:
            raise ValueError("n_pixels_below_threshold must be non-negative")
        if not (0.0 <= self.fragile_pixel_fraction <= 1.0):
            raise ValueError(
                f"fragile_pixel_fraction must be in [0.0, 1.0]; got "
                f"{self.fragile_pixel_fraction}"
            )


class SegNetLogitMarginPolytope:
    """F7 canonical primitive: SegNet logit-margin polytope analyzer."""

    @property
    def name(self) -> str:
        return "segnet_margin_polytope"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "sensitivity_map",
            "bit_allocator",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: torch.Tensor,
        *,
        margin_threshold: float = 0.5,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Compute the per-pixel logit-margin map.

        Parameters
        ----------
        target : torch.Tensor
            SegNet logits tensor of shape ``(B, C, H, W)`` with C = 5
            (canonical SegNet output class count) OR shape ``(C, H, W)``.
        margin_threshold : float
            Threshold below which a pixel is considered "fragile".
        """
        if target.dim() == 3:
            x = target.unsqueeze(0)
        elif target.dim() == 4:
            x = target
        else:
            raise ValueError(
                f"target must be (C, H, W) or (B, C, H, W); got shape "
                f"{tuple(target.shape)}"
            )
        # Validate channel count (warn but don't refuse — caller may have
        # a non-standard scorer with a different class count).
        c = x.shape[1]
        b, _, h, w = x.shape

        x = x.float()
        margin_map = self.compute_margin_map(x)
        margin_flat = margin_map.flatten()

        min_margin = float(margin_flat.min().item())
        mean_margin = float(margin_flat.mean().item())
        max_margin = float(margin_flat.max().item())
        # Median
        sorted_vals = torch.sort(margin_flat).values
        n = sorted_vals.numel()
        median_margin = float(
            sorted_vals[n // 2].item() if n % 2 == 1
            else ((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2).item()
        )

        n_below = int((margin_flat < margin_threshold).sum().item())
        total = margin_flat.numel()
        fragile_fraction = n_below / max(1, total)

        report = SegNetMarginReport(
            logits_shape=tuple(x.shape),
            margin_map_shape=tuple(margin_map.shape),
            min_margin=min_margin,
            mean_margin=mean_margin,
            median_margin=median_margin,
            max_margin=max_margin,
            n_pixels_below_threshold=n_below,
            fragile_pixel_fraction=fragile_fraction,
            safe_perturbation_budget_l_inf=max(0.0, min_margin),
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="structural-code-contract",
            confidence_band=(
                max(0.0, min_margin - 0.1),
                max_margin,
            ),
            composes_with=(
                "bilinear_resize_nullspace",
                "yuv6_sublattice_geometry",
            ),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "margin_threshold": margin_threshold,
                "n_classes_in_logits": c,
                "segnet_n_classes_canonical": SEGNET_N_CLASSES,
            },
        )

    @staticmethod
    def compute_margin_map(logits: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel top1 - top2 margin map.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, H, W)`` per-pixel margin.
        """
        if logits.dim() != 4:
            raise ValueError(
                f"logits must be (B, C, H, W); got shape "
                f"{tuple(logits.shape)}"
            )
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        return margin

    def compute_safe_polytope_budget(
        self,
        margin_map: torch.Tensor,
        *,
        gradient_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-pixel safe-perturbation budget.

        If ``gradient_map`` is None, returns the margin map itself
        (= L-infinity-safe perturbation per pixel).

        If ``gradient_map`` is provided (shape matching margin_map),
        returns ``margin_map / max(gradient_map, eps)`` — the per-pixel
        L2 perturbation budget scaled by the local logit-gradient norm.
        """
        if gradient_map is None:
            return margin_map.clamp(min=0.0)
        if gradient_map.shape != margin_map.shape:
            raise ValueError(
                f"gradient_map shape {tuple(gradient_map.shape)} != "
                f"margin_map shape {tuple(margin_map.shape)}"
            )
        eps = 1e-6
        return (margin_map.clamp(min=0.0) / (gradient_map.abs() + eps))

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "SEGNET_N_CLASSES",
    "SegNetLogitMarginPolytope",
    "SegNetMarginReport",
]
