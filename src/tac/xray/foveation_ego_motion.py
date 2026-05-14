"""F12b: Gibson 1950 ego-motion-matched foveation analyzer.

Per zen_floor council 2026-05-13 and Time-Traveler L5 architecture: in
driving sequences, focus-of-expansion (FOE) concentrates ~70% of usable
visual information in ~25% of pixels. The per-pixel bit-budget should
be foveated — allocate more bits to the FOE region, fewer to the
parafoveal periphery.

This primitive:

1. Given a stream of ego-motion poses (from PoseNet or LAPose), estimates
   the focus-of-expansion location per pair.
2. Returns a per-pixel foveation weight map (Gaussian-decay from FOE).
3. Surfaces the total foveated-vs-uniform bit-budget reduction estimate.

Wire-in hooks engaged:

- ``sensitivity_map``: per-pixel foveation weight is a direct sensitivity
  modulator for any pixel-domain codec.
- ``bit_allocator``: foveated budget weights replace uniform per-pixel
  allocation.
- ``probe_disambiguator``: comparing foveated vs uniform on D4 substrate
  disambiguates "is foveation worth the implementation cost?".

Cross-references
----------------
- Source: Gibson 1950 *The Perception of the Visual World*
- Time-Traveler L5 reverse-engineering memo
- D4 Wyner-Ziv frame-0 nullspace sister memory

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
class FoveationReport:
    """Typed result from :meth:`FoveationEgoMotionAnalyzer.compute`.

    Attributes
    ----------
    n_pairs : int
        Number of pose pairs analyzed.
    foe_x_normalized : float
        Estimated focus-of-expansion x-coordinate (image-normalized in [0, 1]).
    foe_y_normalized : float
        Estimated focus-of-expansion y-coordinate.
    foe_sigma_pixels : float
        Sigma of the Gaussian-decay foveation kernel (in pixels).
    foveated_budget_ratio : float
        Total foveation-weight integral / uniform integral. In (0, 1];
        smaller = more aggressive foveation.
    central_25_percent_weight : float
        Sum of foveation weight in the central 25% of pixels (vs uniform 0.25).
    image_size : tuple[int, int]
        (H, W) of the foveation map.
    """

    n_pairs: int
    foe_x_normalized: float
    foe_y_normalized: float
    foe_sigma_pixels: float
    foveated_budget_ratio: float
    central_25_percent_weight: float
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be non-negative")
        if not (0.0 <= self.foe_x_normalized <= 1.0):
            raise ValueError(
                "foe_x_normalized must be in [0.0, 1.0]"
            )
        if not (0.0 <= self.foe_y_normalized <= 1.0):
            raise ValueError(
                "foe_y_normalized must be in [0.0, 1.0]"
            )
        if self.foe_sigma_pixels <= 0.0:
            raise ValueError("foe_sigma_pixels must be positive")
        if not (0.0 < self.foveated_budget_ratio <= 1.0):
            raise ValueError(
                "foveated_budget_ratio must be in (0.0, 1.0]"
            )
        if not (0.0 <= self.central_25_percent_weight <= 1.0):
            raise ValueError(
                "central_25_percent_weight must be in [0.0, 1.0]"
            )


class FoveationEgoMotionAnalyzer:
    """F12b canonical primitive: Gibson 1950 ego-motion foveation analyzer."""

    @property
    def name(self) -> str:
        return "foveation_ego_motion"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "sensitivity_map",
            "bit_allocator",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: torch.Tensor | None = None,
        *,
        ego_motion_poses: torch.Tensor | None = None,
        image_size: tuple[int, int] = (384, 512),
        foe_sigma_pixels: float = 96.0,
        foveation_floor: float = 0.05,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Compute the per-pixel foveation weight map.

        Parameters
        ----------
        target : torch.Tensor | None
            Alias for ``ego_motion_poses`` (back-compat).
        ego_motion_poses : torch.Tensor | None
            Pose-delta tensor of shape ``(N, 6)`` where the first 3 columns
            are angular components and the last 3 are translation. If None
            and target is None, defaults to forward-driving FOE at image
            center.
        image_size : (H, W)
            Foveation map dimensions. Default (384, 512) = SegNet input size.
        foe_sigma_pixels : float
            Gaussian-decay sigma in pixels. Default 96.0 (~1/4 of image height).
        foveation_floor : float
            Minimum per-pixel weight (in (0, 1)); avoids zero-budget pixels.
        """
        if not (0.0 < foveation_floor < 1.0):
            raise ValueError("foveation_floor must be in (0.0, 1.0)")
        if foe_sigma_pixels <= 0.0:
            raise ValueError("foe_sigma_pixels must be positive")

        poses = target if target is not None else ego_motion_poses
        # Default FOE at image center for forward driving.
        h, w = image_size
        if h <= 0 or w <= 0:
            raise ValueError("image_size dims must be positive")
        n_pairs = 0
        foe_x_norm = 0.5
        foe_y_norm = 0.5

        if poses is not None:
            if poses.dim() != 2 or poses.shape[1] != 6:
                raise ValueError(
                    f"ego_motion_poses must be (N, 6); got shape "
                    f"{tuple(poses.shape)}"
                )
            n_pairs = poses.shape[0]
            # FOE x ~ -lateral_translation / forward_translation; y ~
            # -vertical_translation / forward_translation. For pose
            # tensors where columns 3, 4, 5 are translation:
            forward_t = poses[:, 3].abs().clamp(min=1e-6)
            lateral_t = poses[:, 4]
            vertical_t = poses[:, 5]
            # Mean estimate (driving sequences are roughly forward).
            foe_x = (lateral_t / forward_t).mean().item()
            foe_y = (vertical_t / forward_t).mean().item()
            # Project to image-normalized coordinates centered at (0.5, 0.5).
            foe_x_norm = max(0.0, min(1.0, 0.5 + 0.5 * foe_x))
            foe_y_norm = max(0.0, min(1.0, 0.5 + 0.5 * foe_y))

        # Build foveation weight map (H, W) via Gaussian decay from FOE.
        yy = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
        xx = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
        cy = foe_y_norm * h
        cx = foe_x_norm * w
        sq_dist = (yy - cy).pow(2) + (xx - cx).pow(2)
        weights = torch.exp(-sq_dist / (2 * foe_sigma_pixels**2))
        # Floor + renormalize so max weight is 1.0.
        weights = weights.clamp(min=foveation_floor)
        weights = weights / weights.max()

        total_weight = float(weights.sum().item())
        uniform_total = float(h * w)
        budget_ratio = total_weight / uniform_total

        # Central-25% (centered square of side sqrt(0.25) = 0.5 of image).
        cy_int = int(h * 0.5)
        cx_int = int(w * 0.5)
        half_h = int(h * 0.25)
        half_w = int(w * 0.25)
        y_lo = max(0, cy_int - half_h)
        y_hi = min(h, cy_int + half_h)
        x_lo = max(0, cx_int - half_w)
        x_hi = min(w, cx_int + half_w)
        central_25 = float(weights[y_lo:y_hi, x_lo:x_hi].sum().item())
        central_fraction = central_25 / max(1e-12, total_weight)

        report = FoveationReport(
            n_pairs=n_pairs,
            foe_x_normalized=foe_x_norm,
            foe_y_normalized=foe_y_norm,
            foe_sigma_pixels=foe_sigma_pixels,
            foveated_budget_ratio=budget_ratio,
            central_25_percent_weight=central_fraction,
            image_size=image_size,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="council-deliberation",
            confidence_band=(
                max(0.0, budget_ratio - 0.1),
                min(1.0, budget_ratio + 0.1),
            ),
            composes_with=(
                "predictive_coding_hierarchy",
                "posenet_se3_lie_algebra",
            ),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "foe_sigma_pixels": foe_sigma_pixels,
                "foveation_floor": foveation_floor,
                "image_size_h": image_size[0],
                "image_size_w": image_size[1],
            },
        )

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = ["FoveationEgoMotionAnalyzer", "FoveationReport"]
