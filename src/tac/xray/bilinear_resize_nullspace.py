"""F3: Bilinear-resize left-nullspace primitive.

The contest scorer's first preprocessing step is a low-rank linear projection:
``F.interpolate(x, size=(384, 512), mode='bilinear')`` applied to the
``(1164, 874)`` camera image. Per deep_math §2.4, the resize operator
``R: R^{1164 * 874} -> R^{384 * 512}`` is a fixed linear matrix whose
left-nullspace (= ``ker(R^T)``) is exactly the set of camera-pixel
perturbations that are INVISIBLE to the scorer.

This primitive estimates:

- The left-nullspace dimension (= ``n_camera_pixels - rank(R)``).
- The fraction of camera-pixel directions in the nullspace.
- For a given camera-pixel perturbation ``delta``, the residual
  ``R(camera + delta) - R(camera)`` (when this is below numerical
  precision, ``delta`` lies in the nullspace).

The deep_math memo measures the empirical nullspace fraction as
**80.7% (820,728 of 1,016,536 camera-pixel directions)** — under bilinear
interpolation, ~80% of camera-pixel directions are FREE BITS for any
codec that operates in the camera-pixel domain (e.g., pre-resize sidecars,
camera-to-scorer-frame compression substrates).

Wire-in hooks engaged:

- ``sensitivity_map``: per-camera-pixel "is-in-nullspace" mask is the
  canonical sensitivity weight for camera-pixel-domain codecs.
- ``bit_allocator``: bits-saved-by-perturbing-nullspace-direction is
  approximately FREE (no scorer-distortion contribution).
- ``probe_disambiguator``: comparing the empirical nullspace fraction at
  multiple resize-mode kwargs ('bilinear' vs 'bicubic' vs 'nearest')
  disambiguates which resize kernel the scorer uses.

**Implementation note.** Computing the EXACT nullspace of a 1.0M-by-200K
matrix is intractable; we use a Monte-Carlo Hutchinson-style estimator
sampling random camera-pixel perturbations and counting the fraction
whose resize-output norm is below a tolerance threshold. For exact
column-space analysis, an upstream consumer should call
``estimate_nullspace_fraction_exact`` with a smaller (e.g., 64x64) image.

Cross-references
----------------
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §2.4
- Upstream code contract: ``upstream/modules.py:73, 108-109`` (the resize calls)
- Sister memory: ``feedback_macos_cpu_autopilot_wiring_landed_20260513.md``
  (cooperative-receiver scoring proxy that the nullspace primitive enables)

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)

# Canonical contest sizes (pinned from upstream/modules.py:73, 108-109).
CAMERA_SIZE_H = 874
CAMERA_SIZE_W = 1164
SCORER_INPUT_H = 384
SCORER_INPUT_W = 512


@dataclass(frozen=True)
class BilinearResizeNullspaceReport:
    """Typed result from :meth:`BilinearResizeNullspace.compute`.

    Attributes
    ----------
    camera_n_pixels : int
        Camera-frame pixel count (height * width).
    scorer_n_pixels : int
        Scorer-frame pixel count (height * width).
    upper_bound_nullspace_fraction : float
        Theoretical upper bound: (camera - scorer) / camera. With
        camera=1,016,536 and scorer=196,608, this is ~0.807 — matching
        the deep_math memo's 80.7% finding.
    empirical_nullspace_fraction : float
        Monte-Carlo estimate from ``n_samples`` random perturbations.
        ``None`` if the primitive is run in derivational-only mode.
    n_samples : int
        Number of Monte-Carlo samples (0 if derivational-only).
    perturbation_norm : float
        L2 norm of each random perturbation tested.
    output_residual_norm_tolerance : float
        Threshold below which the output residual norm is considered
        "in the nullspace".
    resize_mode : str
        The resize mode used (default 'bilinear').
    """

    camera_n_pixels: int
    scorer_n_pixels: int
    upper_bound_nullspace_fraction: float
    empirical_nullspace_fraction: float | None
    n_samples: int
    perturbation_norm: float
    output_residual_norm_tolerance: float
    resize_mode: str

    def __post_init__(self) -> None:
        if self.camera_n_pixels <= 0 or self.scorer_n_pixels <= 0:
            raise ValueError("pixel counts must be positive")
        if not (0.0 <= self.upper_bound_nullspace_fraction <= 1.0):
            raise ValueError(
                "upper_bound_nullspace_fraction must be in [0.0, 1.0]"
            )
        if self.empirical_nullspace_fraction is not None:
            if not (0.0 <= self.empirical_nullspace_fraction <= 1.0):
                raise ValueError(
                    "empirical_nullspace_fraction must be in [0.0, 1.0]"
                )
        if self.n_samples < 0:
            raise ValueError("n_samples must be non-negative")


class BilinearResizeNullspace:
    """F3 canonical primitive: bilinear-resize left-nullspace analyzer."""

    @property
    def name(self) -> str:
        return "bilinear_resize_nullspace"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "sensitivity_map",
            "bit_allocator",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: Path | str | None = None,
        *,
        camera_size: tuple[int, int] | None = None,
        scorer_size: tuple[int, int] | None = None,
        resize_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
        n_samples: int = 0,
        perturbation_norm: float = 1.0,
        output_tolerance: float = 1e-4,
        device: str = "cpu",
        seed: int = 0xCAFE,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Estimate left-nullspace fraction of the bilinear resize operator.

        Parameters
        ----------
        target : Path | str | None
            Optional path (for provenance).
        camera_size : (H, W)
            Camera frame dimensions. Defaults to contest (874, 1164).
        scorer_size : (H, W)
            Scorer frame dimensions. Defaults to contest (384, 512).
        resize_mode : str
            Resize mode ('bilinear', 'bicubic', or 'nearest').
        n_samples : int
            If > 0, run Monte-Carlo empirical estimation with this many
            random perturbations. If 0, return derivational-only bound.
        perturbation_norm : float
            L2 norm of each perturbation.
        output_tolerance : float
            Below this output L2 norm, perturbation is considered in nullspace.
        device : str
            Device for tensor ops. 'cpu' is the default per CLAUDE.md
            non-negotiable "MPS auth eval is NOISE".
        seed : int
            Random seed for Monte-Carlo sampling.
        """
        if camera_size is None:
            cam_h, cam_w = CAMERA_SIZE_H, CAMERA_SIZE_W
        else:
            cam_h, cam_w = camera_size
        if scorer_size is None:
            sco_h, sco_w = SCORER_INPUT_H, SCORER_INPUT_W
        else:
            sco_h, sco_w = scorer_size
        camera_n = cam_h * cam_w
        scorer_n = sco_h * sco_w
        if camera_n < scorer_n:
            raise ValueError(
                f"camera_size ({cam_h}x{cam_w}={camera_n}) must be >= "
                f"scorer_size ({sco_h}x{sco_w}={scorer_n}) for nullspace "
                "analysis to make sense (resize must be a projection)"
            )

        # Theoretical upper bound on nullspace fraction = (n_camera - rank)
        # / n_camera. For a low-rank linear projection with rank exactly
        # equal to the output dimension, the upper bound is (n_cam - n_sco)
        # / n_cam = 1 - n_sco / n_cam.
        upper_bound = 1.0 - scorer_n / camera_n

        empirical_fraction: float | None = None
        if n_samples > 0:
            torch.manual_seed(seed)
            n_in_nullspace = 0
            base_camera = torch.zeros(
                (1, 1, cam_h, cam_w), device=device, dtype=torch.float32
            )
            base_output = self._apply_resize(
                base_camera, sco_h, sco_w, resize_mode
            )
            # Sample n_samples random perturbations.
            for _ in range(n_samples):
                delta = torch.randn(
                    (1, 1, cam_h, cam_w), device=device, dtype=torch.float32
                )
                # Normalize to perturbation_norm.
                delta_norm = delta.flatten().norm().item()
                if delta_norm < 1e-12:
                    continue
                delta = delta * (perturbation_norm / delta_norm)
                perturbed_camera = base_camera + delta
                perturbed_output = self._apply_resize(
                    perturbed_camera, sco_h, sco_w, resize_mode
                )
                output_residual = (
                    (perturbed_output - base_output).flatten().norm().item()
                )
                if output_residual < output_tolerance:
                    n_in_nullspace += 1
            empirical_fraction = n_in_nullspace / n_samples

        archive_path: Path | None = None
        archive_sha: str | None = None
        if target is not None:
            archive_path = Path(target)
            if archive_path.exists():
                from tac.repo_io import sha256_bytes

                archive_sha = sha256_bytes(archive_path.read_bytes())

        report = BilinearResizeNullspaceReport(
            camera_n_pixels=camera_n,
            scorer_n_pixels=scorer_n,
            upper_bound_nullspace_fraction=upper_bound,
            empirical_nullspace_fraction=empirical_fraction,
            n_samples=n_samples,
            perturbation_norm=perturbation_norm,
            output_residual_norm_tolerance=output_tolerance,
            resize_mode=resize_mode,
        )

        # Confidence band: theoretical bound (always tight from above) +
        # Monte-Carlo standard error if empirical estimate ran.
        if empirical_fraction is not None and n_samples > 1:
            se = math.sqrt(
                empirical_fraction
                * (1.0 - empirical_fraction)
                / max(1, n_samples)
            )
            band = (
                max(0.0, empirical_fraction - 1.96 * se),
                min(1.0, empirical_fraction + 1.96 * se),
            )
        else:
            band = (0.0, upper_bound)

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=archive_path,
            archive_sha256=archive_sha,
            primitive_value=report,
            evidence_grade="mathematical-derivation",
            confidence_band=band,
            composes_with=(
                "yuv6_sublattice_geometry",
                "segnet_margin_polytope",
            ),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "camera_size": (cam_h, cam_w),
                "scorer_size": (sco_h, sco_w),
                "resize_mode": resize_mode,
                "device": device,
            },
        )

    @staticmethod
    def _apply_resize(
        x: torch.Tensor,
        h: int,
        w: int,
        mode: Literal["bilinear", "bicubic", "nearest"],
    ) -> torch.Tensor:
        if mode == "nearest":
            return F.interpolate(x, size=(h, w), mode="nearest")
        return F.interpolate(
            x, size=(h, w), mode=mode, align_corners=False
        )

    def project_into_nullspace(
        self,
        camera_perturbation: torch.Tensor,
        *,
        camera_size: tuple[int, int] | None = None,
        scorer_size: tuple[int, int] | None = None,
        resize_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """Project ``camera_perturbation`` into the left-nullspace.

        Uses iterative projection: ``delta_proj = delta - R^+ R delta``
        approximated by ``delta - up(down(delta))`` where ``up`` is the
        inverse-shape interpolation.

        Returns the projected perturbation (same shape as input).
        """
        if camera_size is None:
            cam_h, cam_w = CAMERA_SIZE_H, CAMERA_SIZE_W
        else:
            cam_h, cam_w = camera_size
        if scorer_size is None:
            sco_h, sco_w = SCORER_INPUT_H, SCORER_INPUT_W
        else:
            sco_h, sco_w = scorer_size

        if camera_perturbation.dim() == 2:
            x = camera_perturbation.unsqueeze(0).unsqueeze(0)
        elif camera_perturbation.dim() == 3:
            x = camera_perturbation.unsqueeze(0)
        else:
            x = camera_perturbation

        delta = x.clone()
        for _ in range(n_iterations):
            down = self._apply_resize(delta, sco_h, sco_w, resize_mode)
            up = self._apply_resize(down, cam_h, cam_w, resize_mode)
            delta = delta - up

        # Squeeze back to original shape.
        if camera_perturbation.dim() == 2:
            return delta.squeeze(0).squeeze(0)
        if camera_perturbation.dim() == 3:
            return delta.squeeze(0)
        return delta

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "BilinearResizeNullspace",
    "BilinearResizeNullspaceReport",
    "CAMERA_SIZE_H",
    "CAMERA_SIZE_W",
    "SCORER_INPUT_H",
    "SCORER_INPUT_W",
]
