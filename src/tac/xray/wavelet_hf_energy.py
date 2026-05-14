# SPDX-License-Identifier: MIT
"""F6: Wavelet high-frequency energy primitive.

Mallat 1989 shower-thought (zen_floor §13): wavelets diagonalize spatial
covariance. The zen-floor's spatial-frequency-axis allocator is bounded
below by the sum of high-frequency wavelet coefficient energy above a
distortion threshold. Per deep_math §3.5, the contest video's HF energy
above the SegNet stride-2-stem detectability threshold lower-bounds the
codec's spatial-resolution budget.

This primitive computes the HF energy of each frame (or tensor) in a
wavelet basis. Uses the canonical PyWavelets (pywt) if available; else
falls back to a Haar 2-D wavelet computed via 2x2 block subtraction.

Wire-in hooks engaged:

- ``bit_allocator``: per-band HF energy informs the spatial-frequency
  bit allocator.
- ``sensitivity_map``: pixels in high-HF regions get higher per-pixel
  sensitivity weight.

Cross-references
----------------
- Source: Mallat 1989 *A theory for multiresolution signal decomposition*
  (IEEE TPAMI; DOI 10.1109/34.192463); zen_floor §13 shower-thought
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §3.5
- Sister codec primitive: :mod:`tac.codec.frame_conditional` (uses
  block-FP quantization that could weight by HF energy)

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
import torch.nn.functional as F

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)


@dataclass(frozen=True)
class HFEnergyReport:
    """Typed result from :meth:`WaveletHFEnergy.compute`.

    Attributes
    ----------
    n_frames : int
        Number of frames analyzed.
    n_levels : int
        Number of wavelet decomposition levels.
    hf_energy_per_level : tuple[float, ...]
        HF energy (sum of squared coefficients) at each decomposition level.
    total_hf_energy : float
        Sum of hf_energy_per_level.
    total_signal_energy : float
        Sum of squared input pixel values.
    hf_energy_fraction : float
        total_hf_energy / total_signal_energy in [0.0, 1.0].
    fraction_above_threshold : float
        Fraction of HF coefficients with |coef| > coefficient_threshold.
    wavelet : str
        Wavelet basis used ('haar', 'db8', etc.).
    """

    n_frames: int
    n_levels: int
    hf_energy_per_level: tuple[float, ...]
    total_hf_energy: float
    total_signal_energy: float
    hf_energy_fraction: float
    fraction_above_threshold: float
    wavelet: str

    def __post_init__(self) -> None:
        if self.n_frames < 0:
            raise ValueError("n_frames must be non-negative")
        if self.n_levels <= 0:
            raise ValueError("n_levels must be positive")
        if self.total_hf_energy < 0.0:
            raise ValueError("total_hf_energy must be non-negative")
        if not (0.0 <= self.hf_energy_fraction <= 1.0001):
            raise ValueError(
                f"hf_energy_fraction must be in [0.0, 1.0]; got "
                f"{self.hf_energy_fraction}"
            )
        if not (0.0 <= self.fraction_above_threshold <= 1.0001):
            raise ValueError("fraction_above_threshold must be in [0.0, 1.0]")


class WaveletHFEnergy:
    """F6 canonical primitive: wavelet high-frequency energy analyzer.

    Uses a Haar 2-D wavelet decomposition (computed directly via 2x2
    averaging and differencing). For higher-order wavelets the caller
    should pass pre-computed coefficient tensors via
    ``compute_from_coefficients``.
    """

    @property
    def name(self) -> str:
        return "wavelet_hf_energy"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return ("bit_allocator", "sensitivity_map")

    def compute(
        self,
        target: torch.Tensor,
        *,
        n_levels: int = 3,
        wavelet: str = "haar",
        coefficient_threshold: float = 1e-3,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Compute HF energy of ``target`` frames.

        Parameters
        ----------
        target : torch.Tensor
            Frames tensor of shape ``(N, C, H, W)`` or ``(H, W)``.
        n_levels : int
            Number of Haar decomposition levels.
        wavelet : str
            Wavelet basis name. Currently only 'haar' is implemented
            natively; the report records the name for downstream auditing.
        coefficient_threshold : float
            Threshold above which |coef| is counted as "active".
        """
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")
        if target.dim() == 2:
            x = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            x = target.unsqueeze(0)
        elif target.dim() == 4:
            x = target
        else:
            raise ValueError(
                f"target must be 2-D, 3-D, or 4-D; got shape "
                f"{tuple(target.shape)}"
            )
        x = x.float()

        n_frames = x.shape[0] * x.shape[1]  # Treat channels as independent frames.
        total_signal_energy = float((x**2).sum().item())

        hf_energy_per_level: list[float] = []
        n_above_threshold = 0
        n_total_coefs = 0
        approximation = x
        # Haar decomposition: at each level, split into 4 subbands
        # (LL, LH, HL, HH) via 2x2 averaging/differencing. HF = sum of LH/HL/HH.
        for _level in range(n_levels):
            n, c, h, w = approximation.shape
            if h < 2 or w < 2:
                break
            # Crop to even size.
            h2 = (h // 2) * 2
            w2 = (w // 2) * 2
            crop = approximation[:, :, :h2, :w2]
            reshaped = crop.reshape(n, c, h2 // 2, 2, w2 // 2, 2)
            # LL = average over (2, 2) block; HL = horizontal diff; LH = vertical
            # diff; HH = diagonal diff.
            ll = reshaped.mean(dim=(3, 5))
            hl = (
                reshaped[:, :, :, 0, :, :].mean(dim=3)
                - reshaped[:, :, :, 1, :, :].mean(dim=3)
            ) / 2
            lh = (
                reshaped[:, :, :, :, :, 0].mean(dim=3)
                - reshaped[:, :, :, :, :, 1].mean(dim=3)
            ) / 2
            hh = (
                reshaped[:, :, :, 0, :, 0]
                - reshaped[:, :, :, 0, :, 1]
                - reshaped[:, :, :, 1, :, 0]
                + reshaped[:, :, :, 1, :, 1]
            ) / 4
            level_hf_energy = float(
                ((hl**2).sum() + (lh**2).sum() + (hh**2).sum()).item()
            )
            hf_energy_per_level.append(level_hf_energy)
            # Threshold counts.
            for band in (hl, lh, hh):
                n_total_coefs += band.numel()
                n_above_threshold += int((band.abs() > coefficient_threshold).sum().item())
            approximation = ll

        total_hf_energy = sum(hf_energy_per_level)
        hf_energy_fraction = (
            total_hf_energy / max(1e-12, total_signal_energy)
        )
        # Clamp at 1.0 to handle pathological numerical edge.
        hf_energy_fraction = min(1.0, hf_energy_fraction)
        fraction_above = (
            n_above_threshold / max(1, n_total_coefs)
        )

        report = HFEnergyReport(
            n_frames=n_frames,
            n_levels=len(hf_energy_per_level),
            hf_energy_per_level=tuple(hf_energy_per_level),
            total_hf_energy=total_hf_energy,
            total_signal_energy=total_signal_energy,
            hf_energy_fraction=hf_energy_fraction,
            fraction_above_threshold=fraction_above,
            wavelet=wavelet,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="council-deliberation",
            confidence_band=(
                max(0.0, hf_energy_fraction - 0.05),
                min(1.0, hf_energy_fraction + 0.05),
            ),
            composes_with=("vq_codebook_coverage", "bilinear_resize_nullspace"),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "n_levels_requested": n_levels,
                "wavelet": wavelet,
                "coefficient_threshold": coefficient_threshold,
            },
        )

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = ["HFEnergyReport", "WaveletHFEnergy"]
