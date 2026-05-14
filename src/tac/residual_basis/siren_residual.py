# SPDX-License-Identifier: MIT
"""SIREN sinusoidal-activation coordinate-MLP residual basis SCAFFOLD.

SIREN (Sitzmann et al., 2020, "Implicit Neural Representations with Periodic
Activation Functions") replaces ReLU with sine activations in a small coordinate
MLP. Each pixel is rendered as `MLP(x, y)` where `x, y` are normalized
coordinates and the MLP weights ARE the compressed representation.

This module is a research-signal SCAFFOLD per CLAUDE.md HNeRV parity discipline.
The actual SIREN MLP architecture and training are NOT implemented; instead
we compute the FREQUENCY-DOMAIN signature of decoded RGB frames as the
sparsity prior that a SIREN-style coordinate-MLP would need to encode.

It is admissible at L0 if and only if every public API:

  1. emits `score_claim=False`, `promotion_eligible=False`,
     `ready_for_exact_eval_dispatch=False`,
  2. tags `evidence_grade="research_signal"`,
  3. does NOT load PoseNet/SegNet/scorer weights,
  4. does NOT modify or repack any archive bytes,
  5. carries an explicit path to L1+ promotion.

It DOES expose
--------------

* a typed `SirenResidualResult` dataclass with promotion-status invariants
  frozen to research-only,
* a `compute_siren_residual_stats()` entry point that accepts a decoded
  RGB array (T, H, W, 3) and returns 2D FFT magnitude-spectrum statistics
  per frequency band (low / mid / high cuts at fractional radius 1/8 + 1/3),
* a `compute_radial_frequency_buckets()` helper that returns per-band
  energy fractions in the 2D FFT magnitude domain.

It does NOT
-----------

* implement a SIREN MLP architecture (research-only; needs ≤100 LOC inflate
  budget + score-aware loss before promotion),
* claim SIREN codec parity.

Path to L1+ promotion
---------------------

Same 8-archive-grammar-field requirement. SIREN specifically needs:
INT8/FP4 weight packing of the small (~5K-20K param) MLP + a sinusoidal
activation implementation in the contest runtime (a single `np.sin` call;
trivially within the ≤2-dep budget).

References
----------

Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G.
(2020). "Implicit Neural Representations with Periodic Activation Functions."
NeurIPS.

Handoff P3 "Cool-Chic/C3/VQ/SIREN/coordinate MLP" + operator directive
2026-05-11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
RGB_CHANNELS: Final[int] = 3
RESEARCH_SIGNAL_EVIDENCE_GRADE: Final[str] = "research_signal"
LOW_FREQ_RADIUS_FRACTION: Final[float] = 0.125  # 1/8
MID_FREQ_RADIUS_FRACTION: Final[float] = 0.333  # 1/3


class SirenResidualError(ValueError):
    """Raised on shape / dtype contract violations."""


@dataclass(frozen=True)
class SirenFrequencyBandStats:
    """Per-frequency-band 2D-FFT magnitude summary."""

    band_name: str  # "low" | "mid" | "high"
    radius_fraction_inner: float
    radius_fraction_outer: float
    n_coefficients: int
    energy_fraction: float
    log_magnitude_mean: float
    log_magnitude_max: float

    def assert_invariants(self) -> None:
        if self.n_coefficients <= 0:
            raise SirenResidualError(
                f"non-positive n_coefficients={self.n_coefficients}"
            )
        if self.band_name not in ("low", "mid", "high"):
            raise SirenResidualError(f"unknown band_name={self.band_name!r}")
        for name, value in (
            ("radius_fraction_inner", self.radius_fraction_inner),
            ("radius_fraction_outer", self.radius_fraction_outer),
            ("energy_fraction", self.energy_fraction),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise SirenResidualError(f"non-finite-or-negative {name}={value}")
            if value > 1.5:  # Allow some headroom for >1 outer radius (corner pixels).
                raise SirenResidualError(f"{name}={value} > 1.5 (out of range)")
        if not math.isfinite(self.log_magnitude_mean) or not math.isfinite(
            self.log_magnitude_max
        ):
            raise SirenResidualError("non-finite log_magnitude")


@dataclass(frozen=True)
class SirenResidualResult:
    """Result of `compute_siren_residual_stats()`. Promotion-status frozen."""

    n_frames: int
    n_channels: int
    height: int
    width: int
    per_band_stats: tuple[SirenFrequencyBandStats, ...]
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default=RESEARCH_SIGNAL_EVIDENCE_GRADE, init=False)
    schema: str = field(default="siren_residual_pr106_scaffold_v1", init=False)

    def assert_invariants(self) -> None:
        if self.n_frames <= 0 or self.n_channels <= 0:
            raise SirenResidualError(
                f"non-positive n_frames={self.n_frames} or n_channels={self.n_channels}"
            )
        if self.height <= 0 or self.width <= 0:
            raise SirenResidualError(
                f"non-positive h={self.height} w={self.width}"
            )
        if not self.per_band_stats:
            raise SirenResidualError("per_band_stats must be non-empty")
        for stats in self.per_band_stats:
            stats.assert_invariants()
        # Energies must sum approximately to 1.0 (loose tolerance for
        # discretization at the band cutoffs).
        total = sum(s.energy_fraction for s in self.per_band_stats)
        if not (0.5 <= total <= 1.5):
            raise SirenResidualError(
                f"energy_fraction total={total} outside [0.5, 1.5]"
            )
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise SirenResidualError(
                "promotion-status fields must remain False (scaffold-only)"
            )
        if self.evidence_grade != RESEARCH_SIGNAL_EVIDENCE_GRADE:
            raise SirenResidualError(
                f"evidence_grade must be {RESEARCH_SIGNAL_EVIDENCE_GRADE!r}"
            )


def _validate_rgb_array(frames: np.ndarray) -> None:
    if frames.ndim != 4:
        raise SirenResidualError(f"expected (T, H, W, 3); got ndim={frames.ndim}")
    if frames.shape[3] != RGB_CHANNELS:
        raise SirenResidualError(
            f"expected last-axis size 3 (RGB); got {frames.shape[3]}"
        )
    if frames.shape[0] == 0:
        raise SirenResidualError("expected n_frames >= 1; got 0")
    if frames.dtype not in (np.uint8, np.float32, np.float64):
        raise SirenResidualError(
            f"expected dtype uint8 / float32 / float64; got {frames.dtype}"
        )


def _radial_band_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 3 boolean band masks (low, mid, high) for centered 2D FFT.

    Coordinates are computed relative to the FFT-shifted center. Band cutoffs
    are at `LOW_FREQ_RADIUS_FRACTION` and `MID_FREQ_RADIUS_FRACTION` of the
    half-width.
    """

    y = np.fft.fftshift(np.fft.fftfreq(h)) * 2.0  # normalized in [-1, 1]
    x = np.fft.fftshift(np.fft.fftfreq(w)) * 2.0
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)
    low_mask = r < LOW_FREQ_RADIUS_FRACTION
    mid_mask = (r >= LOW_FREQ_RADIUS_FRACTION) & (r < MID_FREQ_RADIUS_FRACTION)
    high_mask = r >= MID_FREQ_RADIUS_FRACTION
    return low_mask, mid_mask, high_mask


def compute_radial_frequency_buckets(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Return masked 2D-FFT-magnitude arrays for each radial band.

    `frame` shape (H, W, 3) uint8 / float. Returns a dict keyed by
    "low" | "mid" | "high" with the magnitude array masked to each band.
    Averages across the 3 RGB channels.
    """

    if frame.ndim != 3 or frame.shape[2] != RGB_CHANNELS:
        raise SirenResidualError(f"expected (H, W, 3); got {frame.shape}")
    h, w, _ = frame.shape
    f = frame.astype(np.float64, copy=False)
    magnitudes_per_channel: list[np.ndarray] = []
    for c in range(RGB_CHANNELS):
        spectrum = np.fft.fftshift(np.fft.fft2(f[..., c]))
        magnitudes_per_channel.append(np.abs(spectrum))
    avg_magnitude = np.mean(magnitudes_per_channel, axis=0)
    low_mask, mid_mask, high_mask = _radial_band_masks(h, w)
    return {
        "low": avg_magnitude * low_mask,
        "mid": avg_magnitude * mid_mask,
        "high": avg_magnitude * high_mask,
    }


def compute_siren_residual_stats(frames: np.ndarray) -> SirenResidualResult:
    """Compute per-band 2D-FFT-magnitude statistics over RGB frames.

    `frames` shape (T, H, W, 3). Returns a typed `SirenResidualResult` with
    promotion-status invariants frozen to research-only.
    """

    _validate_rgb_array(frames)
    n_frames, h, w, _ = frames.shape
    f = frames.astype(np.float64, copy=False)

    # Total energy and per-band accumulators.
    total_energy = 0.0
    band_energies = {"low": 0.0, "mid": 0.0, "high": 0.0}
    band_log_means: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    band_log_maxes: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    low_mask, mid_mask, high_mask = _radial_band_masks(h, w)
    band_masks = {"low": low_mask, "mid": mid_mask, "high": high_mask}

    for t in range(n_frames):
        frame_buckets = compute_radial_frequency_buckets(f[t])
        for band_name, magnitude in frame_buckets.items():
            energy = float((magnitude**2).sum())
            band_energies[band_name] += energy
            total_energy += energy
            # log-magnitude stats over the band-masked region (avoid log(0)).
            non_zero = magnitude[magnitude > 0.0]
            if non_zero.size > 0:
                log_mag = np.log(non_zero + 1e-12)
                band_log_means[band_name].append(float(log_mag.mean()))
                band_log_maxes[band_name].append(float(log_mag.max()))
            else:
                band_log_means[band_name].append(0.0)
                band_log_maxes[band_name].append(0.0)

    if total_energy == 0.0:
        total_energy = 1.0  # All-zero input: divide-by-zero safe; energies remain 0.

    per_band_stats = tuple(
        SirenFrequencyBandStats(
            band_name=band_name,
            radius_fraction_inner=(
                0.0
                if band_name == "low"
                else LOW_FREQ_RADIUS_FRACTION
                if band_name == "mid"
                else MID_FREQ_RADIUS_FRACTION
            ),
            radius_fraction_outer=(
                LOW_FREQ_RADIUS_FRACTION
                if band_name == "low"
                else MID_FREQ_RADIUS_FRACTION
                if band_name == "mid"
                else 1.5
            ),
            n_coefficients=int(band_masks[band_name].sum()),
            energy_fraction=band_energies[band_name] / total_energy,
            log_magnitude_mean=float(np.mean(band_log_means[band_name])),
            log_magnitude_max=float(np.max(band_log_maxes[band_name])),
        )
        for band_name in ("low", "mid", "high")
    )

    result = SirenResidualResult(
        n_frames=int(n_frames),
        n_channels=int(RGB_CHANNELS),
        height=int(h),
        width=int(w),
        per_band_stats=per_band_stats,
    )
    result.assert_invariants()
    return result
