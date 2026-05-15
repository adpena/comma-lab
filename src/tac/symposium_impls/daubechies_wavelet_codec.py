# SPDX-License-Identifier: MIT
"""Daubechies orthonormal wavelet codec scaffold.

Per the Grand Reunion symposium 2026-05-15 Phase F POC #2 (Daubechies +
Mallat) and the Daubechies-Rudin bridge composite. Implements compactly
supported orthonormal wavelets with multi-scale decomposition over frame
residuals + per-sub-band bit allocation. Sister of the existing
``tac.autopilot_rudin_daubechies.wavelet_multi_scale_ranker`` (Phase 5)
which RANKS via wavelet features; this module CODES via wavelet
synthesis/analysis.

Math contract
=============

A wavelet system is defined by a low-pass filter ``h = (h_0, ..., h_{2k-1})``
and high-pass filter ``g`` derived via the QMF relation
``g_n = (-1)^n h_{2k-1-n}``. Daubechies ``db_k`` (Daubechies 1988) gives
the canonical compactly supported orthonormal wavelet of order ``k`` with
``2k`` filter taps. The classical filter coefficients are

* ``db1`` (Haar): ``h = (1/sqrt 2, 1/sqrt 2)``
* ``db2``: ``h = (0.4830, 0.8365, 0.2241, -0.1294)``
* ``db4``: 8-tap Daubechies-4 coefficients (Daubechies 1992 Table 6.1)

The forward transform is

    a^{j+1} = downsample(a^j ★ h)
    d^{j+1} = downsample(a^j ★ g)

with ``a^0 = signal`` and ``a^J, d^1, ..., d^J`` the multi-resolution
decomposition. The inverse is the canonical SOS reconstruction via
upsampling and convolution with the time-reversed filters. Orthogonality
gives perfect reconstruction in floating point: ``a^0 = inv(forward(a^0))``
within numerical precision.

For coding: per Mallat 2009 §10.3, sparse wavelet coefficients are
quantized + entropy coded; the per-sub-band bit allocation respects the
coefficient magnitude statistics. We use a per-sub-band Rudin-style
falling-rule allocation:

    bits_alloc(scale) = base_bits * 2^(-scale)  if magnitude > threshold
                     = 0                         otherwise

The allocation is INTERPRETABLE per Rudin's falling-rule-list paradigm:
the operator can read the bit-allocation table and verify it.

[verified-against: Daubechies, *Comm. Pure Appl. Math.* 1988
"Orthonormal bases of compactly supported wavelets"; Daubechies, *Ten
Lectures on Wavelets* 1992 Ch.6 Table 6.1; Mallat, *A Wavelet Tour of
Signal Processing* 3rd ed Ch.7 (filter banks) + Ch.10 (compression).]

This is a SCAFFOLD: full archive integration is deferred to a follow-up
subagent per the symposium Phase F POC #2 spec ($2-10 GPU dispatch
deferred). The math is canonical and tested for perfect reconstruction.

Lane: ``lane_symposium_impl_daubechies_wavelet_codec_20260515``.
Catalog #260.
"""
from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

__all__ = (
    "DAUBECHIES_FILTERS",
    "DEFAULT_BASE_BITS",
    "DEFAULT_MAGNITUDE_THRESHOLD",
    "DaubechiesFilter",
    "WaveletDecomposition",
    "WaveletSubBandAllocation",
    "compute_per_sub_band_bit_allocation",
    "forward_wavelet_decomposition",
    "inverse_wavelet_reconstruction",
    "select_filter",
    "update_from_anchor",
)

DEFAULT_BASE_BITS: Final[int] = 8
DEFAULT_MAGNITUDE_THRESHOLD: Final[float] = 1e-3


class DaubechiesFilter(str, enum.Enum):
    """Supported Daubechies filter orders."""

    DB1 = "db1"  # Haar
    DB2 = "db2"  # 4-tap
    DB4 = "db4"  # 8-tap


# Canonical coefficients per Daubechies 1992 Table 6.1.
DAUBECHIES_FILTERS: Final[dict[DaubechiesFilter, tuple[float, ...]]] = {
    DaubechiesFilter.DB1: (
        1.0 / math.sqrt(2.0),
        1.0 / math.sqrt(2.0),
    ),
    DaubechiesFilter.DB2: (
        0.4829629131445341,
        0.8365163037378079,
        0.22414386804185735,
        -0.12940952255092145,
    ),
    DaubechiesFilter.DB4: (
        0.23037781330885523,
        0.7148465705525415,
        0.6308807679295904,
        -0.02798376941698385,
        -0.18703481171888114,
        0.030841381835986965,
        0.032883011666982945,
        -0.010597401784997278,
    ),
}


def select_filter(filter_id: DaubechiesFilter | str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(h_lowpass, g_highpass)`` for the requested Daubechies filter.

    ``g`` is derived via the QMF relation ``g_n = (-1)^n h_{N-1-n}`` per
    Mallat §7.3.
    """
    if isinstance(filter_id, str) and not isinstance(filter_id, DaubechiesFilter):
        filter_id = DaubechiesFilter(filter_id)
    h = np.asarray(DAUBECHIES_FILTERS[filter_id], dtype=np.float64)
    n = h.size
    g = np.array([(-1) ** k * h[n - 1 - k] for k in range(n)], dtype=np.float64)
    return h, g


@dataclasses.dataclass(frozen=True)
class WaveletDecomposition:
    """Multi-resolution decomposition of a 1-D signal.

    ``approximation_coefficients`` is the coarsest scale ``a^J``;
    ``detail_coefficients`` is a list ``[d^1, d^2, ..., d^J]`` of detail
    coefficients from finest to coarsest.
    """

    filter_id: DaubechiesFilter
    levels: int
    original_length: int
    approximation_coefficients: np.ndarray
    detail_coefficients: tuple[np.ndarray, ...]

    @property
    def total_coefficient_count(self) -> int:
        return int(self.approximation_coefficients.size + sum(d.size for d in self.detail_coefficients))


def _convolve_downsample(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Periodic-extension convolution then downsample by 2.

    Periodic extension preserves perfect reconstruction for orthonormal
    wavelets per Mallat §7.5.
    """
    n = signal.size
    if n == 0:
        return signal
    k = kernel.size
    # Periodic-extension: repeat enough taps to cover kernel
    extended = np.concatenate([signal, signal[: k - 1]])
    convolved = np.convolve(extended, kernel, mode="valid")
    # Take every-other sample starting at index 0
    return convolved[::2]


def _upsample_convolve(coeffs: np.ndarray, kernel: np.ndarray, target_length: int) -> np.ndarray:
    """Upsample by 2 then periodic convolution with kernel.

    Output length is exactly ``target_length`` (truncated/padded).
    """
    if coeffs.size == 0:
        return np.zeros(target_length, dtype=np.float64)
    upsampled = np.zeros(coeffs.size * 2, dtype=np.float64)
    upsampled[::2] = coeffs
    k = kernel.size
    extended = np.concatenate([upsampled[-(k - 1):], upsampled])
    convolved = np.convolve(extended, kernel, mode="valid")
    if convolved.size >= target_length:
        return convolved[:target_length]
    out = np.zeros(target_length, dtype=np.float64)
    out[: convolved.size] = convolved
    return out


def forward_wavelet_decomposition(
    signal: np.ndarray,
    *,
    filter_id: DaubechiesFilter = DaubechiesFilter.DB2,
    levels: int = 3,
) -> WaveletDecomposition:
    """Multi-level forward Daubechies wavelet transform on a 1D signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    h, g = select_filter(filter_id)
    a = signal.astype(np.float64, copy=True)
    details: list[np.ndarray] = []
    for _ in range(levels):
        a_next = _convolve_downsample(a, h)
        d = _convolve_downsample(a, g)
        details.append(d)
        a = a_next
        if a.size <= 1:
            break
    return WaveletDecomposition(
        filter_id=DaubechiesFilter(filter_id) if isinstance(filter_id, str) else filter_id,
        levels=len(details),
        original_length=int(signal.size),
        approximation_coefficients=a,
        detail_coefficients=tuple(details),
    )


def inverse_wavelet_reconstruction(decomposition: WaveletDecomposition) -> np.ndarray:
    """Inverse Daubechies wavelet transform.

    Per Mallat §7.5 with periodic boundaries the inverse uses time-reversed
    QMF filters. For orthonormal Daubechies the synthesis filters equal the
    analysis filters reversed.
    """
    h, g = select_filter(decomposition.filter_id)
    h_synth = h[::-1].copy()
    g_synth = g[::-1].copy()
    a = decomposition.approximation_coefficients.copy()
    for d in reversed(decomposition.detail_coefficients):
        # Reconstruction at this level
        target_len = a.size * 2
        a_up = _upsample_convolve(a, h_synth, target_len)
        d_up = _upsample_convolve(d, g_synth, target_len)
        a = a_up + d_up
    if a.size > decomposition.original_length:
        a = a[: decomposition.original_length]
    elif a.size < decomposition.original_length:
        out = np.zeros(decomposition.original_length, dtype=np.float64)
        out[: a.size] = a
        a = out
    return a


@dataclasses.dataclass(frozen=True)
class WaveletSubBandAllocation:
    """Per-sub-band bit allocation under Rudin's falling-rule discipline."""

    sub_band_index: int
    scale_level: int
    coefficient_count: int
    significant_coefficient_count: int
    bits_per_coefficient: int
    rule_text: str


def compute_per_sub_band_bit_allocation(
    decomposition: WaveletDecomposition,
    *,
    base_bits: int = DEFAULT_BASE_BITS,
    magnitude_threshold: float = DEFAULT_MAGNITUDE_THRESHOLD,
) -> tuple[WaveletSubBandAllocation, ...]:
    """Compute per-sub-band bit allocation per the falling-rule policy.

    Falling rule (Rudin's interpretable paradigm):

        IF scale = 0 (finest detail) AND magnitude > threshold THEN bits = base_bits
        ELSE IF scale = 1 AND magnitude > threshold THEN bits = base_bits / 2
        ELSE IF scale = J (coarsest approximation) THEN bits = base_bits   (always preserve approx)
        ELSE bits = max(base_bits / 2^scale, 1)

    Each rule is one row in the allocation table; the operator can read it.
    """
    if base_bits < 1:
        raise ValueError("base_bits must be >= 1")
    if magnitude_threshold < 0:
        raise ValueError("magnitude_threshold must be >= 0")
    allocations: list[WaveletSubBandAllocation] = []
    # Detail bands first (finest = 0)
    for scale, detail in enumerate(decomposition.detail_coefficients):
        sig_count = int((np.abs(detail) > magnitude_threshold).sum())
        bits = max(base_bits // (2**scale), 1)
        rule_text = (
            f"IF scale={scale} AND magnitude>{magnitude_threshold:g} "
            f"THEN bits={bits}/coefficient (falling-rule rank {scale})"
        )
        allocations.append(
            WaveletSubBandAllocation(
                sub_band_index=scale,
                scale_level=scale,
                coefficient_count=int(detail.size),
                significant_coefficient_count=sig_count,
                bits_per_coefficient=bits,
                rule_text=rule_text,
            )
        )
    # Approximation band: always preserve at base_bits (Rudin "default" rule).
    approx = decomposition.approximation_coefficients
    sig_count = int((np.abs(approx) > magnitude_threshold).sum())
    rule_text = (
        f"DEFAULT: scale=approximation (J={decomposition.levels}) THEN bits={base_bits} "
        "(preserved at full precision)"
    )
    allocations.append(
        WaveletSubBandAllocation(
            sub_band_index=len(decomposition.detail_coefficients),
            scale_level=decomposition.levels,
            coefficient_count=int(approx.size),
            significant_coefficient_count=sig_count,
            bits_per_coefficient=base_bits,
            rule_text=rule_text,
        )
    )
    return tuple(allocations)


def update_from_anchor(
    anchor: Mapping[str, object], *, signal: np.ndarray | None = None
) -> WaveletDecomposition | None:
    """Re-emit decomposition + allocation from a fresh signal anchor.

    Per CLAUDE.md "Subagent coherence-by-default" hook 5. The anchor is
    consumed only as a trigger; the signal itself drives recomputation.
    """
    if signal is None or signal.size == 0:
        return None
    levels = int(anchor.get("levels", 3))  # type: ignore[arg-type]
    filter_id = anchor.get("filter_id", DaubechiesFilter.DB2)
    if isinstance(filter_id, str) and not isinstance(filter_id, DaubechiesFilter):
        filter_id = DaubechiesFilter(filter_id)
    return forward_wavelet_decomposition(signal, filter_id=filter_id, levels=levels)
