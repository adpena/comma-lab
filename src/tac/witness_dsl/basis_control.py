"""Canonical runtime names for the curvelet-vs-legacy-Fourier basis A/B.

The historical ``polar_fourier`` CLI/checkpoint spelling remains readable, but
normalizes to an explicitly non-shipping regression-control identity. The
windowed-curvelet treatment remains opt-in until the governed n600 byte-closed
realized-through-R no-d_seg-regression verdict exists.
"""

from __future__ import annotations

LEGACY_FOURIER_AB_CONTROL = "legacy_fourier_ab_control"
WINDOWED_CURVELET = "windowed_curvelet"
_LEGACY_ALIASES = frozenset({"polar_fourier", LEGACY_FOURIER_AB_CONTROL})


def normalize_basis_family(value: object) -> str:
    """Return the canonical runtime basis name while accepting the old control alias."""

    raw = str(value)
    if raw in _LEGACY_ALIASES:
        return LEGACY_FOURIER_AB_CONTROL
    if raw == WINDOWED_CURVELET:
        return WINDOWED_CURVELET
    raise ValueError(f"unknown basis family: {raw!r}")


def is_legacy_fourier_ab_control(value: object) -> bool:
    """Whether ``value`` selects the historical byte-compatible control computation."""

    return normalize_basis_family(value) == LEGACY_FOURIER_AB_CONTROL


__all__ = [
    "LEGACY_FOURIER_AB_CONTROL",
    "WINDOWED_CURVELET",
    "is_legacy_fourier_ab_control",
    "normalize_basis_family",
]
