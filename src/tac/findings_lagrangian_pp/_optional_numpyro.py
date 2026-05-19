# SPDX-License-Identifier: MIT
"""Graceful NumPyro import guard for TRACK B (operator-override parallel pursuit).

Per operator-frontier-override 2026-05-19 verbatim *"we shoud pursue PP in
parallel"*: TRACK B (NumPyro hierarchical posteriors over architectures)
dispatches simultaneously with TRACK A (closed-form Gaussian per slot 20
binding). Slot 20 Q5 RATIFIED across 3 rounds REJECTS blanket NumPyro
adoption for TRACK A; the operator override EXTENDS the build to TRACK B
parallel ONLY.

Per CLAUDE.md "Forbidden silent-skip cascades": when NumPyro is unavailable
this module raises a clear ``TrackBNumPyroUnavailableError`` with explicit
remediation; it does NOT silently fall back. The unified
``UnifiedPrediction`` interface degrades to TRACK A explicitly via
``ensemble_prediction_from_tracks(track_b=None)`` which IS a documented
canonical fallback per ``tac.findings_lagrangian.unified``.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": NumPyro is an opt-in
dependency. To enable TRACK B::

    pip install 'numpyro>=0.13' 'jax>=0.4'

Per slot 20 Q5 reactivation criterion: NumPyro adoption is gated by either
(a) Q4 DP mixture triggered, OR (b) any equation accumulates posterior
dimensionality > 20, OR (c) explicit operator-frontier-override (THIS path
per 2026-05-19 override).
"""
from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    "TRACK_B_NUMPYRO_AVAILABLE",
    "TrackBNumPyroUnavailableError",
    "require_numpyro",
    "try_import_numpyro",
]


class TrackBNumPyroUnavailableError(ImportError):
    """Raised when TRACK B is invoked but NumPyro is not installed.

    Per CLAUDE.md "Forbidden silent-skip cascades": every TRACK B entry
    point raises this explicit error rather than silently downgrading.
    Callers may catch this + route through TRACK A's degraded ensemble.
    """


def try_import_numpyro() -> tuple[Any, Any] | None:
    """Attempt to import (numpyro, jax); return (None, None) sentinel on failure.

    Returns:
        (numpyro_module, jax_module) tuple if available; None otherwise.
    """
    try:
        numpyro = importlib.import_module("numpyro")
        jax = importlib.import_module("jax")
        return (numpyro, jax)
    except ImportError:
        return None


def _check_numpyro_available() -> bool:
    """Lightweight check (no import-side-effects on failure)."""
    try:
        importlib.import_module("numpyro")
        importlib.import_module("jax")
        return True
    except ImportError:
        return False


TRACK_B_NUMPYRO_AVAILABLE: bool = _check_numpyro_available()
"""Module-level flag: True iff numpyro + jax can be imported."""


def require_numpyro() -> tuple[Any, Any]:
    """Import numpyro + jax or raise TrackBNumPyroUnavailableError.

    Per CLAUDE.md "Forbidden silent-skip cascades": fail-loud at the
    canonical entry point; do NOT silently return None.

    Returns:
        (numpyro_module, jax_module) tuple.

    Raises:
        TrackBNumPyroUnavailableError: if either dependency is missing,
            with explicit installation remediation.
    """
    result = try_import_numpyro()
    if result is None:
        raise TrackBNumPyroUnavailableError(
            "TRACK B (NumPyro hierarchical posteriors over architectures) "
            "requires numpyro + jax. Install via: "
            "`pip install 'numpyro>=0.13' 'jax>=0.4'`. "
            "Per operator-frontier-override 2026-05-19 "
            "(.omx/research/operator_authorizations/findings_lagrangian_pp_parallel_pursuit_plus_all_voices_matter_override_20260519T080000Z.md): "
            "TRACK B is parallel-pursuit per the operator's verbatim 'we shoud pursue PP in parallel'. "
            "The unified `UnifiedPrediction` interface auto-degrades to TRACK A "
            "(closed-form Gaussian per slot 20 binding) when TRACK B is unavailable; "
            "use `ensemble_prediction_from_tracks(track_b=None)` for the degraded path."
        )
    return result
