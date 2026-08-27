# SPDX-License-Identifier: MIT
"""Lever B — the Jacobian-aligned POSE-NULL projection hook (the orthogonal-solve carrier refinement).

Design memo: ``.omx/research/sophisticated_pose_treatment_design_20260616T222900Z.md``.

THE PRINCIPLE (the #80 POSE CRUX, applied to the FiLM-v2 carrier). The contest ``d_pose`` is the MSE on the
first 6 of 12 PoseNet dims, so the local frame→pose map is a 6×N Jacobian whose ROW SPACE (rank ``r ≤ 6``)
is the POSE-SENSITIVE subspace and whose (N−r)-dim complement is the POSE-NULL. A frame perturbation whose
energy lies in the pose-null moves ``d_pose`` by ~0 to first order, regardless of its pixel-RMSE
(``tac.boundary_math.posenet_subspace_spectrum`` — measured, fail-closed on a severed gradient).

THE FiLM-v2 carrier puts a RESIDUAL on the ``rgb_0`` head only (``f0 = sigmoid(rgb_0(x + film_resid)) ·
255``), leaving the SegNet frame ``f1`` FiLM-clean. ``rgb_0`` carries the pair's relative-pose motion. But
the FiLM residual is a DENSE, geometry-unaware perturbation — part of its energy can leak into the
pose-sensitive subspace (which is exactly the warm-throttled d_pose-drift regression source). This hook
PROJECTS a frame-residual onto the MEASURED pose-null before it is applied, so the carrier's contribution to
``rgb_0`` is pose-null BY CONSTRUCTION: improving rgb_0's frame-0 fidelity (which helps the joint pair
reconstruction) cannot move d_pose.

WHAT THIS MODULE PROVIDES (measurement + projection callable — the honest integration boundary).
  * :func:`project_residual_onto_pose_null` — a thin, tested wrapper over
    ``project_onto_pose_null`` that takes a frame-residual ``(3, H, W)`` (the FiLM ``f0 − f0_clean`` delta, at
    the spectrum's work resolution) + a measured ``PoseSubspaceSpectrum`` and returns the residual with its
    pose-sensitive component REMOVED (the pose-null component only) plus the energy-fraction diagnostics.
  * :func:`pose_null_residual_fraction` — measures what fraction of a given residual is ALREADY pose-null
    (the diagnostic that says whether the FiLM-v2 carrier is incidentally cheap on pose or leaking).

INTEGRATION BOUNDARY (honest per CLAUDE.md "L1 scaffold without overlay" non-negotiable). This hook is a
DEFAULT-OFF callable + a measurement; it is NOT switched on in the live driver this landing. The full
per-step training wire-in (project the FiLM residual every step at the spectrum work resolution, with the
per-pair basis cached) is SPECIFIED in the design memo as the named next step — it requires a per-pair cached
``row_basis`` that the basin run does not yet emit. So this module claims NO score effect; it makes the
pose-null geometry MEASURABLE on the basin (via ``experiments/measure_pose_null_atlas.py``) and the
projection CALLABLE now, so the carrier design is grounded on the measured geometry, not asserted.

AUTHORITY. The spectrum is measured on the frozen CPU-torch PoseNet (NEVER mps) with the differentiable
yuv6 patch active; GT via ``frame_utils.yuv420_to_rgb``. ``[contest-CPU advisory]`` NON-PROMOTABLE.

NO-FAKE. The projection is the exact orthogonal-complement subtraction of the MEASURED row basis; a test
builds a residual INSIDE the measured row space (null_frac ≈ 0) and an isotropic residual (null_frac ≈
(N−r)/N), proving the subspace is load-bearing (reusing the #80 test pattern).
"""

from __future__ import annotations

import numpy as np

from tac.boundary_math.posenet_subspace_spectrum import (
    PoseSubspaceSpectrum,
    project_onto_pose_null,
)

__all__ = [
    "PoseNullProjectionResult",
    "pose_null_residual_fraction",
    "project_residual_onto_pose_null",
]


class PoseNullProjectionResult:
    """The result of projecting a frame-residual onto the measured pose-null.

    ``null_residual`` (3, H, W) float32: the residual with its pose-sensitive component removed (apply THIS
    instead of the raw residual to keep the carrier pose-null). ``null_energy_frac`` / ``sensitive_energy_
    frac``: the energy split of the INPUT residual. ``removed_energy_frac == sensitive_energy_frac`` is the
    d_pose-cost the projection eliminates."""

    __slots__ = ("null_energy_frac", "null_residual", "sensitive_energy_frac", "shape")

    def __init__(self, null_residual, null_energy_frac, sensitive_energy_frac, shape):
        self.null_residual = null_residual
        self.null_energy_frac = float(null_energy_frac)
        self.sensitive_energy_frac = float(sensitive_energy_frac)
        self.shape = tuple(shape)

    def to_summary(self) -> dict[str, float]:
        return {
            "null_energy_frac": self.null_energy_frac,
            "sensitive_energy_frac": self.sensitive_energy_frac,
            "removed_energy_frac": self.sensitive_energy_frac,
        }


def _as_flat_residual(residual_chw, spectrum: PoseSubspaceSpectrum) -> tuple[np.ndarray, tuple]:
    arr = np.asarray(residual_chw, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.shape[0] != spectrum.n_pixels:
        raise ValueError(
            f"residual has {flat.shape[0]} elements but the spectrum row_basis lives in "
            f"{spectrum.n_pixels}-dim pixel space (3*{spectrum.h}*{spectrum.w}). The residual must be "
            "measured/resampled at the spectrum's work resolution before projecting."
        )
    return flat, arr.shape


# PROJECT_PARITY_WAIVED: training-hook residual filter (strips pose-sensitive component of the FiLM residual); nothing stored raw and read transformed
def project_residual_onto_pose_null(residual_chw, spectrum: PoseSubspaceSpectrum) -> PoseNullProjectionResult:
    """Return the frame-residual with its pose-SENSITIVE component removed (the pose-null part only).

    ``residual_chw``: ``(3, H, W)`` frame perturbation (e.g. the FiLM ``f0 − f0_clean`` delta) at the SAME
    work resolution as ``spectrum`` (``spectrum.h`` × ``spectrum.w``). ``spectrum``: a measured
    :class:`PoseSubspaceSpectrum` from ``measure_pose_subspace_spectrum``.

    The returned ``null_residual`` has ~0 first-order d_pose cost (its energy is orthogonal to all ≤6 pose
    Jacobian rows). NO-FAKE: pure orthogonal projection onto the measured basis."""
    flat, shape = _as_flat_residual(residual_chw, spectrum)
    decomp = project_onto_pose_null(spectrum, flat)
    null_flat = np.asarray(decomp["null"], dtype=np.float64).reshape(shape)
    return PoseNullProjectionResult(
        null_residual=null_flat.astype(np.float32),
        null_energy_frac=float(decomp["null_energy_frac"]),
        sensitive_energy_frac=float(decomp["sensitive_energy_frac"]),
        shape=shape,
    )


def pose_null_residual_fraction(residual_chw, spectrum: PoseSubspaceSpectrum) -> dict[str, float]:
    """Measure what fraction of ``residual_chw`` is ALREADY pose-null (the diagnostic).

    Returns ``{"null_energy_frac", "sensitive_energy_frac"}``. A FiLM-v2 carrier residual that is mostly
    pose-null (null_frac ≈ 1) is incidentally cheap on d_pose; a residual leaking into the sensitive
    subspace (sensitive_frac > 0) is the projection's target. NO-FAKE: exact projection energy split."""
    flat, _shape = _as_flat_residual(residual_chw, spectrum)
    decomp = project_onto_pose_null(spectrum, flat)
    return {
        "null_energy_frac": float(decomp["null_energy_frac"]),
        "sensitive_energy_frac": float(decomp["sensitive_energy_frac"]),
    }
