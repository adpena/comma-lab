# SPDX-License-Identifier: MIT
"""Local Bregman/Hessian metric quantities used by the V9 policy.

The ordinary primal Hessian metric and its Fisher-natural cotangent form are
the same quantity when ``delta_eta = H @ delta_theta``.  Raw Euclidean length
in the dual coordinates is a different geometry: it is the squared-Hessian
quadratic form in primal coordinates.  Keeping four named helpers prevents a
no-solve shortcut from silently changing the canonical metric.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector: TypeAlias = NDArray[np.float64]
FloatMatrix: TypeAlias = NDArray[np.float64]

_SYMMETRY_ATOL = 1.0e-12
DELTA_ETA_CONSISTENCY_RTOL = 1.0e-12
DELTA_ETA_CONSISTENCY_ATOL = 1.0e-12


class GeometryValidationError(ValueError):
    """Raised when a local metric input is not finite, symmetric, and SPD."""


def _as_float64(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(f"{name} must be numeric: {exc}") from exc
    if not np.all(np.isfinite(array)):
        raise GeometryValidationError(f"{name} must contain only finite values")
    return array


def _validate_spd_hessian(hessian: ArrayLike) -> FloatMatrix:
    matrix = _as_float64(hessian, name="hessian")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise GeometryValidationError(
            f"hessian must be a non-empty square matrix, got shape={matrix.shape}"
        )
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_ATOL):
        raise GeometryValidationError("hessian must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise GeometryValidationError("hessian must be positive definite") from exc
    return matrix


def _validate_vector(value: ArrayLike, *, name: str, dimension: int) -> FloatVector:
    vector = _as_float64(value, name=name)
    if vector.ndim != 1 or vector.shape != (dimension,):
        raise GeometryValidationError(
            f"{name} must have shape ({dimension},), got shape={vector.shape}"
        )
    return vector


def _quadratic(vector: FloatVector, operator_vector: FloatVector) -> float:
    value = float(vector @ operator_vector)
    if not np.isfinite(value):
        raise GeometryValidationError("quadratic form produced a non-finite value")
    return value


def _validate_dual_coordinate_consistency(
    hessian: FloatMatrix,
    delta_theta: FloatVector,
    delta_eta: FloatVector,
) -> None:
    """Require ``delta_eta = H @ delta_theta`` to fp64 numerical tolerance.

    The identity-bearing aggregate uses a symmetric ``1e-12`` relative and
    absolute tolerance.  Inputs outside that tolerance are different local
    coordinate displacements, so reporting the primal/dual identities for
    them would be a false claim and must fail closed.
    """

    expected_delta_eta = hessian @ delta_theta
    if not np.allclose(
        delta_eta,
        expected_delta_eta,
        rtol=DELTA_ETA_CONSISTENCY_RTOL,
        atol=DELTA_ETA_CONSISTENCY_ATOL,
    ):
        max_abs_residual = float(np.max(np.abs(delta_eta - expected_delta_eta)))
        raise GeometryValidationError(
            "delta_eta must equal hessian @ delta_theta within fp64 tolerance "
            f"(rtol={DELTA_ETA_CONSISTENCY_RTOL:.1e}, "
            f"atol={DELTA_ETA_CONSISTENCY_ATOL:.1e}); "
            f"max_abs_residual={max_abs_residual:.17g}"
        )


def primal_hessian_quadratic(hessian: ArrayLike, delta_theta: ArrayLike) -> float:
    """Return ``delta_theta.T @ H @ delta_theta`` for a validated SPD ``H``."""

    matrix = _validate_spd_hessian(hessian)
    vector = _validate_vector(delta_theta, name="delta_theta", dimension=matrix.shape[0])
    return _quadratic(vector, matrix @ vector)


def solve_fisher_natural_cotangent(
    hessian: ArrayLike, delta_eta: ArrayLike
) -> FloatVector:
    """Solve ``H x = delta_eta`` on the Fisher-natural cotangent path.

    This named, typed solve is the only inverse-Hessian operation in this
    module.  It intentionally calls :func:`numpy.linalg.solve`; an explicit
    inverse and a raw-dual no-solve alias are forbidden.
    """

    matrix = _validate_spd_hessian(hessian)
    cotangent = _validate_vector(delta_eta, name="delta_eta", dimension=matrix.shape[0])
    try:
        solution = np.linalg.solve(matrix, cotangent)
    except np.linalg.LinAlgError as exc:  # Defensive: SPD validation should catch this first.
        raise GeometryValidationError("Fisher-natural cotangent solve failed") from exc
    if not np.all(np.isfinite(solution)):
        raise GeometryValidationError("Fisher-natural cotangent solve returned non-finite values")
    return np.asarray(solution, dtype=np.float64)


def fisher_natural_cotangent_quadratic(
    hessian: ArrayLike, delta_eta: ArrayLike
) -> float:
    """Return ``delta_eta.T @ solve(H, delta_eta)`` via the typed solve."""

    matrix = _validate_spd_hessian(hessian)
    cotangent = _validate_vector(delta_eta, name="delta_eta", dimension=matrix.shape[0])
    solved = solve_fisher_natural_cotangent(matrix, cotangent)
    return _quadratic(cotangent, solved)


def raw_dual_euclidean_quadratic(delta_eta: ArrayLike) -> float:
    """Return raw dual Euclidean length ``delta_eta.T @ delta_eta``."""

    cotangent = _as_float64(delta_eta, name="delta_eta")
    if cotangent.ndim != 1 or cotangent.size == 0:
        raise GeometryValidationError(
            f"delta_eta must be a non-empty vector, got shape={cotangent.shape}"
        )
    return _quadratic(cotangent, cotangent)


def squared_hessian_quadratic(hessian: ArrayLike, delta_theta: ArrayLike) -> float:
    """Return ``delta_theta.T @ H @ H @ delta_theta`` for validated inputs."""

    matrix = _validate_spd_hessian(hessian)
    vector = _validate_vector(delta_theta, name="delta_theta", dimension=matrix.shape[0])
    return _quadratic(vector, matrix @ (matrix @ vector))


def local_hessian_dual_geometry_summary(
    hessian: ArrayLike,
    delta_theta: ArrayLike,
    delta_eta: ArrayLike,
) -> dict[str, float]:
    """Compute four quantities for one consistent primal/dual displacement.

    This aggregate bears the two coordinate identities, so it rejects inputs
    unless ``delta_eta = H @ delta_theta`` within the documented fp64
    tolerance.  The individual quadratic helpers remain available when a
    caller intentionally needs to evaluate unrelated vectors.
    """

    matrix = _validate_spd_hessian(hessian)
    primal = _validate_vector(delta_theta, name="delta_theta", dimension=matrix.shape[0])
    dual = _validate_vector(delta_eta, name="delta_eta", dimension=matrix.shape[0])
    _validate_dual_coordinate_consistency(matrix, primal, dual)
    return {
        "primal_hessian": primal_hessian_quadratic(matrix, primal),
        "fisher_natural_cotangent": fisher_natural_cotangent_quadratic(matrix, dual),
        "raw_dual_euclidean": raw_dual_euclidean_quadratic(dual),
        "squared_hessian": squared_hessian_quadratic(matrix, primal),
    }


__all__ = [
    "DELTA_ETA_CONSISTENCY_ATOL",
    "DELTA_ETA_CONSISTENCY_RTOL",
    "GeometryValidationError",
    "fisher_natural_cotangent_quadratic",
    "local_hessian_dual_geometry_summary",
    "primal_hessian_quadratic",
    "raw_dual_euclidean_quadratic",
    "solve_fisher_natural_cotangent",
    "squared_hessian_quadratic",
]
