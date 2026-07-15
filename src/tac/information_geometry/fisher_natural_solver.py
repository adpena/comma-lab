# SPDX-License-Identifier: MIT
"""Deterministic categorical-Fisher natural-gradient trust solve.

This is the NumPy-fp32 reference for the missing ``H^-1`` surface in the V9
information-geometry stack.  Categorical Fisher is singular in ambient logit
coordinates because adding a constant to every logit changes no probability.
The solve therefore happens in an explicit orthonormal zero-sum (Helmert)
chart, never by adding an unrecorded ambient ridge or by taking an inverse.

For quotient-compatible cotangent ``g`` and probabilities ``p`` the raw step
solves

``(Q.T @ (diag(p) - p p.T) @ Q + damping I) v = -Q.T @ g``

and returns ``u = Q v``.  The trust projection scales ``u`` to satisfy
``u.T H u <= delta_quad`` where ``delta_quad = 2 * delta_kl``.  This is a
local categorical-KL trust region; the exact finite KL is reported but is not
silently substituted for the registered local convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

DeltaConvention = Literal["delta_kl", "delta_quad"]
VALID_DELTA_CONVENTIONS = frozenset({"delta_kl", "delta_quad"})
SIMPLEX_ATOL = 5e-7
GAUGE_ATOL = 2e-6
RESEARCH_ONLY = True
METRIC_ID = "argmax_native_vjp_fidelity_v1"
EQUATION_ID = "categorical_fisher_natural_trust_region_solve_v1"


@dataclass(frozen=True)
class FisherNaturalTrustResult:
    """Complete receipt for a quotient-space categorical-Fisher solve."""

    step: np.ndarray
    unconstrained_step: np.ndarray
    trust_scale: np.ndarray
    clipped: np.ndarray
    fisher_quadratic_before: np.ndarray
    fisher_quadratic_after: np.ndarray
    exact_kl_after: np.ndarray
    projected_residual_linf: np.ndarray
    cotangent_gauge_residual: np.ndarray
    step_gauge_residual: np.ndarray
    damping: float
    delta_convention: str
    delta_kl: float
    delta_quad: float
    metric_id: str = METRIC_ID
    equation_id: str = EQUATION_ID
    authority: str = "numpy-fp32 local categorical quotient solve; no score authority"


def convert_delta_budget(delta: float, delta_convention: DeltaConvention) -> tuple[float, float]:
    """Return explicit ``(delta_kl, delta_quad)`` local trust budgets."""

    value = float(delta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("delta must be finite and non-negative")
    if delta_convention not in VALID_DELTA_CONVENTIONS:
        raise ValueError(
            f"delta_convention must be one of {sorted(VALID_DELTA_CONVENTIONS)}, "
            f"got {delta_convention!r}"
        )
    return (value, 2.0 * value) if delta_convention == "delta_kl" else (0.5 * value, value)


def helmert_zero_sum_basis(class_count: int) -> np.ndarray:
    """Return a deterministic ``K x (K-1)`` orthonormal basis of ``1^perp``."""

    if isinstance(class_count, bool) or not isinstance(class_count, (int, np.integer)):
        raise ValueError("class_count must be an integer")
    k = int(class_count)
    if k < 2:
        raise ValueError("class_count must be >= 2")
    basis = np.zeros((k, k - 1), dtype=np.float64)
    for column in range(k - 1):
        width = column + 1
        denominator = math.sqrt(width * (width + 1))
        basis[:width, column] = 1.0 / denominator
        basis[width, column] = -width / denominator
    return basis


def centre_cotangent(cotangent: Any) -> np.ndarray:
    """Explicitly project a cotangent into the categorical zero-sum chart."""

    value = np.asarray(cotangent, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] < 2:
        raise ValueError("cotangent must have shape (..., K) with K >= 2")
    if not np.all(np.isfinite(value)):
        raise ValueError("cotangent must be finite")
    return np.ascontiguousarray(value - np.mean(value, axis=-1, keepdims=True, dtype=np.float64))


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] < 2:
        raise ValueError("probabilities must have shape (..., K) with K >= 2")
    if not np.all(np.isfinite(value)):
        raise ValueError("probabilities must be finite")
    if np.any(value <= 0.0) or np.any(value > 1.0):
        raise ValueError("probabilities must lie in (0, 1]")
    sums = np.sum(value, axis=-1, dtype=np.float64)
    error = np.abs(sums - 1.0)
    if np.any(error > SIMPLEX_ATOL):
        raise ValueError(
            "probabilities must sum to one on the last axis; "
            f"maximum error is {float(np.max(error)):.6g}"
        )
    return np.ascontiguousarray(value / sums[..., None])


def _validate_cotangent(
    cotangent: Any,
    *,
    shape: tuple[int, ...],
    project_gauge: bool,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(cotangent, dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"cotangent shape {value.shape} does not match probabilities shape {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("cotangent must be finite")
    gauge = np.sum(value, axis=-1, dtype=np.float64)
    tolerance = GAUGE_ATOL * np.maximum(1.0, np.max(np.abs(value), axis=-1))
    if np.any(np.abs(gauge) > tolerance) and not project_gauge:
        raise ValueError(
            "cotangent is not quotient-compatible (last-axis sum must be zero); "
            "pass project_gauge=True only when explicit projection is intended"
        )
    centred = centre_cotangent(value) if project_gauge else np.ascontiguousarray(value)
    return centred, gauge


def categorical_fisher_quadratic(probabilities: Any, step: Any) -> np.ndarray:
    """Return ``u.T (diag(p)-p p.T) u`` without materializing the Hessian."""

    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(step, dtype=np.float64)
    if p.shape != u.shape or p.ndim < 1:
        raise ValueError("probabilities and step must have the same non-scalar shape")
    mean = np.sum(p * u, axis=-1, dtype=np.float64)
    value = np.sum(p * u * u, axis=-1, dtype=np.float64) - mean * mean
    scale = np.sum(np.abs(p * u * u), axis=-1, dtype=np.float64) + mean * mean
    floor = -64.0 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
    if np.any(value < floor):
        raise FloatingPointError("categorical Fisher quadratic became materially negative")
    return np.maximum(value, 0.0)


def categorical_exact_kl(probabilities: Any, step: Any) -> np.ndarray:
    """Return ``KL(p || softmax(log(p)+step))`` stably in float64."""

    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(step, dtype=np.float64)
    if p.shape != u.shape or p.ndim < 1:
        raise ValueError("probabilities and step must have the same non-scalar shape")
    weighted = np.sum(p * u, axis=-1, keepdims=True, dtype=np.float64)
    centred = u - weighted
    maximum = np.max(centred, axis=-1, keepdims=True)
    log_partition = np.squeeze(maximum, axis=-1) + np.log(
        np.sum(p * np.exp(centred - maximum), axis=-1, dtype=np.float64)
    )
    return np.maximum(log_partition, 0.0)


def solve_categorical_fisher_natural_step_numpy_fp32(
    probabilities: Any,
    cotangent: Any,
    *,
    delta: float,
    delta_convention: DeltaConvention = "delta_kl",
    damping: float = 0.0,
    project_gauge: bool = False,
) -> FisherNaturalTrustResult:
    """Solve ``H^-1`` in the categorical quotient and apply a local trust radius.

    ``cotangent`` is the gradient of the scalar objective, so the returned step
    is a descent step.  Ambient non-zero-sum cotangents fail closed unless the
    caller explicitly requests their gauge projection.
    """

    p = _validate_probabilities(probabilities)
    g, gauge_before = _validate_cotangent(
        cotangent,
        shape=p.shape,
        project_gauge=bool(project_gauge),
    )
    ridge = float(damping)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("damping must be finite and non-negative")
    delta_kl, delta_quad = convert_delta_budget(delta, delta_convention)
    k = p.shape[-1]
    basis = helmert_zero_sum_basis(k)
    flat_p = p.reshape(-1, k)
    flat_g = g.reshape(-1, k)
    raw = np.empty_like(flat_g)
    residual = np.empty(flat_p.shape[0], dtype=np.float64)
    identity = np.eye(k - 1, dtype=np.float64)
    for index, (row_p, row_g) in enumerate(zip(flat_p, flat_g, strict=True)):
        hessian = np.diag(row_p) - np.outer(row_p, row_p)
        reduced = basis.T @ hessian @ basis + ridge * identity
        try:
            coordinate = np.linalg.solve(reduced, -(basis.T @ row_g))
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError(
                f"categorical quotient solve failed at flattened row {index}"
            ) from exc
        raw[index] = basis @ coordinate
        projected = basis.T @ (hessian @ raw[index] + ridge * raw[index] + row_g)
        residual[index] = float(np.max(np.abs(projected)))

    unconstrained = raw.reshape(p.shape)
    q_before = categorical_fisher_quadratic(p, unconstrained)
    scale = np.ones_like(q_before, dtype=np.float64)
    positive = q_before > 0.0
    scale[positive] = np.minimum(1.0, np.sqrt(delta_quad / q_before[positive]))
    constrained64 = unconstrained * scale[..., None]
    constrained32 = np.ascontiguousarray(constrained64, dtype=np.float32)

    # The returned fp32 bytes are the authority.  Correct the rare cast-upward
    # case rather than claiming a float64 pre-cast constraint.
    q_after = categorical_fisher_quadratic(p, constrained32)
    excess = q_after > delta_quad
    if np.any(excess):
        correction = np.ones_like(q_after)
        correction[excess] = (
            np.sqrt(delta_quad / q_after[excess])
            * (1.0 - 16.0 * np.finfo(np.float32).eps)
        )
        constrained32 = np.ascontiguousarray(
            constrained32.astype(np.float64) * correction[..., None],
            dtype=np.float32,
        )
        scale *= correction
        q_after = categorical_fisher_quadratic(p, constrained32)
    if np.any(q_after > delta_quad):
        raise FloatingPointError("fp32 trust projection exceeds the registered local budget")

    gauge_after = np.abs(np.sum(constrained32.astype(np.float64), axis=-1, dtype=np.float64))
    if np.any(gauge_after > 8.0 * GAUGE_ATOL):
        raise FloatingPointError("fp32 natural step left the categorical zero-sum chart")
    return FisherNaturalTrustResult(
        step=constrained32,
        unconstrained_step=np.ascontiguousarray(unconstrained, dtype=np.float32),
        trust_scale=np.ascontiguousarray(scale, dtype=np.float32),
        clipped=np.ascontiguousarray(scale < 1.0 - 8.0 * np.finfo(np.float32).eps),
        fisher_quadratic_before=q_before,
        fisher_quadratic_after=q_after,
        exact_kl_after=categorical_exact_kl(p, constrained32),
        projected_residual_linf=residual.reshape(p.shape[:-1]),
        cotangent_gauge_residual=np.asarray(np.abs(gauge_before), dtype=np.float64),
        step_gauge_residual=gauge_after,
        damping=ridge,
        delta_convention=delta_convention,
        delta_kl=delta_kl,
        delta_quad=delta_quad,
    )


solve_categorical_fisher_natural_step = solve_categorical_fisher_natural_step_numpy_fp32


__all__ = [
    "EQUATION_ID",
    "METRIC_ID",
    "FisherNaturalTrustResult",
    "categorical_exact_kl",
    "categorical_fisher_quadratic",
    "centre_cotangent",
    "convert_delta_budget",
    "helmert_zero_sum_basis",
    "solve_categorical_fisher_natural_step",
    "solve_categorical_fisher_natural_step_numpy_fp32",
]
