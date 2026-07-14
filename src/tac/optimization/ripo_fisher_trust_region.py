# SPDX-License-Identifier: MIT
"""Deterministic categorical-Fisher trust-region clipping.

This module is the NumPy-fp32 authority for the RIPO-inspired categorical
output-space trust region.  It operates on complete categorical probability
vectors.  It does *not* claim that SegNet output-space geometry is the exact
pullback geometry of an upstream witness head.

All reductions and clipping decisions use float64.  Returned logit arrays are
contiguous float32 and are checked again after that cast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

DeltaConvention = Literal["delta_kl", "delta_quad"]
ClipMode = Literal[
    "local_directional",
    "exact_kl",
    "local_euclidean_ball",
    "uniform_l2_control",
]

VALID_DELTA_CONVENTIONS = frozenset({"delta_kl", "delta_quad"})
SIMPLEX_ATOL = 5e-7
RESEARCH_ONLY = True
VALID_MODES = frozenset(
    {
        "local_directional",
        "exact_kl",
        "local_euclidean_ball",
        "uniform_l2_control",
    }
)


@dataclass(frozen=True)
class FisherClipResult:
    """Complete deterministic receipt for a categorical logit clip."""

    centred_input: np.ndarray
    centred_output: np.ndarray
    alpha: np.ndarray
    clipped: np.ndarray
    q_before: np.ndarray
    q_after: np.ndarray
    exact_kl_before: np.ndarray
    exact_kl_after: np.ndarray
    lambda_max: np.ndarray
    winner: np.ndarray
    rival: np.ndarray
    top_two_mass: np.ndarray
    delta_convention: str
    delta_kl: float
    delta_quad: float
    mode: str
    quadratic_relative_error_before: np.ndarray
    quadratic_relative_error_after: np.ndarray
    top_two_relative_error_before: np.ndarray
    top_two_relative_error_after: np.ndarray
    probability_shape: tuple[int, ...]
    tolerance: float
    authority: str = "numpy-fp32 categorical output-space; no score authority"


def convert_delta_budget(delta: float, delta_convention: DeltaConvention) -> tuple[float, float]:
    """Return ``(delta_kl, delta_quad)`` without a hidden convention.

    ``delta_quad = 2 * delta_kl`` because the local categorical KL is
    ``0.5 * u.T @ F @ u``.
    """

    value = float(delta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("delta must be finite and non-negative")
    if delta_convention not in VALID_DELTA_CONVENTIONS:
        raise ValueError(
            f"delta_convention must be one of {sorted(VALID_DELTA_CONVENTIONS)}, "
            f"got {delta_convention!r}"
        )
    if delta_convention == "delta_kl":
        return value, 2.0 * value
    return 0.5 * value, value


def _validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] < 2:
        raise ValueError(
            "probabilities must have shape (..., K) with K >= 2; "
            f"got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("probabilities must be finite")
    if np.any(value <= 0.0) or np.any(value > 1.0):
        raise ValueError("probabilities must lie in (0, 1]")
    sums = np.sum(value, axis=-1, dtype=np.float64)
    if not np.all(np.abs(sums - 1.0) <= SIMPLEX_ATOL):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(
            "probabilities must sum to one on the last axis; "
            f"maximum error is {max_error:.6g}"
        )
    # Normalize only inside the accepted simplex tolerance.  This makes the
    # exact formulas insensitive to benign fp32 serialization drift without
    # accepting a malformed categorical vector.
    return np.ascontiguousarray(value / sums[..., None])


def _validate_step(proposed_logit_step: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(proposed_logit_step, dtype=np.float64)
    if value.shape != shape:
        raise ValueError(
            f"proposed_logit_step shape {value.shape} does not match probabilities shape {shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("proposed_logit_step must be finite")
    return np.ascontiguousarray(value)


def centre_logits(logit_step: Any) -> np.ndarray:
    """Arithmetic-gauge centre a full categorical logit update in float64."""

    value = np.asarray(logit_step, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] < 2:
        raise ValueError("logit_step must have shape (..., K) with K >= 2")
    if not np.all(np.isfinite(value)):
        raise ValueError("logit_step must be finite")
    return np.ascontiguousarray(value - np.mean(value, axis=-1, keepdims=True, dtype=np.float64))


def categorical_fisher_quadratic(probabilities: Any, logit_step: Any) -> np.ndarray:
    """Compute ``u.T (diag(p) - p p.T) u`` along the last axis."""

    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(logit_step, dtype=np.float64)
    if p.shape != u.shape or p.ndim < 1:
        raise ValueError("probabilities and logit_step must have the same non-scalar shape")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(u)):
        raise ValueError("probabilities and logit_step must be finite")
    mean = np.sum(p * u, axis=-1, dtype=np.float64)
    q = np.sum(p * u * u, axis=-1, dtype=np.float64) - mean * mean
    # Roundoff can produce a tiny negative value for a PSD quadratic.
    scale = np.sum(np.abs(p * u * u), axis=-1, dtype=np.float64) + mean * mean
    floor = -32.0 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
    if np.any(q < floor):
        raise FloatingPointError("categorical Fisher quadratic became materially negative")
    return np.maximum(q, 0.0)


def categorical_exact_kl(probabilities: Any, logit_step: Any) -> np.ndarray:
    """Compute finite categorical KL with cancellation-safe small steps.

    After weighted centering, ``KL = log1p(E[expm1(v)])``.  For small ``v``
    the expectation is evaluated as ``E[v] + E[expm1(v)-v]`` and the second
    term uses a series, preserving the ``O(v**2)`` signal instead of
    subtracting two nearly equal ``O(v)`` quantities.
    """

    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(logit_step, dtype=np.float64)
    if p.shape != u.shape or p.ndim < 1:
        raise ValueError("probabilities and logit_step must have the same non-scalar shape")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(u)):
        raise ValueError("probabilities and logit_step must be finite")
    if np.any(p < 0.0):
        raise ValueError("probabilities must be non-negative")
    sums = np.sum(p, axis=-1, dtype=np.float64)
    if np.any(sums <= 0.0):
        raise ValueError("probabilities must have positive mass")
    pn = p / sums[..., None]
    flat_p = pn.reshape(-1, pn.shape[-1])
    flat_u = u.reshape(-1, u.shape[-1])
    weighted_mean = np.sum(flat_p * flat_u, axis=-1, dtype=np.float64)
    centred = flat_u - weighted_mean[:, None]
    maximum_abs = np.max(np.abs(centred), axis=-1)
    small = maximum_abs <= 1e-3
    flat_kl = np.empty(flat_p.shape[0], dtype=np.float64)
    if np.any(small):
        value = centred[small]
        square = value * value
        remainder = square * (
            0.5
            + value
            * (
                1.0 / 6.0
                + value
                * (1.0 / 24.0 + value * (1.0 / 120.0 + value * (1.0 / 720.0)))
            )
        )
        residual = np.sum(flat_p[small] * value, axis=-1, dtype=np.float64)
        log1p_argument = residual + np.sum(
            flat_p[small] * remainder,
            axis=-1,
            dtype=np.float64,
        )
        negative_floor = -64.0 * np.finfo(np.float64).eps * np.maximum(
            np.sum(flat_p[small] * square, axis=-1, dtype=np.float64),
            np.finfo(np.float64).tiny,
        )
        if np.any(log1p_argument < negative_floor):
            raise FloatingPointError("small-step categorical KL log1p argument became negative")
        flat_kl[small] = np.log1p(np.maximum(log1p_argument, 0.0))
    if np.any(~small):
        large_u = flat_u[~small]
        large_p = flat_p[~small]
        maximum = np.max(large_u, axis=-1, keepdims=True)
        log_partition = np.squeeze(maximum, axis=-1) + np.log(
            np.sum(large_p * np.exp(large_u - maximum), axis=-1, dtype=np.float64)
        )
        flat_kl[~small] = log_partition - np.sum(
            large_p * large_u,
            axis=-1,
            dtype=np.float64,
        )
    flat_kl[np.ptp(flat_u, axis=-1) == 0.0] = 0.0
    kl = flat_kl.reshape(p.shape[:-1])
    floor = -64.0 * np.finfo(np.float64).eps * np.maximum(np.abs(kl), 1.0)
    if np.any(kl < floor):
        raise FloatingPointError("exact categorical KL became materially negative")
    return np.maximum(kl, 0.0)


def categorical_fisher_lambda_max(probabilities: Any) -> np.ndarray:
    """Return the largest eigenvalue in bounded vectorized batches.

    A whole n600 scorer surface can contain hundreds of millions of rows, so
    materializing ``(..., K, K)`` at once is forbidden.  Batched ``eigvalsh``
    keeps the implementation vectorized without an unbounded tensor.
    """

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] < 2 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite with shape (..., K), K >= 2")
    flat = p.reshape(-1, p.shape[-1])
    values = np.empty(flat.shape[0], dtype=np.float64)
    chunk_rows = 65_536
    diagonal = np.arange(p.shape[-1])
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        row = flat[start:stop]
        fisher = -(row[:, :, None] * row[:, None, :])
        fisher[:, diagonal, diagonal] += row
        values[start:stop] = np.maximum(np.linalg.eigvalsh(fisher)[:, -1], 0.0)
    return values.reshape(p.shape[:-1])


def winner_rival_indices(probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic winner and runner-up indices along the last axis."""

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] < 2 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite with shape (..., K), K >= 2")
    order = np.argsort(-p, axis=-1, kind="stable")
    return order[..., 0], order[..., 1]


def winner_rival_curvature(
    probabilities: Any,
    winner: Any | None = None,
    rival: Any | None = None,
) -> np.ndarray:
    """Return ``p_w + p_r - (p_w - p_r)^2`` for each categorical row."""

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] < 2 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite with shape (..., K), K >= 2")
    if winner is None or rival is None:
        if winner is not None or rival is not None:
            raise ValueError("winner and rival must be supplied together")
        w, r = winner_rival_indices(p)
    else:
        w = np.asarray(winner, dtype=np.int64)
        r = np.asarray(rival, dtype=np.int64)
        if w.shape != p.shape[:-1] or r.shape != p.shape[:-1]:
            raise ValueError("winner and rival shapes must equal probabilities.shape[:-1]")
        if np.any(w < 0) or np.any(w >= p.shape[-1]) or np.any(r < 0) or np.any(r >= p.shape[-1]):
            raise ValueError("winner and rival indices are out of range")
        if np.any(w == r):
            raise ValueError("winner and rival indices must differ")
    pw = np.take_along_axis(p, w[..., None], axis=-1)[..., 0]
    pr = np.take_along_axis(p, r[..., None], axis=-1)[..., 0]
    return np.maximum(pw + pr - (pw - pr) ** 2, 0.0)


def winner_rival_radius(
    probabilities: Any,
    *,
    delta: float,
    delta_convention: DeltaConvention,
    winner: Any | None = None,
    rival: Any | None = None,
) -> np.ndarray:
    """Return the local symmetric winner-rival margin radius ``|t|``."""

    _, delta_quad = convert_delta_budget(delta, delta_convention)
    curvature = winner_rival_curvature(probabilities, winner=winner, rival=rival)
    radius = np.full(curvature.shape, np.inf, dtype=np.float64)
    np.divide(4.0 * delta_quad, curvature, out=radius, where=curvature > 0.0)
    return np.sqrt(radius)


def _largest_exact_kl_alpha(
    p: np.ndarray,
    direction: np.ndarray,
    *,
    delta_kl: float,
    tolerance: float,
) -> np.ndarray:
    shape = p.shape[:-1]
    full_kl = categorical_exact_kl(p, direction)
    if delta_kl == 0.0:
        # With strictly positive p, the Fisher null is exactly the
        # constant-logit gauge direction.  No numerical tolerance may turn a
        # non-null tiny update into an admissible delta=0 update.
        return np.where(np.ptp(direction, axis=-1) == 0.0, 1.0, 0.0)
    active = full_kl > delta_kl
    low = np.zeros(shape, dtype=np.float64)
    high = np.ones(shape, dtype=np.float64)
    # ``tolerance`` is the requested alpha-bracket precision,
    # not a decorative fingerprint field.  Returning the feasible lower edge
    # keeps the hard KL constraint exact even for a deliberately coarse value.
    for _ in range(256):
        middle = 0.5 * (low + high)
        kl = categorical_exact_kl(p, direction * middle[..., None])
        feasible = kl <= delta_kl
        low = np.where(active & feasible, middle, low)
        high = np.where(active & ~feasible, middle, high)
        if np.all(~active | ((high - low) <= tolerance)):
            break
    else:
        raise FloatingPointError("exact-KL bisection did not reach the requested tolerance")
    return np.where(active, low, 1.0)


def _top_two_quadratic(
    probabilities: np.ndarray,
    step: np.ndarray,
    winner: np.ndarray,
    rival: np.ndarray,
) -> np.ndarray:
    curvature = winner_rival_curvature(probabilities, winner=winner, rival=rival)
    uw = np.take_along_axis(step, winner[..., None], axis=-1)[..., 0]
    ur = np.take_along_axis(step, rival[..., None], axis=-1)[..., 0]
    return 0.25 * curvature * (uw - ur) ** 2


def _cast_and_centre(value: np.ndarray) -> np.ndarray:
    cast = np.asarray(value, dtype=np.float32)
    centred = cast - np.mean(cast, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    return np.ascontiguousarray(centred, dtype=np.float32)


def _post_cast_enforce(
    p: np.ndarray,
    output: np.ndarray,
    *,
    mode: str,
    delta_kl: float,
    delta_quad: float,
    lambda_max: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    result = _cast_and_centre(output)
    eps_margin = 1.0 - 32.0 * np.finfo(np.float32).eps
    for _ in range(16):
        value64 = np.asarray(result, dtype=np.float64)
        if mode == "local_directional":
            measure = categorical_fisher_quadratic(p, value64)
            limit = delta_quad
        elif mode == "exact_kl":
            measure = categorical_exact_kl(p, value64)
            limit = delta_kl
        elif mode == "local_euclidean_ball":
            measure = lambda_max * np.sum(value64 * value64, axis=-1, dtype=np.float64)
            limit = delta_quad
        else:
            measure = np.sum(value64 * value64, axis=-1, dtype=np.float64)
            limit = delta_quad
        violation = measure > limit
        if not np.any(violation):
            return result
        factor = np.ones(measure.shape, dtype=np.float64)
        if limit == 0.0:
            factor[violation] = 0.0
        elif mode == "exact_kl":
            exact_alpha = _largest_exact_kl_alpha(
                p,
                value64,
                delta_kl=delta_kl,
                tolerance=tolerance,
            )
            factor[violation] = exact_alpha[violation] * eps_margin
        else:
            factor[violation] = np.sqrt(limit / measure[violation]) * eps_margin
        result = _cast_and_centre(value64 * factor[..., None])
    raise FloatingPointError(f"float32 post-cast {mode} constraint could not be certified")


def clip_categorical_fisher_step_numpy_fp32(
    probabilities: Any,
    proposed_logit_step: Any,
    *,
    delta: float,
    delta_convention: DeltaConvention,
    mode: ClipMode,
    tolerance: float,
) -> FisherClipResult:
    """Clip a complete categorical logit direction and return its receipt."""

    tol = _validate_tolerance(tolerance)
    delta_kl, delta_quad = convert_delta_budget(delta, delta_convention)
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    p = _validate_probabilities(probabilities)
    raw = _validate_step(proposed_logit_step, shape=p.shape)
    direction = centre_logits(raw)
    q_before = categorical_fisher_quadratic(p, direction)
    kl_before = categorical_exact_kl(p, direction)
    lambda_max = categorical_fisher_lambda_max(p)
    norm_sq = np.sum(direction * direction, axis=-1, dtype=np.float64)

    if mode == "local_directional":
        denominator = q_before
        limit = delta_quad
    elif mode == "local_euclidean_ball":
        denominator = lambda_max * norm_sq
        limit = delta_quad
    elif mode == "uniform_l2_control":
        denominator = norm_sq
        limit = delta_quad
    else:
        denominator = None
        limit = delta_kl

    if mode == "exact_kl":
        alpha = _largest_exact_kl_alpha(
            p,
            direction,
            delta_kl=delta_kl,
            tolerance=tol,
        )
    else:
        alpha = np.ones(p.shape[:-1], dtype=np.float64)
        active = denominator > limit
        if limit == 0.0:
            alpha[active] = 0.0
        else:
            alpha[active] = np.sqrt(limit / denominator[active])

    # Leave a deterministic float32 safety margin on every contracted row;
    # this avoids a row-by-row cast crossing while preserving exact identity
    # for directions already inside the selected region.
    contracted = alpha < 1.0
    alpha = np.where(
        contracted,
        alpha * (1.0 - 64.0 * np.finfo(np.float32).eps),
        alpha,
    )

    output = _post_cast_enforce(
        p,
        direction * alpha[..., None],
        mode=mode,
        delta_kl=delta_kl,
        delta_quad=delta_quad,
        lambda_max=lambda_max,
        tolerance=tol,
    )
    output64 = np.asarray(output, dtype=np.float64)
    # Record the realized post-cast scale.  It can be microscopically smaller
    # than the analytic scale after conservative post-cast enforcement.
    direction_norm_sq = np.sum(direction * direction, axis=-1, dtype=np.float64)
    dot = np.sum(direction * output64, axis=-1, dtype=np.float64)
    realized_alpha = np.ones(p.shape[:-1], dtype=np.float64)
    np.divide(dot, direction_norm_sq, out=realized_alpha, where=direction_norm_sq > 0.0)
    realized_alpha = np.clip(realized_alpha, 0.0, 1.0)

    q_after = categorical_fisher_quadratic(p, output64)
    kl_after = categorical_exact_kl(p, output64)
    winner, rival = winner_rival_indices(p)
    pw = np.take_along_axis(p, winner[..., None], axis=-1)[..., 0]
    pr = np.take_along_axis(p, rival[..., None], axis=-1)[..., 0]
    top_before = _top_two_quadratic(p, direction, winner, rival)
    top_after = _top_two_quadratic(p, output64, winner, rival)

    return FisherClipResult(
        centred_input=_cast_and_centre(direction),
        centred_output=output,
        alpha=realized_alpha,
        clipped=realized_alpha < 1.0 - 8.0 * np.finfo(np.float32).eps,
        q_before=q_before,
        q_after=q_after,
        exact_kl_before=kl_before,
        exact_kl_after=kl_after,
        lambda_max=lambda_max,
        winner=winner.astype(np.int64, copy=False),
        rival=rival.astype(np.int64, copy=False),
        top_two_mass=pw + pr,
        delta_convention=str(delta_convention),
        delta_kl=delta_kl,
        delta_quad=delta_quad,
        mode=str(mode),
        quadratic_relative_error_before=np.divide(
            np.abs(kl_before - 0.5 * q_before),
            np.maximum(np.abs(kl_before), np.finfo(np.float64).tiny),
        ),
        quadratic_relative_error_after=np.divide(
            np.abs(kl_after - 0.5 * q_after),
            np.maximum(np.abs(kl_after), np.finfo(np.float64).tiny),
        ),
        top_two_relative_error_before=np.divide(
            np.abs(q_before - top_before),
            np.maximum(np.abs(q_before), np.finfo(np.float64).tiny),
        ),
        top_two_relative_error_after=np.divide(
            np.abs(q_after - top_after),
            np.maximum(np.abs(q_after), np.finfo(np.float64).tiny),
        ),
        probability_shape=tuple(int(value) for value in p.shape),
        tolerance=tol,
    )


__all__ = [
    "RESEARCH_ONLY",
    "VALID_DELTA_CONVENTIONS",
    "VALID_MODES",
    "ClipMode",
    "DeltaConvention",
    "FisherClipResult",
    "categorical_exact_kl",
    "categorical_fisher_lambda_max",
    "categorical_fisher_quadratic",
    "centre_logits",
    "clip_categorical_fisher_step_numpy_fp32",
    "convert_delta_budget",
    "winner_rival_curvature",
    "winner_rival_indices",
    "winner_rival_radius",
]
