# SPDX-License-Identifier: MIT
"""Lazy MLX parity surface for the categorical-Fisher natural trust solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.information_geometry.fisher_natural_solver import (
    FisherNaturalTrustResult,
    categorical_fisher_quadratic,
    centre_cotangent,
    convert_delta_budget,
    helmert_zero_sum_basis,
    solve_categorical_fisher_natural_step_numpy_fp32,
)

MINIMUM_MLX_PARITY = 0.9997
MAXIMUM_STEP_ABSOLUTE_ERROR = 3e-5
MAXIMUM_QUADRATIC_ABSOLUTE_ERROR = 3e-6


@dataclass(frozen=True)
class MlxFisherNaturalTrustResult:
    """Independent MLX step paired with the NumPy-fp32 authority receipt."""

    step: Any
    trust_scale: Any
    numpy_receipt: FisherNaturalTrustResult
    parity: dict[str, float | bool | str]


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("MLX is required for the MLX Fisher natural solver") from exc
    return mx


def _agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.size != cand.size or ref.size == 0:
        raise ValueError("parity arrays must have the same non-empty size")
    if np.allclose(ref, ref[0]) or np.allclose(cand, cand[0]):
        return 1.0 if np.allclose(ref, cand, rtol=3e-4, atol=3e-6) else 0.0
    return float(np.corrcoef(ref, cand)[0, 1])


def solve_categorical_fisher_natural_step_mlx(
    probabilities: Any,
    cotangent: Any,
    *,
    delta: float,
    delta_convention: str = "delta_kl",
    damping: float = 0.0,
    project_gauge: bool = False,
    minimum_parity: float = MINIMUM_MLX_PARITY,
) -> MlxFisherNaturalTrustResult:
    """Run an MLX float32 quotient solve and fail closed below NumPy parity."""

    threshold = float(minimum_parity)
    if not np.isfinite(threshold) or not MINIMUM_MLX_PARITY <= threshold <= 1.0:
        raise ValueError(
            f"minimum_parity is non-lowerable and must be in [{MINIMUM_MLX_PARITY}, 1]"
        )
    reference = solve_categorical_fisher_natural_step_numpy_fp32(
        probabilities,
        cotangent,
        delta=delta,
        delta_convention=delta_convention,
        damping=damping,
        project_gauge=project_gauge,
    )
    mx = _mlx()
    p_np = np.asarray(probabilities, dtype=np.float32)
    g_np = np.asarray(cotangent, dtype=np.float32)
    if project_gauge:
        g_np = centre_cotangent(g_np).astype(np.float32)
    p = mx.array(p_np, dtype=mx.float32)
    g = mx.array(g_np, dtype=mx.float32)
    k = p_np.shape[-1]
    basis = mx.array(helmert_zero_sum_basis(k).astype(np.float32), dtype=mx.float32)
    identity = mx.eye(k - 1, dtype=mx.float32)
    flat_p = mx.reshape(p, (-1, k))
    flat_g = mx.reshape(g, (-1, k))
    rows = []
    for index in range(flat_p.shape[0]):
        row_p = flat_p[index]
        row_g = flat_g[index]
        hessian = mx.diag(row_p) - mx.outer(row_p, row_p)
        reduced = basis.T @ hessian @ basis + float(damping) * identity
        coordinate = mx.linalg.solve(reduced, -(basis.T @ row_g))
        rows.append(basis @ coordinate)
    raw = mx.stack(rows, axis=0)
    mean = mx.sum(flat_p * raw, axis=-1)
    q_before = mx.maximum(mx.sum(flat_p * raw * raw, axis=-1) - mean * mean, 0.0)
    _, delta_quad = convert_delta_budget(delta, delta_convention)
    safe = mx.maximum(q_before, np.finfo(np.float32).tiny)
    scale = mx.minimum(1.0, mx.sqrt(delta_quad / safe))
    scale = mx.where(
        q_before > delta_quad,
        scale * (1.0 - 16.0 * np.finfo(np.float32).eps),
        scale,
    )
    step = mx.reshape(raw * scale[:, None], p.shape)
    mx.eval(step, scale)

    step_np = np.asarray(step)
    q_np = categorical_fisher_quadratic(np.asarray(probabilities, dtype=np.float64), step_np)
    maximum_step_error = float(np.max(np.abs(step_np.astype(np.float64) - reference.step)))
    maximum_q_error = float(np.max(np.abs(q_np - reference.fisher_quadratic_after)))
    step_agreement = _agreement(reference.step, step_np)
    q_agreement = _agreement(reference.fisher_quadratic_after, q_np)
    gauge_error = float(np.max(np.abs(np.sum(step_np.astype(np.float64), axis=-1))))
    passed = bool(
        step_agreement >= threshold
        and q_agreement >= threshold
        and maximum_step_error <= MAXIMUM_STEP_ABSOLUTE_ERROR
        and maximum_q_error <= MAXIMUM_QUADRATIC_ABSOLUTE_ERROR
        and gauge_error <= 3e-6
        and np.all(q_np <= reference.delta_quad)
    )
    if not passed:
        raise FloatingPointError(
            "MLX categorical-Fisher natural-solve parity failed: "
            f"step={step_agreement:.8f}, q={q_agreement:.8f}, "
            f"max_step_error={maximum_step_error:.3e}, max_q_error={maximum_q_error:.3e}"
        )
    return MlxFisherNaturalTrustResult(
        step=step,
        trust_scale=mx.reshape(scale, p.shape[:-1]),
        numpy_receipt=reference,
        parity={
            "minimum_required": threshold,
            "step_correlation": step_agreement,
            "quadratic_correlation": q_agreement,
            "maximum_step_absolute_error": maximum_step_error,
            "maximum_quadratic_absolute_error": maximum_q_error,
            "maximum_gauge_error": gauge_error,
            "update_backend": "mlx_float32_helmert_quotient_linear_solve",
            "passed": passed,
        },
    )


__all__ = [
    "MINIMUM_MLX_PARITY",
    "MlxFisherNaturalTrustResult",
    "solve_categorical_fisher_natural_step_mlx",
]
