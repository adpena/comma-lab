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
    """Run the MLX quotient solve (fp64 CPU-stream assembly+solve, fp32 trust
    scaling) and fail closed below NumPy parity."""

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
    p = mx.array(p_np, dtype=mx.float32)
    k = p_np.shape[-1]
    # Precision-critical quotient assembly + H^-1 solve run in float64 on the
    # CPU stream, mirroring the numpy reference (which assembles and solves in
    # float64 and only casts the FINAL step to fp32).  Two Metal-host MEASURED
    # facts force this (the arm that landed this adapter had no Metal and its
    # parity test was skipped): (a) mx.linalg.solve has NO Metal/GPU kernel
    # (ValueError "not yet supported on the GPU"), so the solve must take a CPU
    # stream regardless; (b) an all-fp32 assembly+solve lands at max_step_error
    # 8.9e-5 > the non-lowerable 3e-5 bound (input/assembly rounding, not the
    # solve — an fp64 solve over fp32 assembly measures identically).  The
    # trust-radius scaling below stays on the default-device fp32 path; the
    # non-lowerable parity constants stay untouched; numpy-fp32 remains the
    # bit-authority.
    p64_np = np.asarray(probabilities, dtype=np.float64)
    g64_np = np.asarray(cotangent, dtype=np.float64)
    if project_gauge:
        g64_np = centre_cotangent(g64_np)
    # Every float64 op below is pinned to the CPU stream — MLX float64 has no
    # GPU support at all ("float64 is not supported on the GPU").
    flat_p64 = mx.array(p64_np.reshape(-1, k), dtype=mx.float64)
    flat_g64 = mx.array(g64_np.reshape(-1, k), dtype=mx.float64)
    basis64 = mx.array(helmert_zero_sum_basis(k), dtype=mx.float64)
    basis64_t = mx.transpose(basis64, stream=mx.cpu)
    identity64 = mx.eye(k - 1, dtype=mx.float64, stream=mx.cpu)
    rows = []
    for index in range(flat_p64.shape[0]):
        row_p = mx.take(flat_p64, index, axis=0, stream=mx.cpu)
        row_g = mx.take(flat_g64, index, axis=0, stream=mx.cpu)
        hessian = mx.subtract(
            mx.diag(row_p, stream=mx.cpu),
            mx.outer(row_p, row_p, stream=mx.cpu),
            stream=mx.cpu,
        )
        reduced = mx.add(
            mx.matmul(
                mx.matmul(basis64_t, hessian, stream=mx.cpu),
                basis64,
                stream=mx.cpu,
            ),
            mx.multiply(
                mx.array(float(damping), dtype=mx.float64), identity64, stream=mx.cpu
            ),
            stream=mx.cpu,
        )
        rhs = mx.negative(
            mx.matmul(basis64_t, row_g, stream=mx.cpu), stream=mx.cpu
        )
        coordinate = mx.linalg.solve(reduced, rhs, stream=mx.cpu)
        rows.append(
            mx.matmul(basis64, coordinate, stream=mx.cpu).astype(
                mx.float32, stream=mx.cpu
            )
        )
    raw = mx.stack(rows, axis=0)
    flat_p = mx.reshape(p, (-1, k))
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
            "update_backend": "mlx_float64_cpu_helmert_quotient_solve_float32_trust_scale",
            "passed": passed,
        },
    )


__all__ = [
    "MINIMUM_MLX_PARITY",
    "MlxFisherNaturalTrustResult",
    "solve_categorical_fisher_natural_step_mlx",
]
