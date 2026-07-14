# SPDX-License-Identifier: MIT
"""Lazy MLX adapter and parity checks for categorical Fisher clipping.

NumPy-fp32 remains verdict authority.  This module never selects a device and
does not import MLX until a public function is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.ripo_fisher_trust_region import (
    FisherClipResult,
    categorical_exact_kl,
    categorical_fisher_lambda_max,
    categorical_fisher_quadratic,
    centre_logits,
    clip_categorical_fisher_step_numpy_fp32,
    convert_delta_budget,
)

MINIMUM_MLX_PARITY = 0.9997
MAXIMUM_ABSOLUTE_ERROR = 2e-6
MAXIMUM_RELATIVE_ERROR = 5e-3
MAXIMUM_ALPHA_ABSOLUTE_ERROR = 2e-5
MAXIMUM_DIRECTION_RESIDUAL = 2e-6


@dataclass(frozen=True)
class MlxFisherClipResult:
    """MLX arrays plus the complete NumPy-authority receipt."""

    centred_input: Any
    centred_output: Any
    alpha: Any
    clipped: Any
    numpy_receipt: FisherClipResult
    parity: dict[str, float | bool | str]


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as error:  # pragma: no cover - host dependent
        raise RuntimeError("MLX is required for the MLX Fisher adapter") from error
    return mx


def categorical_fisher_quadratic_mlx(probabilities: Any, logit_step: Any) -> Any:
    """Evaluate the categorical Fisher quadratic with MLX float32 ops."""

    mx = _mlx()
    p = mx.array(probabilities, dtype=mx.float32)
    u = mx.array(logit_step, dtype=mx.float32)
    if tuple(p.shape) != tuple(u.shape) or len(p.shape) < 1:
        raise ValueError("probabilities and logit_step must have the same non-scalar shape")
    mean = mx.sum(p * u, axis=-1)
    return mx.maximum(mx.sum(p * u * u, axis=-1) - mean * mean, 0.0)


def categorical_exact_kl_mlx(probabilities: Any, logit_step: Any) -> Any:
    """Evaluate finite categorical KL with a small-step stable MLX branch."""

    mx = _mlx()
    p = mx.array(probabilities, dtype=mx.float32)
    u = mx.array(logit_step, dtype=mx.float32)
    if tuple(p.shape) != tuple(u.shape) or len(p.shape) < 1:
        raise ValueError("probabilities and logit_step must have the same non-scalar shape")
    p = p / mx.sum(p, axis=-1, keepdims=True)
    weighted_mean = mx.sum(p * u, axis=-1, keepdims=True)
    centred = u - weighted_mean
    clipped = mx.clip(centred, -1e-3, 1e-3)
    square = clipped * clipped
    remainder = square * (
        0.5
        + clipped
        * (
            1.0 / 6.0
            + clipped
            * (1.0 / 24.0 + clipped * (1.0 / 120.0 + clipped * (1.0 / 720.0)))
        )
    )
    residual = mx.sum(p * clipped, axis=-1)
    small_kl = mx.log1p(mx.maximum(residual + mx.sum(p * remainder, axis=-1), 0.0))
    maximum = mx.max(u, axis=-1, keepdims=True)
    log_partition = mx.squeeze(maximum, axis=-1) + mx.log(
        mx.sum(p * mx.exp(u - maximum), axis=-1)
    )
    large_kl = mx.maximum(log_partition - mx.sum(p * u, axis=-1), 0.0)
    return mx.where(mx.max(mx.abs(centred), axis=-1) <= 1e-3, small_kl, large_kl)


def _agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.size != cand.size or ref.size == 0:
        raise ValueError("parity arrays must have the same non-empty size")
    if np.allclose(ref, ref[0]) or np.allclose(cand, cand[0]):
        return 1.0 if np.allclose(ref, cand, rtol=3e-5, atol=3e-7) else 0.0
    return float(np.corrcoef(ref, cand)[0, 1])


def _absolute_relative_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    difference = np.abs(cand - ref)
    absolute = float(np.max(difference))
    relative = float(np.max(difference / np.maximum(np.abs(ref), 1e-6)))
    return absolute, relative


def clip_categorical_fisher_step_mlx(
    probabilities: Any,
    proposed_logit_step: Any,
    *,
    delta: float,
    delta_convention: str,
    mode: str,
    tolerance: float,
    minimum_parity: float = MINIMUM_MLX_PARITY,
) -> MlxFisherClipResult:
    """Return MLX arrays certified against the NumPy-fp32 authority.

    Root decisions and the float32 post-cast constraint are intentionally made
    by the NumPy authority.  MLX independently recomputes the realized Fisher
    quadratic and finite KL; parity below ``minimum_parity`` fails closed.
    """

    threshold = float(minimum_parity)
    if not np.isfinite(threshold) or not MINIMUM_MLX_PARITY <= threshold <= 1.0:
        raise ValueError(
            f"minimum_parity is non-lowerable and must be in [{MINIMUM_MLX_PARITY}, 1]"
        )
    mx = _mlx()
    receipt = clip_categorical_fisher_step_numpy_fp32(
        probabilities,
        proposed_logit_step,
        delta=delta,
        delta_convention=delta_convention,
        mode=mode,
        tolerance=tolerance,
    )
    p_mx = mx.array(np.asarray(probabilities, dtype=np.float32), dtype=mx.float32)
    proposed_mx = mx.array(np.asarray(proposed_logit_step, dtype=np.float32), dtype=mx.float32)
    if mode == "local_directional":
        # This is a genuine MLX update surface, not just a conversion of the
        # NumPy output.  NumPy independently makes the authoritative decision.
        centred_mx = proposed_mx - mx.mean(proposed_mx, axis=-1, keepdims=True)
        q_before_mx = categorical_fisher_quadratic_mlx(p_mx, centred_mx)
        _, delta_quad = convert_delta_budget(delta, delta_convention)
        safe_q = mx.maximum(q_before_mx, np.finfo(np.float32).tiny)
        analytic_alpha = mx.minimum(1.0, mx.sqrt(delta_quad / safe_q))
        conservative_alpha = mx.where(
            q_before_mx > delta_quad,
            analytic_alpha * (1.0 - 32.0 * np.finfo(np.float32).eps),
            analytic_alpha,
        )
        output_mx = centred_mx * conservative_alpha[..., None]
        update_backend = "mlx_float32_local_directional"
    else:
        # Finite-KL roots deliberately remain NumPy-float64 authority; MLX
        # independently evaluates both defining formulas on the chosen root.
        output_mx = mx.array(receipt.centred_output, dtype=mx.float32)
        update_backend = "numpy_float64_exact_root_mlx_formula_check"
    q_mx = categorical_fisher_quadratic_mlx(p_mx, output_mx)
    kl_mx = categorical_exact_kl_mlx(p_mx, output_mx)
    mx.eval(q_mx, kl_mx, output_mx)
    q_np = np.asarray(q_mx)
    kl_np = np.asarray(kl_mx)
    q_agreement = _agreement(receipt.q_after, q_np)
    kl_agreement = _agreement(receipt.exact_kl_after, kl_np)
    maximum_q_error, maximum_q_relative_error = _absolute_relative_error(receipt.q_after, q_np)
    maximum_kl_error, maximum_kl_relative_error = _absolute_relative_error(
        receipt.exact_kl_after,
        kl_np,
    )
    output_np = np.asarray(output_mx)
    output_agreement = _agreement(receipt.centred_output, output_np)
    maximum_output_error, maximum_output_relative_error = _absolute_relative_error(
        receipt.centred_output,
        output_np,
    )
    direction_np = centre_logits(np.asarray(proposed_logit_step, dtype=np.float32))
    direction_norm_sq = np.sum(direction_np * direction_np, axis=-1, dtype=np.float64)
    realized_alpha = np.ones(direction_np.shape[:-1], dtype=np.float64)
    np.divide(
        np.sum(direction_np * output_np.astype(np.float64), axis=-1, dtype=np.float64),
        direction_norm_sq,
        out=realized_alpha,
        where=direction_norm_sq > 0.0,
    )
    direction_residual = output_np.astype(np.float64) - realized_alpha[..., None] * direction_np
    maximum_direction_residual = float(np.max(np.abs(direction_residual)))
    maximum_alpha_error, maximum_alpha_relative_error = _absolute_relative_error(
        receipt.alpha,
        realized_alpha,
    )

    p_np = np.asarray(probabilities, dtype=np.float64)
    output64 = output_np.astype(np.float64)
    returned_q = categorical_fisher_quadratic(p_np, output64)
    returned_kl = categorical_exact_kl(p_np, output64)
    delta_kl, delta_quad = convert_delta_budget(delta, delta_convention)
    if mode == "local_directional":
        constraint_measure = returned_q
        constraint_limit = delta_quad
    elif mode == "exact_kl":
        constraint_measure = returned_kl
        constraint_limit = delta_kl
    elif mode == "local_euclidean_ball":
        lambda_max = categorical_fisher_lambda_max(p_np)
        constraint_measure = lambda_max * np.sum(output64 * output64, axis=-1, dtype=np.float64)
        constraint_limit = delta_quad
    else:
        constraint_measure = np.sum(output64 * output64, axis=-1, dtype=np.float64)
        constraint_limit = delta_quad
    maximum_constraint_excess = float(np.max(constraint_measure - constraint_limit))
    constraint_passed = bool(maximum_constraint_excess <= 0.0)
    passed = bool(
        q_agreement >= threshold
        and kl_agreement >= threshold
        and output_agreement >= threshold
        and maximum_q_error <= MAXIMUM_ABSOLUTE_ERROR
        and maximum_kl_error <= MAXIMUM_ABSOLUTE_ERROR
        and maximum_output_error <= MAXIMUM_ABSOLUTE_ERROR
        and maximum_q_relative_error <= MAXIMUM_RELATIVE_ERROR
        and maximum_kl_relative_error <= MAXIMUM_RELATIVE_ERROR
        and maximum_output_relative_error <= MAXIMUM_RELATIVE_ERROR
        and maximum_alpha_error <= MAXIMUM_ALPHA_ABSOLUTE_ERROR
        and maximum_alpha_relative_error <= MAXIMUM_RELATIVE_ERROR
        and maximum_direction_residual <= MAXIMUM_DIRECTION_RESIDUAL
        and np.all(realized_alpha >= 0.0)
        and np.all(realized_alpha <= 1.0)
        and constraint_passed
        and np.all(np.isfinite(q_np))
        and np.all(np.isfinite(kl_np))
    )
    if not passed:
        raise FloatingPointError(
            "MLX categorical-Fisher parity failed: "
            f"q={q_agreement:.8f}, kl={kl_agreement:.8f}, "
            f"output={output_agreement:.8f}, required={threshold:.8f}, "
            f"constraint_excess={maximum_constraint_excess:.3e}"
        )
    return MlxFisherClipResult(
        centred_input=mx.array(receipt.centred_input, dtype=mx.float32),
        centred_output=output_mx,
        alpha=mx.array(realized_alpha, dtype=mx.float32),
        clipped=mx.array(realized_alpha < 1.0 - 8.0 * np.finfo(np.float32).eps),
        numpy_receipt=receipt,
        parity={
            "minimum_required": threshold,
            "q_correlation": q_agreement,
            "exact_kl_correlation": kl_agreement,
            "output_correlation": output_agreement,
            "maximum_q_absolute_error": maximum_q_error,
            "maximum_q_relative_error": maximum_q_relative_error,
            "maximum_exact_kl_absolute_error": maximum_kl_error,
            "maximum_exact_kl_relative_error": maximum_kl_relative_error,
            "maximum_output_absolute_error": maximum_output_error,
            "maximum_output_relative_error": maximum_output_relative_error,
            "maximum_alpha_absolute_error": maximum_alpha_error,
            "maximum_alpha_relative_error": maximum_alpha_relative_error,
            "maximum_direction_residual": maximum_direction_residual,
            "maximum_constraint_excess": maximum_constraint_excess,
            "returned_output_constraint_passed": constraint_passed,
            "update_backend": update_backend,
            "passed": passed,
        },
    )


clip_categorical_fisher_step_mlx_fp32 = clip_categorical_fisher_step_mlx


__all__ = [
    "MlxFisherClipResult",
    "categorical_exact_kl_mlx",
    "categorical_fisher_quadratic_mlx",
    "clip_categorical_fisher_step_mlx",
    "clip_categorical_fisher_step_mlx_fp32",
]
