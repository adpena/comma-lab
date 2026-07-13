# SPDX-License-Identifier: MIT
"""Achieved-versus-ceiling and state-stable basis-reuse laws.

The arithmetic ceiling and a measured wall speedup are different objects.  If
``C = F_dense/F_sparse`` and ``A = T_dense/T_sparse``, then ``eta=A/C`` is the
fraction of the ideal arithmetic saving realized in wall time.  The residual
time ``T_sparse - T_dense/C`` charges gather/map traffic, launch latency,
occupancy loss, and any remaining dense work without pretending to identify a
specific hardware cause absent counters.

A rank-r basis computation of duration ``T_basis`` wins only when a
state-stable Jacobian lets it serve ``K`` steps and ``T_basis/K<T_dense``.
These are compute laws; mask prediction and Jacobian-stability accuracy remain
separate gates.
"""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "custom_sparse_adjoint_achieved_vs_ceiling_v1"
MEMO = ".omx/research/custom_sparse_adjoint_kernel_20260713.md"
AXIS = "[macOS-MLX research-signal; NumPy-fp32 parity authority; non-promotable MEANS]"
DERIVED_FLAGSHIP_CEILING_X = 2.208577465069467


def _positive_finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def achieved_vs_ceiling_law(
    *,
    dense_flops: float,
    sparse_flops: float,
    dense_time: float,
    sparse_time: float,
) -> dict[str, float]:
    """Return arithmetic ceiling, wall speedup, efficiency, and residual time."""

    dense_f = _positive_finite(dense_flops, name="dense_flops")
    sparse_f = _positive_finite(sparse_flops, name="sparse_flops")
    dense_t = _positive_finite(dense_time, name="dense_time")
    sparse_t = _positive_finite(sparse_time, name="sparse_time")
    if sparse_f > dense_f:
        raise ValueError("sparse_flops cannot exceed dense_flops")
    ceiling = dense_f / sparse_f
    achieved = dense_t / sparse_t
    efficiency = achieved / ceiling
    ideal_sparse_time = dense_t / ceiling
    return {
        "arithmetic_ceiling_x": ceiling,
        "achieved_wall_speedup_x": achieved,
        "achieved_to_ceiling_ratio": efficiency,
        "relative_ceiling_gap": 1.0 - efficiency,
        "absolute_speedup_gap_x": ceiling - achieved,
        "ideal_sparse_time": ideal_sparse_time,
        "residual_sparse_time_above_flop_scaled_ideal": sparse_t - ideal_sparse_time,
    }


def basis_reuse_law(
    *, dense_vjp_time_per_step: float, basis_vjp_time: float, reuse_steps: int
) -> dict[str, float | bool]:
    """Return the exact compute crossover for a cached basis over ``K`` steps."""

    dense = _positive_finite(dense_vjp_time_per_step, name="dense_vjp_time_per_step")
    basis = _positive_finite(basis_vjp_time, name="basis_vjp_time")
    if isinstance(reuse_steps, bool) or not isinstance(reuse_steps, int) or reuse_steps <= 0:
        raise ValueError("reuse_steps must be a positive integer")
    amortized = basis / reuse_steps
    return {
        "basis_time_per_served_step": amortized,
        "amortized_speedup_x": dense / amortized,
        "no_reuse_speed_factor_x": dense / basis,
        "minimum_reuse_steps_strict": math.floor(basis / dense) + 1,
        "wins": amortized < dense,
    }


def custom_sparse_adjoint_compute_laws(
    *,
    dense_flops: float,
    sparse_flops: float,
    dense_time: float,
    sparse_time: float,
    dense_vjp_time_per_step: float,
    basis_vjp_time: float,
    reuse_steps: int,
) -> dict[str, dict[str, float | bool]]:
    """Evaluate the sparse-wall and basis-reuse laws together."""

    return {
        "sparse_wall": achieved_vs_ceiling_law(
            dense_flops=dense_flops,
            sparse_flops=sparse_flops,
            dense_time=dense_time,
            sparse_time=sparse_time,
        ),
        "basis_reuse": basis_reuse_law(
            dense_vjp_time_per_step=dense_vjp_time_per_step,
            basis_vjp_time=basis_vjp_time,
            reuse_steps=reuse_steps,
        ),
    }


def build_custom_sparse_adjoint_achieved_vs_ceiling_v1() -> CanonicalEquation:
    """Build the analytic compute law; empirical Metal anchor remains fail-closed."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "a real Metal execution surface must pass N=10 cross-process NumPy-fp32 dense-on-"
            "support parity and seal the 125-convolution wall replay receipt"
        ),
        measurement_axis=AXIS,
        hardware_substrate="Apple M5 Max host visible; Metal unavailable in current sandbox",
        captured_at_utc="2026-07-13T23:15:06Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Custom sparse adjoint achieved-versus-ceiling and basis-reuse law",
        one_line_summary=(
            "Wall realization is A/C, while rank-r basis VJPs win only when state-stable "
            "reuse makes T_basis/K smaller than one dense VJP."
        ),
        latex_form=(
            r"C=F_d/F_s,quad A=T_d/T_s,quad \eta=A/C="
            r"(T_d/F_d)/(T_s/F_s);qquad "
            r"A_{r,K}=K T_d/T_{\mathrm{basis}},\quad "
            r"\mathrm{win}\iff T_{\mathrm{basis}}/K<T_d"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.custom_sparse_adjoint_achieved_ceiling_20260713:"
            "custom_sparse_adjoint_compute_laws"
        ),
        domain_of_validity={
            "research_only": True,
            "included": (
                "frozen-convolution input VJP; caller-supplied compact spatial support; "
                "rank-batched basis; matched dense/sparse timing on one hardware fingerprint"
            ),
            "excluded": (
                "mask prediction; global squeeze-excite approximation; nonlinear full-SegNet VJP; "
                "optimizer regret; live training; score or pointer authority"
            ),
            "flagship_derived_ceiling_x": DERIVED_FLAGSHIP_CEILING_X,
            "empirical_status": "BLOCKED_NO_METAL_IN_CURRENT_SANDBOX",
            "verdict_scope": (
                "execution-substrate blocker only; source primitive built default-off; no kernel-wall "
                "or sparse-adjoint-family verdict until the host parity/bench receipt exists"
            ),
            "req_R": (
                "N=10 bit-identical cross-process parity plus measured 125-convolution wall replay; "
                "accuracy admission separately requires oracle-mask predictor and K=2 reuse gates"
            ),
        },
        units_in={
            "dense_flops": "floating_point_operations",
            "sparse_flops": "floating_point_operations",
            "dense_time": "seconds",
            "sparse_time": "seconds",
            "basis_vjp_time": "seconds",
            "reuse_steps": "dimensionless_integer",
        },
        units_out={
            "ceiling": "dimensionless_ratio",
            "wall_speedup": "dimensionless_ratio",
            "efficiency": "dimensionless_ratio",
            "residual_time": "seconds",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-13T23:15:06Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.local_acceleration.metal_sparse_adjoint",
            "tac.cathedral_autopilot",
        ),
        canonical_producers=(
            "tools.bench_custom_sparse_adjoint_kernel",
            "tac.local_acceleration.metal_sparse_adjoint",
        ),
        provenance=provenance,
    )


def populate_custom_sparse_adjoint_achieved_vs_ceiling_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicit append-only registration surface; never called at import time."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_custom_sparse_adjoint_achieved_vs_ceiling_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="FEED-custom-sparse-adjoint; design law; Metal empirical anchor blocked",
    )
    return equation


__all__ = [
    "AXIS",
    "DERIVED_FLAGSHIP_CEILING_X",
    "EQUATION_ID",
    "achieved_vs_ceiling_law",
    "basis_reuse_law",
    "build_custom_sparse_adjoint_achieved_vs_ceiling_v1",
    "custom_sparse_adjoint_compute_laws",
    "populate_custom_sparse_adjoint_achieved_vs_ceiling_v1",
]
