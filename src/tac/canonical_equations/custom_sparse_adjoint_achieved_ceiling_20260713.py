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
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "custom_sparse_adjoint_achieved_vs_ceiling_v1"
MEMO = ".omx/research/custom_sparse_adjoint_kernel_20260713.md"
MEASURED_MEMO = ".omx/research/custom_sparse_adjoint_metal_wall_MEASURED_20260714.md"
MEASURED_RECEIPT = (
    "experiments/results/custom_sparse_adjoint_kernel_metal_bench_20260714/"
    "measurement_receipt.json"
)
AXIS = "[macOS-MLX research-signal; NumPy-fp32 parity authority; non-promotable MEANS]"
DERIVED_FLAGSHIP_CEILING_X = 2.208577465069467
# MEASURED 2026-07-14 (D43 Metal wall memo, 125-conv wall replay, parity GREEN 40/40):
MEASURED_ACHIEVED_WALL_X = 0.7078  # whole-network SLOWDOWN vs the #212 dense Metal kernel
MEASURED_ETA = 0.3205  # achieved / derived flagship ceiling
MEASURED_DENSE_MEDIAN_MS = 65.356
MEASURED_SPARSE_MEDIAN_MS = 92.342


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
            # APPEND-ONLY history: the 2026-07-13 build registered
            # "BLOCKED_NO_METAL_IN_CURRENT_SANDBOX"; the 2026-07-14 D43 memo
            # MEASURED the wall on a live Metal device (parity GREEN 40/40,
            # max abs dev 6.68e-6 vs #212 dense over 125 shapes).
            "empirical_status": "METAL_WALL_MEASURED_20260714_WHOLE_NETWORK_SLOWDOWN_0p7078x",
            "empirical_status_history": ("BLOCKED_NO_METAL_IN_CURRENT_SANDBOX",),
            "verdict_scope": (
                "this custom per-input-site no-atomic compact Metal kernel schedule on this "
                "substrate; whole-network wall 0.7078x is a SLOWDOWN, eta 0.3205 of the 2.2086x "
                "derived ceiling; sparse pays only where support is genuinely sparse "
                "(seg-head/decoder margins), never as a whole-network replacement"
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
        empirical_anchors=(
            EmpiricalAnchor(
                anchor_id="metal_wall_125conv_replay_20260714",
                measurement_utc="2026-07-15T02:18:53.573027Z",
                inputs={
                    "dense_median_ms": MEASURED_DENSE_MEDIAN_MS,
                    "sparse_median_ms": MEASURED_SPARSE_MEDIAN_MS,
                    "wall_replay_convolutions": 125,
                    "parity_trials_bit_identical": "40/40 cross-process NumPy-fp32",
                    "max_abs_dev_vs_212_dense": 6.68e-6,
                },
                predicted_output=DERIVED_FLAGSHIP_CEILING_X,
                empirical_output=MEASURED_ACHIEVED_WALL_X,
                residual=DERIVED_FLAGSHIP_CEILING_X - MEASURED_ACHIEVED_WALL_X,
                source_artifact=MEASURED_RECEIPT,
                measurement_method=(
                    "125-convolution wall replay on a live Metal device via "
                    "tools/bench_custom_sparse_adjoint_kernel.py; median-of-trials; "
                    "sparse kernel parity-gated bit-identical to the NumPy-fp32 "
                    "authority before timing (D43 memo "
                    + MEASURED_MEMO
                    + ")"
                ),
                provenance=build_provenance_for_research_sidecar(
                    sidecar_path=MEASURED_MEMO,
                    reactivation_criteria=(
                        "hybrid layer-routing (sparse only on genuinely-sparse "
                        "seg-head/decoder-margin layers) plus an admitted "
                        "oracle-mask predictor; whole-network sparse replacement "
                        "is measured-dominated at eta 0.3205"
                    ),
                    measurement_axis="[macOS-MLX/Metal research-signal; advisory NON-score]",
                    hardware_substrate="Apple M5 Max, live Metal device, fp contract off",
                    captured_at_utc="2026-07-15T02:18:53.573027Z",
                ),
                empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
            ),
        ),
        predicted_vs_empirical_residual={
            "achieved_wall_vs_derived_ceiling_x": (
                DERIVED_FLAGSHIP_CEILING_X - MEASURED_ACHIEVED_WALL_X
            ),
            "eta_achieved_over_ceiling": MEASURED_ETA,
        },
        last_calibration_utc="2026-07-15T02:18:53.573027Z",
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
