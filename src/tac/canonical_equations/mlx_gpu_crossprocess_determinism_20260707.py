# SPDX-License-Identifier: MIT
"""Canonical equation: MLX-GPU cross-process determinism law (task #348, 2026-07-07).

Closes the #205 risk-register D6 named gap ("no REGISTERED equation yet — candidate for a future
``mlx_gpu_crossprocess_nondeterminism_v1``"). Refines memory L70
(``mlx_gpu_not_bit_identical_crossprocess_bitexact_proof_cpu_locked_20260702``): the 28/28
divergence was NOT "MLX-GPU compute in general" — it is ONE op class.

THE LAW (measured, tools/mlx_gpu_determinism_probe.py, N=10 per cell, M5 Max / MLX 0.31.2):

    det(op) = TRUE  for every fixed-order kernel: GEMM (all shapes incl. huge-K/GEMV),
              conv2d (incl. grouped/strided), reductions/softmax, seeded random,
              elementwise, the custom grouped-backward Metal kernel, the fused-R Metal
              forward AND its fixed-order transpose VJP;
    det(op) = FALSE for duplicate-index ATOMIC accumulation: ``arr.at[idx].add`` scatter
              (10/10 unique hashes, not even in-process repeatable) and the ``mx.take``
              VJP whenever its cotangent is non-trivially strided — i.e. the gather-based
              ``_resize_axis_nhwc`` bicubic-UP backward of the reference R operator.

COROLLARY (the witness consequence): the reference-R backward poisons every gradient from
epoch 1 => the historical "28/28 tensors diverge cross-process" verdict. Swapping R to the
fused Metal kernel (``--fused-r-kernel``, #252 — fixed-order, no atomics) makes the FULL
witness trainer cross-process BIT-IDENTICAL on GPU (0/28 diverged, N=10; Muon-finisher arm
0/28, N=5) at NEGATIVE overhead (~8% FASTER: 25.35s -> 23.44s, 200-ep n=1 smoke).

Authority discipline: every verdict here is a [macOS-MLX research-signal] bit-identity FACT
(hash equality), never a score. CPU (numpy-fp32 / MLX-CPU) remains the verdict authority;
this law WIDENS where bit-exact proofs may run (GPU allowed iff every op in the graph is in
the det=TRUE class — operationally: fused-R ON, no atomic scatter in the graph).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

MLX_GPU_CROSSPROCESS_DETERMINISM_EQUATION_ID = "mlx_gpu_crossprocess_nondeterminism_v1"

_MLX_SIGNAL = "[macOS-MLX research-signal]"

_LEDGER = ".omx/research/deterministic_gpu_accum_348_20260707.md"
_PROBE = "tools/mlx_gpu_determinism_probe.py"
_TESTS = "src/tac/tests/test_mlx_gpu_determinism.py"

# MEASURED constants (M5 Max, MLX 0.31.2, N per the ledger).
N_PROCESSES_PER_CELL = 10
WITNESS_DIVERGED_REFERENCE_R = 28   # /28 tensors, epochs>=1, reference R (gather backward)
WITNESS_DIVERGED_FUSED_R = 0        # /28 tensors, N=10 (+ Muon arm N=5)
# [empirical:.omx/research/deterministic_gpu_accum_348_20260707.md] 200-ep n=1 96x128
# timing smoke: reference 25.35s vs fused-R 23.44s. [macOS-CPU advisory] — never a score.
FUSED_R_SPEEDUP_RATIO = 25.35 / 23.44  # ~1.08x FASTER (negative determinism overhead)


def build_mlx_gpu_crossprocess_determinism_v1() -> CanonicalEquation:
    """Build the MLX-GPU cross-process determinism law (measured localization + cure)."""

    anchor_localization = EmpiricalAnchor(
        anchor_id="mlx_gpu_op_class_localization_n10_20260707",
        measurement_utc="2026-07-07T00:00:00Z",
        inputs={
            "instrument": _PROBE,
            "cells": "19 op classes x N=10 separate processes, sha256 of fp32 output bytes",
            "device": "M5 Max Metal (MLX 0.31.2), default arch + applegpu_g15 override",
        },
        predicted_output={"nondeterministic_ops": ["scatter_add_dup", "r_up_grad", "r_full_grad"]},
        empirical_output={
            "deterministic": "GEMM/conv/reductions/softmax/random/elementwise/custom-grouped-"
                             "backward/fused-R-fwd/fused-R-vjp: 1 unique hash in 10, all cells",
            "nondeterministic": "arr.at[idx].add duplicate-index scatter: 10 unique hashes in 10 "
                                "(in-process non-repeatable); reference-R bicubic-UP backward "
                                "(take-VJP with strided cotangent): unique hash per process",
            "arch_invariance": "divergence persists under MLX_METAL_GPU_ARCH=applegpu_g15 "
                               "(NOT the NAX kernel-selection class)",
        },
        residual=0.0,
        source_artifact=_LEDGER,
        measurement_method="cross_process_sha256_hash_equality_n10_per_op_cell",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=f"re-run {_PROBE} (any MLX upgrade) + {_TESTS}",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_metal_gpu",
        ),
    )
    anchor_witness_cure = EmpiricalAnchor(
        anchor_id="witness_trainer_fused_r_crossprocess_bit_identity_20260707",
        measurement_utc="2026-07-07T00:00:00Z",
        inputs={
            "config": "train_levelset_witness_realized_through_R_mlx.py, n=1 pair, 96x128, 5ep "
                      "CE/tau/l7 curriculum (+ Muon-finisher arm --muon-start-epoch 3), seed 0, "
                      "--mlx-device gpu",
            "cells": "reference-R (N=3) vs --fused-r-kernel (N=10) vs Muon+fused (N=5) vs CPU (N=3)",
        },
        predicted_output={"fused_r_diverged_tensors": 0},
        empirical_output={
            "reference_R_gpu": "28/28 liveP+emaP tensors diverge (reproduces the L70 verdict; "
                               "divergence begins at ep1 — ep0 verdict identical)",
            "fused_R_gpu": "0/28 diverged, N=10; Muon arm 0/28, N=5",
            "cpu": "0/28 diverged (CPU-locked discipline confirmed)",
            "overhead": "fused-R ~8% FASTER (25.35s -> 23.44s, 200-ep timing smoke)",
        },
        residual=0.0,
        source_artifact=_LEDGER,
        measurement_method="full_trainer_cross_process_resume_state_tensor_hash_equality",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="re-run the witness cross-process cells after any MLX/trainer "
                                  "R-path change; scope: verified at n=1 5-20ep smoke configs, "
                                  "self-orient OFF (numpy-side, expected neutral) — n600 composite "
                                  "re-verification owed before relying on GPU bit-identity there",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_metal_gpu",
        ),
    )
    return CanonicalEquation(
        equation_id=MLX_GPU_CROSSPROCESS_DETERMINISM_EQUATION_ID,
        name="MLX-GPU cross-process determinism law (atomic-scatter class vs fixed-order class)",
        one_line_summary=(
            "MLX-GPU nondeterminism is ONE op class (dup-index atomic scatter-add, incl. the "
            "reference-R-UP backward); fused-R fixed-order VJP => witness bit-identical, ~8% faster"
        ),
        latex_form=(
            r"\mathrm{det}(op)=\begin{cases}\mathrm{TRUE}&op\in\{\text{fixed-order kernels}\}\\"
            r"\mathrm{FALSE}&op\in\{\oplus_{\text{atomic}}\ \text{dup-index scatter}\}\end{cases}"
            r"\quad\Rightarrow\quad \mathrm{det}(\text{graph})=\bigwedge_{op\in\text{graph}}\mathrm{det}(op)"
        ),
        python_callable_module_path="mlx_gpu_determinism_probe:probe_cell",  # tools/ (on PYTHONPATH via tools dir)
        domain_of_validity={
            "hardware": ["m5_max_metal_gpu"],
            "mlx_version": "0.31.2 (re-probe on upgrade; the probe is the instrument)",
            "measurement_axis": [_MLX_SIGNAL],
            "note": "bit-identity facts only, never a score; CPU remains the verdict authority; "
                    "witness composite verified at n=1 smoke scale (n600 re-verification owed)",
        },
        units_in={"op": "MLX graph op class", "n_processes": "count"},
        units_out={"det": "boolean (cross-process sha256 hash equality)"},
        empirical_anchors=(anchor_localization, anchor_witness_cure),
        predicted_vs_empirical_residual={"witness_fused_r_diverged_tensors": 0.0},
        last_calibration_utc="2026-07-07T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments/tests/test_levelset_crash_resume_smoke.py",
            "src/tac/tests/test_mlx_gpu_determinism.py",
        ),
        canonical_producers=(
            "tools/mlx_gpu_determinism_probe.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="mlx_gpu_crossprocess_determinism.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_metal_gpu",
        ),
    )


def populate_mlx_gpu_crossprocess_determinism_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins query semantics)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_mlx_gpu_crossprocess_determinism_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="task_348_deterministic_gpu_accumulation_20260707",
    )
    return eq


__all__ = [
    "MLX_GPU_CROSSPROCESS_DETERMINISM_EQUATION_ID",
    "build_mlx_gpu_crossprocess_determinism_v1",
    "populate_mlx_gpu_crossprocess_determinism_equation",
]
