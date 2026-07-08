# SPDX-License-Identifier: MIT
"""Canonical equation: safe-compile per-DEVICE bit-identity (the hosc flip admissibility law).

Crux-engineering landing 2026-07-08 (operator: "All are fixable through crux engineering"):
the GPU per-chip fingerprint bit-cert (memo .omx/research/safe_compile_gpu_bitcert_20260708.md,
commits 5cc9cc1d6/ace035eb8) MEASURED the law this module registers, so it is never re-derived:

THE LAW. Whether ``mx.compile`` of a fused elementwise region is BIT-IDENTICAL to the eager
path is a PER-{chip, macos_build, mlx_version, device} EMPIRICAL FACT — never a global property
of the region. Anchor pair (same chip, same region, same coverage): hosc ``tanh(beta*sin(x))``
under mx.compile is bit-identical on M5-Max-GPU at REAL witness coverage (384x512 grid =
(196608,96), beta uniformly swept over [1,10], N=5 cross-process, max|delta| = 0.0) yet
fp-CONTRACTS by 1 ULP (5.96e-8) on the SAME chip's CPU device. Therefore admissibility of any
compiled-region flip MUST be resolved from a fingerprint-stamped manifest per device
(``resolve_enabled_regions``; launcher b2 fail-closed on fingerprint/device mismatch) — never
assumed, never transferred across devices.

REFINES ``mlx_gpu_crossprocess_nondeterminism_v1`` (memory L70): the GPU wall is CONFINED to
dup-index atomic scatter; fused ELEMENTWISE regions are cross-process deterministic AND
compile-bit-identical on this GPU. The two laws together give the per-config decision procedure:
scatter-class ops -> fixed-order/#348 treatment; elementwise regions -> per-device cert manifest.

means != ends: a bit-identity cert is an on-device fact, never a score. Pointer contest-CPU
0.19110 UNMOVED. Axis tags: all anchors [macOS-MLX research-signal]/[macOS-CPU advisory],
score_claim=false, promotable=false.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

SAFE_COMPILE_DEVICE_BITIDENTITY_EQUATION_ID = "safe_compile_hosc_device_bitidentity_v1"

_MLX_SIGNAL = "[macOS-MLX research-signal]"
_ADVISORY = "[macOS-CPU advisory]"

_MEMO = ".omx/research/safe_compile_gpu_bitcert_20260708.md"
_MODULE = "src/tac/mlx_safe_compile.py"

# MEASURED on M5 Max (fingerprint: Apple M5 Max / macOS 25E246 / mlx 0.31.2), 2026-07-08.
GPU_HOSC_MAX_ABS_DELTA = 0.0          # bit-identical, N=5 cross-process, real 384x512 coverage
CPU_HOSC_MAX_ABS_DELTA = 5.96e-8      # 1-ULP fp-contraction => REFUSE on cpu device
HOSC_BETA_COVERAGE = (1.0, 10.0)      # uniform sweep incl. endpoints (v7 anneal range)
CERT_N_CROSS_PROCESS = 5


def build_safe_compile_hosc_device_bitidentity_v1() -> CanonicalEquation:
    """Build the per-device compile-bit-identity equation. BOTH anchors are MEASURED
    (VERIFIED_VIA_EMPIRICAL_ANCHOR): the GPU ADMIT fact and the CPU REFUSE fact — the
    device-split IS the law."""

    anchor_gpu_admit = EmpiricalAnchor(
        anchor_id="safe_compile_hosc_gpu_bit_identical_real_coverage_20260708",
        measurement_utc="2026-07-08T16:40:00Z",
        inputs={
            "region": "hosc_activation tanh(beta*sin(x)) under mx.compile",
            "device": "gpu", "chip": "Apple M5 Max", "macos_build": "25E246", "mlx": "0.31.2",
            "coverage": "real 384x512 grid (196608,96), 32 adversarial inputs, "
                        "beta uniform over [1,10] incl. endpoints",
        },
        predicted_output={"max_abs_delta": 0.0, "n_cross_process_identical": CERT_N_CROSS_PROCESS},
        empirical_output={
            "max_abs_delta": GPU_HOSC_MAX_ABS_DELTA,
            "n_cross_process_identical": CERT_N_CROSS_PROCESS,
            "verdict": "ADMIT (hosc _act flip admissible at v7 launch on THIS fingerprint)",
            "all_regions": "9/9 SAFE regions CERTIFIED 0.0 + ce_reduction fixed-order; "
                           "kernel_candidates == []",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="mlx_safe_compile_v2_gpu_cert_real_coverage_n5_crossprocess",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="re-run src/tac/mlx_safe_compile.py certification on the target "
                                  "fingerprint; any chip/os/mlx/device change fails closed and "
                                  "recertifies (launcher b2)",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_gpu",
        ),
    )
    anchor_cpu_refuse = EmpiricalAnchor(
        anchor_id="safe_compile_hosc_cpu_fp_contraction_refuse_20260708",
        measurement_utc="2026-07-08T13:00:00Z",
        inputs={
            "region": "hosc_activation tanh(beta*sin(x)) under mx.compile",
            "device": "cpu", "chip": "Apple M5 Max",
            "note": "SAME chip, SAME region, SAME harness as the GPU anchor — only device differs",
        },
        predicted_output={"max_abs_delta": 0.0},
        empirical_output={
            "max_abs_delta": CPU_HOSC_MAX_ABS_DELTA,
            "verdict": "REFUSE (fp-contraction 1 ULP => cpu device never admits the compiled hosc)",
        },
        residual=CPU_HOSC_MAX_ABS_DELTA,
        source_artifact=".omx/research/mlx_safe_compile_v2_finish_20260708.md",
        measurement_method="mlx_safe_compile_v2_cpu_cert_bit_equality_harness",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/mlx_safe_compile_v2_finish_20260708.md",
            reactivation_criteria="re-run the CPU cert; a future mlx version may change the "
                                  "contraction behavior (fingerprint fails closed on version bump)",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=SAFE_COMPILE_DEVICE_BITIDENTITY_EQUATION_ID,
        name="Safe-compile per-device bit-identity (hosc flip admissibility law)",
        one_line_summary=(
            "mx.compile bit-identity is a per-{chip,os,mlx,device} EMPIRICAL fact: hosc ADMITS on "
            "M5-Max-GPU (max|d|=0, N=5) yet REFUSES on the same chip's CPU (5.96e-8); resolve from "
            "the fingerprint manifest"
        ),
        latex_form=(
            r"\mathrm{admit}(r, f) \iff \max_{x \in \mathcal{C}} \left| \mathrm{compile}(r)(x) - "
            r"r(x) \right| = 0 \ \wedge\ \mathrm{fp}(f) = \mathrm{fp}_{\mathrm{cert}},\quad "
            r"f = (\mathrm{chip}, \mathrm{os}, \mathrm{mlx}, \mathrm{device})"
        ),
        python_callable_module_path="tac.mlx_safe_compile:resolve_enabled_regions",
        domain_of_validity={
            "regions": "fused ELEMENTWISE safe-compile regions (scatter-class ops are OUT — they "
                       "ride mlx_gpu_crossprocess_nondeterminism_v1 / #348 fixed-order)",
            "fingerprint": {"chip": "Apple M5 Max", "macos_build": "25E246", "mlx": "0.31.2"},
            "coverage": "finite adversarial sample at real witness scale (empirical cert, not a "
                        "symbolic proof); off-fingerprint fails closed and recertifies",
            "measurement_axis": [_MLX_SIGNAL, _ADVISORY],
        },
        units_in={"region": "callable", "fingerprint": "chip_os_mlx_device", "coverage": "inputs"},
        units_out={"admit": "boolean", "max_abs_delta": "dimensionless"},
        empirical_anchors=(anchor_gpu_admit, anchor_cpu_refuse),
        predicted_vs_empirical_residual={
            "m5_max_gpu": 0.0,
            "m5_max_cpu": CPU_HOSC_MAX_ABS_DELTA,
        },
        last_calibration_utc="2026-07-08T16:40:00Z",
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tools.launch_witness_run",           # admission step b2 (fingerprint gate)
            "tac.witness_autoconfig",             # v7.3 compiler consumes the ADMIT evidence
        ),
        canonical_producers=(
            "tac.mlx_safe_compile",               # the certification harness writes the manifest
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MODULE,
            reactivation_criteria="recertify on fingerprint change; extend anchors when new "
                                  "regions/devices are certified",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_gpu",
        ),
    )


def populate_safe_compile_device_bitidentity_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins query semantics)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_safe_compile_hosc_device_bitidentity_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="crux_engineering_gpu_bitcert_20260708",
    )
    return eq


__all__ = [
    "SAFE_COMPILE_DEVICE_BITIDENTITY_EQUATION_ID",
    "build_safe_compile_hosc_device_bitidentity_v1",
    "populate_safe_compile_device_bitidentity_equation",
]
