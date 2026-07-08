# SPDX-License-Identifier: MIT
"""Canonical equation: frozen-scorer forward BATCH-DEPENDENCE (the micro-batch bit-identity wall).

Crux-engineering falsification 2026-07-08 (commits 1e2978251/0b8b1954c, memo
.omx/research/microbatch_bit_identity_crux_20260708.md): the premise "fixed-order reduction makes
micro-batch B>1 bit-identical to serial" is FALSIFIED at the MECHANISM level — the dominant
divergence enters at the FROZEN-SCORER FORWARD, upstream of any reduction the trainer controls.

THE LAW. For the frozen SegNet/PoseNet forward under MLX, per-sample outputs are BATCH-SIZE-
DEPENDENT: segnet(batch)[k] != segnet(batch[k:k+1])[0]. Measured (real upstream adapter):
GPU logit drift 2.26e-2 max|delta| with 11/196608 argmax px flipped and pose delta 7.7e-3;
CPU drift 7.1e-5 with ZERO argmax flips (argmax-INVARIANT) and pose 2.0e-6. A conv/matmul
kernel-tiling property (sister of mlx_gpu_crossprocess_nondeterminism_v1 + the per-device
bit-identity law) — not a reduction-order the caller can fix. Secondary: the all-cotangents
batched backward reorders ~1e-3..4e-3 AND is itself run-to-run nondeterministic at ULP
boundaries, so it cannot be fixed-order matched either.

CONSEQUENCE: bit-identity <=> per-pair (batch-1) scorer forward <=> the serial path; surviving
speedup AT bit-identity = 1.0x. The micro-batch 2-4x lever stays A/B-gated (bounded n600 d_seg
A/B; GPU flips only 0.006% px so plausibly d_seg-neutral — MEASURE, never assume) unless
batch-invariant scorer Metal kernels are built (#252/#356 program).

verdict_scope: formulation — the MLX frozen-scorer forward on this fingerprint; NOT the
micro-batch paradigm (the A/B path remains open). means != ends; pointer 0.19110 UNMOVED;
all numbers [macOS-MLX research-signal]/[macOS-CPU advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

from tac.canonical_equations.equation import CanonicalEquation, EmpiricalAnchor
from tac.provenance.builders import build_provenance_for_research_sidecar

SCORER_BATCH_DEPENDENCE_EQUATION_ID = "frozen_scorer_forward_batch_dependence_v1"

_MLX_SIGNAL = "[macOS-MLX research-signal]"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/microbatch_bit_identity_crux_20260708.md"

GPU_LOGIT_DRIFT = 2.26e-2
GPU_ARGMAX_FLIPS = 11          # of 196608 px (0.006%)
GPU_POSE_DRIFT = 7.7e-3
CPU_LOGIT_DRIFT = 7.1e-5
CPU_ARGMAX_FLIPS = 0           # argmax-INVARIANT on cpu
CPU_POSE_DRIFT = 2.0e-6


def build_frozen_scorer_forward_batch_dependence_v1() -> CanonicalEquation:
    """Both anchors MEASURED (VERIFIED_VIA_EMPIRICAL_ANCHOR); the GPU/CPU split is the law."""
    a_gpu = EmpiricalAnchor(
        anchor_id="scorer_forward_batch_dependence_gpu_20260708",
        measurement_utc="2026-07-08T17:10:00Z",
        inputs={"op": "frozen SegNet/PoseNet forward, batched-vs-per-sample", "device": "gpu",
                "probe": "tools/micro_batch_bit_identity_probe.py (real upstream adapter)"},
        predicted_output={"batch_invariant": True},
        empirical_output={"max_logit_drift": GPU_LOGIT_DRIFT, "argmax_px_flipped": GPU_ARGMAX_FLIPS,
                          "argmax_px_total": 196608, "pose_drift": GPU_POSE_DRIFT,
                          "verdict": "NOT batch-invariant -> bit-identity-at-speedup impossible"},
        residual=GPU_LOGIT_DRIFT,
        source_artifact=_MEMO,
        measurement_method="micro_batch_bit_identity_probe_decomposition",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="re-run tools/micro_batch_bit_identity_probe.py; batch-invariant "
                                  "scorer Metal kernels (#252/#356) would retire this wall",
            measurement_axis=_MLX_SIGNAL, hardware_substrate="m5_max_gpu"),
    )
    a_cpu = EmpiricalAnchor(
        anchor_id="scorer_forward_batch_dependence_cpu_argmax_invariant_20260708",
        measurement_utc="2026-07-08T17:10:00Z",
        inputs={"op": "frozen SegNet/PoseNet forward, batched-vs-per-sample", "device": "cpu"},
        predicted_output={"batch_invariant": True},
        empirical_output={"max_logit_drift": CPU_LOGIT_DRIFT, "argmax_px_flipped": CPU_ARGMAX_FLIPS,
                          "pose_drift": CPU_POSE_DRIFT,
                          "verdict": "argmax-INVARIANT (0 flips) though not bit-identical"},
        residual=CPU_LOGIT_DRIFT,
        source_artifact=_MEMO,
        measurement_method="micro_batch_bit_identity_probe_decomposition",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="re-run the probe on cpu; the argmax-invariance licenses "
                                  "cpu-side batched evaluation at argmax level (advisory)",
            measurement_axis=_ADVISORY, hardware_substrate="m5_max_cpu"),
    )
    return CanonicalEquation(
        equation_id=SCORER_BATCH_DEPENDENCE_EQUATION_ID,
        name="Frozen-scorer forward batch-dependence (micro-batch bit-identity wall)",
        one_line_summary=(
            "segnet(batch)[k] != segnet(batch[k:k+1])[0]: GPU logit drift 2.26e-2 (11/196608 argmax "
            "flips), CPU argmax-INVARIANT — bit-identity-at-speedup impossible; A/B-gate stands"
        ),
        latex_form=(
            r"\max_k \| S(x_{1:K})_k - S(x_k) \|_\infty = 2.26\cdot 10^{-2}\ (\mathrm{gpu});"
            r"\ 7.1\cdot 10^{-5}\ (\mathrm{cpu,\ argmax\ invariant})"
        ),
        python_callable_module_path=(
            "tac.boundary_math.micro_batch_bit_identity_probe:classify_micro_batch_bit_identity"
        ),
        domain_of_validity={
            "scorers": "frozen SegNet/PoseNet MLX forward on fingerprint {M5 Max, mlx 0.31.2}",
            "scope": "verdict_scope: formulation — this scorer path on this fingerprint; the "
                     "micro-batch PARADIGM stays open via the bounded n600 d_seg A/B",
            "measurement_axis": [_MLX_SIGNAL, _ADVISORY],
        },
        units_in={"batch": "pairs", "device": "gpu|cpu"},
        units_out={"max_logit_drift": "dimensionless", "argmax_px_flipped": "pixels"},
        empirical_anchors=(a_gpu, a_cpu),
        predicted_vs_empirical_residual={"m5_max_gpu": GPU_LOGIT_DRIFT, "m5_max_cpu": CPU_LOGIT_DRIFT},
        last_calibration_utc="2026-07-08T17:10:00Z",
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=("tac.boundary_math.levelset_micro_batch_loss", "tac.witness_autoconfig"),
        canonical_producers=("tac.boundary_math.micro_batch_bit_identity_probe",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="batch-invariant scorer kernels (#252/#356) or new fingerprint",
            measurement_axis=_MLX_SIGNAL, hardware_substrate="m5_max_gpu"),
    )


def populate_frozen_scorer_forward_batch_dependence_equation(*, path=None, lock_path=None,
                                                             agent=None, subagent_id=None):
    """Idempotent APPEND-ONLY registration."""
    from tac.canonical_equations.registry import register_canonical_equation
    eq = build_frozen_scorer_forward_batch_dependence_v1()
    register_canonical_equation(eq, path=path, lock_path=lock_path, agent=agent,
                                subagent_id=subagent_id,
                                notes="microbatch_bit_identity_crux_falsification_20260708")
    return eq


__all__ = [
    "SCORER_BATCH_DEPENDENCE_EQUATION_ID",
    "build_frozen_scorer_forward_batch_dependence_v1",
    "populate_frozen_scorer_forward_batch_dependence_equation",
]
