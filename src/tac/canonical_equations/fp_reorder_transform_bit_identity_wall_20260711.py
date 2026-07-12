# SPDX-License-Identifier: MIT
"""fp-reorder-transform bit-identity WALL (#356 whole-step-megakernel verdict, 2026-07-11).

THE LAW (three MEASURED anchors, ONE mechanism)
-----------------------------------------------
On the level-set witness graph at fp32, ANY framework-level transformation that permits
floating-point reorder/contraction — ``mx.compile`` graph fusion (mul+add -> fma,
reassociated reductions) or batched-axis reductions (``--micro-batch-pairs``) — produces
numerics that are DETERMINISTIC-BUT-DIFFERENT from the untransformed path (re-run of the
transformed graph is byte-stable at 0.0, but transformed-vs-untransformed deltas run
1e-7 .. 1e-3). Such a transform is not an A/B-neutral or bit-identical speed lever. The
historical #410 bit-identity NO-GO remains true, but operator 2026-07-12 WAIVES bit identity
for TRAINING ONLY: a micro-batch transform may be adopted for training after functional
per-pair loss/gradient parity and measured speedup. Score/byte-close authority is untouched.

The SURVIVING speed family is exactly the complement: transforms with EXPLICIT op order
or gradient-free constant caching — the fused-R Metal kernel (fixed-order VJP; L70
cross-process 0/28), the grouped-conv backward (~17x), and ``--cache-gt-skeleton``
(epoch-invariant gradient-free constant; exactly bit-identical, n64 A/B).  Both classes
are MEASURED; the wall is the boundary between them.

Corollary (measured today): the payoff of the failed class is small anyway — compiling
the whole trunk+loss closure buys only 1.12-1.21x on GPU (the step is compute-bound in
the scorer convs; only elementwise glue fuses) and is SLOWER on CPU (0.79-0.83x).  The
mx.compile whole-step megakernel loses on BOTH legs independently.

Scope: FORMULATION-level for mx.compile-on-MLX/Metal at fp32 on THIS graph class; the
CUDA port (#438) has different contraction semantics and must re-measure, and per-chain
explicit-order custom Metal kernels (#252) remain OPEN.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "witness_fp_reorder_transform_bit_identity_wall_v1"

_UTC = "2026-07-11T00:00:00Z"
_ADVISORY = "[macOS-CPU/GPU advisory]"
_MEMO = ".omx/research/whole_step_megakernel_356_20260711.md"
_MICROBATCH_MEMO = ".omx/research/microbatch_bit_identity_smoke_n600_20260710.md"
_SMOKE = "experiments/megakernel_compile_bit_identity_smoke_356.py"

# MEASURED 2026-07-11 (this smoke; M5 Max, MLX, fp32; representative trunk+loss closure).
_COMPILE_GRAD_DELTA = {
    ("cpu", 4096): 2.267e-05,
    ("cpu", 196608): 2.310e-07,
    ("gpu", 4096): 1.302e-05,
    ("gpu", 196608): 1.997e-06,
}
_COMPILE_SPEEDUP = {
    ("cpu", 4096): 0.788,
    ("cpu", 196608): 0.831,
    ("gpu", 4096): 1.117,
    ("gpu", 196608): 1.210,
}


def build_witness_fp_reorder_transform_bit_identity_wall_v1() -> CanonicalEquation:
    """Build the fp-reorder bit-identity wall with its three MEASURED anchors.

    Anchor 1 = mx.compile of the whole trunk+loss closure (#356 smoke, 2026-07-11):
    grad deltas 2.3e-7..2.3e-5, deterministic 0.0, speedup GPU 1.12-1.21x / CPU SLOWER.
    Anchor 2 = --micro-batch-pairs batched reduction order (#410, 2026-07-10):
    CPU grad rel <=5.9e-5, GPU ~1e-3 (matmul tiling), batched re-run 0.0.
    Anchor 3 = mx.compile of the R op (2026-07-03, in-code fail-closed gate):
    fma contraction up to ~4.8e-3 across the uint8 round boundary -> flips d_seg argmax."""

    anchor_compile_closure = EmpiricalAnchor(
        anchor_id="mx_compile_whole_closure_not_bit_identical_marginal_speed_measured_20260711",
        measurement_utc=_UTC,
        inputs={
            "transform": "mx.compile of the whole trunk+loss value_and_grad closure (the #356 megakernel candidate)",
            "graph": (
                "representative witness d_seg closure (tac.local_acceleration.mlx_compile_step."
                "build_representative_dseg_trunk: Linear/FiLM/relu/softmax/palette-matmul/sigmoid "
                "+ CE + finite-diff grid term; op-kind superset direction: the REAL closure adds "
                "MORE elementwise chains, i.e. MORE contraction sites, never fewer)"
            ),
            "grid": "device {cpu,gpu} x P {4096 launch-overhead regime, 196608 full-frame 384x512 regime}",
            "script": _SMOKE,
        },
        predicted_output={
            "hoped": "order-preserving fusion => 0.0 delta + large launch-overhead win",
            "law": "compile fusion reorders fp (fma/reassociation) => delta > 0.0; step is compute-bound => win small",
        },
        empirical_output={
            "grad_max_abs_delta_by_device_P": {f"{d}_{p}": v for (d, p), v in _COMPILE_GRAD_DELTA.items()},
            "loss_delta_smallP": "2.1e-6 (gpu) / 4.8e-7 (cpu); 0.0 at fullframe (mean absorbs, grads do not)",
            "compiled_determinism_delta": 0.0,
            "speedup_by_device_P": {f"{d}_{p}": v for (d, p), v in _COMPILE_SPEEDUP.items()},
            "verdict": (
                "NOT bit-identical anywhere (deterministic-but-DIFFERENT) AND speed marginal "
                "(GPU 1.12-1.21x on the closure ~ approx 5% end-to-end; CPU SLOWER) -> the "
                "mx.compile whole-step megakernel is NO-GO on BOTH legs independently"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="eager_vs_compiled_exact_delta_fwd_bwd_fp32_plus_walltime_bench_cpu_gpu",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "an MLX compile mode that provably preserves fp order (no-fma / strict mode), OR "
                "the CUDA port (#438) measuring 0.0 under torch.compile/inductor strict numerics "
                "on contest hardware, would RE-OPEN whole-step fusion for that backend"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_micro_batch = EmpiricalAnchor(
        anchor_id="micro_batch_pairs_reduction_order_not_bit_identical_measured_20260710",
        measurement_utc="2026-07-10T00:00:00Z",
        inputs={
            "transform": "--micro-batch-pairs batched twin (stacked (B,...) axis mx.mean vs serial per-pair accum)",
            "scale": "n6 and n600 (identical verdict; boolean config guard has no toy-vs-scale gap)",
        },
        predicted_output={
            "law": "batched reduction ORDER differs from serial order => fp-equivalent, NOT bit-identical",
        },
        empirical_output={
            "cpu_loss_rel": 9.4e-08,
            "cpu_per_group_grad_rel_max": 5.9e-05,
            "gpu_batched_fp_noise": "~1e-3 (GPU matmul kernel tiling, pose path)",
            "batched_rerun_delta": 0.0,
            "verdict": "HISTORICAL #410 NO-GO for a bit-identical pointer relaunch",
            "current_training_policy": (
                "operator 2026-07-12 training-only waiver: functional per-pair parity plus measured "
                "speedup admits micro-batching; chroma is now routed"
            ),
        },
        residual=0.0,
        source_artifact=_MICROBATCH_MEMO,
        measurement_method="serial_vs_batched_twin_exact_delta_n6_n600",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MICROBATCH_MEMO,
            reactivation_criteria=(
                "a fixed-order batched reduction (int64 fixed-point or fixed-order tree, #348 family) "
                "measuring 0.0 vs serial at n600 would re-open micro-batch for the pointer lineage"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_r_compile = EmpiricalAnchor(
        anchor_id="mx_compile_r_op_fma_contraction_flips_uint8_ste_argmax_measured_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={
            "transform": "mx.compile of the reference R operator (bicubic-up -> uint8-STE -> bilinear-down)",
            "site": (
                "experiments/train_witness_realized_through_R_mlx.py set_mx_compile_r "
                "(the fail-closed --mx-compile gate documents the measurement in-code)"
            ),
        },
        predicted_output={
            "law": "fma contraction at the uint8 round knife-edge => argmax flips (score-RELEVANT, not just trajectory)",
        },
        empirical_output={
            "contraction_delta": "up to ~4.8e-3 across the uint8 round boundary",
            "consequence": "flips d_seg argmax pixels -> --mx-compile REFUSED fail-closed; --fused-r-kernel is the cure",
            "cure": "fused-R Metal kernel: contraction-off, bit-identical fwd, fixed-order VJP (L70 cross-process 0/28)",
        },
        residual=0.0,
        source_artifact="experiments/train_witness_realized_through_R_mlx.py",
        measurement_method="compiled_vs_reference_r_delta_at_uint8_round_boundary",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "same as the closure anchor: a provably order-preserving compile mode re-opens the R op"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "fp-reorder-transform bit-identity wall: framework transforms that permit fp "
            "reorder/contraction remain deterministic-but-DIFFERENT (delta 1e-7..1e-3). "
            "Bit-identical authority still requires explicit order; training-only micro-batching "
            "may instead pass the functional-parity and measured-speed admission gate"
        ),
        one_line_summary=(
            "Three anchors preserve the fp-reorder wall; a training-only operator waiver now "
            "admits micro-batching by functional parity plus measured speed, never by score authority."
        ),
        latex_form=(
            r"T \in \{\text{compile-fuse},\ \text{batch-reduce}\} \Rightarrow"
            r"\ \exists\,\theta,x:\ \nabla_T \neq \nabla_{\mathrm{eager}}\ (\Delta \in [10^{-7},10^{-3}])"
            r";\quad T \in \{\text{fixed-order kernel},\ \text{const cache}\} \Rightarrow \Delta = 0"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.mlx_compile_step:assert_compile_bit_identical"
        ),
        domain_of_validity={
            "vehicle": ["level_set_witness_v7x_v9_cgauge"],
            "backend": "MLX (Metal/CPU) fp32; CUDA (#438 port) has DIFFERENT contraction semantics -> re-measure there",
            "verdict_scope": (
                "FORMULATION -- mx.compile whole-step fusion on the witness trunk+loss graph class "
                "at fp32. PARADIGM INTACT: per-chain explicit-order custom Metal kernels (#252) and "
                "fixed-order batched reductions (#348 family) remain OPEN speed paths."
            ),
            "scope": (
                "closure anchor measured on the REPRESENTATIVE trunk (op-kind subset of the real "
                "closure; more terms = more contraction sites, so the FAIL direction transfers); "
                "the R-op anchor is real-graph. Advisory, non-promotable; no score claim."
            ),
            "measurement_axis": ["macOS-CPU advisory", "macOS-GPU advisory"],
            "promotion_eligible": False,
            "training_only_override_20260712": (
                "bit identity waived; require per-pair functional loss/gradient parity and measured speedup"
            ),
            "score_authority": "none; exact byte-closed n600 validation remains owed",
            "note": (
                "means != ends: a speed-lever boundary law. Positive guidance: to make the witness "
                "faster, write explicit-order kernels for measured-hot chains or cache gradient-free "
                "constants; never hand the framework freedom to reorder fp on a score-faithful lineage."
            ),
        },
        units_in={"transform": "graph_transformation_class", "closure": "witness_trunk_plus_loss_fp32"},
        units_out={"max_abs_delta": "fp32_abs_delta_eager_vs_transformed", "speedup": "walltime_ratio"},
        empirical_anchors=(anchor_compile_closure, anchor_micro_batch, anchor_r_compile),
        predicted_vs_empirical_residual={
            "compile_closure_measured": 0.0,
            "micro_batch_measured": 0.0,
            "r_op_contraction_measured": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            ".omx/research/whole_step_megakernel_356_20260711.md",  # the #356 verdict memo
            "v9_cgauge_432_speed_lever_selection",                   # which speed levers may ride the arm
            "cuda_port_438_compile_policy",                          # #438 must re-measure per-backend
        ),
        canonical_producers=(
            "tac.local_acceleration.mlx_compile_step",       # compile wrapper + bit-identity assert
            "tac.local_acceleration.metal_fused_r_operator",  # the surviving explicit-order exemplar
            "tac.mlx_safe_compile",                           # certify-then-compile region machinery
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "an order-preserving compile mode (MLX strict/no-fma) measuring 0.0 on the closure; "
                "OR a fixed-order batched reduction (#348) measuring 0.0 at n600; OR the CUDA "
                "backend measuring 0.0 under its compiler on contest hardware"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_witness_fp_reorder_transform_bit_identity_wall_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the fp-reorder bit-identity wall (latest-row-wins).
    EQUATIONS leg of FEED-356; DSL leg = N/A-with-reason (NO lever ships — shipping a
    numerics-changing 'speedup' flag would be the NO-FAKE; the verdict is the deliverable)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_witness_fp_reorder_transform_bit_identity_wall_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="fp_reorder_transform_bit_identity_wall_20260711 (#356 whole-step-megakernel MEASURED "
              "NO-GO on both legs; 3 anchors one mechanism; surviving family = explicit-order "
              "kernels + const caches, both already ON in the #432 argv; advisory NON-PROMOTABLE)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_witness_fp_reorder_transform_bit_identity_wall_v1",
    "populate_witness_fp_reorder_transform_bit_identity_wall_equation",
]
