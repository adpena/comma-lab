# SPDX-License-Identifier: MIT
"""C1 S_R throughput leg — RECONCILED surface (#507, 2026-07-15).

HISTORY + honest state: the original commit of this module (the isolated
c1_throughput worktree's compiler) imported ten Lever factories
(``SafeCompileReference`` / ``SerialPairRouting`` / ``FusedRKernel`` /
``CacheGtSkeleton`` / ``GroupedBackwardReference`` / ``PersistencePoolKernel`` /
``FrozenScorerOneThread`` / ``AsyncVerdictOffload`` / ``VerdictChunking`` /
``ComponentWallclockTelemetry``) plus ``runtime_environment`` fields whose
curriculum_dsl/typed_config hot-file edits were NEVER harvested to main — the
module was import-broken from the moment it landed. The #507 composition task
owned the reconciliation; this rewrite is it.

Two supersessions applied:

1. **Import repair by mapping, not re-invention** — every unharvested factory is
   mapped in :data:`HISTORICAL_FACTORY_RECONCILIATION` onto the surface that
   ALREADY realizes it on main (parent-carried flags, ``PERF_ENV_PREFIX`` env
   carriers, the trainer-native 1-thread law, or the folded telemetry lever in
   ``spec_c1_optimal_form_20260715``). Nothing was silently dropped.
2. **Strict-identity split superseded** — operator 2026-07-15: gradient quality
   (functional parity), not bit identity, is the speed-lever admission bar, and
   the objective is joint wall-clock-to-target. The strict-identity dispositions
   are preserved verbatim in :data:`EXCLUDED_OR_HELD` as the historical record;
   the LIVE re-adjudication lives in
   ``spec_c1_optimal_form_20260715.C1_SPEED_DISPOSITIONS``.

The compile entry points now DELEGATE to the composed #507 factory
(``compile_c1_throughput_launch_config``) and to the official C1a S_R factory
(``compile_c1_sr_parent_launch_config``) — the exact reconciliation the
throughput leg's own audit deferred ("reconcile this compiler with C1a's
official S_R factory"). CONTAINMENT: pure compiles; LAUNCH = operator-GO.
"""

from __future__ import annotations

from pathlib import Path

PROGRAM_NAME = "v9_cgauge_ideal_mod19_sR_c1_throughput"
DEFAULT_OUT_DIR = "experiments/results/v9_cgauge_ideal_mod19_sR_c1_throughput_20260715"
DEFAULT_BENCH_RECEIPT = Path(
    ".omx/research/c1_throughput_composed_bench_20260715.json"
)
BENCH_SCHEMA = "c1_throughput_composed_bench.v1"
C1A_COMMIT = "bdbbf5da175a46c11393ebbe56f53653828fb765"

# The original module's ten factory names -> the main-tree surface that realizes
# each intent today (the #507 reconciliation of the unharvested worktree edits).
HISTORICAL_FACTORY_RECONCILIATION: dict[str, str] = {
    "SafeCompileReference": "SUPERSEDED: strict 'none' pin replaced by the parent-carried "
                            "--safe-compile-regions hosc_activation under relaxed identity "
                            "(curriculum_dsl.SafeCompileRegions is the live owner)",
    "SerialPairRouting": "parent-carried: --micro-batch-pairs 1 pinned in the C1a base "
                         "(trainer fail-closes S_R at micro-batch>1)",
    "FusedRKernel": "parent-carried: curriculum_dsl.FusedRKernel already composed in the "
                    "ideal V9 lineage (--fused-r-kernel emitted)",
    "CacheGtSkeleton": "parent-carried: curriculum_dsl.CacheGtSkeleton (--cache-gt-skeleton "
                       "emitted)",
    "GroupedBackwardReference": "SUPERSEDED: the strict-identity reference downgrade "
                                "(env=0) is refuted under operator 2026-07-15; the ~17x "
                                "custom VJP stays ON via PERF_ENV_PREFIX "
                                "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1",
    "PersistencePoolKernel": "carried by PERF_ENV_PREFIX TAC_MLX_CUSTOM_PERSISTENCE_POOL=1 "
                             "(tac/witness_dsl/typed_config.py PERF_ENV_PREFIX is the SoT)",
    "FrozenScorerOneThread": "trainer-native: torch.set_num_threads(SELECTED_THREADS) from "
                             "canonical_equations.segnet_exact_forward_cpu_thread_law_20260713 "
                             "at trainer startup; the --training-torch-threads flag never "
                             "landed and is unnecessary",
    "AsyncVerdictOffload": "parent-carried: --async-verdict emitted by the C1a base",
    "VerdictChunking": "parent-carried: --verdict-batch 32 --verdict-pairs 0 emitted by the "
                       "C1a base",
    "ComponentWallclockTelemetry": "folded: spec_c1_optimal_form_20260715 lever "
                                   "c1_component_wallclock_telemetry "
                                   "(--component-wallclock-telemetry / probe-every 1 / "
                                   "--profile-timing)",
}

# HISTORICAL strict-identity dispositions (verbatim from the original module; the
# 2026-07-15 live re-adjudication is spec_c1_optimal_form_20260715.C1_SPEED_DISPOSITIONS).
EXCLUDED_OR_HELD: dict[str, dict[str, str]] = {
    "whole_step_megakernel_356": {
        "status": "EXCLUDED_MEASURED_FORMULATION_NO_GO",
        "reason": "fp-reorder grad delta 2.3e-7..2.3e-5; CPU 0.79-0.83x; GPU 1.12-1.21x only",
        "equation_id": "witness_fp_reorder_transform_bit_identity_wall_v1",
    },
    "custom_grouped_backward": {
        "status": "EXCLUDED_STRICT_IDENTITY_CONFLICT",
        "reason": "17.96x backward/5.5x n8 e2e, but primary proof is cosine plus fp32 roundoff, not bit identity",
        "verdict_scope": "current custom Metal grouped/depthwise VJP formulation only",
        "superseded_20260715": "REFUTED under the relaxed-identity directive; ON via PERF_ENV "
                               "(see spec_c1_optimal_form_20260715.C1_SPEED_DISPOSITIONS)",
    },
    "micro_batch_pairs_gt1": {
        "status": "EXCLUDED_SR_AND_IDENTITY_CONFLICT",
        "reason": "S_R has no batched consumer; B2 scorer flips and faithful n24 full-step speed was about 1.001x",
        "verdict_scope": "current batched scorer/reduction formulation on C1",
        "superseded_20260715": "identity half relaxed; the S_R half remains CODE-BLOCKED "
                               "(trainer fail-close) — the named fallen-crack",
    },
    "safe_compile_hosc_activation": {
        "status": "HELD_HOST_CERTIFICATE_ABSENT",
        "reason": "per-chip safe-compile manifest is absent in this worktree; never transfer a certificate",
        "superseded_20260715": "the isolated worktree lacked the certificate; the main-tree "
                               "parent composes hosc_activation via its own custody",
    },
    "fresh_frequency_shift_init": {
        "status": "HELD_UNMEASURED_CONVERGENCE",
        "reason": "wired and tested, but no real GPU epochs-to-target or no-regression receipt",
    },
    "ane_forward": {
        "status": "EXCLUDED_TRAINING_PATH_UNAVAILABLE",
        "reason": "CoreML/ANE forward-only; no differentiable VJP/training placement proof",
        "verdict_scope": "public CoreML training formulation, not the ANE family",
    },
    "pose_verdict_gate": {
        "status": "RETIRED_UNBOUND_CACHE",
        "reason": "no payload-bound pose cache; live PoseNet remains required",
        "verdict_scope": "banked pose substitution formulation",
        "superseded_20260715": "the COMPUTE-ONLY PoseBlindComputeGate (no banked d_pose; live "
                               "PoseNet at every eligible verdict) is a DIFFERENT mechanism "
                               "and is folded in spec_c1_optimal_form_20260715",
    },
    "integer_r_adjoint_348_followon": {
        "status": "HELD_HOST_N600_RECEIPT_OWED",
        "reason": "order-independent Q15/int32 backend has no admitted host n600 parity/determinism/speed receipt",
    },
    "costate_reuse_trust_region_454": {
        "status": "EXCLUDED_MEASURED_ECONOMICS_NO_GO",
        "reason": "validation economics require more than one exact validation per step in the tested region",
        "verdict_scope": "current HVP/Jacobian-drift certification formulation",
    },
    "costate_forward_kill_455_456_465": {
        "status": "EXCLUDED_SCOPED_NO_GO_FAMILIES_OPEN",
        "reason": "tested reuse/linear/student/forward substitutions did not clear fidelity plus economics",
        "verdict_scope": "measured formulations only; nonlinear distilled teacher family remains open",
    },
    "sparse_grouped_backward_486_487_488": {
        "status": "EXCLUDED_NO_ADMITTED_MASK_OR_WALL_WIN",
        "reason": "sparse-adjoint receipt did not realize its ceiling; exact K2 provider remains separate",
        "verdict_scope": "current sparse masks/kernel formulation",
    },
    "transient_task_495": {
        "status": "HELD_IDENTITY_BLOCKED",
        "reason": "no exact canonical task/source object resolves the transient #495 label",
        "verdict_scope": "task identity only; no technical conclusion",
    },
}


def compile_c1_sr_parent_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/v9_cgauge_ideal_mod19_sR_20260715",
):
    """The C1a S_R parent — now a pure delegation to the OFFICIAL committed factory.

    The original compatibility-reconstruction path existed only for the isolated
    worktree; on main the official factory is present and is the single owner.
    """
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_ideal_mod19_sR_launch_config,
    )

    return compile_v9_cgauge_ideal_mod19_sR_launch_config(
        gt_cache_path=gt_cache_path, num_pairs=num_pairs, epochs=epochs,
        out_dir=out_dir)


def compile_c1_throughput_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = DEFAULT_OUT_DIR,
):
    """SUPERSEDED strict-identity compiler — delegates to the #507 composed config.

    Returns ``spec_c1_optimal_form_20260715.compile_c1_optimal_form_launch_config``
    with a supersession record stamped into the manifest, so any caller of the
    historical entry point receives the reconciled composition instead of an
    ImportError (the original module's factories were never harvested to main).
    """
    from tac.witness_dsl.spec_c1_optimal_form_20260715 import (
        compile_c1_optimal_form_launch_config,
    )

    wrapped = compile_c1_optimal_form_launch_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    manifest = dict(wrapped.dsl_program_manifest)
    manifest["superseded_strict_identity_variant"] = {
        "historical_program_name": PROGRAM_NAME,
        "historical_bench_receipt": str(DEFAULT_BENCH_RECEIPT),
        "historical_bench_schema": BENCH_SCHEMA,
        "why": "operator 2026-07-15 relaxed-identity + joint wall-clock directive; the "
               "strict-identity speed split (env=0 reference kernels, safe-compile none) "
               "is superseded — dispositions preserved in EXCLUDED_OR_HELD",
        "factory_reconciliation": dict(HISTORICAL_FACTORY_RECONCILIATION),
    }
    from tac.witness_autoconfig import CrucibleV7LaunchConfig

    return CrucibleV7LaunchConfig(
        typed=wrapped.typed,
        constants_manifest=dict(wrapped.constants_manifest),
        dsl_program_manifest=manifest,
        schedule_governance=dict(wrapped.schedule_governance),
    )


__all__ = [
    "BENCH_SCHEMA",
    "C1A_COMMIT",
    "DEFAULT_BENCH_RECEIPT",
    "DEFAULT_OUT_DIR",
    "EXCLUDED_OR_HELD",
    "HISTORICAL_FACTORY_RECONCILIATION",
    "PROGRAM_NAME",
    "compile_c1_sr_parent_launch_config",
    "compile_c1_throughput_launch_config",
]
