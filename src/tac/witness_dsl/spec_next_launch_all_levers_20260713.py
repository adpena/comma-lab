"""Typed composition for the 2026-07-13 all-compatible-levers witness ticket.

This module is configuration-only.  It does not launch, mutate the trainer, or
claim that missing telemetry exists.  The compiler starts from the sealed V9
CGauge ideal-mod19 lineage, removes the one proven micro-batch incompatibility,
and composes the compatible speed/init/observer levers through the typed DSL.

The returned launch adapter carries explicit ``launch_blockers`` in its DSL
manifest.  The governed launcher consumes those blockers before any real spawn;
``--dry-run`` remains available so the complete zero-dollar preflight chain can
materialize and inspect the held ticket.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

from tac.witness_autoconfig import CrucibleV7LaunchConfig, _crucible_v7_argv_pairs
from tac.witness_dsl.curriculum_dsl import (
    CacheGtSkeleton,
    FreshFrequencyShift,
    FusedRKernel,
    Lever,
    MicroBatch,
    SafeCompileRegions,
)
from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_mod19_launch_config
from tac.witness_dsl.typed_config import TypedLever, build_launch_manifest

PROGRAM_NAME = "next_launch_all_levers_20260713"
DEFAULT_OUT_DIR = "experiments/results/next_launch_all_levers_ticket_20260713"
TRIMMED_PROGRAM_NAME = "next_launch_all_levers_trimmed_20260713"
TRIMMED_OUT_DIR = "experiments/results/next_launch_all_levers_ticket_trimmed_20260713"
DEFAULT_RUNTIME_OUT_DIR = "/Volumes/VertigoDataTier/pact/next_launch_all_levers_20260713"
FULL_VARIANT = "full"
TRIMMED_COMPLIANT_VARIANT = "trimmed_compliant"
MEMORY_VARIANTS = (FULL_VARIANT, TRIMMED_COMPLIANT_VARIANT)

# The exact schemas the requested debts require.  These names are dependency
# slots, not claims that the producer exists.
COMPONENT_WALLCLOCK_SCHEMA = "witness_component_wallclock.v1"
SPS_ENGAGEMENT_SCHEMA = "sps_gradient_role_conflict_engagement.v1"
CAUSAL_MANIFEST_DEPENDENCY = "pact.causal_manifest.v1"

REQUIRED_COMPONENT_FIELDS: tuple[str, ...] = (
    "teacher_forward_s",
    "teacher_backward_s",
    "witness_forward_s",
    "witness_backward_s",
    "realized_R_s",
    "verdict_s",
    "checkpoint_io_s",
    "epoch_total_s",
)

EXCLUDED_LEVERS: dict[str, str] = {
    "tie_locus_displacement": (
        "PROVEN INCOMPATIBLE: emits --seg-subpix-boundary-weight>0, which the real trainer "
        "refuses with --micro-batch-pairs>1 because the batched twin does not consume the term"
    ),
    "whole_step_megakernel": (
        "MEASURED NO-GO for the whole-step fp-reorder formulation; only per-chip, fingerprint-"
        "certified safe regions are admitted"
    ),
    "hardness_oversample": (
        "WIRING_NEEDED: current loop truncates the enlarged order to P visits, so the intended "
        "extra weighted visits are not all consumed"
    ),
    "horizon_weighted_margin": (
        "ISOLATE: exact V9 treatment support is zero and the weight lacks matched run custody"
    ),
    "step_native_activation": (
        "ISOLATE: representation/activation-basin treatment; stacking would confound the FreSh "
        "cold-start treatment"
    ),
    "film_polar_chart_spel_muon_finisher": (
        "EXCLUDED THIS RUN: sibling Muon round-2 finisher remains default-OFF pending its own "
        "governed micro-A/B and operator GO"
    ),
    "fresh_fixed_quality_slice": (
        "EXCLUDED FROM N600 PRODUCTION WINDOW: its per-epoch verdict/checkpoint cadence is the "
        "bounded n8/n64 matched-slice protocol, not this long run"
    ),
}


def excluded_levers_for_variant(variant: str) -> dict[str, str]:
    """Return the explicit, provenance-bearing exclusions for one ticket variant."""

    if variant not in MEMORY_VARIANTS:
        raise ValueError(f"unknown memory variant {variant!r}; expected one of {MEMORY_VARIANTS}")
    excluded = dict(EXCLUDED_LEVERS)
    if variant == TRIMMED_COMPLIANT_VARIANT:
        excluded["fresh_frequency_shift_init"] = (
            "MEMORY TRIM: FreSh re-enables --self-orient with n_dir_freqs=4 over the GO'd "
            "V9 self-orient-OFF parent, creating the measured approximately 47 GiB per-pair "
            "feature-cache tax. The underlying directional transfer measured approximately zero "
            "at the owed-16 n600 receiver surface; FreSh's cold-start treatment delta remains "
            "UNMEASURED, so this exclusion is not claimed score-neutral."
        )
    return excluded


def _typed(lever: Lever) -> TypedLever:
    """Lossless curriculum-DSL Lever -> typed-config adapter."""

    return TypedLever(
        name=lever.name,
        overrides=dict(lever.overrides),
        epochs_delta=int(lever.epochs_delta),
        notes=str(lever.notes),
    )


def _observer_telemetry() -> TypedLever:
    """Compose only real, already-parser-backed score-neutral telemetry.

    ``profile_timing`` is deliberately labeled coarse: it does not pay D-A by
    itself.  ``grad_interaction`` is the existing generic term matrix at stage
    boundaries: it does not pay the exact screw/phase engagement-trigger debt by
    itself.  The missing exact producers remain fail-closed manifest blockers.
    """

    return _typed(
        Lever(
            "next_launch_observer_telemetry",
            overrides={
                "--profile-timing": True,
                "--loss-term-log-every": 0,
                "--grad-interaction-telemetry": True,
                "--grad-interaction-k-pairs": 4,
                "--grad-interaction-every": 0,
            },
            notes=(
                "read-only/default-on-for-ticket observers: coarse epoch/step/verdict/R timing; "
                "standing per-epoch loss-term rows; generic K=4 gradient-cosine matrix at stage "
                "boundaries. Exact D-A component split and exact D-B engagement callbacks are "
                "separate launch-blocking dependencies"
            ),
        )
    )


def _dependency_slots(repo_root: Path) -> tuple[dict[str, dict], list[dict[str, str]]]:
    """Return surfaced dependency slots and the blockers that remain unresolved."""

    causal_module = repo_root / "src" / "tac" / "causal_manifest.py"
    causal_memo = repo_root / ".omx" / "research" / "causal_manifest_build_20260713.md"
    causal_dag_feed = repo_root / ".omx" / "research" / "causal_manifest_DAG_FEED_20260713.md"
    causal_module_landed = causal_module.is_file()
    causal_memo_landed = causal_memo.is_file()
    causal_dag_feed_landed = causal_dag_feed.is_file()
    trainer_path = repo_root / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
    trainer_text = trainer_path.read_text() if trainer_path.is_file() else ""
    try:
        trainer_tree = ast.parse(trainer_text) if trainer_text else None
    except SyntaxError:
        trainer_tree = None
    trainer_string_literals = {
        node.value
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    } if trainer_tree is not None else set()
    causal_trainer_default_on = (
        "CausalManifestWriter" in trainer_text
        and '"mode": "default_on_read_only"' in trainer_text
        and "CAUSAL_MANIFEST_FILENAME" in trainer_text
    )
    component_wallclock_producer_landed = (
        COMPONENT_WALLCLOCK_SCHEMA in trainer_string_literals
        and all(field in trainer_string_literals for field in REQUIRED_COMPONENT_FIELDS)
    )
    sps_engagement_producer_landed = (
        SPS_ENGAGEMENT_SCHEMA in trainer_string_literals
        and "temporal_screw_engaged" in trainer_string_literals
        and "phase_advection_engaged" in trainer_string_literals
    )
    causal_contract_landed = (
        causal_module_landed
        and causal_trainer_default_on
        and (causal_memo_landed or causal_dag_feed_landed)
    )
    causal_schema = CAUSAL_MANIFEST_DEPENDENCY
    if causal_module_landed:
        # Consume the sibling's real schema constant; never invent a parallel
        # launch-ticket schema.  Import is read-only and the module has no side
        # effects beyond definitions.
        from tac.causal_manifest import SCHEMA_ID as causal_schema
    runtime_out = Path(DEFAULT_RUNTIME_OUT_DIR)
    runtime_out_ready = runtime_out.is_dir() and runtime_out.exists()
    slots = {
        "D_A_component_wallclock": {
            "schema": COMPONENT_WALLCLOCK_SCHEMA,
            "required_fields": list(REQUIRED_COMPONENT_FIELDS),
            "status": (
                "landed_exact_in_run_producer"
                if component_wallclock_producer_landed
                else "missing_exact_in_run_producer"
            ),
            "producer_detection": (
                "schema id plus all eight exact field names must exist in the shared trainer"
            ),
            "existing_partial": (
                "--profile-timing emits fused step/verdict/overhead plus isolated-R microbench; "
                "it does not separate all required components"
            ),
        },
        "D_B_sps_engagement_conflict": {
            "schema": SPS_ENGAGEMENT_SCHEMA,
            "required_triggers": [
                "temporal_screw_engaged (event, fail-safe cap ep450)",
                "phase_advection_engaged (ep726 static terminal-band anchor)",
            ],
            "required_math": "gradient cosine/norm/conflict rule from tools/probe_sps_gradient_role_conflict.py",
            "status": (
                "landed_exact_engagement_callback"
                if sps_engagement_producer_landed
                else "missing_exact_engagement_callback"
            ),
            "producer_detection": (
                "schema id plus temporal_screw_engaged and phase_advection_engaged callbacks "
                "must exist in the shared trainer"
            ),
            "existing_partial": (
                "--grad-interaction-telemetry emits the generic term matrix at seg-form boundaries; "
                "screw/phase engagement is not a seg-form boundary"
            ),
        },
        "causal_manifest_transition_logging": {
            "schema": causal_schema,
            "module": str(causal_module.relative_to(repo_root)),
            "memo": str(causal_memo.relative_to(repo_root)),
            "dag_feed": str(causal_dag_feed.relative_to(repo_root)),
            "consumed_evidence": [
                str(path.relative_to(repo_root))
                for path in (causal_memo, causal_dag_feed)
                if path.is_file()
            ],
            "trainer_surface": (
                "default_on_read_only_no_launch_flag"
                if causal_trainer_default_on
                else "not_detected"
            ),
            "output_filename": "causal_manifest.jsonl",
            "status": (
                "landed_default_on_read_only"
                if causal_contract_landed
                else "trainer_default_on_memo_pending"
                if causal_trainer_default_on
                else "module_landed_memo_pending"
                if causal_module_landed
                else "pending_sibling_landing"
            ),
        },
        "memory_waterfill_micro_batch": {
            "required_value": 2,
            "canonical_solver_value": 1,
            "status": "n600_micro_batch_rss_unmeasured",
            "source": "tools/memory_waterfill_config.py on the compiled launch.sh",
            "detail": (
                "canonical waterfill excludes B>1: available points are n8 and contention-confounded; "
                "n600 RSS delta remains unmeasured"
            ),
        },
        "storage_waterfall": {
            "runtime_out_dir": DEFAULT_RUNTIME_OUT_DIR,
            "requested_bytes": 1_026_048_000,
            "requested_bytes_provenance": (
                "100x the MEASURED 10,260,480-byte v9_cgauge_432 run directory, to preserve "
                "stage/periodic checkpoints plus telemetry without local-disk spill"
            ),
            "status": "ready" if runtime_out_ready else "selected_workload_root_missing",
        },
    }
    blockers = []
    if not component_wallclock_producer_landed:
        blockers.append(
            {
                "id": "D_A_EXACT_COMPONENT_TIMERS_MISSING",
                "detail": "exact in-run component timing producer is absent; coarse --profile-timing is insufficient",
            }
        )
    if not sps_engagement_producer_landed:
        blockers.append(
            {
                "id": "D_B_EXACT_ENGAGEMENT_HOOK_MISSING",
                "detail": "no exact screw/phase engagement callback invokes the SPS gradient-conflict observer",
            }
        )
    blockers.append(
        {
            "id": "MEMORY_WATERFILL_B2_UNMEASURED_N600",
            "detail": (
                "canonical memory waterfill pins micro_batch=1 because B=2 has no target-n600 RSS "
                "measurement; do not override the $0 gate"
            ),
        }
    )
    if not runtime_out_ready:
        blockers.append(
            {
                "id": "SSD_WORKLOAD_ROOT_MISSING",
                "detail": (
                    f"storage waterfall selected {DEFAULT_RUNTIME_OUT_DIR}, but the workload root "
                    "does not exist; create it through the authorized storage preflight before GO"
                ),
            }
        )
    if not causal_contract_landed:
        blockers.append(
            {
                "id": (
                    "CAUSAL_MANIFEST_MEMO_PENDING"
                    if causal_module_landed
                    else "CAUSAL_MANIFEST_SCHEMA_PENDING"
                ),
                "detail": (
                    "causal-manifest module exposes pact.causal_manifest.v1 and the shared trainer "
                    "now writes causal_manifest.jsonl by a default-on read-only path with no new "
                    "launch flag, but the sibling memo has not landed; do not certify or guess the "
                    "unfinished dependency contract"
                    if causal_module_landed
                    else "causal-manifest sibling module+memo have not landed; dependency slot is named but not guessed"
                ),
            }
        )
    return slots, blockers


def compile_next_launch_all_levers_ticket(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = DEFAULT_OUT_DIR,
    variant: str = FULL_VARIANT,
) -> CrucibleV7LaunchConfig:
    """Compile the strongest currently compatible held launch ticket.

    Pure/$0: this function cannot spawn or train.  The governed launcher is the
    only intended consumer, and it refuses a real spawn while ``launch_blockers``
    is non-empty.
    """

    if variant not in MEMORY_VARIANTS:
        raise ValueError(f"unknown memory variant {variant!r}; expected one of {MEMORY_VARIANTS}")
    trimmed = variant == TRIMMED_COMPLIANT_VARIANT
    program_name = TRIMMED_PROGRAM_NAME if trimmed else PROGRAM_NAME
    resolved_out_dir = (
        TRIMMED_OUT_DIR
        if trimmed and out_dir == DEFAULT_OUT_DIR
        else out_dir
    )

    parent = compile_v9_cgauge_ideal_mod19_launch_config(
        gt_cache_path=gt_cache_path,
        num_pairs=num_pairs,
        epochs=epochs,
        out_dir=resolved_out_dir,
    )
    # MicroBatch(B=2) is the V9 functional-parity treatment.  Tie-locus is the
    # sole ideal-core lever the real batched trainer explicitly refuses.
    kept = tuple(lv for lv in parent.typed.levers if lv.name != "tie_locus_displacement")
    additions = (
        _typed(MicroBatch(2)),
        _typed(FusedRKernel()),
        _typed(CacheGtSkeleton()),
        _typed(SafeCompileRegions("hosc_activation")),
        *((_typed(FreshFrequencyShift()),) if not trimmed else ()),
        _observer_telemetry(),
    )
    purpose = (
        "HELD operator-GO-only n600 all-compatible-speed-stack"
        + (" with FreSh treatment; " if not trimmed else " with memory-compliant FreSh trim; ")
        + "measure D-A component wall split, D-B screw/phase engagement gradient conflict, "
        + "and causal transition rows; no launch until manifest blockers clear"
    )
    typed = parent.typed.model_copy(
        update={
            "name": program_name,
            "out_dir": str(resolved_out_dir),
            "purpose": purpose,
            "levers": kept + additions,
        }
    )
    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{program_name} DSL gate: {len(violations)} WitnessProgram.validate violation(s): "
            f"{violations[:6]}"
        )

    argv = tuple(typed.to_program().compile_trainer_argv())
    emitted = dict(_crucible_v7_argv_pairs(argv))
    required = {
        "--micro-batch-pairs": "2",
        "--safe-compile-regions": "hosc_activation",
        "--grad-interaction-k-pairs": "4",
        "--grad-interaction-every": "0",
    }
    mismatch = {flag: (emitted.get(flag), value) for flag, value in required.items()
                if emitted.get(flag) != value}
    required_boolean_flags = [
        "--fused-r-kernel",
        "--cache-gt-skeleton",
        "--profile-timing",
        "--grad-interaction-telemetry",
        "--stage-checkpoints",
        "--async-verdict",
    ]
    if not trimmed:
        required_boolean_flags.append("--fresh-init")
    for boolean_flag in required_boolean_flags:
        if boolean_flag not in emitted:
            mismatch[boolean_flag] = (emitted.get(boolean_flag), "present")
    if "--seg-subpix-boundary-weight" in emitted and float(
        emitted["--seg-subpix-boundary-weight"] or 0.0
    ) > 0.0:
        mismatch["--seg-subpix-boundary-weight"] = (
            emitted["--seg-subpix-boundary-weight"],
            "absent-or-zero under MicroBatch(B=2)",
        )
    if mismatch:
        raise ValueError(f"{program_name} actuation REFUSE: {mismatch}")

    repo_root = Path(__file__).resolve().parents[3]
    dependency_slots, blockers = _dependency_slots(repo_root)
    manifest = build_launch_manifest(
        program_name=program_name,
        emitted_flag_names=sorted(emitted),
        typed_config_hash=typed.typed_config_hash(),
        typed_validated=True,
    )
    manifest.update(
        {
            "expected_active_levers": [lv.name for lv in typed.levers],
            "excluded_levers": excluded_levers_for_variant(variant),
            "memory_variant": variant,
            "memory_trim": ({
                "trimmed_lever": "fresh_frequency_shift_init",
                "removed_flags": [
                    "--fresh-init", "--self-orient", "--n-dir-freqs", "--freq-across",
                    "--freq-along", "--fresh-spectrum-size", "--fresh-sample-pairs",
                    "--fresh-reference-freq-along", "--fresh-tangent-deficit",
                    "--fresh-bias-k-min", "--fresh-bias-k-max", "--fresh-bias-k-step",
                ],
                "authority": "owed16_realized_transfer_measured_zero_20260710",
                "score_impact": "UNKNOWN for FreSh treatment; underlying self-orient transfer measured approximately zero",
            } if trimmed else None),
            "readiness_deferrals": {
                "HorizonWeightedMargin": (
                    "isolated exact-V9 warm-start A/B required; stacking it into the FreSh "
                    "all-speed treatment would destroy causal custody"
                ),
                "StepNativeActivation": (
                    "activation-basin treatment must remain isolated because this ticket already "
                    "changes cold-start frequency selection through FreSh"
                ),
            },
            "dependency_slots": dependency_slots,
            "launch_blockers": blockers,
            "operator_go_required": True,
            "held": True,
            "trajectory_authority": (
                "training-only functional parity for MicroBatch(B=2); exact score/byte-close authority unchanged"
            ),
        }
    )

    constants = dict(parent.constants_manifest)
    constants.update(
        {
            "micro_batch_pairs": {
                "value": 2,
                "equation_id": "micro_batch_v9_functional_parity_waiver_20260712",
                "ladder_class": "measured_anchor",
                "fallback_used": False,
                "note": "training-only operator waiver; not bit-identical and not score authority",
            },
            "sps_gradient_conflict_sample_pairs": {
                "value": 4,
                "equation_id": "sps_gradient_role_conflict_probe_20260713",
                "ladder_class": "measured_anchor",
                "fallback_used": False,
                "note": "matches the four-stratum cheap probe count; exact engagement hook remains blocked",
            },
        }
    )
    if not trimmed:
        constants["fresh_frequency_shift"] = {
            "value": {
                "reference_freq_along": 8.0,
                "tangent_deficit": 3.2,
                "bias_grid": [0.0, 3.0, 0.1],
                "sample_pairs": 10,
            },
            "equation_id": "fresh_frequency_shift_init_v1",
            "ladder_class": "measured_anchor",
            "fallback_used": False,
            "note": "cold-start selection; 94 treatment scorer-pair equivalents including epoch-zero commit",
        }
    return replace(
        parent,
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
    )


__all__ = [
    "CAUSAL_MANIFEST_DEPENDENCY",
    "COMPONENT_WALLCLOCK_SCHEMA",
    "DEFAULT_OUT_DIR",
    "TRIMMED_OUT_DIR",
    "FULL_VARIANT",
    "TRIMMED_COMPLIANT_VARIANT",
    "MEMORY_VARIANTS",
    "DEFAULT_RUNTIME_OUT_DIR",
    "EXCLUDED_LEVERS",
    "PROGRAM_NAME",
    "TRIMMED_PROGRAM_NAME",
    "REQUIRED_COMPONENT_FIELDS",
    "SPS_ENGAGEMENT_SCHEMA",
    "compile_next_launch_all_levers_ticket",
    "excluded_levers_for_variant",
]
