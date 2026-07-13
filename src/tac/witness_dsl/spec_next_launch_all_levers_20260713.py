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
DEFAULT_RUNTIME_OUT_DIR = "/Volumes/VertigoDataTier/pact/next_launch_all_levers_20260713"

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
    causal_trainer_default_on = (
        "CausalManifestWriter" in trainer_text
        and '"mode": "default_on_read_only"' in trainer_text
        and "CAUSAL_MANIFEST_FILENAME" in trainer_text
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
            "status": "missing_exact_in_run_producer",
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
            "status": "missing_exact_engagement_callback",
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
    blockers = [
        {
            "id": "D_A_EXACT_COMPONENT_TIMERS_MISSING",
            "detail": "exact in-run component timing producer is absent; coarse --profile-timing is insufficient",
        },
        {
            "id": "D_B_EXACT_ENGAGEMENT_HOOK_MISSING",
            "detail": "no exact screw/phase engagement callback invokes the SPS gradient-conflict observer",
        },
        {
            "id": "MEMORY_WATERFILL_B2_UNMEASURED_N600",
            "detail": (
                "canonical memory waterfill pins micro_batch=1 because B=2 has no target-n600 RSS "
                "measurement; do not override the $0 gate"
            ),
        },
    ]
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
) -> CrucibleV7LaunchConfig:
    """Compile the strongest currently compatible held launch ticket.

    Pure/$0: this function cannot spawn or train.  The governed launcher is the
    only intended consumer, and it refuses a real spawn while ``launch_blockers``
    is non-empty.
    """

    parent = compile_v9_cgauge_ideal_mod19_launch_config(
        gt_cache_path=gt_cache_path,
        num_pairs=num_pairs,
        epochs=epochs,
        out_dir=out_dir,
    )
    # MicroBatch(B=2) is the V9 functional-parity treatment.  Tie-locus is the
    # sole ideal-core lever the real batched trainer explicitly refuses.
    kept = tuple(lv for lv in parent.typed.levers if lv.name != "tie_locus_displacement")
    additions = (
        _typed(MicroBatch(2)),
        _typed(FusedRKernel()),
        _typed(CacheGtSkeleton()),
        _typed(SafeCompileRegions("hosc_activation")),
        _typed(FreshFrequencyShift()),
        _observer_telemetry(),
    )
    typed = parent.typed.model_copy(
        update={
            "name": PROGRAM_NAME,
            "out_dir": str(out_dir),
            "purpose": (
                "HELD operator-GO-only n600 all-compatible-speed-stack + FreSh treatment; "
                "measure D-A component wall split, D-B screw/phase engagement gradient conflict, "
                "and causal transition rows; no launch until manifest blockers clear"
            ),
            "levers": kept + additions,
        }
    )
    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{PROGRAM_NAME} DSL gate: {len(violations)} WitnessProgram.validate violation(s): "
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
    for boolean_flag in (
        "--fused-r-kernel",
        "--cache-gt-skeleton",
        "--fresh-init",
        "--profile-timing",
        "--grad-interaction-telemetry",
        "--stage-checkpoints",
        "--async-verdict",
    ):
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
        raise ValueError(f"{PROGRAM_NAME} actuation REFUSE: {mismatch}")

    repo_root = Path(__file__).resolve().parents[3]
    dependency_slots, blockers = _dependency_slots(repo_root)
    manifest = build_launch_manifest(
        program_name=PROGRAM_NAME,
        emitted_flag_names=sorted(emitted),
        typed_config_hash=typed.typed_config_hash(),
        typed_validated=True,
    )
    manifest.update(
        {
            "expected_active_levers": [lv.name for lv in typed.levers],
            "excluded_levers": dict(EXCLUDED_LEVERS),
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
            "fresh_frequency_shift": {
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
    "DEFAULT_RUNTIME_OUT_DIR",
    "EXCLUDED_LEVERS",
    "PROGRAM_NAME",
    "REQUIRED_COMPONENT_FIELDS",
    "SPS_ENGAGEMENT_SCHEMA",
    "compile_next_launch_all_levers_ticket",
]
