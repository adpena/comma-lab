# SPDX-License-Identifier: MIT
"""Fail-closed C1 S_R throughput/convergence launch composition.

This compiler consumes the committed C1a ``v9_cgauge_ideal_mod19_sR`` parent when it is
available.  The compatibility path exists only so this isolated integration worktree can prove
the exact downstream composition against its older base; it reconstructs C1a from the same named
Lever factories and records that provenance.  Real launch authority still requires a measured,
content-bound composed-path benchmark receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from tac.witness_autoconfig import CrucibleV7LaunchConfig, _crucible_v7_argv_pairs
from tac.witness_dsl.curriculum_dsl import (
    AsyncVerdictOffload,
    CacheGtSkeleton,
    ComponentWallclockTelemetry,
    FrozenScorerOneThread,
    FusedRKernel,
    GroupedBackwardReference,
    Lever,
    MarginSaliency,
    MarginSaliencyReachability,
    PersistencePoolKernel,
    SafeCompileReference,
    SerialPairRouting,
    VerdictChunking,
)
from tac.witness_dsl.typed_config import TypedLever, build_launch_manifest

PROGRAM_NAME = "v9_cgauge_ideal_mod19_sR_c1_throughput"
DEFAULT_OUT_DIR = "experiments/results/v9_cgauge_ideal_mod19_sR_c1_throughput_20260715"
DEFAULT_BENCH_RECEIPT = Path(
    ".omx/research/c1_throughput_composed_bench_20260715.json"
)
BENCH_SCHEMA = "c1_throughput_composed_bench.v1"
C1A_COMMIT = "bdbbf5da175a46c11393ebbe56f53653828fb765"

ACTIVE_FACTORIES = (
    "SafeCompileReference",
    "SerialPairRouting",
    "FusedRKernel",
    "CacheGtSkeleton",
    "GroupedBackwardReference",
    "PersistencePoolKernel",
    "FrozenScorerOneThread",
    "AsyncVerdictOffload",
    "VerdictChunking",
    "ComponentWallclockTelemetry",
)

# Exhaustive disposition of the corpus-level speed/convergence candidates.  Numeric evidence and
# exact source paths live in the dated audit; this machine-readable table is the launch manifest's
# default-off/activation queue and makes omissions visible to the governed launcher.
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
    },
    "micro_batch_pairs_gt1": {
        "status": "EXCLUDED_SR_AND_IDENTITY_CONFLICT",
        "reason": "S_R has no batched consumer; B2 scorer flips and faithful n24 full-step speed was about 1.001x",
        "verdict_scope": "current batched scorer/reduction formulation on C1",
    },
    "safe_compile_hosc_activation": {
        "status": "HELD_HOST_CERTIFICATE_ABSENT",
        "reason": "per-chip safe-compile manifest is absent in this worktree; never transfer a certificate",
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


def _typed(lever: Lever) -> TypedLever:
    return TypedLever(
        name=lever.name,
        overrides=dict(lever.overrides),
        epochs_delta=int(lever.epochs_delta),
        notes=str(lever.notes),
        lawrefs=dict(lever.lawrefs),
        constant_manifest=dict(lever.constant_manifest),
        runtime_environment=dict(lever.runtime_environment),
    )


def _scientific_argv_sha(argv: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(argv), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compile_sr_parent(
    gt_cache_path: str, *, num_pairs: int, epochs: int, out_dir: str
) -> tuple[CrucibleV7LaunchConfig, str]:
    """Consume official C1a when present, else reconstruct its exact typed delta."""

    from tac.witness_dsl import spec_v9_cgauge as v9

    official = getattr(v9, "compile_v9_cgauge_ideal_mod19_sR_launch_config", None)
    if callable(official):
        return official(
            gt_cache_path=gt_cache_path,
            num_pairs=num_pairs,
            epochs=epochs,
            out_dir=out_dir,
        ), f"official:{C1A_COMMIT}"

    parent = v9.compile_v9_cgauge_ideal_mod19_launch_config(
        gt_cache_path=gt_cache_path,
        num_pairs=num_pairs,
        epochs=epochs,
        out_dir=out_dir,
    )
    additions = (
        _typed(MarginSaliency(start_epoch=0, window=0)),
        _typed(MarginSaliencyReachability(window=0)),
    )
    typed = parent.typed.model_copy(update={
        "name": "v9_cgauge_ideal_mod19_sR",
        "base": {**parent.typed.base, "--micro-batch-pairs": 1},
        "levers": tuple(parent.typed.levers) + additions,
    })
    rebound = parent._rebind_typed(typed)
    manifest = dict(rebound.dsl_program_manifest)
    manifest["sr_ab_contract"] = {
        "control_config": "v9_cgauge_ideal_mod19",
        "treatment_config": "v9_cgauge_ideal_mod19_sR",
        "treatment": True,
        "sole_scientific_argv_delta": "--margin-saliency-reachability",
        "micro_batch_pairs": 1,
        "equation_id": "margin_saliency_reachability_replaces_texture_proxy_v1",
        "status": "PREPARED_NOT_FIRED",
    }
    constants = dict(rebound.constants_manifest)
    constants["margin_saliency_reachability"] = {
        "value": True,
        "equation_id": "margin_saliency_reachability_replaces_texture_proxy_v1",
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {"micro_batch_pairs": 1, "source_commit": C1A_COMMIT},
        "note": "isolated-worktree compatibility reconstruction of committed C1a",
    }
    return CrucibleV7LaunchConfig(
        typed=rebound.typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=dict(rebound.schedule_governance),
    ), f"compatibility-reconstruction:{C1A_COMMIT}"


def compile_c1_sr_parent_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/v9_cgauge_ideal_mod19_sR_20260715",
) -> CrucibleV7LaunchConfig:
    """Resolve the committed C1a parent across the isolated-worktree merge boundary."""

    return _compile_sr_parent(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir
    )[0]


def _bench_receipt_status(
    path: Path,
    *,
    expected_program: str,
    expected_argv_sha: str,
    expected_typed_hash: str,
    expected_num_pairs: int,
) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"missing composed-path benchmark receipt: {path}"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable composed-path benchmark receipt: {type(exc).__name__}: {exc}"
    if not isinstance(receipt, dict) or receipt.get("schema") != BENCH_SCHEMA:
        return None, f"benchmark receipt schema must be {BENCH_SCHEMA}"
    required = {
        "program_name": expected_program,
        "scientific_argv_sha256": expected_argv_sha,
        "typed_config_hash": expected_typed_hash,
        "micro_batch_pairs": 1,
        "num_pairs": int(expected_num_pairs),
        "real_gt_cache": True,
        "sr_consumer_exercised": True,
        "bit_identity_spot_check": "PASS",
        "status": "MEASURED_PASS",
    }
    mismatches = {k: (receipt.get(k), v) for k, v in required.items() if receipt.get(k) != v}
    if mismatches:
        return None, f"benchmark receipt binding mismatch: {mismatches}"
    for key in ("seconds_per_epoch", "peak_rss_gib"):
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"benchmark receipt {key} must be numeric"
        if not math.isfinite(float(value)) or float(value) <= 0:
            return None, f"benchmark receipt {key} must be finite and positive"
    for key in ("hardware", "identity_authority", "rss_method"):
        value = receipt.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, f"benchmark receipt {key} must be a non-empty string"
    runtime_custody = receipt.get("runtime_custody")
    if not isinstance(runtime_custody, dict) or not runtime_custody:
        return None, "benchmark receipt runtime_custody must be a non-empty object"
    source_git_sha = receipt.get("source_git_sha")
    if (
        not isinstance(source_git_sha, str)
        or len(source_git_sha) != 40
        or any(c not in "0123456789abcdef" for c in source_git_sha)
    ):
        return None, "benchmark receipt source_git_sha must be a lowercase 40-hex commit"
    measurement_argv = receipt.get("measurement_argv")
    if (
        not isinstance(measurement_argv, list)
        or not measurement_argv
        or any(not isinstance(token, str) or not token for token in measurement_argv)
    ):
        return None, "benchmark receipt measurement_argv must be a non-empty string list"
    epoch_samples = receipt.get("epoch_seconds_samples")
    if (
        not isinstance(epoch_samples, list)
        or len(epoch_samples) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in epoch_samples
        )
    ):
        return None, "benchmark receipt needs at least two positive finite epoch samples"
    if not math.isclose(
        float(receipt["seconds_per_epoch"]),
        float(statistics.median(epoch_samples)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return None, "benchmark seconds_per_epoch must equal the median epoch sample"
    rss_samples = receipt.get("rss_samples_gib")
    if (
        not isinstance(rss_samples, list)
        or not rss_samples
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in rss_samples
        )
    ):
        return None, "benchmark receipt needs positive finite RSS samples"
    if not math.isclose(
        float(receipt["peak_rss_gib"]), max(map(float, rss_samples)), rel_tol=0.0, abs_tol=1e-12
    ):
        return None, "benchmark peak_rss_gib must equal max RSS sample"
    if receipt.get("bit_identity_max_abs") != 0:
        return None, "benchmark bit_identity_max_abs must be exactly zero"
    compared = receipt.get("bit_identity_compared_values")
    if isinstance(compared, bool) or not isinstance(compared, int) or compared < 1:
        return None, "benchmark bit_identity_compared_values must be a positive integer"
    for key in ("gt_cache_sha256", "sr_cache_sha256"):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return None, f"benchmark receipt {key} must be a lowercase SHA-256"
    if receipt.get("sr_source_kind") not in ("sidecar", "in_cache"):
        return None, "benchmark receipt sr_source_kind must be sidecar or in_cache"
    return receipt, None


def compile_c1_throughput_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = DEFAULT_OUT_DIR,
    bench_receipt_path: str | Path = DEFAULT_BENCH_RECEIPT,
) -> CrucibleV7LaunchConfig:
    """Compile the strict-identity C1 S_R throughput vehicle and its launch blockers."""

    parent, parent_source = _compile_sr_parent(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir
    )
    active = (
        SafeCompileReference(),
        SerialPairRouting(),
        FusedRKernel(),
        CacheGtSkeleton(),
        GroupedBackwardReference(),
        PersistencePoolKernel(),
        FrozenScorerOneThread(),
        AsyncVerdictOffload(),
        VerdictChunking(batch=32, pairs=0),
        ComponentWallclockTelemetry(),
    )
    runtime_environment = dict(parent.typed.runtime_environment)
    for lever in active:
        runtime_environment.update(lever.runtime_environment)
    typed = parent.typed.model_copy(update={
        "name": PROGRAM_NAME,
        "out_dir": str(out_dir),
        "purpose": (
            "C1 S_R treatment with every compatible measured strict-identity throughput lever; "
            "serial B1; composed-path benchmark and content custody fail closed"
        ),
        "levers": tuple(parent.typed.levers) + tuple(_typed(lever) for lever in active),
        "runtime_environment": runtime_environment,
    })
    violations = typed.validate_program()
    if violations:
        raise ValueError(f"C1 throughput typed DSL REFUSE: {violations[:6]}")

    argv = tuple(typed.to_program().compile_trainer_argv())
    pairs = dict(_crucible_v7_argv_pairs(argv))
    required_values = {
        "--micro-batch-pairs": "1",
        "--training-torch-threads": "1",
        "--verdict-batch": "32",
        "--verdict-pairs": "0",
        "--component-wallclock-probe-every": "1",
        "--safe-compile-regions": "none",
    }
    mismatches = {
        flag: (pairs.get(flag), value)
        for flag, value in required_values.items()
        if pairs.get(flag) != value
    }
    for flag in (
        "--margin-saliency-reachability",
        "--fused-r-kernel",
        "--cache-gt-skeleton",
        "--async-verdict",
        "--component-wallclock-telemetry",
        "--profile-timing",
    ):
        if flag not in pairs:
            mismatches[flag] = (pairs.get(flag), "present")
    required_env = {
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD": "0",
        "TAC_MLX_CUSTOM_PERSISTENCE_POOL": "1",
    }
    if typed.runtime_environment != required_env:
        mismatches["runtime_environment"] = (typed.runtime_environment, required_env)
    if mismatches:
        raise ValueError(f"C1 throughput actuation REFUSE: {mismatches}")

    argv_sha = _scientific_argv_sha(argv)
    bench_path = Path(bench_receipt_path)
    bench, bench_error = _bench_receipt_status(
        bench_path,
        expected_program=PROGRAM_NAME,
        expected_argv_sha=argv_sha,
        expected_typed_hash=typed.typed_config_hash(),
        expected_num_pairs=num_pairs,
    )
    blockers: list[dict[str, str]] = []
    cache = Path(gt_cache_path)
    sidecar = cache.with_name(cache.stem + "_sR.npz")
    if bench is not None and cache.is_file():
        actual_gt_sha = _file_sha256(cache)
        expected_gt_sha = bench["gt_cache_sha256"]
        if actual_gt_sha != expected_gt_sha:
            bench_error = (
                "benchmark GT-cache SHA mismatch: "
                f"receipt={expected_gt_sha} actual={actual_gt_sha}"
            )
            bench = None
        elif bench["sr_source_kind"] == "in_cache":
            if bench["sr_cache_sha256"] != actual_gt_sha:
                bench_error = "in-cache S_R receipt SHA must equal the GT-cache SHA"
                bench = None
        elif sidecar.is_file():
            actual_sr_sha = _file_sha256(sidecar)
            if bench["sr_cache_sha256"] != actual_sr_sha:
                bench_error = (
                    "benchmark S_R-sidecar SHA mismatch: "
                    f"receipt={bench['sr_cache_sha256']} actual={actual_sr_sha}"
                )
                bench = None
    if not cache.is_file():
        blockers.append({"id": "C1_REAL_GT_CACHE_MISSING", "detail": str(cache)})
    admitted_in_cache_sr = bool(bench and bench.get("sr_source_kind") == "in_cache")
    if not sidecar.is_file() and not admitted_in_cache_sr:
        # An in-cache sR member can clear this at runtime, but compile-time inspection deliberately
        # does not inflate a multi-GiB cache.  A measured receipt bound to its SHA clears launch.
        blockers.append({"id": "C1_SR_SIDECAR_NOT_VISIBLE", "detail": str(sidecar)})
    if bench_error is not None:
        blockers.append({"id": "C1_COMPOSED_BENCH_NOT_ADMITTED", "detail": bench_error})

    manifest = build_launch_manifest(
        program_name=PROGRAM_NAME,
        emitted_flag_names=sorted(pairs),
        typed_config_hash=typed.typed_config_hash(),
        typed_validated=True,
    )
    manifest.update({
        "expected_active_levers": [lever.name for lever in typed.levers],
        "c1_throughput_factories": list(ACTIVE_FACTORIES),
        "parent_source": parent_source,
        "strict_speed_bit_identity": True,
        "scientific_argv_sha256": argv_sha,
        "runtime_environment": dict(typed.runtime_environment),
        "excluded_or_held": EXCLUDED_OR_HELD,
        "sr_microbatch_reconciliation": {
            "sr_requires_micro_batch_pairs": 1,
            "batched_twin_consumes_sr": False,
            "b_gt1_status": "EXCLUDED",
            "faithful_n24_b2_step_speedup_x": 1.001,
            "custom_grouped_backward_status": "EXCLUDED_STRICT_IDENTITY",
        },
        "benchmark_receipt": bench,
        "benchmark_receipt_path": str(bench_path),
        "launch_blockers": blockers,
        "held": bool(blockers),
        "operator_go_required": True,
    })
    constants = dict(parent.constants_manifest)
    constants.update({
        "c1_serial_pair_routing": {
            "value": 1,
            "equation_id": "witness_fp_reorder_transform_bit_identity_wall_v1",
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"sr_batched_consumer": False, "microbatch_identity": "NO-GO"},
        },
        "c1_frozen_scorer_threads": {
            "value": 1,
            "equation_id": "segnet_exact_forward_cpu_thread_control_v1",
            "ladder_class": "measured",
            "fallback_used": False,
            "inputs": {"operator_training_standard": True},
        },
        "c1_grouped_backward_runtime": {
            "value": "reference",
            "equation_id": "witness_fp_reorder_transform_bit_identity_wall_v1",
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"custom_identity_proof": "FUNCTIONAL_PARITY_ONLY"},
        },
    })
    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=dict(parent.schedule_governance),
    )


__all__ = [
    "ACTIVE_FACTORIES",
    "BENCH_SCHEMA",
    "DEFAULT_BENCH_RECEIPT",
    "EXCLUDED_OR_HELD",
    "PROGRAM_NAME",
    "compile_c1_sr_parent_launch_config",
    "compile_c1_throughput_launch_config",
]
