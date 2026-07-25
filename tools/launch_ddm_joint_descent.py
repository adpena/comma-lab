#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Governed DDM #366 joint-descent launcher with bounded and full-run modes.

``--full-run`` is the only campaign loop.  This branch uses it only through
explicit bounded verification controls; MAIN review and standing-GO dispatch
remain required before an unbounded campaign may fire.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.local_acceleration.metal_fused_r_operator import (  # noqa: E402
    assert_metal_matches_cpu_oracle,
)
from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    _custom_metal_backward_status,
    load_mlx_distortion_scorer_adapter_from_upstream,
    temporary_mlx_device,
)
from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    receive_joint_descent_archive,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    BASELINE_DSEG,
    EVIDENCE_AXIS,
    MEMORY_RECEIPT_SCHEMA,
    POINTER,
    AdamStateV1,
    DirectDescriptionJointDescentMLXModule,
    DirectDescriptionJointDescentTypedConfigV1,
    PoseFinishEngageStateV1,
    ProposalGeometryInfeasibleError,
    classify_cumulative_fire_gate,
    classify_governed_stage_exit,
    classify_memory_preflight,
    classify_realized_stage_verdict,
    clipped_adam_step,
    compile_parameterized_archive,
    exact_final_target_gate,
    initial_adam_state,
    lift_v15_archive,
    linear_rewarmup_factor,
    load_stage_checkpoint,
    opening_candidate_gradient,
    parameter_group_indices,
    project_adam_state_geometry,
    realize_parameter_theta,
    realized_training_state,
    save_stage_checkpoint,
    template_camera_state,
    verify_trainable_group_ownership,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.optimization.pure_priced_realized_objective import (  # noqa: E402
    RealizedObjectiveState,
    pure_priced_realized_delta,
)

DEFAULT_TICKET = REPO / ".omx/research/configs/ddm_j5_366_realized_acceptance_warmstart_20260723.json"
POINTER_DSEG = 0.027470296224
WS1_BASELINE_DSEG_BY_ARCHIVE_SHA256 = {
    "264a09abb8f614eca104eb4ab1d0a12005ba65ec6a4fbc6620ff92f1c73281a9": (0.024124510023328993),
    "5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e": (0.07051923116048177),
}
STATIC_BOOTSTRAP_BOUND_GIB = 16.0
# DERIVED: historical static one-pair bootstrap bound plus ceil(stage3
# 4.72976016998291 GiB basis - measured stage1 0.7276554107666016 GiB basis).
STATIC_WORST_GEOMETRY_BOOTSTRAP_BOUND_GIB = STATIC_BOOTSTRAP_BOUND_GIB + math.ceil(
    4.72976016998291 - 0.7276554107666016
)
C1_INTEGER_TARGET_ERROR_MAX = 136_839
C1_ROLE_CLASS_IDS = (1, 3)
C1_RESIDUAL_CLASS_IDS = (0, 2, 4)
EXIT_REFUSE = 4
EXIT_HASH = 8


def _utc() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat()


def _expected_full_run_baseline_dseg(
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> float:
    """Return the SHA-sealed receiver baseline for the selected warm start."""

    return WS1_BASELINE_DSEG_BY_ARCHIVE_SHA256.get(
        config.source_archive_sha256,
        BASELINE_DSEG,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _resolve_input(relative: str, *, allow_authority_cache: bool = False) -> Path:
    candidate = REPO / relative
    if candidate.is_file():
        return candidate
    if allow_authority_cache:
        authority = Path("/Users/adpena/Projects/pact") / relative
        if authority.is_file():
            return authority
    raise DirectDescriptionError(f"bound joint-descent input is unavailable: {relative}")


def _verify_regular(path: Path, *, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"bound input is not a regular file: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise DirectDescriptionError(f"bound input bytes differ: {path}: {size} != {expected_bytes}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise DirectDescriptionError(f"bound input SHA-256 differs: {path}: {actual}")
    return {"path": str(path), "bytes": size, "sha256": actual}


class _RSSMonitor:
    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes = self._process.memory_info().rss
        self.free_floor_bytes = psutil.virtual_memory().available

    def _run(self) -> None:
        import psutil

        while not self._stop.wait(0.05):
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._process.memory_info().rss)
            self.free_floor_bytes = min(self.free_floor_bytes, psutil.virtual_memory().available)

    def __enter__(self) -> _RSSMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._run_once()

    def _run_once(self) -> None:
        import psutil

        self.peak_rss_bytes = max(self.peak_rss_bytes, self._process.memory_info().rss)
        self.free_floor_bytes = min(self.free_floor_bytes, psutil.virtual_memory().available)


def _storage_receipt(out_dir: Path) -> dict[str, Any]:
    resolved = out_dir.resolve()
    allowed = Path("/Volumes/VertigoDataTier/pact").resolve()
    if not resolved.is_relative_to(allowed):
        raise DirectDescriptionError("bounded consumer output must use /Volumes/VertigoDataTier/pact")
    resolved.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved)
    if usage.free < 2 * 1024**3:
        raise DirectDescriptionError("REFUSE_STORAGE_WATERFALL_FREE_SPACE_BELOW_2_GIB")
    return {
        "tier": str(allowed),
        "out_dir": str(resolved),
        "free_bytes": usage.free,
        "minimum_free_bytes": 2 * 1024**3,
        "cleanup": "no_large_scratch_created; immutable small checkpoints and receipts preserved",
    }


@contextmanager
def _same_outdir_guard(out_dir: Path, config: DirectDescriptionJointDescentTypedConfigV1):
    """Single-flight lock plus immutable typed identity for one run directory."""

    import fcntl

    lock_path = out_dir / ".ddm_joint_descent.lock"
    identity_path = out_dir / "run_identity.json"
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DirectDescriptionError("REFUSE_SAME_OUTDIR_ACTIVE_CONSUMER") from exc
        expected = {
            "schema": "ddm_joint_descent_run_identity.v1",
            "typed_config_hash": config.typed_config_hash(),
            "dsl_compile_hash": config.dsl_compile_hash,
            "source_archive_sha256": config.source_archive_sha256,
            "target_cache_sha256": config.target_cache_sha256,
            "seed": config.seed,
        }
        if identity_path.is_file():
            if json.loads(identity_path.read_bytes()) != expected:
                raise DirectDescriptionError("REFUSE_SAME_OUTDIR_INCOMPATIBLE_TYPED_IDENTITY")
        else:
            _atomic_json(identity_path, expected)
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _memory_projection(measured_peak_gib: float) -> float:
    # DERIVED safety envelope: larger of 20%+1 GiB or +2 GiB over measured peak.
    return max(measured_peak_gib * 1.2 + 1.0, measured_peak_gib + 2.0)


def _load_memory_receipt(path: Path, config: DirectDescriptionJointDescentTypedConfigV1) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"REFUSE_MEASURED_MEMORY_RECEIPT_UNAVAILABLE: {path}")
    row = json.loads(path.read_bytes())
    if row.get("schema") != MEMORY_RECEIPT_SCHEMA:
        raise DirectDescriptionError("memory receipt schema differs")
    if row.get("typed_config_hash") != config.typed_config_hash():
        raise DirectDescriptionError("memory receipt typed config hash differs")
    if row.get("target_cache_sha256") != config.target_cache_sha256:
        raise DirectDescriptionError("memory receipt target-cache hash differs")
    if row.get("admission") is not True:
        raise DirectDescriptionError("memory receipt is not SAFE")
    if config.execution_custody is not None:
        expected_memory = config.execution_custody["worst_geometry_memory_receipt"]
        if expected_memory.get("sha256") is None:
            raise DirectDescriptionError("worst-geometry memory receipt SHA-256 is not sealed")
        if path.resolve() != (REPO / expected_memory["path"]).resolve():
            raise DirectDescriptionError("memory receipt path differs from the sealed worst-geometry receipt")
        if _sha256_file(path) != expected_memory["sha256"]:
            raise DirectDescriptionError("memory receipt SHA-256 differs from the sealed worst-geometry receipt")
        _verify_execution_custody(config)
        contract = config.worst_geometry_memory_contract
        if contract is None:
            raise DirectDescriptionError("memory receipt lacks a typed worst-geometry contract")
        _assert_worst_geometry_receipt(row.get("memory_geometry"), contract)
        if row.get("j5_producer_artifacts") != config.execution_custody["j5_producer_artifacts"]:
            raise DirectDescriptionError("memory receipt J5 producer custody differs")
        expected_sources = {
            name: binding["sha256"] for name, binding in config.execution_custody["source_files"].items()
        }
        measured_sources = {
            name: binding["sha256"]
            for name, binding in row.get("source_custody", {}).items()
            if name in {"consumer", "launcher"}
        }
        if measured_sources != expected_sources:
            raise DirectDescriptionError("memory receipt final launcher/consumer custody differs")
    admitted, reason = classify_memory_preflight(
        float(row["projected_peak_gib"]), ceiling_gib=config.memory_ceiling_gib
    )
    if not admitted:
        raise DirectDescriptionError(reason)
    return row


def _assert_worst_geometry_receipt(geometry: Any, contract: Any) -> None:
    expected = {
        "pair_start": contract.selected_pair_start,
        "pair_ids": list(range(contract.selected_pair_start, contract.selected_pair_start + contract.train_batch)),
        "active_groups": list(contract.active_groups),
        "island_secants": contract.expected_island_secants,
        "lane_secants": contract.expected_lane_secants,
        "total_secants": contract.expected_total_secants,
        "derived_basis_gib": contract.derived_basis_gib,
    }
    if not isinstance(geometry, dict) or geometry != expected:
        raise DirectDescriptionError("memory receipt did not measure the sealed worst geometry")


def _verify_execution_custody(config: DirectDescriptionJointDescentTypedConfigV1) -> dict[str, Any]:
    custody = config.execution_custody
    if custody is None:
        return {}
    verified: dict[str, Any] = {"source_files": {}, "j5_producer_artifacts": {}}
    for group in ("source_files", "j5_producer_artifacts"):
        for name, binding in custody[group].items():
            path = Path(binding["path"])
            if not path.is_absolute():
                path = REPO / path
            if not path.is_file() or path.is_symlink():
                raise DirectDescriptionError(f"bound execution-custody file is unavailable: {path}")
            actual = _sha256_file(path)
            if actual != binding["sha256"]:
                raise DirectDescriptionError(f"bound execution-custody SHA-256 differs: {path}")
            verified[group][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
    return verified


def _governor(projected_gib: float) -> dict[str, Any]:
    from tools.witness_memory_preflight import system_aware_admission

    context = system_aware_admission(projected_gib, exclude_pid=os.getpid())
    payload = context.to_json()
    if not context.decision.admit:
        raise DirectDescriptionError("REFUSE_SYSTEM_MEMORY_GOVERNOR: " + context.decision.reason)
    return payload


def _base_receipt(config: DirectDescriptionJointDescentTypedConfigV1) -> dict[str, Any]:
    launcher_source = Path(__file__).resolve()
    consumer_source = REPO / "src/tac/optimization/direct_description_joint_descent.py"
    lift_source = REPO / "src/tac/optimization/direct_description_g1_worldsheet.py"
    payload = {
        "written_at_utc": _utc(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_custody": {
            "launcher": {"path": str(launcher_source.relative_to(REPO)), "sha256": _sha256_file(launcher_source)},
            "consumer": {"path": str(consumer_source.relative_to(REPO)), "sha256": _sha256_file(consumer_source)},
            "g1_lift": {"path": str(lift_source.relative_to(REPO)), "sha256": _sha256_file(lift_source)},
        },
        "typed_config_hash": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash,
        "seed": config.seed,
        "deterministic_algorithms": True,
        "ema_decay": config.ema_decay,
        "grad_clip": config.grad_clip,
        "hardware_axis": EVIDENCE_AXIS,
        "platform": platform.platform(),
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "research_only": True,
        "verdict_scope": "bounded executable-consumer smoke; no formulation, family, or score verdict",
    }
    if config.execution_custody is not None:
        payload["banked_r1_comparator"] = dict(config.execution_custody["banked_r1_comparator"])
    return payload


def _dry_run(
    *,
    config: DirectDescriptionJointDescentTypedConfigV1,
    memory_receipt_path: Path,
    out_dir: Path,
) -> int:
    memory = _load_memory_receipt(memory_receipt_path, config)
    plan = {
        "schema": "ddm_joint_descent_governed_dry_run.v1",
        **_base_receipt(config),
        "memory_receipt": {"path": str(memory_receipt_path), "sha256": _sha256_file(memory_receipt_path)},
        "projected_peak_gib": memory["projected_peak_gib"],
        "storage": _storage_receipt(out_dir),
        "governor": _governor(float(memory["projected_peak_gib"])),
        "execution_allowed": False,
        "long_launch_surface_present": False,
        "verdict": "DRY_RUN_GREEN_MAIN_REVIEW_AND_OPERATOR_GO_STILL_REQUIRED",
    }
    _atomic_json(out_dir / "governed_dry_run.json", plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


def _bounded_smoke_locked(args: argparse.Namespace, config: DirectDescriptionJointDescentTypedConfigV1) -> int:
    import mlx.core as mx

    out_dir = Path(args.out_dir)
    if args.simulate_kill_after_checkpoint and (
        not args.bootstrap_measurement or args.resume_from is not None or args.stop_after_step != 1
    ):
        raise DirectDescriptionError("simulated kill is restricted to bootstrap step 1")
    storage = _storage_receipt(out_dir)
    memory_path = Path(args.memory_receipt) if args.memory_receipt else out_dir / "memory_preflight.json"
    if args.bootstrap_measurement:
        if args.resume_from is not None:
            raise DirectDescriptionError("bootstrap measurement cannot also resume")
        projected_for_gate = STATIC_BOOTSTRAP_BOUND_GIB
        projection_source = "DERIVED_STATIC_BOUND_FOR_ONE_PAIR_SCORER_BACKWARD_PLUS_N600_MEMMAP_TARGETS"
    else:
        memory = _load_memory_receipt(memory_path, config)
        projected_for_gate = float(memory["projected_peak_gib"])
        projection_source = "MEASURED_ACTUAL_CONSUMER_RECEIPT"
    governor = _governor(projected_for_gate)

    source_path = _resolve_input(config.source_archive_path)
    cache_path = _resolve_input(config.target_cache_path, allow_authority_cache=True)
    source_custody = _verify_regular(
        source_path,
        expected_bytes=config.source_archive_bytes,
        expected_sha256=config.source_archive_sha256,
    )
    cache_custody = _verify_regular(
        cache_path,
        expected_bytes=config.target_cache_bytes,
        expected_sha256=config.target_cache_sha256,
    )

    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    np.random.seed(config.seed)
    started = time.monotonic()
    with _RSSMonitor() as monitor, temporary_mlx_device("gpu"):
        mx.random.seed(config.seed)
        fused_r_parity = assert_metal_matches_cpu_oracle(seed=config.seed)
        custom_active, custom_reason = _custom_metal_backward_status()
        if config.custom_grouped_backward_required and not custom_active:
            raise DirectDescriptionError("REFUSE_CUSTOM_GROUPED_BACKWARD_INACTIVE: " + custom_reason)

        archive = source_path.read_bytes()
        lift = lift_v15_archive(archive)
        stage00 = lift.exact_reemit()
        if stage00 != archive:
            raise DirectDescriptionError("stage00 exact archive reemit differs")
        receiver = receive_joint_descent_archive(archive)
        replay_pairs = (0, args.pair_id, 599)
        if not np.array_equal(
            receiver.render_camera_pairs(replay_pairs),
            receive_joint_descent_archive(stage00).render_camera_pairs(replay_pairs),
        ):
            raise DirectDescriptionError("stage00 camera replay differs")

        ownership = verify_trainable_group_ownership(lift) if args.verify_group_ownership else None
        labels = open_stored_npy_memmap(cache_path, "lstars")
        poses = open_stored_npy_memmap(cache_path, "gt_poses")
        if labels.shape != (600, 384, 512) or poses.shape != (600, 6):
            raise DirectDescriptionError("n600 target cache geometry differs")
        base_camera, template_masks = template_camera_state(lift, (args.pair_id,))
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(config.upstream_root, device="cpu")
        model = DirectDescriptionJointDescentMLXModule(
            lift=lift,
            scorer_adapter=adapter,
            seg_targets=labels,
            pose_targets=poses,
        )
        if args.resume_from is None:
            state = initial_adam_state(model.parameter_count)
            resume_metadata = None
        else:
            state, resume_metadata = load_stage_checkpoint(Path(args.resume_from), config=config)
            if state.theta.size != model.parameter_count:
                raise DirectDescriptionError("resume parameter count differs")
        if state.step >= args.stop_after_step:
            raise DirectDescriptionError("resume checkpoint is already at or beyond requested stop")

        initial_metrics = model.measure_components(
            state.theta,
            pair_ids=(args.pair_id,),
            base_camera=base_camera,
            template_masks=template_masks,
        )
        loss, gradient = model.loss_and_grad(
            state.theta,
            pair_ids=(args.pair_id,),
            base_camera=base_camera,
            template_masks=template_masks,
        )
        gradient_norm = float(np.linalg.norm(gradient.astype(np.float64)))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise DirectDescriptionError("bounded real-cache gradient is zero or nonfinite")

        accepted: tuple[AdamStateV1, dict[str, float], float] | None = None
        for learning_rate in (1.0, 0.25, 2.0, 0.0625):
            candidate = clipped_adam_step(
                state,
                gradient,
                learning_rate=learning_rate,
                grad_clip=config.grad_clip,
                ema_decay=config.ema_decay,
            )
            metrics = model.measure_components(
                candidate.theta,
                pair_ids=(args.pair_id,),
                base_camera=base_camera,
                template_masks=template_masks,
            )
            if metrics["d_seg"] < initial_metrics["d_seg"] or metrics["d_pose"] < initial_metrics["d_pose"]:
                accepted = candidate, metrics, learning_rate
                break
        if accepted is None:
            raise DirectDescriptionError(
                "INSTANCE_BLOCKER_BOUNDED_PAIR_NO_DSEG_OR_DPOSE_DESCENT_FOR_PREREGISTERED_STEP_GRID"
            )
        state, final_metrics, learning_rate = accepted
        telemetry = [
            {
                "event": "bounded_step",
                "step": state.step,
                "pair_id": args.pair_id,
                "loss_before": loss,
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
                "initial": initial_metrics,
                "final": final_metrics,
                "d_seg_decreased": final_metrics["d_seg"] < initial_metrics["d_seg"],
                "d_pose_decreased": final_metrics["d_pose"] < initial_metrics["d_pose"],
                "score_claim": False,
            }
        ]
        checkpoint = out_dir / "checkpoints" / f"stage00_step{state.step:06d}.npz"
        checkpoint_sha = save_stage_checkpoint(
            checkpoint,
            state,
            stage_id="00_bounded_consumer_smoke",
            config=config,
            telemetry=telemetry,
        )
        loaded, loaded_metadata = load_stage_checkpoint(checkpoint, config=config)
        checkpoint_bit_exact = (
            all(
                np.array_equal(getattr(state, field), getattr(loaded, field))
                for field in ("theta", "ema", "first_moment", "second_moment")
            )
            and state.step == loaded.step
        )
        if not checkpoint_bit_exact:
            raise DirectDescriptionError("checkpoint immediate parse-back is not bit-exact")
        mx.eval()
    elapsed = time.monotonic() - started

    measured_peak_gib = monitor.peak_rss_bytes / 1024**3
    projected_peak_gib = _memory_projection(measured_peak_gib)
    admission, reason = classify_memory_preflight(projected_peak_gib, ceiling_gib=config.memory_ceiling_gib)
    observations: list[dict[str, Any]] = []
    if memory_path.is_file():
        previous = json.loads(memory_path.read_bytes())
        if previous.get("schema") == MEMORY_RECEIPT_SCHEMA:
            observations.extend(previous.get("observations", ()))
    observations.append(
        {
            "step": state.step,
            "measured_peak_rss_gib": measured_peak_gib,
            "measured_free_memory_floor_gib": monitor.free_floor_bytes / 1024**3,
            "elapsed_seconds": elapsed,
            "process_boundary_resume": args.resume_from is not None,
        }
    )
    maximum_measured = max(float(row["measured_peak_rss_gib"]) for row in observations)
    projected_peak_gib = _memory_projection(maximum_measured)
    admission, reason = classify_memory_preflight(projected_peak_gib, ceiling_gib=config.memory_ceiling_gib)
    memory_receipt = {
        "schema": MEMORY_RECEIPT_SCHEMA,
        **_base_receipt(config),
        "target_cache_sha256": config.target_cache_sha256,
        "num_pairs": config.num_pairs,
        "verdict_batch": config.verdict_batch,
        "consumer_window": "one pair forward+backward with full n600 target cache bound and mapped",
        "observations": observations,
        "maximum_measured_peak_rss_gib": maximum_measured,
        "projection_formula": "DERIVED max(measured*1.2+1GiB, measured+2GiB)",
        "projected_peak_gib": projected_peak_gib,
        "operator_ceiling_gib": config.memory_ceiling_gib,
        "admission": admission,
        "reason": reason,
    }
    _atomic_json(memory_path, memory_receipt)
    receipt = {
        "schema": "ddm_joint_descent_bounded_resume_smoke.v1",
        **_base_receipt(config),
        "source_archive": source_custody,
        "target_cache": cache_custody,
        "stage00": {
            "archive_reemit_byte_identical": True,
            "camera_replay_byte_identical": True,
            "archive_bytes": len(stage00),
            "archive_sha256": config.source_archive_sha256,
            "inherited_d_seg": POINTER_DSEG,
            "inherited_d_seg_provenance": "DERIVED_FROM_EXACT_V15_ARCHIVE_BYTE_IDENTITY_AND_BOUND_RECEIPT",
            "parameter_inventory": lift.inventory(),
        },
        "group_ownership": ownership,
        "kernels": {
            "mlx_default_device": str(mx.default_device()),
            "custom_grouped_backward_active": custom_active,
            "custom_grouped_backward_reason": custom_reason,
            "fused_r_numpy_fp32_parity": fused_r_parity,
        },
        "storage": storage,
        "governor": governor,
        "projection_used_for_governor_gib": projected_for_gate,
        "projection_source": projection_source,
        "memory_receipt": {"path": str(memory_path), "sha256": _sha256_file(memory_path)},
        "resume": {
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "resume_metadata": resume_metadata,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_parseback_bit_exact": checkpoint_bit_exact,
            "loaded_metadata": loaded_metadata,
            "process_boundary": args.resume_from is not None,
        },
        "telemetry": telemetry,
        "memory_admission": admission,
        "simulated_kill_after_checkpoint": bool(args.simulate_kill_after_checkpoint),
        "verdict": ("BOUNDED_STEP_RESUME_GREEN" if admission else "REFUSE_REAL_CONSUMER_MEMORY_PROJECTION"),
    }
    receipt_path = out_dir / f"smoke_step{state.step:06d}.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    if args.simulate_kill_after_checkpoint:
        _atomic_json(
            out_dir / "kill_after_checkpoint.json",
            {
                "schema": "ddm_joint_descent_kill_boundary.v1",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "memory_receipt_sha256": _sha256_file(memory_path),
                "next_action": "start a new process with --resume-from this checkpoint",
                "score_claim": False,
            },
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(23)
    return 0 if admission else EXIT_REFUSE


def _bounded_smoke(args: argparse.Namespace, config: DirectDescriptionJointDescentTypedConfigV1) -> int:
    out_dir = Path(args.out_dir)
    _storage_receipt(out_dir)
    with _same_outdir_guard(out_dir, config):
        return _bounded_smoke_locked(args, config)


def _load_cpu_frozen_scorers(upstream_root: str) -> tuple[Any, Any]:
    """Load frozen CPU-torch scorer mirrors without inflating cached GT frames."""

    import torch
    from safetensors.torch import load_file

    upstream = Path(upstream_root)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import modules as upstream_modules

    modules_path = upstream / "modules.py"
    if Path(upstream_modules.__file__).resolve() != modules_path.resolve():
        raise DirectDescriptionError("frozen scorer imported non-custodied modules.py")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    segnet = upstream_modules.SegNet().eval().to("cpu")
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    segnet.load_state_dict(load_file(str(upstream_modules.segnet_sd_path), device="cpu"))
    posenet.load_state_dict(load_file(str(upstream_modules.posenet_sd_path), device="cpu"))
    for scorer in (segnet, posenet):
        for parameter in scorer.parameters():
            parameter.requires_grad = False
    return segnet, posenet


def _chunked_n600_verdict(
    *,
    archive: bytes,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    batch_size: int,
) -> dict[str, Any]:
    """Realized camera->R->frozen-scorer verdict with at most one RGB chunk resident."""

    from experiments.train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_argmax_batch,
    )

    if labels.shape != (600, 384, 512) or poses.shape != (600, 6):
        raise DirectDescriptionError("chunked full-run verdict requires exact n600 target geometry")
    receiver = receive_joint_descent_archive(archive)
    errors = 0
    sites = 0
    pose_squared_error = 0.0
    class_errors = np.zeros(5, dtype=np.int64)
    class_sites = np.zeros(5, dtype=np.int64)
    started = time.monotonic()
    for start in range(0, 600, batch_size):
        stop = min(600, start + batch_size)
        ids = tuple(range(start, stop))
        camera = receiver.render_camera_pairs(ids)
        target = [np.asarray(labels[index], dtype=np.int64) for index in ids]
        target_pose = [np.asarray(poses[index], dtype=np.float64) for index in ids]
        d_seg_rows, argmax = cpu_verdict_d_seg_argmax_batch(
            segnet,
            [camera[index, 1] for index in range(len(ids))],
            target,
        )
        d_pose_rows = cpu_verdict_d_pose_batch(
            posenet,
            [camera[index, 0] for index in range(len(ids))],
            [camera[index, 1] for index in range(len(ids))],
            target_pose,
        )
        errors += round(sum(d_seg_rows) * 384 * 512)
        sites += len(ids) * 384 * 512
        pose_squared_error += float(sum(d_pose_rows) * 6)
        target_array = np.stack(target)
        mismatched = np.asarray(argmax) != target_array
        for class_id in range(5):
            class_mask = target_array == class_id
            class_errors[class_id] += int(np.count_nonzero(mismatched & class_mask))
            class_sites[class_id] += int(np.count_nonzero(class_mask))
        del camera, argmax, target_array, mismatched
    d_seg = errors / sites
    d_pose = pose_squared_error / (600 * 6)
    if errors != int(class_errors.sum()) or sites != int(class_sites.sum()):
        raise DirectDescriptionError("chunked full-run verdict global/per-class totals differ")
    archive_bytes = len(archive)
    role_errors = int(class_errors[list(C1_ROLE_CLASS_IDS)].sum())
    residual_errors = int(class_errors[list(C1_RESIDUAL_CLASS_IDS)].sum())
    return {
        "schema": "ddm_joint_descent_chunked_stage_verdict.v1",
        "num_pairs": 600,
        "batch_size": batch_size,
        "maximum_rgb_chunks_resident": 1,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "advisory_action": 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / 37_545_489,
        "per_class": {
            name: {
                "class_id": class_id,
                "errors": int(class_errors[class_id]),
                "sites": int(class_sites[class_id]),
                "d_seg": float(class_errors[class_id] / class_sites[class_id]),
            }
            for class_id, name in enumerate(("Road", "Lane", "Undrivable", "Movable", "MyCar"))
        },
        "c1_debt_buckets": {
            "role_correction_owned": {
                "target_classes": ["Lane", "Movable"],
                "errors": role_errors,
                "baseline_ceiling_errors": 726_416,
            },
            "residual_trunk_owned": {
                "target_classes": ["Road", "Undrivable", "MyCar"],
                "errors": residual_errors,
                "integer_target_error_allowance": C1_INTEGER_TARGET_ERROR_MAX,
                "errors_above_target_allowance": max(
                    residual_errors - C1_INTEGER_TARGET_ERROR_MAX,
                    0,
                ),
                "baseline_errors_above_target_allowance": 2_377_273,
            },
            "partition_check": {
                "role_plus_residual_errors": role_errors + residual_errors,
                "global_errors": errors,
            },
        },
        "elapsed_seconds": time.monotonic() - started,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
    }


def _c1_bucket_delta(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Attribute one exact realized move to the fixed C1 error partition."""

    def debt_buckets(verdict: dict[str, Any]) -> dict[str, Any]:
        buckets = verdict.get("c1_debt_buckets")
        if buckets is not None:
            return buckets
        per_class = verdict.get("per_class")
        if not isinstance(per_class, dict):
            raise DirectDescriptionError("C1 bucket delta lacks canonical per-class error custody")
        try:
            role_errors = sum(int(per_class[name]["errors"]) for name in ("Lane", "Movable"))
            residual_errors = sum(int(per_class[name]["errors"]) for name in ("Road", "Undrivable", "MyCar"))
        except (KeyError, TypeError, ValueError) as exc:
            raise DirectDescriptionError("C1 bucket delta per-class custody is incomplete") from exc
        return {
            "role_correction_owned": {"errors": role_errors},
            "residual_trunk_owned": {
                "errors": residual_errors,
                "errors_above_target_allowance": max(
                    residual_errors - C1_INTEGER_TARGET_ERROR_MAX,
                    0,
                ),
            },
        }

    before = debt_buckets(reference)
    after = debt_buckets(candidate)
    role_delta = int(after["role_correction_owned"]["errors"]) - int(before["role_correction_owned"]["errors"])
    residual_delta = int(after["residual_trunk_owned"]["errors"]) - int(before["residual_trunk_owned"]["errors"])
    residual_debt_delta = int(after["residual_trunk_owned"]["errors_above_target_allowance"]) - int(
        before["residual_trunk_owned"]["errors_above_target_allowance"]
    )
    return {
        "role_correction_owned_delta_errors": role_delta,
        "residual_trunk_owned_delta_errors": residual_delta,
        "residual_trunk_debt_delta_errors": residual_debt_delta,
        "global_delta_errors": role_delta + residual_delta,
        "residual_bucket_descended": residual_delta < 0,
        "sign_convention": "candidate_minus_reference; negative removes errors",
    }


def _opening_exact_admitted(
    *,
    policy: str,
    pure_priced_accepted: bool,
    component_safe: bool,
    cumulative_fire_green: bool,
) -> bool:
    """Apply the typed opening policy without weakening campaign acceptance."""

    if policy == "pure_priced_exact_n600":
        return pure_priced_accepted
    if policy == "campaign_component_safe_exact_n600":
        return pure_priced_accepted and component_safe and cumulative_fire_green
    if policy == "component_safe_exact_n600":
        return component_safe
    raise DirectDescriptionError(f"unknown warm-start acceptance policy: {policy}")


def _seg_lexicographic_attempt_key(
    attempt: dict[str, Any],
    *,
    reference_seg_proxy: float,
) -> tuple[int, int, float, int]:
    """Rank within each rung; exact n600 receiver replay remains the gate."""

    seg_delta = float(attempt["metrics"]["seg_ce_margin"]) - reference_seg_proxy
    return (
        int(attempt["multiplier_index"]),
        0 if seg_delta <= 0.0 else 1,
        seg_delta,
        int(attempt["candidate_index"]),
    )


def _measurement_schedule(args: argparse.Namespace) -> dict[str, Any]:
    """Bounded pre-seal schedule; unavailable outside explicit measurement mode."""

    if not args.measure_full_config_window:
        raise DirectDescriptionError("REFUSE_FULL_RUN_TICKET_LACKS_RESEALED_SCHEDULE")
    if args.max_steps is None or args.max_steps <= 0:
        raise DirectDescriptionError("full-config measurement requires positive --max-steps")
    return {
        "train_batch": int(args.measurement_train_batch),
        "learning_rate": 0.25,
        "checkpoint_interval_steps": 1,
        "warm_start_pair": 447,
        "warm_start_steps": 1,
        "plateau_verdicts": 2,
        "warm_start_reform": None,
        "stages": [
            {
                "stage_id": "01_island_worldsheet_joint_descent",
                "active_groups": ("island_worldsheet", "shared_template_dof"),
                "maximum_steps": args.max_steps,
                "verdict_interval_steps": args.max_steps,
                "target_d_seg": POINTER_DSEG * 0.75,
                "target_d_pose": None,
            }
        ],
        "provenance": "BOUNDED_PRESEAL_MEASUREMENT_ONLY: batch candidate explicitly measured on this path; quarter-quantum from v16 warning; pair447 from J2 two-axis descent",
    }


def _sealed_schedule(config: DirectDescriptionJointDescentTypedConfigV1, args: argparse.Namespace) -> dict[str, Any]:
    schedule = config.full_run_schedule
    if schedule is None:
        return _measurement_schedule(args)
    if schedule.event_continuation is not None:
        if not schedule.event_continuation.execution_allowed:
            raise DirectDescriptionError(
                "REFUSE_EVENT_CONTINUATION_EXECUTION_DISABLED_PENDING_MAIN_REVIEW"
            )
        raise DirectDescriptionError(
            "REFUSE_EVENT_CONTINUATION_CAMPAIGN_ACTUATION_REQUIRES_MAIN_REVIEWED_RUNTIME_BINDING"
        )
    reform = schedule.warm_start_reform
    return {
        "train_batch": schedule.train_batch,
        "learning_rate": schedule.learning_rate_quantum_fraction,
        "checkpoint_interval_steps": schedule.checkpoint_interval_steps,
        "warm_start_pair": schedule.warm_start_pair,
        "warm_start_steps": schedule.warm_start_steps,
        "plateau_verdicts": schedule.plateau_verdicts,
        "warm_start_reform": (
            None
            if reform is None
            else {
                "adam_beta2": reform.adam_beta2,
                "lr_rewarmup_c": reform.lr_rewarmup_c,
                "lr_rewarmup_steps": reform.lr_rewarmup_steps,
                "lr_rewarmup_floor": reform.lr_rewarmup_floor,
                "lr_rewarmup_shape": reform.lr_rewarmup_shape,
                "maximum_continuous_update_quantum_fraction": (reform.maximum_continuous_update_quantum_fraction),
                "frozen_groups_until_first_admission": reform.frozen_groups_until_first_admission,
                "group_release_condition": reform.group_release_condition,
                "pose_objective_engage_condition": reform.pose_objective_engage_condition,
                "first_realized_admission": reform.first_realized_admission,
                "realized_acceptance_policy": reform.realized_acceptance_policy,
                "proposal_staging": reform.proposal_staging,
                "proposal_q8_denominator": reform.proposal_q8_denominator,
                "proposal_multipliers": reform.proposal_multipliers,
                "proposal_ordering": reform.proposal_ordering,
                "opening_active_groups": reform.opening_active_groups,
                "opening_candidate_ids": reform.opening_candidate_ids,
                "opening_candidate_pair_ids": reform.opening_candidate_pair_ids,
                "residual_bucket_admission_required": reform.residual_bucket_admission_required,
                "component_fire_gate": reform.component_fire_gate,
            }
        ),
        "pose_finish_engage": (
            None if schedule.pose_finish_engage is None else schedule.pose_finish_engage.to_payload()
        ),
        "stages": [
            {
                "stage_id": stage.stage_id,
                "active_groups": stage.active_groups,
                "maximum_steps": stage.maximum_steps,
                "verdict_interval_steps": stage.verdict_interval_steps,
                "target_d_seg": stage.target_d_seg,
                "target_d_pose": stage.target_d_pose,
            }
            for stage in schedule.stages
        ],
        "provenance": "HASH_SEALED_FULL_RUN_SCHEDULE",
    }


def _full_run_memory_receipt(
    *,
    config: DirectDescriptionJointDescentTypedConfigV1,
    monitor: _RSSMonitor,
    step_seconds: list[float],
    train_batch: int,
) -> dict[str, Any]:
    measured = monitor.peak_rss_bytes / 1024**3
    projected = _memory_projection(measured)
    admission, reason = classify_memory_preflight(projected, ceiling_gib=config.memory_ceiling_gib)
    ordered = sorted(step_seconds)
    median = ordered[len(ordered) // 2] if ordered else None
    return {
        "schema": MEMORY_RECEIPT_SCHEMA,
        **_base_receipt(config),
        "target_cache_sha256": config.target_cache_sha256,
        "num_pairs": config.num_pairs,
        "verdict_batch": config.verdict_batch,
        "consumer_window": "actual --full-run path: exact parameter compile/parse-back + sparse realized secants + paint/uint8-STE/R/frozen MLX scorer forward-backward",
        "train_batch": train_batch,
        "measured_step_seconds": step_seconds,
        "measured_seconds_per_step": median,
        "measured_seconds_per_step_low": min(step_seconds) if step_seconds else None,
        "measured_seconds_per_step_high": max(step_seconds) if step_seconds else None,
        "projected_seconds_per_n600_pass": None if median is None else median * math.ceil(600 / train_batch),
        "maximum_measured_peak_rss_gib": measured,
        "measured_free_memory_floor_gib": monitor.free_floor_bytes / 1024**3,
        "projection_formula": "DERIVED max(measured*1.2+1GiB, measured+2GiB)",
        "projected_peak_gib": projected,
        "operator_ceiling_gib": config.memory_ceiling_gib,
        "admission": admission,
        "reason": reason,
        "score_claim": False,
    }


def _write_immutable_json(
    path: Path,
    payload: dict[str, Any],
    *,
    volatile_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Write once, allowing deterministic replay to reuse timing-only drift."""

    if path.exists():
        existing = json.loads(path.read_bytes())
        expected_stable = {key: value for key, value in payload.items() if key not in volatile_fields}
        existing_stable = {key: value for key, value in existing.items() if key not in volatile_fields}
        if existing_stable != expected_stable:
            raise DirectDescriptionError(f"immutable full-run artifact already differs: {path}")
        return existing
    _atomic_json(path, payload)
    return payload


def _write_structural_proposal_rejection(
    *,
    out_dir: Path,
    candidate_id: str,
    global_step: int,
    multiplier: float,
    multiplier_index: int,
    proposal_staging: str,
    reason: str,
) -> dict[str, Any]:
    """Persist a receiver-invalid proposal and keep smaller rungs live."""

    slug = candidate_id.replace("+", "plus").replace("-", "minus")
    relpath = Path("verdicts") / (
        f"warm_start_proposal_step{global_step:06d}_{slug}_"
        f"shrink{multiplier_index:02d}_structural_reject.json"
    )
    return _write_immutable_json(
        out_dir / relpath,
        {
            "schema": "ddm_joint_descent_structural_proposal_rejection.v1",
            "candidate_id": candidate_id,
            "exact_replay_executed": False,
            "global_step": global_step,
            "proposal_multiplier": multiplier,
            "proposal_staging": proposal_staging,
            "reason": reason,
            "score_claim": False,
            "verdict": "REJECT_AND_SHRINK",
            "verdict_scope": "INSTANCE proposal geometry only; later ladder rungs remain live",
        },
    )


def _write_proposal_geometry_event(
    *,
    out_dir: Path,
    event: dict[str, Any],
    candidate_id: str,
    global_step: int,
    multiplier: float,
    multiplier_index: int,
    proposal_staging: str | None,
) -> dict[str, Any]:
    """Persist one typed cure/rejection without conflating it with score authority."""

    if (
        event.get("event") != "proposal_infeasible_geometry"
        or event.get("status") not in {"cured", "rejected"}
    ):
        raise DirectDescriptionError("proposal geometry telemetry event is invalid")
    slug = candidate_id.replace("+", "plus").replace("-", "minus")
    track_index = int(event["track_index"])
    relpath = Path("telemetry") / (
        f"geometry_step{global_step:06d}_{slug}_shrink{multiplier_index:02d}_"
        f"track{track_index:04d}_{event['status']}.json"
    )
    return _write_immutable_json(
        out_dir / relpath,
        {
            **event,
            "candidate_id": candidate_id,
            "global_step": global_step,
            "proposal_multiplier": multiplier,
            "proposal_multiplier_index": multiplier_index,
            "proposal_staging": proposal_staging,
        },
    )


def _worst_geometry_memory_bootstrap_locked(
    args: argparse.Namespace,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> int:
    """Measure the sealed stage-3 all-groups geometry without a campaign step."""

    import mlx.core as mx

    contract = config.worst_geometry_memory_contract
    if contract is None or config.execution_custody is None:
        raise DirectDescriptionError("worst-geometry bootstrap requires the J6A typed custody contract")
    if args.resume_from is not None:
        raise DirectDescriptionError("worst-geometry bootstrap creates, rather than resumes, its checkpoint")
    out_dir = Path(args.out_dir)
    storage = _storage_receipt(out_dir)
    verified_execution = _verify_execution_custody(config)
    governor = _governor(STATIC_WORST_GEOMETRY_BOOTSTRAP_BOUND_GIB)
    source_path = _resolve_input(config.source_archive_path)
    cache_path = _resolve_input(config.target_cache_path, allow_authority_cache=True)
    source_custody = _verify_regular(
        source_path,
        expected_bytes=config.source_archive_bytes,
        expected_sha256=config.source_archive_sha256,
    )
    cache_custody = _verify_regular(
        cache_path,
        expected_bytes=config.target_cache_bytes,
        expected_sha256=config.target_cache_sha256,
    )
    pair_ids = tuple(range(contract.selected_pair_start, contract.selected_pair_start + contract.train_batch))
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    np.random.seed(config.seed)
    started = time.monotonic()
    with _RSSMonitor() as monitor, temporary_mlx_device("gpu"):
        mx.random.seed(config.seed)
        fused_r_parity = assert_metal_matches_cpu_oracle(seed=config.seed)
        custom_active, custom_reason = _custom_metal_backward_status()
        if config.custom_grouped_backward_required and not custom_active:
            raise DirectDescriptionError("REFUSE_CUSTOM_GROUPED_BACKWARD_INACTIVE: " + custom_reason)
        archive = source_path.read_bytes()
        lift = lift_v15_archive(archive)
        groups = parameter_group_indices(lift)
        labels = open_stored_npy_memmap(cache_path, "lstars")
        poses = open_stored_npy_memmap(cache_path, "gt_poses")
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(config.upstream_root, device="cpu")
        model = DirectDescriptionJointDescentMLXModule(
            lift=lift,
            scorer_adapter=adapter,
            seg_targets=labels,
            pose_targets=poses,
        )
        state = initial_adam_state(model.parameter_count)
        base_camera, template_masks, basis, basis_indices, local_theta, current_archive = realized_training_state(
            lift,
            state.theta,
            pair_ids=pair_ids,
            active_groups=contract.active_groups,
            include_lane_programs=True,
        )
        basis_index_set = {int(value) for value in basis_indices}
        island_secants = len(basis_index_set.intersection(groups["island_worldsheet"]))
        lane_secants = len(basis_index_set.intersection(groups["lane_program"]))
        total_secants = len(basis_indices)
        actual_basis_gib = float(basis.nbytes / 1024**3)
        if (
            island_secants != contract.expected_island_secants
            or lane_secants != contract.expected_lane_secants
            or total_secants != contract.expected_total_secants
            or not math.isclose(actual_basis_gib, contract.derived_basis_gib, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise DirectDescriptionError(
                "REFUSE_WORST_GEOMETRY_DIFFERS: "
                f"island={island_secants}, lane={lane_secants}, total={total_secants}, "
                f"basis_gib={actual_basis_gib}"
            )
        step_started = time.monotonic()
        loss, gradient = model.loss_and_grad(
            local_theta,
            pair_ids=pair_ids,
            base_camera=base_camera,
            template_masks=template_masks,
            realized_secant_basis=basis,
            realized_secant_indices=basis_indices,
            pose_objective_weight=1.0,
        )
        gradient_norm = float(np.linalg.norm(gradient.astype(np.float64)))
        if not math.isfinite(float(loss)) or not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise DirectDescriptionError("REFUSE_WORST_GEOMETRY_FORWARD_BACKWARD_INVALID")
        mx.eval()
        step_seconds = time.monotonic() - step_started
        pose_state = PoseFinishEngageStateV1()
        checkpoint = out_dir / "checkpoints" / "03_pose_coupled_finish_worst_geometry_global000000.npz"
        checkpoint_sha = save_stage_checkpoint(
            checkpoint,
            state,
            stage_id="03_pose_coupled_finish_worst_geometry_memory_only",
            config=config,
            telemetry=(
                {
                    "event": "worst_geometry_memory_only",
                    "pair_ids": list(pair_ids),
                    "island_secants": island_secants,
                    "lane_secants": lane_secants,
                    "total_secants": total_secants,
                    "loss": float(loss),
                    "gradient_norm": gradient_norm,
                    "score_claim": False,
                },
            ),
            run_cursor={
                "stage_index": 0,
                "stage_step": 0,
                "global_step": 0,
                "verdict_history": [],
                "baseline_verdict": None,
                "pose_finish_engage_state": pose_state.to_payload(),
                "stage_end_reason": "memory_measurement_only",
                "campaign_blocker": None,
            },
            realized_archive={
                "bytes": len(current_archive),
                "sha256": hashlib.sha256(current_archive).hexdigest(),
                "lane_programs_materialized": True,
                "parameter_shadow": "zero_state_memory_measurement_only",
                "realized_parameter_count": 0,
            },
        )
        loaded, loaded_metadata = load_stage_checkpoint(checkpoint, config=config)
        if loaded.step != 0 or any(
            not np.array_equal(getattr(state, field), getattr(loaded, field))
            for field in ("theta", "ema", "first_moment", "second_moment")
        ):
            raise DirectDescriptionError("worst-geometry checkpoint immediate parse-back differs")
        PoseFinishEngageStateV1.from_payload(
            loaded_metadata["run_cursor"]["pose_finish_engage_state"],
            config=config.full_run_schedule.pose_finish_engage,
        )
    measured_peak_gib = monitor.peak_rss_bytes / 1024**3
    projected_peak_gib = _memory_projection(measured_peak_gib)
    admission, reason = classify_memory_preflight(
        projected_peak_gib,
        ceiling_gib=config.memory_ceiling_gib,
    )
    receipt = {
        "schema": MEMORY_RECEIPT_SCHEMA,
        **_base_receipt(config),
        "target_cache_sha256": config.target_cache_sha256,
        "num_pairs": config.num_pairs,
        "verdict_batch": config.verdict_batch,
        "consumer_window": (
            "sealed stage3 max window: all receiver-effective groups, 52 realized secants, "
            "paint/uint8-STE/R/frozen MLX scorer forward-backward"
        ),
        "memory_geometry": {
            "pair_start": contract.selected_pair_start,
            "pair_ids": list(pair_ids),
            "active_groups": list(contract.active_groups),
            "island_secants": island_secants,
            "lane_secants": lane_secants,
            "total_secants": total_secants,
            "derived_basis_gib": actual_basis_gib,
        },
        "measured_step_seconds": [step_seconds],
        "maximum_measured_peak_rss_gib": measured_peak_gib,
        "measured_free_memory_floor_gib": monitor.free_floor_bytes / 1024**3,
        "projection_formula": "DERIVED max(measured*1.2+1GiB, measured+2GiB)",
        "projected_peak_gib": projected_peak_gib,
        "operator_ceiling_gib": config.memory_ceiling_gib,
        "admission": admission,
        "reason": reason,
        "bootstrap_governor": governor,
        "storage": storage,
        "source_archive": source_custody,
        "target_cache": cache_custody,
        "kernels": {
            "mlx_default_device": str(mx.default_device()),
            "custom_grouped_backward_active": custom_active,
            "custom_grouped_backward_reason": custom_reason,
            "fused_r_numpy_fp32_parity": fused_r_parity,
        },
        "execution_custody_verified": verified_execution,
        "j5_producer_artifacts": dict(config.execution_custody["j5_producer_artifacts"]),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "immediate_parseback_bit_exact": True,
        },
        "elapsed_seconds": time.monotonic() - started,
        "memory_measurement_only": True,
        "campaign_launched": False,
        "execution_allowed_by_this_receipt": False,
        "score_claim": False,
        "verdict": (
            "SAFE_WORST_GEOMETRY_WITHIN_116_GIB_CEILING"
            if admission
            else "REFUSE_WORST_GEOMETRY_EXCEEDS_116_GIB_CEILING"
        ),
    }
    receipt_path = out_dir / "worst_geometry_memory_preflight.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if admission else EXIT_REFUSE


def _worst_geometry_memory_bootstrap(
    args: argparse.Namespace,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> int:
    out_dir = Path(args.out_dir)
    _storage_receipt(out_dir)
    with _same_outdir_guard(out_dir, config):
        return _worst_geometry_memory_bootstrap_locked(args, config)


def _resume_proof(
    args: argparse.Namespace,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> int:
    if args.resume_from is None or args.memory_receipt is None:
        raise DirectDescriptionError("resume proof requires --resume-from and --memory-receipt")
    storage = _storage_receipt(Path(args.out_dir))
    memory = _load_memory_receipt(Path(args.memory_receipt), config)
    governor = _governor(float(memory["projected_peak_gib"]))
    state, metadata = load_stage_checkpoint(Path(args.resume_from), config=config)
    cursor = metadata.get("run_cursor", {})
    pose_config = config.full_run_schedule.pose_finish_engage
    if pose_config is None:
        raise DirectDescriptionError("resume proof lacks typed pose-finish engage config")
    pose_state = PoseFinishEngageStateV1.from_payload(
        cursor.get("pose_finish_engage_state", {}),
        config=pose_config,
    )
    expected_checkpoint = memory.get("checkpoint", {})
    if _sha256_file(Path(args.resume_from)) != expected_checkpoint.get("sha256") or state.step != int(
        cursor.get("global_step", -1)
    ):
        raise DirectDescriptionError("resume proof checkpoint custody differs")
    receipt = {
        "schema": "ddm_joint_descent_process_boundary_resume_proof.v1",
        **_base_receipt(config),
        "process_pid": os.getpid(),
        "checkpoint": {
            "path": str(Path(args.resume_from)),
            "sha256": expected_checkpoint["sha256"],
            "global_step": state.step,
            "optimizer_arrays_bit_close_loadable": True,
        },
        "pose_finish_engage_state": pose_state.to_payload(),
        "memory_receipt": {
            "path": str(Path(args.memory_receipt)),
            "sha256": _sha256_file(Path(args.memory_receipt)),
            "worst_geometry_total_secants": memory["memory_geometry"]["total_secants"],
        },
        "storage": storage,
        "governor": governor,
        "execution_custody_verified": _verify_execution_custody(config),
        "campaign_launched": False,
        "execution_allowed_by_this_receipt": False,
        "score_claim": False,
        "verdict": "FRESH_PROCESS_RESUME_PROOF_GREEN",
    }
    _atomic_json(Path(args.out_dir) / "process_boundary_resume_proof.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _full_run_locked(args: argparse.Namespace, config: DirectDescriptionJointDescentTypedConfigV1) -> int:
    import mlx.core as mx

    schedule = _sealed_schedule(config, args)
    out_dir = Path(args.out_dir)
    storage = _storage_receipt(out_dir)
    admission_memory_path = (
        Path(args.memory_receipt) if args.memory_receipt else out_dir / "full_run_memory_preflight.json"
    )
    runtime_memory_path = out_dir / "full_run_memory_preflight.json"
    if args.bootstrap_measurement:
        if args.resume_from is not None:
            raise DirectDescriptionError("full-run bootstrap measurement cannot resume")
        projected_for_gate = STATIC_BOOTSTRAP_BOUND_GIB
        projection_source = "DERIVED_STATIC_BOUND_THEN_REPLACED_BY_ACTUAL_FULL_RUN_WINDOW"
    else:
        memory = _load_memory_receipt(admission_memory_path, config)
        projected_for_gate = float(memory["projected_peak_gib"])
        projection_source = "MEASURED_ACTUAL_FULL_RUN_RECEIPT"
    governor = _governor(projected_for_gate)

    source_path = _resolve_input(config.source_archive_path)
    cache_path = _resolve_input(config.target_cache_path, allow_authority_cache=True)
    source_custody = _verify_regular(
        source_path,
        expected_bytes=config.source_archive_bytes,
        expected_sha256=config.source_archive_sha256,
    )
    cache_custody = _verify_regular(
        cache_path,
        expected_bytes=config.target_cache_bytes,
        expected_sha256=config.target_cache_sha256,
    )
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    np.random.seed(config.seed)
    step_seconds: list[float] = []
    full_started = time.monotonic()
    with _RSSMonitor() as monitor, temporary_mlx_device("gpu"):
        mx.random.seed(config.seed)
        fused_r_parity = assert_metal_matches_cpu_oracle(seed=config.seed)
        custom_active, custom_reason = _custom_metal_backward_status()
        if config.custom_grouped_backward_required and not custom_active:
            raise DirectDescriptionError("REFUSE_CUSTOM_GROUPED_BACKWARD_INACTIVE: " + custom_reason)
        archive = source_path.read_bytes()
        lift = lift_v15_archive(archive)
        groups = parameter_group_indices(lift)
        labels = open_stored_npy_memmap(cache_path, "lstars")
        poses = open_stored_npy_memmap(cache_path, "gt_poses")
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(config.upstream_root, device="cpu")
        model = DirectDescriptionJointDescentMLXModule(
            lift=lift,
            scorer_adapter=adapter,
            seg_targets=labels,
            pose_targets=poses,
        )
        if args.resume_from is None:
            state = initial_adam_state(model.parameter_count)
            stage_index = 0
            stage_step = 0
            verdict_history: list[dict[str, Any]] = []
            baseline_verdict = None
            restored_pose_finish_payload = None
        else:
            state, metadata = load_stage_checkpoint(Path(args.resume_from), config=config)
            cursor = metadata.get("run_cursor", {})
            stage_index = int(cursor.get("stage_index", 0))
            stage_step = int(cursor.get("stage_step", 0))
            verdict_history = list(cursor.get("verdict_history", ()))
            baseline_verdict = cursor.get("baseline_verdict")
            restored_pose_finish_payload = cursor.get("pose_finish_engage_state")
            if state.theta.size != model.parameter_count:
                raise DirectDescriptionError("full-run resume parameter count differs")
        if stage_index >= len(schedule["stages"]):
            raise DirectDescriptionError("full-run resume cursor is already complete")

        cpu_scorers: tuple[Any, Any] | None = None
        expected_baseline_dseg = _expected_full_run_baseline_dseg(config)
        baseline_path = out_dir / "verdicts" / "stage00_baseline_n600.json"
        if baseline_verdict is None and baseline_path.is_file():
            baseline_verdict = json.loads(baseline_path.read_bytes())
            if (
                baseline_verdict.get("schema") != "ddm_joint_descent_chunked_stage_verdict.v1"
                or int(baseline_verdict.get("num_pairs", 0)) != 600
                or abs(float(baseline_verdict.get("d_seg", -1.0)) - expected_baseline_dseg) > 5.0e-10
            ):
                raise DirectDescriptionError("existing full-run baseline verdict custody differs")
        if baseline_verdict is None:
            cpu_scorers = _load_cpu_frozen_scorers(config.upstream_root)
            baseline_verdict = _chunked_n600_verdict(
                archive=archive,
                labels=labels,
                poses=poses,
                segnet=cpu_scorers[0],
                posenet=cpu_scorers[1],
                batch_size=config.verdict_batch,
            )
            if abs(float(baseline_verdict["d_seg"]) - expected_baseline_dseg) > 5.0e-10:
                raise DirectDescriptionError(
                    "REFUSE_FULL_RUN_BASELINE_DSEG_CUSTODY_DRIFT: "
                    f"{baseline_verdict['d_seg']} != {expected_baseline_dseg}"
                )
            _write_immutable_json(baseline_path, baseline_verdict)

        pose_finish_config = None if config.full_run_schedule is None else config.full_run_schedule.pose_finish_engage
        pose_finish_state: PoseFinishEngageStateV1 | None = None
        if pose_finish_config is not None:
            if restored_pose_finish_payload is not None:
                pose_finish_state = PoseFinishEngageStateV1.from_payload(
                    restored_pose_finish_payload,
                    config=pose_finish_config,
                )
            else:
                pose_finish_state = PoseFinishEngageStateV1().observe(
                    global_step=0,
                    d_seg=float(baseline_verdict["d_seg"]),
                    strict_seg_admission=False,
                    config=pose_finish_config,
                )
                for historical in verdict_history:
                    if "global_step" not in historical or "d_seg" not in historical:
                        continue
                    pose_finish_state = pose_finish_state.observe(
                        global_step=int(historical["global_step"]),
                        d_seg=float(historical["d_seg"]),
                        strict_seg_admission=bool(historical.get("warm_start_seg_admitted", False)),
                        config=pose_finish_config,
                    )

        stop_global = args.max_steps
        run_telemetry: list[dict[str, Any]] = []
        campaign_blocker: str | None = None
        latest_stage_decision: str | None = None
        reform = schedule.get("warm_start_reform")
        warm_start_admitted = any(row.get("warm_start_realized_admitted") is True for row in verdict_history)
        warm_start_seg_admitted = any(row.get("warm_start_seg_admitted") is True for row in verdict_history)
        latest_warm_start_admission = next(
            (row for row in reversed(verdict_history) if row.get("warm_start_realized_admitted") is True),
            None,
        )
        warm_start_component_safe = bool(
            latest_warm_start_admission is not None
            and latest_warm_start_admission.get("warm_start_component_safe_residual_admitted") is True
        )
        geometry_escape_injected = False
        while stage_index < len(schedule["stages"]):
            stage = schedule["stages"][stage_index]
            if stop_global is not None and state.step >= stop_global:
                break
            pair_start = (
                int(schedule["warm_start_pair"])
                if state.step < int(schedule["warm_start_steps"])
                else ((state.step - 1) * int(schedule["train_batch"])) % 600
            )
            pair_ids = tuple((pair_start + offset) % 600 for offset in range(int(schedule["train_batch"])))
            include_lanes = any(
                "lane_program" in prior["active_groups"] for prior in schedule["stages"][: stage_index + 1]
            )
            reform_active = reform is not None and not warm_start_component_safe and stage_index == 0
            if reform_active and reform.get("opening_active_groups"):
                active_groups = tuple(
                    group for group in stage["active_groups"] if group in set(reform["opening_active_groups"])
                )
                frozen_groups = set(stage["active_groups"]) - set(active_groups)
            else:
                frozen_groups = set(reform["frozen_groups_until_first_admission"]) if reform_active else set()
                active_groups = tuple(group for group in stage["active_groups"] if group not in frozen_groups)
            pose_objective_weight = 1.0 if pose_finish_state is None else 1.0 if pose_finish_state.engaged else 0.0
            rewarmup_factor = (
                linear_rewarmup_factor(
                    completed_steps=state.step,
                    rewarmup_steps=int(reform["lr_rewarmup_steps"]),
                    floor=float(reform["lr_rewarmup_floor"]),
                )
                if reform_active
                else 1.0
            )
            step_started = time.monotonic()
            base_camera, template_masks, basis, basis_indices, local_theta, current_archive = realized_training_state(
                lift,
                state.theta,
                pair_ids=pair_ids,
                active_groups=active_groups,
                include_lane_programs=include_lanes,
            )
            current_realized = realize_parameter_theta(lift, state.theta)
            initial = model.measure_components(
                local_theta,
                pair_ids=pair_ids,
                base_camera=base_camera,
                template_masks=template_masks,
                realized_secant_basis=basis,
                realized_secant_indices=basis_indices,
            )
            loss, gradient = model.loss_and_grad(
                local_theta,
                pair_ids=pair_ids,
                base_camera=base_camera,
                template_masks=template_masks,
                realized_secant_basis=basis,
                realized_secant_indices=basis_indices,
                pose_objective_weight=pose_objective_weight,
            )
            active = set().union(*(groups[name] for name in active_groups))
            gradient[[index for index in range(len(gradient)) if index not in active]] = 0.0
            gradient_norm = float(np.linalg.norm(gradient.astype(np.float64)))
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise DirectDescriptionError("INSTANCE_BLOCKER_FULL_RUN_ACTIVE_GRADIENT_ZERO_OR_NONFINITE")
            accepted: tuple[AdamStateV1, dict[str, float], float, bool, str, float] | None = None
            rejected_realized_verdict: dict[str, Any] | None = None
            candidate_ids = (
                tuple(reform["opening_candidate_ids"])
                if reform_active and reform.get("opening_candidate_ids")
                else ("local_exact_gradient",)
            )
            multipliers = (
                tuple(float(value) for value in reform["proposal_multipliers"]) if reform_active else (1.0, 0.5, 0.25)
            )
            proposal_attempts: list[dict[str, Any]] = []
            for candidate_index, candidate_id in enumerate(candidate_ids):
                proposal_gradient = opening_candidate_gradient(
                    lift,
                    candidate_id,
                    gradient,
                    active_pair_ids=(tuple(reform["opening_candidate_pair_ids"]) if reform_active else ()),
                )
                proposal_gradient[[index for index in range(len(proposal_gradient)) if index not in active]] = 0.0
                if not np.any(proposal_gradient):
                    continue
                for multiplier_index, multiplier in enumerate(multipliers):
                    learning_rate = float(schedule["learning_rate"]) * rewarmup_factor * multiplier
                    candidate = clipped_adam_step(
                        state,
                        proposal_gradient,
                        learning_rate=learning_rate,
                        grad_clip=config.grad_clip,
                        ema_decay=config.ema_decay,
                        beta2=(float(reform["adam_beta2"]) if reform_active else 0.999),
                        maximum_update=(
                            reform["maximum_continuous_update_quantum_fraction"] if reform_active else None
                        ),
                        theta_lattice_denominator=(int(reform["proposal_q8_denominator"]) if reform_active else None),
                    )
                    if args.force_geometry_escape_once and not geometry_escape_injected:
                        forced_theta = candidate.theta.copy()
                        forced_index = next(
                            index
                            for index in groups["island_worldsheet"]
                            if index in active
                        )
                        forced_theta[forced_index] = np.float32(8192.0)
                        candidate = AdamStateV1(
                            step=candidate.step,
                            theta=forced_theta,
                            ema=candidate.ema,
                            first_moment=candidate.first_moment,
                            second_moment=candidate.second_moment,
                        )
                        geometry_escape_injected = True
                    try:
                        candidate, geometry_events = project_adam_state_geometry(lift, candidate)
                    except ProposalGeometryInfeasibleError as exc:
                        _write_proposal_geometry_event(
                            out_dir=out_dir,
                            event=exc.event,
                            candidate_id=candidate_id,
                            global_step=candidate.step,
                            multiplier=multiplier,
                            multiplier_index=multiplier_index,
                            proposal_staging=(reform["proposal_staging"] if reform_active else None),
                        )
                        _write_structural_proposal_rejection(
                            out_dir=out_dir,
                            candidate_id=candidate_id,
                            global_step=candidate.step,
                            multiplier=multiplier,
                            multiplier_index=multiplier_index,
                            proposal_staging=(reform["proposal_staging"] if reform_active else "continuous_uint8"),
                            reason=str(exc),
                        )
                        continue
                    for geometry_event in geometry_events:
                        _write_proposal_geometry_event(
                            out_dir=out_dir,
                            event=geometry_event,
                            candidate_id=candidate_id,
                            global_step=candidate.step,
                            multiplier=multiplier,
                            multiplier_index=multiplier_index,
                            proposal_staging=(reform["proposal_staging"] if reform_active else None),
                        )
                    candidate_local = candidate.theta - (state.theta - local_theta)
                    metrics = model.measure_components(
                        candidate_local,
                        pair_ids=pair_ids,
                        base_camera=base_camera,
                        template_masks=template_masks,
                        realized_secant_basis=basis,
                        realized_secant_indices=basis_indices,
                    )
                    candidate_objective = 100.0 * metrics["seg_ce_margin"] + (
                        pose_objective_weight * math.sqrt(10.0 * metrics["d_pose"] + 1.0e-12)
                    )
                    candidate_realized = realize_parameter_theta(lift, candidate.theta)
                    realized_changed = not np.array_equal(candidate_realized, current_realized)
                    if reform_active and not realized_changed:
                        continue
                    if not reform_active and not realized_changed and candidate_objective >= loss:
                        continue
                    proposal_attempts.append(
                        {
                            "candidate": candidate,
                            "candidate_id": candidate_id,
                            "candidate_index": candidate_index,
                            "candidate_objective": candidate_objective,
                            "candidate_realized": candidate_realized,
                            "learning_rate": learning_rate,
                            "metrics": metrics,
                            "multiplier": multiplier,
                            "multiplier_index": multiplier_index,
                            "realized_changed": realized_changed,
                        }
                    )
            if reform_active and reform["proposal_ordering"] == ("seg_lexicographic_proxy_then_exact_component_gate"):
                initial_seg_proxy = float(initial["seg_ce_margin"])
                proposal_attempts.sort(
                    key=lambda attempt: _seg_lexicographic_attempt_key(
                        attempt,
                        reference_seg_proxy=initial_seg_proxy,
                    )
                )
            for attempt in proposal_attempts:
                candidate = attempt["candidate"]
                candidate_id = str(attempt["candidate_id"])
                candidate_realized = attempt["candidate_realized"]
                learning_rate = float(attempt["learning_rate"])
                metrics = attempt["metrics"]
                multiplier = float(attempt["multiplier"])
                multiplier_index = int(attempt["multiplier_index"])
                realized_changed = bool(attempt["realized_changed"])
                if reform_active and realized_changed:
                    slug = candidate_id.replace("+", "plus").replace("-", "minus")
                    try:
                        candidate_archive, _ = compile_parameterized_archive(
                            lift,
                            candidate.theta,
                            include_lane_programs=include_lanes,
                        )
                    except ProposalGeometryInfeasibleError as exc:
                        _write_proposal_geometry_event(
                            out_dir=out_dir,
                            event=exc.event,
                            candidate_id=candidate_id,
                            global_step=candidate.step,
                            multiplier=multiplier,
                            multiplier_index=multiplier_index,
                            proposal_staging=reform["proposal_staging"],
                        )
                        _write_structural_proposal_rejection(
                            out_dir=out_dir,
                            candidate_id=candidate_id,
                            global_step=candidate.step,
                            multiplier=multiplier,
                            multiplier_index=multiplier_index,
                            proposal_staging=reform["proposal_staging"],
                            reason=str(exc),
                        )
                        continue
                    if cpu_scorers is None:
                        cpu_scorers = _load_cpu_frozen_scorers(config.upstream_root)
                    realized_verdict = _chunked_n600_verdict(
                        archive=candidate_archive,
                        labels=labels,
                        poses=poses,
                        segnet=cpu_scorers[0],
                        posenet=cpu_scorers[1],
                        batch_size=config.verdict_batch,
                    )
                    reference_verdict = next(
                        (row for row in reversed(verdict_history) if row.get("warm_start_realized_admitted") is True),
                        baseline_verdict,
                    )
                    component_decision = classify_realized_stage_verdict(
                        reference_d_seg=float(reference_verdict["d_seg"]),
                        reference_d_pose=float(reference_verdict["d_pose"]),
                        candidate_d_seg=float(realized_verdict["d_seg"]),
                        candidate_d_pose=float(realized_verdict["d_pose"]),
                        target_d_seg=float(stage["target_d_seg"]),
                        target_d_pose=stage["target_d_pose"],
                    )
                    pure_delta = pure_priced_realized_delta(
                        RealizedObjectiveState(
                            float(reference_verdict["d_seg"]),
                            float(reference_verdict["d_pose"]),
                            int(reference_verdict["archive_bytes"]),
                        ),
                        RealizedObjectiveState(
                            float(realized_verdict["d_seg"]),
                            float(realized_verdict["d_pose"]),
                            int(realized_verdict["archive_bytes"]),
                        ),
                    )
                    bucket_delta = _c1_bucket_delta(reference_verdict, realized_verdict)
                    cumulative_bucket_delta = _c1_bucket_delta(baseline_verdict, realized_verdict)
                    component_safe = component_decision in {
                        "REALIZED_STAGE_TARGET_MET",
                        "REALIZED_STAGE_DESCENT_CONTINUE",
                        "REALIZED_STAGE_SEG_FLAT_POSE_DESCENT_CONTINUE",
                    }
                    cumulative_fire_green, cumulative_fire_decision = classify_cumulative_fire_gate(
                        baseline_d_seg=float(baseline_verdict["d_seg"]),
                        baseline_d_pose=float(baseline_verdict["d_pose"]),
                        candidate_d_seg=float(realized_verdict["d_seg"]),
                        candidate_d_pose=float(realized_verdict["d_pose"]),
                        cumulative_residual_delta_errors=int(
                            cumulative_bucket_delta["residual_trunk_owned_delta_errors"]
                        ),
                        residual_descent_required=bool(reform["residual_bucket_admission_required"]),
                    )
                    exact_admitted = _opening_exact_admitted(
                        policy=reform["realized_acceptance_policy"],
                        pure_priced_accepted=pure_delta.accepted,
                        component_safe=component_safe,
                        cumulative_fire_green=cumulative_fire_green,
                    )
                    candidate_pose_finish_state = pose_finish_state
                    if exact_admitted and pose_finish_state is not None:
                        candidate_pose_finish_state = pose_finish_state.observe(
                            global_step=candidate.step,
                            d_seg=float(realized_verdict["d_seg"]),
                            strict_seg_admission=(float(realized_verdict["d_seg"]) < float(baseline_verdict["d_seg"])),
                            config=pose_finish_config,
                        )
                    proposal_receipt_relpath = Path("verdicts") / (
                        f"warm_start_proposal_step{candidate.step:06d}_{slug}_shrink{multiplier_index:02d}_n600.json"
                    )
                    realized_verdict.update(
                        {
                            "stage_id": stage["stage_id"],
                            "stage_step": stage_step + 1,
                            "global_step": candidate.step,
                            "proposal_source": candidate_id,
                            "proposal_multiplier": multiplier,
                            "proposal_staging": reform["proposal_staging"],
                            "reference_d_seg": float(reference_verdict["d_seg"]),
                            "reference_d_pose": float(reference_verdict["d_pose"]),
                            "reference_archive_bytes": int(reference_verdict["archive_bytes"]),
                            "component_gate_decision": component_decision,
                            "pure_priced_delta": {
                                "seg_term": pure_delta.seg_term,
                                "pose_term": pure_delta.pose_term,
                                "rate_term": pure_delta.rate_term,
                                "joint_delta": pure_delta.joint_delta,
                                "accepted": pure_delta.accepted,
                                "acceptance_authority": "strict_joint_delta_lt_zero",
                            },
                            "c1_bucket_delta_vs_last_admitted": bucket_delta,
                            "c1_bucket_delta_cumulative_vs_baseline": cumulative_bucket_delta,
                            "cumulative_fire_gate_decision": cumulative_fire_decision,
                            "warm_start_realized_admitted": exact_admitted,
                            "warm_start_seg_admitted": exact_admitted
                            and float(realized_verdict["d_seg"]) < float(baseline_verdict["d_seg"]),
                            "warm_start_component_safe_residual_admitted": (exact_admitted and cumulative_fire_green),
                            "pose_finish_engage_state": (
                                None
                                if candidate_pose_finish_state is None
                                else candidate_pose_finish_state.to_payload()
                            ),
                            "parameter_shadow": "live_opening_candidate",
                            "realized_parameter_count": int(np.count_nonzero(candidate_realized)),
                            "realized_changed_parameter_count": int(
                                np.count_nonzero(candidate_realized != current_realized)
                            ),
                            "decision_basis": (
                                "PAINT_UINT8_R_FROZEN_SCORERS_EXACT_ARCHIVE_BYTES_PURE_PRICED"
                                if reform["realized_acceptance_policy"] == "pure_priced_exact_n600"
                                else "FIRST_INTEGER_REALIZATION_EXACT_N600_ABORT_ROLLBACK"
                            ),
                            "model_predicted_plateau_used": False,
                            "proposal_receipt_relpath": str(proposal_receipt_relpath),
                        }
                    )
                    realized_verdict = _write_immutable_json(
                        out_dir / proposal_receipt_relpath,
                        realized_verdict,
                        volatile_fields=("elapsed_seconds",),
                    )
                    latest_stage_decision = component_decision
                    if not exact_admitted:
                        rejected_realized_verdict = realized_verdict
                        continue
                    verdict_history.append(realized_verdict)
                    pose_finish_state = candidate_pose_finish_state
                    warm_start_admitted = True
                    warm_start_seg_admitted = warm_start_seg_admitted or realized_verdict["warm_start_seg_admitted"]
                    warm_start_component_safe = bool(realized_verdict["warm_start_component_safe_residual_admitted"])
                accepted = (
                    candidate,
                    metrics,
                    learning_rate,
                    realized_changed,
                    candidate_id,
                    multiplier,
                )
                break
            if rejected_realized_verdict is not None and accepted is None:
                campaign_blocker = "BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER"
                rollback_event = {
                    "schema": "ddm_joint_descent_warm_start_rollback.v1",
                    "event": "warm_start_rejected_proposal_rollback",
                    "rejected_candidate_global_step": int(rejected_realized_verdict["global_step"]),
                    "preserved_global_step": state.step,
                    "stage_id": stage["stage_id"],
                    "stage_step": stage_step,
                    "realized_stage_decision": campaign_blocker,
                    "rejected_joint_delta": rejected_realized_verdict["pure_priced_delta"]["joint_delta"],
                    "rejected_proposal_receipt": (rejected_realized_verdict["proposal_receipt_relpath"]),
                    "rollback_state": "last_admitted_receiver_state",
                    "score_claim": False,
                }
                rollback_cursor = {
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    "global_step": state.step,
                    "verdict_history": verdict_history,
                    "baseline_verdict": baseline_verdict,
                    "pose_finish_engage_state": (None if pose_finish_state is None else pose_finish_state.to_payload()),
                    "stage_end_reason": campaign_blocker,
                    "campaign_blocker": campaign_blocker,
                }
                rejected_step = int(rejected_realized_verdict["global_step"])
                rollback_checkpoint = (
                    out_dir
                    / "checkpoints"
                    / (f"{stage['stage_id']}_blocked_proposal{rejected_step:06d}_rollback_global{state.step:06d}.npz")
                )
                save_stage_checkpoint(
                    rollback_checkpoint,
                    state,
                    stage_id=stage["stage_id"],
                    config=config,
                    telemetry=(rollback_event,),
                    run_cursor=rollback_cursor,
                    realized_archive={
                        "bytes": len(current_archive),
                        "sha256": hashlib.sha256(current_archive).hexdigest(),
                        "lane_programs_materialized": include_lanes,
                        "parameter_shadow": "last_admitted_live_resume_state",
                        "realized_parameter_count": int(np.count_nonzero(current_realized)),
                    },
                )
                loaded_rollback, loaded_rollback_metadata = load_stage_checkpoint(
                    rollback_checkpoint,
                    config=config,
                )
                if (
                    loaded_rollback.step != state.step
                    or loaded_rollback_metadata["run_cursor"]["campaign_blocker"] != campaign_blocker
                    or any(
                        not np.array_equal(getattr(state, field), getattr(loaded_rollback, field))
                        for field in ("theta", "ema", "first_moment", "second_moment")
                    )
                ):
                    raise DirectDescriptionError("warm-start rollback checkpoint immediate parse-back differs")
                break
            if accepted is None:
                raise DirectDescriptionError("INSTANCE_BLOCKER_FULL_RUN_NO_LOCAL_OBJECTIVE_DESCENT_DURING_REWARMUP")
            state, final, learning_rate, realized_boundary_crossed, accepted_candidate_id, accepted_multiplier = (
                accepted
            )
            stage_step += 1
            elapsed = time.monotonic() - step_started
            step_seconds.append(elapsed)
            row = {
                "schema": "ddm_joint_descent_full_run_step.v1",
                "event": "full_run_step",
                "global_step": state.step,
                "stage_index": stage_index,
                "stage_id": stage["stage_id"],
                "stage_step": stage_step,
                "pair_ids": list(pair_ids),
                "active_groups": list(active_groups),
                "frozen_groups": sorted(frozen_groups),
                "active_receiver_effective_coordinates": len(active),
                "realized_secant_coordinates_in_window": len(basis_indices),
                "loss_before": loss,
                "gradient_norm": gradient_norm,
                "learning_rate_uint8_quantum_fraction": learning_rate,
                "lr_rewarmup_factor": rewarmup_factor,
                "pose_objective_weight": pose_objective_weight,
                "pose_finish_engage_state": (None if pose_finish_state is None else pose_finish_state.to_payload()),
                "maximum_continuous_update_quantum_fraction": (
                    reform["maximum_continuous_update_quantum_fraction"] if reform_active else None
                ),
                "proposal_source": accepted_candidate_id,
                "proposal_multiplier": accepted_multiplier,
                "proposal_staging": reform["proposal_staging"] if reform_active else None,
                "proposal_q8_denominator": reform["proposal_q8_denominator"] if reform_active else None,
                "realized_boundary_crossed": realized_boundary_crossed,
                "initial": initial,
                "final": final,
                "seconds": elapsed,
                "source_archive_sha256": _sha256_file(source_path),
                "current_parseback_archive_sha256": hashlib.sha256(current_archive).hexdigest(),
                "first_order_plateau_prediction_used": False,
                "score_claim": False,
            }
            row = _write_immutable_json(
                out_dir / "telemetry" / f"step{state.step:06d}.json",
                row,
                volatile_fields=("seconds",),
            )
            run_telemetry.append(row)
            mx.eval()

            periodic = state.step % int(schedule["checkpoint_interval_steps"]) == 0
            bounded_stop = stop_global is not None and state.step >= stop_global
            stage_limit = stage_step >= int(stage["maximum_steps"])
            verdict_due = stage_step % int(stage["verdict_interval_steps"]) == 0
            if bounded_stop and args.stage_exit_on_stop:
                verdict_due = True
            verdict: dict[str, Any] | None = None
            live_archive, live_realized_theta = compile_parameterized_archive(
                lift, state.theta, include_lane_programs=include_lanes
            )
            if verdict_due:
                scheduled_stage_verdict = stage_step % int(stage["verdict_interval_steps"]) == 0
                verdict_shadow = "ema" if scheduled_stage_verdict else "live_bounded_smoke_only"
                verdict_theta = state.ema if scheduled_stage_verdict else state.theta
                verdict_archive, verdict_realized_theta = compile_parameterized_archive(
                    lift, verdict_theta, include_lane_programs=include_lanes
                )
                if cpu_scorers is None:
                    cpu_scorers = _load_cpu_frozen_scorers(config.upstream_root)
                verdict = _chunked_n600_verdict(
                    archive=verdict_archive,
                    labels=labels,
                    poses=poses,
                    segnet=cpu_scorers[0],
                    posenet=cpu_scorers[1],
                    batch_size=config.verdict_batch,
                )
                verdict.update(
                    {
                        "stage_id": stage["stage_id"],
                        "stage_step": stage_step,
                        "global_step": state.step,
                        "target_d_seg": stage["target_d_seg"],
                        "target_d_pose": stage["target_d_pose"],
                        "realized_parameter_count": int(np.count_nonzero(verdict_realized_theta)),
                        "parameter_shadow": verdict_shadow,
                        "decision_basis": "REALIZED_THROUGH_ARCHIVE_PARSEBACK_PAINT_UINT8_R_FROZEN_SCORERS",
                        "model_predicted_plateau_used": False,
                    }
                )
            target_met = False
            if verdict is not None:
                previous_verdict = verdict_history[-1] if verdict_history else baseline_verdict
                if previous_verdict is None:
                    raise DirectDescriptionError("REFUSE_REALIZED_STAGE_VERDICT_LACKS_REFERENCE")
                latest_stage_decision = classify_realized_stage_verdict(
                    reference_d_seg=float(previous_verdict["d_seg"]),
                    reference_d_pose=float(previous_verdict["d_pose"]),
                    candidate_d_seg=float(verdict["d_seg"]),
                    candidate_d_pose=float(verdict["d_pose"]),
                    target_d_seg=float(stage["target_d_seg"]),
                    target_d_pose=stage["target_d_pose"],
                )
                verdict["reference_d_seg"] = float(previous_verdict["d_seg"])
                verdict["reference_d_pose"] = float(previous_verdict["d_pose"])
                verdict["realized_stage_decision"] = latest_stage_decision
                target_met = latest_stage_decision == "REALIZED_STAGE_TARGET_MET"
                if pose_finish_state is not None and (
                    not pose_finish_state.exact_verdict_steps or state.step > pose_finish_state.exact_verdict_steps[-1]
                ):
                    pose_finish_state = pose_finish_state.observe(
                        global_step=state.step,
                        d_seg=float(verdict["d_seg"]),
                        strict_seg_admission=(
                            latest_stage_decision
                            in {
                                "REALIZED_STAGE_TARGET_MET",
                                "REALIZED_STAGE_DESCENT_CONTINUE",
                            }
                            and float(verdict["d_seg"]) < float(previous_verdict["d_seg"])
                        ),
                        config=pose_finish_config,
                    )
                verdict["pose_finish_engage_state"] = (
                    None if pose_finish_state is None else pose_finish_state.to_payload()
                )
                verdict = _write_immutable_json(
                    out_dir / "verdicts" / f"{stage['stage_id']}_step{state.step:06d}_n600.json",
                    verdict,
                    volatile_fields=("elapsed_seconds",),
                )
                verdict_history.append(verdict)
            recent = [
                row
                for row in verdict_history
                if row.get("stage_id") == stage["stage_id"] and row.get("realized_stage_decision") is not None
            ]
            plateau = len(recent) >= int(schedule["plateau_verdicts"]) and all(
                float(after["d_seg"]) >= float(before["d_seg"]) and float(after["d_pose"]) >= float(before["d_pose"])
                for before, after in zip(
                    recent[-int(schedule["plateau_verdicts"]) : -1],
                    recent[-int(schedule["plateau_verdicts"]) + 1 :],
                    strict=True,
                )
            )
            stage_exit_decision = classify_governed_stage_exit(
                target_met=target_met,
                stage_limit=stage_limit,
                plateau=plateau,
                component_decision=latest_stage_decision,
            )
            stage_end = campaign_blocker is None and stage_exit_decision == "ADVANCE_EXACT_TARGET_MET"
            if stage_exit_decision.startswith(("STOPPED_", "BLOCKED_", "REFUSE_")):
                campaign_blocker = stage_exit_decision
            cursor = {
                "stage_index": stage_index + (1 if stage_end else 0),
                "stage_step": 0 if stage_end else stage_step,
                "global_step": state.step,
                "verdict_history": verdict_history,
                "baseline_verdict": baseline_verdict,
                "pose_finish_engage_state": (None if pose_finish_state is None else pose_finish_state.to_payload()),
                "stage_end_reason": campaign_blocker or stage_exit_decision,
                "campaign_blocker": campaign_blocker,
            }
            accepted_checkpoint = (
                out_dir
                / "checkpoints"
                / f"{stage['stage_id']}_accepted_global{state.step:06d}.npz"
            )
            save_stage_checkpoint(
                accepted_checkpoint,
                state,
                stage_id=stage["stage_id"],
                config=config,
                telemetry=(row,),
                run_cursor=cursor,
                realized_archive={
                    "bytes": len(live_archive),
                    "sha256": hashlib.sha256(live_archive).hexdigest(),
                    "lane_programs_materialized": include_lanes,
                    "parameter_shadow": "live_resume_state",
                    "realized_parameter_count": int(np.count_nonzero(live_realized_theta)),
                    "checkpoint_reason": "every_accepted_step",
                },
            )
            loaded_accepted, loaded_accepted_metadata = load_stage_checkpoint(
                accepted_checkpoint,
                config=config,
            )
            if state.step != loaded_accepted.step or any(
                not np.array_equal(getattr(state, field), getattr(loaded_accepted, field))
                for field in ("theta", "ema", "first_moment", "second_moment")
            ):
                raise DirectDescriptionError("accepted-step checkpoint immediate parse-back differs")
            if loaded_accepted_metadata["run_cursor"] != cursor:
                raise DirectDescriptionError("accepted-step checkpoint cursor parse-back differs")
            if periodic or stage_end or bounded_stop or campaign_blocker or args.simulate_kill_after_checkpoint:
                boundary = "blocked" if campaign_blocker else "stage_end" if stage_end else "intra"
                checkpoint = out_dir / "checkpoints" / f"{stage['stage_id']}_{boundary}_global{state.step:06d}.npz"
                checkpoint_sha = save_stage_checkpoint(
                    checkpoint,
                    state,
                    stage_id=stage["stage_id"],
                    config=config,
                    telemetry=(row,),
                    run_cursor=cursor,
                    realized_archive={
                        "bytes": len(live_archive),
                        "sha256": hashlib.sha256(live_archive).hexdigest(),
                        "lane_programs_materialized": include_lanes,
                        "parameter_shadow": "live_resume_state",
                        "realized_parameter_count": int(np.count_nonzero(live_realized_theta)),
                    },
                )
                loaded, loaded_metadata = load_stage_checkpoint(checkpoint, config=config)
                if state.step != loaded.step or any(
                    not np.array_equal(getattr(state, field), getattr(loaded, field))
                    for field in ("theta", "ema", "first_moment", "second_moment")
                ):
                    raise DirectDescriptionError("full-run checkpoint immediate parse-back differs")
                if pose_finish_state is not None:
                    restored_pose_state = PoseFinishEngageStateV1.from_payload(
                        loaded_metadata["run_cursor"]["pose_finish_engage_state"],
                        config=pose_finish_config,
                    )
                    if restored_pose_state != pose_finish_state:
                        raise DirectDescriptionError("pose-finish engage checkpoint immediate parse-back differs")
                memory_receipt = _full_run_memory_receipt(
                    config=config,
                    monitor=monitor,
                    step_seconds=step_seconds,
                    train_batch=int(schedule["train_batch"]),
                )
                _atomic_json(runtime_memory_path, memory_receipt)
                if args.simulate_kill_after_checkpoint:
                    _atomic_json(
                        out_dir / "kill_after_full_run_checkpoint.json",
                        {
                            "schema": "ddm_joint_descent_full_run_kill_boundary.v1",
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha,
                            "checkpoint_cursor": loaded_metadata["run_cursor"],
                            "memory_receipt": str(runtime_memory_path),
                            "memory_receipt_sha256": _sha256_file(runtime_memory_path),
                            "next_action": "new process --full-run --resume-from this checkpoint",
                            "score_claim": False,
                        },
                    )
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(23)
            if stage_end:
                stage_index += 1
                stage_step = 0
            if bounded_stop or campaign_blocker:
                break

        memory_receipt = _full_run_memory_receipt(
            config=config,
            monitor=monitor,
            step_seconds=step_seconds,
            train_batch=int(schedule["train_batch"]),
        )
        _atomic_json(runtime_memory_path, memory_receipt)
    admission = bool(memory_receipt["admission"])
    final_verdict = verdict_history[-1] if verdict_history else None
    pose_finish_engaged = pose_finish_state is None or pose_finish_state.engaged
    final_target_green = False
    final_target_decision = "FINAL_TARGET_GATE_NOT_REACHED"
    if stage_index >= len(schedule["stages"]):
        final_target_green, final_target_decision = exact_final_target_gate(
            final_verdict=final_verdict,
            final_stage=schedule["stages"][-1],
        )
    if not admission:
        final_run_verdict = "REFUSE_FULL_RUN_MEMORY_PROJECTION"
    elif campaign_blocker is not None:
        final_run_verdict = campaign_blocker
    elif args.max_steps is not None and warm_start_component_safe and pose_finish_engaged:
        final_run_verdict = "READY_TO_FIRE_UNDER_STANDING_GO"
    elif args.max_steps is not None and warm_start_component_safe and not pose_finish_engaged:
        final_run_verdict = "BLOCKED_POSE_FINISH_CONDITIONING_HISTORY_INSUFFICIENT"
    elif args.max_steps is not None and warm_start_admitted:
        final_run_verdict = "BLOCKED_COMPONENT_OR_RESIDUAL_FIRE_GATE_AFTER_PURE_PRICED_ADMISSION"
    elif args.max_steps is not None and latest_stage_decision in {
        "REALIZED_STAGE_TARGET_MET",
        "REALIZED_STAGE_DESCENT_CONTINUE",
        "REALIZED_STAGE_SEG_FLAT_POSE_DESCENT_CONTINUE",
    }:
        final_run_verdict = "FULL_RUN_BOUNDED_REALIZED_DESCENT_GREEN"
    elif args.max_steps is not None:
        final_run_verdict = "FULL_RUN_BOUNDED_EXECUTION_GREEN_NO_N600_VERDICT"
    elif stage_index >= len(schedule["stages"]) and not pose_finish_engaged:
        final_run_verdict = "REFUSE_SCHEDULE_COMPLETE_WITHOUT_POSE_FINISH_ENGAGEMENT"
    elif stage_index >= len(schedule["stages"]):
        final_run_verdict = final_target_decision
    else:
        final_run_verdict = "FULL_RUN_ACTIVE_CHECKPOINTED"
    receipt = {
        "schema": "ddm_joint_descent_full_run_receipt.v1",
        **_base_receipt(config),
        "schedule": schedule,
        "source_archive": source_custody,
        "target_cache": cache_custody,
        "receiver_effective_parameter_groups": {name: len(indexes) for name, indexes in groups.items()},
        "receiver_effective_parameter_count": sum(len(indexes) for indexes in groups.values()),
        "j2_named_parameter_count_superseded": 706,
        "j2_overcount_reason": "aspect/rotation lift metadata and BEV/range seed fields lack current receiver wire coordinates",
        "kernels": {
            "mlx_default_device": str(mx.default_device()),
            "custom_grouped_backward_active": custom_active,
            "custom_grouped_backward_reason": custom_reason,
            "fused_r_numpy_fp32_parity": fused_r_parity,
        },
        "storage": storage,
        "governor": governor,
        "projection_used_for_governor_gib": projected_for_gate,
        "projection_source": projection_source,
        "admission_memory_receipt": {
            "path": str(admission_memory_path),
            "sha256": _sha256_file(admission_memory_path),
        },
        "runtime_memory_receipt": {
            "path": str(runtime_memory_path),
            "sha256": _sha256_file(runtime_memory_path),
        },
        "step_seconds": step_seconds,
        "elapsed_seconds": time.monotonic() - full_started,
        "global_step": state.step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "baseline_verdict": baseline_verdict,
        "final_stage_verdict": final_verdict,
        "telemetry_rows": len(run_telemetry),
        "memory_admission": admission,
        "bounded_verification": args.max_steps is not None,
        "execution_allowed_by_this_receipt": False,
        "verdict": final_run_verdict,
        "campaign_blocker": campaign_blocker,
        "latest_realized_stage_decision": latest_stage_decision,
        "warm_start_realized_admitted": warm_start_admitted,
        "warm_start_seg_admitted": warm_start_seg_admitted,
        "warm_start_component_safe_residual_admitted": warm_start_component_safe,
        "pose_finish_engage_state": (None if pose_finish_state is None else pose_finish_state.to_payload()),
        "pose_finish_engaged": pose_finish_engaged,
        "banked_r1_fallback_harvest_signal": (pose_finish_state is not None and not pose_finish_state.engaged),
        "banked_r1_fallback_is_comparator_only": config.execution_custody is not None,
        "final_target_gate": {
            "evaluated": stage_index >= len(schedule["stages"]),
            "green": final_target_green,
            "decision": final_target_decision,
        },
    }
    _atomic_json(out_dir / "full_run_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    non_promoting_stop = final_run_verdict.startswith(("BLOCKED_", "STOPPED_", "REFUSE_"))
    return 0 if admission and campaign_blocker is None and not non_promoting_stop else EXIT_REFUSE


def _full_run(args: argparse.Namespace, config: DirectDescriptionJointDescentTypedConfigV1) -> int:
    out_dir = Path(args.out_dir)
    _storage_receipt(out_dir)
    with _same_outdir_guard(out_dir, config):
        return _full_run_locked(args, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, default=DEFAULT_TICKET)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--memory-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--bounded-smoke", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    mode.add_argument("--worst-geometry-memory-bootstrap", action="store_true")
    mode.add_argument("--resume-proof", action="store_true")
    parser.add_argument("--bootstrap-measurement", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-step", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pair-id", type=int, choices=range(600), default=447)
    parser.add_argument("--verify-group-ownership", action="store_true")
    parser.add_argument("--simulate-kill-after-checkpoint", action="store_true")
    parser.add_argument("--stage-exit-on-stop", action="store_true")
    parser.add_argument("--force-geometry-escape-once", action="store_true")
    parser.add_argument("--measure-full-config-window", action="store_true")
    parser.add_argument("--measurement-train-batch", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args(argv)
    try:
        config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(args.ticket)
        event_schedule = (
            None
            if config.full_run_schedule is None
            else config.full_run_schedule.event_continuation
        )
        if event_schedule is not None and (args.bounded_smoke or args.full_run):
            raise DirectDescriptionError(
                "REFUSE_EVENT_CONTINUATION_EXECUTION_DISABLED_PENDING_MAIN_REVIEW"
            )
        if args.worst_geometry_memory_bootstrap:
            if (
                args.bootstrap_measurement
                or args.resume_from is not None
                or args.memory_receipt is not None
                or args.simulate_kill_after_checkpoint
                or args.max_steps is not None
                or args.stage_exit_on_stop
                or args.force_geometry_escape_once
                or args.measure_full_config_window
                or args.measurement_train_batch != 1
                or args.verify_group_ownership
                or args.pair_id != 447
                or args.stop_after_step != 1
            ):
                raise DirectDescriptionError("worst-geometry memory bootstrap accepts no campaign controls")
            return _worst_geometry_memory_bootstrap(args, config)
        if args.resume_proof:
            if (
                args.bootstrap_measurement
                or args.simulate_kill_after_checkpoint
                or args.max_steps is not None
                or args.stage_exit_on_stop
                or args.force_geometry_escape_once
                or args.measure_full_config_window
                or args.measurement_train_batch != 1
                or args.verify_group_ownership
                or args.pair_id != 447
                or args.stop_after_step != 1
            ):
                raise DirectDescriptionError("resume proof accepts only its checkpoint and memory receipt")
            with _same_outdir_guard(Path(args.out_dir), config):
                return _resume_proof(args, config)
        if args.dry_run:
            if (
                args.bootstrap_measurement
                or args.resume_from is not None
                or args.simulate_kill_after_checkpoint
                or args.max_steps is not None
                or args.stage_exit_on_stop
                or args.force_geometry_escape_once
                or args.measure_full_config_window
                or args.measurement_train_batch != 1
            ):
                raise DirectDescriptionError("dry-run cannot bootstrap or resume")
            memory_path = args.memory_receipt or args.out_dir / "memory_preflight.json"
            return _dry_run(config=config, memory_receipt_path=memory_path, out_dir=args.out_dir)
        if args.full_run:
            if args.verify_group_ownership or args.pair_id != 447 or args.stop_after_step != 1:
                raise DirectDescriptionError("bounded-smoke-only flags cannot alter --full-run")
            if args.max_steps is not None and args.max_steps <= 0:
                raise DirectDescriptionError("--max-steps must be positive")
            if args.force_geometry_escape_once and (
                args.max_steps != 1 or args.resume_from is not None
            ):
                raise DirectDescriptionError(
                    "--force-geometry-escape-once is restricted to a fresh one-step bounded full-run smoke"
                )
            if args.measure_full_config_window and config.full_run_schedule is not None:
                raise DirectDescriptionError("pre-seal measurement mode is forbidden for a resealed schedule")
            if not args.measure_full_config_window and args.measurement_train_batch != 1:
                raise DirectDescriptionError("--measurement-train-batch requires pre-seal measurement mode")
            return _full_run(args, config)
        if (
            args.max_steps is not None
            or args.stage_exit_on_stop
            or args.force_geometry_escape_once
            or args.measure_full_config_window
            or args.measurement_train_batch != 1
        ):
            raise DirectDescriptionError("full-run-only flags cannot alter bounded smoke")
        if not args.bootstrap_measurement and args.memory_receipt is None:
            args.memory_receipt = args.out_dir / "memory_preflight.json"
        return _bounded_smoke(args, config)
    except DirectDescriptionError as exc:
        message = str(exc)
        print(json.dumps({"verdict": "REFUSE", "reason": message, "score_claim": False}), file=sys.stderr)
        return EXIT_HASH if "hash" in message.lower() or "sha-256" in message.lower() else EXIT_REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
