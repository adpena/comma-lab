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
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    BASELINE_DSEG,
    EVIDENCE_AXIS,
    MEMORY_RECEIPT_SCHEMA,
    POINTER,
    AdamStateV1,
    DirectDescriptionJointDescentMLXModule,
    DirectDescriptionJointDescentTypedConfigV1,
    classify_memory_preflight,
    classify_realized_stage_verdict,
    clipped_adam_step,
    compile_parameterized_archive,
    initial_adam_state,
    lift_v15_archive,
    load_stage_checkpoint,
    parameter_group_indices,
    realized_training_state,
    save_stage_checkpoint,
    template_camera_state,
    verify_trainable_group_ownership,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402

DEFAULT_TICKET = REPO / ".omx/research/configs/ddm_j3_366_joint_descent_witness_program_20260723.json"
POINTER_DSEG = 0.027470296224
STATIC_BOOTSTRAP_BOUND_GIB = 16.0
EXIT_REFUSE = 4
EXIT_HASH = 8


def _utc() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat()


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
    admitted, reason = classify_memory_preflight(
        float(row["projected_peak_gib"]), ceiling_gib=config.memory_ceiling_gib
    )
    if not admitted:
        raise DirectDescriptionError(reason)
    return row


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
    return {
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
        receiver = receive_carrier_compose_archive(archive)
        replay_pairs = (0, args.pair_id, 599)
        if not np.array_equal(
            receiver.render_camera_pairs(replay_pairs),
            receive_carrier_compose_archive(stage00).render_camera_pairs(replay_pairs),
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
        checkpoint_bit_exact = all(
            np.array_equal(getattr(state, field), getattr(loaded, field))
            for field in ("theta", "ema", "first_moment", "second_moment")
        ) and state.step == loaded.step
        if not checkpoint_bit_exact:
            raise DirectDescriptionError("checkpoint immediate parse-back is not bit-exact")
        mx.eval()
    elapsed = time.monotonic() - started

    measured_peak_gib = monitor.peak_rss_bytes / 1024**3
    projected_peak_gib = _memory_projection(measured_peak_gib)
    admission, reason = classify_memory_preflight(
        projected_peak_gib, ceiling_gib=config.memory_ceiling_gib
    )
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
        "verdict": (
            "BOUNDED_STEP_RESUME_GREEN" if admission else "REFUSE_REAL_CONSUMER_MEMORY_PROJECTION"
        ),
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
    receiver = receive_carrier_compose_archive(archive)
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
    return {
        "schema": "ddm_joint_descent_chunked_stage_verdict.v1",
        "num_pairs": 600,
        "batch_size": batch_size,
        "maximum_rgb_chunks_resident": 1,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
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
        "elapsed_seconds": time.monotonic() - started,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
    }


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
    return {
        "train_batch": schedule.train_batch,
        "learning_rate": schedule.learning_rate_quantum_fraction,
        "checkpoint_interval_steps": schedule.checkpoint_interval_steps,
        "warm_start_pair": schedule.warm_start_pair,
        "warm_start_steps": schedule.warm_start_steps,
        "plateau_verdicts": schedule.plateau_verdicts,
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
        else:
            state, metadata = load_stage_checkpoint(Path(args.resume_from), config=config)
            cursor = metadata.get("run_cursor", {})
            stage_index = int(cursor.get("stage_index", 0))
            stage_step = int(cursor.get("stage_step", 0))
            verdict_history = list(cursor.get("verdict_history", ()))
            baseline_verdict = cursor.get("baseline_verdict")
            if state.theta.size != model.parameter_count:
                raise DirectDescriptionError("full-run resume parameter count differs")
        if stage_index >= len(schedule["stages"]):
            raise DirectDescriptionError("full-run resume cursor is already complete")

        cpu_scorers: tuple[Any, Any] | None = None
        baseline_path = out_dir / "verdicts" / "stage00_baseline_n600.json"
        if baseline_verdict is None and baseline_path.is_file():
            baseline_verdict = json.loads(baseline_path.read_bytes())
            if (
                baseline_verdict.get("schema") != "ddm_joint_descent_chunked_stage_verdict.v1"
                or int(baseline_verdict.get("num_pairs", 0)) != 600
                or abs(float(baseline_verdict.get("d_seg", -1.0)) - BASELINE_DSEG) > 5.0e-10
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
            if abs(float(baseline_verdict["d_seg"]) - BASELINE_DSEG) > 5.0e-10:
                raise DirectDescriptionError(
                    "REFUSE_FULL_RUN_BASELINE_DSEG_CUSTODY_DRIFT: "
                    f"{baseline_verdict['d_seg']} != {BASELINE_DSEG}"
                )
            _write_immutable_json(baseline_path, baseline_verdict)

        stop_global = args.max_steps
        run_telemetry: list[dict[str, Any]] = []
        campaign_blocker: str | None = None
        latest_stage_decision: str | None = None
        while stage_index < len(schedule["stages"]):
            stage = schedule["stages"][stage_index]
            if stop_global is not None and state.step >= stop_global:
                break
            pair_start = (
                int(schedule["warm_start_pair"])
                if state.step < int(schedule["warm_start_steps"])
                else ((state.step - 1) * int(schedule["train_batch"])) % 600
            )
            pair_ids = tuple(
                (pair_start + offset) % 600 for offset in range(int(schedule["train_batch"]))
            )
            include_lanes = any(
                "lane_program" in prior["active_groups"]
                for prior in schedule["stages"][: stage_index + 1]
            )
            step_started = time.monotonic()
            base_camera, template_masks, basis, basis_indices, local_theta, current_archive = (
                realized_training_state(
                    lift,
                    state.theta,
                    pair_ids=pair_ids,
                    active_groups=stage["active_groups"],
                    include_lane_programs=include_lanes,
                )
            )
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
            )
            active = set().union(*(groups[name] for name in stage["active_groups"]))
            gradient[[index for index in range(len(gradient)) if index not in active]] = 0.0
            gradient_norm = float(np.linalg.norm(gradient.astype(np.float64)))
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise DirectDescriptionError("INSTANCE_BLOCKER_FULL_RUN_ACTIVE_GRADIENT_ZERO_OR_NONFINITE")
            accepted: tuple[AdamStateV1, dict[str, float], float] | None = None
            for multiplier in (1.0, 0.5, 0.25):
                learning_rate = float(schedule["learning_rate"]) * multiplier
                candidate = clipped_adam_step(
                    state,
                    gradient,
                    learning_rate=learning_rate,
                    grad_clip=config.grad_clip,
                    ema_decay=config.ema_decay,
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
                if metrics["d_seg"] < initial["d_seg"] and metrics["d_pose"] <= initial["d_pose"]:
                    accepted = candidate, metrics, learning_rate
                    break
            if accepted is None:
                raise DirectDescriptionError(
                    "INSTANCE_BLOCKER_FULL_RUN_NO_REALIZED_BATCH_DESCENT_AT_QUARTER_QUANTUM_OR_BELOW"
                )
            state, final, learning_rate = accepted
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
                "active_groups": list(stage["active_groups"]),
                "active_receiver_effective_coordinates": len(active),
                "realized_secant_coordinates_in_window": len(basis_indices),
                "loss_before": loss,
                "gradient_norm": gradient_norm,
                "learning_rate_uint8_quantum_fraction": learning_rate,
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
                if latest_stage_decision.startswith(("BLOCKED_", "REFUSE_")):
                    campaign_blocker = latest_stage_decision
                verdict = _write_immutable_json(
                    out_dir / "verdicts" / f"{stage['stage_id']}_step{state.step:06d}_n600.json",
                    verdict,
                    volatile_fields=("elapsed_seconds",),
                )
                verdict_history.append(verdict)
            recent = [row for row in verdict_history if row.get("stage_id") == stage["stage_id"]]
            plateau = len(recent) >= int(schedule["plateau_verdicts"]) and all(
                float(after["d_seg"]) >= float(before["d_seg"])
                and float(after["d_pose"]) >= float(before["d_pose"])
                for before, after in zip(
                    recent[-int(schedule["plateau_verdicts"]) : -1],
                    recent[-int(schedule["plateau_verdicts"]) + 1 :],
                    strict=True,
                )
            )
            stage_end = campaign_blocker is None and (target_met or stage_limit or plateau)
            cursor = {
                "stage_index": stage_index + (1 if stage_end else 0),
                "stage_step": 0 if stage_end else stage_step,
                "global_step": state.step,
                "verdict_history": verdict_history,
                "baseline_verdict": baseline_verdict,
                "stage_end_reason": (
                    campaign_blocker
                    or (
                        "realized_target_met"
                        if target_met
                        else "realized_plateau"
                        if plateau
                        else "maximum_steps"
                        if stage_limit
                        else None
                    )
                ),
                "campaign_blocker": campaign_blocker,
            }
            if periodic or stage_end or bounded_stop or campaign_blocker or args.simulate_kill_after_checkpoint:
                boundary = "blocked" if campaign_blocker else "stage_end" if stage_end else "intra"
                checkpoint = (
                    out_dir
                    / "checkpoints"
                    / f"{stage['stage_id']}_{boundary}_global{state.step:06d}.npz"
                )
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
    if not admission:
        final_run_verdict = "REFUSE_FULL_RUN_MEMORY_PROJECTION"
    elif campaign_blocker is not None:
        final_run_verdict = campaign_blocker
    elif args.max_steps is not None and latest_stage_decision in {
        "REALIZED_STAGE_TARGET_MET",
        "REALIZED_STAGE_DESCENT_CONTINUE",
    }:
        final_run_verdict = "FULL_RUN_BOUNDED_REALIZED_DESCENT_GREEN"
    elif args.max_steps is not None and latest_stage_decision == "REALIZED_STAGE_NO_TOTAL_DSEG_DESCENT":
        final_run_verdict = "BLOCKED_BOUNDED_NO_TOTAL_DSEG_DESCENT"
        campaign_blocker = final_run_verdict
    elif args.max_steps is not None:
        final_run_verdict = "FULL_RUN_BOUNDED_EXECUTION_GREEN_NO_N600_VERDICT"
    elif stage_index >= len(schedule["stages"]):
        final_run_verdict = "FULL_RUN_SCHEDULE_COMPLETE"
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
    }
    _atomic_json(out_dir / "full_run_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if admission and campaign_blocker is None else EXIT_REFUSE


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
    parser.add_argument("--bootstrap-measurement", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-step", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pair-id", type=int, choices=range(600), default=447)
    parser.add_argument("--verify-group-ownership", action="store_true")
    parser.add_argument("--simulate-kill-after-checkpoint", action="store_true")
    parser.add_argument("--stage-exit-on-stop", action="store_true")
    parser.add_argument("--measure-full-config-window", action="store_true")
    parser.add_argument("--measurement-train-batch", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args(argv)
    try:
        config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(args.ticket)
        if args.dry_run:
            if (
                args.bootstrap_measurement
                or args.resume_from is not None
                or args.simulate_kill_after_checkpoint
                or args.max_steps is not None
                or args.stage_exit_on_stop
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
            if args.measure_full_config_window and config.full_run_schedule is not None:
                raise DirectDescriptionError("pre-seal measurement mode is forbidden for a resealed schedule")
            if not args.measure_full_config_window and args.measurement_train_batch != 1:
                raise DirectDescriptionError("--measurement-train-batch requires pre-seal measurement mode")
            return _full_run(args, config)
        if (
            args.max_steps is not None
            or args.stage_exit_on_stop
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
