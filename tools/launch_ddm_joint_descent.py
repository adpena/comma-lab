#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Governed, bounded launcher for the DDM #366 joint-descent consumer.

This entry point intentionally cannot start the long campaign.  It closes the
J1 executable-consumer gates with an exact stage-00 replay, a real n600-cache
memory measurement, and a one-step-per-process checkpoint/resume smoke.  MAIN
review plus the existing governed campaign launcher remain required for fire.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
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
    EVIDENCE_AXIS,
    MEMORY_RECEIPT_SCHEMA,
    POINTER,
    AdamStateV1,
    DirectDescriptionJointDescentMLXModule,
    DirectDescriptionJointDescentTypedConfigV1,
    classify_memory_preflight,
    clipped_adam_step,
    initial_adam_state,
    lift_v15_archive,
    load_stage_checkpoint,
    save_stage_checkpoint,
    template_camera_state,
    verify_trainable_group_ownership,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402

DEFAULT_TICKET = REPO / ".omx/research/configs/ddm_j1_366_joint_descent_witness_program_20260723.json"
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
        "git_sha": os.popen("git rev-parse HEAD").read().strip(),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, default=DEFAULT_TICKET)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--memory-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--bounded-smoke", action="store_true")
    parser.add_argument("--bootstrap-measurement", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-step", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pair-id", type=int, choices=range(600), default=447)
    parser.add_argument("--verify-group-ownership", action="store_true")
    parser.add_argument("--simulate-kill-after-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(args.ticket)
        if args.dry_run:
            if args.bootstrap_measurement or args.resume_from is not None or args.simulate_kill_after_checkpoint:
                raise DirectDescriptionError("dry-run cannot bootstrap or resume")
            memory_path = args.memory_receipt or args.out_dir / "memory_preflight.json"
            return _dry_run(config=config, memory_receipt_path=memory_path, out_dir=args.out_dir)
        if not args.bootstrap_measurement and args.memory_receipt is None:
            args.memory_receipt = args.out_dir / "memory_preflight.json"
        return _bounded_smoke(args, config)
    except DirectDescriptionError as exc:
        message = str(exc)
        print(json.dumps({"verdict": "REFUSE", "reason": message, "score_claim": False}), file=sys.stderr)
        return EXIT_HASH if "hash" in message.lower() or "sha-256" in message.lower() else EXIT_REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
