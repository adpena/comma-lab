"""Bounded retained offline profile for the JO1 wall-clock critical path.

Only copied 1--3 pair inputs are executed.  Every materialized tensor, camera,
certificate, coder candidate, and timing receipt is retained under the caller's
SSD output root with SHA-256 and byte records.  The live r8 directory is read
for pinned inputs and reference receipts but is never opened for write.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments import ddm_jo2_receiver_close as receiver_close
from experiments import ddm_jo3_joint_objective_entrypoint as entrypoint
from experiments import ddm_wc2_jo1_pair_parallel as pair_parallel

DEFAULT_RUN_DIR = Path(
    "experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final"
)
DEFAULT_CONFIG = Path(".omx/research/ddm_jo5_determinism_cure_reseal_20260821/seal_r8/compiled_config.json")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_wc2_jo1_wallclock/profile_r1")
DEFAULT_CONFIG_SHA256 = "38d2f96dc755fd118eaccdac5985adaf6cff8e8beaea401669c8676600731b90"
MIN_FREE_BYTES = 8 * 1024**3
AXIS = "[macOS-CPU offline wall-clock probe; no score authority]"


class ProfileError(RuntimeError):
    """The bounded profile could not preserve the pinned mechanism or custody."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfileError(f"JSON root is not an object: {path}")
    return value


def _median(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def storage_preflight(output: Path) -> dict[str, Any]:
    probe = output if output.exists() else output.parent
    probe.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(probe)
    if usage.free < MIN_FREE_BYTES:
        raise ProfileError(f"SSD storage preflight failed: free={usage.free},required={MIN_FREE_BYTES}")
    output.mkdir(parents=True, exist_ok=True)
    return entrypoint.atomic_json(
        output / "STORAGE_PREFLIGHT.json",
        {
            "schema": "ddm_wc2_storage_preflight.v1",
            "path": str(output.resolve()),
            "free_bytes_before": usage.free,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "retention_policy": (
                "all copied inputs, solver candidates, winner bytes, coder candidates, and "
                "receipts are durable evidence on the first-priority SSD tier"
            ),
            "automatic_cleanup_path": (
                "no untracked scratch is created; atomic temporary siblings are replaced on "
                "success and retained evidence is certify-or-block, never auto-deleted"
            ),
            "score_claim": False,
        },
    )


def copy_pair_inputs(
    *,
    config: Any,
    run_dir: Path,
    pair: int,
    output: Path,
) -> dict[str, Any]:
    live_root = run_dir / "stages/01_target_birth/fresh_schur_attempt_0000"
    reference_path = live_root / f"pairs/pair_{pair:04d}/RESULT.json"
    if not reference_path.is_file():
        raise ProfileError(f"live reference pair is incomplete: {pair}")
    reference = _read_json(reference_path)
    master_cursor = _read_json(run_dir / "stages/01_target_birth/retained/MATERIALIZE_CURSOR.json")
    frame_cursor = _read_json(run_dir / "retained/FX5_FRAME0_CURSOR.json")
    master_source = Path(master_cursor["candidate_master_path"])
    frame0_source = Path(frame_cursor["retained_field"])
    master = np.load(master_source, mmap_mode="r", allow_pickle=False)
    frame0 = np.load(frame0_source, mmap_mode="r", allow_pickle=False)
    arrays = entrypoint.open_inputs(config)
    pair_root = output / "copied_inputs" / f"pair_{pair:04d}"
    payloads = {
        "candidate_master": entrypoint.atomic_npy(
            pair_root / "candidate_master.uint8.npy", np.asarray(master[pair], dtype=np.uint8)
        ),
        "base_frame0": entrypoint.atomic_npy(
            pair_root / "base_frame0.uint8.npy", np.asarray(frame0[pair], dtype=np.uint8)
        ),
        "base_pose6": entrypoint.atomic_npy(
            pair_root / "base_pose6.float32.npy",
            np.asarray(arrays["base_pose6"][pair], dtype=np.float32),
        ),
        "tokens": entrypoint.atomic_npy(
            pair_root / "tokens.uint8.npy", np.asarray(arrays["tokens"][pair], dtype=np.uint8)
        ),
        "target": entrypoint.atomic_npy(
            pair_root / "target.uint8.npy", np.asarray(arrays["target"][pair], dtype=np.uint8)
        ),
        "base_argmax": entrypoint.atomic_npy(
            pair_root / "base_argmax.uint8.npy",
            np.asarray(arrays["base_argmax"][pair], dtype=np.uint8),
        ),
        "pose_target": entrypoint.atomic_npy(
            pair_root / "pose_target.float32.npy",
            np.asarray(arrays["pose_target"][pair], dtype=np.float32),
        ),
        "live_final_codes": entrypoint.atomic_npy(
            pair_root / "live_final_codes.int32.npy",
            np.asarray(reference["final_codes"], dtype=np.int32),
        ),
        "live_final_pose6": entrypoint.atomic_npy(
            pair_root / "live_final_pose6.float32.npy",
            np.asarray(reference["final_pose6"], dtype=np.float32),
        ),
    }
    result = {
        "schema": "ddm_wc2_copied_pair_inputs.v1",
        "pair": pair,
        "payloads": payloads,
        "live_reference": entrypoint.file_record(reference_path),
        "source_candidate_master": dict(reference["candidate_master"]),
        "source_base_pose6": dict(config.inputs.fx5_base_pose6.model_dump()),
        "semantic_object_sha256": reference["semantic_object_sha256"],
        "source_object_sha256": config.inputs.rc2_archive.source_object_sha256,
        "copy_scope": "one pair only; no live payload was modified",
        "all_materialized_payloads_retained": True,
        "score_claim": False,
    }
    entrypoint.atomic_json(pair_root / "INPUT_MANIFEST.json", result)
    return result


@contextmanager
def timed_leaves(timings: MutableMapping[str, float]) -> Any:
    originals: list[tuple[Any, str, Callable[..., Any]]] = []

    def install(module: Any, name: str, bucket: str) -> None:
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[bucket] = timings.get(bucket, 0.0) + time.perf_counter() - started

        setattr(module, name, wrapper)

    install(receiver_close, "pose_vectors", "posenet_forward_seconds")
    install(receiver_close, "render_frame0", "carrier_render_seconds")
    for module, names in (
        (receiver_close, ("atomic_bytes", "atomic_json", "atomic_npy")),
        (entrypoint, ("atomic_bytes", "atomic_json", "atomic_npy", "atomic_compact_json")),
    ):
        for name in names:
            install(module, name, "retention_cert_io_seconds")
    install(entrypoint, "raw_array_record", "retention_cert_io_seconds")
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def run_pair_repeat(
    *,
    config: Any,
    copied: Mapping[str, Any],
    output: Path,
    threads: int,
    surface: Any,
    modules: Any,
    posenet: Any,
    require_live_identity: bool,
) -> dict[str, Any]:
    pair = int(copied["pair"])
    torch.set_num_threads(threads)
    timings: dict[str, float] = {}
    retention = entrypoint.CertifiedCandidateRetention(
        solve_root=output,
        stage_id="01_target_birth_wc2_offline_copy",
        workload_config_sha256=str(config.workload_config_sha256),
        base_archive_sha256=config.inputs.rc2_archive.sha256,
    )
    with timed_leaves(timings):
        row = pair_parallel.solve_pair_exact(
            surface=surface,
            modules=modules,
            posenet=posenet,
            master=np.load(copied["payloads"]["candidate_master"]["path"], allow_pickle=False),
            baseline_pose6=np.load(copied["payloads"]["base_pose6"]["path"], allow_pickle=False),
            pair=pair,
            root=output / f"pairs/pair_{pair:04d}",
            retention=retention,
            candidate_master=copied["source_candidate_master"],
            base_pose6=copied["source_base_pose6"],
            semantic_object_sha256=str(copied["semantic_object_sha256"]),
            timings=timings,
        )
    total = float(timings["pair_total_seconds"])
    named = sum(
        float(timings.get(name, 0.0))
        for name in (
            "posenet_forward_seconds",
            "carrier_solve_seconds",
            "retention_cert_io_seconds",
        )
    )
    timings["schur_compensation_other_seconds"] = max(0.0, total - named)
    live_codes = np.load(copied["payloads"]["live_final_codes"]["path"], allow_pickle=False)
    live_pose = np.load(copied["payloads"]["live_final_pose6"]["path"], allow_pickle=False)
    codes_equal = np.array_equal(np.asarray(row["final_codes"], dtype=np.int32), live_codes)
    pose_equal = np.array_equal(np.asarray(row["final_pose6"], dtype=np.float32), live_pose)
    result = {
        "schema": "ddm_wc2_offline_pair_profile.v1",
        "pair": pair,
        "threads": threads,
        "timings": timings,
        "final_codes_equal_live_r8": codes_equal,
        "final_pose6_equal_live_r8": pose_equal,
        "live_identity_required": require_live_identity,
        "pair_worker_source_sha256": pair_parallel.source_sha256(),
        "per_pair_batch_integrity_preserved": True,
        "all_materialized_payloads_retained": True,
        "axis": AXIS,
        "score_claim": False,
    }
    entrypoint.atomic_json(output / "PROFILE_RESULT.json", result)
    if require_live_identity and (not codes_equal or not pose_equal):
        raise ProfileError(f"offline pair {pair} differs from live r8: codes={codes_equal},pose={pose_equal}")
    return result


def profile_training_step(
    *,
    config: Any,
    copied: Mapping[str, Any],
    output: Path,
    semantic: Any,
    segnet: Any,
    posenet: Any,
    repeats: int,
    threads: int,
) -> list[dict[str, Any]]:
    pair = int(copied["pair"])
    payloads = copied["payloads"]
    token = torch.from_numpy(np.load(payloads["tokens"]["path"], allow_pickle=False))[None].long()
    target = torch.from_numpy(np.load(payloads["target"]["path"], allow_pickle=False))[None].long()
    base_argmax = torch.from_numpy(np.load(payloads["base_argmax"]["path"], allow_pickle=False))[None].long()
    pose_target = torch.from_numpy(np.load(payloads["pose_target"]["path"], allow_pickle=False))[None].float()
    first = (
        torch.from_numpy(np.load(payloads["base_frame0"]["path"], allow_pickle=False)).permute(2, 0, 1)[None].float()
    )
    results = []
    for repeat in range(repeats):
        entrypoint.configure_determinism(config.seed)
        torch.set_num_threads(threads)
        model = entrypoint.worker.HybridOutputResidual(
            config.actuation.hidden_channels, float(config.actuation.max_rgb_delta.value)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.stages[0].learning_rate.value)
        optimizer.zero_grad(set_to_none=True)
        timing: dict[str, float] = {}
        total_started = time.perf_counter()
        started = time.perf_counter()
        pre_r, camera, seg_input = entrypoint.render_training_pair(
            semantic=semantic, model=model, tokens=token, pair=pair
        )
        timing["render_residual_r_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        seg_logits = segnet(seg_input)
        timing["segnet_forward_seconds"] = time.perf_counter() - started
        pair_camera = torch.stack((first[0], camera[0]), dim=0)[None]
        started = time.perf_counter()
        pose_input = posenet.preprocess_input(pair_camera)
        pose6 = posenet(pose_input)["pose"][..., :6]
        timing["posenet_forward_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        loss, metrics = entrypoint.worker.joint_inner_objective(
            seg_logits=seg_logits,
            target=target,
            retained_base_argmax=base_argmax,
            pose6_candidate=pose6,
            pose6_target=pose_target,
            rate_proxy=entrypoint.rate_proxy(model),
            duals=entrypoint.worker.DualState(),
            stage=config.stages[0],
        )
        timing["objective_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        timing["backward_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        optimizer.step()
        timing["optimizer_seconds"] = time.perf_counter() - started
        gradient_norm = float(
            torch.sqrt(
                sum(value.grad.detach().square().sum() for value in model.parameters() if value.grad is not None)
            )
        )
        retained_metrics = dict(metrics)
        retained_metrics["gradient_norm"] = torch.tensor(gradient_norm)
        started = time.perf_counter()
        retained = entrypoint.retain_training_step(
            output / f"repeat_{repeat:02d}",
            pair=pair,
            tokens=token,
            pre_r=pre_r,
            camera=camera,
            seg_input=seg_input,
            seg_logits=seg_logits,
            pose_input=pose_input,
            pose6=pose6,
            target=target,
            base_argmax=base_argmax,
            metrics=retained_metrics,
        )
        timing["retention_io_seconds"] = time.perf_counter() - started
        timing["total_seconds"] = time.perf_counter() - total_started
        result = {
            "repeat": repeat,
            "pair": pair,
            "threads": threads,
            "timings": timing,
            "gradient_norm": gradient_norm,
            "retained": retained,
            "axis": AXIS,
            "all_materialized_payloads_retained": True,
            "score_claim": False,
        }
        entrypoint.atomic_json(output / f"repeat_{repeat:02d}/TIMING.json", result)
        results.append(result)
    return results


def profile_coder_race(
    *,
    config: Any,
    run_dir: Path,
    output: Path,
    surface: Any,
    modules: Any,
) -> dict[str, Any]:
    residual_path = run_dir / "stages/01_target_birth/retained/residual.j2s1"
    if not residual_path.is_file():
        raise ProfileError("target-birth retained residual is absent")
    payload = residual_path.read_bytes()
    entrypoint.residual_runtime.decode_residual_state(payload)
    semantic_body = entrypoint.residual_runtime.pack_semantic_blob(surface.parts.semantic_blob, payload)
    carrier_body = receiver_close.encode_carrier_body(surface.codes, surface, modules)
    retained = {
        "residual_payload": entrypoint.atomic_bytes(output / "retained/residual.j2s1", payload),
        "semantic_body": entrypoint.atomic_bytes(output / "retained/semantic.j2r1", semantic_body),
        "carrier_body": entrypoint.atomic_bytes(output / "retained/carrier.raw", carrier_body),
        "candidate_codes": entrypoint.atomic_npy(output / "retained/candidate_codes.int32.npy", surface.codes),
    }
    rows = []
    best: tuple[int, int, int, bytes] | None = None
    started = time.perf_counter()
    for semantic_quality in range(12):
        semantic_stream = receiver_close.brotli.compress(semantic_body, quality=semantic_quality, lgwin=24)
        if receiver_close.brotli.decompress(semantic_stream) != semantic_body:
            raise ProfileError("semantic Brotli round-trip differs")
        for carrier_quality in range(12):
            carrier_stream = receiver_close.brotli.compress(carrier_body, quality=carrier_quality, lgwin=24)
            if receiver_close.brotli.decompress(carrier_stream) != carrier_body:
                raise ProfileError("carrier Brotli round-trip differs")
            member = (
                receiver_close.rx1.RX1_HEADER.pack(
                    receiver_close.rx1.RX1_MAGIC,
                    receiver_close.rx1.RX1_VERSION,
                    int(surface.outer["codec"]),
                    int(surface.outer["table_mode"]),
                    0,
                    len(surface.outer["hpac_stream"]),
                    len(semantic_stream),
                    len(carrier_stream),
                )
                + surface.outer["hpac_stream"]
                + semantic_stream
                + carrier_stream
                + surface.outer["tail"]
            )
            archive_payload = receiver_close.rx1.deterministic_zip(member)
            root = output / "retained/rate_race" / (f"sq{semantic_quality:02d}_cq{carrier_quality:02d}")
            payloads = {
                "semantic_stream": entrypoint.atomic_bytes(root / "semantic.br", semantic_stream),
                "carrier_stream": entrypoint.atomic_bytes(root / "carrier.br", carrier_stream),
                "member_p": entrypoint.atomic_bytes(root / "p", member),
                "archive": entrypoint.atomic_bytes(root / "archive.zip", archive_payload),
            }
            row = {
                "semantic_quality": semantic_quality,
                "carrier_quality": carrier_quality,
                "archive_bytes": len(archive_payload),
                "payloads": payloads,
            }
            entrypoint.atomic_json(root / "RESULT.json", row)
            rows.append(row)
            key = (len(archive_payload), semantic_quality, carrier_quality)
            if best is None or key < best[:3]:
                best = (*key, archive_payload)
    elapsed = time.perf_counter() - started
    if best is None:
        raise ProfileError("coder race produced no candidate")
    archive_bytes, semantic_quality, carrier_quality, archive_payload = best
    selected = entrypoint.atomic_bytes(output / "selected/archive.zip", archive_payload)
    result = {
        "schema": "ddm_wc2_real_coder_race_profile.v1",
        "control_object": (
            "current target-birth residual plus the exact fx5 base carrier codes; this is a "
            "coder-cost control, not a completed fresh-solve package"
        ),
        "candidate_denominator": len(rows),
        "selection_mode": "minimum exact archive bytes; lower qualities break ties",
        "selected_semantic_quality": semantic_quality,
        "selected_carrier_quality": carrier_quality,
        "selected_archive_bytes": archive_bytes,
        "wall_seconds": elapsed,
        "retained": retained,
        "selected_archive": selected,
        "all_materialized_payloads_retained": True,
        "axis": AXIS,
        "score_claim": False,
    }
    entrypoint.atomic_json(output / "CODER_PROFILE.json", result)
    return result


def _parallel_task(task: Mapping[str, Any]) -> dict[str, Any]:
    config_path = Path(str(task["config_path"]))
    config = entrypoint.load_config(config_path, str(task["config_sha256"]))
    entrypoint.configure_determinism(config.seed)
    torch.set_num_threads(int(task["threads"]))
    _segnet, posenet, _patch = entrypoint.load_scorers(config)  # SCORER_LOADER_ORDER_OK: jo3 entrypoint wrapper returns (segnet, posenet, receipt); unpack matches its verified order
    surface, modules = receiver_close.load_surface(
        Path(config.inputs.rc2_archive.path), Path(config.inputs.rc2_runtime.path)
    )
    copied = _read_json(Path(str(task["input_manifest"])))
    return run_pair_repeat(
        config=config,
        copied=copied,
        output=Path(str(task["output"])),
        threads=int(task["threads"]),
        surface=surface,
        modules=modules,
        posenet=posenet,
        require_live_identity=bool(task["require_live_identity"]),
    )


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    storage = storage_preflight(output)
    config_path = args.compiled_config.resolve()
    config = entrypoint.load_config(config_path, args.expected_config_sha256)
    entrypoint.configure_determinism(config.seed)
    semantic, surface, modules = entrypoint.load_semantic(config)
    segnet, posenet, patch = entrypoint.load_scorers(config)  # SCORER_LOADER_ORDER_OK: jo3 entrypoint wrapper returns (segnet, posenet, receipt); unpack matches its verified order
    patch_record = entrypoint.atomic_json(output / "YUV6_PATCH_RECEIPT.json", patch)
    pairs = sorted(set(args.pairs))
    if not 1 <= len(pairs) <= 3:
        raise ProfileError("offline profile requires 1--3 distinct pairs")
    copied = [
        copy_pair_inputs(config=config, run_dir=args.run_dir.resolve(), pair=pair, output=output) for pair in pairs
    ]
    serial_results = []
    for threads in args.thread_counts:
        repeats = args.repeats if threads == args.reference_threads else 1
        for repeat in range(repeats):
            for value in copied:
                serial_results.append(
                    run_pair_repeat(
                        config=config,
                        copied=value,
                        output=output
                        / "serial"
                        / f"threads_{threads:02d}"
                        / f"repeat_{repeat:02d}"
                        / f"pair_{int(value['pair']):04d}",
                        threads=threads,
                        surface=surface,
                        modules=modules,
                        posenet=posenet,
                        require_live_identity=threads == args.reference_threads,
                    )
                )
    parallel_threads = args.parallel_threads
    tasks = [
        {
            "config_path": str(config_path),
            "config_sha256": args.expected_config_sha256,
            "threads": parallel_threads,
            "require_live_identity": parallel_threads == args.reference_threads,
            "input_manifest": str(output / "copied_inputs" / f"pair_{int(value['pair']):04d}/INPUT_MANIFEST.json"),
            "output": str(output / "parallel" / f"pair_{int(value['pair']):04d}"),
        }
        for value in copied
    ]
    parallel_started = time.perf_counter()
    processes: list[tuple[subprocess.Popen[str], Path, Path]] = []
    for ordinal, task in enumerate(tasks):
        worker_root = output / "parallel" / f"worker_{ordinal:02d}"
        task_path = worker_root / "TASK.json"
        result_path = worker_root / "RESULT.json"
        entrypoint.atomic_json(task_path, task)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.ddm_wc2_jo1_step_profile",
                "--worker-task",
                str(task_path),
                "--worker-result",
                str(result_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        processes.append((process, worker_root / "worker.log", result_path))
    parallel_results = []
    parallel_failures = []
    for process, log_path, result_path in processes:
        stdout, _ = process.communicate()
        entrypoint.atomic_bytes(log_path, stdout.encode())
        if process.returncode != 0 or not result_path.is_file():
            parallel_failures.append(f"rc={process.returncode}; log={log_path}")
            continue
        row = _read_json(result_path)
        if row.get("pair_worker_source_sha256") != pair_parallel.source_sha256():
            parallel_failures.append(f"pair worker source SHA-256 differs: {result_path}")
            continue
        parallel_results.append(row)
    if parallel_failures:
        raise ProfileError("offline subprocess pair worker failed: " + "; ".join(parallel_failures))
    parallel_wall = time.perf_counter() - parallel_started
    training_results = profile_training_step(
        config=config,
        copied=copied[0],
        output=output / "training_step",
        semantic=semantic,
        segnet=segnet,
        posenet=posenet,
        repeats=args.training_repeats,
        threads=args.reference_threads,
    )
    coder = profile_coder_race(
        config=config,
        run_dir=args.run_dir.resolve(),
        output=output / "coder_race",
        surface=surface,
        modules=modules,
    )
    source_sha256 = pair_parallel.source_sha256()
    if any(row.get("pair_worker_source_sha256") != source_sha256 for row in serial_results):
        raise ProfileError("serial profile pair-worker source changed during execution")
    reference_rows = [row for row in serial_results if int(row["threads"]) == args.reference_threads]
    thread_summary = {}
    for threads in args.thread_counts:
        rows = [row for row in serial_results if int(row["threads"]) == threads]
        thread_summary[str(threads)] = {
            "pair_total_seconds_median": _median([row["timings"] for row in rows], "pair_total_seconds"),
            "sample_denominator": len(rows),
            "byte_identity_denominator": sum(
                bool(row["final_codes_equal_live_r8"] and row["final_pose6_equal_live_r8"]) for row in rows
            ),
        }
    serial_parallel_baseline = sum(
        float(row["timings"]["pair_total_seconds"]) for row in serial_results if int(row["threads"]) == parallel_threads
    )
    if args.repeats > 1 and parallel_threads == args.reference_threads:
        serial_parallel_baseline /= args.repeats
    result = {
        "schema": "ddm_wc2_jo1_step_cost_profile.v1",
        "status": "COMPLETE",
        "axis": AXIS,
        "pairs": pairs,
        "reference_threads": args.reference_threads,
        "reference_repeat_denominator": args.repeats,
        "reference_pair_profile_median_seconds": {
            name: _median([row["timings"] for row in reference_rows], name)
            for name in (
                "pair_total_seconds",
                "posenet_forward_seconds",
                "carrier_solve_seconds",
                "retention_cert_io_seconds",
                "schur_compensation_other_seconds",
            )
        },
        "training_step_profile_median_seconds": {
            name: _median([row["timings"] for row in training_results], name) for name in training_results[0]["timings"]
        },
        "thread_tuning": thread_summary,
        "pair_process_parallel": {
            "workers": len(tasks),
            "threads_per_worker": parallel_threads,
            "wall_seconds_including_spawn_and_model_load": parallel_wall,
            "serial_same-thread_pair_seconds": serial_parallel_baseline,
            "gross_speedup_including_spawn": (serial_parallel_baseline / parallel_wall if parallel_wall else None),
            "byte_identity_denominator": sum(
                bool(row["final_codes_equal_live_r8"] and row["final_pose6_equal_live_r8"]) for row in parallel_results
            ),
            "pair_denominator": len(parallel_results),
            "merge_order": pairs,
        },
        "coder_race": coder,
        "storage_preflight": storage,
        "yuv6_patch_receipt": patch_record,
        "pair_worker_source": entrypoint.file_record(Path(pair_parallel.__file__)),
        "profile_source": entrypoint.file_record(Path(__file__)),
        "all_materialized_payloads_retained": True,
        "live_run_written": False,
        "score_claim": False,
        "frontier_moved": False,
    }
    entrypoint.atomic_json(output / "STEP_COST_PROFILE.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    result.add_argument("--compiled-config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--expected-config-sha256", default=DEFAULT_CONFIG_SHA256)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--pairs", type=int, nargs="+", default=[0, 184, 369])
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--training-repeats", type=int, default=3)
    result.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4])
    result.add_argument("--reference-threads", type=int, default=4)
    result.add_argument("--parallel-threads", type=int, default=2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.worker_task is not None:
            if args.worker_result is None:
                raise ProfileError("--worker-result is required with --worker-task")
            result = _parallel_task(_read_json(args.worker_task))
            entrypoint.atomic_json(args.worker_result, result)
        else:
            result = run_profile(args)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        json.JSONDecodeError,
        ProfileError,
        pair_parallel.PairParallelError,
    ) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
