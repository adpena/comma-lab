"""Determinism-preserving whole-pair process parallelism for JO1 fresh solve.

This is an r9 candidate, not a mutation of the sacred r8 process.  A worker owns
one complete pair and therefore preserves every PoseNet exploration batch's
shape, order, and intra-pair control flow.  The parent merges only complete pair
receipts in numeric pair order and retains the same full winners and certified
non-winner rebuild tuples as the pinned serial implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments import ddm_jo2_receiver_close as receiver_close
from experiments import ddm_jo3_joint_objective_entrypoint as entrypoint

AXIS = "[macOS-CPU offline wall-clock probe; no score authority]"


class PairParallelError(RuntimeError):
    """A pair worker or deterministic merge violated the r8 mechanism."""


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _record_timing(timings: MutableMapping[str, float] | None, name: str, elapsed: float) -> None:
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + float(elapsed)


def solve_pair_exact(
    *,
    surface: Any,
    modules: Any,
    posenet: Any,
    master: np.ndarray,
    baseline_pose6: np.ndarray,
    pair: int,
    root: Path,
    retention: entrypoint.CertifiedCandidateRetention,
    candidate_master: Mapping[str, Any],
    base_pose6: Mapping[str, Any],
    semantic_object_sha256: str,
    timings: MutableMapping[str, float] | None = None,
) -> dict[str, Any]:
    """Execute the pinned serial algorithm for exactly one whole pair.

    The body is intentionally line-for-line mechanism-equivalent to the pair
    body in ``ddm_jo2_receiver_close.solve_fresh_compensation``.  Scope is the
    only reduction; candidate generation, batch boundaries, tie breaks,
    retention, and same-batch winner replay are unchanged.
    """
    if not 0 <= pair < receiver_close.N:
        raise PairParallelError(f"pair is outside n600: {pair}")
    master = np.asarray(master, dtype=np.uint8)
    baseline_pose6 = np.asarray(baseline_pose6)
    if master.shape != (receiver_close.CAMERA_H, receiver_close.CAMERA_W, 3):
        raise PairParallelError(f"candidate master geometry differs: {master.shape}")
    if baseline_pose6.shape != (receiver_close.POSE_DIMS,):
        raise PairParallelError(f"base Pose6 geometry differs: {baseline_pose6.shape}")
    if baseline_pose6.dtype not in (np.float32, np.float64):
        raise PairParallelError(f"base Pose6 dtype differs: {baseline_pose6.dtype}")
    result_path = root / "RESULT.json"
    fingerprint = receiver_close.candidate_object_fingerprint(
        pair=pair,
        semantic_object_sha256=semantic_object_sha256,
        candidate_master=candidate_master,
        base_pose6=base_pose6,
    )
    if result_path.is_file():
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if row.get("candidate_object_fingerprint_sha256") != fingerprint:
            raise PairParallelError(f"stale compensation at pair {pair}")
        resumed_codes = np.asarray(row["final_codes"], dtype=np.int32)
        if resumed_codes.shape != (receiver_close.D,) or np.any(resumed_codes < -2048) or np.any(resumed_codes > 2047):
            raise PairParallelError(f"resumed carrier codes differ at pair {pair}")
        retention.verify_winner(row.get("winner_retention"))
        return row
    started = time.perf_counter()
    base_codes = surface.codes[pair].copy()
    event = receiver_close.evaluate_codes(
        surface=surface,
        modules=modules,
        posenet=posenet,
        codes=(base_codes,),
        master=master,
        pair=pair,
        stage_root=root / "stage_10_event",
        retention=retention,
    )[0]
    leak = event.astype(np.float64) - baseline_pose6.astype(np.float64)
    jacobian_codes = [base_codes.copy()]
    jacobian_probe_modes: list[str] = []
    for dimension in range(receiver_close.D):
        offsets, _denominator, mode = receiver_close.jacobian_probe_offsets(int(base_codes[dimension]))
        jacobian_probe_modes.append(mode)
        for delta in offsets:
            candidate = base_codes.copy()
            candidate[dimension] += delta
            if not -2048 <= candidate[dimension] <= 2047:
                raise PairParallelError("endpoint-safe probe left the int12 domain")
            jacobian_codes.append(candidate)
    jacobian_vectors = receiver_close.evaluate_codes(
        surface=surface,
        modules=modules,
        posenet=posenet,
        codes=tuple(jacobian_codes),
        master=master,
        pair=pair,
        stage_root=root / "stage_20_jacobian",
        retention=retention,
    )
    carrier_started = time.perf_counter()
    jacobian = np.empty((receiver_close.POSE_DIMS, receiver_close.D), dtype=np.float64)
    for dimension in range(receiver_close.D):
        _offsets, denominator, _mode = receiver_close.jacobian_probe_offsets(int(base_codes[dimension]))
        jacobian[:, dimension] = (
            jacobian_vectors[2 + 2 * dimension].astype(np.float64)
            - jacobian_vectors[1 + 2 * dimension].astype(np.float64)
        ) / denominator
    entrypoint.atomic_npy(root / "stage_20_jacobian/J_POSE0.float64.npy", jacobian)
    update, diagnostics = receiver_close._damped_solve(jacobian, -leak)
    centre = np.rint(base_codes.astype(np.float64) + update).astype(np.int32)
    centre = np.clip(centre, -2048, 2047)
    neighbourhood, active = receiver_close._nearby_candidates(base_codes, centre, jacobian, update)
    _record_timing(timings, "carrier_solve_seconds", time.perf_counter() - carrier_started)
    vectors = receiver_close.evaluate_codes(
        surface=surface,
        modules=modules,
        posenet=posenet,
        codes=neighbourhood,
        master=master,
        pair=pair,
        stage_root=root / "stage_30_integer_cube",
        retention=retention,
    )
    carrier_started = time.perf_counter()
    objectives = np.mean(np.square(vectors.astype(np.float64) - baseline_pose6[None]), axis=1)
    best = min(range(len(neighbourhood)), key=lambda index: (float(objectives[index]), index))
    current = np.asarray(neighbourhood[best], dtype=np.int32)
    objective = float(objectives[best])
    final_vector = np.asarray(vectors[best], dtype=np.float32)
    _record_timing(timings, "carrier_solve_seconds", time.perf_counter() - carrier_started)
    passes = 0
    while True:
        candidates = [current.copy()]
        for dimension in range(receiver_close.D):
            for delta in (-1, 1):
                candidate = current.copy()
                candidate[dimension] += delta
                if -2048 <= candidate[dimension] <= 2047:
                    candidates.append(candidate)
        vectors = receiver_close.evaluate_codes(
            surface=surface,
            modules=modules,
            posenet=posenet,
            codes=tuple(candidates),
            master=master,
            pair=pair,
            stage_root=root / f"stage_40_descent/pass_{passes:04d}",
            retention=retention,
        )
        carrier_started = time.perf_counter()
        objectives = np.mean(np.square(vectors.astype(np.float64) - baseline_pose6[None]), axis=1)
        best = min(range(len(candidates)), key=lambda index: (float(objectives[index]), index))
        value = float(objectives[best])
        final_vector = np.asarray(vectors[best], dtype=np.float32)
        passes += 1
        if not value < objective:
            _record_timing(timings, "carrier_solve_seconds", time.perf_counter() - carrier_started)
            break
        current = candidates[best]
        objective = value
        _record_timing(timings, "carrier_solve_seconds", time.perf_counter() - carrier_started)
    if not np.array_equal(np.asarray(candidates[best], dtype=np.int32), current):
        raise PairParallelError("converged winner differs from final explored row")
    winner_repeat = retention.recompute_selected_winner(
        root=root / "stage_50_winner_repeat_batch",
        pair=pair,
        base_codes=base_codes,
        candidate_codes=candidates,
        selected_index=best,
        master=master,
        surface=surface,
        modules=modules,
        posenet=posenet,
    )
    winner_pose = np.asarray(winner_repeat["pose_vector"], dtype=np.float32)
    if not np.array_equal(winner_pose, final_vector):
        raise PairParallelError(f"winner repeat Pose6 differs at pair {pair}")
    winner_retention = retention.retain_winner(
        root=root / "stage_50_winner_full",
        pair=pair,
        base_codes=base_codes,
        codes=np.asarray(winner_repeat["codes"], dtype=np.int32),
        slave_camera=np.asarray(winner_repeat["slave_camera"], dtype=np.uint8),
        pose_input=np.asarray(winner_repeat["pose_input"], dtype=np.uint8),
        pose_vector=winner_pose,
    )
    solved_codes = current
    row = {
        "schema": "ddm_jo2_fresh_schur_pair.v1",
        "pair": pair,
        "candidate_object_fingerprint_sha256": fingerprint,
        "semantic_object_sha256": semantic_object_sha256,
        "candidate_master": dict(candidate_master),
        "base_codes": base_codes.tolist(),
        "final_codes": solved_codes.tolist(),
        "final_code_delta": (solved_codes - base_codes).tolist(),
        "float_update": update.tolist(),
        "active_dimensions": list(active),
        "integer_cube_candidates": len(neighbourhood),
        "coordinate_descent_full_passes": passes,
        "final_objective_mse_to_base_pose6": objective,
        "final_pose6": winner_pose.tolist(),
        "jacobian": diagnostics,
        "jacobian_probe_modes": jacobian_probe_modes,
        "winner_retention": winner_retention,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
    }
    entrypoint.atomic_npy(root / "FINAL_CODES.int32.npy", solved_codes)
    entrypoint.atomic_npy(root / "FINAL_POSE6.float32.npy", winner_pose)
    entrypoint.atomic_json(result_path, row)
    _record_timing(timings, "pair_total_seconds", time.perf_counter() - started)
    return row


def run_worker_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    """Load the pinned substrate once and own every whole pair in one shard."""
    config = entrypoint.load_config(Path(str(task["compiled_config"])), str(task["expected_config_sha256"]))
    entrypoint.configure_determinism(config.seed)
    threads_per_worker = int(task["threads_per_worker"])
    torch.set_num_threads(threads_per_worker)
    _segnet, posenet, _patch = entrypoint.load_scorers(config)
    archive = Path(str(task["archive"]))
    runtime_root = Path(str(task["runtime_root"]))
    surface, modules = receiver_close.load_surface(archive, runtime_root)
    candidate_master = dict(task["candidate_master"])
    base_pose6 = dict(task["base_pose6"])
    masters = np.load(receiver_close.verify_record(candidate_master), mmap_mode="r", allow_pickle=False)
    baseline = np.load(receiver_close.verify_record(base_pose6), mmap_mode="r", allow_pickle=False)
    output = Path(str(task["output"]))
    retention = entrypoint.CertifiedCandidateRetention(
        solve_root=output,
        stage_id=str(task["stage_id"]),
        workload_config_sha256=str(task["workload_config_sha256"]),
        base_archive_sha256=entrypoint.file_record(archive)["sha256"],
    )
    results = []
    for pair_value in task["pairs"]:
        pair = int(pair_value)
        timings: dict[str, float] = {}
        row = solve_pair_exact(
            surface=surface,
            modules=modules,
            posenet=posenet,
            master=np.asarray(masters[pair]),
            baseline_pose6=np.asarray(baseline[pair]),
            pair=pair,
            root=output / f"pairs/pair_{pair:04d}",
            retention=retention,
            candidate_master=candidate_master,
            base_pose6=base_pose6,
            semantic_object_sha256=str(task["semantic_object_sha256"]),
            timings=timings,
        )
        results.append(
            {
                "pair": pair,
                "row": row,
                "timings": timings,
                "pid": os.getpid(),
                "threads_per_worker": threads_per_worker,
            }
        )
    return {
        "schema": "ddm_wc2_pair_worker_shard.v1",
        "worker_ordinal": int(task["worker_ordinal"]),
        "pid": os.getpid(),
        "threads_per_worker": threads_per_worker,
        "pairs": [int(value) for value in task["pairs"]],
        "results": results,
        "pair_worker_source_sha256": source_sha256(),
        "all_materialized_payloads_retained": True,
        "score_claim": False,
    }


def solve_fresh_compensation_parallel(
    *,
    candidate_master: Mapping[str, Any],
    base_pose6: Mapping[str, Any],
    semantic_object_sha256: str,
    output: Path,
    compiled_config: Path,
    expected_config_sha256: str,
    workload_config_sha256: str,
    stage_id: str,
    archive: Path,
    runtime_root: Path,
    workers: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    """Solve n600 with whole-pair workers and a deterministic numeric merge."""
    if workers < 1 or threads_per_worker < 1:
        raise PairParallelError("worker and thread counts must be positive")
    masters = np.load(receiver_close.verify_record(candidate_master), mmap_mode="r", allow_pickle=False)
    baseline = np.load(receiver_close.verify_record(base_pose6), mmap_mode="r", allow_pickle=False)
    if (
        masters.shape
        != (
            receiver_close.N,
            receiver_close.CAMERA_H,
            receiver_close.CAMERA_W,
            3,
        )
        or masters.dtype != np.uint8
    ):
        raise PairParallelError("candidate master field must be n600 camera uint8")
    if baseline.shape != (receiver_close.N, receiver_close.POSE_DIMS) or baseline.dtype not in (
        np.float32,
        np.float64,
    ):
        raise PairParallelError("base Pose6 table must be n600x6")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shards = [list(range(ordinal, receiver_close.N, workers)) for ordinal in range(workers)]
    processes: list[tuple[subprocess.Popen[str], Path, Path]] = []
    for ordinal, pairs in enumerate(shards):
        worker_root = output / "parallel_workers" / f"worker_{ordinal:02d}"
        task_path = worker_root / "TASK.json"
        result_path = worker_root / "RESULT.json"
        entrypoint.atomic_json(
            task_path,
            {
                "schema": "ddm_wc2_pair_worker_task.v1",
                "worker_ordinal": ordinal,
                "pairs": pairs,
                "compiled_config": str(compiled_config.resolve()),
                "expected_config_sha256": expected_config_sha256,
                "candidate_master": dict(candidate_master),
                "base_pose6": dict(base_pose6),
                "semantic_object_sha256": semantic_object_sha256,
                "output": str(output.resolve()),
                "stage_id": stage_id,
                "workload_config_sha256": workload_config_sha256,
                "archive": str(archive.resolve()),
                "runtime_root": str(runtime_root.resolve()),
                "threads_per_worker": threads_per_worker,
                "pair_worker_source_sha256": source_sha256(),
                "all_materialized_payloads_retained": True,
                "score_claim": False,
            },
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.ddm_wc2_jo1_pair_parallel",
                "worker-shard",
                "--task",
                str(task_path),
                "--result",
                str(result_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        processes.append((process, worker_root / "worker.log", result_path))
    worker_results = []
    worker_failures = []
    for process, log_path, result_path in processes:
        stdout, _ = process.communicate()
        entrypoint.atomic_bytes(log_path, stdout.encode())
        if process.returncode != 0 or not result_path.is_file():
            worker_failures.append(f"rc={process.returncode}; log={log_path}")
            continue
        shard = json.loads(result_path.read_text(encoding="utf-8"))
        if shard.get("pair_worker_source_sha256") != source_sha256():
            worker_failures.append(f"pair worker source SHA-256 differs: {result_path}")
            continue
        worker_results.extend(shard["results"])
    if worker_failures:
        raise PairParallelError("; ".join(worker_failures))
    worker_results.sort(key=lambda value: int(value["pair"]))
    rows = [value["row"] for value in worker_results]
    if [int(row["pair"]) for row in rows] != list(range(receiver_close.N)):
        raise PairParallelError("deterministic merge pair census differs")
    solved_codes = np.stack([np.asarray(row["final_codes"], dtype=np.int32) for row in rows])
    surface, modules = receiver_close.load_surface(archive, runtime_root)
    codes_record = entrypoint.atomic_npy(output / "candidate_codes.int32.npy", solved_codes)
    frame0_record = receiver_close.materialize_candidate_frame0(
        surface=surface, modules=modules, codes=solved_codes, output=output
    )
    retention = entrypoint.CertifiedCandidateRetention(
        solve_root=output,
        stage_id=stage_id,
        workload_config_sha256=workload_config_sha256,
        base_archive_sha256=entrypoint.file_record(archive)["sha256"],
    )
    for row in rows:
        retention.verify_winner(row.get("winner_retention"))
    retention_inventory = retention.finalize()
    timing_values = [
        float(value["timings"]["pair_total_seconds"])
        for value in worker_results
        if "pair_total_seconds" in value["timings"]
    ]
    execution = entrypoint.atomic_json(
        output / "PAIR_PARALLEL_EXECUTION.json",
        {
            "schema": "ddm_wc2_pair_parallel_execution.v1",
            "pair_worker_source_sha256": source_sha256(),
            "worker_count": workers,
            "threads_per_worker": threads_per_worker,
            "pair_merge_order": list(range(receiver_close.N)),
            "fresh_pair_timing_denominator": len(timing_values),
            "pair_total_seconds_median": statistics.median(timing_values) if timing_values else None,
            "wall_seconds": time.perf_counter() - started,
            "per_pair_batch_shape_preserved": True,
            "all_materialized_payloads_retained": True,
            "score_claim": False,
        },
    )
    result = {
        "schema": "ddm_jo2_fresh_schur_n600.v1",
        "status": "COMPLETE",
        "pair_denominator": receiver_close.N,
        "semantic_object_sha256": semantic_object_sha256,
        "candidate_master": dict(candidate_master),
        "base_pose6": dict(base_pose6),
        "candidate_codes": codes_record,
        "candidate_frame0": frame0_record,
        "changed_pairs": int(np.count_nonzero(np.any(solved_codes != surface.codes, axis=1))),
        "changed_coordinates": int(np.count_nonzero(solved_codes != surface.codes)),
        "fresh_per_candidate_asserted_in_code": True,
        "retention_inventory": retention_inventory,
        "pair_results": rows,
        "pair_parallel_execution": execution,
        "pair_worker_source_sha256": source_sha256(),
        "score_claim": False,
    }
    entrypoint.atomic_json(output / "FRESH_SCHUR_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker-shard")
    worker.add_argument("--task", required=True, type=Path)
    worker.add_argument("--result", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        task = json.loads(args.task.read_text(encoding="utf-8"))
        if task.get("pair_worker_source_sha256") != source_sha256():
            raise PairParallelError("task pair-worker source SHA-256 differs")
        value = run_worker_shard(task)
        entrypoint.atomic_json(args.result, value)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "COMPLETE", "result": str(args.result.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
