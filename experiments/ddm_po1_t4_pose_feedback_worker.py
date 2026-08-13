#!/usr/bin/env python3
"""Retained T4 PoseNet error-feedback instrument for one exact archive.

This worker is component-only.  It decodes one pinned archive, re-emits the
GT/candidate SegNet argmax fields, and evaluates frozen PoseNet first-six
vectors for GT plus two decoded repeats in the same T4 process.  It retains
every raw/scorer/vector payload and immutable stage checkpoint on the Modal
volume; it never promotes or scores an archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

try:
    from experiments.ddm_js1b_cuda_argmax_field_materializer_worker import (
        BATCH_SIZE,
        FIELD_DATA_BYTES,
        LOGIT_BYTES,
        N_PAIRS,
        RAW_BYTES,
        SEED,
        SEG_INPUT_BYTES,
        STORAGE_RESERVE_BYTES,
        UPSTREAM,
        WorkerError,
        atomic_json,
        checkpoint_once,
        decode_exact_receiver,
        extract_zip_once,
        file_record,
        load_posenet,
        load_segnet,
        require_record,
        score_argmax_field,
        score_pose_vectors,
    )
except ModuleNotFoundError:
    from ddm_js1b_cuda_argmax_field_materializer_worker import (  # type: ignore[no-redef]
        BATCH_SIZE,
        FIELD_DATA_BYTES,
        LOGIT_BYTES,
        N_PAIRS,
        RAW_BYTES,
        SEED,
        SEG_INPUT_BYTES,
        STORAGE_RESERVE_BYTES,
        UPSTREAM,
        WorkerError,
        atomic_json,
        checkpoint_once,
        decode_exact_receiver,
        extract_zip_once,
        file_record,
        load_posenet,
        load_segnet,
        require_record,
        score_argmax_field,
        score_pose_vectors,
    )

AXIS_POSE: Final = (
    "[contest-CUDA T4 frozen-PoseNet first6 vectors, n600, batch=16] COMPONENT-ONLY"
)
AXIS_SEG: Final = (
    "[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY"
)
POSE_INPUT_BYTES: Final = N_PAIRS * 12 * 384 * 512 * 4
POSE_OUTPUT_BYTES: Final = N_PAIRS * 12 * 4
POSE_VECTOR_BYTES: Final = N_PAIRS * 6 * 4
EXPECTED_RETAINED_PAYLOAD_BYTES: Final = (
    RAW_BYTES
    + RAW_BYTES
    + 2 * (SEG_INPUT_BYTES + LOGIT_BYTES + FIELD_DATA_BYTES)
    + 3 * (POSE_INPUT_BYTES + POSE_OUTPUT_BYTES + POSE_VECTOR_BYTES)
    + 2 * N_PAIRS * 8
)
NOISE_COMPARABLE_RATIO: Final = 0.5


def current_retained_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def storage_preflight(run_root: Path) -> dict[str, Any]:
    """Fail closed unless the retained volume can hold every materialized payload."""
    usage = shutil.disk_usage(run_root)
    already_retained = current_retained_bytes(run_root)
    remaining = max(0, EXPECTED_RETAINED_PAYLOAD_BYTES - already_retained)
    required = remaining + STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_po1_storage_preflight.v1",
        "tier": str(run_root),
        "free_bytes": usage.free,
        "already_retained_bytes": already_retained,
        "expected_total_retained_payload_bytes": EXPECTED_RETAINED_PAYLOAD_BYTES,
        "remaining_payload_bytes": remaining,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "block rather than delete; all raw and scorer payloads retained",
    }
    atomic_json(run_root / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise WorkerError(
            f"PO1 storage preflight failed: free={usage.free}, required={required}"
        )
    return result


def pose_feedback_metrics(
    gt: np.ndarray,
    decoded_first: np.ndarray,
    decoded_repeat: np.ndarray,
) -> dict[str, Any]:
    """Compute the exact vector MSE and the pre-registered F1 noise diagnostic."""
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (gt, decoded_first, decoded_repeat))
    if any(value.shape != (N_PAIRS, 6) for value in arrays):
        raise WorkerError("PO1 PoseNet vector arrays must each have shape (600, 6)")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise WorkerError("PO1 PoseNet vector arrays must be finite")
    gt32, first32, repeat32 = arrays
    error = first32 - gt32
    repeat_delta = repeat32 - first32
    error_squared = error * error
    repeat_squared = repeat_delta * repeat_delta
    pair_error_rms = np.sqrt(np.mean(error_squared, axis=1, dtype=np.float32))
    pair_repeat_rms = np.sqrt(np.mean(repeat_squared, axis=1, dtype=np.float32))
    comparable = (pair_error_rms > 0.0) & (
        pair_repeat_rms >= NOISE_COMPARABLE_RATIO * pair_error_rms
    )
    comparable_pairs = int(np.count_nonzero(comparable))
    return {
        "schema": "ddm_po1_pose_feedback_metrics.v1",
        "pairs": N_PAIRS,
        "components_per_pair": 6,
        "d_pose_decoded_first": float(np.mean(error_squared, dtype=np.float32)),
        "d_pose_decoded_repeat": float(
            np.mean((repeat32 - gt32) * (repeat32 - gt32), dtype=np.float32)
        ),
        "repeat_noise_mse": float(np.mean(repeat_squared, dtype=np.float32)),
        "pair_error_rms": pair_error_rms.astype(np.float64),
        "pair_repeat_noise_rms": pair_repeat_rms.astype(np.float64),
        "noise_comparable_ratio": NOISE_COMPARABLE_RATIO,
        "noise_comparable_pairs": comparable_pairs,
        "noise_comparable_fraction": comparable_pairs / N_PAIRS,
        "f1_instrument_floor_closed": comparable_pairs > N_PAIRS // 2,
        "f1_rule": (
            "close when repeat RMS is at least half decoded-vs-GT RMS for more than "
            "300 of 600 pairs"
        ),
    }


def seg_feedback_metrics(gt_path: Path, decoded_path: Path) -> dict[str, Any]:
    gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    decoded = np.load(decoded_path, mmap_mode="r", allow_pickle=False)
    if gt.shape != decoded.shape or gt.shape != (N_PAIRS, 384, 512):
        raise WorkerError("PO1 SegNet fields have unexpected geometry")
    flips = int(np.count_nonzero(gt != decoded))
    denominator = int(gt.size)
    return {
        "schema": "ddm_po1_seg_feedback_metrics.v1",
        "flips": flips,
        "denominator_pixels": denominator,
        "d_seg": flips / denominator,
        "gt_field": file_record(gt_path),
        "decoded_field": file_record(decoded_path),
    }


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    import timm
    import torch
    import torchvision

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    started = time.time()
    request_path = run_root / "inputs/REQUEST.json"
    request = json.loads(request_path.read_text())
    if resume_from != str(request["resume_from"]) or resume_from != str(request["run_id"]):
        raise WorkerError("PO1 resume token differs from the retained run identity")
    final_path = run_root / "FINAL_RESULT.json"
    if final_path.is_file():
        result = json.loads(final_path.read_text())
        if result.get("execution_status") != "COMPLETE":
            raise WorkerError("retained PO1 final result is not complete")
        return result

    storage = storage_preflight(run_root)
    for filename, record in request["inputs"].items():
        require_record(run_root / f"inputs/{filename}", record)
    checkpoint_once(
        run_root / "checkpoints/stage_00_inputs_and_storage.json",
        {
            "schema": "ddm_po1_stage_checkpoint.v1",
            "stage": "inputs_and_storage",
            "inputs": request["inputs"],
            "storage": storage,
            "resume_from": resume_from,
            "complete": True,
        },
    )

    runtime_root = run_root / "work/runtime/candidate"
    extract_zip_once(
        run_root / "inputs/candidate_runtime.zip",
        runtime_root,
        "RUNTIME_EXTRACTED.json",
    )
    checkpoint_once(
        run_root / "checkpoints/stage_05_runtime_extracted.json",
        {
            "schema": "ddm_po1_stage_checkpoint.v1",
            "stage": "runtime_extracted",
            "runtime_root": str(runtime_root),
            "complete": True,
        },
    )

    receiver = decode_exact_receiver(
        name="candidate",
        archive_path=run_root / "inputs/candidate_archive.zip",
        archive_record=request["candidate_archive"],
        runtime_root=runtime_root,
        run_root=run_root,
    )
    checkpoint_once(
        run_root / "checkpoints/stage_10_candidate_decoded.json",
        {
            "schema": "ddm_po1_stage_checkpoint.v1",
            "stage": "candidate_decoded",
            "receiver": receiver,
            "complete": True,
        },
    )

    upstream_snapshot_sha256 = compute_upstream_snapshot_sha256(
        UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    if not upstream_snapshot_sha256:
        raise WorkerError("remote canonical upstream snapshot is missing")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    raw_root = Path(receiver["raw"]["path"]).parent
    raw_record = receiver["raw"]

    segnet = load_segnet(device)
    seg_results = {
        "gt": score_argmax_field(
            source="gt",
            raw_root=None,
            raw_record=None,
            scorer=segnet,
            device=device,
            run_root=run_root,
        ),
        "candidate": score_argmax_field(
            source="candidate",
            raw_root=raw_root,
            raw_record=raw_record,
            scorer=segnet,
            device=device,
            run_root=run_root,
        ),
    }
    del segnet
    torch.cuda.empty_cache()
    checkpoint_once(
        run_root / "checkpoints/stage_20_segnet_complete.json",
        {
            "schema": "ddm_po1_stage_checkpoint.v1",
            "stage": "segnet_complete",
            "scorers": seg_results,
            "complete": True,
        },
    )

    posenet = load_posenet(device)
    pose_results = {
        "gt": score_pose_vectors(
            source="gt",
            raw_root=None,
            raw_record=None,
            scorer=posenet,
            device=device,
            run_root=run_root,
        ),
        "candidate_first": score_pose_vectors(
            source="candidate_first",
            raw_root=raw_root,
            raw_record=raw_record,
            scorer=posenet,
            device=device,
            run_root=run_root,
        ),
        "candidate_repeat": score_pose_vectors(
            source="candidate_repeat",
            raw_root=raw_root,
            raw_record=raw_record,
            scorer=posenet,
            device=device,
            run_root=run_root,
        ),
    }
    checkpoint_once(
        run_root / "checkpoints/stage_30_posenet_complete.json",
        {
            "schema": "ddm_po1_stage_checkpoint.v1",
            "stage": "posenet_complete",
            "scorers": pose_results,
            "complete": True,
        },
    )

    vector_paths = {
        name: Path(value["first6_vectors"]["path"])
        for name, value in pose_results.items()
    }
    metrics = pose_feedback_metrics(
        np.load(vector_paths["gt"], allow_pickle=False),
        np.load(vector_paths["candidate_first"], allow_pickle=False),
        np.load(vector_paths["candidate_repeat"], allow_pickle=False),
    )
    pair_error_path = run_root / "retained/pose_vectors/pair_error_rms_n600.npy"
    pair_noise_path = run_root / "retained/pose_vectors/pair_repeat_noise_rms_n600.npy"
    try:
        from experiments.ddm_js1b_cuda_argmax_field_materializer_worker import atomic_npy
    except ModuleNotFoundError:
        from ddm_js1b_cuda_argmax_field_materializer_worker import atomic_npy

    pair_error = metrics.pop("pair_error_rms")
    pair_noise = metrics.pop("pair_repeat_noise_rms")
    metrics["pair_error_rms"] = atomic_npy(pair_error_path, pair_error)
    metrics["pair_repeat_noise_rms"] = atomic_npy(pair_noise_path, pair_noise)
    seg_metrics = seg_feedback_metrics(
        Path(seg_results["gt"]["argmax"]["path"]),
        Path(seg_results["candidate"]["argmax"]["path"]),
    )

    final = {
        "schema": "ddm_po1_t4_pose_feedback_result.v1",
        "execution_status": "COMPLETE",
        "status": "F1_INSTRUMENT_FLOOR" if metrics["f1_instrument_floor_closed"] else "FEEDBACK_USABLE",
        "axis_pose": AXIS_POSE,
        "axis_seg": AXIS_SEG,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "candidate_archive": request["candidate_archive"],
        "receiver": receiver,
        "pose_scorers": pose_results,
        "seg_scorers": seg_results,
        "pose_feedback": metrics,
        "seg_feedback": seg_metrics,
        "retention": {
            "volume_run_root": str(run_root),
            "exact_receiver_raw": True,
            "all_gt_rgb_batches": True,
            "all_seg_inputs_logits_and_fields": True,
            "all_pose_inputs_full_outputs_and_first6_vectors": True,
            "decoded_repeat_same_job": True,
            "expected_payload_bytes_before_metadata": EXPECTED_RETAINED_PAYLOAD_BYTES,
        },
        "provenance": {
            "source_git_head": request["source_git_head"],
            "source_git_dirty_at_dispatch": request["source_git_dirty"],
            "source_git_status_sha256": request["source_git_status_sha256"],
            "dispatcher_source_sha256": request["dispatcher_source_sha256"],
            "worker_source_sha256": request["worker_source_sha256"],
            "js1b_worker_source_sha256": request["js1b_worker_source_sha256"],
            "upstream_snapshot_sha256": upstream_snapshot_sha256,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "timm_version": timm.__version__,
            "numpy_version": np.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "boundaries": {
            "measured": (
                "exact receiver raw, frozen PoseNet first-six vectors with same-job repeat, "
                "and frozen SegNet argmax fields on contest-CUDA T4"
            ),
            "not_measured": (
                "full contest score, contest-CPU, any coefficient update, or any new archive"
            ),
        },
    }
    checkpoint_once(final_path, final)
    checkpoint_once(run_root / "checkpoints/stage_40_final.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args(argv)
    result = run(args.run_root.resolve(), args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
