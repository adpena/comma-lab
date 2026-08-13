#!/usr/bin/env python3
"""Retain one hash-pinned CUDA decode and its frozen-scorer measurements.

This worker deliberately does not adjudicate.  Its legacy mode runs the exact
RE1T SegNet-only measurement byte-for-byte as before.  The explicitly sealed
dual-axis mode additionally retains official PoseNet first-six vectors for GT
and two candidate passes from the same decoded payload.  All complete-S
arithmetic remains local.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Final

# The scorer import otherwise writes upstream/__pycache__, making the final
# snapshot verifier reject the worker's own residue (the SA1 20260813h defect).
sys.dont_write_bytecode = True

import numpy as np

try:
    from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1b
except ModuleNotFoundError:
    import ddm_js1b_cuda_argmax_field_materializer_worker as js1b  # type: ignore[no-redef]

AXIS: Final = "[contest-CUDA T4 frozen-SegNet argmax field, n600, batch=16] COMPONENT-ONLY"
AXIS_POSE: Final = (
    "[contest-CUDA T4 frozen-PoseNet first6 vectors, n600, batch=16] COMPONENT-ONLY"
)
PRIOR_RUN: Final = Path("/ddm_js1b_retained/ddm_js1b_20260813b")
GT_FIELD: Final = PRIOR_RUN / "retained/fields/gt_argmax_n600.npy"
BASE_FIELD: Final = PRIOR_RUN / "retained/fields/cp135_base_argmax_n600.npy"
GT_FIELD_RECORD: Final = {
    "bytes": 117_964_928,
    "sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
}
BASE_FIELD_RECORD: Final = {
    "bytes": 117_964_928,
    "sha256": "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727",
}
BASE_FLIPS: Final = 34_970
EXPECTED_RETAINED_PAYLOAD_BYTES: Final = (
    js1b.RAW_BYTES + js1b.SEG_INPUT_BYTES + js1b.LOGIT_BYTES + js1b.FIELD_DATA_BYTES + 128
)
POSE_INPUT_BYTES: Final = js1b.N_PAIRS * 12 * js1b.SEG_HEIGHT * js1b.SEG_WIDTH * 4
POSE_OUTPUT_BYTES: Final = js1b.N_PAIRS * 12 * 4
POSE_VECTOR_BYTES: Final = js1b.N_PAIRS * 6 * 4
EXPECTED_POSE_RETENTION_BYTES: Final = (
    js1b.RAW_BYTES
    + 3 * (POSE_INPUT_BYTES + POSE_OUTPUT_BYTES + POSE_VECTOR_BYTES)
    + 2 * js1b.N_PAIRS * 8
)


class RE1TWorkerError(RuntimeError):
    """A retained-input, receiver, scorer, or resume invariant failed."""


def storage_preflight(
    run_root: Path,
    *,
    retain_pose_vectors: bool = False,
) -> dict[str, Any]:
    """Fail closed unless the volume can retain every materialized payload."""
    usage = shutil.disk_usage(run_root)
    already_retained = js1b.current_retained_bytes(run_root)
    expected_total = EXPECTED_RETAINED_PAYLOAD_BYTES
    if retain_pose_vectors:
        expected_total += EXPECTED_POSE_RETENTION_BYTES
    remaining = max(0, expected_total - already_retained)
    required = remaining + js1b.STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_re1t_t4_storage_preflight.v1",
        "tier": str(run_root),
        "free_bytes": usage.free,
        "already_retained_bytes": already_retained,
        "expected_total_retained_payload_bytes": expected_total,
        "remaining_payload_bytes": remaining,
        "reserve_bytes": js1b.STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "block and retain; no generated payload is deleted",
    }
    if retain_pose_vectors:
        result["retain_pose_vectors"] = True
        result["expected_pose_retention_bytes"] = EXPECTED_POSE_RETENTION_BYTES
    js1b.atomic_json(run_root / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise RE1TWorkerError(
            f"RE1T storage preflight failed: free={usage.free}, required={required}"
        )
    return result


def pose_measurement(
    gt: np.ndarray,
    candidate_first: np.ndarray,
    candidate_repeat: np.ndarray,
) -> dict[str, Any]:
    """Reduce retained official first-six vectors without discarding pair payloads."""
    arrays = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (gt, candidate_first, candidate_repeat)
    )
    if any(value.shape != (js1b.N_PAIRS, 6) for value in arrays):
        raise RE1TWorkerError("PoseNet vector arrays must each have shape (600, 6)")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise RE1TWorkerError("PoseNet vector arrays must be finite")
    gt32, first32, repeat32 = arrays
    first_error = first32 - gt32
    repeat_error = repeat32 - gt32
    repeat_delta = repeat32 - first32
    pair_error_rms = np.sqrt(np.mean(first_error * first_error, axis=1, dtype=np.float32))
    pair_repeat_rms = np.sqrt(np.mean(repeat_delta * repeat_delta, axis=1, dtype=np.float32))
    return {
        "schema": "ddm_re1t_t4_pose_measurement.v1",
        "pairs": js1b.N_PAIRS,
        "components_per_pair": 6,
        "d_pose_candidate_first": float(
            np.mean(first_error * first_error, dtype=np.float32)
        ),
        "d_pose_candidate_repeat": float(
            np.mean(repeat_error * repeat_error, dtype=np.float32)
        ),
        "repeat_noise_mse": float(
            np.mean(repeat_delta * repeat_delta, dtype=np.float32)
        ),
        "pair_error_rms": pair_error_rms.astype(np.float64),
        "pair_repeat_noise_rms": pair_repeat_rms.astype(np.float64),
        "adjudicated_remotely": False,
    }


def field_measurement(
    candidate: np.ndarray,
    gt: np.ndarray,
    base: np.ndarray,
) -> dict[str, Any]:
    """Reduce full retained fields without applying an admission verdict."""
    arrays = tuple(np.asarray(value) for value in (candidate, gt, base))
    if arrays[0].shape != arrays[1].shape or arrays[1].shape != arrays[2].shape:
        raise RE1TWorkerError("candidate, GT, and base fields have different geometry")
    candidate_array, gt_array, base_array = arrays
    base_flips = int(np.count_nonzero(base_array != gt_array))
    candidate_flips = int(np.count_nonzero(candidate_array != gt_array))
    candidate_base_changes = int(np.count_nonzero(candidate_array != base_array))
    return {
        "schema": "ddm_re1t_t4_field_measurement.v1",
        "denominator_pixels": int(gt_array.size),
        "base_flips_vs_gt": base_flips,
        "candidate_flips_vs_gt": candidate_flips,
        "candidate_minus_base_flips": candidate_flips - base_flips,
        "candidate_changed_pixels_vs_cp135": candidate_base_changes,
        "candidate_field_identical_to_cp135": candidate_base_changes == 0,
        "adjudicated_remotely": False,
    }


def run(
    run_root: Path,
    resume_from: str,
    *,
    retain_pose_vectors: bool = False,
) -> dict[str, Any]:
    import timm
    import torch
    import torchvision

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    started = time.time()
    request = json.loads((run_root / "inputs/REQUEST.json").read_text(encoding="utf-8"))
    if resume_from != request["resume_from"] or resume_from != request["run_id"]:
        raise RE1TWorkerError("resume token differs from retained run identity")
    if request.get("score_claim") is not False or request.get("promotion_eligible") is not False:
        raise RE1TWorkerError("request crossed the component-only authority boundary")
    if request.get("local_pose_delta") != 0.0 or request.get("pose_unmeasured") is not True:
        raise RE1TWorkerError("request lost the explicit Pose-unknown placeholder law")
    requested_pose_vectors = bool(request.get("retain_pose_vectors", False))
    if requested_pose_vectors != retain_pose_vectors:
        raise RE1TWorkerError("worker Pose-vector mode differs from the sealed request")
    evidence_input = (
        "POSE_SCREEN_RESULT.json"
        if retain_pose_vectors
        else "RE1X_FULL_N600_BLOCKER.json"
    )
    if set(request.get("inputs", {})) != {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        evidence_input,
    }:
        raise RE1TWorkerError("request input census differs")
    source_shas = {
        "worker_source_sha256": js1b.sha256_file(Path(__file__)),
        "js1b_worker_source_sha256": js1b.sha256_file(Path(js1b.__file__)),
    }
    for key, observed in source_shas.items():
        if request.get(key) != observed:
            raise RE1TWorkerError(f"remote source differs from the sealed request: {key}")
    final_path = run_root / "FINAL_RESULT.json"
    if final_path.is_file():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("execution_status") != "MEASUREMENT_COMPLETE":
            raise RE1TWorkerError("retained final result is not complete")
        if result.get("candidate_archive") != request["inputs"]["candidate_archive.zip"]:
            raise RE1TWorkerError("retained final result belongs to a different candidate")
        if result.get("score_claim") is not False or result.get("axis") != AXIS:
            raise RE1TWorkerError("retained final result crossed its component-only authority")
        retained_pose = bool(
            result.get("retention", {}).get("all_pose_inputs_outputs_and_first6_vectors", False)
        )
        if retained_pose != retain_pose_vectors:
            raise RE1TWorkerError("retained final result belongs to another Pose-vector mode")
        final_checkpoint = (
            "stage_40_measurement_final.json"
            if retain_pose_vectors
            else "stage_30_measurement_final.json"
        )
        js1b.checkpoint_once(run_root / "checkpoints" / final_checkpoint, result)
        return result

    storage_preflight(run_root, retain_pose_vectors=retain_pose_vectors)
    for filename, record in request["inputs"].items():
        js1b.require_record(run_root / "inputs" / filename, record)
    js1b.require_record(GT_FIELD, GT_FIELD_RECORD)
    js1b.require_record(BASE_FIELD, BASE_FIELD_RECORD)
    base = np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False)
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    if int(np.count_nonzero(base != gt)) != BASE_FLIPS:
        raise RE1TWorkerError("retained T4 control no longer has 34,970 flips")
    js1b.checkpoint_once(
        run_root / "checkpoints/stage_00_inputs.json",
        {
            "schema": "ddm_re1t_t4_stage_checkpoint.v1",
            "stage": "inputs",
            "inputs": request["inputs"],
            "prior_gt_field": GT_FIELD_RECORD,
            "prior_base_field": BASE_FIELD_RECORD,
            # STORAGE_PREFLIGHT.json is intentionally absent: free/retained bytes
            # are volatile, so checkpointing them makes byte-identical resume
            # impossible (the SA1 20260813g defect).
            "resume_from": resume_from,
            "complete": True,
        },
    )

    runtime_root = run_root / "work/runtime/candidate"
    js1b.extract_zip_once(
        run_root / "inputs/candidate_runtime.zip",
        runtime_root,
        "RUNTIME_EXTRACTED.json",
    )
    receiver = js1b.decode_exact_receiver(
        name="candidate",
        archive_path=run_root / "inputs/candidate_archive.zip",
        archive_record=request["inputs"]["candidate_archive.zip"],
        runtime_root=runtime_root,
        run_root=run_root,
    )
    js1b.checkpoint_once(
        run_root / "checkpoints/stage_10_candidate_decoded.json",
        {
            "schema": "ddm_re1t_t4_stage_checkpoint.v1",
            "stage": "candidate_decoded",
            "receiver": receiver,
            "complete": True,
        },
    )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(js1b.SEED)
    np.random.seed(js1b.SEED)
    scorer = js1b.load_segnet(device)
    scorer_result = js1b.score_argmax_field(
        source="candidate",
        raw_root=Path(receiver["raw"]["path"]).parent,
        raw_record=receiver["raw"],
        scorer=scorer,
        device=device,
        run_root=run_root,
    )
    js1b.checkpoint_once(
        run_root / "checkpoints/stage_20_candidate_scored.json",
        {
            "schema": "ddm_re1t_t4_stage_checkpoint.v1",
            "stage": "candidate_scored",
            "scorer": scorer_result,
            "complete": True,
        },
    )
    pose_results: dict[str, Any] | None = None
    pose_metrics: dict[str, Any] | None = None
    if retain_pose_vectors:
        del scorer
        torch.cuda.empty_cache()
        posenet = js1b.load_posenet(device)
        raw_root = Path(receiver["raw"]["path"]).parent
        raw_record = receiver["raw"]
        pose_results = {
            "gt": js1b.score_pose_vectors(
                source="gt",
                raw_root=None,
                raw_record=None,
                scorer=posenet,
                device=device,
                run_root=run_root,
            ),
            "candidate_first": js1b.score_pose_vectors(
                source="candidate_first",
                raw_root=raw_root,
                raw_record=raw_record,
                scorer=posenet,
                device=device,
                run_root=run_root,
            ),
            "candidate_repeat": js1b.score_pose_vectors(
                source="candidate_repeat",
                raw_root=raw_root,
                raw_record=raw_record,
                scorer=posenet,
                device=device,
                run_root=run_root,
            ),
        }
        vector_paths = {
            name: Path(value["first6_vectors"]["path"])
            for name, value in pose_results.items()
        }
        pose_metrics = pose_measurement(
            np.load(vector_paths["gt"], mmap_mode="r", allow_pickle=False),
            np.load(vector_paths["candidate_first"], mmap_mode="r", allow_pickle=False),
            np.load(vector_paths["candidate_repeat"], mmap_mode="r", allow_pickle=False),
        )
        pair_error = pose_metrics.pop("pair_error_rms")
        pair_repeat = pose_metrics.pop("pair_repeat_noise_rms")
        pose_metrics["pair_error_rms"] = js1b.atomic_npy(
            run_root / "retained/pose_vectors/pair_error_rms_n600.npy",
            pair_error,
        )
        pose_metrics["pair_repeat_noise_rms"] = js1b.atomic_npy(
            run_root / "retained/pose_vectors/pair_repeat_noise_rms_n600.npy",
            pair_repeat,
        )
        js1b.checkpoint_once(
            run_root / "checkpoints/stage_30_posenet_complete.json",
            {
                "schema": "ddm_re1t_t4_stage_checkpoint.v1",
                "stage": "posenet_complete",
                "scorers": pose_results,
                "measurement": pose_metrics,
                "complete": True,
            },
        )
    candidate_path = Path(scorer_result["argmax"]["path"])
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    metrics = field_measurement(candidate, gt, base)
    if metrics["denominator_pixels"] != js1b.N_PAIRS * js1b.SEG_HEIGHT * js1b.SEG_WIDTH:
        raise RE1TWorkerError("RE1T full-population denominator differs")
    if metrics["base_flips_vs_gt"] != BASE_FLIPS:
        raise RE1TWorkerError("RE1T base control changed during measurement")

    upstream_sha = compute_upstream_snapshot_sha256(
        js1b.UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    final = {
        "schema": "ddm_re1t_t4_measurement_result.v1",
        "execution_status": "MEASUREMENT_COMPLETE",
        "status": "RETURN_TO_LOCAL_ADJUDICATOR",
        "axis": AXIS,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "candidate_archive": request["inputs"]["candidate_archive.zip"],
        "receiver": receiver,
        "scorer": scorer_result,
        "field_measurement": metrics,
        "retained_prior_fields": {
            "gt": js1b.file_record(GT_FIELD),
            "cp135_base": js1b.file_record(BASE_FIELD),
        },
        "retention": {
            "candidate_exact_receiver_raw": True,
            "all_candidate_seg_inputs": True,
            "all_candidate_logits": True,
            "candidate_argmax_field": js1b.file_record(candidate_path),
            "volume_run_root": str(run_root),
        },
        "provenance": {
            "source_git_head": request["source_git_head"],
            "source_git_dirty_at_prepare": request["source_git_dirty"],
            "source_git_status_sha256": request["source_git_status_sha256"],
            "dispatcher_source_sha256": request["dispatcher_source_sha256"],
            "worker_source_sha256": request["worker_source_sha256"],
            "js1b_worker_source_sha256": request["js1b_worker_source_sha256"],
            "upstream_snapshot_sha256": upstream_sha,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "timm_version": timm.__version__,
            "numpy_version": np.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "remote_adjudication_performed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "boundaries": {
            "measured": "exact candidate public decode plus n600 frozen T4 SegNet field",
            "not_measured": "PoseNet, contest-CPU, full evaluate.py score, or mixed-axis admission",
            "adjudication_surface": "local dispatcher after harvest",
        },
    }
    if retain_pose_vectors:
        final["axis_pose"] = AXIS_POSE
        final["pose_scorers"] = pose_results
        final["pose_measurement"] = pose_metrics
        final["retention"]["all_pose_inputs_outputs_and_first6_vectors"] = True
        final["retention"]["candidate_pose_forward_repeat_same_decoded_payload"] = True
        final["boundaries"] = {
            "measured": (
                "exact candidate public decode, n600 frozen T4 SegNet field, and official "
                "PoseNet first-six vectors for GT plus two candidate passes"
            ),
            "not_measured": (
                "contest-CPU or upstream evaluate.py; complete-S adjudication remains local"
            ),
            "adjudication_surface": "local dispatcher after harvest",
        }
    js1b.checkpoint_once(final_path, final)
    final_checkpoint = (
        "stage_40_measurement_final.json"
        if retain_pose_vectors
        else "stage_30_measurement_final.json"
    )
    js1b.checkpoint_once(run_root / "checkpoints" / final_checkpoint, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", required=True)
    parser.add_argument(
        "--retain-pose-vectors",
        action="store_true",
        help="also retain official PoseNet first-six vectors for GT and two candidate passes",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                args.run_root.resolve(),
                args.resume_from,
                retain_pose_vectors=args.retain_pose_vectors,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
