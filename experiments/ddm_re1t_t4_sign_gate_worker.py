#!/usr/bin/env python3
"""Retain the RE1 Round-1 CUDA decode and SegNet field measurement.

This worker deliberately does not adjudicate.  It runs the exact hash-pinned
public receiver, retains every raw/scorer payload, reduces the full candidate
field against the retained GT/base controls, and returns the measurement for
local mixed-axis arithmetic.
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


class RE1TWorkerError(RuntimeError):
    """A retained-input, receiver, scorer, or resume invariant failed."""


def storage_preflight(run_root: Path) -> dict[str, Any]:
    """Fail closed unless the volume can retain every materialized payload."""
    usage = shutil.disk_usage(run_root)
    already_retained = js1b.current_retained_bytes(run_root)
    remaining = max(0, EXPECTED_RETAINED_PAYLOAD_BYTES - already_retained)
    required = remaining + js1b.STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_re1t_t4_storage_preflight.v1",
        "tier": str(run_root),
        "free_bytes": usage.free,
        "already_retained_bytes": already_retained,
        "expected_total_retained_payload_bytes": EXPECTED_RETAINED_PAYLOAD_BYTES,
        "remaining_payload_bytes": remaining,
        "reserve_bytes": js1b.STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "block and retain; no generated payload is deleted",
    }
    js1b.atomic_json(run_root / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise RE1TWorkerError(
            f"RE1T storage preflight failed: free={usage.free}, required={required}"
        )
    return result


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


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
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
    if set(request.get("inputs", {})) != {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "RE1X_FULL_N600_BLOCKER.json",
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
        js1b.checkpoint_once(run_root / "checkpoints/stage_30_measurement_final.json", result)
        return result

    storage_preflight(run_root)
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
    js1b.checkpoint_once(final_path, final)
    js1b.checkpoint_once(run_root / "checkpoints/stage_30_measurement_final.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.run_root.resolve(), args.resume_from), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
