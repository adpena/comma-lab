#!/usr/bin/env python3
"""Retained candidate-only T4 field gate for the SA1 shipping-axis actuator."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

REMOTE_REPO: Final = Path("/workspace/pact")
if str(REMOTE_REPO) not in sys.path:
    sys.path.insert(0, str(REMOTE_REPO))

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1b

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
BASE_BYTES: Final = 186_252
DENOMINATOR: Final = 600 * 384 * 512
POSE_LINEAR_PRICE: Final = 603.0
RATE_PRICE: Final = 25.0 / 37_545_489.0
EXPECTED_RETAINED_PAYLOAD_BYTES: Final = (
    js1b.RAW_BYTES + js1b.SEG_INPUT_BYTES + js1b.LOGIT_BYTES + js1b.FIELD_DATA_BYTES + 128
)


class SA1T4Error(RuntimeError):
    """A retained-input, receiver, scorer, or gate invariant failed."""


def storage_preflight(run_root: Path) -> dict[str, Any]:
    usage = __import__("shutil").disk_usage(run_root)
    already_retained = js1b.current_retained_bytes(run_root)
    remaining = max(0, EXPECTED_RETAINED_PAYLOAD_BYTES - already_retained)
    required = remaining + js1b.STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_sa1_t4_storage_preflight.v1",
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
        raise SA1T4Error("T4 storage preflight failed")
    return result


def adjudicate(
    candidate: np.ndarray,
    gt: np.ndarray,
    *,
    candidate_archive_bytes: int,
    local_pose_delta: float,
) -> dict[str, Any]:
    candidate_flips = int(np.count_nonzero(candidate != gt))
    delta_flips = candidate_flips - BASE_FLIPS
    delta_bytes = candidate_archive_bytes - BASE_BYTES
    seg_delta_s = 100.0 * delta_flips / DENOMINATOR
    pose_delta_s = POSE_LINEAR_PRICE * local_pose_delta
    rate_delta_s = RATE_PRICE * delta_bytes
    joint_delta_s = seg_delta_s + pose_delta_s + rate_delta_s
    per_flip_s = 100.0 / DENOMINATOR
    pose_rate_cost_s = pose_delta_s + rate_delta_s
    minimum_reduction = max(1, math.floor(pose_rate_cost_s / per_flip_s) + 1)
    maximum_admissible_candidate_flips = BASE_FLIPS - minimum_reduction
    admitted = candidate_flips <= maximum_admissible_candidate_flips and joint_delta_s < 0.0
    return {
        "status": "ADMITTED_FOR_FULL_EXACT_ROW" if admitted else "REJECTED_BY_T4_SIGN_GATE",
        "admitted_for_full_exact_row": admitted,
        "base_flips": BASE_FLIPS,
        "candidate_flips": candidate_flips,
        "delta_flips": delta_flips,
        "candidate_archive_bytes": candidate_archive_bytes,
        "delta_bytes": delta_bytes,
        "local_pose_delta_stratified_n32": local_pose_delta,
        "seg_delta_s_exact_t4_field": seg_delta_s,
        "pose_delta_s_local_linearized": pose_delta_s,
        "rate_delta_s_exact_archive": rate_delta_s,
        "pose_plus_rate_cost_s": pose_rate_cost_s,
        "minimum_required_flip_reduction": minimum_reduction,
        "maximum_admissible_candidate_flips": maximum_admissible_candidate_flips,
        "joint_delta_s_mixed_axis_admission_only": joint_delta_s,
        "gate_rule": (
            "candidate flips at or below the pose-plus-rate break-even ceiling and "
            "mixed-axis joint delta S below zero"
        ),
        "full_exact_score_claim": False,
    }


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    import timm
    import torch
    import torchvision

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    started = time.time()
    request = json.loads((run_root / "inputs/REQUEST.json").read_text())
    if resume_from != request["resume_from"] or resume_from != request["run_id"]:
        raise SA1T4Error("resume token differs from retained run identity")
    final_path = run_root / "FINAL_RESULT.json"
    if final_path.is_file():
        result = json.loads(final_path.read_text())
        if result.get("execution_status") != "COMPLETE":
            raise SA1T4Error("retained final result is not complete")
        return result

    preflight = storage_preflight(run_root)
    for filename, record in request["inputs"].items():
        js1b.require_record(run_root / "inputs" / filename, record)
    js1b.require_record(GT_FIELD, GT_FIELD_RECORD)
    js1b.require_record(BASE_FIELD, BASE_FIELD_RECORD)
    base = np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False)
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    if int(np.count_nonzero(base != gt)) != BASE_FLIPS:
        raise SA1T4Error("retained T4 control no longer has 34,970 flips")
    js1b.checkpoint_once(
        run_root / "checkpoints/stage_00_inputs_and_storage.json",
        {
            "schema": "ddm_sa1_t4_stage_checkpoint.v1",
            "stage": "inputs_and_storage",
            "inputs": request["inputs"],
            "prior_gt_field": GT_FIELD_RECORD,
            "prior_base_field": BASE_FIELD_RECORD,
            "storage_preflight": preflight,
            "resume_from": resume_from,
            "complete": True,
        },
    )

    runtime_root = run_root / "work/runtime/sa1_candidate"
    js1b.extract_zip_once(
        run_root / "inputs/sa1_candidate_runtime.zip",
        runtime_root,
        "RUNTIME_EXTRACTED.json",
    )
    receiver = js1b.decode_exact_receiver(
        name="sa1_candidate",
        archive_path=run_root / "inputs/sa1_candidate_archive.zip",
        archive_record=request["inputs"]["sa1_candidate_archive.zip"],
        runtime_root=runtime_root,
        run_root=run_root,
    )
    js1b.checkpoint_once(
        run_root / "checkpoints/stage_10_candidate_decoded.json",
        {
            "schema": "ddm_sa1_t4_stage_checkpoint.v1",
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
        source="sa1_candidate",
        raw_root=Path(receiver["raw"]["path"]).parent,
        raw_record=receiver["raw"],
        scorer=scorer,
        device=device,
        run_root=run_root,
    )
    js1b.checkpoint_once(
        run_root / "checkpoints/stage_20_candidate_scored.json",
        {
            "schema": "ddm_sa1_t4_stage_checkpoint.v1",
            "stage": "candidate_scored",
            "scorer": scorer_result,
            "complete": True,
        },
    )
    candidate_path = Path(scorer_result["argmax"]["path"])
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    gate = adjudicate(
        candidate,
        gt,
        candidate_archive_bytes=int(request["inputs"]["sa1_candidate_archive.zip"]["bytes"]),
        local_pose_delta=float(request["local_pose_delta_stratified_n32"]),
    )
    upstream_sha = compute_upstream_snapshot_sha256(
        js1b.UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    final = {
        "schema": "ddm_sa1_t4_sign_gate_result.v1",
        "execution_status": "COMPLETE",
        "status": gate["status"],
        "axis": AXIS,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "receiver": receiver,
        "scorer": scorer_result,
        "gate": gate,
        "retained_prior_fields": {
            "gt": js1b.file_record(GT_FIELD),
            "cp135_base": js1b.file_record(BASE_FIELD),
        },
        "retention": {
            "candidate_exact_receiver_raw": True,
            "all_candidate_seg_inputs": True,
            "all_candidate_logits": True,
            "candidate_argmax_field": True,
            "volume_run_root": str(run_root),
        },
        "provenance": {
            "source_git_head": request["source_git_head"],
            "source_git_dirty_at_dispatch": request["source_git_dirty"],
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
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "boundaries": {
            "measured": "candidate exact receiver plus n600 frozen T4 SegNet argmax field",
            "not_measured": "T4 PoseNet, contest-CPU, or full upstream exact score",
            "mixed_axis_joint_gate_uses_local_pose_guard": True,
        },
    }
    js1b.checkpoint_once(final_path, final)
    js1b.checkpoint_once(run_root / "checkpoints/stage_30_final.json", final)
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
