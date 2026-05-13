"""Reusable custody logic for CPU/CUDA auth-eval device-axis comparisons.

The contest stack has two separate questions that should not be conflated:

* which scorer device produced the final score; and
* which device the submission runtime used while inflating frames.

This module keeps the shared, non-promotional bookkeeping in one place so
planners, harvesters, and xray tools classify exact pairs consistently.
"""

from __future__ import annotations

import math
from typing import Any


CONTEST_N_BYTES = 37_545_489
CONTEST_CPU_AXIS = "contest_cpu"
CONTEST_CUDA_AXIS = "contest_cuda"
CONTEST_CUDA_EQUIVALENT_GPU_MODEL_TOKENS = (
    "t4",
    "a100",
    "4090",
    "h100",
    "a10g",
    "l40s",
)


def is_contest_cuda_equivalent_gpu(
    *,
    gpu_model: str | None,
    gpu_t4_match: bool = False,
) -> bool:
    """Return whether a CUDA auth-eval GPU belongs to the accepted contest axis.

    The public scorer uses NVIDIA T4, but internal exact CUDA custody also
    treats Linux x86_64 A100/4090/H100/A10G/L40S replays as contest-CUDA
    evidence. This helper intentionally only classifies the GPU family; callers
    still own device, sample-count, and OS/architecture checks.
    """

    if gpu_t4_match is True:
        return True
    model = str(gpu_model or "").lower()
    return any(token in model for token in CONTEST_CUDA_EQUIVALENT_GPU_MODEL_TOKENS)


def score_terms(*, pose: float, seg: float, archive_bytes: int) -> dict[str, float]:
    """Return official contest score terms for one auth-eval artifact."""

    rate = 25.0 * archive_bytes / CONTEST_N_BYTES
    return {
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(10.0 * pose),
        "rate_term": rate,
        "score": 100.0 * seg + math.sqrt(10.0 * pose) + rate,
    }


def cuda_minus_cpu_gaps(
    *,
    cpu_terms: dict[str, float],
    cuda_terms: dict[str, float],
) -> dict[str, float | None]:
    """Return CUDA-minus-CPU score/component gaps with gap-share fields."""

    score_gap = cuda_terms["score"] - cpu_terms["score"]
    seg_gap = cuda_terms["seg_term"] - cpu_terms["seg_term"]
    pose_gap = cuda_terms["pose_term"] - cpu_terms["pose_term"]
    rate_gap = cuda_terms["rate_term"] - cpu_terms["rate_term"]
    return {
        "score": score_gap,
        "seg_term": seg_gap,
        "pose_term": pose_gap,
        "rate_term": rate_gap,
        "seg_gap_share": seg_gap / score_gap if score_gap else None,
        "pose_gap_share": pose_gap / score_gap if score_gap else None,
        "rate_gap_share": rate_gap / score_gap if score_gap else None,
    }


def raw_output_pairing(
    *,
    cpu_raw: dict[str, Any] | None,
    cuda_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify inflated-frame custody for a paired CPU/CUDA eval.

    Missing raw manifests do not invalidate the score pair, but they do block
    mechanism-complete conclusions because scorer-device and runtime-device
    effects cannot be separated.
    """

    same: bool | None = None
    status = "raw_output_manifest_missing"
    if cpu_raw and cuda_raw:
        same = cpu_raw.get("aggregate_sha256") == cuda_raw.get("aggregate_sha256")
        status = "same_inflated_outputs" if same else "different_inflated_outputs"
    elif cpu_raw or cuda_raw:
        status = "partial_raw_output_manifest"

    mechanism_blockers: list[str] = []
    if status == "raw_output_manifest_missing":
        mechanism_blockers.append("raw_output_manifest_missing")
    elif status == "partial_raw_output_manifest":
        mechanism_blockers.append("partial_raw_output_manifest")

    return {
        "cpu": cpu_raw,
        "cuda": cuda_raw,
        "same_inflated_output_aggregate_sha256": same,
        "raw_output_pairing_status": status,
        "mechanism_blockers": mechanism_blockers,
        "mechanism_analysis_complete": not mechanism_blockers,
    }


def mechanism_class_from_pair(
    *,
    same_inflated_output_aggregate_sha256: bool | None,
    same_runtime_tree_sha256: bool | None,
    same_archive_sha256: bool,
) -> str:
    """Return the narrowest mechanism class justified by pair custody."""

    if same_inflated_output_aggregate_sha256 is True:
        return "same_raw_outputs_scorer_or_loader_drift"
    if same_inflated_output_aggregate_sha256 is False:
        return "different_raw_outputs_runtime_or_inflate_drift"
    if same_runtime_tree_sha256 is True and same_archive_sha256:
        return "same_archive_runtime_raw_outputs_unmeasured"
    return "custody_incomplete"
