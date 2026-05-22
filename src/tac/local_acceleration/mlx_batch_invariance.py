# SPDX-License-Identifier: MIT
"""Batch-invariance audit for local MLX scorer adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    run_mlx_distortion_scorer_nchw,
    temporary_mlx_device,
    torch_distortion_net_to_mlx,
)
from tac.local_acceleration.mlx_scorer_response import (
    GPU_RESEARCH_SIGNAL_BLOCKER,
    _load_upstream_distortion_net,
    load_scorer_input_cache,
)

SCHEMA_VERSION = "mlx_scorer_batch_invariance.v1"
PASS_VERDICT = "PASS_MLX_BATCH_INVARIANCE"
FAIL_VERDICT = "FAIL_MLX_BATCH_INVARIANCE"


@dataclass(frozen=True)
class MLXBatchInvarianceThresholds:
    """Tolerances for treating batched MLX scorer output as singleton-equivalent."""

    max_posenet_output_abs_delta: float = 1.0e-4
    max_segnet_logit_abs_delta: float = 1.0e-3
    max_segnet_argmax_diff_pixels: int = 0


def write_batch_invariance_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_scorer_batch_invariance_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str,
    start_pair: int,
    batch_pairs: int,
    allow_gpu_research_signal: bool = False,
    thresholds: MLXBatchInvarianceThresholds | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare one batched scorer response against singleton scorer responses."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: device_type='gpu' is local MLX "
            "research signal only; pass allow_gpu_research_signal=True after "
            "recording a non-promotional rationale"
        )
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(batch_pairs) < 2:
        raise ValueError(f"batch_pairs must be >= 2 for invariance audit, got {batch_pairs}")

    cache = load_scorer_input_cache(cache_dir)
    total_pair_count = int(cache.pair_indices.shape[0])
    start = int(start_pair)
    stop = start + int(batch_pairs)
    if stop > total_pair_count:
        raise ValueError(
            f"requested pair window [{start}, {stop}) exceeds cache pair count {total_pair_count}"
        )

    pose = np.asarray(cache.posenet_yuv6_pair[start:stop], dtype=np.float32)
    seg = np.asarray(cache.segnet_last_rgb[start:stop], dtype=np.float32)
    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    with temporary_mlx_device(device_type):
        adapter = torch_distortion_net_to_mlx(dist)
        batched = run_mlx_distortion_scorer_nchw(adapter, pose, seg)
        singleton_outputs = [
            run_mlx_distortion_scorer_nchw(adapter, pose[index : index + 1], seg[index : index + 1])
            for index in range(int(batch_pairs))
        ]

    singleton = concatenate_distortion_outputs(singleton_outputs)
    return build_batch_invariance_manifest_from_outputs(
        batched_outputs=batched,
        singleton_outputs=singleton,
        thresholds=thresholds,
        run_id=run_id,
        device_type=device_type,
        cache_dir=str(cache_dir),
        start_pair=start,
        batch_pairs=int(batch_pairs),
        total_pair_count=total_pair_count,
        allow_gpu_research_signal=allow_gpu_research_signal,
    )


def concatenate_distortion_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not outputs:
        raise ValueError("outputs must not be empty")
    pose_keys = sorted(outputs[0]["posenet"].keys())
    return {
        "posenet": {
            key: np.concatenate(
                [np.asarray(item["posenet"][key], dtype=np.float32) for item in outputs],
                axis=0,
            )
            for key in pose_keys
        },
        "segnet": np.concatenate(
            [np.asarray(item["segnet"], dtype=np.float32) for item in outputs],
            axis=0,
        ),
    }


def build_batch_invariance_manifest_from_outputs(
    *,
    batched_outputs: dict[str, Any],
    singleton_outputs: dict[str, Any],
    thresholds: MLXBatchInvarianceThresholds | None = None,
    run_id: str | None = None,
    device_type: str | None = None,
    cache_dir: str | None = None,
    start_pair: int | None = None,
    batch_pairs: int | None = None,
    total_pair_count: int | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    limits = thresholds or MLXBatchInvarianceThresholds()
    blockers: list[str] = []

    pose_deltas = _pose_output_deltas(batched_outputs, singleton_outputs)
    max_pose_delta = max(pose_deltas.values(), default=0.0)
    seg_batched = np.asarray(batched_outputs["segnet"], dtype=np.float32)
    seg_singleton = np.asarray(singleton_outputs["segnet"], dtype=np.float32)
    if seg_batched.shape != seg_singleton.shape:
        blockers.append(
            "segnet_shape_mismatch:"
            f"batched={list(seg_batched.shape)}:singleton={list(seg_singleton.shape)}"
        )
        seg_logit_delta = None
        seg_argmax_diff_pixels = None
    else:
        seg_logit_delta = float(np.max(np.abs(seg_batched - seg_singleton)))
        seg_argmax_diff_pixels = int(
            np.sum(np.argmax(seg_batched, axis=1) != np.argmax(seg_singleton, axis=1))
        )

    if max_pose_delta > limits.max_posenet_output_abs_delta:
        blockers.append(
            "posenet_output_abs_delta_exceeds_threshold:"
            f"{max_pose_delta:.12g}>{limits.max_posenet_output_abs_delta:.12g}"
        )
    if (
        seg_logit_delta is not None
        and seg_logit_delta > limits.max_segnet_logit_abs_delta
    ):
        blockers.append(
            "segnet_logit_abs_delta_exceeds_threshold:"
            f"{seg_logit_delta:.12g}>{limits.max_segnet_logit_abs_delta:.12g}"
        )
    if (
        seg_argmax_diff_pixels is not None
        and seg_argmax_diff_pixels > limits.max_segnet_argmax_diff_pixels
    ):
        blockers.append(
            "segnet_argmax_diff_pixels_exceeds_threshold:"
            f"{seg_argmax_diff_pixels}>{limits.max_segnet_argmax_diff_pixels}"
        )
    if device_type == "gpu" and not allow_gpu_research_signal:
        blockers.append(GPU_RESEARCH_SIGNAL_BLOCKER)

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "thresholds": asdict(limits),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "device_type": device_type,
        "gpu_research_signal_allowed": bool(allow_gpu_research_signal),
        "device_contract": {
            "gpu_research_signal_required": device_type == "gpu",
            "gpu_research_signal_allowed": bool(allow_gpu_research_signal),
            "gpu_research_signal_blocker": GPU_RESEARCH_SIGNAL_BLOCKER,
        },
        "cache_dir": cache_dir,
        "start_pair": start_pair,
        "batch_pairs": batch_pairs,
        "total_pair_count": total_pair_count,
        "deltas": {
            "posenet_outputs_max_abs": pose_deltas,
            "posenet_output_abs_max": max_pose_delta,
            "segnet_logit_abs_max": seg_logit_delta,
            "segnet_argmax_diff_pixels": seg_argmax_diff_pixels,
        },
        "allowed_use": (
            [
                "local_mlx_batch_training_signal",
                "local_mlx_scorer_response_profiling",
            ]
            if passed
            else [
                "singletons_or_smaller_batches_only",
                "do_not_use_this_batch_shape_for_scorer_training_signal",
            ]
        ),
        "authority_status": (
            "Batch-invariance is local MLX evidence only; CPU/CUDA auth eval remains "
            "required for score claims and promotion."
        ),
    }


def _pose_output_deltas(
    batched_outputs: dict[str, Any],
    singleton_outputs: dict[str, Any],
) -> dict[str, float]:
    batched_pose = batched_outputs.get("posenet")
    singleton_pose = singleton_outputs.get("posenet")
    if not isinstance(batched_pose, dict) or not isinstance(singleton_pose, dict):
        raise ValueError("outputs must contain posenet dictionaries")
    keys = sorted(set(batched_pose) | set(singleton_pose))
    deltas: dict[str, float] = {}
    for key in keys:
        lhs = np.asarray(batched_pose.get(key), dtype=np.float32)
        rhs = np.asarray(singleton_pose.get(key), dtype=np.float32)
        if lhs.shape != rhs.shape:
            deltas[key] = float("inf")
        else:
            deltas[key] = float(np.max(np.abs(lhs - rhs)))
    return deltas
