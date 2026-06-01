# SPDX-License-Identifier: MIT
"""Fidelity contract between the MLX scorer lane and upstream auth eval.

The MLX implementation is deliberately split into two pieces:

1. ``tac.local_acceleration.mlx_preprocess`` mirrors upstream preprocessing and
   writes fixed scorer-input NumPy tensors.
2. ``tac.local_acceleration.mlx_scorer_adapters`` consumes those fixed tensors
   with MLX network adapters.

This module makes that contract machine-readable and fail-closed.  It is not a
numerical parity proof by itself; it is the source/shape/authority gate that
keeps MLX fast lanes from pretending to be contest auth eval.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.contest_eval_contract import (
    build_upstream_eval_contract,
)
from tac.local_acceleration import (
    EVIDENCE_GRADE_MLX,
    EVIDENCE_TAG_MLX,
    mlx_preprocess,
    mlx_scorer_adapters,
)

MLX_UPSTREAM_SCORER_CONTRACT_SCHEMA = "mlx_upstream_scorer_contract_fidelity.v1"


def build_mlx_upstream_scorer_contract_fidelity(
    *,
    upstream_root: str | Path = "upstream",
    cache_manifest: Mapping[str, Any] | None = None,
    require_full_contest_shapes: bool = True,
) -> dict[str, Any]:
    """Build a fail-closed MLX-vs-upstream scorer contract manifest."""

    upstream_root_path = Path(upstream_root)
    upstream = build_upstream_eval_contract(
        repo_root=upstream_root_path.parent,
        upstream_dir=upstream_root_path.name,
    )
    blockers: list[str] = []
    blockers.extend(_upstream_blockers(upstream))
    blockers.extend(_constant_blockers(upstream))
    blockers.extend(_source_contract_blockers())
    cache_validation = None
    if cache_manifest is not None:
        cache_validation = validate_mlx_cache_manifest_against_upstream_contract(
            cache_manifest,
            require_full_contest_shapes=require_full_contest_shapes,
        )
        blockers.extend(cache_validation["blockers"])
    passed = not blockers
    return {
        "schema": MLX_UPSTREAM_SCORER_CONTRACT_SCHEMA,
        "passed": passed,
        "verdict": (
            "PASS_STATIC_MLX_UPSTREAM_SCORER_CONTRACT"
            if passed
            else "FAIL_STATIC_MLX_UPSTREAM_SCORER_CONTRACT"
        ),
        "blockers": blockers,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "upstream_contract": upstream,
        "mlx_contract": {
            "schema": "mlx_fixed_scorer_input_contract.v1",
            "cache_producer": "tac.local_acceleration.mlx_preprocess",
            "network_adapter": "tac.local_acceleration.mlx_scorer_adapters.MLXDistortionScorerAdapter",
            "segnet_last_rgb_shape_nchw": [
                "N",
                3,
                *mlx_preprocess.SEGNET_INPUT_HW,
            ],
            "posenet_yuv6_pair_shape_nchw": [
                "N",
                12,
                *mlx_preprocess.YUV6_INPUT_HW,
            ],
            "pair_indices_shape": ["N", mlx_preprocess.SEQ_LEN],
            "frame_shape_hwc": [*mlx_preprocess.CAMERA_HW, 3],
            "adapter_consumes_fixed_inputs_not_raw_frames": True,
            "preprocess_authority": (
                "cache producer must match upstream DistortionNet.preprocess_input; "
                "adapter parity alone is insufficient"
            ),
            "component_math": {
                "pose": "mean((pose_ref[..., :6]-pose_cand[..., :6])**2)",
                "seg": "mean(argmax(seg_ref,axis=1)!=argmax(seg_cand,axis=1))",
            },
        },
        "cache_manifest_validation": cache_validation,
        "required_numerical_parity_steps": [
            "preprocess_scorer_inputs_from_pairs versus upstream DistortionNet.preprocess_input",
            "MLX PoseNet adapter versus upstream PoseNet on fixed YUV6 input",
            "MLX SegNet adapter versus upstream SegNet on fixed RGB input",
            "MLX component reductions versus upstream compute_distortion",
            "paired full-video cache calibration versus contest CPU/CUDA auth payload",
        ],
        "forbidden_inferences": [
            "MLX score equals contest CPU/CUDA score without calibration payload",
            "SegNet boundary saliency substitutes for PoseNet pair saliency",
            "adapter parity substitutes for raw inflate/evaluate.py replay",
        ],
    }


def validate_mlx_cache_manifest_against_upstream_contract(
    cache_manifest: Mapping[str, Any],
    *,
    require_full_contest_shapes: bool = True,
) -> dict[str, Any]:
    """Validate an MLX scorer-input cache manifest against upstream shapes."""

    blockers: list[str] = []
    pair_count = _positive_int(cache_manifest.get("pair_count"))
    if pair_count is None:
        blockers.append("cache_pair_count_missing_or_invalid")
        pair_count = -1
    expected_seg = [pair_count, 3, *mlx_preprocess.SEGNET_INPUT_HW]
    expected_pose = [pair_count, 12, *mlx_preprocess.YUV6_INPUT_HW]
    expected_pairs = [pair_count, mlx_preprocess.SEQ_LEN]
    expected_frame = [*mlx_preprocess.CAMERA_HW, 3]
    _shape_check(blockers, cache_manifest, "segnet_last_rgb_shape", expected_seg)
    _shape_check(blockers, cache_manifest, "posenet_yuv6_pair_shape", expected_pose)
    _shape_check(blockers, cache_manifest, "pair_indices_shape", expected_pairs)
    if require_full_contest_shapes:
        _shape_check(blockers, cache_manifest, "frame_shape_hwc", expected_frame)
    for flag in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "promotable",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if cache_manifest.get(flag) is True:
            blockers.append(f"cache_manifest_attempts_{flag}")
    return {
        "schema": "mlx_cache_manifest_upstream_contract_validation.v1",
        "passed": not blockers,
        "blockers": blockers,
        "expected": {
            "segnet_last_rgb_shape": expected_seg,
            "posenet_yuv6_pair_shape": expected_pose,
            "pair_indices_shape": expected_pairs,
            "frame_shape_hwc": expected_frame,
        },
        "observed": {
            "segnet_last_rgb_shape": cache_manifest.get("segnet_last_rgb_shape"),
            "posenet_yuv6_pair_shape": cache_manifest.get("posenet_yuv6_pair_shape"),
            "pair_indices_shape": cache_manifest.get("pair_indices_shape"),
            "frame_shape_hwc": cache_manifest.get("frame_shape_hwc"),
        },
    }


def _upstream_blockers(upstream: Mapping[str, Any]) -> list[str]:
    rows = upstream.get("implementation_snippet_checks")
    if not isinstance(rows, list):
        return ["upstream_implementation_snippet_checks_missing"]
    return [
        f"upstream_fragment_{row.get('relative_path')}:{row.get('name')}_missing"
        for row in rows
        if isinstance(row, Mapping) and row.get("present") is not True
    ]


def _constant_blockers(upstream: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    score_contract = upstream.get("score_allocation_contract")
    if not isinstance(score_contract, Mapping):
        return ["upstream_score_allocation_contract_missing"]
    video = score_contract.get("pair_geometry")
    seg = score_contract.get("segnet")
    pose = score_contract.get("posenet")
    if not isinstance(video, Mapping) or video.get("seq_len") != mlx_preprocess.SEQ_LEN:
        blockers.append("seq_len_mismatch")
    if not isinstance(video, Mapping) or list(video.get("camera_size_wh", [])) != [
        mlx_preprocess.CAMERA_SIZE[0],
        mlx_preprocess.CAMERA_SIZE[1],
    ]:
        blockers.append("camera_size_w_h_mismatch")
    expected_input_wh = [mlx_preprocess.SEGNET_INPUT_HW[1], mlx_preprocess.SEGNET_INPUT_HW[0]]
    if not isinstance(seg, Mapping) or list(seg.get("input_size_wh", [])) != expected_input_wh:
        blockers.append("segnet_input_hw_mismatch")
    expected_yuv6 = [
        6,
        mlx_preprocess.YUV6_INPUT_HW[0],
        mlx_preprocess.YUV6_INPUT_HW[1],
    ]
    if not isinstance(pose, Mapping) or list(pose.get("yuv6_shape_per_frame_chw", [])) != expected_yuv6:
        blockers.append("posenet_yuv6_hw_mismatch")
    return blockers


def _source_contract_blockers() -> list[str]:
    blockers: list[str] = []
    preprocess_source = inspect.getsource(mlx_preprocess.preprocess_scorer_inputs_from_pairs)
    distortion_source = inspect.getsource(
        mlx_scorer_adapters.scorer_distortion_components_numpy
    )
    adapter_source = inspect.getsource(mlx_scorer_adapters.MLXDistortionScorerAdapter)
    required = [
        ("preprocess_bilinear_resize", preprocess_source, "F.interpolate(x, size=SEGNET_INPUT_HW, mode=\"bilinear\")"),
        ("preprocess_segnet_last_frame", preprocess_source, "resized_pair[:, -1, ...]"),
        ("preprocess_yuv6_pair", preprocess_source, "_rgb_to_yuv6_torch(resized)"),
        ("component_pose_first_six", distortion_source, "reference_pose[..., :6]"),
        ("component_seg_argmax", distortion_source, "np.argmax(reference_seg, axis=1)"),
        ("adapter_fixed_posenet_input", adapter_source, "posenet_yuv6_pair_nhwc"),
        ("adapter_fixed_segnet_input", adapter_source, "segnet_last_rgb_nhwc"),
    ]
    for name, source, fragment in required:
        if fragment not in source:
            blockers.append(f"mlx_source_fragment_{name}_missing")
    return blockers


def _shape_check(
    blockers: list[str],
    manifest: Mapping[str, Any],
    key: str,
    expected: list[int],
) -> None:
    value = manifest.get(key)
    if not isinstance(value, (list, tuple)) or list(value) != expected:
        blockers.append(f"{key}_mismatch")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "MLX_UPSTREAM_SCORER_CONTRACT_SCHEMA",
    "build_mlx_upstream_scorer_contract_fidelity",
    "validate_mlx_cache_manifest_against_upstream_contract",
]
