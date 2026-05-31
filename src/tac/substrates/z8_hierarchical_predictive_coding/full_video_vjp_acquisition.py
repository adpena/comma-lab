# SPDX-License-Identifier: MIT
"""Full-video VJP acquisition lane for Z8 joint P18/P19 water-fill.

The contest optimizer may use mini-batch/window gradients as cheap ranking
probes, but budget-spending Z8 coefficient attacks must be ratified by a
full-video, archive-pinned joint P18/P19 surface. This module is the queue-owned
spine for that workflow: deterministic pair shards, target-mode policy, and a
surface bundle contract consumed by the coefficient materializer.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.joint_p18_p19_waterfill import (
    FULL_VIDEO_AUTHORITY_SCOPE,
    FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
    PROPOSAL_ONLY_SCOPE,
    JointP18P19WaterfillConfig,
    build_joint_p18_p19_waterfill_surface,
)
from tac.optimization.target_modes import (
    CONTEST_VIDEO_OVERFIT_MODE,
    CORPUS_GENERALIZATION_MODE,
    HYBRID_CONTEST_PLUS_CORPUS_MODE,
    normalize_target_optimization_mode,
    target_mode_declares_overfit_allowed,
    target_mode_requires_corpus_manifest,
)
from tac.repo_io import write_json
from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive

Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA = "z8_full_video_vjp_acquisition_plan.v1"
Z8_FULL_VIDEO_VJP_SURFACE_SHARD_SCHEMA = "z8_full_video_vjp_surface_shard.v1"
Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA = "z8_full_video_vjp_surface_bundle.v1"
Z8_FULL_VIDEO_VJP_MLX_SHARD_BACKEND = "mlx_autograd_raw_pair_p18_p19_vjp.v1"
Z8_FULL_VIDEO_MLX_REPLAY_SCHEMA = "z8_full_video_mlx_replay.v1"
CONTEST_RATE_NORMALIZER_BYTES = 37_545_489.0

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}
SINGLE_UPDATE_AFTER_FULL_REDUCTION = "single_update_after_all_pair_shards_reduce"
TRUE_P19_POSE_SURFACE_KIND = "per_axis_posenet_jacobian_mahalanobis_v1"
SCALAR_POSE_LOSS_VJP_SURFACE_KIND = "scalar_first6_pose_mse_vjp_proxy_v1"
P19_POSE_SURFACE_BLOCKER = "p19_pose_surface_not_true_per_axis_jacobian"


@dataclass(frozen=True)
class Z8FullVideoVjpAcquisitionConfig:
    """Config for deterministic full-video VJP shard planning."""

    target_mode: str = CONTEST_VIDEO_OVERFIT_MODE
    pair_chunk_size: int = 64
    parallel_workers: int | None = None
    corpus_manifest_path: str | None = None
    allow_minibatch_probe_between_full_passes: bool = True
    allow_partial_production_probe_surface: bool = False

    def __post_init__(self) -> None:
        normalize_target_optimization_mode(self.target_mode)
        if self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if self.parallel_workers is not None and self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive when provided")
        if target_mode_requires_corpus_manifest(self.target_mode) and not self.corpus_manifest_path:
            raise ValueError("corpus_manifest_path is required for production/hybrid target modes")

    @property
    def normalized_target_mode(self) -> str:
        return normalize_target_optimization_mode(self.target_mode)


@dataclass(frozen=True)
class Z8FullVideoMlxVjpShardConfig:
    """Config for one MLX-autograd P18/P19 VJP shard.

    The shard backprops through the MLX scorer preprocessing and scorer models
    to candidate pair RGB pixels. It is not budget-spend authority by itself;
    the complete full-video reduction is the authority boundary.
    """

    shard_index: int
    pair_start: int
    pair_end: int
    full_video_pair_count: int
    full_video_d_pose: float
    target_mode: str = CONTEST_VIDEO_OVERFIT_MODE
    rgb_value_range: float = 255.0
    scorer_hw: tuple[int, int] = (384, 512)
    seg_margin_delta: float = 1.0
    pose_null_threshold: float = 1e-8
    require_archive_runtime_candidate_custody: bool = True
    candidate_custody_atol: float = 0.0

    def __post_init__(self) -> None:
        normalize_target_optimization_mode(self.target_mode)
        if self.shard_index < 0:
            raise ValueError("shard_index must be non-negative")
        if self.pair_start < 0 or self.pair_end <= self.pair_start:
            raise ValueError("pair_start/pair_end must describe a non-empty positive span")
        if self.full_video_pair_count <= 0:
            raise ValueError("full_video_pair_count must be positive")
        if self.pair_end > self.full_video_pair_count:
            raise ValueError("pair_end cannot exceed full_video_pair_count")
        if self.full_video_d_pose < 0.0:
            raise ValueError("full_video_d_pose must be >= 0")
        if self.rgb_value_range <= 0.0:
            raise ValueError("rgb_value_range must be positive")
        if self.scorer_hw[0] <= 1 or self.scorer_hw[1] <= 1:
            raise ValueError("scorer_hw must be positive and at least 2x2")
        if self.seg_margin_delta < 0.0:
            raise ValueError("seg_margin_delta must be >= 0")
        if self.pose_null_threshold < 0.0:
            raise ValueError("pose_null_threshold must be >= 0")
        if self.candidate_custody_atol < 0.0:
            raise ValueError("candidate_custody_atol must be >= 0")

    @property
    def normalized_target_mode(self) -> str:
        return normalize_target_optimization_mode(self.target_mode)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _as_pair_rgb_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[1] != 2 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (pairs, 2, H, W, 3); got {arr.shape}")
    if arr.shape[2] < 2 or arr.shape[3] < 2:
        raise ValueError(f"{name} spatial dimensions must be at least 2x2; got {arr.shape[2:4]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _mlx_pairs_to_scorer_inputs_nhwc(pairs_rgb_255: Any, *, scorer_hw: tuple[int, int]) -> tuple[Any, Any]:
    """Return ``(segnet_last_rgb_nhwc, posenet_yuv6_pair_nhwc)`` in MLX."""

    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        resize_nhwc_align_corners_false,
        rgb_to_yuv6_mlx,
    )

    shape = tuple(int(dim) for dim in pairs_rgb_255.shape)
    if len(shape) != 5 or shape[1] != 2 or shape[-1] != 3:
        raise ValueError(f"expected pair RGB MLX tensor (B,2,H,W,3), got {shape}")
    flat = mx.reshape(pairs_rgb_255, (-1, shape[2], shape[3], 3))
    scorer_flat = resize_nhwc_align_corners_false(flat, size=scorer_hw, mode="bilinear")
    scorer_pairs = mx.reshape(
        scorer_flat,
        (shape[0], 2, int(scorer_hw[0]), int(scorer_hw[1]), 3),
    )
    segnet_last_rgb = scorer_pairs[:, 1, :, :, :]
    yuv6 = rgb_to_yuv6_mlx(scorer_pairs)
    h2, w2 = int(yuv6.shape[2]), int(yuv6.shape[3])
    pose = mx.reshape(mx.transpose(yuv6, (0, 2, 3, 1, 4)), (shape[0], h2, w2, 12))
    return segnet_last_rgb, pose


def _mlx_value_to_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    try:
        mx.synchronize()
    except AttributeError:
        pass
    return np.asarray(value)


def _candidate_archive_runtime_custody_report(
    archive_bytes: bytes,
    *,
    candidate_pairs_rgb: np.ndarray,
    rgb_value_range: float,
    required: bool,
    atol: float,
    archive_runtime_candidate_pairs_rgb: Any | None,
) -> dict[str, Any]:
    archive_sha = _sha256_bytes(archive_bytes)
    report: dict[str, Any] = {
        "schema": "z8_archive_runtime_candidate_custody.v1",
        "archive_sha256": archive_sha,
        "required": bool(required),
        "rgb_value_range": float(rgb_value_range),
        "candidate_custody_atol": float(atol),
    }
    if not required:
        report.update(
            {
                "archive_runtime_candidate_custody": False,
                "blocker": "archive_runtime_candidate_custody_not_required_probe_only",
            }
        )
        return report

    archive_pairs = (
        _as_pair_rgb_array(archive_runtime_candidate_pairs_rgb, name="archive_runtime_candidate_pairs_rgb")
        if archive_runtime_candidate_pairs_rgb is not None
        else reconstruct_z8_archive_pairs_rgb255(archive_bytes)
    )
    candidate_255 = np.ascontiguousarray(
        np.asarray(candidate_pairs_rgb, dtype=np.float32) * (255.0 / float(rgb_value_range)),
        dtype=np.float32,
    )
    if archive_pairs.shape != candidate_255.shape:
        report.update(
            {
                "archive_runtime_candidate_custody": False,
                "blocker": "archive_runtime_candidate_shape_mismatch",
                "archive_runtime_shape": list(archive_pairs.shape),
                "candidate_shape": list(candidate_255.shape),
            }
        )
        return report

    archive_255 = np.ascontiguousarray(archive_pairs, dtype=np.float32)
    delta = np.abs(candidate_255.astype(np.float64) - archive_255.astype(np.float64))
    max_abs_delta = float(np.max(delta)) if delta.size else 0.0
    archive_pairs_sha = _array_sha256(archive_255)
    candidate_pairs_sha = _array_sha256(candidate_255)
    ok = bool(candidate_pairs_sha == archive_pairs_sha or max_abs_delta <= float(atol))
    report.update(
        {
            "archive_runtime_candidate_custody": ok,
            "candidate_pairs_source": "archive_runtime_reconstruction_or_verified_equal",
            "archive_runtime_candidate_pairs_sha256": archive_pairs_sha,
            "candidate_pairs_sha256_rgb255": candidate_pairs_sha,
            "max_abs_delta_vs_archive_runtime_candidate": max_abs_delta,
            "blocker": None if ok else "candidate_pairs_do_not_match_archive_runtime_reconstruction",
        }
    )
    return report


def build_z8_full_video_mlx_vjp_surface_shard(
    archive_bytes: bytes,
    *,
    reference_pairs_rgb: Any,
    candidate_pairs_rgb: Any,
    mlx_scorer: Any,
    config: Z8FullVideoMlxVjpShardConfig,
    archive_runtime_candidate_pairs_rgb: Any | None = None,
) -> dict[str, Any]:
    """Emit one exact chunked MLX P18/P19 VJP shard over pair RGB pixels.

    ``reference_pairs_rgb`` and ``candidate_pairs_rgb`` must describe the same
    full-video pair grid. The function slices ``config.pair_start:pair_end``,
    computes MLX VJPs of a nonsmooth SegNet margin surrogate and the PoseNet
    MSE term through scorer preprocessing, and converts the two gradients into
    the canonical joint water-fill surface. No optimizer update is performed.
    """

    import mlx.core as mx

    ref_full = _as_pair_rgb_array(reference_pairs_rgb, name="reference_pairs_rgb")
    cand_full = _as_pair_rgb_array(candidate_pairs_rgb, name="candidate_pairs_rgb")
    if ref_full.shape != cand_full.shape:
        raise ValueError(
            "reference_pairs_rgb and candidate_pairs_rgb must have identical shape; "
            f"got {ref_full.shape} vs {cand_full.shape}"
        )
    if ref_full.shape[0] != config.full_video_pair_count:
        raise ValueError(
            f"full_video_pair_count must match pair array length: {config.full_video_pair_count} vs {ref_full.shape[0]}"
        )
    custody = _candidate_archive_runtime_custody_report(
        archive_bytes,
        candidate_pairs_rgb=cand_full,
        rgb_value_range=float(config.rgb_value_range),
        required=bool(config.require_archive_runtime_candidate_custody),
        atol=float(config.candidate_custody_atol),
        archive_runtime_candidate_pairs_rgb=archive_runtime_candidate_pairs_rgb,
    )
    if config.require_archive_runtime_candidate_custody and not custody["archive_runtime_candidate_custody"]:
        raise ValueError(str(custody["blocker"]))
    ref = np.ascontiguousarray(
        ref_full[config.pair_start : config.pair_end] * (255.0 / float(config.rgb_value_range)),
        dtype=np.float32,
    )
    cand = np.ascontiguousarray(
        cand_full[config.pair_start : config.pair_end] * (255.0 / float(config.rgb_value_range)),
        dtype=np.float32,
    )
    archive_sha = _sha256_bytes(archive_bytes)

    ref_mx = mx.array(ref)
    ref_seg_input, ref_pose_input = _mlx_pairs_to_scorer_inputs_nhwc(
        ref_mx,
        scorer_hw=config.scorer_hw,
    )
    ref_seg_logits = mlx_scorer.segnet(ref_seg_input)
    ref_classes = mx.argmax(ref_seg_logits, axis=-1)
    ref_pose = mlx_scorer.posenet(ref_pose_input)["pose"]
    mx.eval(ref_classes, ref_pose)

    cand_mx = mx.array(cand)

    def seg_margin_loss(pair_rgb_255: Any) -> Any:
        seg_input, _pose_input = _mlx_pairs_to_scorer_inputs_nhwc(
            pair_rgb_255,
            scorer_hw=config.scorer_hw,
        )
        logits = mlx_scorer.segnet(seg_input)
        class_count = int(logits.shape[-1])
        labels = mx.arange(class_count, dtype=ref_classes.dtype)
        one_hot = labels.reshape((1, 1, 1, class_count)) == ref_classes[..., None]
        true_logit = mx.sum(mx.where(one_hot, logits, 0.0), axis=-1)
        other_logit = mx.max(mx.where(one_hot, -1.0e9, logits), axis=-1)
        hinge = mx.maximum(0.0, other_logit - true_logit + float(config.seg_margin_delta))
        return mx.mean(hinge)

    pose_axis_count = min(6, int(ref_pose.shape[-1]))

    def pose_mse_loss(pair_rgb_255: Any) -> Any:
        _seg_input, pose_input = _mlx_pairs_to_scorer_inputs_nhwc(
            pair_rgb_255,
            scorer_hw=config.scorer_hw,
        )
        pose = mlx_scorer.posenet(pose_input)["pose"]
        dims = min(pose_axis_count, int(pose.shape[-1]))
        diff = pose[..., :dims] - ref_pose[..., :dims]
        return mx.mean(diff * diff)

    seg_grad = mx.grad(seg_margin_loss)(cand_mx)
    pose_grad = mx.grad(pose_mse_loss)(cand_mx)
    mx.eval(seg_grad, pose_grad)
    objective_shard_weight = float(config.pair_end - config.pair_start) / float(config.full_video_pair_count)
    seg_grad_np = (
        np.abs(_mlx_value_to_numpy(seg_grad)).astype(np.float64, copy=False) * objective_shard_weight
    )
    pose_grad_np = (
        np.abs(_mlx_value_to_numpy(pose_grad)).astype(np.float64, copy=False) * objective_shard_weight
    )

    full_atom_count = int(config.full_video_pair_count * np.prod(cand_full.shape[1:]))
    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg_grad_np,
        pose_jacobian=pose_grad_np[..., None],
        config=JointP18P19WaterfillConfig(
            d_pose=float(config.full_video_d_pose),
            pose_inverse_variance=(1.0,),
            target_mode=config.normalized_target_mode,
            pose_null_threshold=float(config.pose_null_threshold),
            evidence_scope=PROPOSAL_ONLY_SCOPE,
            full_video_atom_count=full_atom_count,
            linearization_archive_sha=archive_sha,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
    )
    joint = np.asarray(surface["joint_weight"], dtype=np.float64)
    mask = np.asarray(surface["rate_attack_deadzone_mask"], dtype=bool)
    shard_pair_count = int(config.pair_end - config.pair_start)
    return {
        "schema": Z8_FULL_VIDEO_VJP_SURFACE_SHARD_SCHEMA,
        "surface_generation_backend": Z8_FULL_VIDEO_VJP_MLX_SHARD_BACKEND,
        "local_axis": EVIDENCE_TAG_MLX,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "target_mode": config.normalized_target_mode,
        "archive_sha256": archive_sha,
        "linearization_archive_sha": archive_sha,
        "shard_index": int(config.shard_index),
        "pair_start": int(config.pair_start),
        "pair_end": int(config.pair_end),
        "pair_count": shard_pair_count,
        "full_video_pair_count": int(config.full_video_pair_count),
        "full_video_atom_count": full_atom_count,
        "covered_atom_count": int(joint.size),
        "gradient_reduction_semantics": FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        "pair_chunk_updates_forbidden": True,
        "gradient_values_are_full_video_objective_contributions": True,
        "full_video_objective_shard_weight": objective_shard_weight,
        "optimizer_update_applied": False,
        "budget_spend_authority": False,
        "optimizer_update_authority": False,
        "surface_relinearization_required_after_accepted_mutation": True,
        "vjp_coordinate_system": "candidate_pair_rgb_before_scorer_preprocess",
        "preprocess_backend": "mlx_bilinear_resize_align_corners_false_plus_mlx_yuv6",
        "segnet_loss_kind": "reference_argmax_nonsmooth_margin_hinge_vjp",
        "seg_margin_delta": float(config.seg_margin_delta),
        "pose_loss_kind": "reference_pose_head_first6_mse_vjp",
        "pose_surface_kind": SCALAR_POSE_LOSS_VJP_SURFACE_KIND,
        "pose_axis_count": int(pose_axis_count),
        "pose_inverse_variance_source": "unit_scalar_proxy_not_contest_mahalanobis",
        "pose_jacobian_abs_is_true_jacobian": False,
        "pose_surface_authority": False,
        "pose_surface_blockers": [P19_POSE_SURFACE_BLOCKER],
        "full_video_d_pose": float(config.full_video_d_pose),
        "pose_null_threshold": float(config.pose_null_threshold),
        "scorer_hw": [int(config.scorer_hw[0]), int(config.scorer_hw[1])],
        "rgb_value_range": float(config.rgb_value_range),
        "reference_pairs_sha256": _array_sha256(ref),
        "candidate_pairs_sha256": _array_sha256(cand),
        "archive_runtime_candidate_custody": bool(custody["archive_runtime_candidate_custody"]),
        "archive_runtime_candidate_custody_report": custody,
        "segnet_vjp_abs_max": float(seg_grad_np.max(initial=0.0)),
        "pose_vjp_abs_max": float(pose_grad_np.max(initial=0.0)),
        "segnet_argmax_gradient_abs": seg_grad_np,
        "pose_jacobian_abs": pose_grad_np,
        "joint_weight": joint,
        "rate_attack_deadzone_mask": mask,
        "joint_surface_report": {
            key: value
            for key, value in surface.items()
            if key
            not in {
                "joint_weight",
                "rate_attack_deadzone_mask",
                "safe_rate_spend_mask",
                "distortion_protect_mask",
                "segnet_term",
                "pose_term",
                "pose_null_mask",
            }
        },
        **FALSE_AUTHORITY,
    }


def reconstruct_z8_archive_pairs_rgb255(archive_bytes: bytes) -> np.ndarray:
    """Reconstruct Z8 archive pairs as ``(pairs,2,H,W,3)`` float32 RGB in [0,255]."""

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    binding, pair_pyramids, _stats = projected_pair_pyramids_from_archive_bytes(archive_bytes)
    pairs: list[np.ndarray] = []
    for pyramid in pair_pyramids:
        r0, r1 = reconstruct_pair_rgb_from_pyramid(binding, pyramid)
        f0 = np.transpose(np.asarray(r0[0], dtype=np.float32), (1, 2, 0))
        f1 = np.transpose(np.asarray(r1[0], dtype=np.float32), (1, 2, 0))
        pairs.append(np.stack([f0, f1], axis=0))
    if not pairs:
        raise ValueError("Z8 archive reconstructs zero pairs")
    return np.clip(np.stack(pairs, axis=0) * 255.0, 0.0, 255.0).astype(np.float32)


def compute_full_video_mlx_pose_distortion(
    *,
    reference_pairs_rgb: Any,
    candidate_pairs_rgb: Any,
    mlx_scorer: Any,
    rgb_value_range: float = 255.0,
    scorer_hw: tuple[int, int] = (384, 512),
    pair_chunk_size: int = 64,
) -> float:
    """Compute full-video PoseNet d_pose with MLX scorer preprocessing."""

    import mlx.core as mx

    ref = _as_pair_rgb_array(reference_pairs_rgb, name="reference_pairs_rgb")
    cand = _as_pair_rgb_array(candidate_pairs_rgb, name="candidate_pairs_rgb")
    if ref.shape != cand.shape:
        raise ValueError(f"reference/candidate pair shapes disagree: {ref.shape} vs {cand.shape}")
    if rgb_value_range <= 0.0:
        raise ValueError("rgb_value_range must be positive")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    pose_sum = 0.0
    pose_count = 0
    scale = 255.0 / float(rgb_value_range)
    for start in range(0, int(ref.shape[0]), int(pair_chunk_size)):
        end = min(int(ref.shape[0]), start + int(pair_chunk_size))
        ref_mx = mx.array(np.ascontiguousarray(ref[start:end] * scale, dtype=np.float32))
        cand_mx = mx.array(np.ascontiguousarray(cand[start:end] * scale, dtype=np.float32))
        _ref_seg, ref_pose_input = _mlx_pairs_to_scorer_inputs_nhwc(ref_mx, scorer_hw=scorer_hw)
        _cand_seg, cand_pose_input = _mlx_pairs_to_scorer_inputs_nhwc(cand_mx, scorer_hw=scorer_hw)
        ref_pose = mlx_scorer.posenet(ref_pose_input)["pose"]
        cand_pose = mlx_scorer.posenet(cand_pose_input)["pose"]
        dims = min(6, int(ref_pose.shape[-1]), int(cand_pose.shape[-1]))
        diff = cand_pose[..., :dims] - ref_pose[..., :dims]
        per_pair = mx.mean(diff * diff, axis=1)
        per_pair_np = _mlx_value_to_numpy(per_pair).astype(np.float64, copy=False)
        pose_sum += float(np.sum(per_pair_np))
        pose_count += int(per_pair_np.size)
    return float(pose_sum / max(pose_count, 1))


def compute_full_video_mlx_distortion_replay(
    *,
    reference_pairs_rgb: Any,
    candidate_pairs_rgb: Any,
    mlx_scorer: Any,
    archive_rate_bytes: int,
    archive_rate_bytes_source: str,
    rgb_value_range: float = 255.0,
    scorer_hw: tuple[int, int] = (384, 512),
    pair_chunk_size: int = 64,
    rate_normalizer_bytes: float = CONTEST_RATE_NORMALIZER_BYTES,
) -> dict[str, Any]:
    """Replay the full-video local MLX scorer action for one candidate."""

    import mlx.core as mx

    ref = _as_pair_rgb_array(reference_pairs_rgb, name="reference_pairs_rgb")
    cand = _as_pair_rgb_array(candidate_pairs_rgb, name="candidate_pairs_rgb")
    if ref.shape != cand.shape:
        raise ValueError(f"reference/candidate pair shapes disagree: {ref.shape} vs {cand.shape}")
    if rgb_value_range <= 0.0:
        raise ValueError("rgb_value_range must be positive")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    if archive_rate_bytes < 0:
        raise ValueError("archive_rate_bytes must be >= 0")
    if rate_normalizer_bytes <= 0.0:
        raise ValueError("rate_normalizer_bytes must be positive")

    scale = 255.0 / float(rgb_value_range)
    seg_sum = 0.0
    seg_count = 0
    pose_sum = 0.0
    pose_count = 0
    for start in range(0, int(ref.shape[0]), int(pair_chunk_size)):
        end = min(int(ref.shape[0]), start + int(pair_chunk_size))
        ref_mx = mx.array(np.ascontiguousarray(ref[start:end] * scale, dtype=np.float32))
        cand_mx = mx.array(np.ascontiguousarray(cand[start:end] * scale, dtype=np.float32))
        ref_seg_input, ref_pose_input = _mlx_pairs_to_scorer_inputs_nhwc(ref_mx, scorer_hw=scorer_hw)
        cand_seg_input, cand_pose_input = _mlx_pairs_to_scorer_inputs_nhwc(cand_mx, scorer_hw=scorer_hw)
        ref_seg = mlx_scorer.segnet(ref_seg_input)
        cand_seg = mlx_scorer.segnet(cand_seg_input)
        ref_pose = mlx_scorer.posenet(ref_pose_input)["pose"]
        cand_pose = mlx_scorer.posenet(cand_pose_input)["pose"]

        seg_diff = mx.argmax(ref_seg, axis=-1) != mx.argmax(cand_seg, axis=-1)
        seg_np = _mlx_value_to_numpy(seg_diff).astype(np.float64, copy=False)
        seg_sum += float(np.sum(seg_np))
        seg_count += int(seg_np.size)

        dims = min(6, int(ref_pose.shape[-1]), int(cand_pose.shape[-1]))
        pose_diff = cand_pose[..., :dims] - ref_pose[..., :dims]
        pose_np = _mlx_value_to_numpy(pose_diff * pose_diff).astype(np.float64, copy=False)
        pose_sum += float(np.sum(pose_np))
        pose_count += int(pose_np.size)

    d_seg = float(seg_sum / max(seg_count, 1))
    d_pose = float(pose_sum / max(pose_count, 1))
    rate = float(archive_rate_bytes) / float(rate_normalizer_bytes)
    action = 100.0 * d_seg + math.sqrt(max(0.0, 10.0 * d_pose)) + 25.0 * rate
    return {
        "schema": Z8_FULL_VIDEO_MLX_REPLAY_SCHEMA,
        "local_axis": EVIDENCE_TAG_MLX,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "pair_count": int(ref.shape[0]),
        "frame_count": int(ref.shape[0] * 2),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "rate": rate,
        "archive_rate_bytes": int(archive_rate_bytes),
        "archive_rate_bytes_source": archive_rate_bytes_source,
        "rate_normalizer_bytes": float(rate_normalizer_bytes),
        "contest_action_proxy": float(action),
        "full_video_local_replay_executed": True,
        "full_video_local_replay_scope": "full_video",
        "replay_ok": True,
        **FALSE_AUTHORITY,
    }


def build_z8_full_video_mlx_replay_evaluator(
    *,
    reference_pairs_rgb: Any,
    mlx_scorer: Any,
    rgb_value_range: float = 255.0,
    scorer_hw: tuple[int, int] = (384, 512),
    pair_chunk_size: int = 64,
    rate_source: str = "byte_closed_zip",
    rate_normalizer_bytes: float = CONTEST_RATE_NORMALIZER_BYTES,
    repo_root: str | Path | None = None,
    artifact_dir: str | Path | None = None,
):
    """Return a candidate-archive replay gate for the relinearized loop."""

    if rate_source not in {"payload_bytes", "byte_closed_zip"}:
        raise ValueError("rate_source must be 'payload_bytes' or 'byte_closed_zip'")
    ref = _as_pair_rgb_array(reference_pairs_rgb, name="reference_pairs_rgb")
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None

    def _byte_closed_rate_bytes(
        candidate_archive: bytes,
        *,
        iteration_index: int,
        candidate_index: int,
    ) -> tuple[int, str | None, str | None]:
        from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
            export_z8hpc1_archive_bytes,
        )

        if artifact_root is not None:
            out_dir = (
                artifact_root
                / "local_replay_byte_closed"
                / f"iter_{iteration_index:04d}_cand_{candidate_index:05d}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            archive_zip_path, archive_zip_sha, archive_zip_bytes = export_z8hpc1_archive_bytes(
                candidate_archive,
                out_dir,
                repo_root=repo_root,
                emit_archive_bound_candidate_package=False,
                emit_byte_mutation_proof=False,
                emit_runtime_payload_bridge_report=False,
                retain_receiver_proof_output=False,
            )
            return int(archive_zip_bytes), archive_zip_path.as_posix(), str(archive_zip_sha)

        with tempfile.TemporaryDirectory(prefix="z8_mlx_replay_zip_") as tmp:
            archive_zip_path, archive_zip_sha, archive_zip_bytes = export_z8hpc1_archive_bytes(
                candidate_archive,
                Path(tmp),
                repo_root=repo_root,
                emit_archive_bound_candidate_package=False,
                emit_byte_mutation_proof=False,
                emit_runtime_payload_bridge_report=False,
                retain_receiver_proof_output=False,
            )
            return int(archive_zip_bytes), archive_zip_path.as_posix(), str(archive_zip_sha)

    def _evaluator(
        *,
        candidate_archive_bytes: bytes,
        source_archive_bytes: bytes,
        current_archive_bytes: bytes,
        iteration_index: int,
        candidate_index: int,
        candidate_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_archive = bytes(candidate_archive_bytes)
        candidate_pairs = reconstruct_z8_archive_pairs_rgb255(candidate_archive)
        if candidate_pairs.shape != ref.shape:
            raise ValueError(
                "reference_pairs_rgb shape must match candidate archive reconstruction shape: "
                f"{ref.shape} vs {candidate_pairs.shape}"
            )
        archive_zip_path: str | None = None
        archive_zip_sha: str | None = None
        if rate_source == "byte_closed_zip":
            rate_bytes, archive_zip_path, archive_zip_sha = _byte_closed_rate_bytes(
                candidate_archive,
                iteration_index=int(iteration_index),
                candidate_index=int(candidate_index),
            )
            rate_bytes_source = "byte_closed_archive_zip"
        else:
            rate_bytes = len(candidate_archive)
            rate_bytes_source = "z8hpc1_payload_bytes"
        report = compute_full_video_mlx_distortion_replay(
            reference_pairs_rgb=ref,
            candidate_pairs_rgb=candidate_pairs,
            mlx_scorer=mlx_scorer,
            archive_rate_bytes=rate_bytes,
            archive_rate_bytes_source=rate_bytes_source,
            rgb_value_range=rgb_value_range,
            scorer_hw=scorer_hw,
            pair_chunk_size=pair_chunk_size,
            rate_normalizer_bytes=rate_normalizer_bytes,
        )
        report.update(
            {
                "candidate_archive_sha256": _sha256_bytes(candidate_archive),
                "source_archive_sha256": _sha256_bytes(bytes(source_archive_bytes)),
                "current_archive_sha256": _sha256_bytes(bytes(current_archive_bytes)),
                "iteration_index": int(iteration_index),
                "candidate_index": int(candidate_index),
                "candidate_metadata": _jsonable(dict(candidate_metadata or {})),
                "rate_source": rate_source,
                "byte_closed_archive_zip_path": archive_zip_path if artifact_root is not None else None,
                "byte_closed_archive_zip_sha256": archive_zip_sha,
            }
        )
        return report

    return _evaluator


def build_z8_full_video_mlx_surface_provider(
    *,
    reference_pairs_rgb: Any,
    mlx_scorer: Any,
    acquisition_config: Z8FullVideoVjpAcquisitionConfig | None = None,
    rgb_value_range: float = 255.0,
    scorer_hw: tuple[int, int] = (384, 512),
    seg_margin_delta: float = 1.0,
    pose_null_threshold: float = 1e-8,
    artifact_dir: str | Path | None = None,
):
    """Return a fresh-surface provider for relinearized dead-zone search.

    The returned callable implements the loop spine's acquisition side:
    current archive -> reconstruct candidate pairs -> full-video d_pose ->
    all MLX VJP shards -> deterministic reduction bundle. It performs no
    accept/reject decision itself; the materializer/search loop remains the
    authority for hard archive projection and replay.
    """

    ref = _as_pair_rgb_array(reference_pairs_rgb, name="reference_pairs_rgb")
    cfg = acquisition_config or Z8FullVideoVjpAcquisitionConfig()
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None

    def _provider(iteration_index: int, current_archive_bytes: bytes) -> dict[str, Any]:
        current_archive = bytes(current_archive_bytes)
        candidate_pairs = reconstruct_z8_archive_pairs_rgb255(current_archive)
        if candidate_pairs.shape != ref.shape:
            raise ValueError(
                "reference_pairs_rgb shape must match archive reconstruction shape: "
                f"{ref.shape} vs {candidate_pairs.shape}"
            )
        d_pose = compute_full_video_mlx_pose_distortion(
            reference_pairs_rgb=ref,
            candidate_pairs_rgb=candidate_pairs,
            mlx_scorer=mlx_scorer,
            rgb_value_range=rgb_value_range,
            scorer_hw=scorer_hw,
            pair_chunk_size=cfg.pair_chunk_size,
        )
        plan = build_z8_full_video_vjp_acquisition_plan(current_archive, config=cfg)
        shards: list[dict[str, Any]] = []
        iter_dir = None
        if artifact_root is not None:
            iter_dir = artifact_root / f"iteration_{int(iteration_index):04d}"
            write_z8_full_video_vjp_acquisition_plan(current_archive, iter_dir / "plan", config=cfg)
        for shard_row in plan["pair_shards"]:
            shard = build_z8_full_video_mlx_vjp_surface_shard(
                current_archive,
                reference_pairs_rgb=ref,
                candidate_pairs_rgb=candidate_pairs,
                mlx_scorer=mlx_scorer,
                config=Z8FullVideoMlxVjpShardConfig(
                    shard_index=int(shard_row["shard_index"]),
                    pair_start=int(shard_row["pair_start"]),
                    pair_end=int(shard_row["pair_end"]),
                    full_video_pair_count=int(plan["archive_num_pairs"]),
                    full_video_d_pose=d_pose,
                    target_mode=cfg.normalized_target_mode,
                    rgb_value_range=rgb_value_range,
                    scorer_hw=scorer_hw,
                    seg_margin_delta=seg_margin_delta,
                    pose_null_threshold=pose_null_threshold,
                ),
                archive_runtime_candidate_pairs_rgb=candidate_pairs,
            )
            shards.append(shard)
            if iter_dir is not None:
                write_z8_full_video_vjp_surface_shard(shard, iter_dir / "shards")
        bundle = assemble_z8_full_video_vjp_surface_bundle(
            current_archive,
            shard_surfaces=shards,
            config=cfg,
        )
        bundle["provider_iteration_index"] = int(iteration_index)
        bundle["full_video_d_pose"] = d_pose
        bundle["surface_provider_backend"] = "z8_full_video_mlx_archive_fresh_surface_provider.v1"
        if iter_dir is not None:
            manifest = write_z8_full_video_vjp_surface_bundle(bundle, iter_dir / "bundle")
            bundle["surface_provider_bundle_manifest"] = manifest
        return bundle

    return _provider


def _pair_shards(num_pairs: int, chunk_size: int) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, int(num_pairs), int(chunk_size))):
        end = min(start + int(chunk_size), int(num_pairs))
        shards.append(
            {
                "schema": "z8_full_video_vjp_pair_shard.v1",
                "shard_index": int(shard_index),
                "pair_start": int(start),
                "pair_end": int(end),
                "pair_count": int(end - start),
                "execution_hint": "mlx_full_video_resident_pair_chunk_vjp",
            }
        )
    return shards


def build_z8_full_video_vjp_acquisition_plan(
    archive_bytes: bytes,
    *,
    config: Z8FullVideoVjpAcquisitionConfig | None = None,
) -> dict[str, Any]:
    """Return a deterministic work plan for full-video local VJP acquisition."""

    cfg = config or Z8FullVideoVjpAcquisitionConfig()
    arc = parse_archive(archive_bytes)
    archive_sha = _sha256_bytes(archive_bytes)
    target_mode = cfg.normalized_target_mode
    shards = _pair_shards(arc.num_pairs, cfg.pair_chunk_size)
    parallel_workers = cfg.parallel_workers or min(
        len(shards), max(1, (arc.num_pairs + cfg.pair_chunk_size - 1) // cfg.pair_chunk_size)
    )
    return {
        "schema": Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA,
        "local_axis": "[macOS-MLX research-signal]",
        "archive_sha256": archive_sha,
        "archive_num_pairs": int(arc.num_pairs),
        "target_mode": target_mode,
        "declared_overfit_allowed": target_mode_declares_overfit_allowed(target_mode),
        "corpus_manifest_required": target_mode_requires_corpus_manifest(target_mode),
        "corpus_manifest_path": cfg.corpus_manifest_path,
        "full_video_vjp_is_first_class_acquisition_lane": True,
        "full_video_residency_required": True,
        "surface_linearization_archive_sha_required": True,
        "surface_relinearization_required_after_accepted_mutation": True,
        "pair_chunk_updates_forbidden": True,
        "gradient_reduction_semantics": FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        "optimizer_update_semantics": SINGLE_UPDATE_AFTER_FULL_REDUCTION,
        "minibatch_window_gradients_policy": (
            "ranking_probe_only_between_full_video_passes"
            if cfg.allow_minibatch_probe_between_full_passes
            else "disabled"
        ),
        "minibatch_window_gradients_budget_spend_authority": False,
        "contest_mode_budget_spend_requires_full_video_archive_pinned_surface": (
            target_mode == CONTEST_VIDEO_OVERFIT_MODE
        ),
        "production_mode_requires_declared_corpus_manifest": target_mode
        in {
            CORPUS_GENERALIZATION_MODE,
            HYBRID_CONTEST_PLUS_CORPUS_MODE,
        },
        "pair_chunk_size": int(cfg.pair_chunk_size),
        "parallel_workers": int(parallel_workers),
        "shard_count": len(shards),
        "pair_shards": shards,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _as_array(row: Mapping[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    if key not in row:
        raise ValueError(f"surface shard missing {key}")
    arr = np.asarray(row[key], dtype=dtype)
    if arr.ndim != 5:
        raise ValueError(f"{key} shard must have shape (pairs, frames, H, W, C); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        nonfinite = int(arr.size - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"{key} shard contains non-finite values ({nonfinite}/{arr.size})")
    return arr


def _finite_surface_array(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        nonfinite = int(arr.size - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"{name} contains non-finite values ({nonfinite}/{arr.size})")
    return arr


def assemble_z8_full_video_vjp_surface_bundle(
    archive_bytes: bytes,
    *,
    shard_surfaces: Sequence[Mapping[str, Any]],
    config: Z8FullVideoVjpAcquisitionConfig | None = None,
) -> dict[str, Any]:
    """Assemble pair-sharded VJP outputs into one materializer-ready surface."""

    cfg = config or Z8FullVideoVjpAcquisitionConfig()
    plan = build_z8_full_video_vjp_acquisition_plan(archive_bytes, config=cfg)
    archive_sha = str(plan["archive_sha256"])
    expected_start = 0
    joint_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    seg_grad_chunks: list[np.ndarray] = []
    pose_grad_chunks: list[np.ndarray] = []
    shard_reports: list[dict[str, Any]] = []
    pose_surface_blockers: list[str] = []
    full_video_d_pose: float | None = None
    pose_null_threshold: float = 1e-8
    for shard_index, raw in enumerate(sorted(shard_surfaces, key=lambda row: int(row.get("pair_start", -1)))):
        if (
            raw.get("optimizer_update_applied")
            or raw.get("budget_spend_authority")
            or raw.get("optimizer_update_authority")
            or raw.get("gradient_reduction_authority")
        ):
            raise ValueError(
                "full-video VJP shards cannot carry optimizer update authority; "
                "assemble the complete archive-pinned pair grid before updating"
            )
        if raw.get("archive_runtime_candidate_custody") is not True:
            blocker = (raw.get("archive_runtime_candidate_custody_report", {}) or {}).get(
                "blocker",
                "archive_runtime_candidate_custody_missing",
            )
            raise ValueError(f"full-video VJP shard lacks archive runtime candidate custody: {blocker}")
        if raw.get("gradient_values_are_full_video_objective_contributions") is not True:
            raise ValueError("full-video VJP shard gradients are not full-video objective contributions")
        pair_start = int(raw.get("pair_start", -1))
        pair_end = int(raw.get("pair_end", -1))
        if pair_start != expected_start or pair_end <= pair_start:
            raise ValueError(
                "full-video VJP shards must be contiguous and ordered; "
                f"expected_start={expected_start} got=({pair_start},{pair_end})"
            )
        pinned = str(raw.get("linearization_archive_sha") or "")
        if pinned != archive_sha:
            raise ValueError(
                "full-video VJP shard linearization archive mismatch: "
                f"expected={archive_sha} got={pinned or '<missing>'}"
            )
        joint = _as_array(raw, "joint_weight", dtype=np.float64)
        mask = _as_array(raw, "rate_attack_deadzone_mask", dtype=bool)
        seg_grad = _as_array(raw, "segnet_argmax_gradient_abs", dtype=np.float64)
        pose_grad = _as_array(raw, "pose_jacobian_abs", dtype=np.float64)
        if joint.shape[0] != pair_end - pair_start or mask.shape != joint.shape:
            raise ValueError("surface shard pair span and tensor shape disagree")
        if seg_grad.shape != joint.shape or pose_grad.shape != joint.shape:
            raise ValueError("surface shard raw gradient tensors must match joint_weight shape")
        pose_surface_kind = str(raw.get("pose_surface_kind") or "missing_pose_surface_kind")
        pose_surface_true = (
            pose_surface_kind == TRUE_P19_POSE_SURFACE_KIND
            and raw.get("pose_jacobian_abs_is_true_jacobian") is True
            and raw.get("pose_surface_authority") is True
        )
        if not pose_surface_true:
            shard_pose_blockers = list(raw.get("pose_surface_blockers") or [])
            pose_surface_blockers.extend(shard_pose_blockers or [P19_POSE_SURFACE_BLOCKER])
        shard_d_pose = raw.get("full_video_d_pose")
        if shard_d_pose is None:
            raise ValueError("surface shard missing full_video_d_pose")
        if full_video_d_pose is None:
            full_video_d_pose = float(shard_d_pose)
            pose_null_threshold = float(raw.get("pose_null_threshold", pose_null_threshold))
        elif abs(full_video_d_pose - float(shard_d_pose)) > 1e-12:
            raise ValueError("surface shards disagree on full_video_d_pose")
        joint_chunks.append(joint)
        mask_chunks.append(mask)
        seg_grad_chunks.append(seg_grad)
        pose_grad_chunks.append(pose_grad)
        expected_start = pair_end
        shard_reports.append(
            {
                "schema": "z8_full_video_vjp_surface_shard_report.v1",
                "shard_index": int(raw.get("shard_index", shard_index)),
                "pair_start": pair_start,
                "pair_end": pair_end,
                "pair_count": int(pair_end - pair_start),
                "linearization_archive_sha": pinned,
                "archive_runtime_candidate_custody": True,
                "gradient_values_are_full_video_objective_contributions": True,
                "optimizer_update_applied": False,
                "pose_surface_kind": pose_surface_kind,
                "pose_surface_authority": bool(pose_surface_true),
            }
        )

    archive_num_pairs = int(plan["archive_num_pairs"])
    full_coverage = expected_start == archive_num_pairs
    if not full_coverage and not cfg.allow_partial_production_probe_surface:
        raise ValueError(
            "full-video VJP surface does not cover archive pair grid: "
            f"covered={expected_start} required={archive_num_pairs}"
        )
    pose_surface_authority = full_coverage and not pose_surface_blockers
    joint_full = np.concatenate(joint_chunks, axis=0) if joint_chunks else np.zeros((0, 2, 1, 1, 1))
    mask_full = np.concatenate(mask_chunks, axis=0) if mask_chunks else np.zeros_like(joint_full, dtype=bool)
    if full_coverage:
        seg_grad_full = np.concatenate(seg_grad_chunks, axis=0)
        pose_grad_full = np.concatenate(pose_grad_chunks, axis=0)
        global_surface = build_joint_p18_p19_waterfill_surface(
            segnet_argmax_gradient=seg_grad_full,
            pose_jacobian=pose_grad_full[..., None],
            config=JointP18P19WaterfillConfig(
                d_pose=float(full_video_d_pose if full_video_d_pose is not None else 0.0),
                pose_inverse_variance=(1.0,),
                target_mode=str(plan["target_mode"]),
                pose_null_threshold=float(pose_null_threshold),
                evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE if pose_surface_authority else PROPOSAL_ONLY_SCOPE,
                full_video_atom_count=int(seg_grad_full.size),
                linearization_archive_sha=archive_sha,
                gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
            ),
        )
        joint_full = np.asarray(global_surface["joint_weight"], dtype=np.float64)
        mask_full = np.asarray(global_surface["rate_attack_deadzone_mask"], dtype=bool)
        if not pose_surface_authority:
            global_surface = {
                **global_surface,
                "implicit_allocator_authority": False,
                "implicit_allocator_blockers": sorted(set(pose_surface_blockers)),
            }
    else:
        global_surface = {
            "implicit_allocator_authority": False,
            "implicit_allocator_blockers": ["partial_full_video_vjp_surface_probe_only"],
        }
    authority_blockers = sorted(set(pose_surface_blockers))
    budget_spend_authority = bool(full_coverage and pose_surface_authority)
    return {
        "schema": Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA,
        "surface_assembly_backend": "global_kkt_dykstra_after_full_shard_reduction.v1",
        "target_mode": plan["target_mode"],
        "local_axis": plan["local_axis"],
        "archive_sha256": archive_sha,
        "linearization_archive_sha": archive_sha,
        "evidence_scope": "full_video" if full_coverage else "proposal_only",
        "full_video_pair_count": archive_num_pairs,
        "covered_pair_count": int(expected_start),
        "pair_coverage_fraction": float(expected_start / archive_num_pairs) if archive_num_pairs else 1.0,
        "full_video_surface_coverage": bool(full_coverage),
        "full_video_vjp_is_first_class_acquisition_lane": True,
        "full_video_reduction_complete": bool(full_coverage),
        "gradient_reduction_semantics": (
            FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION if full_coverage else "proposal_or_sampled"
        ),
        "gradient_reduction_authority": bool(full_coverage),
        "minibatch_window_gradients_budget_spend_authority": False,
        "budget_spend_authority": budget_spend_authority,
        "budget_spend_blockers": authority_blockers,
        "optimizer_update_authority": budget_spend_authority,
        "optimizer_update_semantics": (
            SINGLE_UPDATE_AFTER_FULL_REDUCTION
            if budget_spend_authority
            else (
                "no_update_pose_surface_not_true_p19_jacobian"
                if full_coverage
                else "no_update_partial_surface_probe_only"
            )
        ),
        "pose_surface_kind": TRUE_P19_POSE_SURFACE_KIND
        if pose_surface_authority
        else SCALAR_POSE_LOSS_VJP_SURFACE_KIND,
        "pose_surface_authority": bool(pose_surface_authority),
        "pose_surface_blockers": authority_blockers,
        "implicit_allocator_authority": bool(
            budget_spend_authority and global_surface.get("implicit_allocator_authority", False)
        ),
        "implicit_allocator_blockers": list(global_surface.get("implicit_allocator_blockers") or []),
        "global_joint_surface_report": {
            key: value
            for key, value in global_surface.items()
            if key
            not in {
                "joint_weight",
                "rate_attack_deadzone_mask",
                "safe_rate_spend_mask",
                "distortion_protect_mask",
                "segnet_term",
                "pose_term",
                "pose_null_mask",
                "pose_mahalanobis_norm",
                "coarsening_priority",
                "rate_attack_allocation",
            }
        },
        "surface_relinearization_required_after_accepted_mutation": True,
        "joint_weight": joint_full,
        "rate_attack_deadzone_mask": mask_full,
        "shard_count": len(shard_reports),
        "shard_reports": shard_reports,
        "acquisition_plan": plan,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def write_z8_full_video_vjp_surface_bundle(
    bundle: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a materializer-ready NPZ surface plus a compact manifest."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surface_path = out_dir / "z8_full_video_vjp_surface_bundle.npz"
    joint_weight = _finite_surface_array(bundle["joint_weight"], name="joint_weight", dtype=np.float32)
    rate_attack_deadzone_mask = np.asarray(bundle["rate_attack_deadzone_mask"], dtype=bool)
    np.savez_compressed(
        surface_path,
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=rate_attack_deadzone_mask,
        linearization_archive_sha=np.asarray(str(bundle["linearization_archive_sha"])),
        evidence_scope=np.asarray(str(bundle["evidence_scope"])),
        target_mode=np.asarray(str(bundle["target_mode"])),
        gradient_reduction_semantics=np.asarray(str(bundle["gradient_reduction_semantics"])),
        gradient_reduction_authority=np.asarray(bool(bundle["gradient_reduction_authority"])),
        optimizer_update_authority=np.asarray(bool(bundle["optimizer_update_authority"])),
        optimizer_update_semantics=np.asarray(str(bundle["optimizer_update_semantics"])),
        full_video_reduction_complete=np.asarray(bool(bundle["full_video_reduction_complete"])),
        budget_spend_authority=np.asarray(bool(bundle["budget_spend_authority"])),
        implicit_allocator_authority=np.asarray(bool(bundle.get("implicit_allocator_authority", False))),
        pose_surface_kind=np.asarray(str(bundle.get("pose_surface_kind", ""))),
        pose_surface_authority=np.asarray(bool(bundle.get("pose_surface_authority", False))),
    )
    manifest = {
        "schema": "z8_full_video_vjp_surface_bundle_manifest.v1",
        "surface_bundle_schema": bundle["schema"],
        "surface_path": surface_path.as_posix(),
        "surface_sha256": _sha256_bytes(surface_path.read_bytes()),
        "archive_sha256": bundle["archive_sha256"],
        "linearization_archive_sha": bundle["linearization_archive_sha"],
        "target_mode": bundle["target_mode"],
        "evidence_scope": bundle["evidence_scope"],
        "full_video_surface_coverage": bundle["full_video_surface_coverage"],
        "covered_pair_count": bundle["covered_pair_count"],
        "full_video_pair_count": bundle["full_video_pair_count"],
        "gradient_reduction_semantics": bundle["gradient_reduction_semantics"],
        "gradient_reduction_authority": bundle["gradient_reduction_authority"],
        "budget_spend_authority": bundle["budget_spend_authority"],
        "optimizer_update_authority": bundle["optimizer_update_authority"],
        "optimizer_update_semantics": bundle["optimizer_update_semantics"],
        "budget_spend_blockers": bundle.get("budget_spend_blockers", []),
        "pose_surface_kind": bundle.get("pose_surface_kind"),
        "pose_surface_authority": bundle.get("pose_surface_authority", False),
        "pose_surface_blockers": bundle.get("pose_surface_blockers", []),
        "implicit_allocator_authority": bundle.get("implicit_allocator_authority", False),
        "implicit_allocator_blockers": bundle.get("implicit_allocator_blockers", []),
        "shard_count": bundle["shard_count"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    manifest_path = out_dir / "z8_full_video_vjp_surface_bundle_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    write_json(manifest_path, manifest)
    return manifest


def write_z8_full_video_vjp_surface_shard(
    shard: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write one archive-pinned MLX VJP shard plus queue manifest."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_index = int(shard.get("shard_index", 0))
    shard_path = out_dir / f"z8_full_video_vjp_surface_shard_{shard_index:04d}.npz"
    metadata = {
        key: value
        for key, value in shard.items()
        if key
        not in {
            "joint_weight",
            "rate_attack_deadzone_mask",
            "segnet_argmax_gradient_abs",
            "pose_jacobian_abs",
        }
    }
    joint_weight = _finite_surface_array(shard["joint_weight"], name="joint_weight", dtype=np.float32)
    seg_grad = _finite_surface_array(
        shard["segnet_argmax_gradient_abs"],
        name="segnet_argmax_gradient_abs",
        dtype=np.float32,
    )
    pose_grad = _finite_surface_array(
        shard["pose_jacobian_abs"],
        name="pose_jacobian_abs",
        dtype=np.float32,
    )
    np.savez_compressed(
        shard_path,
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=np.asarray(shard["rate_attack_deadzone_mask"], dtype=bool),
        segnet_argmax_gradient_abs=seg_grad,
        pose_jacobian_abs=pose_grad,
        shard_index=np.asarray(shard_index),
        pair_start=np.asarray(int(shard["pair_start"])),
        pair_end=np.asarray(int(shard["pair_end"])),
        linearization_archive_sha=np.asarray(str(shard["linearization_archive_sha"])),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
    )
    manifest = {
        "schema": "z8_full_video_vjp_surface_shard_manifest.v1",
        "surface_shard_schema": shard.get("schema", Z8_FULL_VIDEO_VJP_SURFACE_SHARD_SCHEMA),
        "shard_path": shard_path.as_posix(),
        "shard_sha256": _sha256_bytes(shard_path.read_bytes()),
        "archive_sha256": shard.get("archive_sha256"),
        "linearization_archive_sha": shard.get("linearization_archive_sha"),
        "target_mode": shard.get("target_mode"),
        "evidence_grade": shard.get("evidence_grade"),
        "local_axis": shard.get("local_axis"),
        "shard_index": shard_index,
        "pair_start": int(shard["pair_start"]),
        "pair_end": int(shard["pair_end"]),
        "pair_count": int(shard["pair_end"]) - int(shard["pair_start"]),
        "full_video_pair_count": shard.get("full_video_pair_count"),
        "gradient_reduction_semantics": shard.get("gradient_reduction_semantics"),
        "archive_runtime_candidate_custody": bool(shard.get("archive_runtime_candidate_custody", False)),
        "gradient_values_are_full_video_objective_contributions": bool(
            shard.get("gradient_values_are_full_video_objective_contributions", False)
        ),
        "optimizer_update_applied": False,
        "budget_spend_authority": False,
        **FALSE_AUTHORITY,
    }
    manifest_path = out_dir / f"z8_full_video_vjp_surface_shard_{shard_index:04d}_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    write_json(manifest_path, manifest)
    return manifest


def write_z8_full_video_vjp_acquisition_plan(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    config: Z8FullVideoVjpAcquisitionConfig | None = None,
) -> dict[str, Any]:
    """Write the deterministic full-video VJP shard plan for queue execution."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_z8_full_video_vjp_acquisition_plan(archive_bytes, config=config)
    plan_path = out_dir / "z8_full_video_vjp_acquisition_plan.json"
    write_json(plan_path, plan)
    plan["plan_path"] = plan_path.as_posix()
    write_json(plan_path, plan)
    return plan


def load_z8_full_video_vjp_surface_shard_file(path: str | Path) -> dict[str, Any]:
    """Load one archive-pinned full-video VJP shard from NPZ or JSON."""

    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        required = {
            "joint_weight",
            "rate_attack_deadzone_mask",
            "segnet_argmax_gradient_abs",
            "pose_jacobian_abs",
            "pair_start",
            "pair_end",
            "linearization_archive_sha",
        }
        missing = sorted(key for key in required if key not in data)
        if missing:
            raise ValueError(f"{p} missing required shard keys: {missing}")
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata_text = str(np.asarray(data["metadata_json"]).reshape(-1)[0])
            metadata = json.loads(metadata_text) if metadata_text else {}
        return {
            **metadata,
            "shard_index": int(np.asarray(data.get("shard_index", 0)).reshape(-1)[0]),
            "pair_start": int(np.asarray(data["pair_start"]).reshape(-1)[0]),
            "pair_end": int(np.asarray(data["pair_end"]).reshape(-1)[0]),
            "linearization_archive_sha": str(np.asarray(data["linearization_archive_sha"]).reshape(-1)[0]),
            "joint_weight": np.asarray(data["joint_weight"], dtype=np.float64),
            "rate_attack_deadzone_mask": np.asarray(
                data["rate_attack_deadzone_mask"],
                dtype=bool,
            ),
            "segnet_argmax_gradient_abs": np.asarray(data["segnet_argmax_gradient_abs"], dtype=np.float64),
            "pose_jacobian_abs": np.asarray(data["pose_jacobian_abs"], dtype=np.float64),
            "optimizer_update_applied": bool(
                np.asarray(metadata.get("optimizer_update_applied", False)).reshape(-1)[0]
            ),
            "optimizer_update_authority": bool(
                np.asarray(metadata.get("optimizer_update_authority", False)).reshape(-1)[0]
            ),
            "gradient_reduction_authority": bool(
                np.asarray(metadata.get("gradient_reduction_authority", False)).reshape(-1)[0]
            ),
            "budget_spend_authority": bool(np.asarray(metadata.get("budget_spend_authority", False)).reshape(-1)[0]),
            "archive_runtime_candidate_custody": bool(
                np.asarray(metadata.get("archive_runtime_candidate_custody", False)).reshape(-1)[0]
            ),
            "gradient_values_are_full_video_objective_contributions": bool(
                np.asarray(metadata.get("gradient_values_are_full_video_objective_contributions", False)).reshape(-1)[0]
            ),
        }
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {
        "shard_index": int(payload.get("shard_index", 0)),
        "pair_start": int(payload["pair_start"]),
        "pair_end": int(payload["pair_end"]),
        "linearization_archive_sha": str(payload["linearization_archive_sha"]),
        "joint_weight": np.asarray(payload["joint_weight"], dtype=np.float64),
        "rate_attack_deadzone_mask": np.asarray(
            payload["rate_attack_deadzone_mask"],
            dtype=bool,
        ),
        "segnet_argmax_gradient_abs": np.asarray(payload["segnet_argmax_gradient_abs"], dtype=np.float64),
        "pose_jacobian_abs": np.asarray(payload["pose_jacobian_abs"], dtype=np.float64),
        "optimizer_update_applied": bool(payload.get("optimizer_update_applied", False)),
        "optimizer_update_authority": bool(payload.get("optimizer_update_authority", False)),
        "gradient_reduction_authority": bool(payload.get("gradient_reduction_authority", False)),
        "budget_spend_authority": bool(payload.get("budget_spend_authority", False)),
        "archive_runtime_candidate_custody": bool(payload.get("archive_runtime_candidate_custody", False)),
        "gradient_values_are_full_video_objective_contributions": bool(
            payload.get("gradient_values_are_full_video_objective_contributions", False)
        ),
    }


def build_z8_full_video_vjp_acquisition_contract() -> dict[str, Any]:
    """Return the stable contract embedded in Z8 driver metadata."""

    return {
        "schema": "z8_full_video_vjp_acquisition_contract.v1",
        "plan_schema": Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA,
        "surface_bundle_schema": Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA,
        "contest_mode": CONTEST_VIDEO_OVERFIT_MODE,
        "production_modes": [CORPUS_GENERALIZATION_MODE, HYBRID_CONTEST_PLUS_CORPUS_MODE],
        "contest_budget_spend_requires": [
            "full_video_pair_grid_coverage",
            "linearization_archive_sha_equals_current_archive_sha",
            "candidate_pairs_equal_archive_runtime_reconstruction",
            "raw_p18_p19_gradients_reduced_before_global_kkt_dykstra_allocation",
            "true_per_axis_posenet_jacobian_mahalanobis_surface",
            "single_optimizer_update_after_full_shard_reduction",
            "relinearize_after_each_accepted_archive_mutation",
            "receiver_proof_plus_exact_cpu_cuda_before_score_authority",
        ],
        "production_budget_spend_requires": [
            "declared_corpus_manifest",
            "same_surface_schema",
            "explicit_generalization_target_mode",
        ],
        "minibatch_window_gradients_role": "ranking_probe_only_between_full_video_passes",
        "mlx_execution_model": "keep_full_video_resident_accumulate_pair_chunk_vjp_in_parallel",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "P19_POSE_SURFACE_BLOCKER",
    "SCALAR_POSE_LOSS_VJP_SURFACE_KIND",
    "TRUE_P19_POSE_SURFACE_KIND",
    "Z8_FULL_VIDEO_MLX_REPLAY_SCHEMA",
    "Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA",
    "Z8_FULL_VIDEO_VJP_MLX_SHARD_BACKEND",
    "Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA",
    "Z8_FULL_VIDEO_VJP_SURFACE_SHARD_SCHEMA",
    "Z8FullVideoMlxVjpShardConfig",
    "Z8FullVideoVjpAcquisitionConfig",
    "assemble_z8_full_video_vjp_surface_bundle",
    "build_z8_full_video_mlx_replay_evaluator",
    "build_z8_full_video_mlx_surface_provider",
    "build_z8_full_video_mlx_vjp_surface_shard",
    "build_z8_full_video_vjp_acquisition_contract",
    "build_z8_full_video_vjp_acquisition_plan",
    "compute_full_video_mlx_distortion_replay",
    "compute_full_video_mlx_pose_distortion",
    "load_z8_full_video_vjp_surface_shard_file",
    "reconstruct_z8_archive_pairs_rgb255",
    "write_z8_full_video_vjp_acquisition_plan",
    "write_z8_full_video_vjp_surface_bundle",
    "write_z8_full_video_vjp_surface_shard",
]
