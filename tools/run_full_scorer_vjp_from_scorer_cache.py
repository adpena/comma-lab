#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compute MLX-first direct scorer VJP shards from retained scorer-input caches.

The runner is an acquisition tool for compact-carrier training and repair. It
backprops through the local MLX PoseNet/SegNet adapters from fixed NumPy cache
tensors and writes deterministic shard bundles. It is never score authority:
SegNet uses a differentiable cross-entropy boundary surrogate, and promotion
still requires receiver proof plus contest auth eval replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.operator_storage_waterfall import (  # noqa: E402
    operator_storage_policy_payload,
    operator_work_tiers,
    storage_preflight_artifact_catalog_metadata,
)
from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    nchw_to_nhwc,
    nhwc_to_nchw,
    temporary_mlx_device,
    torch_distortion_net_to_mlx,
)
from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    MANIFEST_CACHE_INTEGRITY_MODE,
    _load_upstream_distortion_net,
    _resolve_upstream_dir,
    load_scorer_input_cache,
)

SCHEMA = "direct_full_scorer_vjp_bundle.v1"
SHARD_SCHEMA = "direct_full_scorer_vjp_shard.v1"
TOOL = "tools/run_full_scorer_vjp_from_scorer_cache.py"
FALSE_AUTHORITY: dict[str, bool] = {
    "research_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "dispatch_attempted": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
STORAGE_PREFLIGHT_SCHEMA = "direct_full_scorer_vjp_storage_preflight.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_xray(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _full_d_pose_from_inputs(xray: dict[str, Any] | None, explicit: float | None) -> float:
    if explicit is not None:
        value = float(explicit)
    elif xray is not None:
        value = float((xray.get("component_summary") or {}).get("avg_posenet_dist"))
    else:
        raise ValueError("full-video d_pose requires --full-video-d-pose or --pair-window-json")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"full-video d_pose must be finite and >= 0, got {value}")
    return value


def _cache_paths(cache: Any) -> dict[str, Any]:
    return {
        "root": cache.root.as_posix(),
        "manifest_sha256": _sha256_file(cache.root / "manifest.json"),
        "pair_count": int(cache.pair_indices.shape[0]),
        "array_sha256": dict(cache.manifest.get("array_sha256") or {}),
        "cache_integrity": cache.cache_integrity,
    }


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    nonfinite_count = int(arr.size - finite_count)
    if finite_count:
        finite_arr = arr[finite].astype(np.float64, copy=False)
        abs_arr = np.abs(finite_arr)
        abs_sum = float(np.sum(abs_arr, dtype=np.float64))
        abs_mean = float(np.mean(abs_arr, dtype=np.float64))
        abs_max = float(np.max(abs_arr))
        l2_norm = float(np.sqrt(np.sum(finite_arr * finite_arr, dtype=np.float64)))
    else:
        abs_sum = None
        abs_mean = None
        abs_max = None
        l2_norm = None
    return {
        "finite_fraction": float(finite_count / arr.size) if arr.size else 0.0,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "nan_count": int(np.count_nonzero(np.isnan(arr))),
        "inf_count": int(np.count_nonzero(np.isinf(arr))),
        "abs_sum": abs_sum,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "l2_norm": l2_norm,
    }


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _estimate_vjp_output_bytes(
    *,
    candidate: Any,
    start_pair: int,
    stop_pair: int,
    summary_only: bool,
) -> int:
    """Conservative uncompressed shard-byte estimate for storage preflight."""

    pair_count = max(0, int(stop_pair) - int(start_pair))
    if summary_only:
        return max(64 * 1024, pair_count * 4096)
    pose = np.asarray(candidate.posenet_yuv6_pair)
    seg = np.asarray(candidate.segnet_last_rgb)
    pose_per_pair = int(np.prod(pose.shape[1:], dtype=np.int64)) if pose.ndim else 0
    seg_per_pair = int(np.prod(seg.shape[1:], dtype=np.int64)) if seg.ndim else 0
    arrays = pair_count * (pose_per_pair + seg_per_pair) * np.dtype(np.float32).itemsize
    metadata_slack = max(16 * 1024 * 1024, int(arrays * 0.25))
    return int(arrays + metadata_slack)


def _storage_preflight(
    *,
    output_dir: Path,
    requested_bytes: int,
    reserve_free_bytes: int,
    allow_local_output: bool,
    storage_plan_path: Path | None,
    cleanup_plan_path: Path | None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = operator_work_tiers(allow_local_disk=allow_local_output)
    selected = None
    for tier in tiers:
        tier_root = Path(tier.root).expanduser().resolve(strict=False)
        if _path_is_relative_to(output_dir, tier_root):
            selected = tier.to_dict()
            break
    local_output_allowed = bool(allow_local_output)
    blockers: list[str] = []
    if selected is None and not local_output_allowed:
        blockers.append("direct_full_scorer_vjp_output_dir_not_on_operator_ssd_tier")
    usage = shutil.disk_usage(output_dir)
    free_after = int(usage.free) - int(requested_bytes)
    if free_after < int(reserve_free_bytes):
        blockers.append("direct_full_scorer_vjp_output_storage_preflight_insufficient_free_space")
    payload = {
        "schema": STORAGE_PREFLIGHT_SCHEMA,
        "output_dir": output_dir.as_posix(),
        "requested_bytes": int(requested_bytes),
        "reserve_free_bytes": int(reserve_free_bytes),
        "free_bytes_before": int(usage.free),
        "free_bytes_after_estimate": int(free_after),
        "selected_storage_tier": selected,
        "allow_local_output": local_output_allowed,
        "operator_storage_policy": operator_storage_policy_payload(
            allow_local_disk=local_output_allowed
        ),
        "artifact_catalog_metadata": storage_preflight_artifact_catalog_metadata(
            storage_plan_path=storage_plan_path,
            cleanup_plan_path=cleanup_plan_path,
            journal_path=None,
        ),
        "blockers": blockers,
        "passed": not blockers,
        **FALSE_AUTHORITY,
    }
    if blockers:
        raise SystemExit(json.dumps(_jsonable(payload), sort_keys=True))
    return payload


def _per_pair_l2(array_nchw: np.ndarray) -> np.ndarray:
    arr = np.asarray(array_nchw, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    axes = tuple(range(1, arr.ndim))
    return np.sqrt(np.sum(arr.astype(np.float64) * arr.astype(np.float64), axis=axes)).astype(np.float32)


def _gradient_quality_blockers(
    *,
    name: str,
    stats: dict[str, Any],
    max_abs_sanity_limit: float | None = None,
) -> list[str]:
    blockers: list[str] = []
    if int(stats["nonfinite_count"]) > 0:
        blockers.append(f"{name}_gradient_nonfinite:{stats['nonfinite_count']}")
    abs_max = stats.get("abs_max")
    if max_abs_sanity_limit is not None and abs_max is not None:
        abs_max_f = float(abs_max)
        limit_f = float(max_abs_sanity_limit)
        if abs_max_f > limit_f:
            blockers.append(
                f"{name}_gradient_abs_max_exceeds_sanity_limit:"
                f"{abs_max_f:.12g}>{limit_f:.12g}"
            )
    return blockers


def _save_shard_arrays(
    *,
    output_path: Path,
    pair_start: int,
    pair_end: int,
    pose_grad_nchw: np.ndarray,
    seg_grad_nchw: np.ndarray,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pair_start=np.asarray(pair_start, dtype=np.int64),
        pair_end=np.asarray(pair_end, dtype=np.int64),
        posenet_yuv6_pair_grad=pose_grad_nchw.astype(np.float32, copy=False),
        segnet_last_rgb_grad=seg_grad_nchw.astype(np.float32, copy=False),
    )
    return {
        "path": output_path.as_posix(),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256_file(output_path),
    }


def _make_loss_fn(
    *,
    adapter: Any,
    ref_pose_nhwc: Any,
    ref_seg_nhwc: Any,
    full_video_pair_count: int,
    full_video_d_pose: float,
    seg_ce_weight: float,
    pose_eps: float,
) -> Any:
    import mlx.core as mx

    ref_pose = adapter.posenet(ref_pose_nhwc)["pose"][..., :6]
    ref_one_hot = None
    if seg_ce_weight > 0.0:
        ref_seg_logits = adapter.segnet(ref_seg_nhwc)
        ref_classes = mx.argmax(ref_seg_logits, axis=-1)
        class_axis = int(ref_seg_logits.shape[-1])
        class_ids = mx.reshape(mx.arange(class_axis), (1, 1, 1, class_axis))
        ref_one_hot = (mx.expand_dims(ref_classes, axis=-1) == class_ids).astype(mx.float32)
    pose_gain = 5.0 / math.sqrt(10.0 * max(float(full_video_d_pose), float(pose_eps)))
    total_pairs = float(full_video_pair_count)

    def _loss(candidate_pose_nhwc: Any, candidate_seg_nhwc: Any) -> Any:
        cand_pose = adapter.posenet(candidate_pose_nhwc)["pose"][..., :6]
        pose_diff = cand_pose - ref_pose
        per_pair_pose_mse = mx.mean(mx.square(pose_diff), axis=tuple(range(1, len(pose_diff.shape))))
        pose_loss = pose_gain * mx.sum(per_pair_pose_mse) / total_pairs

        if seg_ce_weight > 0.0:
            cand_logits = adapter.segnet(candidate_seg_nhwc)
            logsum = mx.logsumexp(cand_logits, axis=-1)
            target_logits = mx.sum(cand_logits * ref_one_hot, axis=-1)
            per_pair_ce = mx.mean(logsum - target_logits, axis=tuple(range(1, len(logsum.shape))))
            seg_loss = float(seg_ce_weight) * mx.sum(per_pair_ce) / total_pairs
        else:
            seg_loss = mx.array(0.0, dtype=pose_loss.dtype)
        return pose_loss + seg_loss

    return _loss


def _compute_shard_mlx(
    *,
    adapter: Any,
    reference: Any,
    candidate: Any,
    pair_start: int,
    pair_end: int,
    full_video_pair_count: int,
    full_video_d_pose: float,
    seg_ce_weight: float,
    pose_eps: float,
    output_dir: Path,
    shard_index: int,
    summary_only: bool,
    max_gradient_abs_sanity_limit: float | None,
    mlx_device_type: str,
) -> dict[str, Any]:
    import mlx.core as mx

    ref_pose_np = np.asarray(reference.posenet_yuv6_pair[pair_start:pair_end], dtype=np.float32)
    ref_seg_np = np.asarray(reference.segnet_last_rgb[pair_start:pair_end], dtype=np.float32)
    cand_pose_np = np.asarray(candidate.posenet_yuv6_pair[pair_start:pair_end], dtype=np.float32)
    cand_seg_np = np.asarray(candidate.segnet_last_rgb[pair_start:pair_end], dtype=np.float32)
    ref_pose = mx.array(nchw_to_nhwc(ref_pose_np))
    ref_seg = mx.array(nchw_to_nhwc(ref_seg_np))
    cand_pose = mx.array(nchw_to_nhwc(cand_pose_np))
    cand_seg = mx.array(nchw_to_nhwc(cand_seg_np))
    loss_fn = _make_loss_fn(
        adapter=adapter,
        ref_pose_nhwc=ref_pose,
        ref_seg_nhwc=ref_seg,
        full_video_pair_count=full_video_pair_count,
        full_video_d_pose=full_video_d_pose,
        seg_ce_weight=seg_ce_weight,
        pose_eps=pose_eps,
    )
    started = time.time()
    loss_value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(cand_pose, cand_seg)
    mx.eval(loss_value, grads[0], grads[1])
    try:
        mx.synchronize()
    except AttributeError:
        pass
    pose_grad_nchw = nhwc_to_nchw(np.asarray(grads[0], dtype=np.float32))
    seg_grad_nchw = nhwc_to_nchw(np.asarray(grads[1], dtype=np.float32))
    elapsed = time.time() - started
    pose_pair_l2 = _per_pair_l2(pose_grad_nchw)
    seg_pair_l2 = _per_pair_l2(seg_grad_nchw)
    pose_stats = _array_stats(pose_grad_nchw)
    seg_stats = _array_stats(seg_grad_nchw)
    shard_blockers = [
        *_gradient_quality_blockers(
            name="posenet_yuv6_pair",
            stats=pose_stats,
            max_abs_sanity_limit=max_gradient_abs_sanity_limit,
        ),
        *_gradient_quality_blockers(
            name="segnet_last_rgb",
            stats=seg_stats,
            max_abs_sanity_limit=max_gradient_abs_sanity_limit,
        ),
    ]
    arrays = None
    if not summary_only:
        arrays = _save_shard_arrays(
            output_path=output_dir / f"shard_{shard_index:04d}_{pair_start:04d}_{pair_end:04d}.npz",
            pair_start=pair_start,
            pair_end=pair_end,
            pose_grad_nchw=pose_grad_nchw,
            seg_grad_nchw=seg_grad_nchw,
        )
    return {
        "schema": SHARD_SCHEMA,
        "backend": "mlx",
        "device_type": mlx_device_type,
        "shard_index": int(shard_index),
        "pair_start": int(pair_start),
        "pair_end": int(pair_end),
        "pair_count": int(pair_end - pair_start),
        "elapsed_seconds": elapsed,
        "loss_contribution": float(np.asarray(loss_value)),
        "gradient_quality_blockers": shard_blockers,
        "arrays": arrays,
        "posenet_yuv6_pair_grad": {
            "shape": list(pose_grad_nchw.shape),
            "stats": pose_stats,
            "per_pair_l2": pose_pair_l2.tolist(),
        },
        "segnet_last_rgb_grad": {
            "shape": list(seg_grad_nchw.shape),
            "stats": seg_stats,
            "per_pair_l2": seg_pair_l2.tolist(),
        },
        "top_pairs_by_grad_l2": [
            {
                "pair_idx": int(pair_start + idx),
                "combined_grad_l2": float(pose_pair_l2[idx] + seg_pair_l2[idx]),
                "pose_grad_l2": float(pose_pair_l2[idx]),
                "seg_grad_l2": float(seg_pair_l2[idx]),
            }
            for idx in np.argsort(-(pose_pair_l2 + seg_pair_l2), kind="stable")[: min(16, len(pose_pair_l2))]
        ],
    }


def _compute_shard_torch(
    *,
    dist: Any,
    reference: Any,
    candidate: Any,
    pair_start: int,
    pair_end: int,
    full_video_pair_count: int,
    full_video_d_pose: float,
    seg_ce_weight: float,
    pose_eps: float,
    output_dir: Path,
    shard_index: int,
    summary_only: bool,
    torch_device: str,
    max_gradient_abs_sanity_limit: float | None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch.device(torch_device)
    ref_pose = torch.from_numpy(np.array(reference.posenet_yuv6_pair[pair_start:pair_end], dtype=np.float32, copy=True)).to(device)
    ref_seg = torch.from_numpy(np.array(reference.segnet_last_rgb[pair_start:pair_end], dtype=np.float32, copy=True)).to(device)
    cand_pose = torch.from_numpy(np.array(candidate.posenet_yuv6_pair[pair_start:pair_end], dtype=np.float32, copy=True)).to(device)
    cand_seg = torch.from_numpy(np.array(candidate.segnet_last_rgb[pair_start:pair_end], dtype=np.float32, copy=True)).to(device)
    cand_pose = cand_pose.detach().clone().requires_grad_(True)
    cand_seg = cand_seg.detach().clone().requires_grad_(seg_ce_weight > 0.0)

    with torch.no_grad():
        ref_pose_out = dist.posenet(ref_pose)["pose"][..., :6]
        ref_classes = dist.segnet(ref_seg).argmax(dim=1) if seg_ce_weight > 0.0 else None
    started = time.time()
    cand_pose_out = dist.posenet(cand_pose)["pose"][..., :6]
    pose_diff = cand_pose_out - ref_pose_out
    per_pair_pose_mse = pose_diff.square().mean(dim=tuple(range(1, pose_diff.ndim)))
    pose_gain = 5.0 / math.sqrt(10.0 * max(float(full_video_d_pose), float(pose_eps)))
    pose_loss = pose_gain * per_pair_pose_mse.sum() / float(full_video_pair_count)
    if seg_ce_weight > 0.0:
        cand_logits = dist.segnet(cand_seg)
        seg_map = F.cross_entropy(cand_logits, ref_classes, reduction="none")
        per_pair_ce = seg_map.mean(dim=tuple(range(1, seg_map.ndim)))
        seg_loss = float(seg_ce_weight) * per_pair_ce.sum() / float(full_video_pair_count)
    else:
        seg_loss = torch.zeros((), dtype=pose_loss.dtype, device=device)
    loss_value = pose_loss + seg_loss
    loss_value.backward()

    pose_grad_nchw = cand_pose.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    if cand_seg.grad is None:
        seg_grad_nchw = np.zeros_like(np.asarray(candidate.segnet_last_rgb[pair_start:pair_end], dtype=np.float32))
    else:
        seg_grad_nchw = cand_seg.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    elapsed = time.time() - started
    pose_pair_l2 = _per_pair_l2(pose_grad_nchw)
    seg_pair_l2 = _per_pair_l2(seg_grad_nchw)
    pose_stats = _array_stats(pose_grad_nchw)
    seg_stats = _array_stats(seg_grad_nchw)
    shard_blockers = [
        *_gradient_quality_blockers(
            name="posenet_yuv6_pair",
            stats=pose_stats,
            max_abs_sanity_limit=max_gradient_abs_sanity_limit,
        ),
        *_gradient_quality_blockers(
            name="segnet_last_rgb",
            stats=seg_stats,
            max_abs_sanity_limit=max_gradient_abs_sanity_limit,
        ),
    ]
    arrays = None
    if not summary_only:
        arrays = _save_shard_arrays(
            output_path=output_dir / f"shard_{shard_index:04d}_{pair_start:04d}_{pair_end:04d}.npz",
            pair_start=pair_start,
            pair_end=pair_end,
            pose_grad_nchw=pose_grad_nchw,
            seg_grad_nchw=seg_grad_nchw,
        )
    return {
        "schema": SHARD_SCHEMA,
        "backend": "torch",
        "torch_device": torch_device,
        "shard_index": int(shard_index),
        "pair_start": int(pair_start),
        "pair_end": int(pair_end),
        "pair_count": int(pair_end - pair_start),
        "elapsed_seconds": elapsed,
        "loss_contribution": float(loss_value.detach().cpu().item()),
        "gradient_quality_blockers": shard_blockers,
        "arrays": arrays,
        "posenet_yuv6_pair_grad": {
            "shape": list(pose_grad_nchw.shape),
            "stats": pose_stats,
            "per_pair_l2": pose_pair_l2.tolist(),
        },
        "segnet_last_rgb_grad": {
            "shape": list(seg_grad_nchw.shape),
            "stats": seg_stats,
            "per_pair_l2": seg_pair_l2.tolist(),
        },
        "top_pairs_by_grad_l2": [
            {
                "pair_idx": int(pair_start + idx),
                "combined_grad_l2": float(pose_pair_l2[idx] + seg_pair_l2[idx]),
                "pose_grad_l2": float(pose_pair_l2[idx]),
                "seg_grad_l2": float(seg_pair_l2[idx]),
            }
            for idx in np.argsort(-(pose_pair_l2 + seg_pair_l2), kind="stable")[: min(16, len(pose_pair_l2))]
        ],
    }


def _prepare_torch_distortion_net(dist: Any, torch_device: str) -> tuple[Any, str]:
    import torch

    device = torch.device(torch_device)
    dist = dist.to(device).eval()
    for param in dist.parameters():
        param.requires_grad_(False)
    return dist, str(device)


def run_vjp(args: argparse.Namespace) -> dict[str, Any]:
    candidate = load_scorer_input_cache(
        args.candidate_cache_dir,
        mmap_mode="r",
        integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
    )
    reference = load_scorer_input_cache(
        args.reference_cache_dir,
        mmap_mode="r",
        integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
    )
    if candidate.pair_indices.shape != reference.pair_indices.shape or not np.array_equal(
        np.asarray(candidate.pair_indices),
        np.asarray(reference.pair_indices),
    ):
        raise SystemExit("candidate/reference pair_indices must match exactly for this VJP runner")
    total_pairs = int(candidate.pair_indices.shape[0])
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    pair_limit = min(total_pairs, int(args.max_pairs)) if args.max_pairs is not None else total_pairs
    if args.start_pair < 0 or args.start_pair >= total_pairs:
        raise SystemExit("--start-pair outside cache pair range")
    stop_pair = min(total_pairs, args.start_pair + pair_limit)
    if stop_pair <= args.start_pair:
        raise SystemExit("empty VJP pair window")
    xray = _load_xray(args.pair_window_json)
    full_video_d_pose = _full_d_pose_from_inputs(xray, args.full_video_d_pose)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    storage_preflight = _storage_preflight(
        output_dir=output_dir,
        requested_bytes=_estimate_vjp_output_bytes(
            candidate=candidate,
            start_pair=args.start_pair,
            stop_pair=stop_pair,
            summary_only=bool(args.summary_only),
        ),
        reserve_free_bytes=int(max(float(args.storage_reserve_gb), 0.0) * (1024**3)),
        allow_local_output=bool(args.allow_local_output),
        storage_plan_path=args.storage_plan_path,
        cleanup_plan_path=args.cleanup_plan_path,
    )
    upstream_dir = _resolve_upstream_dir(REPO_ROOT, upstream_dir=args.upstream_dir)
    dist = _load_upstream_distortion_net(upstream_dir)
    torch_device = args.torch_device
    if args.backend == "torch":
        dist, torch_device = _prepare_torch_distortion_net(dist, args.torch_device)
    shards: list[dict[str, Any]] = []
    started = time.time()
    for shard_index, pair_start in enumerate(range(args.start_pair, stop_pair, args.shard_pairs)):
        pair_end = min(stop_pair, pair_start + args.shard_pairs)
        if args.backend == "mlx":

            def compute_mlx_on(device_type: str, *, summary_only: bool) -> dict[str, Any]:
                with temporary_mlx_device(device_type):
                    adapter = torch_distortion_net_to_mlx(dist)
                    return _compute_shard_mlx(
                        adapter=adapter,
                        reference=reference,
                        candidate=candidate,
                        pair_start=pair_start,
                        pair_end=pair_end,
                        full_video_pair_count=total_pairs,
                        full_video_d_pose=full_video_d_pose,
                        seg_ce_weight=args.seg_ce_weight,
                        pose_eps=args.pose_eps,
                        output_dir=output_dir / "shards",
                        shard_index=shard_index,
                        summary_only=summary_only,
                        max_gradient_abs_sanity_limit=args.max_gradient_abs_sanity_limit,
                        mlx_device_type=device_type,
                    )

            if args.device_type == "auto":
                metal_probe = compute_mlx_on("gpu", summary_only=True)
                if metal_probe.get("gradient_quality_blockers"):
                    shard = compute_mlx_on("cpu", summary_only=args.summary_only)
                    shard["gradient_backend_fallback"] = {
                        "schema": "direct_full_scorer_vjp_gradient_backend_fallback.v1",
                        "from_backend": "mlx",
                        "from_device_type": "gpu",
                        "to_backend": "mlx",
                        "to_device_type": "cpu",
                        "reason": "metal_gradient_quality_blocked",
                        "metal_gradient_quality_blockers": list(
                            metal_probe.get("gradient_quality_blockers") or []
                        ),
                        "metal_probe_loss_contribution": metal_probe.get("loss_contribution"),
                    }
                else:
                    shard = (
                        compute_mlx_on("gpu", summary_only=False)
                        if not args.summary_only
                        else metal_probe
                    )
                    shard["gradient_backend_fallback"] = None
            else:
                shard = compute_mlx_on(args.device_type, summary_only=args.summary_only)
        else:
            shard = _compute_shard_torch(
                dist=dist,
                reference=reference,
                candidate=candidate,
                pair_start=pair_start,
                pair_end=pair_end,
                full_video_pair_count=total_pairs,
                full_video_d_pose=full_video_d_pose,
                seg_ce_weight=args.seg_ce_weight,
                pose_eps=args.pose_eps,
                output_dir=output_dir / "shards",
                shard_index=shard_index,
                summary_only=args.summary_only,
                torch_device=torch_device,
                max_gradient_abs_sanity_limit=args.max_gradient_abs_sanity_limit,
            )
        shards.append(shard)
        if args.progress_every and (shard_index + 1) % args.progress_every == 0:
            done = pair_end - args.start_pair
            print(
                json.dumps(
                    {
                        "event": "direct_full_scorer_vjp_progress",
                        "done_pairs": done,
                        "total_pairs": stop_pair - args.start_pair,
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    full_reduction_complete = args.start_pair == 0 and stop_pair == total_pairs
    blockers = ["direct_full_scorer_vjp_is_acquisition_only_false_authority"]
    if not full_reduction_complete:
        blockers.append("direct_full_scorer_vjp_partial_pair_window_not_exact_full_video_reduction")
    if args.summary_only:
        blockers.append("direct_full_scorer_vjp_summary_only_no_gradient_arrays")
    shard_quality_blockers = sorted(
        {
            blocker
            for shard in shards
            for blocker in shard.get("gradient_quality_blockers", [])
        }
    )
    if shard_quality_blockers:
        blockers.append("direct_full_scorer_vjp_gradient_quality_failed")
    bundle = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "producer": TOOL,
        "backend": args.backend,
        "device_type": args.device_type,
        "torch_device": torch_device if args.backend == "torch" else None,
        "candidate_cache": _cache_paths(candidate),
        "reference_cache": _cache_paths(reference),
        "pair_window": [int(args.start_pair), int(stop_pair)],
        "full_video_pair_count": total_pairs,
        "full_video_d_pose": full_video_d_pose,
        "seg_ce_weight": float(args.seg_ce_weight),
        "pose_gain": 5.0 / math.sqrt(10.0 * max(float(full_video_d_pose), float(args.pose_eps))),
        "max_gradient_abs_sanity_limit": args.max_gradient_abs_sanity_limit,
        "seg_surrogate": "cross_entropy_to_reference_segnet_argmax",
        "pose_objective": "full_video_scaled_sqrt_pose_mse_linearized_gain",
        "exact_reduction_contract": {
            "single_update_after_all_shards_reduce": True,
            "budget_spend_allowed_before_full_reduction": False,
            "full_reduction_complete": full_reduction_complete,
        },
        "archive": _archive_record(args.archive),
        "source_xray_report": _source_report_record(args.pair_window_json, xray),
        "storage_preflight": storage_preflight,
        "shard_count": len(shards),
        "shards": shards,
        "gradient_quality_blockers": shard_quality_blockers,
        "elapsed_seconds": time.time() - started,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    bundle_path = output_dir / "direct_full_scorer_vjp_bundle.json"
    bundle["bundle_path"] = bundle_path.as_posix()
    _write_json(bundle_path, bundle)
    return bundle


def _source_report_record(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
    }


def _archive_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=False)
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size if resolved.exists() else None,
        "sha256": _sha256_file(resolved) if resolved.is_file() else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair-window-json", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--backend", default="mlx", choices=("mlx", "torch"))
    parser.add_argument("--device-type", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--shard-pairs", type=int, default=4)
    parser.add_argument("--seg-ce-weight", type=float, default=100.0)
    parser.add_argument("--full-video-d-pose", type=float)
    parser.add_argument("--pose-eps", type=float, default=1.0e-12)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--max-gradient-abs-sanity-limit",
        type=float,
        default=1.0e6,
        help=(
            "Fail closed when a finite gradient abs max exceeds this numerical "
            "sanity bound. Use a negative value only for explicit drift probes."
        ),
    )
    parser.add_argument(
        "--storage-reserve-gb",
        type=float,
        default=5.0,
        help="Free GiB to preserve on the selected output tier after estimated VJP shards.",
    )
    parser.add_argument(
        "--allow-local-output",
        action="store_true",
        help="Explicitly allow VJP shard output outside the operator SSD waterfall.",
    )
    parser.add_argument("--storage-plan-path", type=Path)
    parser.add_argument("--cleanup-plan-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.shard_pairs < 1:
        raise SystemExit("--shard-pairs must be >= 1")
    if args.seg_ce_weight < 0.0:
        raise SystemExit("--seg-ce-weight must be >= 0")
    if args.pose_eps <= 0.0:
        raise SystemExit("--pose-eps must be > 0")
    if args.max_gradient_abs_sanity_limit is not None and args.max_gradient_abs_sanity_limit < 0.0:
        args.max_gradient_abs_sanity_limit = None
    bundle = run_vjp(args)
    print(
        json.dumps(
            {
                "bundle": bundle["bundle_path"],
                "shard_count": bundle["shard_count"],
                "full_reduction_complete": bundle["exact_reduction_contract"]["full_reduction_complete"],
                "blockers": bundle["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
