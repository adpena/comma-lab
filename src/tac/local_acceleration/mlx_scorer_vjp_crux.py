# SPDX-License-Identifier: MIT
"""Localize MLX Metal scorer-VJP drift against MLX CPU and PyTorch.

This module is diagnostic only. It consumes fixed scorer-input NumPy caches and
compares first-order acquisition gradients for the same pair window across
MLX CPU, MLX GPU, and PyTorch. The intent is to identify the smallest live
scorer branch whose Metal reverse-mode path is unsafe before any allocator or
trainer spends budget on that gradient.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    nchw_to_nhwc,
    nhwc_to_nchw,
    temporary_mlx_device,
    torch_distortion_net_to_mlx,
    torch_posenet_to_mlx,
)
from tac.local_acceleration.mlx_scorer_response import (
    MANIFEST_CACHE_INTEGRITY_MODE,
    _load_upstream_distortion_net,
    _resolve_upstream_dir,
    load_scorer_input_cache,
)

SCHEMA_VERSION = "mlx_scorer_vjp_crux.v1"
POSENET_YUV6_VJP_PARITY_SCHEMA_VERSION = "mlx_posenet_yuv6_vjp_parity.v1"
BRANCHES = ("pose", "seg", "joint")

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


def build_mlx_scorer_vjp_crux_manifest(
    *,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    repo_root: str | Path,
    start_pair: int,
    max_pairs: int,
    full_video_d_pose: float,
    seg_ce_weight: float = 100.0,
    pose_eps: float = 1.0e-12,
    branches: tuple[str, ...] = BRANCHES,
    max_abs_ratio_warn: float = 1.0e3,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare scorer VJP gradients across MLX CPU, MLX GPU, and PyTorch."""

    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if float(seg_ce_weight) < 0.0:
        raise ValueError(f"seg_ce_weight must be >= 0, got {seg_ce_weight}")
    if float(pose_eps) <= 0.0:
        raise ValueError(f"pose_eps must be > 0, got {pose_eps}")
    if float(max_abs_ratio_warn) < 1.0:
        raise ValueError(f"max_abs_ratio_warn must be >= 1, got {max_abs_ratio_warn}")
    if not math.isfinite(float(full_video_d_pose)) or float(full_video_d_pose) < 0.0:
        raise ValueError(f"full_video_d_pose must be finite and >= 0, got {full_video_d_pose}")
    branch_tuple = tuple(str(branch) for branch in branches)
    unknown = sorted(set(branch_tuple) - set(BRANCHES))
    if unknown:
        raise ValueError(f"unsupported VJP crux branches: {unknown}")

    candidate = load_scorer_input_cache(
        candidate_cache_dir,
        mmap_mode="r",
        integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
    )
    reference = load_scorer_input_cache(
        reference_cache_dir,
        mmap_mode="r",
        integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
    )
    if candidate.pair_indices.shape != reference.pair_indices.shape or not np.array_equal(
        np.asarray(candidate.pair_indices),
        np.asarray(reference.pair_indices),
    ):
        raise ValueError("candidate/reference pair_indices must match exactly")

    total_pairs = int(candidate.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pairs:
        raise ValueError(f"start_pair {start} outside cache pair count {total_pairs}")
    stop = min(total_pairs, start + int(max_pairs))
    repo = Path(repo_root).expanduser().resolve(strict=False)
    upstream_dir = _resolve_upstream_dir(repo)
    dist = _load_upstream_distortion_net(upstream_dir)
    torch_model = _prepare_torch(dist)
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    started = time.time()

    for branch in branch_tuple:
        branch_result = {
            "torch_cpu": _torch_branch_grad(
                dist=torch_model,
                candidate=candidate,
                reference=reference,
                start=start,
                stop=stop,
                total_pairs=total_pairs,
                full_video_d_pose=float(full_video_d_pose),
                seg_ce_weight=float(seg_ce_weight),
                pose_eps=float(pose_eps),
                branch=branch,
            ),
            "mlx_cpu": _mlx_branch_grad(
                dist=dist,
                candidate=candidate,
                reference=reference,
                start=start,
                stop=stop,
                total_pairs=total_pairs,
                full_video_d_pose=float(full_video_d_pose),
                seg_ce_weight=float(seg_ce_weight),
                pose_eps=float(pose_eps),
                branch=branch,
                device_type="cpu",
            ),
            "mlx_gpu": _mlx_branch_grad(
                dist=dist,
                candidate=candidate,
                reference=reference,
                start=start,
                stop=stop,
                total_pairs=total_pairs,
                full_video_d_pose=float(full_video_d_pose),
                seg_ce_weight=float(seg_ce_weight),
                pose_eps=float(pose_eps),
                branch=branch,
                device_type="gpu",
            ),
        }
        branch_rows = _comparison_rows(
            branch=branch,
            branch_result=branch_result,
            max_abs_ratio_warn=float(max_abs_ratio_warn),
        )
        results[branch] = branch_result
        rows.extend(branch_rows)

    blockers = sorted(
        {
            blocker
            for row in rows
            for blocker in row.get("blockers", [])
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "producer": "tac.local_acceleration.mlx_scorer_vjp_crux",
        "run_id": run_id,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "passed": not blockers,
        "verdict": "PASS_MLX_METAL_VJP_CRUX" if not blockers else "FAIL_MLX_METAL_VJP_CRUX",
        "blockers": blockers,
        "candidate_cache": _cache_record(candidate),
        "reference_cache": _cache_record(reference),
        "pair_window": [start, stop],
        "full_video_pair_count": total_pairs,
        "full_video_d_pose": float(full_video_d_pose),
        "seg_ce_weight": float(seg_ce_weight),
        "pose_gain": _pose_gain(float(full_video_d_pose), float(pose_eps)),
        "max_abs_ratio_warn": float(max_abs_ratio_warn),
        "branches": list(branch_tuple),
        "results": results,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "authority_status": (
            "MLX scorer VJP crux manifests are diagnostic local implementation "
            "evidence only; they identify usable acquisition backends but do not "
            "replace receiver proof or contest auth eval."
        ),
        **FALSE_AUTHORITY,
    }


def write_mlx_scorer_vjp_crux_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_posenet_yuv6_vjp_parity_record(
    *,
    torch_posenet: Any,
    reference_yuv6_pair_nchw: np.ndarray,
    candidate_yuv6_pair_nchw: np.ndarray,
    full_video_pair_count: int,
    full_video_d_pose: float,
    pose_eps: float = 1.0e-12,
    device_type: Literal["cpu", "gpu"] = "cpu",
    max_abs_pose_delta_warn: float = 2.0e-3,
    max_abs_grad_delta_warn: float = 2.0e-4,
    measured_passes: int = 1,
) -> dict[str, Any]:
    """Compare fixed-YUV6 PoseNet VJP parity for PyTorch CPU and MLX.

    This intentionally bypasses RGB/YUV preprocessing and does not write VJP
    shards. It proves the local PoseNet acquisition-gradient surface only.
    """

    reference = np.asarray(reference_yuv6_pair_nchw, dtype=np.float32)
    candidate = np.asarray(candidate_yuv6_pair_nchw, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference_yuv6_pair_nchw and candidate_yuv6_pair_nchw must have "
            f"matching shapes, got {reference.shape} and {candidate.shape}"
        )
    if reference.ndim != 4 or int(reference.shape[1]) != 12:
        raise ValueError(
            "PoseNet YUV6 tensors must be NCHW with 12 channels, "
            f"got {reference.shape}"
        )
    if int(full_video_pair_count) < int(reference.shape[0]):
        raise ValueError(
            "full_video_pair_count must cover the compared pair batch, "
            f"got {full_video_pair_count} for batch {reference.shape[0]}"
        )
    if float(pose_eps) <= 0.0:
        raise ValueError(f"pose_eps must be > 0, got {pose_eps}")
    if not math.isfinite(float(full_video_d_pose)) or float(full_video_d_pose) < 0.0:
        raise ValueError(
            f"full_video_d_pose must be finite and >= 0, got {full_video_d_pose}"
        )
    if int(measured_passes) < 1:
        raise ValueError(f"measured_passes must be >= 1, got {measured_passes}")

    torch_result = _torch_posenet_yuv6_vjp(
        torch_posenet=torch_posenet,
        reference_yuv6_pair_nchw=reference,
        candidate_yuv6_pair_nchw=candidate,
        full_video_pair_count=int(full_video_pair_count),
        full_video_d_pose=float(full_video_d_pose),
        pose_eps=float(pose_eps),
    )
    mlx_result = _mlx_posenet_yuv6_vjp(
        torch_posenet=torch_posenet,
        reference_yuv6_pair_nchw=reference,
        candidate_yuv6_pair_nchw=candidate,
        full_video_pair_count=int(full_video_pair_count),
        full_video_d_pose=float(full_video_d_pose),
        pose_eps=float(pose_eps),
        device_type=device_type,
        measured_passes=int(measured_passes),
    )
    pose_delta = np.asarray(mlx_result["candidate_pose"], dtype=np.float32) - np.asarray(
        torch_result["candidate_pose"],
        dtype=np.float32,
    )
    grad_delta = np.asarray(mlx_result["candidate_grad"], dtype=np.float32) - np.asarray(
        torch_result["candidate_grad"],
        dtype=np.float32,
    )
    pose_delta_stats = _stats(pose_delta)
    grad_delta_stats = _stats(grad_delta)
    loss_abs_delta = abs(float(mlx_result["loss"]) - float(torch_result["loss"]))
    blockers: list[str] = []
    if int(torch_result["candidate_grad_stats"]["nonfinite_count"]) > 0:
        blockers.append("torch_posenet_yuv6_grad_nonfinite")
    if int(mlx_result["candidate_grad_stats"]["nonfinite_count"]) > 0:
        blockers.append(f"mlx_{device_type}_posenet_yuv6_grad_nonfinite")
    if float(torch_result["candidate_grad_stats"]["abs_max"] or 0.0) == 0.0:
        blockers.append("torch_posenet_yuv6_grad_zero")
    if float(mlx_result["candidate_grad_stats"]["abs_max"] or 0.0) == 0.0:
        blockers.append(f"mlx_{device_type}_posenet_yuv6_grad_zero")
    pose_abs_max = float(pose_delta_stats["abs_max"] or 0.0)
    grad_abs_max = float(grad_delta_stats["abs_max"] or 0.0)
    measured_pose_spread_abs_max = float(
        mlx_result.get("measured_candidate_pose_spread_stats", {}).get("abs_max")
        or 0.0
    )
    measured_grad_spread_abs_max = float(
        mlx_result.get("measured_candidate_grad_spread_stats", {}).get("abs_max")
        or 0.0
    )
    if pose_abs_max > float(max_abs_pose_delta_warn):
        blockers.append(
            "posenet_yuv6_pose_forward_delta_exceeds:"
            f"{pose_abs_max:.12g}>{float(max_abs_pose_delta_warn):.12g}"
        )
    if grad_abs_max > float(max_abs_grad_delta_warn):
        blockers.append(
            "posenet_yuv6_vjp_grad_delta_exceeds:"
            f"{grad_abs_max:.12g}>{float(max_abs_grad_delta_warn):.12g}"
        )
    if measured_pose_spread_abs_max > float(max_abs_pose_delta_warn):
        blockers.append(
            f"mlx_{device_type}_posenet_yuv6_measured_pose_spread_exceeds:"
            f"{measured_pose_spread_abs_max:.12g}>{float(max_abs_pose_delta_warn):.12g}"
        )
    if measured_grad_spread_abs_max > float(max_abs_grad_delta_warn):
        blockers.append(
            f"mlx_{device_type}_posenet_yuv6_measured_grad_spread_exceeds:"
            f"{measured_grad_spread_abs_max:.12g}>{float(max_abs_grad_delta_warn):.12g}"
        )

    return {
        "schema_version": POSENET_YUV6_VJP_PARITY_SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "branch": "pose",
        "device_type": device_type,
        "passed": not blockers,
        "verdict": "PASS_POSENET_YUV6_VJP_PARITY" if not blockers else "FAIL_POSENET_YUV6_VJP_PARITY",
        "blockers": blockers,
        "full_video_pair_count": int(full_video_pair_count),
        "full_video_d_pose": float(full_video_d_pose),
        "pose_gain": _pose_gain(float(full_video_d_pose), float(pose_eps)),
        "max_abs_pose_delta_warn": float(max_abs_pose_delta_warn),
        "max_abs_grad_delta_warn": float(max_abs_grad_delta_warn),
        "measured_passes": int(measured_passes),
        "torch_cpu": _posenet_vjp_summary(torch_result),
        f"mlx_{device_type}": _posenet_vjp_summary(mlx_result),
        "comparison": {
            "loss_abs_delta": loss_abs_delta,
            "candidate_pose_delta": pose_delta_stats,
            "candidate_grad_delta": grad_delta_stats,
        },
        "authority_status": (
            "Fixed-cache PoseNet YUV6 VJP parity is diagnostic local "
            "implementation evidence only; it does not provide contest score, "
            "receiver proof, promotion, or exact-eval authority."
        ),
        **FALSE_AUTHORITY,
    }


def _prepare_torch(dist: Any) -> Any:
    dist = dist.to("cpu").eval()
    for param in dist.parameters():
        param.requires_grad_(False)
    return dist


def _torch_posenet_yuv6_vjp(
    *,
    torch_posenet: Any,
    reference_yuv6_pair_nchw: np.ndarray,
    candidate_yuv6_pair_nchw: np.ndarray,
    full_video_pair_count: int,
    full_video_d_pose: float,
    pose_eps: float,
) -> dict[str, Any]:
    import torch

    model = torch_posenet.to("cpu").eval()
    for param in model.parameters():
        param.requires_grad_(False)
    ref = torch.from_numpy(
        np.array(reference_yuv6_pair_nchw, dtype=np.float32, copy=True)
    )
    cand = torch.from_numpy(
        np.array(candidate_yuv6_pair_nchw, dtype=np.float32, copy=True)
    ).requires_grad_(True)
    with torch.no_grad():
        ref_pose = model(ref)["pose"][..., :6]
    cand_pose = model(cand)["pose"][..., :6]
    diff = cand_pose - ref_pose
    per_pair = diff.square().mean(dim=tuple(range(1, cand_pose.ndim)))
    loss = (
        _pose_gain(full_video_d_pose, pose_eps)
        * per_pair.sum()
        / float(full_video_pair_count)
    )
    loss.backward()
    grad = cand.grad.detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "loss": float(loss.detach().cpu().item()),
        "reference_pose": ref_pose.detach().cpu().numpy().astype(np.float32, copy=True),
        "candidate_pose": cand_pose.detach().cpu().numpy().astype(np.float32, copy=True),
        "candidate_grad": grad,
        "candidate_grad_stats": _stats(grad),
        "candidate_pose_stats": _stats(
            cand_pose.detach().cpu().numpy().astype(np.float32, copy=True)
        ),
    }


def _mlx_posenet_yuv6_vjp(
    *,
    torch_posenet: Any,
    reference_yuv6_pair_nchw: np.ndarray,
    candidate_yuv6_pair_nchw: np.ndarray,
    full_video_pair_count: int,
    full_video_d_pose: float,
    pose_eps: float,
    device_type: Literal["cpu", "gpu"],
    measured_passes: int,
) -> dict[str, Any]:
    import mlx.core as mx

    with temporary_mlx_device(device_type):
        adapter = torch_posenet_to_mlx(torch_posenet)
        ref = mx.array(nchw_to_nhwc(reference_yuv6_pair_nchw))
        cand = mx.array(nchw_to_nhwc(candidate_yuv6_pair_nchw))
        ref_pose_lazy = adapter(ref)["pose"][..., :6]
        mx.eval(ref_pose_lazy)
        try:
            mx.synchronize()
        except AttributeError:
            pass
        # Keep the VJP reference pose as a host-materialized constant. Reusing
        # the lazy reference-forward graph inside value_and_grad can make
        # repeated MLX CPU PoseNet VJP parity probes report inflated losses.
        ref_pose_np = np.asarray(ref_pose_lazy, dtype=np.float32)
        ref_pose = mx.stop_gradient(mx.array(ref_pose_np))
        gain = _pose_gain(full_video_d_pose, pose_eps)

        def loss_fn(candidate_x: Any) -> Any:
            cand_pose = adapter(candidate_x)["pose"][..., :6]
            diff = cand_pose - ref_pose
            per_pair = mx.mean(mx.square(diff), axis=tuple(range(1, len(diff.shape))))
            return gain * mx.sum(per_pair) / float(full_video_pair_count)

        warmup_loss, warmup_grad = mx.value_and_grad(loss_fn)(cand)
        warmup_pose = adapter(cand)["pose"][..., :6]
        mx.eval(warmup_loss, warmup_grad, warmup_pose)
        try:
            mx.synchronize()
        except AttributeError:
            pass
        measured: list[dict[str, Any]] = []
        for _ in range(int(measured_passes)):
            loss, grad = mx.value_and_grad(loss_fn)(cand)
            cand_pose = adapter(cand)["pose"][..., :6]
            mx.eval(loss, grad, cand_pose)
            try:
                mx.synchronize()
            except AttributeError:
                pass
            measured.append(
                {
                    "loss": float(np.asarray(loss)),
                    "candidate_pose": np.asarray(cand_pose, dtype=np.float32),
                    "candidate_grad": nhwc_to_nchw(
                        np.asarray(grad, dtype=np.float32)
                    ),
                }
            )
        losses = np.asarray([row["loss"] for row in measured], dtype=np.float64)
        selected_index = int(np.argsort(losses)[len(losses) // 2])
        selected = measured[selected_index]
        grad_np = np.asarray(selected["candidate_grad"], dtype=np.float32)
        cand_pose_np = np.asarray(selected["candidate_pose"], dtype=np.float32)
        pose_stack = np.stack(
            [np.asarray(row["candidate_pose"], dtype=np.float32) for row in measured],
            axis=0,
        )
        grad_stack = np.stack(
            [np.asarray(row["candidate_grad"], dtype=np.float32) for row in measured],
            axis=0,
        )
        pose_spread = np.max(pose_stack, axis=0) - np.min(pose_stack, axis=0)
        grad_spread = np.max(grad_stack, axis=0) - np.min(grad_stack, axis=0)
        return {
            "loss": float(selected["loss"]),
            "warmup_loss": float(np.asarray(warmup_loss)),
            "warmup_grad_stats": _stats(
                nhwc_to_nchw(np.asarray(warmup_grad, dtype=np.float32))
            ),
            "warmup_candidate_pose_stats": _stats(
                np.asarray(warmup_pose, dtype=np.float32)
            ),
            "measured_passes": int(measured_passes),
            "measured_loss_min": float(np.min(losses)),
            "measured_loss_max": float(np.max(losses)),
            "measured_loss_spread": float(np.max(losses) - np.min(losses)),
            "measured_candidate_pose_spread_stats": _stats(pose_spread),
            "measured_candidate_grad_spread_stats": _stats(grad_spread),
            "reference_pose": ref_pose_np,
            "candidate_pose": cand_pose_np,
            "candidate_grad": grad_np,
            "candidate_grad_stats": _stats(grad_np),
            "candidate_pose_stats": _stats(cand_pose_np),
        }


def _posenet_vjp_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "loss": float(result["loss"]),
        "reference_pose_stats": _stats(np.asarray(result["reference_pose"])),
        "candidate_pose_stats": dict(result["candidate_pose_stats"]),
        "candidate_grad_stats": dict(result["candidate_grad_stats"]),
    }
    if "warmup_loss" in result:
        summary["warmup_loss"] = float(result["warmup_loss"])
    if "warmup_grad_stats" in result:
        summary["warmup_grad_stats"] = dict(result["warmup_grad_stats"])
    for key in (
        "measured_passes",
        "measured_loss_min",
        "measured_loss_max",
        "measured_loss_spread",
        "measured_candidate_pose_spread_stats",
        "measured_candidate_grad_spread_stats",
    ):
        if key in result:
            value = result[key]
            summary[key] = dict(value) if isinstance(value, dict) else value
    return summary


def _pose_gain(full_video_d_pose: float, pose_eps: float) -> float:
    return 5.0 / math.sqrt(10.0 * max(float(full_video_d_pose), float(pose_eps)))


def _torch_branch_grad(
    *,
    dist: Any,
    candidate: Any,
    reference: Any,
    start: int,
    stop: int,
    total_pairs: int,
    full_video_d_pose: float,
    seg_ce_weight: float,
    pose_eps: float,
    branch: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ref_pose = torch.from_numpy(np.array(reference.posenet_yuv6_pair[start:stop], dtype=np.float32, copy=True))
    ref_seg = torch.from_numpy(np.array(reference.segnet_last_rgb[start:stop], dtype=np.float32, copy=True))
    cand_pose = torch.from_numpy(np.array(candidate.posenet_yuv6_pair[start:stop], dtype=np.float32, copy=True)).requires_grad_(
        branch in {"pose", "joint"}
    )
    cand_seg = torch.from_numpy(np.array(candidate.segnet_last_rgb[start:stop], dtype=np.float32, copy=True)).requires_grad_(
        branch in {"seg", "joint"}
    )
    with torch.no_grad():
        ref_pose_out = dist.posenet(ref_pose)["pose"][..., :6]
        ref_classes = dist.segnet(ref_seg).argmax(dim=1)
    loss = torch.zeros((), dtype=torch.float32)
    if branch in {"pose", "joint"}:
        cand_pose_out = dist.posenet(cand_pose)["pose"][..., :6]
        per_pair_pose_mse = (cand_pose_out - ref_pose_out).square().mean(dim=tuple(range(1, cand_pose_out.ndim)))
        loss = loss + _pose_gain(full_video_d_pose, pose_eps) * per_pair_pose_mse.sum() / float(total_pairs)
    if branch in {"seg", "joint"}:
        cand_logits = dist.segnet(cand_seg)
        seg_map = F.cross_entropy(cand_logits, ref_classes, reduction="none")
        loss = loss + float(seg_ce_weight) * seg_map.mean(dim=tuple(range(1, seg_map.ndim))).sum() / float(total_pairs)
    loss.backward()
    pose_grad = (
        np.zeros_like(np.asarray(candidate.posenet_yuv6_pair[start:stop], dtype=np.float32))
        if cand_pose.grad is None
        else cand_pose.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    seg_grad = (
        np.zeros_like(np.asarray(candidate.segnet_last_rgb[start:stop], dtype=np.float32))
        if cand_seg.grad is None
        else cand_seg.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    return _gradient_record(loss=float(loss.detach().cpu().item()), pose_grad=pose_grad, seg_grad=seg_grad)


def _mlx_branch_grad(
    *,
    dist: Any,
    candidate: Any,
    reference: Any,
    start: int,
    stop: int,
    total_pairs: int,
    full_video_d_pose: float,
    seg_ce_weight: float,
    pose_eps: float,
    branch: str,
    device_type: Literal["cpu", "gpu"],
) -> dict[str, Any]:
    import mlx.core as mx

    ref_pose_np = np.asarray(reference.posenet_yuv6_pair[start:stop], dtype=np.float32)
    ref_seg_np = np.asarray(reference.segnet_last_rgb[start:stop], dtype=np.float32)
    cand_pose_np = np.asarray(candidate.posenet_yuv6_pair[start:stop], dtype=np.float32)
    cand_seg_np = np.asarray(candidate.segnet_last_rgb[start:stop], dtype=np.float32)
    with temporary_mlx_device(device_type):
        adapter = torch_distortion_net_to_mlx(dist)
        ref_pose = mx.array(nchw_to_nhwc(ref_pose_np))
        ref_seg = mx.array(nchw_to_nhwc(ref_seg_np))
        cand_pose = mx.array(nchw_to_nhwc(cand_pose_np))
        cand_seg = mx.array(nchw_to_nhwc(cand_seg_np))
        ref_pose_out = adapter.posenet(ref_pose)["pose"][..., :6]
        ref_seg_logits = adapter.segnet(ref_seg)
        ref_classes = mx.argmax(ref_seg_logits, axis=-1)
        class_ids = mx.reshape(mx.arange(int(ref_seg_logits.shape[-1])), (1, 1, 1, -1))
        ref_one_hot = (mx.expand_dims(ref_classes, axis=-1) == class_ids).astype(mx.float32)
        gain = _pose_gain(full_video_d_pose, pose_eps)

        def loss_fn(pose_x: Any, seg_x: Any) -> Any:
            loss = mx.array(0.0, dtype=pose_x.dtype)
            if branch in {"pose", "joint"}:
                cand_pose_out = adapter.posenet(pose_x)["pose"][..., :6]
                diff = cand_pose_out - ref_pose_out
                per_pair = mx.mean(mx.square(diff), axis=tuple(range(1, len(diff.shape))))
                loss = loss + gain * mx.sum(per_pair) / float(total_pairs)
            if branch in {"seg", "joint"}:
                cand_logits = adapter.segnet(seg_x)
                logsum = mx.logsumexp(cand_logits, axis=-1)
                target_logits = mx.sum(cand_logits * ref_one_hot, axis=-1)
                per_pair = mx.mean(logsum - target_logits, axis=tuple(range(1, len(logsum.shape))))
                loss = loss + float(seg_ce_weight) * mx.sum(per_pair) / float(total_pairs)
            return loss

        loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(cand_pose, cand_seg)
        mx.eval(loss, grads[0], grads[1])
        try:
            mx.synchronize()
        except AttributeError:
            pass
        pose_grad = nhwc_to_nchw(np.asarray(grads[0], dtype=np.float32))
        seg_grad = nhwc_to_nchw(np.asarray(grads[1], dtype=np.float32))
        return _gradient_record(loss=float(np.asarray(loss)), pose_grad=pose_grad, seg_grad=seg_grad)


def _gradient_record(*, loss: float, pose_grad: np.ndarray, seg_grad: np.ndarray) -> dict[str, Any]:
    return {
        "loss": float(loss),
        "posenet_yuv6_pair_grad": _stats(pose_grad),
        "segnet_last_rgb_grad": _stats(seg_grad),
    }


def _stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    if finite_count:
        values = arr[finite].astype(np.float64, copy=False)
        abs_values = np.abs(values)
        return {
            "shape": list(arr.shape),
            "finite_fraction": float(finite_count / arr.size),
            "nonfinite_count": int(arr.size - finite_count),
            "abs_max": float(np.max(abs_values)),
            "abs_mean": float(np.mean(abs_values)),
            "l2_norm": float(np.sqrt(np.sum(values * values, dtype=np.float64))),
        }
    return {
        "shape": list(arr.shape),
        "finite_fraction": 0.0,
        "nonfinite_count": int(arr.size),
        "abs_max": None,
        "abs_mean": None,
        "l2_norm": None,
    }


def _comparison_rows(
    *,
    branch: str,
    branch_result: dict[str, dict[str, Any]],
    max_abs_ratio_warn: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tensor_key in ("posenet_yuv6_pair_grad", "segnet_last_rgb_grad"):
        torch_stats = branch_result["torch_cpu"][tensor_key]
        cpu_stats = branch_result["mlx_cpu"][tensor_key]
        gpu_stats = branch_result["mlx_gpu"][tensor_key]
        blockers: list[str] = []
        for backend, stats in (("mlx_cpu", cpu_stats), ("mlx_gpu", gpu_stats)):
            if int(stats["nonfinite_count"]) > 0:
                blockers.append(f"{branch}_{backend}_{tensor_key}_nonfinite:{stats['nonfinite_count']}")
        cpu_ratio = _ratio(cpu_stats.get("abs_max"), torch_stats.get("abs_max"))
        gpu_ratio = _ratio(gpu_stats.get("abs_max"), torch_stats.get("abs_max"))
        if gpu_ratio is not None and gpu_ratio > max_abs_ratio_warn:
            blockers.append(
                f"{branch}_mlx_gpu_{tensor_key}_abs_max_ratio_exceeds:"
                f"{gpu_ratio:.12g}>{max_abs_ratio_warn:.12g}"
            )
        if cpu_ratio is not None and cpu_ratio > max_abs_ratio_warn:
            blockers.append(
                f"{branch}_mlx_cpu_{tensor_key}_abs_max_ratio_exceeds:"
                f"{cpu_ratio:.12g}>{max_abs_ratio_warn:.12g}"
            )
        rows.append(
            {
                "branch": branch,
                "tensor": tensor_key,
                "torch_abs_max": torch_stats.get("abs_max"),
                "mlx_cpu_abs_max": cpu_stats.get("abs_max"),
                "mlx_gpu_abs_max": gpu_stats.get("abs_max"),
                "mlx_cpu_to_torch_abs_max_ratio": cpu_ratio,
                "mlx_gpu_to_torch_abs_max_ratio": gpu_ratio,
                "torch_l2_norm": torch_stats.get("l2_norm"),
                "mlx_cpu_l2_norm": cpu_stats.get("l2_norm"),
                "mlx_gpu_l2_norm": gpu_stats.get("l2_norm"),
                "blockers": blockers,
            }
        )
    return rows


def _ratio(value: Any, reference: Any) -> float | None:
    if value is None or reference is None:
        return None
    value_f = float(value)
    reference_f = float(reference)
    if reference_f == 0.0:
        return None if value_f == 0.0 else math.inf
    return abs(value_f) / abs(reference_f)


def _cache_record(cache: Any) -> dict[str, Any]:
    return {
        "root": cache.root.as_posix(),
        "pair_count": int(cache.pair_indices.shape[0]),
        "array_sha256": dict(cache.manifest.get("array_sha256") or {}),
    }


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


__all__ = [
    "BRANCHES",
    "FALSE_AUTHORITY",
    "POSENET_YUV6_VJP_PARITY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_mlx_scorer_vjp_crux_manifest",
    "build_posenet_yuv6_vjp_parity_record",
    "write_mlx_scorer_vjp_crux_manifest",
]
