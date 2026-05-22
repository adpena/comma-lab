# SPDX-License-Identifier: MIT
"""PyTorch-vs-MLX scorer parity audit on fixed scorer-input cache windows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    run_mlx_distortion_scorer_nchw,
    scorer_distortion_components_numpy,
    temporary_mlx_device,
    torch_distortion_net_to_mlx,
)
from tac.local_acceleration.mlx_scorer_response import (
    GPU_RESEARCH_SIGNAL_BLOCKER,
    _load_upstream_distortion_net,
    load_scorer_input_cache,
)

SCHEMA_VERSION = "mlx_scorer_torch_parity.v1"
PASS_VERDICT = "PASS_MLX_TORCH_SCORER_PARITY"
FAIL_VERDICT = "FAIL_MLX_TORCH_SCORER_PARITY"


@dataclass(frozen=True)
class MLXTorchParityThresholds:
    """Tolerances for MLX scorer outputs against upstream PyTorch outputs."""

    max_posenet_output_abs_delta: float = 2.0e-3
    max_segnet_logit_abs_delta: float = 1.0e-2
    max_posenet_component_abs_delta: float = 2.0e-5
    max_segnet_argmax_diff_pixels: int = 0


def build_mlx_scorer_torch_parity_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    thresholds: MLXTorchParityThresholds | None = None,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    """Compare MLX scorer outputs against upstream PyTorch on one cache window."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU parity audits are local "
            "research-signal implementation checks only; pass "
            "allow_gpu_research_signal=True after recording that rationale"
        )
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")

    cache = load_scorer_input_cache(cache_dir)
    total_pair_count = int(cache.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pair_count:
        raise ValueError(f"start_pair {start} is outside cache pair count {total_pair_count}")
    stop = min(total_pair_count, start + int(max_pairs))
    pose = np.asarray(cache.posenet_yuv6_pair[start:stop], dtype=np.float32)
    seg = np.asarray(cache.segnet_last_rgb[start:stop], dtype=np.float32)

    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    torch_outputs = run_torch_distortion_scorer_nchw(dist, pose, seg)
    with temporary_mlx_device(device_type):
        mlx_outputs = run_mlx_distortion_scorer_nchw(
            torch_distortion_net_to_mlx(dist),
            pose,
            seg,
        )

    return build_mlx_scorer_torch_parity_manifest_from_outputs(
        torch_outputs=torch_outputs,
        mlx_outputs=mlx_outputs,
        thresholds=thresholds,
        run_id=run_id,
        device_type=device_type,
        cache_dir=str(cache_dir),
        start_pair=start,
        max_pairs=int(max_pairs),
        pair_window=[start, stop],
        n_samples=stop - start,
        total_pair_count=total_pair_count,
        gpu_research_signal_allowed=bool(allow_gpu_research_signal),
    )


def run_torch_distortion_scorer_nchw(
    distortion_net: Any,
    posenet_yuv6_pair_nchw: np.ndarray,
    segnet_last_rgb_nchw: np.ndarray,
) -> dict[str, Any]:
    """Run upstream PyTorch PoseNet+SegNet on fixed scorer-input tensors."""

    import torch

    with torch.inference_mode():
        pose_tensor = torch.from_numpy(
            np.array(posenet_yuv6_pair_nchw, dtype=np.float32, copy=True, order="C")
        )
        seg_tensor = torch.from_numpy(
            np.array(segnet_last_rgb_nchw, dtype=np.float32, copy=True, order="C")
        )
        pose_outputs = distortion_net.posenet(pose_tensor)
        seg_outputs = distortion_net.segnet(seg_tensor)
    return {
        "posenet": {
            name: value.detach().cpu().numpy()
            for name, value in pose_outputs.items()
        },
        "segnet": seg_outputs.detach().cpu().numpy(),
    }


def build_mlx_scorer_torch_parity_manifest_from_outputs(
    *,
    torch_outputs: dict[str, Any],
    mlx_outputs: dict[str, Any],
    thresholds: MLXTorchParityThresholds | None = None,
    run_id: str | None = None,
    device_type: str | None = None,
    cache_dir: str | None = None,
    start_pair: int | None = None,
    max_pairs: int | None = None,
    pair_window: list[int] | None = None,
    n_samples: int | None = None,
    total_pair_count: int | None = None,
    gpu_research_signal_allowed: bool = False,
) -> dict[str, Any]:
    """Build a parity manifest from already-computed PyTorch and MLX outputs."""

    limits = thresholds or MLXTorchParityThresholds()
    blockers: list[str] = []

    pose_deltas = _pose_output_deltas(torch_outputs, mlx_outputs)
    max_pose_delta = max(pose_deltas.values(), default=0.0)
    seg_logit_delta, seg_argmax_diff_pixels = _segnet_deltas(
        torch_outputs,
        mlx_outputs,
        blockers,
    )
    components = scorer_distortion_components_numpy(torch_outputs, mlx_outputs)
    pose_component_abs_max = float(np.max(np.abs(components["posenet"])))
    seg_component_diff_samples = int(np.sum(components["segnet"] > 0.0))

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
    if pose_component_abs_max > limits.max_posenet_component_abs_delta:
        blockers.append(
            "posenet_component_abs_delta_exceeds_threshold:"
            f"{pose_component_abs_max:.12g}>{limits.max_posenet_component_abs_delta:.12g}"
        )
    if (
        seg_argmax_diff_pixels is not None
        and seg_argmax_diff_pixels > limits.max_segnet_argmax_diff_pixels
    ):
        blockers.append(
            "segnet_argmax_diff_pixels_exceeds_threshold:"
            f"{seg_argmax_diff_pixels}>{limits.max_segnet_argmax_diff_pixels}"
        )

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
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "device_type": device_type,
        "gpu_research_signal_allowed": bool(gpu_research_signal_allowed),
        "cache_dir": cache_dir,
        "start_pair": start_pair,
        "max_pairs": max_pairs,
        "pair_window": pair_window,
        "n_samples": n_samples,
        "total_pair_count": total_pair_count,
        "deltas": {
            "posenet_outputs_max_abs": pose_deltas,
            "posenet_output_abs_max": max_pose_delta,
            "segnet_logit_abs_max": seg_logit_delta,
            "segnet_argmax_diff_pixels": seg_argmax_diff_pixels,
            "posenet_component_abs_max": pose_component_abs_max,
            "segnet_component_diff_samples": seg_component_diff_samples,
        },
        "allowed_use": (
            [
                "local_mlx_scorer_training_signal",
                "local_mlx_scorer_response_profiling",
                "candidate_generation_prior",
            ]
            if passed
            else [
                "do_not_use_mlx_scorer_signal_for_training_priority",
                "fall_back_to_upstream_pytorch_or_exact_eval_axis",
            ]
        ),
        "authority_status": (
            "PyTorch-vs-MLX parity is local implementation evidence only; paired "
            "CPU/CUDA auth eval remains required for score claims and promotion."
        ),
        "device_contract": {
            "gpu_research_signal_blocker": GPU_RESEARCH_SIGNAL_BLOCKER,
            "gpu_research_signal_required": device_type == "gpu",
            "forbidden_uses": [
                "score_claim",
                "promotion",
                "rank_or_kill",
                "replacement_for_cpu_or_cuda_auth_eval",
            ],
        },
    }


def write_mlx_scorer_torch_parity_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a parity manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pose_output_deltas(torch_outputs: dict[str, Any], mlx_outputs: dict[str, Any]) -> dict[str, float]:
    torch_pose = torch_outputs.get("posenet")
    mlx_pose = mlx_outputs.get("posenet")
    if not isinstance(torch_pose, dict) or not isinstance(mlx_pose, dict):
        raise ValueError("outputs must contain posenet dictionaries")
    keys = sorted(set(torch_pose) | set(mlx_pose))
    deltas: dict[str, float] = {}
    for key in keys:
        if key not in torch_pose or key not in mlx_pose:
            deltas[key] = float("inf")
            continue
        lhs = np.asarray(torch_pose.get(key), dtype=np.float32)
        rhs = np.asarray(mlx_pose.get(key), dtype=np.float32)
        deltas[key] = float("inf") if lhs.shape != rhs.shape else float(np.max(np.abs(lhs - rhs)))
    return deltas


def _segnet_deltas(
    torch_outputs: dict[str, Any],
    mlx_outputs: dict[str, Any],
    blockers: list[str],
) -> tuple[float | None, int | None]:
    torch_seg = np.asarray(torch_outputs.get("segnet"), dtype=np.float32)
    mlx_seg = np.asarray(mlx_outputs.get("segnet"), dtype=np.float32)
    if torch_seg.shape != mlx_seg.shape:
        blockers.append(
            "segnet_shape_mismatch:"
            f"torch={list(torch_seg.shape)}:mlx={list(mlx_seg.shape)}"
        )
        return None, None
    logit_delta = float(np.max(np.abs(torch_seg - mlx_seg)))
    argmax_diff = int(np.sum(np.argmax(torch_seg, axis=1) != np.argmax(mlx_seg, axis=1)))
    return logit_delta, argmax_diff


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "FAIL_VERDICT",
    "PASS_VERDICT",
    "SCHEMA_VERSION",
    "MLXTorchParityThresholds",
    "build_mlx_scorer_torch_parity_manifest",
    "build_mlx_scorer_torch_parity_manifest_from_outputs",
    "run_torch_distortion_scorer_nchw",
    "write_mlx_scorer_torch_parity_manifest",
]
