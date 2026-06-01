# SPDX-License-Identifier: MIT
"""Joint P18/P19 recon-pixel-weight surface producer.

This module turns the contest scorer geometry into the canonical
``RendererBundle.recon_pixel_weight`` artifact consumed by MLX substrate
training. It is deliberately substrate-agnostic: the output is a file-backed
``(N, 2, H, W, 1)`` weight map plus a manifest, so HiNeRV/SNeRV/RNeRV-style
carriers can consume the same P18/P19 surface without duplicating scorer math.

Authority boundary: this is a local MLX acquisition signal, not an exact score.
Only byte-closed local CPU replay and exact contest auth eval can promote.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.repo_io import write_json

JOINT_RECON_PIXEL_WEIGHT_SCHEMA = "joint_p18_p19_recon_pixel_weight.v1"
JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA = (
    "joint_p18_p19_recon_pixel_weight_manifest.v1"
)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


@dataclass(frozen=True)
class JointReconPixelWeightConfig:
    """Config for full-video chunked P18/P19 recon-pixel weighting."""

    num_pairs: int
    pair_chunk_size: int = 8
    scorer_hw: tuple[int, int] = (384, 512)
    d_pose_operating_point: float = 3.4e-5
    seg_weight: float = 100.0
    pose_axis_count: int = 6
    pose_inverse_variance: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    seg_margin_delta: float = 1.0
    weight_floor_fraction: float = 0.05
    normalize: str = "mean"

    def __post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if self.scorer_hw[0] <= 1 or self.scorer_hw[1] <= 1:
            raise ValueError("scorer_hw must be at least 2x2")
        if self.d_pose_operating_point < 0.0:
            raise ValueError("d_pose_operating_point must be >= 0")
        if self.seg_weight < 0.0:
            raise ValueError("seg_weight must be >= 0")
        if self.pose_axis_count <= 0:
            raise ValueError("pose_axis_count must be positive")
        if len(self.pose_inverse_variance) < self.pose_axis_count:
            raise ValueError("pose_inverse_variance must cover pose_axis_count")
        if any(float(v) <= 0.0 for v in self.pose_inverse_variance):
            raise ValueError("pose_inverse_variance entries must be > 0")
        if self.seg_margin_delta < 0.0:
            raise ValueError("seg_margin_delta must be >= 0")
        if self.weight_floor_fraction < 0.0:
            raise ValueError("weight_floor_fraction must be >= 0")
        if self.normalize not in {"mean", "none"}:
            raise ValueError("normalize must be 'mean' or 'none'")

    @property
    def pose_gain(self) -> float:
        if self.d_pose_operating_point <= 0.0:
            return 0.0
        return 5.0 / float(np.sqrt(10.0 * self.d_pose_operating_point))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        mean_value = float(np.mean(finite))
        std_value = float(np.std(finite))
    else:
        min_value = 0.0
        max_value = 0.0
        mean_value = 0.0
        std_value = 0.0
    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(np.asarray(array).dtype),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "std": std_value,
        "nonfinite_count": int(arr.size - finite.size),
        "nonzero_fraction": float(np.count_nonzero(arr) / max(int(arr.size), 1)),
    }


def _as_target_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (N,H,W,3); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr)


def _sanitize_gradient_component(
    arr: np.ndarray,
    *,
    component: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(value)
    nonfinite_count = int(value.size - int(np.count_nonzero(finite_mask)))
    return (
        np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
        {
            "component": component,
            "nonfinite_count": nonfinite_count,
            "sanitized_to_zero": nonfinite_count > 0,
        },
    )


def _pairs_to_scorer_inputs_nhwc(
    pairs_rgb_255: Any,
    *,
    scorer_hw: tuple[int, int],
) -> tuple[Any, Any]:
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
    scorer_flat = resize_nhwc_align_corners_false(
        flat,
        size=scorer_hw,
        mode="bilinear",
    )
    scorer_pairs = mx.reshape(
        scorer_flat,
        (shape[0], 2, int(scorer_hw[0]), int(scorer_hw[1]), 3),
    )
    segnet_last_rgb = scorer_pairs[:, 1, :, :, :]
    yuv6 = rgb_to_yuv6_mlx(scorer_pairs)
    h2, w2 = int(yuv6.shape[2]), int(yuv6.shape[3])
    pose_input = mx.reshape(
        mx.transpose(yuv6, (0, 2, 3, 1, 4)),
        (shape[0], h2, w2, 12),
    )
    return segnet_last_rgb, pose_input


def _mlx_to_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    try:
        mx.synchronize()
    except AttributeError:
        pass
    return np.asarray(value)


def build_joint_p18_p19_recon_pixel_weight(
    target_rgb_0: Any,
    target_rgb_1: Any,
    *,
    mlx_scorer: Any,
    config: JointReconPixelWeightConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a chunk-reduced joint P18/P19 per-pair/frame weight map.

    ``target_rgb_0`` and ``target_rgb_1`` are NHWC ``[0,1]`` target frames. The
    output is ``float32`` shape ``(N,2,H,W,1)`` and can be passed directly to
    the compact renderer runner's ``--recon-pixel-weight-path``.
    """

    import mlx.core as mx

    t0 = _as_target_array(target_rgb_0, name="target_rgb_0")
    t1 = _as_target_array(target_rgb_1, name="target_rgb_1")
    if t0.shape != t1.shape:
        raise ValueError(f"target shapes must match; got {t0.shape} vs {t1.shape}")
    if int(t0.shape[0]) != int(config.num_pairs):
        raise ValueError(
            f"config.num_pairs={config.num_pairs} does not match targets {t0.shape[0]}"
        )

    n, h, w, _c = t0.shape
    joint = np.zeros((n, 2, h, w), dtype=np.float64)
    seg = np.zeros_like(joint)
    pose = np.zeros_like(joint)
    sanitization: list[dict[str, Any]] = []
    pose_inverse_variance = [
        float(v) for v in config.pose_inverse_variance[: config.pose_axis_count]
    ]

    for start in range(0, n, int(config.pair_chunk_size)):
        end = min(start + int(config.pair_chunk_size), n)
        pairs_np = np.stack([t0[start:end], t1[start:end]], axis=1) * 255.0
        pairs = mx.array(np.ascontiguousarray(pairs_np, dtype=np.float32))
        def seg_margin_loss(pair_rgb_255: Any) -> Any:
            current_seg, _current_pose = _pairs_to_scorer_inputs_nhwc(
                pair_rgb_255,
                scorer_hw=config.scorer_hw,
            )
            logits = mlx_scorer.segnet(current_seg)
            sorted_logits = mx.sort(logits, axis=-1)
            top1 = sorted_logits[..., -1]
            top2 = sorted_logits[..., -2]
            margin = top1 - top2
            hinge = mx.maximum(
                0.0,
                float(config.seg_margin_delta) - margin,
            )
            return mx.mean(hinge)

        def pose_axis_mean(pair_rgb_255: Any, *, axis: int) -> Any:
            _current_seg, current_pose = _pairs_to_scorer_inputs_nhwc(
                pair_rgb_255,
                scorer_hw=config.scorer_hw,
            )
            pose_out = mlx_scorer.posenet(current_pose)["pose"]
            return mx.mean(pose_out[..., int(axis)])

        seg_grad = mx.grad(seg_margin_loss)(pairs)
        pose_axis_grads = [
            mx.grad(
                lambda pair_rgb_255, axis=axis: pose_axis_mean(
                    pair_rgb_255,
                    axis=axis,
                )
            )(pairs)
            for axis in range(int(config.pose_axis_count))
        ]
        mx.eval(seg_grad, *pose_axis_grads)

        seg_g_raw, seg_sanitized = _sanitize_gradient_component(
            np.abs(_mlx_to_numpy(seg_grad)),
            component=f"seg_margin_grad_pairs_{start}_{end}",
        )
        sanitization.append(seg_sanitized)
        seg_g = seg_g_raw.astype(np.float64)
        seg_saliency = np.sum(seg_g, axis=-1)
        pose_sq = np.zeros_like(seg_saliency)
        for axis, (inv_var, grad) in enumerate(zip(
            pose_inverse_variance,
            pose_axis_grads,
            strict=False,
        )):
            g_raw, pose_sanitized = _sanitize_gradient_component(
                _mlx_to_numpy(grad),
                component=f"pose_axis_{axis}_grad_pairs_{start}_{end}",
            )
            sanitization.append(pose_sanitized)
            g = g_raw.astype(np.float64)
            pose_sq += float(inv_var) * np.sum(np.square(g), axis=-1)
        pose_saliency = np.sqrt(np.maximum(pose_sq, 0.0))

        seg[start:end] = seg_saliency
        pose[start:end] = pose_saliency
        joint[start:end] = (
            float(config.seg_weight) * seg_saliency
            + float(config.pose_gain) * pose_saliency
        )

    nonfinite_records = [
        item for item in sanitization if int(item["nonfinite_count"]) > 0
    ]
    blockers = [
        f"nonfinite_gradient_sanitized:{item['component']}"
        for item in nonfinite_records
    ]
    joint_positive = bool(np.any(joint > 0.0))
    if not joint_positive:
        blockers.append("joint_recon_pixel_weight_surface_degenerate_uniform")
    positive_mean = float(np.mean(joint[joint > 0.0])) if joint_positive else 1.0
    if float(config.weight_floor_fraction) > 0.0:
        joint = joint + float(config.weight_floor_fraction) * positive_mean
    if config.normalize == "mean":
        mean = float(np.mean(joint))
        if mean <= 0.0 or not np.isfinite(mean):
            joint = np.ones_like(joint, dtype=np.float64)
            mean = 1.0
        joint = joint / mean
    weight = np.ascontiguousarray(joint[..., None].astype(np.float32))
    metadata = {
        "schema": JOINT_RECON_PIXEL_WEIGHT_SCHEMA,
        "local_axis": EVIDENCE_TAG_MLX,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "num_pairs": int(n),
        "height": int(h),
        "width": int(w),
        "channels": 1,
        "shape": [int(v) for v in weight.shape],
        "scorer_terms": {
            "p18_segnet": "mlx_segnet_top2_margin_vjp_on_last_frame",
            "p19_posenet": "mlx_posenet_per_axis_jacobian_norm_on_pair",
        },
        "pose_gain": float(config.pose_gain),
        "seg_weight": float(config.seg_weight),
        "d_pose_operating_point": float(config.d_pose_operating_point),
        "normalize": config.normalize,
        "weight_floor_fraction": float(config.weight_floor_fraction),
        "weight_stats": _stats(weight),
        "seg_saliency_stats": _stats(seg),
        "pose_saliency_stats": _stats(pose),
        "gradient_sanitization": sanitization,
        "blockers": blockers,
        "training_consumption_recommended": not blockers,
        **FALSE_AUTHORITY,
    }
    return weight, metadata


def build_joint_p18_p19_recon_pixel_weight_from_video(
    *,
    source_video_path: str | Path,
    upstream_dir: str | Path,
    config: JointReconPixelWeightConfig,
    scorer_device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the real video, load the MLX scorer adapter, and build weights."""

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )
    from tac.substrates._shared.mlx_score_aware import decode_mlx_targets

    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=int(config.num_pairs),
        output_height=int(config.scorer_hw[0]),
        output_width=int(config.scorer_hw[1]),
    )
    mlx_scorer = load_mlx_distortion_scorer_adapter_from_upstream(
        str(upstream_dir),
        device=scorer_device,
    )
    return build_joint_p18_p19_recon_pixel_weight(
        target_rgb_0,
        target_rgb_1,
        mlx_scorer=mlx_scorer,
        config=config,
    )


def write_joint_p18_p19_recon_pixel_weight_artifact(
    *,
    output_dir: str | Path,
    source_video_path: str | Path,
    upstream_dir: str | Path,
    config: JointReconPixelWeightConfig,
    scorer_device: str = "cpu",
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Build and write a queue-consumable joint recon-pixel-weight artifact."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"output_dir is non-empty; pass overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    weight, metadata = build_joint_p18_p19_recon_pixel_weight_from_video(
        source_video_path=source_video_path,
        upstream_dir=upstream_dir,
        config=config,
        scorer_device=scorer_device,
    )
    weight_path = out / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(weight_path, weight=weight)
    manifest_path = out / "joint_p18_p19_recon_pixel_weight_manifest.json"
    manifest: dict[str, Any] = {
        "schema": JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA,
        "manifest_path": manifest_path.as_posix(),
        "weight_path": weight_path.as_posix(),
        "weight_sha256": _sha256_file(weight_path),
        "weight_array_sha256": _array_sha256(weight),
        "weight_bytes": int(weight_path.stat().st_size),
        "source_video_path": Path(source_video_path).expanduser().as_posix(),
        "upstream_dir": Path(upstream_dir).expanduser().as_posix(),
        "scorer_device": scorer_device,
        "config": asdict(config),
        "elapsed_seconds": time.time() - started,
        "metadata": metadata,
        **FALSE_AUTHORITY,
    }
    write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "FALSE_AUTHORITY",
    "JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA",
    "JOINT_RECON_PIXEL_WEIGHT_SCHEMA",
    "JointReconPixelWeightConfig",
    "build_joint_p18_p19_recon_pixel_weight",
    "build_joint_p18_p19_recon_pixel_weight_from_video",
    "write_joint_p18_p19_recon_pixel_weight_artifact",
]
