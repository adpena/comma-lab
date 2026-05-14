#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reverse-engineer the fixed contest video into overfit-planning metadata.

This script does not run the scorer and makes no score claim. It records the
known upstream contest contract, source-video anatomy, camera/FoE geometry, and
cheap pairwise motion/foveation statistics that can drive later exact-eval
archive hypotheses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
UPSTREAM_ROOT = REPO_ROOT / "upstream"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from tac.camera import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    COMMA_EXTRINSICS,
    COMMA_INTRINSICS_NATIVE,
    FRAME_H,
    FRAME_W,
    HORIZON_BAND,
    VANISHING_POINT,
)


SCHEMA = "contest_video_reverse_engineering_v1"
PRODUCER = "experiments/reverse_engineer_contest_video.py"
SCORE_CLAIM_WARNING = (
    "This artifact is source-video and rule anatomy only. It is not score "
    "evidence and cannot promote, rank, kill, or retire a lane."
)


class VideoReverseEngineeringError(ValueError):
    """Raised when the video analysis inputs are invalid."""


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _safe_float(value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise VideoReverseEngineeringError("non-finite statistic produced")
    return out


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": _safe_float(arr.mean()),
        "std": _safe_float(arr.std()),
        "min": _safe_float(arr.min()),
        "max": _safe_float(arr.max()),
    }


def _run_json(cmd: list[str]) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _ffprobe(path: Path) -> dict[str, Any] | None:
    return _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,pix_fmt,color_space,color_transfer,color_primaries",
            "-of",
            "json",
            str(path),
        ]
    )


def _upstream_git() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=UPSTREAM_ROOT, text=True).strip()
        except Exception:
            return None

    return {
        "remote": run(["git", "remote", "get-url", "origin"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "head_summary": run(["git", "log", "-1", "--oneline"]),
    }


def _native_foe() -> dict[str, float]:
    scale_x = CAMERA_W / FRAME_W
    scale_y = CAMERA_H / FRAME_H
    return {
        "scorer_x": float(VANISHING_POINT[0]),
        "scorer_y": float(VANISHING_POINT[1]),
        "native_x": float(VANISHING_POINT[0] * scale_x),
        "native_y": float(VANISHING_POINT[1] * scale_y),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
    }


def _ring_masks(height: int, width: int, *, center_x: float, center_y: float) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt((xx.astype(np.float32) - center_x) ** 2 + (yy.astype(np.float32) - center_y) ** 2)
    max_radius = float(radius.max())
    edges = [0.0, 80.0, 160.0, 260.0, 400.0, max_radius + 1.0]
    masks: dict[str, np.ndarray] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        masks[f"r_{int(lo)}_{int(hi) if hi <= max_radius else 'max'}"] = (radius >= lo) & (radius < hi)
    horizon_top = int(round(HORIZON_BAND[0] * height / FRAME_H))
    horizon_bottom = int(round(HORIZON_BAND[1] * height / FRAME_H))
    horizon = np.zeros((height, width), dtype=bool)
    horizon[max(0, horizon_top) : min(height, horizon_bottom), :] = True
    masks["horizon_band"] = horizon
    return masks


def _decode_luma_plane(frame: Any) -> np.ndarray:
    plane = frame.planes[0]
    arr = np.frombuffer(plane, dtype=np.uint8)
    return arr.reshape(frame.height, plane.line_size)[:, : frame.width].copy()


def analyze_video_luma(
    video_path: Path,
    *,
    max_frames: int | None = None,
    pair_limit: int | None = None,
) -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:
        raise VideoReverseEngineeringError("PyAV is required for video analysis") from exc

    foe = _native_foe()
    masks = _ring_masks(
        CAMERA_H,
        CAMERA_W,
        center_x=foe["native_x"],
        center_y=foe["native_y"],
    )
    luma_means: list[float] = []
    luma_stds: list[float] = []
    pair_abs_deltas: list[float] = []
    pair_ring_deltas: dict[str, list[float]] = {key: [] for key in masks}
    pair_radial_biases: list[float] = []
    top_pairs: list[dict[str, Any]] = []

    prev: np.ndarray | None = None
    frame_count = 0
    pair_index = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            y = _decode_luma_plane(frame).astype(np.float32)
            luma_means.append(_safe_float(y.mean()))
            luma_stds.append(_safe_float(y.std()))
            if prev is not None and frame_count % 2 == 1:
                abs_delta = np.abs(y - prev)
                mean_abs = _safe_float(abs_delta.mean())
                pair_abs_deltas.append(mean_abs)
                ring_record: dict[str, float] = {}
                for name, mask in masks.items():
                    value = _safe_float(abs_delta[mask].mean())
                    pair_ring_deltas[name].append(value)
                    ring_record[name] = value
                inner = ring_record.get("r_0_80", 0.0) + ring_record.get("r_80_160", 0.0)
                outer = ring_record.get("r_260_400", 0.0) + ring_record.get("r_400_max", 0.0)
                radial_bias = _safe_float(inner - outer)
                pair_radial_biases.append(radial_bias)
                top_pairs.append(
                    {
                        "pair_index": pair_index,
                        "mean_abs_luma_delta": mean_abs,
                        "radial_inner_minus_outer_delta": radial_bias,
                        "horizon_band_delta": ring_record.get("horizon_band"),
                    }
                )
                pair_index += 1
                if pair_limit is not None and pair_index >= pair_limit:
                    break
            prev = y
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

    top_pairs_sorted = sorted(
        top_pairs,
        key=lambda item: float(item["mean_abs_luma_delta"]),
        reverse=True,
    )[:20]
    return {
        "frame_count_decoded": frame_count,
        "pair_count_analyzed": len(pair_abs_deltas),
        "luma_mean": _summary(luma_means),
        "luma_std": _summary(luma_stds),
        "pair_mean_abs_luma_delta": _summary(pair_abs_deltas),
        "pair_radial_inner_minus_outer_delta": _summary(pair_radial_biases),
        "ring_pair_delta": {name: _summary(values) for name, values in pair_ring_deltas.items()},
        "top_motion_pairs_by_luma_delta": top_pairs_sorted,
    }


def analyze_masks(mask_path: Path | None) -> dict[str, Any] | None:
    if mask_path is None:
        return None
    if not mask_path.exists():
        raise VideoReverseEngineeringError(f"mask path does not exist: {mask_path}")
    import torch
    from tac.lane_mark_pose import compute_zero_cost_poses_from_masks
    from tac.lane_mark_speed import zoom_from_masks
    from tac.mask_codec import decode_masks_auto

    masks = decode_masks_auto(mask_path).long()
    if masks.ndim != 3:
        raise VideoReverseEngineeringError(f"decoded masks must be (N,H,W), got {tuple(masks.shape)}")
    class_counts = torch.bincount(masks.reshape(-1), minlength=5).to(torch.float64)
    class_fracs = class_counts / class_counts.sum().clamp_min(1)
    zoom = zoom_from_masks(masks)
    zero_cost = compute_zero_cost_poses_from_masks(masks)
    zoom_abs = zoom.abs()
    topk = min(20, int(zoom.numel()))
    top_values, top_indices = torch.topk(zoom_abs, k=topk)
    return {
        "path": str(mask_path),
        "sha256": _sha256_file(mask_path),
        "bytes": int(mask_path.stat().st_size),
        "shape": list(masks.shape),
        "class_pixel_counts": [int(v.item()) for v in class_counts],
        "class_pixel_fractions": [float(v.item()) for v in class_fracs],
        "lane_mark_fraction": float(class_fracs[1].item()) if len(class_fracs) > 1 else None,
        "log_zoom_summary": _summary([float(v) for v in zoom.tolist()]),
        "log_zoom_abs_summary": _summary([float(v) for v in zoom_abs.tolist()]),
        "zero_cost_pose_dim0_summary": _summary([float(v) for v in zero_cost[:, 0].tolist()]),
        "top_pairs_by_abs_log_zoom": [
            {
                "pair_index": int(idx.item()),
                "abs_log_zoom": float(value.item()),
                "log_zoom": float(zoom[int(idx.item())].item()),
            }
            for value, idx in zip(top_values, top_indices)
        ],
    }


def build_video_reverse_engineering_report(
    *,
    video_path: Path,
    output_json: Path,
    mask_path: Path | None = None,
    max_frames: int | None = None,
    pair_limit: int | None = None,
) -> dict[str, Any]:
    video_path = video_path.resolve()
    if not video_path.exists():
        raise VideoReverseEngineeringError(f"video path does not exist: {video_path}")
    if video_path.stat().st_size <= 0:
        raise VideoReverseEngineeringError(f"video path is empty: {video_path}")

    luma = analyze_video_luma(video_path, max_frames=max_frames, pair_limit=pair_limit)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": PRODUCER,
        "evidence_grade": "empirical_video_anatomy",
        "score_claim": False,
        "promotion_eligible": False,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "contest_contract": {
            "upstream_repo": "https://github.com/commaai/comma_video_compression_challenge",
            "upstream_git": _upstream_git(),
            "public_test_video_names": (UPSTREAM_ROOT / "public_test_video_names.txt").read_text().splitlines(),
            "score_formula": "100*segnet_dist + sqrt(10*posenet_dist) + 25*archive_bytes/original_bytes",
            "original_size_bytes": 37_545_489,
            "expected_raw_rgb_bytes_per_video": CAMERA_W * CAMERA_H * 1200 * 3,
            "samples": 600,
            "seq_len": 2,
            "official_inflate_budget_seconds": 1800,
            "official_gpu_class_for_gpu_inflate": "Tesla T4 equivalent",
        },
        "video": {
            "path": str(video_path),
            "sha256": _sha256_file(video_path),
            "bytes": int(video_path.stat().st_size),
            "ffprobe": _ffprobe(video_path),
        },
        "camera_model": {
            "source": "repo tac.camera plus upstream frame_utils constants",
            "native_width": CAMERA_W,
            "native_height": CAMERA_H,
            "scorer_width": FRAME_W,
            "scorer_height": FRAME_H,
            "native_intrinsics": {
                "fx": COMMA_INTRINSICS_NATIVE.fx,
                "fy": COMMA_INTRINSICS_NATIVE.fy,
                "cx": COMMA_INTRINSICS_NATIVE.cx,
                "cy": COMMA_INTRINSICS_NATIVE.cy,
            },
            "extrinsics_prior": {
                "height_m": COMMA_EXTRINSICS.height,
                "pitch_rad": COMMA_EXTRINSICS.pitch,
            },
            "vanishing_point": _native_foe(),
            "horizon_band_scorer_y": list(HORIZON_BAND),
        },
        "luma_motion_anatomy": luma,
        "mask_proxy_anatomy": analyze_masks(mask_path.resolve() if mask_path is not None else None),
        "overfit_hypotheses": [
            {
                "id": "hardware_calibrated_foveation_atoms",
                "description": (
                    "Use the known EON/openpilot-style pinhole geometry and fixed "
                    "FoE to parameterize Telescope/hyperbolic foveation atoms."
                ),
                "evidence_type": "hypothesis_from_video_anatomy",
                "required_next_evidence": "exact CUDA archive eval of concrete foveation archive",
            },
            {
                "id": "ego_motion_pair_atoms",
                "description": (
                    "Use high luma-delta, horizon-band, and lane-mark log-zoom "
                    "pairs as PoseNet-sensitive pair atoms."
                ),
                "evidence_type": "empirical_video_prior",
                "required_next_evidence": "component-manifold probe or exact archive eval",
            },
            {
                "id": "openpilot_compress_time_prior",
                "description": (
                    "Use openpilot/supercombo or classical calibrated flow only at "
                    "compress time to seed pose/foveation/latent atoms; ship only "
                    "charged compact outputs."
                ),
                "evidence_type": "contest-compliant hypothesis",
                "required_next_evidence": "deterministic archive closure and exact CUDA eval",
            },
        ],
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }
    _write_json(output_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=UPSTREAM_ROOT / "videos" / "0.mkv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pair-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_video_reverse_engineering_report(
        video_path=args.video,
        output_json=args.output_json,
        mask_path=args.mask_path,
        max_frames=args.max_frames,
        pair_limit=args.pair_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
