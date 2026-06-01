#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the SR-NeRV resolution-axis mirror against upstream scorer preprocess.

This is a $0, local, false-authority gate for the low-res-encode +
super-resolve enhancer.  It asks whether:

    camera RGB -> low internal resolution -> legal 1164x874 output
      -> scorer downsample

lands near the direct scorer downsample used by upstream SegNet/PoseNet.  It
does not load scorer weights and never claims score authority.  Passing this
gate only means the SR-resolution idea is plausible enough to train/export a
charged receiver candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

SR_NERV_RESOLUTION_AXIS_MIRROR_SCHEMA = "sr_nerv_resolution_axis_mirror.v1"

FALSE_AUTHORITY: dict[str, Any] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def run_mirror_check(
    *,
    video_path: Path,
    output_json: Path,
    num_pairs: int,
    internal_width: int,
    internal_height: int,
    upsample_mode: str,
    max_abs_pass: float,
    mean_abs_pass: float,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    upstream_dir = repo_root / "upstream"
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    from frame_utils import camera_size, rgb_to_yuv6, segnet_model_input_size, seq_len  # type: ignore

    if num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if internal_width <= 0 or internal_height <= 0:
        raise ValueError("internal dimensions must be positive")
    frames_hwc = _decode_real_pairs(
        video_path=video_path,
        num_pairs=num_pairs,
        repo_root=repo_root,
    )
    btchw = frames_hwc.permute(0, 1, 4, 2, 3).to(dtype=torch.float32)
    camera_w, camera_h = int(camera_size[0]), int(camera_size[1])
    scorer_w, scorer_h = (
        int(segnet_model_input_size[0]),
        int(segnet_model_input_size[1]),
    )
    direct = _scorer_preprocess_tensors(
        btchw,
        scorer_hw=(scorer_h, scorer_w),
        rgb_to_yuv6=rgb_to_yuv6,
        seq_len=int(seq_len),
    )
    restored = _roundtrip_lowres_to_camera(
        btchw,
        internal_hw=(int(internal_height), int(internal_width)),
        camera_hw=(camera_h, camera_w),
        upsample_mode=upsample_mode,
    )
    mirrored = _scorer_preprocess_tensors(
        restored,
        scorer_hw=(scorer_h, scorer_w),
        rgb_to_yuv6=rgb_to_yuv6,
        seq_len=int(seq_len),
    )
    seg_metrics = _diff_metrics(direct["segnet_rgb"], mirrored["segnet_rgb"])
    pose_metrics = _diff_metrics(direct["posenet_yuv6"], mirrored["posenet_yuv6"])
    pass_gate = (
        seg_metrics["max_abs"] <= float(max_abs_pass)
        and pose_metrics["max_abs"] <= float(max_abs_pass)
        and seg_metrics["mean_abs"] <= float(mean_abs_pass)
        and pose_metrics["mean_abs"] <= float(mean_abs_pass)
    )
    result = {
        "schema": SR_NERV_RESOLUTION_AXIS_MIRROR_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "repo_head": _git_head(repo_root),
        "video_path": video_path.as_posix(),
        "video_sha256": _sha256_file(video_path),
        "num_pairs": int(num_pairs),
        "source_shape": {
            "batch": int(btchw.shape[0]),
            "seq_len": int(btchw.shape[1]),
            "channels": int(btchw.shape[2]),
            "height": int(btchw.shape[3]),
            "width": int(btchw.shape[4]),
        },
        "scorer_contract": {
            "camera_size_wh": [camera_w, camera_h],
            "segnet_model_input_size_wh": [scorer_w, scorer_h],
            "segnet_frame_index": 1,
            "posenet_frames": [0, 1],
        },
        "sr_nerv_candidate": {
            "internal_size_wh": [int(internal_width), int(internal_height)],
            "upsample_mode": upsample_mode,
            "internal_pixels": int(internal_width) * int(internal_height),
            "camera_pixels": camera_w * camera_h,
            "pixel_ratio_vs_camera": (
                (int(internal_width) * int(internal_height)) / (camera_w * camera_h)
            ),
        },
        "metrics": {
            "segnet_preprocess_rgb": seg_metrics,
            "posenet_preprocess_yuv6": pose_metrics,
        },
        "mirror_gate": {
            "schema": "sr_nerv_resolution_axis_mirror_gate.v1",
            "pass": bool(pass_gate),
            "max_abs_pass": float(max_abs_pass),
            "mean_abs_pass": float(mean_abs_pass),
            "next_action_if_pass": (
                "train_charged_lowres_carrier_plus_sr_adapter_under_packet_spine"
            ),
            "next_action_if_fail": (
                "demote_or_tighten_internal_resolution_before_training"
            ),
        },
        "blockers": [
            "preprocess_mirror_is_not_scorer_or_exact_eval_authority",
            "receiver_proven_archive_required_before_candidate_promotion",
            "contest_cpu_cuda_exact_eval_required_before_score_claim",
        ],
        **FALSE_AUTHORITY,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _decode_real_pairs(
    *,
    video_path: Path,
    num_pairs: int,
    repo_root: Path,
) -> torch.Tensor:
    upstream_dir = repo_root / "upstream"
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    from frame_utils import AVVideoDataset, camera_size, seq_len  # type: ignore

    ds = AVVideoDataset(
        [video_path.name],
        data_dir=video_path.parent,
        batch_size=int(num_pairs),
        device=torch.device("cpu"),
    )
    ds.prepare_data()
    for _, _, batch in ds:
        if batch.shape[0] < num_pairs:
            raise RuntimeError(f"decoded {batch.shape[0]} pairs, need {num_pairs}")
        expected_tail = (int(seq_len), int(camera_size[1]), int(camera_size[0]), 3)
        if tuple(batch.shape[1:]) != expected_tail:
            raise RuntimeError(
                f"unexpected upstream batch shape {tuple(batch.shape)}; "
                f"expected (*, {expected_tail})"
            )
        return batch[:num_pairs].to(dtype=torch.uint8)
    raise RuntimeError(f"decoded no batches from {video_path}")


def _roundtrip_lowres_to_camera(
    btchw: torch.Tensor,
    *,
    internal_hw: tuple[int, int],
    camera_hw: tuple[int, int],
    upsample_mode: str,
) -> torch.Tensor:
    n, t, c, _h, _w = btchw.shape
    flat = btchw.reshape(n * t, c, btchw.shape[-2], btchw.shape[-1])
    low = F.interpolate(flat, size=internal_hw, mode="bilinear", align_corners=False)
    kwargs: dict[str, Any] = {}
    if upsample_mode in {"bilinear", "bicubic", "linear", "trilinear"}:
        kwargs["align_corners"] = False
    restored = F.interpolate(low, size=camera_hw, mode=upsample_mode, **kwargs)
    return restored.clamp(0.0, 255.0).reshape(n, t, c, camera_hw[0], camera_hw[1])


def _scorer_preprocess_tensors(
    btchw: torch.Tensor,
    *,
    scorer_hw: tuple[int, int],
    rgb_to_yuv6: Any,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    n, t, c, h, w = btchw.shape
    if t != seq_len:
        raise ValueError(f"expected seq_len={seq_len}; got {t}")
    flat = btchw.reshape(n * t, c, h, w)
    resized = F.interpolate(flat, size=scorer_hw, mode="bilinear")
    pose = rgb_to_yuv6(resized).reshape(n, t * 6, scorer_hw[0] // 2, scorer_hw[1] // 2)
    seg = F.interpolate(btchw[:, -1, ...], size=scorer_hw, mode="bilinear")
    return {"posenet_yuv6": pose, "segnet_rgb": seg}


def _diff_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.to(dtype=torch.float32) - b.to(dtype=torch.float32)).abs().flatten()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p95_abs": float(torch.quantile(diff, 0.95).item()),
        "p99_abs": float(torch.quantile(diff, 0.99).item()),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--internal-width", type=int, default=512)
    parser.add_argument("--internal-height", type=int, default=384)
    parser.add_argument(
        "--upsample-mode",
        choices=("nearest", "bilinear", "bicubic"),
        default="bilinear",
    )
    parser.add_argument("--max-abs-pass", type=float, default=2.0)
    parser.add_argument("--mean-abs-pass", type=float, default=0.25)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    video = args.video.expanduser()
    if not video.is_absolute():
        video = repo_root / video
    output = args.output_json.expanduser()
    if not output.is_absolute():
        output = repo_root / output
    result = run_mirror_check(
        video_path=video,
        output_json=output,
        num_pairs=int(args.num_pairs),
        internal_width=int(args.internal_width),
        internal_height=int(args.internal_height),
        upsample_mode=str(args.upsample_mode),
        max_abs_pass=float(args.max_abs_pass),
        mean_abs_pass=float(args.mean_abs_pass),
        repo_root=repo_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"check_sr_nerv_resolution_axis_mirror failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
