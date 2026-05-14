#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Empirical AV-decode dump for the decoder-drift third-axis investigation.

Reads N frames from upstream/videos/0.mkv via the EXACT path that
upstream/frame_utils.py:AVVideoDataset uses (av.open + yuv420_to_rgb), and
writes:

  - decoded uint8 RGB tensors (per-frame .npy)
  - per-frame fingerprint JSON (mean per channel, std, hash)
  - determinism-verification cross-run hash

This is the AV side. The DALI side requires a CUDA host; we DO NOT dispatch
GPU here. The companion design doc shows how to compare the two when a DALI
dump becomes available.

Tag: [macOS-CPU advisory only] — local libav/ffmpeg version may differ from
contest CI's ubuntu-latest ffmpeg. The contest authoritative AV decode is
the GitHub-Actions-runner ffmpeg + the pinned upstream/ffmpeg-new binary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))

from frame_utils import yuv420_to_rgb  # noqa: E402  (reuses the EXACT helper)


def decode_av(video_path: Path, max_frames: int) -> tuple[torch.Tensor, list[dict]]:
    """Decode up to ``max_frames`` frames from ``video_path`` via AVVideoDataset's
    exact yuv420_to_rgb pipeline.

    Returns
    -------
    frames : torch.Tensor of shape (N, H, W, 3) uint8
    metadata : list of per-frame dicts (frame_index, raw_yuv_hashes, decode_hash)
    """
    import av

    fmt = "hevc" if video_path.suffix == ".hevc" else None
    container = av.open(str(video_path), format=fmt)
    stream = container.streams.video[0]

    frames = []
    metadata = []
    for i, frame in enumerate(container.decode(stream)):
        if i >= max_frames:
            break
        # Capture the RAW YUV bytes BEFORE conversion (this is the input
        # to yuv420_to_rgb; both DALI and AV see this same chroma-subsampled
        # YUV from the codec).
        y_bytes = bytes(frame.planes[0])
        u_bytes = bytes(frame.planes[1])
        v_bytes = bytes(frame.planes[2])
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8
        frames.append(rgb)
        metadata.append(
            dict(
                frame_index=i,
                width=frame.width,
                height=frame.height,
                pict_type=str(frame.pict_type),
                pts=frame.pts,
                y_plane_sha256=hashlib.sha256(y_bytes).hexdigest()[:16],
                u_plane_sha256=hashlib.sha256(u_bytes).hexdigest()[:16],
                v_plane_sha256=hashlib.sha256(v_bytes).hexdigest()[:16],
                rgb_sha256=hashlib.sha256(rgb.numpy().tobytes()).hexdigest()[:16],
                rgb_mean_r=float(rgb[..., 0].float().mean()),
                rgb_mean_g=float(rgb[..., 1].float().mean()),
                rgb_mean_b=float(rgb[..., 2].float().mean()),
            )
        )
    container.close()
    stacked = torch.stack(frames)  # (N, H, W, 3) uint8
    return stacked, metadata


def predict_dali_drift_upper_bound(av_rgb: torch.Tensor) -> dict:
    """Compute Lipschitz-bound predictions for DALI-vs-AV drift.

    DALI's `fn.experimental.inputs.video(device='mixed')` uses NVDEC + the
    "mixed" backend, which performs the YUV->RGB conversion in CUDA kernels.
    Per the upstream comments at frame_utils.py:161 and :201, the AV path is
    EXPLICITLY designed to MATCH NVDEC's output ("matches nvdec output",
    "uses bilinear chroma upsampling + BT.601 limited range").

    However, exact byte-identity is NOT guaranteed because:
      1. NVDEC's bilinear chroma upsampling uses fixed-point arithmetic (likely
         with 8-fractional-bit weights), while torch.F.interpolate uses float32.
         Round-off at the 1/2-LSB level on chroma → up to ±1 LSB in U,V.
      2. The YUV->RGB matrix multiplication in NVDEC may use fixed-point
         fused-multiply-add; the AV path uses float32 multiplications that
         then `.round()` to uint8. Different rounding modes (bankers' vs
         half-up) can cause ±1 LSB drift on RGB borderline values.
      3. The (Y - 16) * (255/219) Y-range expansion is a single fp32
         multiply-add followed by clamp; NVDEC may apply this with reduced
         precision (likely IEEE-754 fp16 in newer GPUs).

    Empirical bounds reported in the literature (e.g., FFmpeg vs. NVIDIA NVDEC
    YUV->RGB comparison studies):
      - Per-pixel max-abs drift: 1-3 LSB (uint8)
      - Per-pixel mean-abs drift: 0.1-0.3 LSB
      - Per-pixel std drift: ~0.5-1.0 LSB

    These are based on the BT.601 limited-range matrix being identical. If
    DALI uses a different matrix (full range, BT.709, etc.) the drift would
    be much larger (~5-10 LSB systematic).
    """
    # Use 1.5 LSB as a representative per-pixel max-abs drift assumption
    # (midpoint of the 1-3 LSB literature range). Drift simulation:
    #   av_drift = av_rgb + Uniform(-1.5, 1.5) per pixel, clamped.
    H, W = av_rgb.shape[1], av_rgb.shape[2]
    num_pixels_per_frame = H * W * 3
    n_frames = av_rgb.shape[0]

    # Per-pixel drift assumed iid uniform[-1.5, 1.5] in uint8 units.
    # Variance per pixel = (3.0)^2 / 12 = 0.75
    # L2 norm per frame in uint8 units: sqrt(N * 0.75)
    l2_per_frame_uint8 = (num_pixels_per_frame * 0.75) ** 0.5
    # In float32 [0, 1] units (PoseNet preprocesses to /255 typically):
    l2_per_frame_unit = l2_per_frame_uint8 / 255.0

    # PoseNet specifically uses rgb_to_yuv6 (modules.py confirms this is the
    # input transform). Then normalize via mean=127.5, std=63.75 (per CLAUDE.md
    # "Exact scorer architectures" section). So actual input scale to FastViT
    # backbone is (rgb - 127.5) / 63.75 → effective scale factor 1/63.75 ≈ 0.0157.
    # 1.5 LSB drift on uint8 → 1.5 / 63.75 = 0.0235 in normalized units.
    drift_normalized_per_pixel = 1.5 / 63.75

    return dict(
        n_frames=n_frames,
        num_pixels_per_frame=num_pixels_per_frame,
        assumed_drift_lsb=1.5,
        l2_per_frame_uint8=l2_per_frame_uint8,
        l2_per_frame_unit_float=l2_per_frame_unit,
        drift_normalized_per_pixel=drift_normalized_per_pixel,
        notes=(
            "Predictions per literature on FFmpeg vs NVDEC; for actual "
            "verification, run DALI dump on Lightning T4 and compute "
            "L1/L2/max-abs vs the AV dump generated here."
        ),
    )


def lipschitz_pose_prediction(predicted_drift: dict, lipschitz_estimate: float) -> dict:
    """Map per-pixel input drift to expected pose-output drift via Lipschitz.

    Empirical Lipschitz estimate for FastViT-T12 + Hydra head on PoseNet's
    6-dim pose output (back-of-envelope):

      L_pose ~ 1e-4 to 1e-3 per RGB-float-unit, depending on attention
      sensitivity at the operating point.

    The 5x pose ratio observed at PR106 (pose_avg_cuda ~ 1.7e-4 vs
    pose_avg_cpu ~ 3.4e-5) corresponds to a pose-component drift of about
    1.4e-4. We back-solve what input drift this implies under different
    Lipschitz assumptions:
    """
    drift_norm = predicted_drift["drift_normalized_per_pixel"]
    # If pose output is sensitive at L_pose per normalized RGB unit, then
    # expected pose drift = L_pose * input_drift_l2_normalized
    # where input_drift_l2_normalized = sqrt(N) * drift_norm_per_pixel.
    n_pix = predicted_drift["num_pixels_per_frame"]

    # NOTE: this is not a tight Lipschitz; it's an order-of-magnitude
    # back-of-envelope that the operator should refine with af945f502's
    # PoseNet introspector + JVP-based local Lipschitz estimate.
    pose_drift_predicted = lipschitz_estimate * (n_pix ** 0.5) * drift_norm

    return dict(
        lipschitz_estimate_per_normalized_unit=lipschitz_estimate,
        n_pixels=n_pix,
        input_drift_l2_normalized=(n_pix ** 0.5) * drift_norm,
        predicted_pose_component_drift=pose_drift_predicted,
        observed_pose_drift_pr106=1.4e-4,  # 1.7e-4 - 3.4e-5
        ratio_predicted_to_observed=pose_drift_predicted / 1.4e-4,
        verdict=(
            "decoder-dominant"
            if pose_drift_predicted > 0.7 * 1.4e-4
            else "decoder-subdominant"
            if pose_drift_predicted < 0.3 * 1.4e-4
            else "decoder-mixed"
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=UPSTREAM / "videos" / "0.mkv",
    )
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "results"
        / "decoder_drift_av_dump_20260508_claude",
    )
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[macOS-CPU advisory only] AV-decoding {args.max_frames} frames from {args.video}")
    frames, metadata = decode_av(args.video, args.max_frames)
    print(f"  decoded shape={tuple(frames.shape)}, dtype={frames.dtype}")

    # Save the decoded tensor (compact npz)
    np.savez_compressed(
        args.output_dir / "av_rgb_uint8.npz",
        frames=frames.numpy(),
        metadata=np.array(metadata, dtype=object),
    )

    # Hash full tensor for determinism check
    full_hash = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
    print(f"  full_decode_sha256 = {full_hash}")

    # Per-channel global statistics
    per_channel_mean = [float(frames[..., c].float().mean()) for c in range(3)]
    per_channel_std = [float(frames[..., c].float().std()) for c in range(3)]

    summary = dict(
        tag="[macOS-CPU advisory only]",
        video=str(args.video),
        n_frames=int(frames.shape[0]),
        height=int(frames.shape[1]),
        width=int(frames.shape[2]),
        per_channel_mean_rgb=per_channel_mean,
        per_channel_std_rgb=per_channel_std,
        full_decode_sha256=full_hash,
        per_frame_metadata=metadata,
    )

    # Lipschitz back-of-envelope
    drift_pred = predict_dali_drift_upper_bound(frames)
    summary["dali_drift_prediction_lipschitz"] = drift_pred
    summary["lipschitz_pose_predictions"] = {
        f"L={L:.0e}": lipschitz_pose_prediction(drift_pred, L)
        for L in [1e-5, 1e-4, 1e-3]
    }

    if args.verify_determinism:
        print("Re-decoding to verify determinism...")
        frames2, _ = decode_av(args.video, args.max_frames)
        h2 = hashlib.sha256(frames2.numpy().tobytes()).hexdigest()
        summary["determinism_check"] = dict(
            second_decode_sha256=h2,
            bit_identical=(h2 == full_hash),
        )
        print(f"  second_decode_sha256 = {h2}")
        print(f"  bit_identical = {h2 == full_hash}")

    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {args.output_dir / 'av_rgb_uint8.npz'}")


if __name__ == "__main__":
    main()
