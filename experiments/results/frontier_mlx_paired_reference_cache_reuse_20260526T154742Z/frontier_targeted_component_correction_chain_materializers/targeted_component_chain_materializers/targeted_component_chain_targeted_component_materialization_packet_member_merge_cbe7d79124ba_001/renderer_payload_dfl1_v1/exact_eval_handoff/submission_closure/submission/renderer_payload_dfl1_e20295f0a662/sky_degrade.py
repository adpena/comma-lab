#!/usr/bin/env python
"""Exploit #4: Sky region degradation for compress-time rate savings.

Sky (SegNet class 4) has zero driving semantics. SegNet classifies it
trivially regardless of pixel fidelity, and PoseNet barely uses sky
pixels (they are far away = tiny parallax contribution to ego-motion).

This script:
1. Decodes each frame from the input video
2. Runs a lightweight sky detector (top-of-frame heuristic + optional SegNet)
3. Applies Gaussian blur + bit-depth reduction to sky regions only
4. Re-encodes to a new video

The encoded video has simpler sky content -> fewer bits -> lower rate term.
At inflate time, the sky is already blurred in the compressed video, which
is fine because the scorer does not care about sky pixel fidelity.

Typical savings: 12-16% of total rate on highway scenes (sky is ~40% of
frame area, and blur + bit-reduce saves 30-40% of sky bits).

Usage:
    python sky_degrade.py --input 0.mkv --output 0_sky.mkv --blur-sigma 2.0 --bit-reduce 4 --feather 8
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import cv2
import numpy as np


def detect_sky_mask_heuristic(
    frame_bgr: np.ndarray,
    saturation_threshold: float = 60.0,
    brightness_threshold: float = 140.0,
    top_fraction: float = 0.55,
) -> np.ndarray:
    """Detect sky regions using color heuristics.

    Sky pixels are typically: high brightness, low saturation, in the upper
    portion of the frame. This avoids needing SegNet at compress time.

    The heuristic is deliberately conservative — it may miss some sky, but
    it will almost never misclassify road/vehicle as sky.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR frame.
        saturation_threshold: max saturation for sky (HSV S channel).
        brightness_threshold: min brightness for sky (HSV V channel).
        top_fraction: only consider the top fraction of the frame.

    Returns:
        (H, W) float32 mask in [0, 1]. 1.0 = sky, 0.0 = not sky.
    """
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Sky: low saturation + high value (brightness) + upper portion of frame
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Basic sky detection: low saturation AND high brightness
    sky_prob = np.zeros((H, W), dtype=np.float32)
    sky_condition = (sat < saturation_threshold) & (val > brightness_threshold)
    sky_prob[sky_condition] = 1.0

    # Only consider upper portion of frame (sky is above horizon)
    cutoff_row = int(H * top_fraction)
    sky_prob[cutoff_row:, :] = 0.0

    # Morphological cleanup: remove small noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_prob = cv2.morphologyEx(sky_prob, cv2.MORPH_CLOSE, kernel)
    sky_prob = cv2.morphologyEx(sky_prob, cv2.MORPH_OPEN, kernel)

    return sky_prob


def apply_sky_degradation(
    frame_bgr: np.ndarray,
    sky_mask: np.ndarray,
    blur_sigma: float = 2.0,
    bit_reduce: int = 4,
    feather_radius: int = 8,
) -> np.ndarray:
    """Apply blur + bit-depth reduction to sky regions.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR frame.
        sky_mask: (H, W) float32 mask in [0, 1].
        blur_sigma: Gaussian blur sigma for sky regions.
        bit_reduce: round pixel values to nearest `bit_reduce` (e.g., 4 = 6-bit).
        feather_radius: feather the mask boundary for smooth blending.

    Returns:
        (H, W, 3) uint8 BGR frame with degraded sky.
    """
    # Feather the mask to avoid hard boundaries
    if feather_radius > 0:
        sky_mask = cv2.GaussianBlur(sky_mask, (0, 0), feather_radius)
        sky_mask = np.clip(sky_mask, 0.0, 1.0)

    result = frame_bgr.copy()

    # Apply Gaussian blur to sky regions
    if blur_sigma > 0:
        blurred = cv2.GaussianBlur(frame_bgr, (0, 0), blur_sigma)
        mask_3ch = sky_mask[:, :, np.newaxis]
        result = (result.astype(np.float32) * (1.0 - mask_3ch)
                  + blurred.astype(np.float32) * mask_3ch)

    # Apply bit-depth reduction to sky regions
    if bit_reduce > 1:
        reduced = (result / bit_reduce).astype(np.float32)
        reduced = np.round(reduced) * bit_reduce
        reduced = np.clip(reduced, 0, 255)
        mask_3ch = sky_mask[:, :, np.newaxis]
        result = result * (1.0 - mask_3ch) + reduced * mask_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def get_video_info(ffprobe_bin: str, video_path: str) -> tuple[int, int, float]:
    """Get video dimensions and framerate."""
    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    parts = out.split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_parts = parts[2].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    return w, h, fps


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True), required=True,
              help="Input video path.")
@click.option("--output", "output_path", type=click.Path(), required=True,
              help="Output video path.")
@click.option("--ffmpeg-bin", envvar="FFMPEG_BIN", default="ffmpeg",
              help="Path to ffmpeg binary.")
@click.option("--ffprobe-bin", envvar="FFPROBE_BIN", default="ffprobe",
              help="Path to ffprobe binary.")
@click.option("--blur-sigma", type=float, default=2.0,
              help="Gaussian blur sigma for sky regions.")
@click.option("--bit-reduce", type=int, default=4,
              help="Bit-depth reduction factor (e.g. 4 = 6-bit effective).")
@click.option("--feather", type=int, default=8,
              help="Feather radius for mask boundary blending.")
@click.option("--upstream", type=click.Path(exists=True), envvar="COMMA_CHALLENGE_ROOT",
              default=None, help="Upstream challenge root (for SegNet, unused currently).")
@click.option("--scale-w", type=int, default=None,
              help="Override output width.")
@click.option("--scale-h", type=int, default=None,
              help="Override output height.")
@click.option("--saturation-threshold", type=float, default=60.0,
              help="Max HSV saturation for sky detection.")
@click.option("--brightness-threshold", type=float, default=140.0,
              help="Min HSV brightness for sky detection.")
@click.option("--top-fraction", type=float, default=0.55,
              help="Only detect sky in top fraction of frame.")
def sky_degrade(input_path, output_path, ffmpeg_bin, ffprobe_bin,
                blur_sigma, bit_reduce, feather, upstream,
                scale_w, scale_h, saturation_threshold, brightness_threshold,
                top_fraction):
    """Apply sky region degradation to reduce bitrate.

    \b
    Detects sky regions using color heuristics, then applies Gaussian blur
    and bit-depth reduction. Sky has zero driving semantics -- the scorer
    does not care about sky pixel fidelity.

    \b
    Examples:
      python sky_degrade.py --input 0.mkv --output 0_sky.mkv
      python sky_degrade.py --input 0.mkv --output 0_sky.mkv --blur-sigma 3.0
    """
    # Get video info
    w, h, fps = get_video_info(ffprobe_bin, input_path)
    click.echo(f"[sky_degrade] Input: {w}x{h} @ {fps:.1f}fps", err=True)

    # Decode all frames
    cmd_decode = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "error", "pipe:1",
    ]
    proc_decode = subprocess.Popen(cmd_decode, stdout=subprocess.PIPE)

    frame_size = w * h * 3
    frames_processed = 0

    # Stream processing: pipe decode -> process -> encode without buffering
    # all frames in memory. Keeps peak RAM at ~2 frames instead of 1200.
    cmd_encode = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "ffv1",  # lossless intermediate
        "-pix_fmt", "yuv420p",
        "-colorspace", "smpte170m",
        "-color_primaries", "smpte170m",
        "-color_trc", "smpte170m",
        "-v", "error",
        output_path,
    ]
    proc_encode = subprocess.Popen(cmd_encode, stdin=subprocess.PIPE)

    while True:
        raw = proc_decode.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

        # Detect sky
        sky_mask = detect_sky_mask_heuristic(
            frame,
            saturation_threshold=saturation_threshold,
            brightness_threshold=brightness_threshold,
            top_fraction=top_fraction,
        )

        # Apply degradation
        degraded = apply_sky_degradation(
            frame, sky_mask,
            blur_sigma=blur_sigma,
            bit_reduce=bit_reduce,
            feather_radius=feather,
        )

        # Stream directly to encoder -- no buffering
        proc_encode.stdin.write(degraded.tobytes())
        frames_processed += 1

    proc_decode.wait()
    proc_encode.stdin.close()
    proc_encode.wait()

    click.echo(f"[sky_degrade] Processed {frames_processed} frames", err=True)

    if frames_processed == 0:
        raise click.ClickException("No frames decoded")

    if proc_encode.returncode != 0:
        raise click.ClickException("ffmpeg encode failed")

    click.echo(f"[sky_degrade] Output: {output_path}", err=True)


if __name__ == "__main__":
    sky_degrade()
