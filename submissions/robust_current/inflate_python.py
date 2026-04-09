#!/usr/bin/env python3
"""Python-based inflate path with PyTorch/PIL bicubic upscale and binomial USM.

The top entries on the leaderboard use Python (PyTorch bicubic or PIL Lanczos)
for upscaling instead of ffmpeg's scale filter, and a 9x9 binomial kernel for
unsharp masking instead of ffmpeg's Gaussian unsharp.

This script replaces the ffmpeg-based inflate path in inflate.sh for the
non-ROI (flat) encode path.

Usage:
  python inflate_python.py \\
    --archive-dir <dir_with_mkv> \\
    --inflated-dir <output_dir> \\
    --video-names-file <names.txt> \\
    --source-w 1164 --source-h 874 \\
    --upscale-method bicubic \\
    --usm-strength 0.40 \\
    --usm-kernel-size 9
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


def _binomial_kernel_2d(size: int) -> np.ndarray:
    """Build a 2D binomial (Pascal's triangle) kernel.

    This matches the kernel used by the top leaderboard entries:
    Row N of Pascal's triangle, normalized. For size=9:
    [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256
    """
    # Build 1D kernel from Pascal's triangle
    k = np.array([1.0])
    for _ in range(size - 1):
        k = np.convolve(k, [1.0, 1.0])
    k = k / k.sum()
    # Outer product for 2D
    return np.outer(k, k).astype(np.float32)


def _apply_usm(frame: np.ndarray, kernel: np.ndarray, strength: float) -> np.ndarray:
    """Apply unsharp masking: output = input + strength * (input - blur).

    Uses the binomial kernel for the blur step, matching the top entries.
    """
    try:
        from scipy.signal import fftconvolve
        f = frame.astype(np.float32)
        # Apply blur per channel
        blurred = np.zeros_like(f)
        for c in range(f.shape[2]):
            blurred[:, :, c] = fftconvolve(f[:, :, c], kernel, mode='same')
        sharpened = f + strength * (f - blurred)
        return sharpened.clip(0, 255).astype(np.uint8)
    except ImportError:
        # Fallback: use numpy convolution (slower)
        from numpy.fft import fft2, ifft2
        f = frame.astype(np.float32)
        kh, kw = kernel.shape
        fh, fw = f.shape[:2]
        # Pad kernel to frame size
        padded_k = np.zeros((fh, fw), dtype=np.float32)
        padded_k[:kh, :kw] = kernel
        # Center the kernel
        padded_k = np.roll(padded_k, -(kh // 2), axis=0)
        padded_k = np.roll(padded_k, -(kw // 2), axis=1)
        K = fft2(padded_k)
        blurred = np.zeros_like(f)
        for c in range(f.shape[2]):
            blurred[:, :, c] = np.real(ifft2(fft2(f[:, :, c]) * K))
        sharpened = f + strength * (f - blurred)
        return sharpened.clip(0, 255).astype(np.uint8)


def _upscale_frame(frame: np.ndarray, target_h: int, target_w: int, method: str) -> np.ndarray:
    """Upscale a frame using PIL or PyTorch."""
    if method == "bicubic_torch":
        try:
            import torch
            import torch.nn.functional as F
            t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
            t = F.interpolate(t, size=(target_h, target_w), mode='bicubic', align_corners=False)
            return t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        except ImportError:
            method = "bicubic"  # fallback to PIL

    from PIL import Image
    if method == "bicubic":
        resample = Image.BICUBIC
    elif method == "lanczos":
        resample = Image.LANCZOS
    else:
        resample = Image.BICUBIC

    img = Image.fromarray(frame)
    img = img.resize((target_w, target_h), resample)
    return np.array(img)


def inflate_video(
    mkv_path: Path,
    out_path: Path,
    source_w: int,
    source_h: int,
    upscale_method: str,
    usm_strength: float,
    usm_kernel_size: int,
    ffmpeg_bin: str,
) -> None:
    """Decode MKV, upscale with Python, apply binomial USM, write raw RGB."""
    # Decode compressed video to raw frames via ffmpeg
    # We decode at the compressed resolution, then upscale in Python
    probe_cmd = [
        ffmpeg_bin, "-v", "error",
        "-i", str(mkv_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-"
    ]

    # Get compressed resolution from the MKV
    probe = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(mkv_path).replace(ffmpeg_bin, "ffprobe")],  # hack
        capture_output=True, text=True
    )
    # Fallback: just use ffprobe
    ffprobe_bin = ffmpeg_bin.replace("ffmpeg", "ffprobe")
    probe = subprocess.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(mkv_path)],
        capture_output=True, text=True, check=True
    )
    cw, ch = [int(x) for x in probe.stdout.strip().split(",")]

    # YUV420p: Y plane = cw*ch, U plane = (cw//2)*(ch//2), V plane = (cw//2)*(ch//2)
    y_size = cw * ch
    uv_size = (cw // 2) * (ch // 2)
    frame_bytes_yuv = y_size + 2 * uv_size
    frame_bytes_out = source_w * source_h * 3

    # Build USM kernel
    kernel = _binomial_kernel_2d(usm_kernel_size)

    # Decode to YUV420p (NOT rgb24) so we can apply BT.601 conversion
    # matching the evaluator's yuv420_to_rgb() exactly
    decode_proc = subprocess.Popen(
        [ffmpeg_bin, "-v", "error", "-i", str(mkv_path),
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    frame_idx = 0
    with out_path.open("wb") as out_f:
        while True:
            raw = decode_proc.stdout.read(frame_bytes_yuv)
            if len(raw) < frame_bytes_yuv:
                break

            # Parse YUV420p planes
            y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape((ch, cw)).astype(np.float32)
            u = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape((ch // 2, cw // 2)).astype(np.float32)
            v = np.frombuffer(raw[y_size + uv_size:], dtype=np.uint8).reshape((ch // 2, cw // 2)).astype(np.float32)

            # Bilinear chroma upsampling — use torch F.interpolate to match evaluator EXACTLY
            import torch
            import torch.nn.functional as F_torch
            u_t = torch.from_numpy(u).unsqueeze(0).unsqueeze(0)
            v_t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
            u_up = F_torch.interpolate(u_t, size=(ch, cw), mode='bilinear', align_corners=False).squeeze().numpy()
            v_up = F_torch.interpolate(v_t, size=(ch, cw), mode='bilinear', align_corners=False).squeeze().numpy()

            # BT.601 limited range conversion (matching evaluator's yuv420_to_rgb exactly)
            yf = (y - 16.0) * (255.0 / 219.0)
            uf = (u_up - 128.0) * (255.0 / 224.0)
            vf = (v_up - 128.0) * (255.0 / 224.0)

            r = np.clip(yf + 1.402 * vf, 0, 255)
            g = np.clip(yf - 0.344136 * uf - 0.714136 * vf, 0, 255)
            b = np.clip(yf + 1.772 * uf, 0, 255)

            frame = np.round(np.stack([r, g, b], axis=-1)).astype(np.uint8)

            # Upscale with Python
            upscaled = _upscale_frame(frame, source_h, source_w, upscale_method)

            # Apply binomial USM
            if usm_strength > 0:
                upscaled = _apply_usm(upscaled, kernel, usm_strength)

            # Write raw RGB
            out_f.write(upscaled.tobytes())

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  Inflated {frame_idx} frames ...", file=sys.stderr, flush=True)

    decode_proc.stdout.close()
    decode_proc.wait()
    print(f"Inflated {frame_idx} frames -> {out_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="Python-based inflate with bicubic upscale + binomial USM")
    p.add_argument("--archive-dir", required=True, type=Path)
    p.add_argument("--inflated-dir", required=True, type=Path)
    p.add_argument("--video-names-file", required=True, type=Path)
    p.add_argument("--source-w", type=int, default=1164)
    p.add_argument("--source-h", type=int, default=874)
    p.add_argument("--upscale-method", default="bicubic", choices=["bicubic", "lanczos", "bicubic_torch"])
    p.add_argument("--usm-strength", type=float, default=0.40)
    p.add_argument("--usm-kernel-size", type=int, default=9)
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    args = p.parse_args()

    args.inflated_dir.mkdir(parents=True, exist_ok=True)

    for line in args.video_names_file.read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        stem = rel.rsplit(".", 1)[0]
        mkv_path = args.archive_dir / f"{stem}.mkv"
        out_path = args.inflated_dir / f"{stem}.raw"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Inflating {mkv_path} -> {out_path}", file=sys.stderr)
        inflate_video(
            mkv_path, out_path,
            args.source_w, args.source_h,
            args.upscale_method, args.usm_strength, args.usm_kernel_size,
            args.ffmpeg_bin,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
