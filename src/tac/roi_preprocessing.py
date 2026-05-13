#!/usr/bin/env python3
"""ROI-based video preprocessing for the comma video compression challenge.

Degrades non-important regions of dashcam video BEFORE encoding so the encoder
spends fewer bits on regions the evaluator does not care about.

Pipeline:
  1. Define a driving corridor polygon mask (trapezoid covering the road).
  2. Inside the mask: preserve original video fidelity.
  3. Outside the mask: apply Gaussian blur and optional chroma smoothing.
  4. Feather mask edges with a Gaussian blur on the alpha channel.
  5. Output a lossless intermediate (ffv1/mkv) for the existing encode pipeline.

Dependencies: numpy, scipy (for gaussian_filter), ffmpeg/ffprobe via subprocess.
All available through ``uv run``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------

def _ffprobe_meta(video: Path, ffprobe_bin: str) -> dict:
    """Return width, height, fps, and total_frames for *video*."""
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video),
    ]
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(cp.stdout)
    stream = data["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])

    # fps
    rate = stream.get("avg_frame_rate", "0/1")
    num, den = (int(x) for x in rate.split("/"))
    fps = num / den if den else 0.0

    # total frames
    nf = stream.get("nb_read_frames") or stream.get("nb_frames")
    if nf in (None, "N/A"):
        dur = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
        nf = int(round(dur * fps))
    else:
        nf = int(nf)

    return {"width": w, "height": h, "fps": fps, "total_frames": nf}


# ---------------------------------------------------------------------------
# Polygon mask rasterisation (pure numpy)
# ---------------------------------------------------------------------------

def _rasterize_polygon(vertices: Sequence[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """Return a float32 mask (H, W) with 1.0 inside the polygon, 0.0 outside.

    *vertices* are in normalised (0-1) coordinates: (x_frac, y_frac).
    Uses a scanline even-odd fill rule.
    """
    # Convert normalised coords to pixel coords
    pts = [(x * (width - 1), y * (height - 1)) for x, y in vertices]
    n = len(pts)
    mask = np.zeros((height, width), dtype=np.float32)

    # Determine bounding box in pixel rows
    ys = [p[1] for p in pts]
    y_min = max(0, int(np.floor(min(ys))))
    y_max = min(height - 1, int(np.ceil(max(ys))))

    for y in range(y_min, y_max + 1):
        # Find x-intersections with polygon edges
        intersections: list[float] = []
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            if y0 == y1:
                continue
            if (y < min(y0, y1)) or (y >= max(y0, y1)):
                continue
            t = (y - y0) / (y1 - y0)
            x_int = x0 + t * (x1 - x0)
            intersections.append(x_int)
        intersections.sort()
        # Fill between pairs
        for j in range(0, len(intersections) - 1, 2):
            xl = max(0, int(np.floor(intersections[j])))
            xr = min(width - 1, int(np.ceil(intersections[j + 1])))
            mask[y, xl:xr + 1] = 1.0
    return mask


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Return a normalised 1-D Gaussian kernel (pure numpy)."""
    radius = int(np.ceil(3.0 * sigma))
    if radius < 1:
        radius = 1
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _convolve_1d_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1-D convolution along *axis* with reflect-padding (pure numpy)."""
    radius = len(kernel) // 2
    # Pad along the target axis with reflect mode
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (radius, radius)
    padded = np.pad(arr, pad_widths, mode="reflect")

    # Sliding-window dot product via np.lib.stride_tricks is fast but tricky
    # with arbitrary ndim; use a simple accumulation loop instead -- kernel is
    # typically small (radius ~= 3*sigma, e.g. 15 for sigma=5).
    out = np.zeros_like(arr)
    n = len(kernel)
    for i in range(n):
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(i, i + arr.shape[axis])
        out += padded[tuple(slices)] * kernel[i]
    return out


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable Gaussian blur to a 2-D float32 array.

    Uses scipy if available (milliseconds), falls back to pure numpy (slower).
    """
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except ImportError:
        pass
    # Pure numpy fallback — cap kernel to avoid extreme slowness
    effective_sigma = min(sigma, 12.0)
    if effective_sigma < sigma:
        print(f"  [warn] _gaussian_blur_2d: sigma capped from {sigma} to {effective_sigma} (numpy fallback)", file=sys.stderr)
    kernel = _gaussian_kernel_1d(effective_sigma)
    tmp = _convolve_1d_axis(arr, kernel, axis=1)
    return _convolve_1d_axis(tmp, kernel, axis=0)


# ---------------------------------------------------------------------------
# Frame-level processing
# ---------------------------------------------------------------------------

def _sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude via 3x3 Sobel on a float32 grayscale image."""
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    # Pad with reflect
    padded = np.pad(gray, 1, mode="reflect")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            gx += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]] * kx[dy, dx]
            gy += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]] * ky[dy, dx]
    return np.sqrt(gx * gx + gy * gy)


def _compute_adaptive_mask(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    spatial_prior: np.ndarray,
    feather_sigma: float,
) -> np.ndarray:
    """Build a per-frame adaptive importance mask from gradient + temporal + spatial signals.

    Returns a float32 mask in [0, 1] where 1 = important (protect), 0 = unimportant (degrade).
    """
    h, w = frame.shape[:2]
    # Grayscale
    gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.float32)

    # Gradient magnitude (normalised)
    grad = _sobel_gradient_magnitude(gray)
    grad_max = grad.max()
    grad_norm = grad / grad_max if grad_max > 0 else grad

    # Temporal difference (normalised)
    if prev_frame is not None:
        prev_gray = (0.299 * prev_frame[:, :, 0] + 0.587 * prev_frame[:, :, 1] + 0.114 * prev_frame[:, :, 2]).astype(np.float32)
        temporal = np.abs(gray - prev_gray)
        temp_max = temporal.max()
        temporal_norm = temporal / temp_max if temp_max > 0 else temporal
    else:
        temporal_norm = np.zeros((h, w), dtype=np.float32)

    # Fuse: spatial prior + gradient + temporal
    # spatial_prior is the feathered trapezoid (1 inside corridor)
    # gradient captures edges (lane markings, car boundaries)
    # temporal captures moving objects
    raw_mask = np.maximum(
        spatial_prior * 0.7,
        0.35 * grad_norm + 0.45 * temporal_norm + 0.20 * spatial_prior,
    )

    # Soft threshold: push values toward 0 or 1
    raw_mask = np.clip(raw_mask * 1.5, 0.0, 1.0)

    # Light feathering to smooth per-frame noise
    if feather_sigma > 0:
        raw_mask = _gaussian_blur_2d(raw_mask, sigma=min(feather_sigma, 8.0))
        mm = raw_mask.max()
        if mm > 0:
            raw_mask = np.clip(raw_mask / mm, 0.0, 1.0)

    return raw_mask


def _blur_frame_rgb(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur an (H, W, 3) uint8 frame per-channel."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(frame.astype(np.float32), sigma=(sigma, sigma, 0)).clip(0, 255).astype(np.uint8)
    except ImportError:
        kernel = _gaussian_kernel_1d(min(sigma, 12.0))
        f = frame.astype(np.float32)
        tmp = _convolve_1d_axis(f, kernel, axis=1)
        blurred = _convolve_1d_axis(tmp, kernel, axis=0)
        return blurred.clip(0, 255).astype(np.uint8)


def _degrade_chroma_only(frame: np.ndarray, sigma: float, mask: np.ndarray) -> np.ndarray:
    """Degrade only chroma channels outside the mask, preserving luminance.

    Converts RGB→YCbCr, blurs Cb/Cr outside the mask, then converts back.
    This preserves the luminance gradients that PoseNet relies on while
    making chroma regions compress better.
    """
    f = frame.astype(np.float32)
    # RGB to YCbCr (BT.601 approximation)
    y  = 0.299 * f[:,:,0] + 0.587 * f[:,:,1] + 0.114 * f[:,:,2]
    cb = -0.169 * f[:,:,0] - 0.331 * f[:,:,1] + 0.500 * f[:,:,2] + 128.0
    cr = 0.500 * f[:,:,0] - 0.419 * f[:,:,1] - 0.081 * f[:,:,2] + 128.0

    # Blur chroma channels
    cb_blur = _gaussian_blur_2d(cb, sigma)
    cr_blur = _gaussian_blur_2d(cr, sigma)

    # Blend: inside mask keep original chroma, outside use blurred
    alpha = mask[:, :]  # 1.0 = inside corridor (protect), 0.0 = outside (degrade)
    cb_out = cb * alpha + cb_blur * (1.0 - alpha)
    cr_out = cr * alpha + cr_blur * (1.0 - alpha)

    # YCbCr back to RGB
    r = y + 1.402 * (cr_out - 128.0)
    g = y - 0.344 * (cb_out - 128.0) - 0.714 * (cr_out - 128.0)
    b = y + 1.772 * (cb_out - 128.0)

    result = np.stack([r, g, b], axis=-1)
    return result.clip(0, 255).astype(np.uint8)


def _blend_frames(
    original: np.ndarray,
    degraded: np.ndarray,
    mask: np.ndarray,
    blend_outside: float,
) -> np.ndarray:
    """Blend *original* and *degraded* using the soft *mask*.

    Inside the mask (mask=1.0): keep original.
    Outside the mask (mask=0.0): lerp toward *degraded* by *blend_outside*.
    """
    # mask shape (H, W) -> (H, W, 1) for broadcasting
    alpha = mask[:, :, np.newaxis]  # 1.0 inside corridor, 0.0 outside

    # Outside the corridor: blend original toward degraded by blend_outside factor.
    # effective = original * (alpha + (1-alpha)*(1-blend_outside)) + degraded * (1-alpha)*blend_outside
    keep_factor = alpha + (1.0 - alpha) * (1.0 - blend_outside)
    deg_factor = (1.0 - alpha) * blend_outside

    result = (original.astype(np.float32) * keep_factor +
              degraded.astype(np.float32) * deg_factor)
    return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Temporal segment support
# ---------------------------------------------------------------------------

def _load_temporal_segments(path: Path) -> list[dict]:
    """Load a JSON file mapping frame ranges to corridor polygons.

    Expected format::

        [
            {
                "frame_start": 0,
                "frame_end": 900,
                "corridor": [[0.20, 0.45], [0.80, 0.45], [1.0, 1.0], [0.0, 1.0]]
            },
            ...
        ]
    """
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def preprocess(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    corridor_vertices: list[tuple[float, float]],
    outside_blur_sigma: float,
    feather_radius: int,
    outside_blend: float,
    temporal_segments: list[dict] | None,
    adaptive_mask: bool = False,
    chroma_only: bool = False,
    mask_file: Path | None = None,
    dry_run: bool,
) -> int:
    meta = _ffprobe_meta(input_path, ffprobe_bin)
    width = meta["width"]
    height = meta["height"]
    fps = meta["fps"]
    total_frames = meta["total_frames"]

    print(f"Source: {input_path}", file=sys.stderr)
    print(f"Resolution: {width}x{height}, FPS: {fps:.3f}, Frames: {total_frames}", file=sys.stderr)
    print(f"Corridor vertices (normalised): {corridor_vertices}", file=sys.stderr)
    print(f"Outside blur sigma: {outside_blur_sigma}", file=sys.stderr)
    print(f"Feather radius: {feather_radius}px", file=sys.stderr)
    print(f"Outside blend factor: {outside_blend}", file=sys.stderr)

    if dry_run:
        # Build and display the default mask to verify geometry
        mask = _rasterize_polygon(corridor_vertices, width, height)
        inside_px = int(mask.sum())
        total_px = width * height
        print(f"Mask inside pixels: {inside_px}/{total_px} ({100*inside_px/total_px:.1f}%)", file=sys.stderr)
        print("Dry run complete -- no video processed.", file=sys.stderr)
        return 0

    # Pre-compute the feathered corridor mask for the default corridor.
    # If temporal segments are provided, we will recompute per-segment.
    default_mask_hard = _rasterize_polygon(corridor_vertices, width, height)
    default_mask = _gaussian_blur_2d(default_mask_hard, sigma=feather_radius) if feather_radius > 0 else default_mask_hard
    # Renormalise so the peak is 1.0 (feathering can reduce the max)
    mask_max = default_mask.max()
    if mask_max > 0:
        default_mask = np.clip(default_mask / mask_max, 0.0, 1.0)

    inside_px = int((default_mask > 0.5).sum())
    total_px = width * height
    print(f"Effective mask coverage (>0.5): {inside_px}/{total_px} ({100*inside_px/total_px:.1f}%)", file=sys.stderr)

    # Load pre-generated ML masks if provided (overrides polygon/adaptive)
    ml_masks: np.ndarray | None = None
    if mask_file is not None:
        print(f"Loading pre-generated masks from {mask_file} ...", file=sys.stderr)
        ml_masks = np.load(str(mask_file))
        print(f"  Loaded: shape={ml_masks.shape}, mean={ml_masks.mean():.3f}, coverage(>0.5)={100*(ml_masks > 0.5).mean():.1f}%", file=sys.stderr)
        if ml_masks.ndim == 3 and ml_masks.shape[1] == height and ml_masks.shape[2] == width:
            print(f"  Per-frame masks: {ml_masks.shape[0]} frames", file=sys.stderr)
        else:
            print(f"  WARNING: mask shape {ml_masks.shape} doesn't match video {height}x{width}", file=sys.stderr)

    # Build per-segment mask lookup if temporal segments are provided
    segment_masks: list[tuple[int, int, np.ndarray]] | None = None
    if temporal_segments:
        segment_masks = []
        for seg in temporal_segments:
            verts = [(float(v[0]), float(v[1])) for v in seg["corridor"]]
            hard = _rasterize_polygon(verts, width, height)
            soft = _gaussian_blur_2d(hard, sigma=feather_radius) if feather_radius > 0 else hard
            sm = soft.max()
            if sm > 0:
                soft = np.clip(soft / sm, 0.0, 1.0)
            segment_masks.append((int(seg["frame_start"]), int(seg["frame_end"]), soft))

    frame_bytes = width * height * 3  # rgb24

    # --- Open ffmpeg reader pipe ---
    reader_cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-i", str(input_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    # --- Open ffmpeg writer pipe (lossless ffv1 in mkv) ---
    # Use the original fps from the source.
    fps_str = f"{fps:.6f}" if fps != int(fps) else str(int(fps))
    writer_cmd = [
        ffmpeg_bin,
        "-y",
        "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", fps_str,
        "-i", "-",
        "-c:v", "ffv1",
        "-level", "3",
        "-slicecrc", "1",
        str(output_path),
    ]

    reader_proc = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    writer_proc = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    if adaptive_mask:
        print("Adaptive masking ENABLED (gradient + temporal + spatial prior)", file=sys.stderr)

    frame_idx = 0
    prev_frame: np.ndarray | None = None
    try:
        while True:
            raw = reader_proc.stdout.read(frame_bytes)  # type: ignore[union-attr]
            if len(raw) < frame_bytes:
                break

            # .copy() is required because np.frombuffer returns a read-only view;
            # downstream processing (blur, blend) needs a writable array.
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

            # Pick the mask for this frame
            if ml_masks is not None:
                # Use pre-generated ML mask (interpolate if fewer masks than frames)
                if ml_masks.shape[0] >= total_frames:
                    mask = ml_masks[frame_idx]
                else:
                    # Linear interpolation between nearest keyframes
                    ratio = frame_idx / max(total_frames - 1, 1)
                    ml_idx = ratio * (ml_masks.shape[0] - 1)
                    lo = int(ml_idx)
                    hi = min(lo + 1, ml_masks.shape[0] - 1)
                    t = ml_idx - lo
                    mask = ml_masks[lo] * (1.0 - t) + ml_masks[hi] * t
            elif adaptive_mask:
                # Use the spatial prior as the base, but modulate with gradient + temporal
                base_spatial = default_mask
                if segment_masks is not None:
                    for seg_start, seg_end, seg_mask in segment_masks:
                        if seg_start <= frame_idx < seg_end:
                            base_spatial = seg_mask
                            break
                mask = _compute_adaptive_mask(frame, prev_frame, base_spatial, feather_sigma=4.0)
                prev_frame = frame
            else:
                mask = default_mask
                if segment_masks is not None:
                    for seg_start, seg_end, seg_mask in segment_masks:
                        if seg_start <= frame_idx < seg_end:
                            mask = seg_mask
                            break

            # Degrade outside the corridor
            if chroma_only:
                blended = _degrade_chroma_only(frame, outside_blur_sigma, mask)
            else:
                degraded = _blur_frame_rgb(frame, outside_blur_sigma)
                blended = _blend_frames(frame, degraded, mask, outside_blend)

            writer_proc.stdin.write(blended.tobytes())  # type: ignore[union-attr]

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  Processed {frame_idx} frames ...", file=sys.stderr, flush=True)

    finally:
        # Clean shutdown -- close both pipe ends before waiting
        if reader_proc.stdout:
            reader_proc.stdout.close()
        if writer_proc.stdin:
            writer_proc.stdin.close()
        writer_proc.wait()
        reader_proc.wait()

    if reader_proc.returncode != 0:
        print(f"ffmpeg reader error (rc={reader_proc.returncode})", file=sys.stderr)
        return 1
    if writer_proc.returncode != 0:
        print(f"ffmpeg writer error (rc={writer_proc.returncode})", file=sys.stderr)
        return 1

    print(f"Preprocessed {frame_idx} frames -> {output_path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_coord(s: str) -> tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'x,y' but got '{s}'")
    return (float(parts[0]), float(parts[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ROI-based video preprocessing: degrade non-corridor regions before encoding.",
    )
    p.add_argument("--input", required=True, type=Path, help="Source video path")
    p.add_argument("--output", required=True, type=Path, help="Preprocessed output path (lossless ffv1/mkv)")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")

    # Corridor polygon (normalised 0-1 coords)
    p.add_argument("--corridor-top-left", type=_parse_coord, default=(0.20, 0.45),
                    help="Top-left vertex as 'x,y' in 0-1 range (default: 0.20,0.45)")
    p.add_argument("--corridor-top-right", type=_parse_coord, default=(0.80, 0.45),
                    help="Top-right vertex as 'x,y' in 0-1 range (default: 0.80,0.45)")
    p.add_argument("--corridor-bottom-right", type=_parse_coord, default=(1.0, 1.0),
                    help="Bottom-right vertex as 'x,y' in 0-1 range (default: 1.0,1.0)")
    p.add_argument("--corridor-bottom-left", type=_parse_coord, default=(0.0, 1.0),
                    help="Bottom-left vertex as 'x,y' in 0-1 range (default: 0.0,1.0)")

    # Degradation parameters
    p.add_argument("--outside-blur-sigma", type=float, default=2.5,
                    help="Gaussian blur sigma for outside-corridor degradation (default: 2.5)")
    p.add_argument("--feather-radius", type=int, default=48,
                    help="Gaussian blur radius in pixels for mask edge feathering (default: 48)")
    p.add_argument("--outside-blend", type=float, default=0.60,
                    help="Blend factor for outside region: 0=keep original, 1=fully degraded (default: 0.60)")

    # Temporal segments
    p.add_argument("--temporal-segments", type=Path, default=None,
                    help="Optional JSON file with per-segment corridor definitions")

    # Adaptive masking
    p.add_argument("--adaptive-mask", action="store_true",
                    help="Use gradient+temporal adaptive mask instead of static polygon")

    # Chroma-only degradation
    p.add_argument("--chroma-only", action="store_true",
                    help="Degrade only chroma channels outside corridor (preserves luminance for PoseNet)")

    # Pre-generated ML masks (overrides polygon/adaptive mask)
    p.add_argument("--mask-file", type=Path, default=None,
                    help="Path to .npy file with pre-generated masks (N, H, W) float32. "
                         "Overrides corridor polygon and adaptive masking. "
                         "If N < total_frames, masks are linearly interpolated.")

    # Utility
    p.add_argument("--dry-run", action="store_true",
                    help="Print mask geometry and exit without processing")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    corridor_vertices = [
        args.corridor_top_left,
        args.corridor_top_right,
        args.corridor_bottom_right,
        args.corridor_bottom_left,
    ]

    temporal_segments = None
    if args.temporal_segments is not None:
        temporal_segments = _load_temporal_segments(args.temporal_segments)

    return preprocess(
        args.input,
        args.output,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        corridor_vertices=corridor_vertices,
        outside_blur_sigma=args.outside_blur_sigma,
        feather_radius=args.feather_radius,
        outside_blend=args.outside_blend,
        temporal_segments=temporal_segments,
        adaptive_mask=args.adaptive_mask,
        chroma_only=args.chroma_only,
        mask_file=args.mask_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# ROI map generation (migrated from submissions/robust_current/generate_roi_map.py)
# ---------------------------------------------------------------------------

def masks_to_roi_map(
    masks: "np.ndarray",
    encode_w: int,
    encode_h: int,
    total_frames: int,
    road_qp_delta: int = -10,
    sky_qp_delta: int = 10,
    neutral_qp_delta: int = 0,
    threshold_high: float = 0.6,
    threshold_low: float = 0.3,
) -> list[str]:
    """Convert importance masks to SVT-AV1 ROI map lines.

    Args:
        masks: (N, H, W) float32 importance masks (1=important, 0=unimportant)
        encode_w: encode resolution width (e.g., 524)
        encode_h: encode resolution height (e.g., 394)
        total_frames: total video frames
        road_qp_delta: QP offset for important regions (negative = more bits)
        sky_qp_delta: QP offset for unimportant regions (positive = fewer bits)
        neutral_qp_delta: QP offset for neutral regions
        threshold_high: mask value above which region is "important"
        threshold_low: mask value below which region is "unimportant"

    Returns:
        List of ROI map lines in SVT-AV1 format.
    """
    import math

    cols = math.ceil(encode_w / 64)
    rows = math.ceil(encode_h / 64)

    mask_h, mask_w = masks.shape[1], masks.shape[2]
    scale_y = mask_h / encode_h
    scale_x = mask_w / encode_w

    lines: list[str] = []
    prev_offsets: list[int] | None = None

    for frame_idx in range(total_frames):
        if masks.shape[0] >= total_frames:
            mask = masks[frame_idx]
        else:
            ratio = frame_idx / max(total_frames - 1, 1)
            ml_idx = ratio * (masks.shape[0] - 1)
            lo = int(ml_idx)
            hi = min(lo + 1, masks.shape[0] - 1)
            t = ml_idx - lo
            mask = masks[lo] * (1.0 - t) + masks[hi] * t

        offsets: list[int] = []
        for by in range(rows):
            for bx in range(cols):
                y0 = int(by * 64 * scale_y)
                y1 = int(min((by + 1) * 64, encode_h) * scale_y)
                x0 = int(bx * 64 * scale_x)
                x1 = int(min((bx + 1) * 64, encode_w) * scale_x)
                block_mean = float(mask[y0:y1, x0:x1].mean())
                if block_mean >= threshold_high:
                    offsets.append(road_qp_delta)
                elif block_mean <= threshold_low:
                    offsets.append(sky_qp_delta)
                else:
                    offsets.append(neutral_qp_delta)

        if offsets != prev_offsets:
            offset_str = " ".join(str(o) for o in offsets)
            lines.append(f"{frame_idx} {offset_str}")
            prev_offsets = offsets

    return lines
