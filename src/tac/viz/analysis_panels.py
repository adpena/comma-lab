# SPDX-License-Identifier: MIT
"""Six-panel multipane analysis visualization for TTO frames.

Generates a side-by-side comparison of ground truth vs reconstructed frames
with SegNet semantic analysis, suitable for research diagnostics and paper
figures.

Layout (3 columns x 2 rows):
    Row 1: GT Original       | Our Reconstruction  | Pixel Error (hot colormap)
    Row 2: GT SegNet Overlay  | Our SegNet Overlay   | SegNet Disagreement

The visualization accepts TTO frames (at model resolution 384x512) and
optionally upscales to camera resolution (874x1164) to match the
authoritative scorer pipeline.

Usage as library::

    from tac.viz.analysis_panels import generate_analysis_panels

    result = generate_analysis_panels(
        tto_frames=frames_tensor,
        gt_video_path=Path("upstream/videos/0.mkv"),
        segnet=segnet_model,
        output_dir=Path("/tmp/viz"),
        auth_matched=True,
    )

Usage from CLI::

    python scripts/generate_analysis_viz.py \\
        --frames /tmp/tto_frames.pt \\
        --upstream upstream/ \\
        --output /tmp/viz/ \\
        --auth-matched
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.camera import CAMERA_H, CAMERA_W, FRAME_H, FRAME_W

# SegNet class colors for overlay visualization (5 classes, RGB order).
# Matches the upstream comma challenge color convention.
SEGNET_OVERLAY_COLORS = np.array(
    [
        [64, 64, 64],      # 0: road (dark gray)
        [255, 255, 0],     # 1: lane marking (yellow)
        [0, 0, 255],       # 2: vehicle / undrivable (blue)
        [255, 0, 0],       # 3: movable / pedestrian (red)
        [0, 255, 0],       # 4: other (green)
    ],
    dtype=np.uint8,
)

OVERLAY_ALPHA = 0.45

# Clip defaults (seconds 12-18 captures highway driving with lane changes)
DEFAULT_CLIP_START_SEC = 12.0
DEFAULT_CLIP_DURATION_SEC = 6.0
DEFAULT_FPS = 20


def _segnet_classes(
    frame_chw: torch.Tensor,
    segnet: nn.Module,
) -> np.ndarray:
    """Run SegNet on a single frame and return per-pixel argmax classes.

    Args:
        frame_chw: (1, 3, H, W) float tensor on the model device.
        segnet: frozen SegNet module with ``preprocess_input`` method.

    Returns:
        (Hs, Ws) numpy int array of class indices (0 .. NUM_CLASSES-1).
    """
    # SegNet expects (B, T, C, H, W) with T=2; duplicate frame to form pair.
    pair = frame_chw.unsqueeze(1).expand(-1, 2, -1, -1, -1)
    seg_input = segnet.preprocess_input(pair)
    seg_output = segnet(seg_input)
    # seg_output: (B, num_classes, Hs, Ws) -- SegNet selects last frame from pair internally.
    classes = seg_output[0].argmax(dim=0).cpu().numpy()
    return classes


def _blend_segnet_overlay(
    frame_rgb: np.ndarray,
    class_map: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend SegNet class colors onto an RGB frame.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB array.
        class_map: (Hs, Ws) int array of class indices.
        alpha: overlay opacity (0 = frame only, 1 = color only).

    Returns:
        (H, W, 3) uint8 blended array at the frame resolution.
    """
    from PIL import Image

    h, w = frame_rgb.shape[:2]
    hs, ws = class_map.shape

    # Map classes to colors at SegNet resolution.
    color_mask = SEGNET_OVERLAY_COLORS[class_map]  # (Hs, Ws, 3)

    # Upscale color mask to frame resolution if needed.
    if (hs, ws) != (h, w):
        color_mask = np.array(
            Image.fromarray(color_mask).resize((w, h), Image.NEAREST)
        )

    blended = (
        (1.0 - alpha) * frame_rgb.astype(np.float32)
        + alpha * color_mask.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _pixel_error_heatmap(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel absolute error and render as a hot colormap.

    Args:
        frame_a: (H, W, 3) uint8 RGB -- ground truth.
        frame_b: (H, W, 3) uint8 RGB -- reconstruction.

    Returns:
        (H, W, 3) uint8 RGB heatmap (matplotlib 'hot' colormap).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    # Mean absolute error across channels, normalized to [0, 1].
    diff = np.abs(
        frame_a.astype(np.float32) - frame_b.astype(np.float32)
    ).mean(axis=2)
    max_val = diff.max()
    if max_val > 0:
        diff_norm = diff / max_val
    else:
        diff_norm = diff

    # Apply 'hot' colormap.
    colored = cm.hot(diff_norm)[:, :, :3]  # (H, W, 3) float [0, 1]
    return (colored * 255).astype(np.uint8)


def _segnet_disagreement_mask(
    classes_a: np.ndarray,
    classes_b: np.ndarray,
    display_h: int,
    display_w: int,
) -> np.ndarray:
    """Binary disagreement mask: white where class predictions differ.

    Args:
        classes_a: (Hs, Ws) int array (e.g. GT SegNet classes).
        classes_b: (Hs, Ws) int array (e.g. reconstruction SegNet classes).
        display_h: target display height.
        display_w: target display width.

    Returns:
        (display_h, display_w, 3) uint8 RGB array (white on black).
    """
    from PIL import Image

    diff = (classes_a != classes_b).astype(np.uint8) * 255
    diff_img = Image.fromarray(diff, mode="L").resize(
        (display_w, display_h), Image.NEAREST
    )
    diff_np = np.array(diff_img)
    return np.stack([diff_np, diff_np, diff_np], axis=-1)


def _load_mono_font(size: int = 14):
    """Try to load a monospace font; fall back to PIL default."""
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_label_bar(
    width: int,
    height: int,
    labels: list[str],
    bg: tuple[int, int, int] = (10, 10, 10),
    fg: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 14,
) -> np.ndarray:
    """Render a horizontal bar with evenly spaced centered text labels.

    Args:
        width: total bar width in pixels.
        height: bar height in pixels.
        labels: list of text strings, one per column.
        bg: background RGB color.
        fg: foreground text RGB color.
        font_size: approximate font size.

    Returns:
        (height, width, 3) uint8 RGB array.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = _load_mono_font(font_size)

    section_w = width // max(len(labels), 1)
    for i, text in enumerate(labels):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = i * section_w + (section_w - tw) // 2
        y = (height - th) // 2
        draw.text((x, y), text, fill=fg, font=font)

    return np.array(img)


def _upscale_frames(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Bicubic upscale (N, H, W, 3) uint8 tensor to target resolution.

    Uses the same bicubic interpolation as evaluate.py to match
    authoritative scorer behavior.

    Args:
        frames: (N, H, W, 3) uint8 tensor.
        target_h: target height.
        target_w: target width.
        device: compute device.
        batch_size: frames per batch for GPU memory management.

    Returns:
        (N, target_h, target_w, 3) uint8 tensor on CPU.
    """
    n = frames.shape[0]
    out = []
    for i in range(0, n, batch_size):
        batch = frames[i : i + batch_size].float().permute(0, 3, 1, 2).to(device)
        batch_up = F.interpolate(
            batch, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        batch_uint8 = batch_up.round().clamp(0, 255).to(torch.uint8)
        out.append(batch_uint8.permute(0, 2, 3, 1).cpu())
    return torch.cat(out, dim=0)


def generate_analysis_panels(
    tto_frames: torch.Tensor,
    gt_video_path: Path,
    segnet: nn.Module,
    output_dir: Path,
    *,
    clip_start_sec: float = DEFAULT_CLIP_START_SEC,
    clip_duration_sec: float = DEFAULT_CLIP_DURATION_SEC,
    fps: int = DEFAULT_FPS,
    auth_matched: bool = True,
    camera_h: int = CAMERA_H,
    camera_w: int = CAMERA_W,
    gif_scale: float = 0.5,
    gif_fps: int = 15,
    label_height: int = 36,
) -> dict[str, Any]:
    """Generate 6-panel analysis visualization.

    Layout:
        Row 1: GT Original | Our Reconstruction | Pixel Error (hot)
        Row 2: GT SegNet   | Our SegNet         | SegNet Disagreement

    Args:
        tto_frames: (N, H, W, 3) uint8 tensor at model resolution (384x512).
        gt_video_path: path to ground-truth ``.mkv`` video.
        segnet: frozen SegNet module (on device, with ``preprocess_input``).
        output_dir: directory for output GIF and MP4 files.
        clip_start_sec: start of the clip to visualize (seconds).
        clip_duration_sec: duration of the clip (seconds).
        fps: video frame rate.
        auth_matched: if True, upscale frames to camera resolution for display.
        camera_h: camera native height (used when ``auth_matched=True``).
        camera_w: camera native width (used when ``auth_matched=True``).
        gif_scale: downscale factor for GIF output (0.5 = half resolution).
        gif_fps: GIF playback frame rate.
        label_height: height of text label bars in pixels.

    Returns:
        Dictionary with keys:
            ``gif_path``: Path to generated GIF.
            ``mp4_path``: Path to generated MP4 (or None if ffmpeg unavailable).
            ``seg_disagree_mean``: mean SegNet disagreement fraction.
            ``pixel_error_mean``: mean pixel error (0-255 scale).
            ``n_frames_rendered``: number of frames in the visualization.
    """
    from PIL import Image

    t_start = time.monotonic()

    gt_video_path = Path(gt_video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs.
    assert tto_frames.ndim == 4, f"Expected 4D tensor, got {tto_frames.ndim}D"
    assert tto_frames.dtype == torch.uint8, f"Expected uint8, got {tto_frames.dtype}"
    n_total = tto_frames.shape[0]

    # Determine clip frame range.
    frame_start = int(clip_start_sec * fps)
    frame_end = min(int((clip_start_sec + clip_duration_sec) * fps), n_total)
    if frame_start >= n_total:
        frame_start = 0
        frame_end = min(int(clip_duration_sec * fps), n_total)

    tto_clip = tto_frames[frame_start:frame_end]
    n_clip = tto_clip.shape[0]

    if n_clip == 0:
        raise ValueError(
            f"Empty clip: start={frame_start}, end={frame_end}, total={n_total}"
        )

    # Decode GT video.  Always need model resolution for SegNet; camera
    # resolution is only needed when auth_matched is True for display.
    print(f"  [viz] Decoding GT video: {gt_video_path}")
    from tac.data import decode_video

    gt_frames_model = decode_video(
        str(gt_video_path), target_h=FRAME_H, target_w=FRAME_W
    )
    gt_clip_model = gt_frames_model[frame_start:frame_end]

    if auth_matched:
        gt_frames_camera = decode_video(
            str(gt_video_path), target_h=camera_h, target_w=camera_w
        )
        gt_clip_camera = gt_frames_camera[frame_start:frame_end]
    else:
        gt_clip_camera = None  # not used when auth_matched is False

    # Determine display resolution.
    if auth_matched:
        display_h, display_w = camera_h, camera_w
    else:
        display_h, display_w = FRAME_H, FRAME_W

    # Detect device from segnet parameters.
    device = next(segnet.parameters()).device

    # Upscale TTO frames if auth_matched.
    if auth_matched:
        print(f"  [viz] Upscaling {n_clip} TTO frames to {camera_h}x{camera_w}")
        tto_display = _upscale_frames(tto_clip, camera_h, camera_w, device)
    else:
        tto_display = tto_clip

    # Panel dimensions.
    panel_h = display_h
    panel_w = display_w
    n_cols = 3
    total_w = panel_w * n_cols
    total_h = label_height + panel_h + label_height + panel_h

    # Pre-render label bars.
    top_labels = _draw_label_bar(
        total_w,
        label_height,
        ["GT Original", "Our Reconstruction", "Pixel Error"],
        font_size=max(12, label_height // 3),
    )
    bottom_labels = _draw_label_bar(
        total_w,
        label_height,
        ["GT SegNet", "Our SegNet", "SegNet Disagreement"],
        font_size=max(12, label_height // 3),
        fg=(180, 180, 180),
    )

    # Accumulate metrics and GIF frames.
    gif_pil_frames: list[Image.Image] = []
    seg_disagree_fractions: list[float] = []
    pixel_errors: list[float] = []

    print(f"  [viz] Rendering {n_clip} frames (display: {display_w}x{display_h})")
    with torch.no_grad():
        for i in range(n_clip):
            # Display-resolution frames (numpy).
            gt_display = gt_clip_camera[i] if auth_matched else gt_clip_model[i]
            gt_np = gt_display.cpu().numpy()
            tto_np = tto_display[i].cpu().numpy()

            # Ensure both are the same size.
            assert gt_np.shape[:2] == (display_h, display_w), (
                f"GT shape {gt_np.shape[:2]} != display ({display_h}, {display_w})"
            )
            assert tto_np.shape[:2] == (display_h, display_w), (
                f"TTO shape {tto_np.shape[:2]} != display ({display_h}, {display_w})"
            )

            # SegNet inference at model resolution (384x512).
            gt_model_chw = (
                gt_clip_model[i]
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            tto_model_chw = (
                tto_clip[i]
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            gt_cls = _segnet_classes(gt_model_chw, segnet)
            tto_cls = _segnet_classes(tto_model_chw, segnet)

            # Row 1, Col 1: GT Original.
            # Row 1, Col 2: Our Reconstruction.
            # Row 1, Col 3: Pixel Error heatmap.
            error_heatmap = _pixel_error_heatmap(gt_np, tto_np)

            top_row = np.concatenate([gt_np, tto_np, error_heatmap], axis=1)

            # Row 2, Col 1: GT SegNet overlay.
            gt_overlay = _blend_segnet_overlay(gt_np, gt_cls)

            # Row 2, Col 2: Our SegNet overlay.
            tto_overlay = _blend_segnet_overlay(tto_np, tto_cls)

            # Row 2, Col 3: SegNet disagreement mask.
            disagree_mask = _segnet_disagreement_mask(
                gt_cls, tto_cls, panel_h, panel_w
            )

            bottom_row = np.concatenate(
                [gt_overlay, tto_overlay, disagree_mask], axis=1
            )

            # Full composite.
            composite = np.concatenate(
                [top_labels, top_row, bottom_labels, bottom_row], axis=0
            )

            gif_pil_frames.append(Image.fromarray(composite))

            # Metrics.
            total_pixels = gt_cls.shape[0] * gt_cls.shape[1]
            disagree_count = int((gt_cls != tto_cls).sum())
            seg_disagree_fractions.append(disagree_count / total_pixels)

            pixel_err = np.abs(
                gt_np.astype(np.float32) - tto_np.astype(np.float32)
            ).mean()
            pixel_errors.append(float(pixel_err))

            if (i + 1) % 20 == 0 or i == 0:
                print(
                    f"    frame {frame_start + i}: "
                    f"seg_disagree={seg_disagree_fractions[-1]:.4f} "
                    f"pixel_err={pixel_errors[-1]:.2f}"
                )

    # Compute summary metrics.
    seg_disagree_mean = float(np.mean(seg_disagree_fractions))
    pixel_error_mean = float(np.mean(pixel_errors))

    # Write GIF.
    gif_path = output_dir / "analysis_panels.gif"
    gif_w = int(total_w * gif_scale)
    gif_h = int(total_h * gif_scale)

    resized = [f.resize((gif_w, gif_h), Image.LANCZOS) for f in gif_pil_frames]
    duration_ms = int(1000 / gif_fps)
    gif_buf = io.BytesIO()
    resized[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=resized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    gif_path.write_bytes(gif_buf.getvalue())
    gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"  [viz] GIF saved: {gif_path} ({gif_size_mb:.1f} MB)")

    # Write MP4 via ffmpeg if available.
    mp4_path: Path | None = None
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        mp4_path = output_dir / "analysis_panels.mp4"
        _write_mp4_ffmpeg(
            gif_pil_frames, mp4_path, fps=gif_fps, ffmpeg_bin=ffmpeg_bin
        )
        mp4_size_mb = mp4_path.stat().st_size / (1024 * 1024)
        print(f"  [viz] MP4 saved: {mp4_path} ({mp4_size_mb:.1f} MB)")
    else:
        print("  [viz] ffmpeg not found, skipping MP4 generation")

    elapsed = time.monotonic() - t_start
    print(
        f"  [viz] Done in {elapsed:.1f}s: "
        f"seg_disagree_mean={seg_disagree_mean:.4f}, "
        f"pixel_error_mean={pixel_error_mean:.2f}"
    )

    return {
        "gif_path": gif_path,
        "mp4_path": mp4_path,
        "seg_disagree_mean": seg_disagree_mean,
        "pixel_error_mean": pixel_error_mean,
        "n_frames_rendered": n_clip,
        "elapsed_seconds": elapsed,
    }


def _write_mp4_ffmpeg(
    frames: list,
    output_path: Path,
    fps: int,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Write a list of PIL Image frames to MP4 using ffmpeg subprocess.

    Uses rawvideo pipe input to avoid PyAV dependency in headless
    environments.

    Args:
        frames: list of PIL Image objects (all same size).
        output_path: destination MP4 path.
        fps: output frame rate.
        ffmpeg_bin: path to ffmpeg binary.
    """
    if not frames:
        return

    w, h = frames[0].size
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for frame in frames:
        raw = np.array(frame.convert("RGB")).tobytes()
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait(timeout=120)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
