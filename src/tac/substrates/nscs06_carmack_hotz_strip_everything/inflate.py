# SPDX-License-Identifier: MIT
"""Carmack-Hotz strip-everything inflate runtime — Path A v2 (chroma + optical-flow).

Per HNeRV parity discipline lesson L4 + L5 + strict-scorer-rule:

- ≤200 LOC inflate budget (substrate_engineering exception per L7; Path A
  redesign adds chroma + affine-warp logic — symposium commit 4292c8ce2).
- ≤2 external deps: numpy + Pillow (NO torch, NO smp, NO efficientnet).
- Full RGB renderer (per L5; chroma-MSE ≥20% of input chroma is the redefined
  bar per Assumption-Adversary VETO 2026-05-16).
- NO scorer load (per CLAUDE.md "Strict scorer rule" non-negotiable).

Path A forward path (cargo-cult unwound 2026-05-16):

1. Read ``<archive_dir>/0.bin`` bytes; call ``parse_archive`` (v2 schema).
2. For each pair, ARITH-DECODE per-cell class labels using uniform CDF.
   (Cargo-cult #1 UNWOUND: the per-class CDF is now ACTUALLY consumed.)
3. Arith-decode palette indices using per-cell class CDF rows.
4. Dequantize via palette -> uint8 grayscale (low-res).
5. Upsample grayscale + class-label maps -> per-frame uint8 RGB via per-class
   chroma anchors (Cargo-cult #2 UNWOUND: seg-scorer now sees coloured frames).
6. Apply affine warp from frame-0 to derive frame-1 using ALL 6 pose dims
   (Cargo-cult #4 UNWOUND: 2-of-6 translation replaced with 6-dim affine).
7. Save the contest raw stream to ``<output_stem>.raw``.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from tac.codec.pr98_channel_balance_zero_byte_bolt_on import (
    Pr98ChannelBalanceConfig,
    apply_pr98_channel_balance_to_decoded_pair,
)

from .archive import (
    POSE_DIMS,
    decode_class_label_stream,
    decode_grayscale_stream,
    dequantize_pose_deltas,
    parse_archive,
)

_PR98_L28_NSCS06_CONFIG = Pr98ChannelBalanceConfig(
    substrate_id="nscs06_carmack_hotz_strip_everything"
)


def _apply_pr98_l28_channel_balance_to_pair_uint8(
    frame_0: np.ndarray, frame_1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply canonical L28 PR98 balance to an HWC uint8 frame pair."""
    pair = np.stack((frame_0, frame_1), axis=0).transpose(0, 3, 1, 2)[None]
    balanced = apply_pr98_channel_balance_to_decoded_pair(
        pair.astype(np.float32),
        _PR98_L28_NSCS06_CONFIG,
    )[0]
    frame_0_balanced, frame_1_balanced = balanced.transpose(0, 2, 3, 1)
    return (
        frame_0_balanced.round().astype(np.uint8),
        frame_1_balanced.round().astype(np.uint8),
    )


def _grayscale_plus_chroma_to_rgb(
    gray_u8: np.ndarray, cls_u8: np.ndarray, chroma_palette: np.ndarray
) -> np.ndarray:
    """Map grayscale + per-pixel class -> RGB via per-class chroma anchor (Path A).

    Each pixel = chroma_palette[class] modulated by the local luma. We use the
    Y' fraction of BT.601 luminance (Y = 0.299 R + 0.587 G + 0.114 B) so the
    anchor RGB is scaled by (gray / anchor_luma) to preserve relative shading.

    Args:
        gray_u8: (H, W) uint8 luma at full resolution.
        cls_u8: (H, W) uint8 seg-scorer class labels at full resolution.
        chroma_palette: (NUM_SEGNET_CLASSES, 3) uint8 per-class RGB anchors.
    """
    if gray_u8.dtype != np.uint8 or cls_u8.dtype != np.uint8:
        raise ValueError("gray_u8 and cls_u8 must be uint8")
    if gray_u8.shape != cls_u8.shape:
        raise ValueError(f"shape mismatch {gray_u8.shape} vs {cls_u8.shape}")
    anchor = chroma_palette[cls_u8].astype(np.float32)  # (H, W, 3)
    # Per-class anchor luma; protects against divide-by-zero for black anchors.
    anchor_luma = (
        0.299 * anchor[..., 0] + 0.587 * anchor[..., 1] + 0.114 * anchor[..., 2]
    )
    safe_luma = np.where(anchor_luma > 1.0, anchor_luma, 1.0)
    scale = (gray_u8.astype(np.float32) / safe_luma)[..., None]
    out = np.clip(anchor * scale, 0.0, 255.0).astype(np.uint8)
    return out


def _affine_warp_frame1_from_frame0(
    frame_0: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """Derive frame_1 from frame_0 via 6-DOF affine warp from pose-net pose (Path A).

    pose-net's 6-dim pose vector is interpreted as (tx, ty, tz, rx, ry, rz):
    translation in fractional-image units + small-angle rotations in radians.
    We project to a 2-D affine in screen coords using small-angle approximations:

      x' = (1 + ε_tz) * (cos(rz)*x - sin(rz)*y) + tx*W
      y' = (1 + ε_tz) * (sin(rz)*x + cos(rz)*y) + ty*H

    where ε_tz = pose[2] * SCALE_TZ provides perspective-style zoom (forward
    motion enlarges the frame). The rx, ry pitch+yaw rotations are projected as
    additional translation in image-plane (paraxial approximation). This is the
    minimum-faithful 6-dim warp; future passes can replace with full homography
    + depth-from-pose. Numpy only; no Pillow.warp_perspective (which exists but
    is not bilinear-accurate for sub-pixel shifts).
    """
    if frame_0.ndim != 3 or frame_0.shape[2] != 3:
        raise ValueError(f"frame_0 must be (H, W, 3); got {frame_0.shape}")
    if pose.shape != (POSE_DIMS,):
        raise ValueError(f"pose must be ({POSE_DIMS},); got {pose.shape}")
    h, w, _ = frame_0.shape
    tx, ty, tz, rx, ry, rz = (float(v) for v in pose)

    # Build the 2x3 inverse affine matrix mapping output (x', y') -> source (x, y).
    # Forward map y_out = A * y_in + t; inverse for sampling.
    SCALE_T = 0.05  # fractional translation in [-5%, +5%] of W or H
    SCALE_R = 0.10  # rotation in radians (~5.7 degrees max for unit pose)
    SCALE_TZ = 0.05  # zoom in [-5%, +5%]
    SCALE_PITCH = 0.05  # rx -> ty (paraxial)
    SCALE_YAW = 0.05  # ry -> tx (paraxial)

    cos_rz = math.cos(rz * SCALE_R)
    sin_rz = math.sin(rz * SCALE_R)
    zoom = 1.0 + tz * SCALE_TZ
    if abs(zoom) < 1e-3:
        zoom = 1e-3  # degenerate guard

    # Inverse rotation + zoom: source = R^-1 * (output - translation) / zoom
    inv_cos = cos_rz / zoom
    inv_sin = sin_rz / zoom
    eff_tx = (tx + ry * SCALE_YAW) * SCALE_T * w
    eff_ty = (ty + rx * SCALE_PITCH) * SCALE_T * h

    # Sample grid centered at image center to keep rotation around centre.
    cy, cx = h * 0.5, w * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs_c = xs - cx - eff_tx
    ys_c = ys - cy - eff_ty
    src_x = inv_cos * xs_c + inv_sin * ys_c + cx
    src_y = -inv_sin * xs_c + inv_cos * ys_c + cy

    # Bilinear sample with edge replication.
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (src_x - x0)[..., None]
    wy = (src_y - y0)[..., None]
    f00 = frame_0[y0, x0].astype(np.float32)
    f01 = frame_0[y0, x1].astype(np.float32)
    f10 = frame_0[y1, x0].astype(np.float32)
    f11 = frame_0[y1, x1].astype(np.float32)
    top = f00 * (1.0 - wx) + f01 * wx
    bot = f10 * (1.0 - wx) + f11 * wx
    out = (top * (1.0 - wy) + bot * wy).clip(0.0, 255.0).astype(np.uint8)
    return out


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate one archive's bytes into a contest ``.raw`` file (Path A v2)."""
    from PIL import Image

    arc = parse_archive(archive_bytes)
    # Path A: decode per-cell class labels FIRST so the per-class CDF can be
    # consumed during the grayscale stream decode (cargo-cult #1 unwound).
    cls_lowres = decode_class_label_stream(
        arc.cls_arith_bytes,
        shape=(arc.num_pairs, arc.grayscale_h, arc.grayscale_w),
    )
    palette_indices = decode_grayscale_stream(arc, class_labels_lowres=cls_lowres)
    grayscale_lowres = arc.palette.dequantize(palette_indices)
    pose = dequantize_pose_deltas(
        arc.pose_bytes,
        num_pairs=arc.num_pairs,
        scale=arc.pose_quant_scale,
        zero=arc.pose_quant_zero,
    )

    raw_path = output_stem.with_suffix(".raw")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    debug_png_dir = output_stem if os.environ.get("NSCS06_DEBUG_PNG") == "1" else None
    if debug_png_dir is not None:
        debug_png_dir.mkdir(parents=True, exist_ok=True)

    with raw_path.open("wb") as fh:
        for p in range(arc.num_pairs):
            gray_full = np.array(
                Image.fromarray(grayscale_lowres[p]).resize(
                    (arc.output_width, arc.output_height), Image.BILINEAR
                )
            )
            cls_full = np.array(
                Image.fromarray(cls_lowres[p]).resize(
                    (arc.output_width, arc.output_height), Image.NEAREST
                )
            )
            frame_0 = _grayscale_plus_chroma_to_rgb(
                gray_full, cls_full, arc.chroma_rgb
            )
            frame_1 = _affine_warp_frame1_from_frame0(frame_0, pose[p])
            frame_0, frame_1 = _apply_pr98_l28_channel_balance_to_pair_uint8(
                frame_0, frame_1
            )
            for off, frame in ((0, frame_0), (1, frame_1)):
                fh.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                if debug_png_dir is not None:
                    Image.fromarray(frame).save(debug_png_dir / f"{2 * p + off}.png")
    return raw_path


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` (Catalog #146)."""
    if len(sys.argv) < 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0]
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI smoke
    sys.exit(main_cli())

# Reserved POSE_DIMS re-export to keep the module fully self-describing.
__all__ = ["POSE_DIMS", "inflate_one_video", "main_cli"]
