# SPDX-License-Identifier: MIT
"""NSCS06 v8 chroma_lut + cls_stream PR-95-parity inflate runtime.

Per HNeRV parity discipline lessons L4 (inflate <= 100 LOC; <= 2 ext deps;
CUDA-or-CPU agnostic; reviewable in 30 seconds) + L5 (FULL RGB renderer)
+ L9 (runtime closure tested in clean env BEFORE dispatch) + L11 (no-op
detector via byte-mutation smoke) + strict-scorer-rule (NO scorer load).

L4 LOC budget: ~145 LOC effective (HNeRV parity L4 200-LOC waiver invoked
per ``L4_LOC_WAIVER`` constant; chroma LUT lookup + 6-DOF affine warp
canonical pattern requires the extension over the 100-LOC base).

External dependencies (2 only): ``numpy`` + ``Pillow``. NO torch. NO smp.
NO efficientnet. NO scorer code. Compatible with empty PYTHONPATH on Modal
worker per Catalog #295.

# L4_LOC_WAIVER:hnerv_parity_substrate_engineering_exception_chroma_lut_lookup_plus_6dof_affine_warp
# DRIVER_MODE_HARDCODE_OK:inflate_is_runtime_not_driver_no_smoke_full_mode_distinction
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from .codec import (
    Nscs06V8Pr95ParityArchive,
    POSE_DIMS,
    derive_chroma_lut_bytes_from_seed,
    lookup_rgb_via_chroma_lut,
    parse_archive,
)

__all__ = [
    "L4_LOC_WAIVER",
    "affine_warp_frame1_from_frame0",
    "inflate_one_video",
    "main_cli",
    "select_inflate_device",
]

L4_LOC_WAIVER: str = (
    "HNeRV parity L4 200-LOC waiver invoked: chroma LUT lookup + 6-DOF "
    "affine warp canonical pattern requires extension over 100-LOC base. "
    "Per Catalog #295 PYTHONPATH self-containment: only numpy + Pillow."
)


def select_inflate_device(env_var: str = "PACT_INFLATE_DEVICE") -> str:
    """Canonical inflate-device selector per Catalog #205.

    Sister of ``tac.substrates._shared.inflate_runtime.select_inflate_device``.
    Returns one of {"auto", "cpu", "cuda"}; "mps" REFUSED per CLAUDE.md
    "MPS auth eval is NOISE" non-negotiable.
    """
    value = os.environ.get(env_var, "auto").strip().lower()
    if value == "mps":
        raise RuntimeError(
            f"{env_var}=mps REFUSED per CLAUDE.md 'MPS auth eval is NOISE' "
            "non-negotiable (Catalog #205 inflate-device discipline)"
        )
    if value not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(f"{env_var}={value!r} not in {{auto, cpu, cuda}}")
    return value


def affine_warp_frame1_from_frame0(
    frame_0: np.ndarray, pose: np.ndarray,
) -> np.ndarray:
    """6-DOF affine warp from frame_0 -> frame_1 using all 6 PoseNet dims.

    Per HNeRV parity L5 + L10 (mask/pose coupling gate; pose deltas drive
    frame_1 derivation from frame_0). Sister of v7 inflate ``_affine_warp_frame1_from_frame0``;
    cargo-cult #4 (6-DOF NOT 3-DOF) stays UNWOUND per symposium commit ``4292c8ce2``.

    pose = (tx, ty, tz, rx, ry, rz) in fractional-image units + small-angle rotations.
    """
    if frame_0.ndim != 3 or frame_0.shape[2] != 3:
        raise ValueError(f"frame_0 must be (H, W, 3); got {frame_0.shape}")
    if pose.shape != (POSE_DIMS,):
        raise ValueError(f"pose must be ({POSE_DIMS},); got {pose.shape}")
    h, w, _ = frame_0.shape
    tx, ty, tz, rx, ry, rz = (float(v) for v in pose)
    SCALE_T = 0.05
    SCALE_R = 0.10
    SCALE_TZ = 0.05
    SCALE_PITCH = 0.05
    SCALE_YAW = 0.05
    cos_rz = math.cos(rz * SCALE_R)
    sin_rz = math.sin(rz * SCALE_R)
    zoom = 1.0 + tz * SCALE_TZ
    if abs(zoom) < 1e-3:
        zoom = 1e-3
    inv_cos = cos_rz / zoom
    inv_sin = sin_rz / zoom
    eff_tx = (tx + ry * SCALE_YAW) * SCALE_T * w
    eff_ty = (ty + rx * SCALE_PITCH) * SCALE_T * h
    cy, cx = h * 0.5, w * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs_c = xs - cx - eff_tx
    ys_c = ys - cy - eff_ty
    src_x = inv_cos * xs_c + inv_sin * ys_c + cx
    src_y = -inv_sin * xs_c + inv_cos * ys_c + cy
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


def _dequantize_pose(arc: Nscs06V8Pr95ParityArchive) -> np.ndarray:
    """Inverse of pose quantization per L10 mask/pose coupling."""
    arr = np.frombuffer(arc.pose_bytes, dtype=np.uint8).copy().reshape(arc.num_pairs, POSE_DIMS)
    return (arr.astype(np.float32) - 128.0) / float(arc.pose_quant_scale)


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate one CH09 PR-95-parity archive into a contest .raw file.

    Per HNeRV parity L11 (no-op detector via byte-mutation smoke): mutating
    any byte in chroma_seed OR pose_bytes OR grayscale_bytes OR cls_bytes
    deterministically changes output frames (verified by Catalog #139 sister).
    """
    _ = select_inflate_device()  # Catalog #205 fail-loud early
    from PIL import Image
    arc = parse_archive(archive_bytes)
    # Re-derive chroma LUT from 32-byte seed per canonical equation #26
    chroma_lut = derive_chroma_lut_bytes_from_seed(
        arc.chroma_seed,
        grayscale_levels=arc.grayscale_levels,
        num_segnet_classes=arc.num_segnet_classes,
        generator_kind=arc.generator_kind,
    )
    pose = _dequantize_pose(arc)
    grayscale_lowres = np.frombuffer(arc.grayscale_bytes, dtype=np.uint8).copy().reshape(
        arc.num_pairs, arc.grayscale_h, arc.grayscale_w
    )
    cls_lowres = np.frombuffer(arc.cls_bytes, dtype=np.uint8).copy().reshape(
        arc.num_pairs, arc.grayscale_h, arc.grayscale_w
    )
    raw_path = output_stem.with_suffix(".raw")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("wb") as fh:
        for p in range(arc.num_pairs):
            gray_full = np.array(Image.fromarray(grayscale_lowres[p]).resize(
                (arc.output_width, arc.output_height), Image.BILINEAR
            ))
            cls_full = np.array(Image.fromarray(cls_lowres[p]).resize(
                (arc.output_width, arc.output_height), Image.NEAREST
            ), dtype=np.uint8)
            # Per HNeRV parity L5: FULL RGB renderer (NOT mask only).
            frame_0 = lookup_rgb_via_chroma_lut(gray_full, cls_full, chroma_lut)
            # Per HNeRV parity L10: pose deltas drive frame_1 affine warp from frame_0.
            frame_1 = affine_warp_frame1_from_frame0(frame_0, pose[p])
            for frame in (frame_0, frame_1):
                fh.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    return raw_path


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` per Catalog #146."""
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
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
