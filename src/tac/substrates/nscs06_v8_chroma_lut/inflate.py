# SPDX-License-Identifier: MIT
"""nscs06 v8 chroma-LUT inflate runtime — CH08 v1/v2 + 6-DOF affine warp.

Per HNeRV parity discipline lessons L4 + L5 + strict-scorer-rule:

- ~120 LOC inflate budget (substrate_engineering exception per L7; the chroma
  LUT lookup + 6-DOF affine warp logic carry over from v7 sister inflate).
- <=2 external deps: numpy + Pillow (NO torch, NO smp, NO efficientnet).
- Full RGB renderer per L5 (chroma reconstructed from grayscale + class via
  per-(level, class) LUT).
- NO scorer load per CLAUDE.md "Strict scorer rule" non-negotiable.

v8 forward path:

1. Read ``<archive_dir>/0.bin`` bytes; call :func:`parse_archive`.
2. Resolve the chroma LUT:
   - v1: read inline ``(grayscale_levels, num_segnet_classes, 3)`` uint8 LUT.
   - v2: re-derive the LUT bytes deterministically from a 32-byte PCG64 seed
     via :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`.
3. For each pair:
   - Decode the per-cell ``(grayscale, class)`` map. v8 SCAFFOLD uses
     class=0 uniformly (parser-pass-through; v7 sister parses a separate
     CLS_STREAM). v8 evolves to consume the v7 CLS_STREAM at L1 promotion.
   - Upsample grayscale + class maps to output resolution.
   - Lookup per-pixel RGB via :func:`lookup_rgb_via_chroma_lut`.
   - Apply 6-DOF affine warp (sister of v7 ``_affine_warp_frame1_from_frame0``)
     to derive frame_1.
4. Write contest raw stream to ``<output_stem>.raw``.

Catalog #205 inline-device-fork is via canonical helper :func:`select_inflate_device`
(local helper mirroring the canonical contract at
``tac.substrates._shared.inflate_runtime.select_inflate_device``).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from .architecture import lookup_rgb_via_chroma_lut
from .archive import (
    CH08_SCHEMA_VERSION_INLINE_LUT,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM,
    POSE_DIMS,
    Nscs06V8Archive,
    parse_archive,
)


def select_inflate_device(env_var: str = "PACT_INFLATE_DEVICE") -> str:
    """Canonical inflate-device selector per Catalog #205.

    Sister of ``tac.substrates._shared.inflate_runtime.select_inflate_device``.
    Returns one of {``"auto"``, ``"cpu"``, ``"cuda"``}; ``"mps"`` is REFUSED
    per CLAUDE.md "MPS auth eval is NOISE" non-negotiable. The v8 inflate
    runtime is numpy/Pillow-only so device is informational; the env var is
    honored for forward-compat with future torch-using variants.
    """
    value = os.environ.get(env_var, "auto").strip().lower()
    if value == "mps":
        raise RuntimeError(
            f"{env_var}=mps REFUSED per CLAUDE.md 'MPS auth eval is NOISE' "
            "non-negotiable (Catalog #205 inflate-device discipline)"
        )
    if value not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(
            f"{env_var}={value!r} not in {{auto, cpu, cuda}}"
        )
    return value


def _resolve_chroma_lut(arc: Nscs06V8Archive) -> np.ndarray:
    """Resolve the chroma LUT from a parsed CH08 archive (v1 inline, v2 seed, or v3 seed+cls)."""
    if arc.schema_version == CH08_SCHEMA_VERSION_INLINE_LUT:
        if arc.chroma_lut is None:
            raise ValueError("v1 archive missing inline chroma_lut")
        return arc.chroma_lut
    if arc.schema_version in (
        CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
        CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM,
    ):
        if arc.chroma_seed is None:
            raise ValueError(
                f"v{arc.schema_version} archive missing chroma_seed"
            )
        # Re-derive the LUT bytes deterministically via the canonical helper.
        from tac.procedural_codebook_generator import derive_codebook_from_seed
        flat = derive_codebook_from_seed(
            seed_bytes=arc.chroma_seed,
            output_shape=(arc.grayscale_levels * arc.num_segnet_classes * 3,),
            dtype=np.uint8,
            generator_kind=arc.generator_kind or "pcg64",
        )
        return flat.reshape(arc.grayscale_levels, arc.num_segnet_classes, 3)
    raise ValueError(f"unsupported schema_version {arc.schema_version}")


def _affine_warp_frame1_from_frame0(
    frame_0: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """6-DOF affine warp from frame_0 -> frame_1 using all 6 PoseNet dims.

    Sister of v7 nscs06_carmack_hotz_strip_everything.inflate._affine_warp_frame1_from_frame0
    (cargo-cult #4 stays UNWOUND in v8 per HNeRV parity L5 + symposium commit
    4292c8ce2). pose = (tx, ty, tz, rx, ry, rz) in fractional-image units +
    small-angle rotations.
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


def _dequantize_pose_deltas(arc: Nscs06V8Archive) -> np.ndarray:
    """Inverse of pose quantization stored in the CH08 archive."""
    arr = np.frombuffer(arc.pose_bytes, dtype=np.uint8).copy().reshape(
        arc.num_pairs, POSE_DIMS
    )
    return (arr.astype(np.float32) - 128.0) / float(arc.pose_quant_scale)


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate one CH08 archive's bytes into a contest .raw file."""
    # Touch the device selector so it raises early on mps/invalid setting.
    _ = select_inflate_device()
    from PIL import Image

    arc = parse_archive(archive_bytes)
    chroma_lut = _resolve_chroma_lut(arc)
    pose = _dequantize_pose_deltas(arc)

    grayscale_lowres = np.frombuffer(arc.grayscale_bytes, dtype=np.uint8).copy().reshape(
        arc.num_pairs, arc.grayscale_h, arc.grayscale_w
    )

    raw_path = output_stem.with_suffix(".raw")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("wb") as fh:
        for p in range(arc.num_pairs):
            gray_full = np.array(
                Image.fromarray(grayscale_lowres[p]).resize(
                    (arc.output_width, arc.output_height), Image.BILINEAR
                )
            )
            # v8 cls_stream consumption (Catalog #233 L1→L2 promotion canonical
            # 4-gate unblocker per T3 council #1335 REVISION #2 Yousfi BLOCKER).
            # v3 archives carry per-cell uint8 class labels at low-res; upsample
            # NEAREST so the LUT lookup uses the canonical formula
            # ``RGB = chroma_lut[gray_full[y,x], cls_full[y,x], :]`` with
            # non-uniform cls_full. v1/v2 (legacy) fall back to cls=0 uniform
            # which is the cargo-cult #5 L0 SCAFFOLD behavior preserved for
            # backward compat.
            if arc.cls_lowres is not None:
                cls_full = np.array(
                    Image.fromarray(arc.cls_lowres[p]).resize(
                        (arc.output_width, arc.output_height), Image.NEAREST
                    ),
                    dtype=np.uint8,
                )
            else:
                cls_full = np.zeros_like(gray_full, dtype=np.uint8)
            frame_0 = lookup_rgb_via_chroma_lut(gray_full, cls_full, chroma_lut)
            frame_1 = _affine_warp_frame1_from_frame0(frame_0, pose[p])
            for frame in (frame_0, frame_1):
                fh.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
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


__all__ = [
    "POSE_DIMS",
    "inflate_one_video",
    "main_cli",
    "select_inflate_device",
]
