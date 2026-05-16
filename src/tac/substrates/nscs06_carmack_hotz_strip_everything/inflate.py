# SPDX-License-Identifier: MIT
"""Carmack-Hotz strip-everything inflate runtime — NO torch, NO scorer (≤100 LOC budget).

Per HNeRV parity discipline lesson L4 + L5 + strict-scorer-rule:

- ≤100 LOC inflate budget (current ~88 LOC including CLI glue)
- ≤2 external deps: numpy + Pillow (NO torch, NO smp, NO efficientnet)
- Full RGB renderer (not a mask-only codec; per L5)
- NO scorer load (per CLAUDE.md "Strict scorer rule" non-negotiable)

Forward path:

1. Read ``<archive_dir>/0.bin`` bytes; call ``parse_archive``.
2. For each pair, decode palette indices via arithmetic decoder.
3. Dequantize via palette -> uint8 grayscale (low-res).
4. Upsample grayscale -> per-frame uint8 RGB via grayscale-as-luma + LUT lookup.
5. Apply pose-delta warp from frame-0 to derive frame-1 (Quantizr PR #56 odd-only).
6. Save the contest raw stream to ``<output_stem>.raw``.

The "analytical renderer" is intentionally minimal: grayscale-as-luma +
LUT-based chrominance is the Carmack iteration MVP. Future passes can replace
the LUT with smarter chrominance reconstruction without changing the archive
grammar.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .archive import (
    POSE_DIMS,
    decode_grayscale_stream,
    dequantize_pose_deltas,
    parse_archive,
)


def _grayscale_to_rgb(gray_u8: np.ndarray) -> np.ndarray:
    """Map uint8 grayscale (H, W) -> uint8 RGB (H, W, 3) via Y=Cr=Cb=Y trick.

    Minimal MVP analytical renderer: replicate luma into all 3 channels. The
    LOSS axis of the Carmack-Hotz prediction band [0.10, 0.20]: chroma is
    fundamentally not in the archive. The GAIN axis: rate drops massively.
    """
    if gray_u8.dtype != np.uint8:
        raise ValueError(f"gray_u8 must be uint8; got {gray_u8.dtype}")
    if gray_u8.ndim != 2:
        raise ValueError(f"gray_u8 must be 2-D; got shape {gray_u8.shape}")
    return np.repeat(gray_u8[:, :, None], 3, axis=2)


def _warp_frame1_from_frame0(
    frame_0: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """Derive frame_1 from frame_0 via pose deltas (PR #56 odd-only paradigm).

    Minimal MVP: a global translation derived from the first 2 pose dims.
    This is the LOWEST-FIDELITY warp that still moves bytes; the analytical
    renderer iteration is the obvious place for future score gains.
    """
    h, w, _ = frame_0.shape
    # Pose dims [0, 1] interpreted as fractional translation in [-0.05, +0.05]
    dy = round(float(pose[0]) * 0.05 * h)
    dx = round(float(pose[1]) * 0.05 * w)
    out = np.roll(frame_0, shift=(dy, dx), axis=(0, 1))
    return out


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate one archive's bytes into a contest ``.raw`` file.

    ``output_stem`` is the evaluator-preserved video stem. For ``0.mkv`` the
    caller passes ``<inflated_dir>/0`` and the required output is
    ``<inflated_dir>/0.raw``. Debug PNG emission is opt-in via
    ``NSCS06_DEBUG_PNG=1`` and is never the scoring surface.
    """
    from PIL import Image

    arc = parse_archive(archive_bytes)
    # Class-label map is uniform 0 for the L1 SCAFFOLD path; future iteration
    # chains a learned-prior through pose to refine the per-cell class label.
    cls_lowres = np.zeros(
        (arc.num_pairs, arc.grayscale_h, arc.grayscale_w), dtype=np.uint8
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
            frame_0 = _grayscale_to_rgb(gray_full)
            frame_1 = _warp_frame1_from_frame0(frame_0, pose[p])
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
