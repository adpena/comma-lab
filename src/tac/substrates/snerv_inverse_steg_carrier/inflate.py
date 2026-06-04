# SPDX-License-Identifier: MIT
"""SNeRV SNAR1 contest inflate runtime.

Consumes one SNAR member from ``archive_dir`` (preferring the byte-minimal
``x`` convention, with ``0.bin`` fallback), decodes receiver-visible frames,
and writes one contest raw RGB file. This module is scorer-free and torch-free;
the final camera-size lowering is a small NumPy bilinear renderer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.archive import decode_snerv_archive_frames

CAMERA_HW: tuple[int, int] = (874, 1164)


class SnervInflateError(ValueError):
    """Raised when SNeRV inflate input/output contracts are violated."""


def snerv_frames_to_raw_bytes(frames: np.ndarray) -> bytes:
    """Return contest raw bytes for ``(pairs, 2, 3, H, W)`` byte-scale frames."""

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[1] != 2 or arr.shape[2] != 3:
        raise SnervInflateError(
            "expected frames shaped (n_pairs, 2, 3, H, W); "
            f"got {tuple(arr.shape)}"
        )
    flat = arr.reshape(-1, 3, int(arr.shape[-2]), int(arr.shape[-1]))
    resized = _resize_nchw_bilinear(flat, out_hw=CAMERA_HW)
    u8 = (
        np.rint(np.clip(resized, 0.0, 255.0))
        .astype(np.uint8)
        .transpose(0, 2, 3, 1)
    )
    return u8.tobytes(order="C")


def write_snerv_frames_to_raw(fh: BinaryIO, frames: np.ndarray) -> int:
    """Append SNeRV frames to an open contest raw file and return frame count."""

    arr = np.asarray(frames)
    fh.write(snerv_frames_to_raw_bytes(arr))
    return int(arr.shape[0]) * 2


def inflate_one_video(
    archive_bytes: bytes,
    output_path: Path,
    *,
    device: str | None = None,
) -> None:
    """Decode one SNAR1 archive packet into one contest ``.raw`` file.

    ``device`` is accepted for compatibility with generated substrate wrappers;
    SNeRV's current receiver implementation is NumPy-only.
    """

    del device
    frames = decode_snerv_archive_frames(bytes(archive_bytes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        write_snerv_frames_to_raw(fh, frames)


def main_cli() -> int:
    if len(sys.argv) != 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = _read_archive_bytes(archive_dir)
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        inflate_one_video(archive_bytes, _raw_output_path(output_dir, line))
    return 0


def _read_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single charged archive member, accepting ``x`` or ``0.bin``.

    The upstream evaluator extracts ``archive.zip`` before calling
    ``inflate.sh``; it does not prescribe the member name. ``x`` saves charged
    ZIP filename bytes versus ``0.bin``. Ambiguity fails closed so a stale
    sidecar cannot silently choose the wrong packet.
    """

    candidates = [archive_dir / "x", archive_dir / "0.bin"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        names = ", ".join(path.name for path in present) or "none"
        raise SnervInflateError(
            "expected exactly one SNeRV archive member named 'x' or '0.bin'; "
            f"found {names}"
        )
    return present[0].read_bytes()


def _resize_nchw_bilinear(frames: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise SnervInflateError(f"expected NCHW RGB frames; got {tuple(arr.shape)}")
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    in_h, in_w = int(arr.shape[2]), int(arr.shape[3])
    if (in_h, in_w) == (out_h, out_w):
        return arr
    y0, y1, wy = _bilinear_axis(in_h, out_h)
    x0, x1, wx = _bilinear_axis(in_w, out_w)
    out = np.empty((arr.shape[0], 3, out_h, out_w), dtype=np.float32)
    wx_row = wx.reshape(1, out_w)
    for frame_idx in range(arr.shape[0]):
        for channel_idx in range(3):
            plane = arr[frame_idx, channel_idx]
            rows = (1.0 - wy[:, None]) * plane[y0, :] + wy[:, None] * plane[y1, :]
            out[frame_idx, channel_idx] = (
                (1.0 - wx_row) * rows[:, x0] + wx_row * rows[:, x1]
            )
    return out


def _bilinear_axis(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if in_size < 1 or out_size < 1:
        raise SnervInflateError(
            f"invalid resize axis in_size={in_size} out_size={out_size}"
        )
    if out_size == 1:
        coords = np.array([0.0], dtype=np.float32)
    else:
        scale = float(in_size) / float(out_size)
        coords = (np.arange(out_size, dtype=np.float32) + 0.5) * scale - 0.5
        coords = np.clip(coords, 0.0, float(in_size - 1))
    lo = np.floor(coords).astype(np.int64)
    hi = np.minimum(lo + 1, in_size - 1)
    weight = (coords - lo.astype(np.float32)).astype(np.float32)
    return lo, hi, weight


def _raw_output_path(output_dir: Path, video_name: str) -> Path:
    raw = str(video_name).replace("\\", "/").strip()
    rel = Path(raw)
    if (
        not raw
        or "//" in raw
        or rel.is_absolute()
        or any(part in {"", ".."} for part in rel.parts)
    ):
        raise SnervInflateError(f"unsafe file_list video name: {video_name!r}")
    root = output_dir.resolve(strict=False)
    target = (output_dir / rel.with_suffix(".raw")).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SnervInflateError(
            f"file_list video name escapes output directory: {video_name!r}"
        ) from exc
    return target


__all__ = [
    "SnervInflateError",
    "_read_archive_bytes",
    "inflate_one_video",
    "main_cli",
    "snerv_frames_to_raw_bytes",
    "write_snerv_frames_to_raw",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_cli())
