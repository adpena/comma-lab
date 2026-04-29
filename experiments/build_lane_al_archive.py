#!/usr/bin/env python3
"""Lane AL archive builder — Analog Latent canvas → grayscale.mkv.

Takes ``optimized_grayscale.npy`` (uint8, (N, H, W)) produced by
``experiments/optimize_grayscale_canvas.py`` and packs it into a
contest-ready archive with the SAME layout as Lane MM:

    archive_lane_al.zip
        ├── renderer.bin          (from anchor archive — Lane A)
        ├── grayscale.mkv         (ffmpeg AV1 monochrome from optimized npy)
        └── optimized_poses.pt    (from anchor archive)

Inflate dispatch reuses Lane MM's ``PYTHON_INFLATE=renderer_grayscale``
arm — no inflate-side changes required. The Gaussian-LUT decoder
already handles arbitrary uint8 values via the soft-projection +
nearest-neighbour argmax path.

Usage:
    python experiments/build_lane_al_archive.py \\
        --anchor-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --grayscale-npy experiments/results/lane_al/iter_0/optimized_grayscale.npy \\
        --output experiments/results/lane_al/iter_0/archive_lane_al.zip
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _encode_grayscale_mkv(
    pixels_uint8: np.ndarray,
    output_path: Path,
    crf: int = 50,
    fps: int = 20,
) -> int:
    """Encode (N, H, W) uint8 grayscale as an AV1 monochrome stream.

    Uses ``-pix_fmt gray`` for both input and output so AV1 skips chroma
    planes entirely. Returns the encoded byte size.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pixels_uint8.dtype != np.uint8:
        raise ValueError(
            f"pixels must be uint8; got {pixels_uint8.dtype}"
        )
    if pixels_uint8.ndim != 3:
        raise ValueError(
            f"pixels must be (N, H, W); got {pixels_uint8.shape}"
        )
    n, h, w = pixels_uint8.shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "gray",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libsvtav1",
        "-crf", str(crf), "-preset", "6",
        "-svtav1-params", "enable-restoration=0:enable-cdef=0",
        "-pix_fmt", "gray", "-an",
        str(output_path),
    ]
    proc = subprocess.run(
        cmd, input=pixels_uint8.tobytes(),
        capture_output=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg AV1 encode failed (rc={proc.returncode}):\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return output_path.stat().st_size


def build_lane_al_archive(
    anchor_archive: Path,
    grayscale_npy: Path,
    output: Path,
    crf: int = 50,
) -> dict:
    """Build the Lane AL archive from Lane A renderer/poses + optimized
    grayscale canvas.
    """
    anchor_archive = Path(anchor_archive)
    grayscale_npy = Path(grayscale_npy)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pixels = np.load(str(grayscale_npy))
    if pixels.dtype != np.uint8:
        raise ValueError(
            f"optimized grayscale must be uint8; got {pixels.dtype}"
        )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(anchor_archive, "r") as zf:
            zf.extractall(td_path)

        renderer = td_path / "renderer.bin"
        if not renderer.exists():
            raise FileNotFoundError(
                f"anchor archive missing renderer.bin: {anchor_archive}"
            )

        grayscale_mkv = td_path / "grayscale.mkv"
        gray_bytes = _encode_grayscale_mkv(
            pixels, grayscale_mkv, crf=crf,
        )

        # Deterministic zip (codex R5-r6 #5 archive-builder rule).
        def _det_write(zout: zipfile.ZipFile, src: Path, arcname: str) -> None:
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zout.writestr(info, src.read_bytes(), compresslevel=9)

        with zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED, compresslevel=9,
        ) as zout:
            _det_write(zout, renderer, "renderer.bin")
            _det_write(zout, grayscale_mkv, "grayscale.mkv")
            for poses_name in ("optimized_poses.pt", "poses.pt"):
                p = td_path / poses_name
                if p.exists():
                    _det_write(zout, p, poses_name)
                    break

    return {
        "grayscale_bytes": gray_bytes,
        "output_bytes": output.stat().st_size,
        "n_frames": int(pixels.shape[0]),
        "frame_h": int(pixels.shape[1]),
        "frame_w": int(pixels.shape[2]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-archive", type=Path, required=True,
                   help="Lane A archive.zip with renderer.bin + poses.")
    p.add_argument("--grayscale-npy", type=Path, required=True,
                   help="optimized_grayscale.npy from optimize_grayscale_canvas.py.")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the Lane AL archive.zip.")
    p.add_argument("--crf", type=int, default=50,
                   help="AV1 CRF (default: 50, matches Lane A/MM).")
    args = p.parse_args(argv)

    info = build_lane_al_archive(
        anchor_archive=args.anchor_archive,
        grayscale_npy=args.grayscale_npy,
        output=args.output,
        crf=args.crf,
    )
    print(
        f"[lane-al] grayscale.mkv={info['grayscale_bytes']:,}B"
        f"  output_archive={info['output_bytes']:,}B"
        f"  n_frames={info['n_frames']}  "
        f"frame={info['frame_h']}x{info['frame_w']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
