#!/usr/bin/env python3
"""Lane MM archive builder — Selfcomp grayscale-LUT mask re-encoding.

Takes the canonical Lane A archive (renderer.bin + masks.mkv +
optimized_poses.pt) and produces a new archive where masks.mkv has been
re-encoded as grayscale.mkv using the Selfcomp class targets
[0, 255, 64, 192, 128] (instead of the legacy linear ramp [0, 63, 127,
191, 255]).

The renderer + poses are unchanged. The hypothesis (per Lane MM
provenance): grayscale.mkv ~50% smaller than masks.mkv at the same
quality due to (1) AV1 monochrome skipping chroma planes and (2) the
Selfcomp targets being 64-pixel-spaced (3 gaps) plus 63-pixel-spaced (1 gap) so AV1's quantizer can absorb
~10-15 levels of noise without flipping the nearest-neighbour class.

Outputs:
    archive_lane_mm.zip with renderer.bin + grayscale.mkv +
                                    optimized_poses.pt.

Usage:
    python experiments/build_lane_mm_archive.py \\
        --anchor-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --output experiments/results/lane_mm/archive_lane_mm.zip
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.mask_grayscale_lut import CLASS_TO_GRAY  # noqa: E402
from tac.submission_archive import safe_extract_zip  # noqa: E402


def _decode_legacy_masks_mkv(mkv_path: Path) -> torch.Tensor:
    """Decode the Lane A masks.mkv -> (N, H, W) int64 class ids.

    Lane A uses encoding ``pixel_value = class * (255 // 4)`` so values
    map [0, 63, 127, 191, 255]. We invert with rounding division.
    """
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0",
            str(mkv_path),
        ],
        capture_output=True, text=True, timeout=30, check=True,
    )
    parts = probe.stdout.strip().split(",")
    src_w, src_h = int(parts[0]), int(parts[1])

    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(mkv_path),
            "-f", "rawvideo", "-pix_fmt", "gray", "-v", "error",
            "pipe:1",
        ],
        capture_output=True, timeout=300, check=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = src_h * src_w
    n = len(raw) // frame_size
    if len(raw) % frame_size != 0:
        raise ValueError(
            f"decoded gray data {len(raw)} not divisible by {src_h}x{src_w}={frame_size}"
        )
    pixels = raw.reshape(n, src_h, src_w)
    # Lane A encoding: pixel = class * 63. Decode = round(pixel / 63).
    scale = 255 // 4
    classes = np.round(pixels.astype(np.float32) / scale).astype(np.int64)
    classes = np.clip(classes, 0, 4)
    return torch.from_numpy(classes)


def _encode_grayscale_mkv(
    classes: torch.Tensor,
    output_path: Path,
    crf: int = 50,
    fps: int = 20,
) -> int:
    """Encode (N, H, W) int64 class ids as a Selfcomp-mapped grayscale AV1 stream.

    Uses CLASS_TO_GRAY = {0:0, 1:255, 2:64, 3:192, 4:128} (Selfcomp inflate.py
    CLASS_TARGETS). Returns the encoded byte size.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n, h, w = classes.shape
    targets = torch.tensor(
        [CLASS_TO_GRAY[c] for c in range(5)], dtype=torch.uint8
    )
    pixels = targets[classes].numpy()  # (N, H, W) uint8

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
        cmd, input=pixels.tobytes(), capture_output=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg AV1 encode failed (rc={proc.returncode}):\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return output_path.stat().st_size


def build_lane_mm_archive(
    anchor_archive: Path,
    output: Path,
    crf: int = 50,
    keep_legacy_masks: bool = False,
    sigma: float = 15.0,
) -> dict:
    """Re-encode the anchor archive's masks.mkv into grayscale.mkv (Lane MM).

    Returns a dict with byte sizes (anchor_masks_bytes, grayscale_bytes,
    output_bytes) for provenance logging.
    """
    anchor_archive = Path(anchor_archive)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        safe_extract_zip(anchor_archive, td_path)

        masks_mkv = td_path / "masks.mkv"
        if not masks_mkv.exists():
            raise FileNotFoundError(
                f"anchor archive missing masks.mkv: {anchor_archive}"
            )
        anchor_masks_bytes = masks_mkv.stat().st_size

        classes = _decode_legacy_masks_mkv(masks_mkv)
        grayscale = td_path / "grayscale.mkv"
        gray_bytes = _encode_grayscale_mkv(
            classes, grayscale, crf=crf,
        )

        # Build the new archive: renderer.bin + grayscale.mkv (+ poses if present).
        renderer = td_path / "renderer.bin"
        if not renderer.exists():
            raise FileNotFoundError(f"anchor archive missing renderer.bin")

        # Deterministic zip: ZipInfo with fixed mtime so byte hash is stable
        # across machines (Codex R5-r6 #5; check_archive_builders_use_deterministic_zip).
        def _det_write(zout: zipfile.ZipFile, src: Path, arcname: str) -> None:
            info = zipfile.ZipInfo(arcname)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zout.writestr(info, src.read_bytes(), compresslevel=9)

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
            _det_write(zout, renderer, "renderer.bin")
            _det_write(zout, grayscale, "grayscale.mkv")
            if keep_legacy_masks:
                _det_write(zout, masks_mkv, "masks.mkv")
            for poses_name in ("optimized_poses.pt", "poses.pt"):
                p = td_path / poses_name
                if p.exists():
                    _det_write(zout, p, poses_name)
                    break

    output_bytes = output.stat().st_size
    return {
        "anchor_masks_bytes": anchor_masks_bytes,
        "grayscale_bytes": gray_bytes,
        "output_bytes": output_bytes,
        "rate_delta_bytes": gray_bytes - anchor_masks_bytes,
        "lane_mm_sigma": float(sigma),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--anchor-archive", type=Path, required=True,
        help="Path to a Lane A archive (renderer.bin + masks.mkv + poses).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the Lane MM archive.zip.",
    )
    p.add_argument(
        "--crf", type=int, default=50,
        help="AV1 CRF for grayscale.mkv (default: 50, same as Lane A).",
    )
    p.add_argument(
        "--keep-legacy-masks", action="store_true",
        help="Also include masks.mkv in the output for A/B testing.",
    )
    p.add_argument(
        "--sigma", type=float, default=15.0,
        help="Inflate-time Gaussian-LUT sigma (default: 15.0). Recorded in "
             "the build manifest for provenance; the inflate path reads "
             "LANE_MM_SIGMA env var (set by lane scripts) at decode time.",
    )
    args = p.parse_args(argv)

    info = build_lane_mm_archive(
        anchor_archive=args.anchor_archive,
        output=args.output,
        crf=args.crf,
        keep_legacy_masks=args.keep_legacy_masks,
        sigma=args.sigma,
    )
    print(
        f"[lane-mm] anchor masks.mkv={info['anchor_masks_bytes']:,}B"
        f"  grayscale.mkv={info['grayscale_bytes']:,}B"
        f"  delta={info['rate_delta_bytes']:+,}B"
        f"  output_archive={info['output_bytes']:,}B"
        f"  sigma={info['lane_mm_sigma']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
