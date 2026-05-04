#!/usr/bin/env python3
"""Compress the renderer archive for rate savings.

Technique 3 from the eureka list: reduce archive.zip from ~184KB to ~100KB
for a free 0.055 score improvement (rate term: 25 * size / gt_size).

Components:
  1. FP4 quantize renderer.bin (150KB -> ~72KB)
  2. Re-encode masks at lower CRF or smaller resolution (79KB -> ~40KB)
  3. FP16 poses (14.4KB -> 7.2KB, or FP8 -> 3.6KB)
  4. Rebuild archive.zip with DEFLATE-9

Usage:
    PYTHONPATH=src python scripts/compress_archive.py \
        --archive submissions/robust_current/archive.zip \
        --output submissions/robust_current/archive_compressed.zip

    # Measure only (no write):
    PYTHONPATH=src python scripts/compress_archive.py \
        --archive submissions/robust_current/archive.zip --dry-run

    # Custom settings:
    PYTHONPATH=src python scripts/compress_archive.py \
        --archive submissions/robust_current/archive.zip \
        --renderer-bits 4 --pose-format fp16 --mask-crf 25
"""
from __future__ import annotations

import argparse
import io
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

from tac.submission_archive import (
    ORIGINAL_VIDEO_BYTES as GT_SIZE,
    deterministic_zip_directory,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compress renderer archive for rate savings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--archive", type=str, required=True,
                   help="Path to input archive.zip")
    p.add_argument("--output", type=str, default=None,
                   help="Output path (default: archive_compressed.zip alongside input)")
    p.add_argument("--renderer-bits", type=int, default=None, choices=[4, 6, 8],
                   help="Re-quantize renderer.bin to N bits (None=keep original)")
    p.add_argument("--pose-format", type=str, default="fp16",
                   choices=["fp32", "fp16", "fp8"],
                   help="Pose storage format")
    p.add_argument("--mask-crf", type=int, default=None,
                   help="Re-encode masks.mkv at this CRF (None=keep original)")
    p.add_argument("--dry-run", action="store_true",
                   help="Analyze only, do not write output")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose output")
    return p.parse_args()


def analyze_archive(archive_path: str) -> dict[str, int]:
    """Analyze contents of archive.zip."""
    result = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir():
                result[info.filename] = info.file_size
    return result


def requantize_poses(poses_data: bytes, output_format: str, pose_dim: int = 6) -> bytes:
    """Convert poses from whatever format to target format.

    Poses are stored as torch tensors. We load, convert, and re-save.
    """
    import torch

    # Try loading as torch tensor first
    buf = io.BytesIO(poses_data)
    try:
        poses = torch.load(buf, map_location="cpu", weights_only=True).float()
    except Exception:
        # Might be raw binary (fp16 buffer)
        poses = torch.frombuffer(bytearray(poses_data), dtype=torch.float16).reshape(-1, pose_dim).float()

    if output_format == "fp32":
        out_buf = io.BytesIO()
        torch.save(poses.float(), out_buf)
        return out_buf.getvalue()
    elif output_format == "fp16":
        out_buf = io.BytesIO()
        torch.save(poses.half(), out_buf)
        return out_buf.getvalue()
    elif output_format == "fp8":
        # FP8 via quantize-to-int8 with per-row scale
        scale = poses.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        quantized = (poses / scale * 127).round().clamp(-127, 127).to(torch.int8)
        # Pack: [n_pairs (4B)] [scale (n_pairs*4B fp32)] [quantized (n_pairs*6B int8)]
        n_pairs = poses.shape[0]
        buf = struct.pack("<I", n_pairs)
        buf += scale.squeeze(1).numpy().astype("float32").tobytes()
        buf += quantized.numpy().tobytes()
        return buf
    else:
        raise ValueError(f"Unknown pose format: {output_format}")


def compress_archive(args: argparse.Namespace) -> None:
    """Main compression pipeline."""
    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"ERROR: Archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)

    # Analyze original
    contents = analyze_archive(str(archive_path))
    original_size = archive_path.stat().st_size

    print(f"Original archive: {archive_path}")
    print(f"  Total size: {original_size:,} bytes")
    print(f"  Rate contribution: 25 * {original_size} / {GT_SIZE} = {25 * original_size / GT_SIZE:.4f}")
    print(f"\n  Contents:")
    for name, size in sorted(contents.items(), key=lambda x: -x[1]):
        print(f"    {name}: {size:,} bytes ({size / original_size * 100:.1f}%)")

    if args.dry_run:
        print("\n[dry-run] No output written.")
        return

    # Extract to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(str(archive_path), "r") as zf:
            zf.extractall(str(tmpdir))

        # Process each component
        changes = []

        # 1. Poses: requantize
        for pose_file in ["poses.pt", "optimized_poses.pt"]:
            pose_path = tmpdir / pose_file
            if pose_path.exists():
                old_size = pose_path.stat().st_size
                poses_data = pose_path.read_bytes()
                new_data = requantize_poses(poses_data, args.pose_format)
                pose_path.write_bytes(new_data)
                new_size = len(new_data)
                changes.append((pose_file, old_size, new_size))
                print(f"\n  {pose_file}: {old_size:,} -> {new_size:,} bytes "
                      f"({args.pose_format}, {(1 - new_size / old_size) * 100:.0f}% saved)")

        # 2. Masks: re-encode at different CRF
        if args.mask_crf is not None:
            mask_path = tmpdir / "masks.mkv"
            if mask_path.exists():
                old_size = mask_path.stat().st_size
                reencoded = _reencode_masks(mask_path, args.mask_crf)
                if reencoded is not None:
                    new_size = reencoded.stat().st_size
                    shutil.move(str(reencoded), str(mask_path))
                    changes.append(("masks.mkv", old_size, new_size))
                    print(f"\n  masks.mkv: {old_size:,} -> {new_size:,} bytes "
                          f"(CRF {args.mask_crf}, {(1 - new_size / old_size) * 100:.0f}% saved)")

        # 3. Renderer.bin: re-quantize at lower bits
        if args.renderer_bits is not None:
            renderer_path = tmpdir / "renderer.bin"
            if renderer_path.exists():
                old_size = renderer_path.stat().st_size
                new_data = _requantize_renderer(renderer_path, args.renderer_bits)
                if new_data is not None:
                    renderer_path.write_bytes(new_data)
                    new_size = len(new_data)
                    changes.append(("renderer.bin", old_size, new_size))
                    print(f"\n  renderer.bin: {old_size:,} -> {new_size:,} bytes "
                          f"({args.renderer_bits}-bit, {(1 - new_size / old_size) * 100:.0f}% saved)")

        # 4. Rebuild archive with maximum DEFLATE compression
        output_path = Path(args.output) if args.output else (
            archive_path.parent / f"{archive_path.stem}_compressed.zip"
        )

        deterministic_zip_directory(
            tmpdir,
            output_path,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )

        new_archive_size = output_path.stat().st_size
        print(f"\n{'=' * 60}")
        print(f"RESULTS:")
        print(f"  Original: {original_size:,} bytes (rate: {25 * original_size / GT_SIZE:.4f})")
        print(f"  Compressed: {new_archive_size:,} bytes (rate: {25 * new_archive_size / GT_SIZE:.4f})")
        print(f"  Saved: {original_size - new_archive_size:,} bytes "
              f"({(1 - new_archive_size / original_size) * 100:.1f}%)")
        print(f"  Rate improvement: {25 * (original_size - new_archive_size) / GT_SIZE:.4f}")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}")


def _reencode_masks(mask_path: Path, crf: int) -> Path | None:
    """Re-encode mask video at a different CRF."""
    import subprocess

    output = mask_path.parent / "masks_reencoded.mkv"
    cmd = [
        "ffmpeg", "-y", "-i", str(mask_path),
        "-c:v", "libsvtav1", "-crf", str(crf),
        "-pix_fmt", "gray",
        "-v", "error",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"  WARNING: mask re-encoding failed: {result.stderr.decode()}", file=sys.stderr)
        return None
    return output


def _requantize_renderer(renderer_path: Path, bits: int) -> bytes | None:
    """Re-quantize renderer.bin at lower bit depth.

    This requires loading the model, re-exporting at lower bits.
    Only works with ASYM format.
    """
    raw = renderer_path.read_bytes()
    if raw[:4] != b"ASYM":
        print(f"  WARNING: renderer.bin is not ASYM format, skipping re-quantization",
              file=sys.stderr)
        return None

    try:
        from tac.renderer_export import load_asymmetric_checkpoint, export_asymmetric_checkpoint

        # Load the model from the binary
        model = load_asymmetric_checkpoint(raw, device="cpu")

        # Re-export at lower bits
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tmp_path = Path(tf.name)

        export_asymmetric_checkpoint(model, tmp_path, default_bits=bits)
        result = tmp_path.read_bytes()
        tmp_path.unlink()
        return result
    except ImportError as e:
        print(f"  WARNING: Cannot re-quantize renderer (missing tac.renderer_export): {e}",
              file=sys.stderr)
        return None
    except Exception as e:
        print(f"  WARNING: Renderer re-quantization failed: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    args = parse_args()
    compress_archive(args)
