# SPDX-License-Identifier: MIT
"""Archive pruning: strip unnecessary bytes from archive.zip.

Every byte saved reduces the rate term in the scoring formula:
    rate = archive_size / (n_frames * width * height * 3)
This directly improves the final score with ZERO quality loss.

Targets:
  - ZIP metadata: file comments, extra fields, padding
  - MKV container: tags, chapters, cues, seekhead overhead
  - Postfilter checkpoint: remove training-only metadata

Usage::

    from tac.archive_optimizer import optimize_archive, estimate_rate_savings
    savings = optimize_archive("submissions/robust_current/archive.zip")
    print(estimate_rate_savings(original_size=820_000, optimized_size=savings["optimized_bytes"]))
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from tac.submission_archive import (
    safe_extract_zip,
    validate_zip_member_infos,
    write_deterministic_zip_member,
)


def _strip_zip_metadata(archive_path: Path, output_path: Path) -> int:
    """Repack a ZIP file with minimal metadata.

    Strips: file comments, archive comment, extra fields, uses STORED
    for already-compressed content (MKV, .pt files are already compressed).
    Uses maximum deflate compression for any uncompressed entries.

    Returns the new archive size in bytes.
    """
    with (
        zipfile.ZipFile(archive_path, "r") as zin,
        zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zout,
    ):
        infos = zin.infolist()
        validate_zip_member_infos(infos)
        for info in infos:
            data = zin.read(info.filename)

            # Try both STORED and DEFLATED, keep whichever is smaller
            # (already-compressed data like MKV may be smaller with STORED)
            buf_deflated = io.BytesIO()
            with zipfile.ZipFile(buf_deflated, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as ztmp:
                ztmp.writestr(info.filename, data)
            deflated_size = buf_deflated.tell()

            buf_stored = io.BytesIO()
            with zipfile.ZipFile(buf_stored, "w", compression=zipfile.ZIP_STORED) as ztmp:
                ztmp.writestr(info.filename, data)
            stored_size = buf_stored.tell()

            compress_type = zipfile.ZIP_STORED if stored_size <= deflated_size else zipfile.ZIP_DEFLATED

            write_deterministic_zip_member(
                zout,
                info.filename,
                data,
                compress_type=compress_type,
                compresslevel=9 if compress_type != zipfile.ZIP_STORED else None,
            )

    return output_path.stat().st_size


def _strip_mkv_container(mkv_path: Path, output_path: Path) -> int:
    """Strip MKV container overhead using ffmpeg remux (strips metadata/tags).

    Returns the new file size in bytes.
    """
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-i", str(mkv_path),
        "-c", "copy",
        "-map_metadata", "-1",  # strip all metadata
        "-fflags", "+bitexact",  # deterministic output
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        # If ffmpeg fails, just copy the original
        shutil.copy2(mkv_path, output_path)
    return output_path.stat().st_size


def _strip_checkpoint_metadata(pt_path: Path, output_path: Path) -> int:
    """Strip training-only metadata from a .pt checkpoint.

    Removes __meta__ fields that are not needed at inference time
    (training config, loss history, etc.) while keeping architecture
    metadata needed to reconstruct the model.

    Returns the new file size in bytes.
    """
    import torch

    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if "__meta__" in state:
        meta = state["__meta__"]
        # Keep only essential fields for model reconstruction
        essential_keys = {"variant", "hidden", "kernel", "version"}
        cleaned_meta = {k: v for k, v in meta.items() if k in essential_keys}
        state["__meta__"] = cleaned_meta

    torch.save(state, output_path)
    return output_path.stat().st_size


def optimize_archive(archive_path: str | Path) -> dict:
    """Optimize an archive.zip by stripping all unnecessary bytes.

    Process:
      1. Extract archive contents
      2. Strip MKV container metadata from video files
      3. Strip checkpoint metadata from .pt files
      4. Repack with minimal ZIP metadata and optimal compression

    Args:
        archive_path: path to the archive.zip to optimize

    Returns:
        dict with keys:
            original_bytes: original archive size
            optimized_bytes: optimized archive size
            savings_bytes: bytes saved
            savings_pct: percentage reduction
            per_file: dict of per-file size changes
    """
    archive_path = Path(archive_path)
    original_bytes = archive_path.stat().st_size

    with tempfile.TemporaryDirectory(prefix="archive_opt_") as tmpdir:
        tmpdir = Path(tmpdir)
        extract_dir = tmpdir / "extracted"
        optimized_dir = tmpdir / "optimized"
        extract_dir.mkdir()
        optimized_dir.mkdir()

        # Extract through the canonical safe extractor; raw extractall is a
        # contest-custody bug class because it can hide duplicate/zip-slip
        # members until much later in the pipeline.
        safe_extract_zip(archive_path, extract_dir)

        per_file: dict[str, dict] = {}

        # Process each file
        for fpath in extract_dir.rglob("*"):
            if fpath.is_dir():
                continue
            rel = fpath.relative_to(extract_dir)
            out_path = optimized_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            orig_size = fpath.stat().st_size
            suffix = fpath.suffix.lower()

            if suffix == ".mkv":
                new_size = _strip_mkv_container(fpath, out_path)
            elif suffix == ".pt":
                try:
                    new_size = _strip_checkpoint_metadata(fpath, out_path)
                except Exception:
                    shutil.copy2(fpath, out_path)
                    new_size = orig_size
            else:
                shutil.copy2(fpath, out_path)
                new_size = orig_size

            per_file[str(rel)] = {
                "original": orig_size,
                "optimized": new_size,
                "saved": orig_size - new_size,
            }

        # Repack with minimal ZIP metadata
        optimized_archive = tmpdir / "archive_optimized.zip"
        with zipfile.ZipFile(optimized_archive, "w", compression=zipfile.ZIP_STORED) as zout:
            for fpath in sorted(optimized_dir.rglob("*")):
                if fpath.is_dir():
                    continue
                rel = fpath.relative_to(optimized_dir).as_posix()
                write_deterministic_zip_member(
                    zout,
                    rel,
                    fpath.read_bytes(),
                    compress_type=zipfile.ZIP_STORED,
                    compresslevel=None,
                )

        optimized_bytes = optimized_archive.stat().st_size

        # If optimization actually saved space, replace the original
        if optimized_bytes < original_bytes:
            shutil.copy2(optimized_archive, archive_path)

    savings_bytes = original_bytes - optimized_bytes
    savings_pct = (savings_bytes / original_bytes * 100) if original_bytes > 0 else 0.0

    print(f"Archive optimization: {original_bytes:,}B -> {optimized_bytes:,}B "
          f"({savings_pct:.1f}% saved, {savings_bytes:,}B)")
    for fname, info in per_file.items():
        if info["saved"] > 0:
            print(f"  {fname}: {info['original']:,}B -> {info['optimized']:,}B "
                  f"(-{info['saved']:,}B)")

    return {
        "original_bytes": original_bytes,
        "optimized_bytes": optimized_bytes,
        "savings_bytes": savings_bytes,
        "savings_pct": savings_pct,
        "per_file": per_file,
    }


def estimate_rate_savings(
    current_size: int,
    optimized_size: int,
    n_frames: int = 1200,
    width: int = 1164,
    height: int = 874,
) -> dict:
    """Estimate score impact from archive size reduction.

    The rate term in the scoring formula is:
        rate = archive_bytes / (n_frames * width * height * channels)

    Args:
        current_size: current archive size in bytes
        optimized_size: optimized archive size in bytes
        n_frames: number of frames (default 1200 for 0.mkv)
        width: frame width (default 1164)
        height: frame height (default 874)

    Returns:
        dict with rate_before, rate_after, rate_delta, score_impact_estimate
    """
    total_pixels = n_frames * width * height * 3
    rate_before = current_size / total_pixels
    rate_after = optimized_size / total_pixels
    rate_delta = rate_before - rate_after

    # The score formula adds rate directly, so delta rate = delta score
    return {
        "rate_before": rate_before,
        "rate_after": rate_after,
        "rate_delta": rate_delta,
        "score_impact_estimate": rate_delta,
        "bytes_saved": current_size - optimized_size,
        "description": (
            f"Rate: {rate_before:.6f} -> {rate_after:.6f} "
            f"(delta={rate_delta:.6f}, est. score improvement={rate_delta:.6f})"
        ),
    }
