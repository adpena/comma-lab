"""Canonical submission archive builder and validator.

This is the SINGLE SOURCE OF TRUTH for what goes into archive.zip.
Every auth eval, every compress run, every deployment MUST use this module
to build or validate the archive. Ad hoc archive construction is forbidden.

The archive measurement disaster of 2026-04-21 (119KB renderer-only archive
evaluated as if it were the 338KB full submission) happened because archive
construction was scattered across 6+ locations with no validation.
This module makes the wrong thing impossible by construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # R41 fix: ruff F821 flagged torch.Tensor string annotations as undefined
    # because torch wasn't importable in this scope. TYPE_CHECKING import keeps
    # the runtime light (torch never loaded at module import) while letting type
    # checkers AND ruff resolve the symbol.
    import torch

logger = logging.getLogger(__name__)


# ============================================================
# Brotli compression utilities
# ============================================================

def brotli_compress(data: bytes, quality: int = 11, lgwin: int = 24) -> bytes:
    """Compress bytes with Brotli (quality 11 = max, matches Quantizr)."""
    import brotli
    return brotli.compress(data, quality=quality, lgwin=lgwin)


def brotli_decompress(data: bytes) -> bytes:
    """Decompress Brotli-compressed bytes."""
    import brotli
    return brotli.decompress(data)


def compress_file_brotli(
    input_path: Path, output_path: Path | None = None, quality: int = 11
) -> Path:
    """Compress a file with Brotli, producing .br suffix.

    Args:
        input_path: file to compress
        output_path: destination (default: input_path + '.br')
        quality: Brotli quality 0-11 (11 = max, matches Quantizr)

    Returns:
        Path to the compressed file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + ".br")
    data = input_path.read_bytes()
    compressed = brotli_compress(data, quality=quality)
    output_path.write_bytes(compressed)
    ratio = len(compressed) / len(data) * 100
    logger.info(
        "Brotli: %s %d B -> %d B (%.1f%%)", input_path.name, len(data), len(compressed), ratio
    )
    print(f"  Brotli: {input_path.name} {len(data):,}B -> {len(compressed):,}B ({ratio:.1f}%)")
    return output_path


def decompress_file_brotli(input_path: Path, output_path: Path | None = None) -> Path:
    """Decompress a .br file.

    Args:
        input_path: Brotli-compressed file (typically *.br)
        output_path: destination (default: strip .br suffix)

    Returns:
        Path to the decompressed file.
    """
    input_path = Path(input_path)
    if output_path is None:
        # Strip .br suffix
        if input_path.suffix == ".br":
            output_path = input_path.with_suffix("")
        else:
            raise ValueError(
                f"Cannot infer output path: {input_path} does not end in .br. "
                f"Provide output_path explicitly."
            )
    data = input_path.read_bytes()
    decompressed = brotli_decompress(data)
    output_path.write_bytes(decompressed)
    logger.info(
        "Brotli decompress: %s %d B -> %s %d B",
        input_path.name, len(data), output_path.name, len(decompressed),
    )
    return output_path


def decompress_brotli_files_in_dir(directory: Path | str) -> int:
    """Decompress all .br files in a directory, removing the .br originals.

    This is the canonical inflate-time decompression step: after the archive
    is extracted, any .br files are decompressed in place before the inflate
    pipeline reads them.

    Args:
        directory: directory to scan for .br files

    Returns:
        Number of files decompressed.
    """
    directory = Path(directory)
    count = 0
    for br_file in sorted(directory.glob("*.br")):
        decompressed_path = decompress_file_brotli(br_file)
        br_file.unlink()  # remove .br, keep decompressed
        logger.info("Decompressed and removed: %s -> %s", br_file.name, decompressed_path.name)
        count += 1
    return count


# ============================================================
# Contest data contract — single source of truth
# Both builder (compress) and inflater (inflate) import these.
# Upstream evaluate.py expects these exact dimensions.
# ============================================================
ORIGINAL_VIDEO_BYTES = 37_545_489
NUM_FRAMES = 1200
NUM_PAIRS = 600
HALF_FRAMES = 600  # For half-frame mask encoding (Quantizr paradigm)
OUT_W, OUT_H = 1164, 874  # Camera output resolution
SEG_W, SEG_H = 512, 384  # SegNet / renderer resolution
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES  # 3,662,409,600
POSE_DIM = 6  # FiLM conditioning vector dimension


@dataclass
class ArchiveManifest:
    """Declares what the archive MUST contain for a given inflate mode."""

    renderer_bin: bool = False
    masks_mkv: bool = False
    optimized_poses_pt: bool = False
    optimized_poses_bin: bool = False  # raw fp16 binary (half the size of .pt)
    optimized_embedding_pt: bool = False
    poses_pt: bool = False
    corrections_bin: bool = False
    gradient_corrections_bin: bool = False
    mini_segnet_bin: bool = False
    mini_posenet_bin: bool = False
    posenet_targets_bin: bool = False

    def required_files(self) -> list[str]:
        mapping = {
            "renderer_bin": "renderer.bin",
            "masks_mkv": "masks.mkv",
            "optimized_poses_pt": "optimized_poses.pt",
            "optimized_poses_bin": "optimized_poses.bin",
            "optimized_embedding_pt": "optimized_embedding.pt",
            "poses_pt": "poses.pt",
            "corrections_bin": "corrections.bin",
            "gradient_corrections_bin": "gradient_corrections.bin",
            "mini_segnet_bin": "mini_segnet.bin",
            "mini_posenet_bin": "mini_posenet.bin",
            "posenet_targets_bin": "posenet_targets.bin",
        }
        return [
            mapping[k]
            for k, v in vars(self).items()
            if k in mapping and v
        ]


# The manifest for our current renderer-based submission
RENDERER_SUBMISSION_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_mkv=True,
    optimized_poses_pt=True,
)

# Compact manifest: raw binary poses instead of .pt (saves ~8KB)
RENDERER_COMPACT_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_mkv=True,
    optimized_poses_bin=True,
)


def detect_pose_manifest(archive_path) -> ArchiveManifest:
    """R38 fix: inspect the archive and return whichever manifest matches
    the pose format actually present. Prior code at validation sites used
    a fixed manifest and reported false-negative MISSING for the other
    pose format. Use this helper at every validation site to auto-pick.
    """
    import zipfile
    try:
        with zipfile.ZipFile(str(archive_path), "r") as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, FileNotFoundError):
        # Default to .pt manifest; the validator will catch the missing zip.
        return RENDERER_SUBMISSION_MANIFEST
    if "optimized_poses.bin" in names:
        return RENDERER_COMPACT_MANIFEST
    return RENDERER_SUBMISSION_MANIFEST


@dataclass
class ArchiveValidationResult:
    """Result of archive validation."""

    valid: bool
    archive_path: Path
    archive_bytes: int
    rate_term: float
    files_found: dict[str, int] = field(default_factory=dict)
    files_missing: list[str] = field(default_factory=list)
    files_unexpected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    md5: str = ""

    def summary(self) -> str:
        lines = [
            f"Archive: {self.archive_path}",
            f"  Size: {self.archive_bytes:,} bytes ({self.archive_bytes / 1024:.1f} KB)",
            f"  Rate term: 25 * {self.archive_bytes} / {ORIGINAL_VIDEO_BYTES} = {self.rate_term:.4f}",
            f"  MD5: {self.md5}",
            f"  Valid: {self.valid}",
        ]
        if self.files_found:
            lines.append("  Contents:")
            for name, size in sorted(self.files_found.items()):
                lines.append(f"    {name}: {size:,} bytes")
        if self.files_missing:
            lines.append(f"  MISSING: {', '.join(self.files_missing)}")
        if self.files_unexpected:
            lines.append(f"  UNEXPECTED: {', '.join(self.files_unexpected)}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


def validate_archive(
    archive_path: Path | str,
    manifest: ArchiveManifest = RENDERER_SUBMISSION_MANIFEST,
    strict: bool = True,
) -> ArchiveValidationResult:
    """Validate a submission archive against a manifest.

    Args:
        archive_path: Path to archive.zip
        manifest: Expected contents declaration
        strict: If True, unexpected files are errors. If False, warnings.

    Returns:
        ArchiveValidationResult with full provenance.

    Raises:
        FileNotFoundError: if archive_path does not exist.
        zipfile.BadZipFile: if file is not a valid zip.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    archive_bytes = archive_path.stat().st_size
    md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    rate_term = 25 * archive_bytes / ORIGINAL_VIDEO_BYTES

    result = ArchiveValidationResult(
        valid=True,
        archive_path=archive_path,
        archive_bytes=archive_bytes,
        rate_term=rate_term,
        md5=md5,
    )

    required = set(manifest.required_files())

    with zipfile.ZipFile(archive_path, "r") as zf:
        archive_names = set(zf.namelist())
        for info in zf.infolist():
            result.files_found[info.filename] = info.file_size

        # Check required files present
        for req in required:
            if req not in archive_names:
                result.files_missing.append(req)
                result.errors.append(f"Required file missing: {req}")
                result.valid = False

        # Check for unexpected files
        for name in archive_names:
            if name not in required:
                result.files_unexpected.append(name)
                msg = f"Unexpected file in archive: {name}"
                if strict:
                    result.errors.append(msg)
                    result.valid = False
                else:
                    result.warnings.append(msg)

    # Sanity checks on known file sizes (skip for Brotli-compressed archives
    # since .br files have unpredictable sizes after compression)
    is_brotli = any(name.endswith(".br") for name in result.files_found)

    if not is_brotli:
        if "renderer.bin" in result.files_found:
            size = result.files_found["renderer.bin"]
            if size < 10_000:
                result.errors.append(
                    f"renderer.bin suspiciously small: {size} bytes (expected ~150-300KB)"
                )
                result.valid = False
            elif size > 5_000_000:
                result.warnings.append(
                    f"renderer.bin unusually large: {size} bytes (expected ~150-300KB)"
                )

        if "masks.mkv" in result.files_found:
            size = result.files_found["masks.mkv"]
            if size < 1_000:
                result.errors.append(
                    f"masks.mkv suspiciously small: {size} bytes (expected ~50-200KB)"
                )
                result.valid = False

        if "optimized_poses.pt" in result.files_found:
            size = result.files_found["optimized_poses.pt"]
            # 600 pairs * 6 values * 2 bytes (fp16) = 7.2KB minimum
            if size < 1_000:
                result.errors.append(
                    f"optimized_poses.pt suspiciously small: {size} bytes (expected ~7-20KB)"
                )
                result.valid = False

    return result


def compute_rate_term(archive_path: Path | str) -> float:
    """Compute the rate term for a submission archive.

    Returns: 25 * archive_bytes / original_video_bytes
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    return 25 * archive_path.stat().st_size / ORIGINAL_VIDEO_BYTES


def save_poses_binary(poses: torch.Tensor, output_path: Path | str) -> int:
    """Save poses as raw fp16 binary (minimal overhead, ~7.2KB for 600×6).

    Args:
        poses: (N, 6) float tensor of optimized pose vectors
        output_path: output .bin file path

    Returns:
        File size in bytes
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = poses.half().cpu().numpy().tobytes()
    output_path.write_bytes(raw)
    return len(raw)


def load_poses_binary(path: Path | str, pose_dim: int = 6) -> torch.Tensor:
    """Load poses from raw fp16 binary.

    Returns:
        (N, pose_dim) float32 tensor
    """
    import torch

    raw = Path(path).read_bytes()
    return torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(-1, pose_dim).float()


def build_submission_archive(
    output_path: Path | str,
    renderer_bin: Path | str | None = None,
    masks_mkv: Path | str | None = None,
    optimized_poses_pt: Path | str | None = None,
    optimized_poses_bin: Path | str | None = None,
    optimized_embedding_pt: Path | str | None = None,
    gradient_corrections_bin: Path | str | None = None,
    manifest: ArchiveManifest = RENDERER_SUBMISSION_MANIFEST,
    validate: bool = True,
    use_brotli: bool = False,
    brotli_quality: int = 11,
) -> ArchiveValidationResult:
    """Build and validate a submission archive.

    This is the ONLY correct way to build an archive for submission or eval.

    Args:
        output_path: Where to write archive.zip
        renderer_bin: Path to renderer.bin
        masks_mkv: Path to masks.mkv
        optimized_poses_pt: Path to optimized_poses.pt
        optimized_poses_bin: Path to optimized_poses.bin (raw fp16, smaller)
        optimized_embedding_pt: Path to optimized_embedding.pt (optional)
        manifest: Expected contents manifest
        validate: If True, validate after building (default True)
        use_brotli: If True, Brotli-compress each artifact before adding
            to the ZIP. Archive will contain .br files (e.g. renderer.bin.br).
            The inflate side must call decompress_brotli_files_in_dir()
            after extraction.
        brotli_quality: Brotli quality level 0-11 (default 11 = max,
            matches Quantizr). Only used when use_brotli=True.

    Returns:
        ArchiveValidationResult with full provenance.

    Raises:
        FileNotFoundError: if any required source file is missing.
        ValueError: if validation fails.
    """
    output_path = Path(output_path)

    # Map manifest fields to source paths
    source_map: dict[str, Path | None] = {
        "renderer.bin": Path(renderer_bin) if renderer_bin else None,
        "masks.mkv": Path(masks_mkv) if masks_mkv else None,
        "optimized_poses.pt": Path(optimized_poses_pt) if optimized_poses_pt else None,
        "optimized_poses.bin": Path(optimized_poses_bin) if optimized_poses_bin else None,
        "optimized_embedding.pt": Path(optimized_embedding_pt) if optimized_embedding_pt else None,
        "gradient_corrections.bin": Path(gradient_corrections_bin) if gradient_corrections_bin else None,
    }

    required = set(manifest.required_files())

    # Verify all required source files exist BEFORE creating the archive
    for name in required:
        src = source_map.get(name)
        if src is None:
            raise FileNotFoundError(
                f"Required artifact {name} not provided. "
                f"Pass it as an argument to build_submission_archive()."
            )
        if not src.exists():
            raise FileNotFoundError(
                f"Required artifact {name} not found at {src}. "
                f"Build it first, then pass the path."
            )

    # ── Validate artifact integrity before building ──
    # Mask frame count: must be NUM_FRAMES (1200) or HALF_FRAMES (600)
    masks_src = source_map.get("masks.mkv")
    if masks_src and masks_src.exists():
        import subprocess as _sp
        try:
            probe = _sp.run(
                ["ffprobe", "-v", "error", "-count_frames",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", str(masks_src)],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "ffprobe is required to validate mask frame count but was not found. "
                "Install ffmpeg (which includes ffprobe) to enable mask validation."
            )
        except _sp.TimeoutExpired:
            logger.warning(
                "ffprobe timed out on %s — mask frame count NOT validated. "
                "File may be corrupted or on a slow mount.", masks_src,
            )
            probe = None
        if probe is not None and probe.returncode != 0:
            logger.warning(
                "ffprobe returned non-zero (%d) for %s — mask frame count NOT validated. "
                "stderr: %s", probe.returncode, masks_src, probe.stderr.strip(),
            )
        elif probe is not None and not probe.stdout.strip():
            logger.warning(
                "ffprobe returned empty output for %s — mask frame count NOT validated.",
                masks_src,
            )
        else:
            n_mask_frames = int(probe.stdout.strip())
            if n_mask_frames not in (NUM_FRAMES, HALF_FRAMES):
                raise ValueError(
                    f"masks.mkv has {n_mask_frames} frames. "
                    f"Expected {NUM_FRAMES} (full) or {HALF_FRAMES} (half-frame). "
                    f"Rebuild masks with correct frame count."
                )
            logger.info("Mask frame count: %d (%s)",
                        n_mask_frames,
                        "full" if n_mask_frames == NUM_FRAMES else "half-frame")

    # FP4 without QAT warning: check renderer.bin header
    renderer_src = source_map.get("renderer.bin")
    if renderer_src and renderer_src.exists():
        header = renderer_src.read_bytes()[:4]
        is_fp4 = header == b"FP4A"
        renderer_size = renderer_src.stat().st_size
        if is_fp4:
            logger.warning(
                "FP4 renderer detected (%d bytes). Ensure QAT was used — "
                "FP4 without QAT introduces ~11.6 pixel mean error (39x pose signal).",
                renderer_size,
            )

    # Build the archive
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # When using Brotli, we store pre-compressed .br blobs with ZIP_STORED
    # (no double compression). Without Brotli, use ZIP_DEFLATED as before.
    zip_method = zipfile.ZIP_STORED if use_brotli else zipfile.ZIP_DEFLATED
    zip_kwargs: dict = {} if use_brotli else {"compresslevel": 9}

    if use_brotli:
        print("Brotli compression enabled (quality=%d)" % brotli_quality)

    with zipfile.ZipFile(output_path, "w", zip_method, **zip_kwargs) as zf:
        for name in required:
            src = source_map[name]
            assert src is not None  # guaranteed by check above

            if use_brotli:
                # Brotli-compress the artifact data, store as name.br
                raw_data = src.read_bytes()
                compressed_data = brotli_compress(raw_data, quality=brotli_quality)
                br_name = name + ".br"
                zf.writestr(br_name, compressed_data)
                ratio = len(compressed_data) / len(raw_data) * 100
                logger.info(
                    "  Added %s (%d -> %d bytes, %.1f%%)",
                    br_name, len(raw_data), len(compressed_data), ratio,
                )
                print(
                    f"  Brotli: {name} {len(raw_data):,}B -> "
                    f"{len(compressed_data):,}B ({ratio:.1f}%) as {br_name}"
                )
            else:
                zf.write(src, arcname=name)
                logger.info("  Added %s (%d bytes)", name, src.stat().st_size)

    # Determine expected archive names for validation
    if use_brotli:
        brotli_manifest = _brotli_manifest(manifest)
        result = validate_archive(output_path, manifest=brotli_manifest, strict=True)
    else:
        result = validate_archive(output_path, manifest=manifest, strict=True)

    logger.info("Archive built: %s", result.summary())

    if validate and not result.valid:
        raise ValueError(
            f"Archive validation FAILED after build:\n{result.summary()}"
        )

    return result


def _brotli_manifest(manifest: ArchiveManifest) -> ArchiveManifest:
    """Create a Brotli-aware manifest that expects .br suffixed filenames.

    This is used internally for validation of Brotli-compressed archives.
    The manifest mirrors the original but maps each file to name.br.
    """
    # We cannot reuse ArchiveManifest directly because required_files()
    # returns hardcoded names. Instead, create a thin wrapper manifest
    # that validates .br names.
    return _BrotliManifest(manifest)


class _BrotliManifest(ArchiveManifest):
    """Manifest wrapper that expects .br suffixed filenames."""

    def __init__(self, inner: ArchiveManifest):
        # Copy all fields from inner
        for k, v in vars(inner).items():
            setattr(self, k, v)
        self._inner = inner

    def required_files(self) -> list[str]:
        return [name + ".br" for name in self._inner.required_files()]


def require_valid_archive(
    archive_path: Path | str,
    manifest: ArchiveManifest = RENDERER_SUBMISSION_MANIFEST,
    context: str = "auth eval",
) -> ArchiveValidationResult:
    """Validate an archive and raise if invalid.

    Use this at the START of any auth eval, deployment, or submission pipeline.

    Args:
        archive_path: Path to archive.zip
        manifest: Expected contents
        context: Human-readable context for error messages

    Returns:
        ArchiveValidationResult (only if valid)

    Raises:
        FileNotFoundError: if archive does not exist
        ValueError: if archive is invalid
    """
    result = validate_archive(archive_path, manifest=manifest, strict=True)

    if not result.valid:
        raise ValueError(
            f"ARCHIVE VALIDATION FAILED for {context}.\n"
            f"This means the {context} would produce an INVALID score.\n"
            f"\n{result.summary()}\n"
            f"\nFix the archive before running {context}."
        )

    # Always print provenance
    logger.info("[%s] Archive validated:\n%s", context, result.summary())
    print(f"\n[{context}] Archive: {result.archive_bytes:,} bytes, "
          f"rate={result.rate_term:.4f}, md5={result.md5[:12]}...")

    return result


def score_from_components(
    segnet_dist: float,
    posenet_dist: float,
    archive_bytes: int,
) -> dict[str, float]:
    """Compute contest score from components with full breakdown.

    Returns dict with 'total', 'segnet_term', 'posenet_term', 'rate_term',
    and all input values for provenance.
    """
    import math

    rate = 25 * archive_bytes / ORIGINAL_VIDEO_BYTES
    segnet_term = 100 * segnet_dist
    posenet_term = math.sqrt(10 * posenet_dist)

    return {
        "total": segnet_term + posenet_term + rate,
        "segnet_term": segnet_term,
        "posenet_term": posenet_term,
        "rate_term": rate,
        "segnet_dist": segnet_dist,
        "posenet_dist": posenet_dist,
        "archive_bytes": archive_bytes,
        "original_video_bytes": ORIGINAL_VIDEO_BYTES,
    }
