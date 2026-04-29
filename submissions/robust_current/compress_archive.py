#!/usr/bin/env python
"""Build a submission archive using the canonical archive builder.

Replaces the manual `zip -9 -r` in compress.sh with the validated
build_submission_archive() pipeline from tac.submission_archive. Supports:

  - Half-frame masks (600 odd frames, Quantizr paradigm)
  - Binary poses (.bin instead of .pt, ~50% smaller)
  - Brotli pre-compression of individual artifacts
  - Full provenance (contents, sizes, rate term, MD5)
  - Post-build validation against the chosen manifest

Usage:
    PYTHONPATH=src python submissions/robust_current/compress_archive.py \\
        --renderer-bin renderer.bin \\
        --masks-path masks.mkv \\
        --poses-path optimized_poses.pt \\
        --output archive.zip

    # Half-frame masks + binary poses (compact manifest):
    PYTHONPATH=src python submissions/robust_current/compress_archive.py \\
        --renderer-bin renderer.bin \\
        --masks-path masks.mkv \\
        --poses-path optimized_poses.pt \\
        --half-frame --binary-poses \\
        --output archive.zip

    # Dry-run (analyze artifacts, do not build):
    PYTHONPATH=src python submissions/robust_current/compress_archive.py \\
        --renderer-bin renderer.bin \\
        --masks-path masks.mkv \\
        --poses-path optimized_poses.pt \\
        --dry-run
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a validated submission archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--renderer-bin", required=True,
        help="Path to renderer.bin (ASYM-format checkpoint)",
    )
    p.add_argument(
        "--masks-path", required=True,
        help="Path to masks.mkv (AV1 monochrome encoded SegNet masks)",
    )
    p.add_argument(
        "--poses-path", required=True,
        help="Path to poses (.pt or .bin). Converted to .bin if --binary-poses.",
    )
    p.add_argument(
        "--output", default="archive.zip",
        help="Output archive path (default: archive.zip)",
    )
    p.add_argument(
        "--half-frame", action="store_true",
        help="Extract only odd-frame masks (600 frames from 1200). "
             "Requires --masks-path to point to a full 1200-frame masks.mkv; "
             "a new half-frame masks.mkv is written to a temp directory.",
    )
    p.add_argument(
        "--binary-poses", action="store_true",
        help="Convert poses to raw fp16 binary (.bin) instead of .pt. "
             "Saves ~50%% on pose storage (~7.2KB vs ~15KB).",
    )
    p.add_argument(
        "--pose-delta", action="store_true",
        help="Encode poses as anchor + int8 deltas (Lane PD, "
             "src/tac/pose_delta_codec.py). Driving poses are smooth so "
             "delta range is ~10x smaller than absolute range; quantising "
             "deltas to int8 saves ~49%% vs fp16 absolute (3.7KB vs 7.2KB "
             "for 600 pairs). Decoder is pure-Python in submission_archive."
             ".load_optimized_poses (already wired). Mutually exclusive "
             "with --binary-poses and --pose-delta-v2.",
    )
    p.add_argument(
        "--pose-delta-v2", action="store_true",
        help="Lane PD-V2: V1 (anchor + int8 deltas) PLUS static-histogram "
             "arithmetic coding of the int8 stream "
             "(src/tac/pose_delta_codec_v2.py). [prediction] +7-11 basis "
             "points deterministic vs V1 (grand-council stacking codex "
             "memory project_codec_stacking_composition_canonical_orders"
             "_20260429.md). Includes a hard overhead gate that falls back "
             "to V1 transparently if the AC blob is no smaller than V1 — "
             "degenerate inputs (constant pose) silently land as V1 so a "
             "regression cannot ship. Mutually exclusive with "
             "--binary-poses and --pose-delta.",
    )
    p.add_argument(
        "--brotli", action="store_true",
        help="Apply Brotli compression to renderer.bin before archiving "
             "(requires brotli package).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Analyze artifacts and print sizes without building the archive.",
    )
    p.add_argument(
        "--strict", action="store_true", default=True,
        help="Strict validation: unexpected files in archive are errors (default).",
    )
    p.add_argument(
        "--no-strict", action="store_false", dest="strict",
        help="Relaxed validation: unexpected files are warnings, not errors.",
    )
    return p.parse_args()


def _extract_half_frame_masks(
    masks_mkv_path: Path,
    output_dir: Path,
) -> Path:
    """Decode full masks, keep only odd frames, re-encode.

    The Quantizr half-frame paradigm: only odd-frame (frame2) masks are stored
    in the archive. The renderer reuses each odd-frame mask for its paired
    even frame. This halves mask storage with negligible quality loss for the
    asymmetric warp renderer.

    Returns:
        Path to the half-frame masks.mkv in output_dir.
    """
    from tac.mask_codec import decode_masks, encode_masks_monochrome

    print("  Decoding full masks for half-frame extraction ...", file=sys.stderr)
    t0 = time.monotonic()
    full_masks = decode_masks(masks_mkv_path)
    n_full = full_masks.shape[0]
    print(f"    Decoded {n_full} masks ({time.monotonic() - t0:.1f}s)", file=sys.stderr)

    if n_full <= 600:
        print(
            f"    WARNING: Masks already have {n_full} frames (<= 600), "
            f"skipping half-frame extraction.",
            file=sys.stderr,
        )
        return masks_mkv_path

    # Keep only odd-indexed frames (1, 3, 5, ..., 1199)
    half_masks = full_masks[1::2]
    print(
        f"    Half-frame: {n_full} -> {half_masks.shape[0]} masks "
        f"(odd frames only)",
        file=sys.stderr,
    )

    half_path = output_dir / "masks.mkv"
    t1 = time.monotonic()
    file_size = encode_masks_monochrome(half_masks, half_path, crf=20)
    print(
        f"    Re-encoded half-frame masks: {file_size:,} bytes ({time.monotonic() - t1:.1f}s)",
        file=sys.stderr,
    )
    return half_path


def _convert_poses_to_binary(
    poses_path: Path,
    output_dir: Path,
) -> Path:
    """Convert .pt poses to raw fp16 binary format.

    Raw binary: N*6*2 bytes (fp16), no torch overhead.
    Typical saving: 15KB .pt -> 7.2KB .bin for 600 pairs.

    Returns:
        Path to the binary poses file in output_dir.
    """
    import torch
    from tac.submission_archive import save_poses_binary

    if poses_path.suffix == ".bin":
        print(
            f"    Poses already in .bin format ({poses_path.stat().st_size:,} bytes)",
            file=sys.stderr,
        )
        return poses_path

    poses = torch.load(str(poses_path), map_location="cpu", weights_only=True).float()
    bin_path = output_dir / "optimized_poses.bin"
    nbytes = save_poses_binary(poses, bin_path)
    print(
        f"    Converted poses: {poses_path.name} ({poses_path.stat().st_size:,} bytes) "
        f"-> .bin ({nbytes:,} bytes, {poses.shape})",
        file=sys.stderr,
    )
    return bin_path


def _convert_poses_to_pose_delta(
    poses_path: Path,
    output_dir: Path,
) -> Path:
    """Convert poses (.pt or .bin) to Lane PD pose-delta encoding.

    Selfcomp/Quantizr/Lane A all ship optimized_poses.pt as ABSOLUTE poses
    of shape (N_pairs, 6) in fp16. Driving poses are SMOOTH — consecutive
    pose vectors differ by tiny deltas. Lane PD stores frame-0 absolute +
    (N-1) int8 deltas, saving ~49% vs fp16 absolute.

    Math: 600 pairs * 6 dims * fp16 = 7200 B → anchor (12 B) + delta_scale
    (12 B) + 599*6 int8 deltas (3594 B) ≈ 3618 B + dict overhead. Score
    delta: 25 * (7200 - 3618) / 37545489 ≈ -0.0024. Mutually exclusive
    with --binary-poses.

    Output is written as optimized_poses.pt (a torch-saved dict carrying
    the format='pose_delta_v1' sentinel) so the existing decoder in
    submission_archive.load_optimized_poses picks it up automatically.

    Returns: Path to the saved .pt file.
    """
    import torch
    from tac.pose_delta_codec import encode_pose_deltas

    if poses_path.suffix == ".bin":
        # Load raw fp16 binary back to a torch tensor.
        from tac.submission_archive import load_poses_binary
        poses = load_poses_binary(poses_path).float()
    else:
        poses = torch.load(
            str(poses_path), map_location="cpu", weights_only=True
        ).float()

    obj = encode_pose_deltas(poses)
    pt_path = output_dir / "optimized_poses.pt"
    torch.save(obj, str(pt_path))
    print(
        f"    Encoded poses (Lane PD): {poses_path.name} "
        f"({poses_path.stat().st_size:,} bytes) -> pose_delta_v1 .pt "
        f"({pt_path.stat().st_size:,} bytes, {poses.shape})",
        file=sys.stderr,
    )
    return pt_path


def _convert_poses_to_pose_delta_v2(
    poses_path: Path,
    output_dir: Path,
) -> Path:
    """Convert poses (.pt or .bin) to Lane PD-V2 arithmetic-coded format.

    [prediction] +7-11 basis points deterministic vs V1 (grand-council
    stacking codex 2026-04-29). The hard overhead gate inside
    ``encode_pose_delta_v2_or_fallback`` falls back to V1 transparently if
    V2 fails to beat V1's torch.save bytes — so degenerate inputs (constant
    pose) cannot ship a regression. Mutually exclusive with
    ``--binary-poses`` and ``--pose-delta``.

    Output is written as optimized_poses.pt (a torch-saved dict carrying
    EITHER the format='pose_delta_v2' sentinel + AC blob, OR the V1
    sentinel after fallback) so the existing decoder in
    submission_archive.load_optimized_poses picks it up automatically.

    Returns: Path to the saved .pt file.
    """
    import torch
    from tac.pose_delta_codec_v2 import (
        POSE_DELTA_FORMAT_SENTINEL_V2,
        encode_pose_delta_v2_or_fallback,
    )

    if poses_path.suffix == ".bin":
        from tac.submission_archive import load_poses_binary
        poses = load_poses_binary(poses_path).float()
    else:
        poses = torch.load(
            str(poses_path), map_location="cpu", weights_only=True
        ).float()

    obj = encode_pose_delta_v2_or_fallback(poses)
    pt_path = output_dir / "optimized_poses.pt"
    torch.save(obj, str(pt_path))
    used_v2 = obj.get("format") == POSE_DELTA_FORMAT_SENTINEL_V2
    label = "pose_delta_v2" if used_v2 else "pose_delta_v1 (overhead-gate fallback)"
    print(
        f"    Encoded poses (Lane PD-V2 / {label}): {poses_path.name} "
        f"({poses_path.stat().st_size:,} bytes) -> "
        f"{pt_path.stat().st_size:,} bytes, {poses.shape}",
        file=sys.stderr,
    )
    return pt_path



def main() -> int:
    args = _parse_args()

    renderer_bin = Path(args.renderer_bin)
    masks_path = Path(args.masks_path)
    poses_path = Path(args.poses_path)
    output_path = Path(args.output)

    # Validate inputs exist
    missing = []
    for label, p in [("renderer-bin", renderer_bin), ("masks-path", masks_path),
                     ("poses-path", poses_path)]:
        if not p.exists():
            missing.append(f"  --{label}: {p}")
    if missing:
        print("ERROR: Required artifacts not found:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        return 1

    # Print input analysis
    print("=" * 60, file=sys.stderr)
    print("Submission archive builder", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Renderer:    {renderer_bin} ({renderer_bin.stat().st_size:,} bytes)", file=sys.stderr)
    print(f"  Masks:       {masks_path} ({masks_path.stat().st_size:,} bytes)", file=sys.stderr)
    print(f"  Poses:       {poses_path} ({poses_path.stat().st_size:,} bytes)", file=sys.stderr)
    print(f"  Output:      {output_path}", file=sys.stderr)
    print(f"  Half-frame:  {args.half_frame}", file=sys.stderr)
    print(f"  Binary poses:{args.binary_poses}", file=sys.stderr)
    print(f"  Brotli:      {args.brotli}", file=sys.stderr)
    print(file=sys.stderr)

    if args.dry_run:
        from tac.submission_archive import ORIGINAL_VIDEO_BYTES

        total_raw = renderer_bin.stat().st_size + masks_path.stat().st_size + poses_path.stat().st_size
        print(f"[dry-run] Total raw artifact size: {total_raw:,} bytes", file=sys.stderr)
        print(f"[dry-run] Estimated rate term (raw): {25 * total_raw / ORIGINAL_VIDEO_BYTES:.4f}", file=sys.stderr)
        print(f"[dry-run] No archive written.", file=sys.stderr)
        return 0

    from tac.submission_archive import (
        ArchiveManifest,
        RENDERER_COMPACT_MANIFEST,
        RENDERER_SUBMISSION_MANIFEST,
        build_submission_archive,
    )

    with tempfile.TemporaryDirectory(prefix="compress_archive_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Step 1: Half-frame mask extraction (if requested)
        final_masks = masks_path
        if args.half_frame:
            print("Step 1: Extracting half-frame masks ...", file=sys.stderr)
            final_masks = _extract_half_frame_masks(masks_path, tmpdir_path)
        else:
            print("Step 1: Using full-frame masks (no half-frame extraction)", file=sys.stderr)

        # Step 2: Pose conversion (binary fp16 OR Lane PD pose-delta OR
        # Lane PD-V2 arithmetic-coded pose-delta OR none).
        final_poses_pt = None
        final_poses_bin = None
        pose_flag_count = sum(
            int(bool(x))
            for x in (args.binary_poses, args.pose_delta, args.pose_delta_v2)
        )
        if pose_flag_count > 1:
            raise SystemExit(
                "ERROR: --binary-poses, --pose-delta, and --pose-delta-v2 are "
                "mutually exclusive (all three are pose-encoding choices). "
                "Pick exactly one. Lane PD-V2 [prediction] +7-11 bp over "
                "Lane PD V1 (grand-council stacking codex). For pinned-"
                "baseline reproductions where the decoder might be old, "
                "use --binary-poses."
            )
        if args.binary_poses:
            print("Step 2: Converting poses to binary format ...", file=sys.stderr)
            final_poses_bin = _convert_poses_to_binary(poses_path, tmpdir_path)
        elif args.pose_delta_v2:
            print("Step 2: Encoding poses as Lane PD-V2 (AC + anchor+deltas) ...", file=sys.stderr)
            final_poses_pt = _convert_poses_to_pose_delta_v2(poses_path, tmpdir_path)
        elif args.pose_delta:
            print("Step 2: Encoding poses as Lane PD anchor+deltas ...", file=sys.stderr)
            final_poses_pt = _convert_poses_to_pose_delta(poses_path, tmpdir_path)
        else:
            print("Step 2: Using .pt poses (no binary conversion)", file=sys.stderr)
            final_poses_pt = poses_path

        # Step 3: Select manifest based on options
        if args.binary_poses:
            manifest = RENDERER_COMPACT_MANIFEST
        else:
            manifest = RENDERER_SUBMISSION_MANIFEST

        use_brotli = bool(args.brotli)
        if use_brotli:
            print("Step 3: Brotli compression ENABLED (applied by archive builder)", file=sys.stderr)
        else:
            print("Step 3: Brotli compression disabled", file=sys.stderr)

        print(
            f"\nStep 4: Building archive with manifest: "
            f"{manifest.required_files()}"
            f"{' + Brotli' if use_brotli else ''}",
            file=sys.stderr,
        )

        # Step 5: Build and validate (Brotli handled inside build_submission_archive)
        t0 = time.monotonic()
        result = build_submission_archive(
            output_path=output_path,
            renderer_bin=renderer_bin,
            masks_mkv=final_masks,
            optimized_poses_pt=final_poses_pt,
            optimized_poses_bin=final_poses_bin,
            manifest=manifest,
            validate=True,
            use_brotli=use_brotli,
        )
        elapsed = time.monotonic() - t0

    # Print results
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("ARCHIVE BUILT SUCCESSFULLY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(result.summary(), file=sys.stderr)
    print(f"  Build time: {elapsed:.1f}s", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # Also print to stdout for shell capture
    print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
