#!/usr/bin/env python3
"""Materialize a byte-closed PR106 + wavelet residual sidecar candidate.

Per CLAUDE.md HNeRV parity discipline + operator directive 2026-05-11
pre-stage layer: this tool builds a research-only candidate archive that
the inflate runtime at ``submissions/pr106_wavelet_residual_sidecar/``
consumes. NO score claim. NO GPU dispatch. NO MPS. NO /tmp paths.

Pipeline:

1. Read PR106 r2 archive (``submissions/pr106_latent_sidecar_r2/archive.zip``).
2. Build a synthetic zero-or-near-zero wavelet residual blob in the canonical
   wire format: per-frame [4×4B band scales (cA, cH, cV, cD)] +
   [3×437×582 int8 coefficients per band] (single-level 2D Haar at half
   camera resolution). Default --residual-mode=zero emits an all-zero residual
   (identity bolt-on; proves wire-format closure without claiming any score
   movement). The score-aware residual generation (driven by SegNet/PoseNet
   gradients on PR106 decoded outputs) is downstream of this scaffold.
3. Wrap via ``tac.residual_basis.pr106_sidecar_packing.build_archive(...)``.
4. Emit ``wavelet_pr106_residual_sidecar_archive.zip`` + manifest.

Promotion-status: score_claim=False, ready_for_exact_eval_dispatch=False,
promotion_eligible=False — pinned at every emission boundary.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.residual_basis.pr106_materializer_helpers import (  # noqa: E402
    DEFAULT_PR106_ARCHIVE,
    MaterializerError,
    materialize_family_archive,
    run_no_op_detector_byte_mutation,
)
from tac.residual_basis.pr106_sidecar_packing import PR106_RESIDUAL_FORMAT_IDS  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164
HALF_H, HALF_W = CAMERA_H // 2, CAMERA_W // 2
RGB_CHANNELS = 3
BAND_SIZE_PER_CHANNEL = HALF_H * HALF_W
PER_FRAME_BYTES = 4 * 4 + 4 * RGB_CHANNELS * BAND_SIZE_PER_CHANNEL


def build_wavelet_residual_blob(
    *, n_frames: int, mode: str = "zero", default_scale: float = 0.0
) -> bytes:
    """Build a wire-format-compliant wavelet residual blob.

    mode='zero':    all-zero coefficients + zero scales (identity bolt-on).
    mode='probe':   alternating ±1 int8 coefficients at scale=default_scale
                    (used by the byte-mutation no-op detector smoke).
    """
    if mode not in ("zero", "probe"):
        raise MaterializerError(f"unknown residual mode: {mode!r}")
    parts: list[bytes] = []
    for _ in range(n_frames):
        if mode == "zero":
            scales = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
            band_bytes = b"\x00" * (RGB_CHANNELS * BAND_SIZE_PER_CHANNEL)
            parts.append(scales + band_bytes * 4)
        else:  # probe
            scales = struct.pack(
                "<4f", default_scale, default_scale, default_scale, default_scale
            )
            # Alternating ±1 int8 across the band so any mutation is observable.
            probe = (b"\x01\xFF") * (RGB_CHANNELS * BAND_SIZE_PER_CHANNEL // 2)
            if len(probe) < RGB_CHANNELS * BAND_SIZE_PER_CHANNEL:
                probe += b"\x01"
            parts.append(scales + probe * 4)
    return b"".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize PR106 + wavelet residual sidecar candidate"
    )
    parser.add_argument(
        "--pr106-archive",
        type=Path,
        default=DEFAULT_PR106_ARCHIVE,
        help="PR106 r2 canonical archive.zip path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination dir for archive.zip + manifest (must not exist)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=0,
        help=(
            "Frame count for the residual blob; 0 (default) emits an empty "
            "residual (scaffold mode — wire-format closure only, no per-frame "
            "data). Pass --n-frames 1200 with --residual-mode probe to produce "
            "the full-size residual for the inflate.py byte-mutation smoke."
        ),
    )
    parser.add_argument(
        "--residual-mode",
        choices=("empty", "zero", "probe"),
        default="empty",
        help=(
            "empty: no residual bytes (scaffold-readiness archive); "
            "zero: all-zero per-frame residual (identity bolt-on); "
            "probe: alternating ±1 for inflate-runtime byte-mutation smoke."
        ),
    )
    parser.add_argument(
        "--default-scale",
        type=float,
        default=0.0,
        help="Per-band float32 scale (zero by default; non-zero requires explicit opt-in)",
    )
    parser.add_argument(
        "--skip-no-op-smoke", action="store_true", help="Skip the byte-mutation smoke"
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(
            f"ERROR: --output-dir must be empty or not exist: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    if args.residual_mode == "empty":
        residual_bytes = b""
    else:
        if args.n_frames <= 0:
            print(
                "ERROR: --n-frames must be > 0 when --residual-mode is not 'empty'",
                file=sys.stderr,
            )
            return 2
        residual_bytes = build_wavelet_residual_blob(
            n_frames=args.n_frames,
            mode=args.residual_mode,
            default_scale=args.default_scale,
        )
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family="wavelet",
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "default_scale": args.default_scale,
            "n_frames": args.n_frames,
            "per_frame_bytes": PER_FRAME_BYTES,
            "wavelet": "haar_db1_single_level",
            "rationale": (
                "single-level Haar matches the numpy_inverse_dwt scaffold; "
                "multi-level extensions belong to the L2-promotion bolt-on."
            ),
        },
    )
    if not args.skip_no_op_smoke:
        smoke = run_no_op_detector_byte_mutation(
            archive_bytes=build.archive_bytes,
            expected_format_id=PR106_RESIDUAL_FORMAT_IDS["wavelet"],
        )
        print(f"[no_op_detector_byte_mutation_smoke] {smoke}", file=sys.stderr)
    print(f"materialized archive: {archive_zip}")
    print(f"manifest:             {manifest_path}")
    print(f"archive sha256:       {manifest.archive_sha256}")
    print(f"archive size bytes:   {manifest.archive_bytes_size}")
    print(f"residual bytes:       {manifest.residual_bytes_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
