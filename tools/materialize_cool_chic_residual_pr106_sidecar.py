#!/usr/bin/env python3
"""Materialize a byte-closed PR106 + Cool-Chic residual sidecar candidate.

See ``submissions/pr106_cool_chic_residual_sidecar/inflate.py`` for the wire
format consumed by this scaffold. Pipeline + promotion-status pinning are
identical to the wavelet sister materializer; only the residual wire format
differs.

Wire format (residual blob):
    2B n_levels (uint16 LE)
    per level: 4B scale (float32) + int8 coefs at (H/2^L * W/2^L * 3) per frame.

Default mode='empty' emits an empty residual (scaffold-readiness archive).
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
RGB_CHANNELS = 3


def build_cool_chic_residual_blob(
    *, n_frames: int, n_levels: int, mode: str, default_scale: float = 0.0
) -> bytes:
    if mode not in ("zero", "probe"):
        raise MaterializerError(f"unknown residual mode: {mode!r}")
    if n_levels < 1 or n_levels > 6:
        raise MaterializerError(f"n_levels must be in [1, 6]; got {n_levels}")
    parts = [struct.pack("<H", n_levels)]
    for L in range(n_levels):
        h_L = CAMERA_H // (2 ** L)
        w_L = CAMERA_W // (2 ** L)
        n = n_frames * h_L * w_L * RGB_CHANNELS
        parts.append(struct.pack("<f", default_scale))
        if mode == "zero":
            parts.append(b"\x00" * n)
        else:  # probe
            probe = (b"\x01\xFF") * (n // 2)
            if len(probe) < n:
                probe += b"\x01"
            parts.append(probe)
    return b"".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize PR106 + Cool-Chic residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=0)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument(
        "--residual-mode", choices=("empty", "zero", "probe"), default="empty"
    )
    parser.add_argument("--default-scale", type=float, default=0.0)
    parser.add_argument("--skip-no-op-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"ERROR: --output-dir must be empty or not exist: {args.output_dir}", file=sys.stderr)
        return 2
    if args.residual_mode == "empty":
        residual_bytes = b""
    else:
        if args.n_frames <= 0:
            print("ERROR: --n-frames > 0 required when --residual-mode != empty", file=sys.stderr)
            return 2
        residual_bytes = build_cool_chic_residual_blob(
            n_frames=args.n_frames,
            n_levels=args.n_levels,
            mode=args.residual_mode,
            default_scale=args.default_scale,
        )
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family="cool_chic",
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "n_frames": args.n_frames,
            "n_levels": args.n_levels,
            "default_scale": args.default_scale,
            "pyramid_kind": "upsample_cascade_int8",
            "rationale": "Ladune et al. 2023 Cool-Chic hierarchical pyramid; INT8-quantised per level",
        },
    )
    if not args.skip_no_op_smoke and residual_bytes:
        smoke = run_no_op_detector_byte_mutation(
            archive_bytes=build.archive_bytes,
            expected_format_id=PR106_RESIDUAL_FORMAT_IDS["cool_chic"],
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
