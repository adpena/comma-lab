#!/usr/bin/env python3
"""Materialize a byte-closed PR106 + C3 residual sidecar candidate.

See ``submissions/pr106_c3_residual_sidecar/inflate.py`` for the wire format.
C3's conditional residual = frame-delta at quarter resolution (Kim et al.,
CVPR 2024). The residual is integrated (cumulative-sum across time) by the
inflate runtime then bilinear-upsampled to camera resolution.

Wire format (residual blob):
    per frame: 4B scale (float32) + int8 coefs at (CAMERA_H/4 * CAMERA_W/4 * 3).
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
QUARTER_H, QUARTER_W = CAMERA_H // 4, CAMERA_W // 4
RGB_CHANNELS = 3
PER_FRAME_BYTES = 4 + QUARTER_H * QUARTER_W * RGB_CHANNELS


def build_c3_residual_blob(*, n_frames: int, mode: str, default_scale: float = 0.0) -> bytes:
    if mode not in ("zero", "probe"):
        raise MaterializerError(f"unknown residual mode: {mode!r}")
    parts: list[bytes] = []
    block_bytes = QUARTER_H * QUARTER_W * RGB_CHANNELS
    for _ in range(n_frames):
        parts.append(struct.pack("<f", default_scale))
        if mode == "zero":
            parts.append(b"\x00" * block_bytes)
        else:  # probe
            probe = (b"\x01\xFF") * (block_bytes // 2)
            if len(probe) < block_bytes:
                probe += b"\x01"
            parts.append(probe)
    return b"".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize PR106 + C3 residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=0)
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
        residual_bytes = build_c3_residual_blob(
            n_frames=args.n_frames, mode=args.residual_mode, default_scale=args.default_scale
        )
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family="c3",
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "n_frames": args.n_frames,
            "per_frame_bytes": PER_FRAME_BYTES,
            "default_scale": args.default_scale,
            "conditioning_mode": "frame_delta_quarter_resolution_int8",
            "rationale": "Kim et al. 2024 C3 conditional residual; cumulative-sum integration in inflate",
        },
    )
    if not args.skip_no_op_smoke and residual_bytes:
        smoke = run_no_op_detector_byte_mutation(
            archive_bytes=build.archive_bytes,
            expected_format_id=PR106_RESIDUAL_FORMAT_IDS["c3"],
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
