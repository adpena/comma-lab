#!/usr/bin/env python3
"""Materialize a byte-closed PR106 + coordinate-MLP residual sidecar candidate.

See ``submissions/pr106_coord_mlp_residual_sidecar/inflate.py`` for the wire
format. The coord_mlp family (SIREN/NeRV/HNeRV/Cool-Chic/C3/Fourier-feat MLPs)
shares the Laplacian-smoothness prior: the residual is locally smooth so most
variation lives in a coarse representation. This scaffold encodes the residual
at 1/8 resolution with per-frame INT8 + float32 scale.

Wire format (residual blob):
    per frame: 4B scale (float32) + int8 coefs at (CAMERA_H/8 * CAMERA_W/8 * 3).
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
    repack_dense_as_sparse,
    run_no_op_detector_byte_mutation,
)
from tac.residual_basis.pr106_sidecar_packing import (  # noqa: E402
    PR106_RESIDUAL_FORMAT_IDS,
    sparse_family_name,
)

CAMERA_H, CAMERA_W = 874, 1164
DOWNSAMPLE_FACTOR = 8
LOW_H, LOW_W = CAMERA_H // DOWNSAMPLE_FACTOR, CAMERA_W // DOWNSAMPLE_FACTOR
RGB_CHANNELS = 3
PER_FRAME_BYTES = 4 + LOW_H * LOW_W * RGB_CHANNELS


def build_coord_mlp_residual_blob(
    *, n_frames: int, mode: str, default_scale: float = 0.0
) -> bytes:
    if mode not in ("zero", "probe"):
        raise MaterializerError(f"unknown residual mode: {mode!r}")
    block_bytes = LOW_H * LOW_W * RGB_CHANNELS
    parts: list[bytes] = []
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
        description="Materialize PR106 + coordinate-MLP residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=0)
    parser.add_argument(
        "--residual-mode", choices=("empty", "zero", "probe"), default="empty"
    )
    parser.add_argument("--default-scale", type=float, default=0.0)
    parser.add_argument("--skip-no-op-smoke", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=("dense", "sparse"),
        default="dense",
        help="Wire-format encoding: dense (0x14) or sparse PacketIR (0x24).",
    )
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
        residual_bytes = build_coord_mlp_residual_blob(
            n_frames=args.n_frames,
            mode=args.residual_mode,
            default_scale=args.default_scale,
        )
    is_sparse = args.encoding == "sparse"
    if is_sparse and residual_bytes:
        try:
            residual_bytes = repack_dense_as_sparse(
                family="coord_mlp",
                dense_residual_bytes=residual_bytes,
                n_frames=args.n_frames,
            )
        except MaterializerError as exc:
            print(f"ERROR: sparse repack failed: {exc}", file=sys.stderr)
            return 2
    family = sparse_family_name("coord_mlp") if is_sparse else "coord_mlp"
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family=family,
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "encoding": args.encoding,
            "n_frames": args.n_frames,
            "per_frame_bytes": PER_FRAME_BYTES,
            "downsample_factor": DOWNSAMPLE_FACTOR,
            "low_h": LOW_H,
            "low_w": LOW_W,
            "default_scale": args.default_scale,
            "rationale": (
                "Tancik et al. 2020 Fourier-features coord-MLP shared Laplacian-"
                "smoothness prior; 1/8 resolution INT8 + bicubic upsample in inflate."
            ),
        },
    )
    if not args.skip_no_op_smoke and residual_bytes:
        smoke = run_no_op_detector_byte_mutation(
            archive_bytes=build.archive_bytes,
            expected_format_id=PR106_RESIDUAL_FORMAT_IDS[family],
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
