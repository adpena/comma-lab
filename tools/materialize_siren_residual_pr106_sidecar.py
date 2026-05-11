#!/usr/bin/env python3
"""Materialize a byte-closed PR106 + SIREN residual sidecar candidate.

See ``submissions/pr106_siren_residual_sidecar/inflate.py`` for the wire format.
SIREN's coordinate-MLP with sinusoidal activations is canonically encoded
here as a SPARSE FREQUENCY-DOMAIN coefficient set. Each coefficient is a
tuple: (frame_idx u16, k_row i16, k_col i16, channel u8, real i8, imag i8) = 9B.

This is the smallest-byte SIREN-compatible residual representation: the
inflate runtime places each coef into the 2D-FFT spectrum then inverse-FFTs
per frame per channel.

Default mode='empty' emits empty residual (scaffold-readiness archive).
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

PER_COEF_BYTES = 9  # u16 + i16 + i16 + u8 + i8 + i8


def build_siren_residual_blob(
    *, n_coefs: int, n_frames: int, mode: str, default_scale: float = 0.0, max_k: int = 32
) -> bytes:
    """Build a sparse-FFT-coefs blob. Each coef is 9B; header = 4B scale + 2B count = 6B.

    mode='zero': all-zero coefs (identity bolt-on; scale also zero).
    mode='probe': low-frequency probe coefs at (k_row=0, k_col in [1..n_coefs], frame 0, channel 0, real=1, imag=0).
    """
    if mode not in ("zero", "probe"):
        raise MaterializerError(f"unknown residual mode: {mode!r}")
    if n_coefs < 0:
        raise MaterializerError(f"n_coefs must be >= 0; got {n_coefs}")
    if n_coefs > 0xFFFF:
        raise MaterializerError(f"n_coefs={n_coefs} exceeds 16-bit cap 65535")
    header = struct.pack("<fH", default_scale, n_coefs)
    coefs: list[bytes] = []
    for i in range(n_coefs):
        frame_idx = i % max(n_frames, 1)
        k_row = 0
        k_col = (i % max_k) + 1
        channel = 0
        if mode == "zero":
            real_q, imag_q = 0, 0
        else:  # probe
            real_q, imag_q = 1, 0
        coefs.append(
            struct.pack("<HhhBbb", frame_idx, k_row, k_col, channel, real_q, imag_q)
        )
    return header + b"".join(coefs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize PR106 + SIREN residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=1200)
    parser.add_argument(
        "--n-coefs",
        type=int,
        default=0,
        help="Number of sparse FFT coefs (0 = empty residual scaffold mode).",
    )
    parser.add_argument(
        "--residual-mode", choices=("empty", "zero", "probe"), default="empty"
    )
    parser.add_argument("--default-scale", type=float, default=0.0)
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument("--skip-no-op-smoke", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=("dense", "sparse"),
        default="dense",
        help="Wire-format encoding: dense (0x13) or sparse PacketIR (0x23).",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"ERROR: --output-dir must be empty or not exist: {args.output_dir}", file=sys.stderr)
        return 2
    if args.residual_mode == "empty":
        residual_bytes = b""
    else:
        if args.n_coefs <= 0:
            print(
                "ERROR: --n-coefs > 0 required when --residual-mode != empty",
                file=sys.stderr,
            )
            return 2
        residual_bytes = build_siren_residual_blob(
            n_coefs=args.n_coefs,
            n_frames=args.n_frames,
            mode=args.residual_mode,
            default_scale=args.default_scale,
            max_k=args.max_k,
        )
    is_sparse = args.encoding == "sparse"
    if is_sparse and residual_bytes:
        try:
            residual_bytes = repack_dense_as_sparse(
                family="siren",
                dense_residual_bytes=residual_bytes,
                n_frames=args.n_frames,
            )
        except MaterializerError as exc:
            print(f"ERROR: sparse repack failed: {exc}", file=sys.stderr)
            return 2
    family = sparse_family_name("siren") if is_sparse else "siren"
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family=family,
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "encoding": args.encoding,
            "n_frames": args.n_frames,
            "n_coefs": args.n_coefs,
            "per_coef_bytes": PER_COEF_BYTES,
            "default_scale": args.default_scale,
            "max_k": args.max_k,
            "rationale": (
                "Sitzmann et al. 2020 SIREN sinusoidal coord-MLP encoded as "
                "sparse 2D FFT coefs; inverse-FFT in inflate."
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
