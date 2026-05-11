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
    repack_dense_as_sparse,
    run_no_op_detector_byte_mutation,
    truncate_wavelet_dense_to_top_k,
)
from tac.residual_basis.pr106_sidecar_packing import (  # noqa: E402
    PR106_RESIDUAL_FORMAT_IDS,
    sparse_family_name,
)

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


def _build_l2_encoded_residual_blob(
    *,
    decoded_raw_path: Path,
    gt_raw_path: Path,
    n_frames: int,
    byte_budget: int,
    n_iterations: int,
    sparse_wire: bool,
    sparse_aware: bool = False,
    use_hinton_distilled_scorer: bool = False,
    use_saliency_masking: bool = False,
) -> tuple[bytes, dict[str, float]]:
    """Run the L2 score-aware wavelet encoder on decoded/GT raw frame streams.

    Emits permanent ``score_claim=False`` / ``promotion_eligible=False`` /
    ``ready_for_exact_eval_dispatch=False`` invariants per CLAUDE.md HNeRV
    parity discipline.

    When ``sparse_aware=True``, the encoder's Lagrangian uses the sparse-
    encoded byte cost in the rate term and emits sparse-repacked bytes
    directly (ready for the 0x20 wavelet_sparse family wrapper).
    """
    import numpy as np
    from tac.residual_basis import dense_wavelet_residual_blob_bytes, encode_wavelet_residual_l2

    frame_bytes = CAMERA_H * CAMERA_W * RGB_CHANNELS
    decoded_total = decoded_raw_path.stat().st_size
    gt_total = gt_raw_path.stat().st_size
    if decoded_total % frame_bytes != 0:
        raise MaterializerError(
            f"decoded raw file {decoded_raw_path} size {decoded_total} not divisible by frame_bytes"
        )
    if gt_total % frame_bytes != 0:
        raise MaterializerError(
            f"gt raw file {gt_raw_path} size {gt_total} not divisible by frame_bytes"
        )
    n_decoded = decoded_total // frame_bytes
    n_gt = gt_total // frame_bytes
    n_to_use = min(n_frames, n_decoded, n_gt) if n_frames > 0 else min(n_decoded, n_gt)
    if n_to_use <= 0:
        raise MaterializerError("no frames to encode")
    if n_to_use % 2 != 0:
        n_to_use -= 1
    if n_to_use <= 0:
        raise MaterializerError("need at least 2 frames after even-count truncation")

    decoded_mm = np.memmap(
        decoded_raw_path, dtype=np.uint8, mode="r",
        shape=(n_decoded, CAMERA_H, CAMERA_W, RGB_CHANNELS),
    )
    gt_mm = np.memmap(
        gt_raw_path, dtype=np.uint8, mode="r",
        shape=(n_gt, CAMERA_H, CAMERA_W, RGB_CHANNELS),
    )
    decoded = np.array(decoded_mm[:n_to_use])
    gt = np.array(gt_mm[:n_to_use])
    encoder_byte_budget = int(byte_budget)
    if sparse_wire and not sparse_aware:
        # The L2 encoder still emits the dense oracle stream; sparse PacketIR
        # repack happens immediately after and is the charged wire. Keep the
        # dense floor internal so sparse budget checks happen on the consumed
        # runtime format instead of blocking before sparse bytes exist.
        encoder_byte_budget = max(
            encoder_byte_budget,
            dense_wavelet_residual_blob_bytes(n_to_use),
        )
    distilled_segnet = None
    distilled_posenet = None
    if use_hinton_distilled_scorer or use_saliency_masking:
        from tac.residual_basis.hinton_distilled_scorer_surrogate import (
            ScorerSurrogateConfig,
            load_pretrained_distilled_scorer_pair,
        )

        config = ScorerSurrogateConfig.council_canonical()
        distilled_segnet, distilled_posenet = load_pretrained_distilled_scorer_pair(
            config=config
        )

    try:
        result = encode_wavelet_residual_l2(
            decoded, gt, byte_budget=encoder_byte_budget, n_iterations=n_iterations,
            sparse_aware=sparse_aware,
            use_hinton_distilled_scorer=use_hinton_distilled_scorer,
            distilled_segnet=distilled_segnet,
            distilled_posenet=distilled_posenet,
            use_saliency_masking=use_saliency_masking,
        )
    except ValueError as exc:
        raise MaterializerError(str(exc)) from exc
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    diag = dict(result.diagnostics)
    if sparse_wire and not sparse_aware:
        diag["requested_sparse_byte_budget"] = float(byte_budget)
        diag["dense_oracle_byte_budget"] = float(encoder_byte_budget)
    diag["sparse_aware_lagrangian"] = float(1.0 if sparse_aware else 0.0)
    diag["use_hinton_distilled_scorer"] = float(1.0 if use_hinton_distilled_scorer else 0.0)
    diag["use_saliency_masking"] = float(1.0 if use_saliency_masking else 0.0)
    diag["n_frames_encoded"] = float(n_to_use)
    return result.residual_bytes, diag


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
        choices=("empty", "zero", "probe", "l2_encoded"),
        default="empty",
        help=(
            "empty/zero/probe = L1 scaffold modes; "
            "l2_encoded = L2 score-aware encoder (requires --decoded-raw + --gt-raw). "
            "L2 emits permanent score_claim=False invariants."
        ),
    )
    parser.add_argument(
        "--default-scale",
        type=float,
        default=0.0,
        help="Per-band float32 scale (zero by default; non-zero requires explicit opt-in)",
    )
    parser.add_argument(
        "--decoded-raw", type=Path, default=None,
        help="(l2_encoded only) Path to (N,874,1164,3) uint8 PR106 decoded raw frames",
    )
    parser.add_argument(
        "--gt-raw", type=Path, default=None,
        help="(l2_encoded only) Path to (N,874,1164,3) uint8 ground-truth raw frames",
    )
    parser.add_argument(
        "--byte-budget", type=int, default=0,
        help=(
            "(l2_encoded only) Residual byte budget. Encoder enforces dense "
            "floor; smaller budgets refused."
        ),
    )
    parser.add_argument(
        "--l2-iterations", type=int, default=2,
        help="(l2_encoded only) Coordinate-descent outer-loop iterations",
    )
    parser.add_argument(
        "--skip-no-op-smoke", action="store_true", help="Skip the byte-mutation smoke"
    )
    parser.add_argument(
        "--encoding",
        choices=("dense", "sparse"),
        default="dense",
        help=(
            "Wire-format encoding for the residual blob. 'dense' (default) uses "
            "format_id 0x10 + per-frame fixed-size layout. 'sparse' repacks the "
            "same residual into format_id 0x20 with temporal-subsampled outer + "
            "RLE-of-zeros inner. Closes O's L2 wire-format ceiling."
        ),
    )
    parser.add_argument(
        "--sparse-top-k-per-frame",
        type=int,
        default=None,
        help=(
            "Sparse wavelet only: keep the largest-magnitude int8 coefficients "
            "per frame before PacketIR repack. This enforces real sparse-byte "
            "budgets without pretending the dense L2 proxy rate still applies."
        ),
    )
    parser.add_argument(
        "--sparse-aware",
        action="store_true",
        help=(
            "(l2_encoded + --encoding sparse) Activate score-aware encoder "
            "Lagrangian that optimizes for byte cost AFTER sparse PacketIR "
            "encoding directly (instead of dense + posthoc repack)."
        ),
    )
    parser.add_argument(
        "--use-hinton-distilled-scorer",
        action="store_true",
        help=(
            "(l2_encoded only; opt-in research mode) Replace the YUV6 MSE proxy "
            "with the Hinton-distilled SegNet+PoseNet surrogate. Per W's DEFERRED "
            "reactivation criterion #1."
        ),
    )
    parser.add_argument(
        "--use-saliency-masking",
        action="store_true",
        help=(
            "(l2_encoded only; opt-in research mode) Apply per-pixel saliency mask "
            "to the residual via score gradient on distilled scorer per Catalog #123. "
            "REQUIRES --use-hinton-distilled-scorer."
        ),
    )
    args = parser.parse_args(argv)
    if args.sparse_aware and args.encoding != "sparse":
        print("ERROR: --sparse-aware requires --encoding sparse", file=sys.stderr)
        return 2
    if args.sparse_aware and args.residual_mode != "l2_encoded":
        print("ERROR: --sparse-aware requires --residual-mode l2_encoded", file=sys.stderr)
        return 2
    if args.use_hinton_distilled_scorer and args.residual_mode != "l2_encoded":
        print("ERROR: --use-hinton-distilled-scorer requires --residual-mode l2_encoded", file=sys.stderr)
        return 2
    if args.use_saliency_masking and not args.use_hinton_distilled_scorer:
        print("ERROR: --use-saliency-masking requires --use-hinton-distilled-scorer (per Catalog #123)", file=sys.stderr)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(
            f"ERROR: --output-dir must be empty or not exist: {args.output_dir}",
            file=sys.stderr,
        )
        return 2
    encoder_diagnostics: dict[str, float] = {}
    if args.residual_mode == "empty":
        residual_bytes = b""
    elif args.residual_mode == "l2_encoded":
        if args.decoded_raw is None or args.gt_raw is None:
            print(
                "ERROR: --decoded-raw and --gt-raw required for --residual-mode l2_encoded",
                file=sys.stderr,
            )
            return 2
        if args.byte_budget <= 0:
            print(
                "ERROR: --byte-budget > 0 required for --residual-mode l2_encoded",
                file=sys.stderr,
            )
            return 2
        try:
            residual_bytes, encoder_diagnostics = _build_l2_encoded_residual_blob(
                decoded_raw_path=args.decoded_raw,
                gt_raw_path=args.gt_raw,
                n_frames=args.n_frames,
                byte_budget=args.byte_budget,
                n_iterations=args.l2_iterations,
                sparse_wire=args.encoding == "sparse",
                sparse_aware=args.sparse_aware,
                use_hinton_distilled_scorer=args.use_hinton_distilled_scorer,
                use_saliency_masking=args.use_saliency_masking,
            )
        except MaterializerError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
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
    is_sparse = args.encoding == "sparse"
    if args.sparse_top_k_per_frame is not None and not is_sparse:
        print("ERROR: --sparse-top-k-per-frame requires --encoding sparse", file=sys.stderr)
        return 2
    if is_sparse and residual_bytes and not args.sparse_aware:
        try:
            n_frames_for_repack = int(
                encoder_diagnostics.get("n_frames_encoded", args.n_frames)
            ) or args.n_frames
            if args.sparse_top_k_per_frame is not None:
                residual_bytes = truncate_wavelet_dense_to_top_k(
                    dense_residual_bytes=residual_bytes,
                    n_frames=n_frames_for_repack,
                    top_k_per_frame=args.sparse_top_k_per_frame,
                )
                encoder_diagnostics["sparse_top_k_per_frame"] = float(
                    args.sparse_top_k_per_frame
                )
            residual_bytes = repack_dense_as_sparse(
                family="wavelet",
                dense_residual_bytes=residual_bytes,
                n_frames=n_frames_for_repack,
            )
            if args.residual_mode == "l2_encoded" and len(residual_bytes) > args.byte_budget:
                print(
                    "ERROR: sparse residual bytes "
                    f"{len(residual_bytes)} exceed --byte-budget {args.byte_budget}",
                    file=sys.stderr,
                )
                return 2
            if args.residual_mode == "l2_encoded":
                encoder_diagnostics["sparse_residual_blob_bytes"] = float(len(residual_bytes))
                encoder_diagnostics["sparse_rate_term_is_posthoc_not_encoder_loss"] = 1.0
        except MaterializerError as exc:
            print(f"ERROR: sparse repack failed: {exc}", file=sys.stderr)
            return 2
    elif is_sparse and residual_bytes and args.sparse_aware:
        encoder_diagnostics["sparse_residual_blob_bytes"] = float(len(residual_bytes))
        encoder_diagnostics["sparse_rate_term_is_posthoc_not_encoder_loss"] = 0.0
    family = sparse_family_name("wavelet") if is_sparse else "wavelet"
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family=family,
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "encoding": args.encoding,
            "sparse_aware_lagrangian": bool(args.sparse_aware),
            "default_scale": args.default_scale,
            "n_frames": args.n_frames,
            "per_frame_bytes": PER_FRAME_BYTES,
            "wavelet": "haar_db1_single_level",
            "rationale": (
                "single-level Haar matches the numpy_inverse_dwt scaffold; "
                "multi-level extensions belong to the L2-promotion bolt-on."
            ),
            "l2_encoder_diagnostics": encoder_diagnostics,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    if not args.skip_no_op_smoke:
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
