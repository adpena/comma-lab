#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
    repack_dense_as_sparse,
    run_no_op_detector_byte_mutation,
)
from tac.residual_basis.pr106_sidecar_packing import (  # noqa: E402
    PR106_RESIDUAL_FORMAT_IDS,
    sparse_family_name,
)

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


def _build_l2_encoded_residual_blob(
    *,
    decoded_raw_path: Path,
    gt_raw_path: Path,
    n_frames: int,
    byte_budget: int,
    candidate_n_levels: tuple[int, ...],
    sparse_wire: bool,
    sparse_aware: bool = False,
    use_hinton_distilled_scorer: bool = False,
    use_saliency_masking: bool = False,
    per_level_top_k_budget: dict[int, int] | None = None,
    pose_only_mode: bool = False,
    pose_marginal_multiplier: float = 1.0,
) -> tuple[bytes, dict[str, float]]:
    """Run the L2 score-aware cool_chic encoder on decoded/GT raw frame streams.

    When ``sparse_aware=True``, the encoder's Lagrangian uses sparse-encoded
    byte cost and emits sparse-repacked bytes for the 0x21 family wrapper.
    """
    import numpy as np
    from tac.residual_basis import (
        dense_cool_chic_residual_blob_bytes,
        encode_cool_chic_residual_l2,
    )

    frame_bytes = CAMERA_H * CAMERA_W * RGB_CHANNELS
    decoded_total = decoded_raw_path.stat().st_size
    gt_total = gt_raw_path.stat().st_size
    if decoded_total % frame_bytes != 0 or gt_total % frame_bytes != 0:
        raise MaterializerError(
            "decoded_raw / gt_raw size not divisible by frame_bytes"
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
        dense_floors = [
            dense_cool_chic_residual_blob_bytes(n_to_use, int(levels))
            for levels in candidate_n_levels
            if 1 <= int(levels) <= 4
        ]
        if not dense_floors:
            raise MaterializerError("no valid --l2-candidate-n-levels")
        # The encoder chooses among dense oracle pyramids; sparse PacketIR
        # repack is the charged runtime wire and is budget-checked below.
        encoder_byte_budget = max(encoder_byte_budget, max(dense_floors))

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
        result = encode_cool_chic_residual_l2(
            decoded, gt,
            byte_budget=encoder_byte_budget,
            candidate_n_levels=candidate_n_levels,
            sparse_aware=sparse_aware,
            per_level_top_k_budget=per_level_top_k_budget,
            use_hinton_distilled_scorer=use_hinton_distilled_scorer,
            distilled_segnet=distilled_segnet,
            distilled_posenet=distilled_posenet,
            use_saliency_masking=use_saliency_masking,
            pose_only_mode=pose_only_mode,
            pose_marginal_multiplier=pose_marginal_multiplier,
        )
    except ValueError as exc:
        raise MaterializerError(str(exc)) from exc
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    diag = dict(result.diagnostics)
    diag["chosen_n_levels"] = float(result.n_levels_used)
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
        description="Materialize PR106 + Cool-Chic residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=0)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument(
        "--residual-mode", choices=("empty", "zero", "probe", "l2_encoded"), default="empty",
        help=(
            "empty/zero/probe = L1 scaffold modes; l2_encoded = L2 score-aware encoder "
            "(requires --decoded-raw + --gt-raw). L2 emits permanent score_claim=False invariants."
        ),
    )
    parser.add_argument("--default-scale", type=float, default=0.0)
    parser.add_argument(
        "--decoded-raw", type=Path, default=None,
        help="(l2_encoded only) Path to (N,874,1164,3) uint8 PR106 decoded raw frames",
    )
    parser.add_argument(
        "--gt-raw", type=Path, default=None,
        help="(l2_encoded only) Path to (N,874,1164,3) uint8 GT raw frames",
    )
    parser.add_argument(
        "--byte-budget", type=int, default=0,
        help="(l2_encoded only) Residual byte budget. Must be explicit; encoder enforces dense floor.",
    )
    parser.add_argument(
        "--l2-candidate-n-levels", type=int, nargs="+", default=(1, 2, 3),
        help="(l2_encoded only) Candidate pyramid depths to sweep",
    )
    parser.add_argument("--skip-no-op-smoke", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=("dense", "sparse"),
        default="dense",
        help="Wire-format encoding: dense (0x11) or sparse PacketIR (0x21).",
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
    parser.add_argument(
        "--per-level-top-k-budget",
        type=str,
        default=None,
        help=(
            "(l2_encoded only) Per-level top-K coefficient budget as comma-"
            "separated 'L:K' pairs (e.g. '0:1024,1:512,2:256,3:128'). The "
            "post-quantization int8 array at each level is truncated to keep "
            "only the K largest-magnitude non-zero coefficients (zeroing rest), "
            "enforcing a hard per-level byte budget. Operator decision "
            "2026-05-11: cool_chic 28MB minimum stream is too large for a 2KB "
            "global cap; per-level allocation enables sparse output that fits."
        ),
    )
    parser.add_argument(
        "--pose-only-mode",
        action="store_true",
        help=(
            "(l2_encoded only) Zero the SegNet term in the L2 Lagrangian. "
            "Per W DEFERRED criterion #4 + CLAUDE.md operating-point analysis."
        ),
    )
    parser.add_argument(
        "--pose-marginal-multiplier",
        type=float,
        default=1.0,
        help=(
            "(l2_encoded only) Multiply the pose Lagrangian term by this scalar. "
            "Default 1.0 = contest-faithful. Set to 2.79 for the PR106 r2 "
            "operating-point upweight."
        ),
    )
    args = parser.parse_args(argv)
    if args.pose_only_mode and args.residual_mode != "l2_encoded":
        print("ERROR: --pose-only-mode requires --residual-mode l2_encoded", file=sys.stderr)
        return 2
    if args.pose_marginal_multiplier <= 0.0:
        print(
            f"ERROR: --pose-marginal-multiplier={args.pose_marginal_multiplier} must be > 0",
            file=sys.stderr,
        )
        return 2
    per_level_top_k: dict[int, int] | None = None
    if args.per_level_top_k_budget is not None:
        per_level_top_k = {}
        try:
            for tok in args.per_level_top_k_budget.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                lvl_s, k_s = tok.split(":")
                per_level_top_k[int(lvl_s)] = int(k_s)
        except (ValueError, IndexError) as exc:
            print(
                f"ERROR: --per-level-top-k-budget parse error: {exc}; "
                "expected 'L:K[,L:K,...]' (e.g. '0:1024,1:512')",
                file=sys.stderr,
            )
            return 2
        if args.residual_mode != "l2_encoded":
            print(
                "ERROR: --per-level-top-k-budget requires --residual-mode l2_encoded",
                file=sys.stderr,
            )
            return 2
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
        print(f"ERROR: --output-dir must be empty or not exist: {args.output_dir}", file=sys.stderr)
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
                candidate_n_levels=tuple(args.l2_candidate_n_levels),
                sparse_wire=args.encoding == "sparse",
                sparse_aware=args.sparse_aware,
                use_hinton_distilled_scorer=args.use_hinton_distilled_scorer,
                use_saliency_masking=args.use_saliency_masking,
                per_level_top_k_budget=per_level_top_k,
                pose_only_mode=args.pose_only_mode,
                pose_marginal_multiplier=args.pose_marginal_multiplier,
            )
        except MaterializerError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
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
    is_sparse = args.encoding == "sparse"
    if is_sparse and residual_bytes and not args.sparse_aware:
        try:
            n_frames_for_repack = int(
                encoder_diagnostics.get("n_frames_encoded", args.n_frames)
            ) or args.n_frames
            residual_bytes = repack_dense_as_sparse(
                family="cool_chic",
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
    family = sparse_family_name("cool_chic") if is_sparse else "cool_chic"
    archive_zip, manifest_path, manifest, build = materialize_family_archive(
        family=family,
        pr106_archive=args.pr106_archive,
        residual_bytes=residual_bytes,
        output_dir=args.output_dir,
        extra={
            "residual_mode": args.residual_mode,
            "encoding": args.encoding,
            "sparse_aware_lagrangian": bool(args.sparse_aware),
            "n_frames": args.n_frames,
            "n_levels": args.n_levels,
            "default_scale": args.default_scale,
            "pyramid_kind": "upsample_cascade_int8",
            "rationale": "Ladune et al. 2023 Cool-Chic hierarchical pyramid; INT8-quantised per level",
            "l2_encoder_diagnostics": encoder_diagnostics,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
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
