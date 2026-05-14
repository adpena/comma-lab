#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
    repack_dense_as_sparse,
    run_no_op_detector_byte_mutation,
)
from tac.residual_basis.pr106_sidecar_packing import (  # noqa: E402
    PR106_RESIDUAL_FORMAT_IDS,
    sparse_family_name,
)

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
    pose_only_mode: bool = False,
    pose_marginal_multiplier: float = 1.0,
) -> tuple[bytes, dict[str, float]]:
    """Run the L2 score-aware encoder on decoded/GT raw frame streams.

    Both paths must point at ``(N, 874, 1164, 3)`` uint8 raw files (the
    canonical PR106 inflate output format). Returns ``(residual_bytes,
    encoder_diagnostics)``. The encoder emits permanent
    ``score_claim=False`` / ``promotion_eligible=False`` /
    ``ready_for_exact_eval_dispatch=False`` invariants per CLAUDE.md
    HNeRV parity discipline.

    When ``sparse_aware=True``, the encoder's Lagrangian uses the sparse-
    encoded byte cost in the rate term (instead of the dense byte cost
    + posthoc repack). The returned bytes are sparse-repacked directly by
    the encoder and ready for the 0x22 (c3_sparse) family wrapper.
    """
    import numpy as np  # local import keeps the materializer's main path light
    from tac.residual_basis import dense_c3_residual_blob_bytes, encode_c3_residual_l2

    frame_bytes = CAMERA_H * CAMERA_W * RGB_CHANNELS
    decoded_total = decoded_raw_path.stat().st_size
    gt_total = gt_raw_path.stat().st_size
    if decoded_total % frame_bytes != 0:
        raise MaterializerError(
            f"decoded raw file {decoded_raw_path} size {decoded_total} not divisible by frame_bytes {frame_bytes}"
        )
    if gt_total % frame_bytes != 0:
        raise MaterializerError(
            f"gt raw file {gt_raw_path} size {gt_total} not divisible by frame_bytes {frame_bytes}"
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
        # The encoder emits the dense semantic oracle; the charged bytes are
        # enforced after sparse PacketIR repack below.
        encoder_byte_budget = max(
            encoder_byte_budget,
            dense_c3_residual_blob_bytes(n_to_use),
        )

    distilled_segnet = None
    distilled_posenet = None
    if use_hinton_distilled_scorer or use_saliency_masking:
        # Per W reactivation criteria #1+#2 + Catalog #123: load the
        # Hinton-distilled scorer surrogate so the L2 encoder's Lagrangian
        # uses the score-aware proxy instead of the YUV6 MSE proxy. Both
        # `use_hinton_distilled_scorer` and `use_saliency_masking` require
        # the surrogate (saliency is computed via score gradient on the
        # surrogate per Catalog #123).
        from tac.residual_basis.hinton_distilled_scorer_surrogate import (
            ScorerSurrogateConfig,
            load_pretrained_distilled_scorer_pair,
        )

        config = ScorerSurrogateConfig.council_canonical()
        distilled_segnet, distilled_posenet = load_pretrained_distilled_scorer_pair(
            config=config
        )

    try:
        result = encode_c3_residual_l2(
            decoded, gt,
            byte_budget=encoder_byte_budget,
            n_iterations=n_iterations,
            sparse_aware=sparse_aware,
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
        description="Materialize PR106 + C3 residual sidecar candidate"
    )
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=0)
    parser.add_argument(
        "--residual-mode", choices=("empty", "zero", "probe", "l2_encoded"), default="empty",
        help=(
            "empty/zero/probe = L1 scaffold modes; l2_encoded = run the L2 "
            "score-aware encoder (requires --decoded-raw + --gt-raw). The "
            "encoder emits permanent score_claim=False / promotion_eligible="
            "False / ready_for_exact_eval_dispatch=False invariants."
        ),
    )
    parser.add_argument("--default-scale", type=float, default=0.0)
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
        help="(l2_encoded only) Residual byte budget. Must be explicit; encoder enforces dense floor.",
    )
    parser.add_argument(
        "--l2-iterations", type=int, default=2,
        help="(l2_encoded only) Coordinate-descent outer-loop iterations",
    )
    parser.add_argument("--skip-no-op-smoke", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=("dense", "sparse"),
        default="dense",
        help="Wire-format encoding: dense (0x12) or sparse PacketIR (0x22).",
    )
    parser.add_argument(
        "--sparse-aware",
        action="store_true",
        help=(
            "(l2_encoded + --encoding sparse) Activate score-aware encoder "
            "Lagrangian that optimizes for byte cost AFTER sparse PacketIR "
            "encoding directly (instead of dense + posthoc repack). The "
            "encoder Lagrangian uses sparse byte cost in the rate term so "
            "coordinate descent picks parameterizations that maximize zero "
            "coefficient count for a given seg+pose proxy."
        ),
    )
    parser.add_argument(
        "--use-hinton-distilled-scorer",
        action="store_true",
        help=(
            "(l2_encoded only; opt-in research mode) Replace the YUV6 MSE proxy "
            "in the L2 Lagrangian with Hinton-distilled SegNet+PoseNet surrogate "
            "outputs (KL-distill at T=2.0 on seg + MSE on first-6 pose dims). "
            "Per W's DEFERRED reactivation criterion #1, this is the canonical "
            "fix for the YUV6 MSE proxy mismatch that produced 0.2066 (noise) "
            "on the c3 sparse-empty dispatch."
        ),
    )
    parser.add_argument(
        "--use-saliency-masking",
        action="store_true",
        help=(
            "(l2_encoded only; opt-in research mode) Apply per-pixel saliency "
            "mask to the residual before encoding. Pixels with low score-aware "
            "saliency (computed via score gradient on distilled scorer per "
            "Catalog #123) are zeroed; bytes saved by RLE-of-zeros runs. "
            "REQUIRES --use-hinton-distilled-scorer (saliency uses surrogate)."
        ),
    )
    parser.add_argument(
        "--pose-only-mode",
        action="store_true",
        help=(
            "(l2_encoded only) Zero the SegNet term in the L2 Lagrangian; "
            "encoder solves only for pose-axis residual improvement. Per W "
            "DEFERRED reactivation criterion #4 + CLAUDE.md operating-point "
            "analysis: at PR106 r2 frontier, pose marginal-value-per-byte is "
            "2.79x SegNet's so a residual that improves pose without breaking "
            "seg dominates marginal-score-per-byte."
        ),
    )
    parser.add_argument(
        "--pose-marginal-multiplier",
        type=float,
        default=1.0,
        help=(
            "(l2_encoded only) Multiply the pose Lagrangian term by this scalar. "
            "Default 1.0 = contest-faithful. Set to 2.79 (= "
            "PR106_R2_POSE_MARGINAL_MULTIPLIER) when running pose-only mode "
            "at the PR106 r2 operating point."
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
                n_iterations=args.l2_iterations,
                sparse_wire=args.encoding == "sparse",
                sparse_aware=args.sparse_aware,
                use_hinton_distilled_scorer=args.use_hinton_distilled_scorer,
                use_saliency_masking=args.use_saliency_masking,
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
        residual_bytes = build_c3_residual_blob(
            n_frames=args.n_frames, mode=args.residual_mode, default_scale=args.default_scale
        )
    is_sparse = args.encoding == "sparse"
    # If --sparse-aware, the encoder already emitted sparse-repacked bytes; skip
    # the posthoc repack to avoid double-encoding the same data.
    if is_sparse and residual_bytes and not args.sparse_aware:
        try:
            # n_frames for the repack: if the encoder ran in l2_encoded mode the
            # actual encoded frame count lives in encoder_diagnostics; otherwise
            # use args.n_frames.
            n_frames_for_repack = int(
                encoder_diagnostics.get("n_frames_encoded", args.n_frames)
            ) or args.n_frames
            residual_bytes = repack_dense_as_sparse(
                family="c3",
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
    family = sparse_family_name("c3") if is_sparse else "c3"
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
            "per_frame_bytes": PER_FRAME_BYTES,
            "default_scale": args.default_scale,
            "conditioning_mode": "frame_delta_quarter_resolution_int8",
            "rationale": "Kim et al. 2024 C3 conditional residual; cumulative-sum integration in inflate",
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
