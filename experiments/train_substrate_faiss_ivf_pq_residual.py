#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# NO_GRAD_WAIVED:faiss_substrate_training_is_kmeans_codebook_fitting_not_gradient_descent_no_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:faiss_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:faiss_substrate_training_is_kmeans_not_a_pytorch_gradient_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_only_in_smoke_full_path_fits_codebook_on_real_contest_video_residuals_via_decode_video_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

"""train_substrate_faiss_ivf_pq_residual — codebook-fitting trainer (Path 3 #I).

Path 3 candidate #I Faiss IVF-PQ residual codec.

MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: ``_full_main`` is UNBLOCKED, but —
**honestly** — NOT via the gradient-reachable MLX score-aware harness. Per the
operator directive ("faiss_ivf_pq may need codebook-FITTING hook not gradient —
implement honestly"): this substrate's distinguishing primitive is PRODUCT
QUANTIZATION (Jégou-Douze-Schmid 2011), whose "training" is K-MEANS CODEBOOK
FITTING on per-pair residual tiles — structurally NOT gradient descent. Routing
it through the gradient harness would be mathematically wrong (PQ codebook
assignment is argmin-over-centroids, non-differentiable). So this ``_full_main``
implements the substrate's ACTUAL training algorithm: a codebook-fitting trainer
that decodes the real contest video, computes per-pair residuals against a
per-pair-mean base, tiles them, fits the PQ codebook via deterministic K-means
(``numpy_reference.train_pq_codebook``), encodes the per-pair codeword stream,
builds the byte-deterministic FAISSPQ1 archive, and measures round-trip
reconstruction MSE.

## Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL_BECAUSE_SERVES: real-video decode via ``tac.data.decode_video``
  (Catalog #114); byte-deterministic archive grammar (``archive.build_archive_bytes``);
  non-promotable Provenance markers (Catalog #192/#317/#341).
- FORK_BECAUSE_PRINCIPLED_MISMATCH (this substrate's UNIQUE primitive AND its
  training algorithm): PQ codebook fitting via K-means is NOT a gradient path;
  the canonical gradient-reachable ``mlx_score_aware`` harness does NOT apply.
  The codebook-fitting trainer here IS the canonical-for-this-method training
  loop per UNIQUE-AND-COMPLETE-PER-METHOD.

## Dispatch gating (Catalog #325)

MLX-LOCAL ($0 M5 Max; numpy K-means runs anywhere — no MLX/CUDA required for
fitting). research_only=True; dispatch_enabled=False; output is non-promotable
``[macOS-MLX research-signal]`` per Catalog #192/#341. The per-substrate
symposium (Catalog #325), MLX/PyTorch parity gate (Catalog #1265), Catalog #319
deliverability_proof, and paired [contest-CUDA]+[contest-CPU] anchor remain
DEFERRED to the paid-dispatch path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Catalog #151 manifest (ast.AnnAssign per Catalog #168). --full requires the
# real contest video at --video-path (required_input_file=True).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "FAISS_IVF_PQ_OUTPUT_DIR",
        "rationale": (
            "Output dir for the codebook-fitting artifacts: FAISSPQ1 archive + "
            "fitting stats JSON + observability surface (NOT /tmp per Catalog #208)."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "FAISS_IVF_PQ_VIDEO_PATH",
        "rationale": (
            "Real contest video the PQ codebook is fit against (Catalog #114; "
            "real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faiss IVF-PQ residual codec codebook-fitting trainer (Path 3 #I).",
    )
    parser.add_argument("--smoke", action="store_true", help="run L0 SCAFFOLD smoke")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Fit the PQ codebook on real contest-video residuals (the "
        "substrate's actual training algorithm; NOT the gradient harness).",
    )
    parser.add_argument(
        "--validate-archive-roundtrip",
        action="store_true",
        help="validate synthetic archive build+parse round-trip (smoke)",
    )
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--m-sub-quantizers", type=int, default=4)
    parser.add_argument("--ksub-codebook-size", type=int, default=256)
    parser.add_argument("--tile-h", type=int, default=96)
    parser.add_argument("--tile-w", type=int, default=128)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output dir (NOT /tmp)."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full codebook fitting (Catalog #114).",
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """L0 SCAFFOLD smoke entry point (synthetic archive round-trip)."""
    print(
        "[faiss_ivf_pq_residual L0 SCAFFOLD smoke] "
        "[macOS-MLX research-signal] "
        "research_only=True dispatch_enabled=False score_claim=False"
    )
    from tac.substrates.faiss_ivf_pq_residual import (
        FAISSPQ1_HEADER_SIZE,
        FAISSPQ1_MAGIC,
        FaissIVFPQResidualConfig,
        build_archive_bytes,
        estimate_archive_bytes,
        estimate_per_pair_codeword_bytes_raw,
        parse_archive,
    )

    print(f"[L0] FAISSPQ1_MAGIC={FAISSPQ1_MAGIC!r}; HEADER_SIZE={FAISSPQ1_HEADER_SIZE}")

    if args.validate_archive_roundtrip:
        import numpy as np

        cfg = FaissIVFPQResidualConfig(
            m_sub_quantizers=2,
            ksub_codebook_size=8,
            tile_h=192,
            tile_w=256,
            num_pairs=4,
        )
        raw_bytes_per_pair = estimate_per_pair_codeword_bytes_raw(cfg)
        est_archive = estimate_archive_bytes(cfg)
        print(f"[L0] estimated per-pair codeword bytes (raw): {raw_bytes_per_pair}")
        print(f"[L0] estimated archive bytes: {est_archive}")
        rng = np.random.default_rng(42)
        codebook = rng.standard_normal(
            (cfg.m_sub_quantizers, cfg.ksub_codebook_size, cfg.sub_dim)
        ).astype(np.float32)
        codewords = rng.integers(
            0,
            cfg.ksub_codebook_size,
            size=(cfg.num_pairs, cfg.tiles_per_pair, cfg.m_sub_quantizers),
            dtype=np.uint16,
        )
        data = build_archive_bytes(
            codebook, codewords, tile_h=cfg.tile_h, tile_w=cfg.tile_w
        )
        arch = parse_archive(data)
        assert np.array_equal(arch.codebook, codebook)
        assert np.array_equal(arch.per_pair_codewords, codewords)
        print(f"[L0] archive round-trip OK: actual_bytes={len(data)}")

    print("[L0] smoke complete; NO paid dispatch authorized")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Codebook-fitting training on real contest-video residuals (honest path).

    NOT the gradient-reachable harness — PQ codebook assignment is argmin (non-
    differentiable). The training algorithm is deterministic K-means codebook
    fitting on per-pair residual tiles (the substrate's actual learning
    procedure). Emits a byte-deterministic FAISSPQ1 archive + round-trip
    reconstruction MSE + non-promotable Provenance markers.
    """
    import hashlib
    import time

    import numpy as np

    from tac.data import decode_video
    from tac.substrates.faiss_ivf_pq_residual import (
        FaissIVFPQResidualConfig,
        build_archive_bytes,
    )
    from tac.substrates.faiss_ivf_pq_residual.mlx_renderer import (
        EVAL_HW,
        EVIDENCE_TAG,
        LANE_ID,
    )
    from tac.substrates.faiss_ivf_pq_residual.numpy_reference import (
        encode_per_pair_residual,
        frame_to_tiles_nhwc,
        pq_reconstruct_tile_vectors,
        tiles_to_frame_nhwc,
        train_pq_codebook,
    )

    if args.output_dir is None:
        raise SystemExit(
            "--output-dir is required for --full codebook fitting "
            "(Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS)."
        )
    out_dir = Path(args.output_dir)
    out_dir_str = str(out_dir.resolve())
    if out_dir_str.startswith(("/tmp/", "/private/tmp/")):
        raise SystemExit(
            f"output-dir {out_dir} under /tmp per CLAUDE.md FORBIDDEN_PATTERN "
            "'Forbidden /tmp paths in any persisted artifact'"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FaissIVFPQResidualConfig(
        m_sub_quantizers=int(args.m_sub_quantizers),
        ksub_codebook_size=int(args.ksub_codebook_size),
        tile_h=int(args.tile_h),
        tile_w=int(args.tile_w),
        num_pairs=int(args.num_pairs),
    )
    started = time.perf_counter()

    # 1. Decode real contest video into per-pair frames at EVAL_HW (Catalog #114).
    frames = decode_video(
        args.video_path,
        target_h=EVAL_HW[0],
        target_w=EVAL_HW[1],
        max_frames=2 * cfg.num_pairs,
    )
    if len(frames) < 2 * cfg.num_pairs:
        raise SystemExit(
            f"decoded {len(frames)} frames; need {2 * cfg.num_pairs} for "
            f"{cfg.num_pairs} pairs"
        )
    gt = np.stack([f.numpy() for f in frames[: 2 * cfg.num_pairs]], axis=0).astype(
        np.float32
    )
    gt_pairs = gt.reshape(cfg.num_pairs, 2, EVAL_HW[0], EVAL_HW[1], 3) / 255.0

    # 2. Per-pair residual against the per-pair mean base (PQ encodes residual).
    #    The current FAISSPQ1 grammar stores one codeword stream per pair, so
    #    the fitting target is frame_0 residual as the deterministic proxy.
    #    A two-stream grammar is a separate archive-format change, not a hidden
    #    side channel inside this trainer.
    bases = gt_pairs.mean(axis=1)  # (P, H, W, 3)
    residual_0 = gt_pairs[:, 0] - bases

    # 3. Tile the residuals; stack all per-pair tiles as K-means training data.
    all_tiles = []
    per_pair_tiles_0 = []
    for p in range(cfg.num_pairs):
        t0 = frame_to_tiles_nhwc(residual_0[p], tile_h=cfg.tile_h, tile_w=cfg.tile_w)
        per_pair_tiles_0.append(t0)
        all_tiles.append(t0)
    train_data = np.concatenate(all_tiles, axis=0)  # (P*tiles_per_pair, tile_dim)

    # 4. Fit the PQ codebook via deterministic K-means (the actual training).
    codebook = train_pq_codebook(
        train_data,
        m_sub_quantizers=cfg.m_sub_quantizers,
        ksub_codebook_size=cfg.ksub_codebook_size,
        num_kmeans_iters=int(args.kmeans_iters),
        seed=int(args.seed),
    )

    # 5. Encode the per-pair codeword stream + measure round-trip residual MSE.
    per_pair_codewords = np.zeros(
        (cfg.num_pairs, cfg.tiles_per_pair, cfg.m_sub_quantizers), dtype=np.uint16
    )
    recon_mse_sum = 0.0
    for p in range(cfg.num_pairs):
        indices = encode_per_pair_residual(per_pair_tiles_0[p], codebook)
        per_pair_codewords[p] = indices
        recon_tiles = pq_reconstruct_tile_vectors(codebook, indices.astype(np.int64))
        recon_frame = tiles_to_frame_nhwc(
            recon_tiles, frame_h=EVAL_HW[0], frame_w=EVAL_HW[1],
            tile_h=cfg.tile_h, tile_w=cfg.tile_w,
        )
        recon_mse_sum += float(np.mean((recon_frame - residual_0[p]) ** 2))
    recon_mse = recon_mse_sum / cfg.num_pairs

    # 6. Build the byte-deterministic FAISSPQ1 archive (non-promotable markers).
    meta = {
        "lane_id": LANE_ID,
        "training_algorithm": "kmeans_codebook_fitting_not_gradient",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": EVIDENCE_TAG,
    }
    archive_bytes = build_archive_bytes(
        codebook, per_pair_codewords, tile_h=cfg.tile_h, tile_w=cfg.tile_w, meta=meta
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    elapsed = time.perf_counter() - started

    stats = {
        "schema_version": "faiss_ivf_pq_residual_codebook_fit_stats_v1",
        "substrate_id": "faiss_ivf_pq_residual",
        "lane_id": LANE_ID,
        "training_algorithm": "kmeans_codebook_fitting_not_gradient",
        "config": {
            "m_sub_quantizers": cfg.m_sub_quantizers,
            "ksub_codebook_size": cfg.ksub_codebook_size,
            "tile_h": cfg.tile_h,
            "tile_w": cfg.tile_w,
            "num_pairs": cfg.num_pairs,
            "kmeans_iters": int(args.kmeans_iters),
            "seed": int(args.seed),
        },
        "residual_recon_mse_mean": recon_mse,
        "archive_member_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "elapsed_seconds": elapsed,
        # Non-promotable markers per Catalog #127/#192/#317/#341.
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "result_review_blockers": [
            "codebook_fit_residual_mse_proxy_not_contest_score",
            "no_score_aware_loss_no_segnet_no_posenet_feedback",
            "no_paired_contest_cpu_plus_cuda_anchor",
            "per_substrate_symposium_pending_catalog_325",
        ],
    }
    (out_dir / "codebook_fit_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[faiss_ivf_pq_residual:_full_main] DONE (k-means codebook fitting) "
        f"num_pairs={cfg.num_pairs} residual_recon_mse={recon_mse:.6f} "
        f"archive_bytes={len(archive_bytes):,} sha256={archive_sha[:16]}… "
        f"wall={elapsed:.1f}s [macOS-MLX research-signal] non-promotable"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    _build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
