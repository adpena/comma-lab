# SPDX-License-Identifier: MIT
"""Z6 predictive-coding MLX-local trainer — L1 PROMOTION: REAL contest video training.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Per Path 3 candidate #D L1 promotion (operator 2026-05-26 cascade: per-substrate
symposium PROCEED_WITH_REVISIONS at
`.omx/research/path_3_d_z6_per_substrate_symposium_l1_promotion_20260526.md`).
The L1 promotion converts the L0 SCAFFOLD's synthetic MSE proxy training into
INFRASTRUCTURE-CONVERGENCE-VERIFICATION on REAL contest video frames decoded
via canonical `tac.data.decode_video` pyav helper, with canonical EMA decay
0.997 per CLAUDE.md "EMA -- NON-NEGOTIABLE".

Per Contrarian sextet dissent + Yousfi sextet dissent: the L1 surface is
INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim. The score-aware
Lagrangian via SegNet/PoseNet remains the PyTorch sister L2 promotion path
per Catalog #164 + #226. The L1 promotion is non-promotable per Catalog
#287/#192/#317/#341 throughout.

Purpose
-------

Train the Z6 FiLM-conditioned next-frame predictor + encoder + decoder on Apple
Silicon (MLX) against REAL contest video frames at $0, with EMA shadow weights
as the canonical inference checkpoint, then export to a contest Z6PCWM1 archive
that:

1. Loads byte-stably into the PyTorch sister substrate
   :class:`tac.substrates.time_traveler_l5_z6.architecture.Z6PredictiveCodingSubstrate`
2. Inflates via :mod:`tac.substrates.time_traveler_l5_z6.inflate` to contest-grade raw frames
3. Passes the canonical #1265 contest-equivalence gate (or Z6 sister gate when
   landed per per-substrate symposium operator-routable #2) before any paid CUDA dispatch

L1 PROMOTION scope (per per-substrate symposium binding revisions)
------------------------------------------------------------------

- REAL contest video frames via `tac.data.decode_video` (per Yousfi dissent + symposium step 2)
- Canonical EMA decay=0.997 + save EMA shadow as inference checkpoint (per Catalog #2)
- Multi-epoch convergence (default 50ep at smoke subset; 50-100ep at contest scale)
- Single-layer FiLM predictor preserved (multi-layer DEFERRED per L0 SCAFFOLD scope guard)
- MSE proxy loss on real frames + residual L2 Lagrangian (Rao-Ballard term)
- Pinned ego-motion buffer preserved (PoseNet-derived ego-motion DEFERRED to L2 per Yousfi)
- Output: .pt + EMA shadow .pt + Z6PCWM1 archive + training_manifest.json

Non-promotable canonical contract
---------------------------------

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable:
- All outputs tagged ``[macOS-MLX research-signal]``
- ``score_claim=False`` / ``promotion_eligible=False`` / ``ready_for_exact_eval_dispatch=False``
- Operator routes paid-CUDA promotion via existing PyTorch trainer
  ``experiments/train_substrate_time_traveler_l5_z6.py`` + ``operator_authorize.py``

Catalog #311 ego-motion conditioning honored
--------------------------------------------

The MLX trainer pins a randomized ego_motion buffer (one-time draw with fixed
seed) AND threads it through the FiLM predictor's modulation pipeline. The
ego-motion gradient surface IS exercised, satisfying the canonical Catalog
#311 ego-motion-conditioning requirement at the structural level. PoseNet-
derived ego-motion is the L2 promotion's responsibility per Yousfi sextet
dissent (`.omx/research/path_3_d_z6_per_substrate_symposium_l1_promotion_20260526.md`).

Catalog #2 EMA NON-NEGOTIABLE honored
-------------------------------------

L1 promotion adds MLX-native EMA with decay=0.997 (canonical Quantizr PR101
anchor per CLAUDE.md "EMA -- NON-NEGOTIABLE"). The EMA shadow is exported
separately as ``z6_mlx_state_dict_ema_shadow.pt`` and the Z6PCWM1 archive
uses the EMA shadow weights (not the live training weights) per CLAUDE.md
"NEVER call ema.apply(model) inside train_epoch" + "Inference / archive bytes
come from ema.state_dict()".

Usage
-----

L1 smoke (real video; small subset; fast convergence verification):

    .venv/bin/python experiments/train_substrate_z6_predictive_coding_mlx.py \\
        --smoke --num-pairs 50 --epochs 5 \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z6_mlx_l1_smoke_$(date -u +%Y%m%dT%H%M%SZ)/

L1 contest-scale (real video; 600 pairs; 50-100 epochs; ~1-3h on M-series):

    .venv/bin/python experiments/train_substrate_z6_predictive_coding_mlx.py \\
        --num-pairs 600 --epochs 50 \\
        --video-path upstream/videos/0.mkv \\
        --output-height 384 --output-width 512 \\
        --output-dir experiments/results/z6_mlx_l1_full_$(date -u +%Y%m%dT%H%M%SZ)/

The trainer produces:
- ``z6_mlx_state_dict.pt`` (PyTorch state_dict via #1251 bridge; LIVE weights)
- ``z6_mlx_state_dict_ema_shadow.pt`` (EMA shadow; canonical inference checkpoint per Catalog #2)
- ``0.bin`` (Z6PCWM1 archive built from EMA shadow weights; ready for #1265 gate)
- ``training_manifest.json`` (canonical Provenance + per-epoch metrics + EMA decay)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #1: this
# trainer is MLX-only by design; PyTorch CUDA dispatch is the operator's
# explicit routing decision via the canonical sister trainer.

# Ensure repo root on sys.path so canonical Z6 imports resolve under
# README's PYTHONPATH=src:upstream entry-point.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _require_mlx() -> None:
    try:
        import mlx.core
        import mlx.nn  # noqa: F401
    except ImportError as exc:
        print(
            "[z6-mlx-trainer] FATAL: MLX not installed. This is an Apple-Silicon-"
            "only trainer. Install via `pip install mlx`. The PyTorch sister "
            "trainer experiments/train_substrate_time_traveler_l5_z6.py covers "
            "the CUDA / paid-dispatch path.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force smoke-only minimal config (overrides --num-pairs/--epochs).",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=8,
        help="Per-pair count (default 8 for smoke; 600 for contest).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epoch count (default 3 for smoke).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=24,
        help="Per-pair latent dimensionality (default 24).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate (default 1e-3).",
    )
    parser.add_argument(
        "--lambda-residual",
        type=float,
        default=1.0,
        help="Residual L2 Lagrangian weight (default 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic-data + ego-motion buffer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dir (state_dict .pt + Z6PCWM1 0.bin + training_manifest.json).",
    )
    parser.add_argument(
        "--predictor-depth",
        type=int,
        default=1,
        choices=[1],
        help="L1 promotion preserves L0 SCAFFOLD scope guard: only depth=1 (single-layer FiLM); depth>=2 DEFERRED per per-substrate symposium binding revision #4.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=48,
        help="Decoded RGB height (default 48 for smoke; 384 for contest).",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=64,
        help="Decoded RGB width (default 64 for smoke; 512 for contest).",
    )
    # ----- L1 PROMOTION: real contest video frames + EMA -----
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Path to canonical contest video for real-frame training (default upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--use-synthetic-frames",
        action="store_true",
        help="Use L0-SCAFFOLD-equivalent synthetic random frames instead of real contest video. For backward-compat smoke; default L1 uses real video.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help="EMA decay per CLAUDE.md 'EMA NON-NEGOTIABLE' (default 0.997 = Quantizr PR101 canonical anchor).",
    )
    parser.add_argument(
        "--disable-ema",
        action="store_true",
        help="(NOT RECOMMENDED) Disable EMA shadow. Honored only for sister-trainer comparison; canonical L1 promotion requires EMA.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _require_mlx()
    args = _parse_args(argv)

    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim

    # Import after MLX check so non-Apple-Silicon CI gets the clean SystemExit
    # from _require_mlx() above instead of an opaque module-import error.
    from tac.substrates.time_traveler_l5_z6.architecture import (
        Z6PredictiveCodingConfig,
    )
    from tac.substrates.time_traveler_l5_z6.mlx_export_bridge import (
        build_z6_pytorch_pt_from_mlx_renderer,
        build_z6pcwm1_archive_from_mlx_renderer,
    )
    from tac.substrates.time_traveler_l5_z6.mlx_renderer import (
        EVIDENCE_GRADE,
        EVIDENCE_TAG,
        LANE_ID,
        SCHEMA_VERSION,
        Z6PredictiveCodingMLXRenderer,
    )

    if args.smoke:
        # L1 PROMOTION: smoke widened from L0's (8 pairs, 5 epochs) to (50 pairs,
        # 5 epochs) because L1 uses REAL frames and benefits from more pair
        # diversity; runtime budget remains <10s at 48x64 + 1-block decoder.
        num_pairs = min(args.num_pairs, 50)
        epochs = min(args.epochs, 5)
    else:
        num_pairs = args.num_pairs
        epochs = args.epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build a small Z6 config sized for the trainer's bounded budget.
    cfg = Z6PredictiveCodingConfig(
        latent_dim=args.latent_dim,
        num_pairs=num_pairs,
        output_height=args.output_height,
        output_width=args.output_width,
        # Decoder param budget: shrink decoder_channels/num_upsample_blocks to
        # match the (output_height, output_width) target. Default contest
        # config uses 6 PixelShuffle blocks (factor 64x); for smoke we use 1
        # block (factor 2x) so the smoke runs in a few seconds.
        decoder_num_upsample_blocks=(
            6 if args.output_height >= 384 else 1
        ),
        decoder_channels=(
            (24, 20, 16, 12, 8, 6) if args.output_height >= 384 else (6,)
        ),
        decoder_embed_dim=32 if args.output_height >= 384 else 16,
        predictor_depth=args.predictor_depth,
        lambda_residual_entropy=args.lambda_residual,
    )

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print(
        f"[z6-mlx-trainer] Building Z6 MLX renderer: "
        f"latent_dim={cfg.latent_dim} num_pairs={cfg.num_pairs} "
        f"output={cfg.output_height}x{cfg.output_width} "
        f"predictor_depth={cfg.predictor_depth}"
    )
    renderer = Z6PredictiveCodingMLXRenderer(cfg)
    breakdown = renderer.num_parameters_breakdown()
    print(
        f"[z6-mlx-trainer] params: encoder={breakdown['encoder']} "
        f"decoder={breakdown['decoder']} predictor={breakdown['predictor']} "
        f"latent_init={breakdown['latent_init']} residuals={breakdown['residuals']} "
        f"total={breakdown['total']}"
    )

    # Pin ego-motion buffer with deterministic synthetic values per Catalog
    # #311 — the FiLM predictor's modulation pipeline must be exercised.
    ego_motion_np = np.random.RandomState(args.seed + 100).randn(
        cfg.num_pairs, cfg.predictor_ego_motion_dim,
    ).astype(np.float32) * 0.1
    renderer.ego_motion_buffer = mx.array(ego_motion_np)

    # L1 PROMOTION: REAL contest video targets via canonical pyav helper
    # ----------------------------------------------------------------
    # Per per-substrate symposium binding revision #2 (Yousfi dissent acceptance
    # at L1 surface): the L1 promotion replaces L0 SCAFFOLD's synthetic random
    # RGB with REAL contest video frames decoded via canonical
    # `tac.data.decode_video` (pyav). Per symposium Catalog #324:
    # `predicted_band_validation_status=post_training_mlx_50_100ep_local`.
    if args.use_synthetic_frames:
        # Backward-compat path: L0 SCAFFOLD synthetic targets (for sister
        # comparison / unit tests / debug).
        print(
            "[z6-mlx-trainer] [L1-PROMOTION] --use-synthetic-frames=True; "
            "using L0-SCAFFOLD-equivalent synthetic random RGB (non-canonical-L1)"
        )
        target_rgb_0_np = np.random.RandomState(args.seed + 200).rand(
            cfg.num_pairs, cfg.output_height, cfg.output_width, 3,
        ).astype(np.float32)
        target_rgb_1_np = np.random.RandomState(args.seed + 300).rand(
            cfg.num_pairs, cfg.output_height, cfg.output_width, 3,
        ).astype(np.float32)
        frame_loader_metadata: dict[str, Any] = {
            "source": "synthetic_random_rgb",
            "video_path": None,
            "frames_decoded": 0,
            "decode_wall_clock_seconds": 0.0,
            "frame_resolution_hw": [int(cfg.output_height), int(cfg.output_width)],
        }
    else:
        # CANONICAL L1: REAL contest video frames via tac.data.decode_video
        from tac.data import decode_video

        if not args.video_path.exists():
            raise SystemExit(
                f"[z6-mlx-trainer] FATAL: contest video not found at "
                f"{args.video_path}. Pass --use-synthetic-frames for "
                f"backward-compat L0 SCAFFOLD synthetic targets, OR "
                f"verify the upstream submodule is initialized."
            )
        # Decode 2 frames per pair: non-overlapping pairs (0,1), (2,3), ...
        frames_needed = 2 * cfg.num_pairs
        print(
            f"[z6-mlx-trainer] [L1-PROMOTION] Decoding {frames_needed} real "
            f"contest video frames at {cfg.output_height}x{cfg.output_width} "
            f"from {args.video_path}"
        )
        t_decode = time.time()
        gt_frames = decode_video(
            args.video_path,
            target_h=cfg.output_height,
            target_w=cfg.output_width,
            max_frames=frames_needed,
        )
        decode_wall = time.time() - t_decode
        if len(gt_frames) < frames_needed:
            raise SystemExit(
                f"[z6-mlx-trainer] FATAL: video decoded {len(gt_frames)} "
                f"frames; needed {frames_needed} for {cfg.num_pairs} pairs."
            )
        # Stack into (num_pairs, 2, H, W, 3) uint8; split into rgb_0 / rgb_1
        # per Catalog #311 ego-motion-conditioned pair structure.
        gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)  # (frames_needed, H, W, 3) uint8
        gt_arr_pairs = gt_arr.reshape(
            cfg.num_pairs, 2, cfg.output_height, cfg.output_width, 3,
        )
        # Convert uint8 [0,255] -> float32 [0,1] for MSE proxy training
        target_rgb_0_np = (gt_arr_pairs[:, 0, :, :, :].astype(np.float32) / 255.0)
        target_rgb_1_np = (gt_arr_pairs[:, 1, :, :, :].astype(np.float32) / 255.0)
        print(
            f"[z6-mlx-trainer] [L1-PROMOTION] decoded {len(gt_frames)} frames "
            f"in {decode_wall:.1f}s; pairs={cfg.num_pairs} "
            f"rgb_0 shape={target_rgb_0_np.shape} dtype={target_rgb_0_np.dtype}"
        )
        frame_loader_metadata = {
            "source": "real_contest_video_pyav",
            "video_path": str(args.video_path),
            "frames_decoded": int(len(gt_frames)),
            "decode_wall_clock_seconds": float(decode_wall),
            "frame_resolution_hw": [int(cfg.output_height), int(cfg.output_width)],
            "domain": "unit_float32_0_to_1",
            "canonical_helper": "tac.data.decode_video",
        }

    target_rgb_0 = mx.array(target_rgb_0_np)
    target_rgb_1 = mx.array(target_rgb_1_np)

    # Bounded budget: full-pair-batch reconstruct + MSE proxy loss + residual
    # L2. Per CLAUDE.md "HNeRV / leaderboard-implementation parity" L6: a true
    # contest-grade score-aware Lagrangian must route through SegNet/PoseNet;
    # the L0 SCAFFOLD's MSE proxy is non-promotable by construction and the
    # canonical Provenance metadata stamps that.
    pair_indices_all = mx.arange(cfg.num_pairs, dtype=mx.int32)

    def _loss_fn(model: Z6PredictiveCodingMLXRenderer) -> Any:
        rgb_0, rgb_1, _z = model.reconstruct_pair(pair_indices_all)
        # MSE reconstruction loss (proxy; non-promotable per Catalog #287)
        mse_0 = mx.mean((rgb_0 - target_rgb_0) ** 2)
        mse_1 = mx.mean((rgb_1 - target_rgb_1) ** 2)
        recon_loss = mse_0 + mse_1
        # Residual L2 Lagrangian (Rao-Ballard predictive-coding signal)
        residual_l2 = mx.mean(model.residuals ** 2)
        total_loss = recon_loss + args.lambda_residual * residual_l2
        return total_loss

    optimizer = mlx_optim.AdamW(learning_rate=args.learning_rate)
    loss_grad = mlx_nn.value_and_grad(renderer, _loss_fn)

    # ----- L1 PROMOTION: MLX-native EMA shadow -----
    # Per CLAUDE.md "EMA -- NON-NEGOTIABLE, HIGHEST EMPHASIS" + Catalog #2:
    # every training path producing an inference checkpoint MUST instantiate
    # EMA decay=0.997 + save EMA shadow as inference checkpoint.
    #
    # MLX-native EMA: shadow = dict of `tree_flatten`-keyed mx.array copies
    # of every trainable parameter; update via shadow := decay*shadow +
    # (1-decay)*live after every optimizer.step(); apply via snapshot+restore
    # only at archive-export time (NEVER inside train loop per CLAUDE.md
    # "NEVER call ema.apply(model) inside train_epoch").
    from mlx.utils import tree_flatten, tree_unflatten

    ema_enabled = not args.disable_ema
    ema_shadow: dict[str, Any] = {}
    if ema_enabled:
        print(
            f"[z6-mlx-trainer] [L1-PROMOTION] Initializing MLX-native EMA "
            f"decay={args.ema_decay} (Catalog #2 NON-NEGOTIABLE)"
        )
        live_params_flat = tree_flatten(renderer.parameters())
        for k, v in live_params_flat:
            # Clone live parameter (detached snapshot; mx.array copies via
            # arithmetic identity multiplication preserves dtype + shape).
            ema_shadow[k] = mx.array(v)
    else:
        print(
            "[z6-mlx-trainer] [L1-PROMOTION] EMA DISABLED via --disable-ema; "
            "non-canonical L1 — use sister-trainer comparison ONLY"
        )

    def _ema_update_after_optimizer_step() -> None:
        """Update EMA shadow weights per canonical Polyak averaging."""
        if not ema_enabled:
            return
        live_flat = tree_flatten(renderer.parameters())
        for k, v in live_flat:
            if k not in ema_shadow:
                # Late-bound module (e.g. lazy init) — seed from live per
                # Codex finding 2 hardening in canonical EMA.
                ema_shadow[k] = mx.array(v)
                continue
            # shadow := decay*shadow + (1-decay)*live
            ema_shadow[k] = args.ema_decay * ema_shadow[k] + (1.0 - args.ema_decay) * v

    per_epoch_metrics: list[dict[str, float]] = []
    t_start = time.time()
    for epoch in range(epochs):
        loss_val, grads = loss_grad(renderer)
        optimizer.update(renderer, grads)
        mx.eval(renderer.parameters(), optimizer.state)
        # EMA update AFTER optimizer step (canonical Polyak pattern)
        _ema_update_after_optimizer_step()
        if ema_enabled:
            mx.eval(list(ema_shadow.values()))
        loss_scalar = float(loss_val.item())
        per_epoch_metrics.append({
            "epoch": int(epoch),
            "loss": loss_scalar,
            "wall_clock_seconds": float(time.time() - t_start),
            "ema_enabled": bool(ema_enabled),
        })
        print(
            f"[z6-mlx-trainer] epoch {epoch+1}/{epochs} "
            f"loss={loss_scalar:.6f} "
            f"wall={time.time() - t_start:.1f}s "
            f"ema={'on' if ema_enabled else 'off'}"
        )
    total_wall = time.time() - t_start

    # Export LIVE state_dict .pt via canonical #1251 bridge (for sister comparison)
    pt_path = args.output_dir / "z6_mlx_state_dict.pt"
    print(f"[z6-mlx-trainer] Exporting LIVE .pt to {pt_path}")
    pt_manifest = build_z6_pytorch_pt_from_mlx_renderer(
        renderer, pt_path, overwrite=True,
    )

    # ----- L1 PROMOTION: EMA shadow as canonical inference checkpoint -----
    # Per CLAUDE.md "Inference / archive bytes come from ema.state_dict()" +
    # snapshot+restore pattern at export time (NEVER inside train_epoch).
    # We export the EMA shadow weights as the canonical inference checkpoint
    # AND build the Z6PCWM1 archive from the EMA shadow per Catalog #2 +
    # Quantizr PR101 anchor.
    if ema_enabled:
        ema_shadow_path = args.output_dir / "z6_mlx_state_dict_ema_shadow.pt"
        print(
            f"[z6-mlx-trainer] [L1-PROMOTION] Exporting EMA SHADOW (canonical "
            f"inference checkpoint) to {ema_shadow_path}"
        )
        # Snapshot live params, swap in EMA shadow for export, restore live
        live_params_snapshot = tree_flatten(renderer.parameters())
        try:
            shadow_as_unflattened = tree_unflatten(list(ema_shadow.items()))
            renderer.update(shadow_as_unflattened)
            mx.eval(renderer.parameters())
            ema_pt_manifest = build_z6_pytorch_pt_from_mlx_renderer(
                renderer, ema_shadow_path, overwrite=True,
            )
            # Build Z6PCWM1 archive from EMA shadow weights (canonical inference)
            archive_path = args.output_dir / "0.bin"
            print(
                f"[z6-mlx-trainer] [L1-PROMOTION] Building Z6PCWM1 archive "
                f"from EMA SHADOW (canonical inference checkpoint) at {archive_path}"
            )
            arc_manifest = build_z6pcwm1_archive_from_mlx_renderer(
                renderer, archive_path, overwrite=True,
                lambda_residual_entropy=args.lambda_residual,
            )
        finally:
            # Restore live params (so future calls observe training state)
            live_snapshot_unflattened = tree_unflatten(list(live_params_snapshot))
            renderer.update(live_snapshot_unflattened)
            mx.eval(renderer.parameters())
    else:
        # No EMA: archive uses live weights (non-canonical fallback)
        archive_path = args.output_dir / "0.bin"
        print(
            f"[z6-mlx-trainer] [L1-PROMOTION] (--disable-ema) Building Z6PCWM1 "
            f"archive from LIVE weights (non-canonical) at {archive_path}"
        )
        arc_manifest = build_z6pcwm1_archive_from_mlx_renderer(
            renderer, archive_path, overwrite=True,
            lambda_residual_entropy=args.lambda_residual,
        )
        ema_pt_manifest = None

    # Build training manifest (canonical Provenance per Catalog #287/#323)
    training_manifest = {
        "schema_version": f"{SCHEMA_VERSION}_training_manifest_l1_promotion",
        "substrate_id": "time_traveler_l5_z6",
        "lane_id": "lane_path_3_d_z6_l1_promotion_20260526",
        "lane_id_l0_scaffold_predecessor": LANE_ID,  # FAKE_LANE_OK:dict_key_name_lane_id_l0_scaffold_predecessor_is_field_label_not_a_lane_id_reference_per_catalog_126_false_positive_per_comprehensive_bug_audit_cascade_20260526
        "promotion_status": "L1_INFRASTRUCTURE_CONVERGENCE_VERIFICATION",
        "predicted_band_validation_status": "post_training_mlx_50_100ep_local",
        "predicted_band_design_memo_cite": "[0.13, 0.16] per Z6 design memo Section 18 (planning prior; non-promotable per CLAUDE.md 'Apples-to-apples evidence discipline')",
        "per_substrate_symposium_memo": ".omx/research/path_3_d_z6_per_substrate_symposium_l1_promotion_20260526.md",
        "per_substrate_symposium_verdict": "PROCEED_WITH_REVISIONS",
        "run_id": datetime.now(UTC).strftime("z6_mlx_train_%Y%m%dT%H%M%SZ"),
        "config": {
            "latent_dim": cfg.latent_dim,
            "num_pairs": cfg.num_pairs,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "predictor_depth": cfg.predictor_depth,
            "lambda_residual_entropy": cfg.lambda_residual_entropy,
            "epochs": epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "smoke_mode": bool(args.smoke),
            "ema_decay": float(args.ema_decay) if ema_enabled else None,
            "ema_enabled": bool(ema_enabled),
            "use_synthetic_frames": bool(args.use_synthetic_frames),
        },
        "frame_loader": frame_loader_metadata,
        "parameter_breakdown": breakdown,
        "per_epoch_metrics": per_epoch_metrics,
        "total_wall_clock_seconds": total_wall,
        "outputs": {
            "pt_manifest_live": pt_manifest,
            "pt_manifest_ema_shadow": ema_pt_manifest,
            "archive_manifest": arc_manifest,
            "archive_source": "ema_shadow_canonical" if ema_enabled else "live_weights_non_canonical",
        },
        "training_evidence_grade": EVIDENCE_GRADE,
        "training_evidence_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": EVIDENCE_TAG,
        "hardware_substrate": "darwin_arm64_mlx_local",
        "blockers": [
            "macos_mlx_research_signal_training_axis_only",
            "mse_proxy_loss_on_real_video_non_promotable_per_catalog_287_symposium_contrarian_dissent",
            "score_aware_lagrangian_via_segnet_posenet_routes_through_pytorch_sister_l2_promotion",
            "requires_pass_verdict_from_sister_z6_gate_or_pr95_parameterized_gate_per_catalog_325_op_routable_2",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_any_score_claim",
            "posenet_derived_ego_motion_deferred_to_l2_per_yousfi_dissent",
        ],
        "operator_routable_next_steps": [
            (
                f"Run #1265 gate (PR95-parameterized OR Z6 sister): "
                f".venv/bin/python tools/gate_mlx_candidate_contest_equivalence.py "
                f"--archive-zip {archive_path} --candidate-label z6_predictive_coding_l1 "
                f"--output-json {args.output_dir / 'gate_verdict.json'} "
                f"  # NOTE: hardwired for PR95 grammar; D=Z6 needs sister gate parameterization per symposium op-routable #2"
            ),
            (
                "On PASS: operator routes paired CUDA dispatch via "
                "experiments/train_substrate_time_traveler_l5_z6.py + "
                "tools/operator_authorize.py per Catalog #313 predecessor-probe check"
            ),
            (
                "L2 PoseNet ego-motion (Yousfi dissent op-routable #3): "
                "sister subagent wires PoseNet projections into ego_motion_buffer"
            ),
            (
                "L2 score-aware loss (Contrarian dissent op-routable #4): "
                "PyTorch sister trainer + paid CUDA dispatch"
            ),
        ],
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(training_manifest, indent=2, sort_keys=True))
    print(f"[z6-mlx-trainer] training_manifest.json written to {manifest_path}")
    print(f"[z6-mlx-trainer] DONE. Total wall-clock: {total_wall:.1f}s")
    print(f"[z6-mlx-trainer] axis_tag={EVIDENCE_TAG} promotable=False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
