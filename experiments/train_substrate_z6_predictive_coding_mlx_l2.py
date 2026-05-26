# SPDX-License-Identifier: MIT
"""Z6 L2 long-training trainer — canonical proof-of-pattern.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Per Path 3 canonical-substrate-development-cascade doctrine §"L2 LONG-
TRAINING INFRASTRUCTURE" + L2-INFRA-BUILD landing memo: each substrate's
L2 trainer is ~30 LOC of substrate-specific config + ONE canonical helper
invocation. This is the canonical PROOF-OF-PATTERN for D=Z6 (sister
substrates inherit the same shape at L1->L2 promotion).

Contrast with the L1 promotion trainer at
``experiments/train_substrate_z6_predictive_coding_mlx.py`` (~600 LOC of
hand-rolled training loop + EMA + checkpoint + Provenance + posterior
anchor + archive emission) — that LOC budget moves into the canonical
helper here; the substrate side only carries the substrate-axis decisions
(curriculum overrides + ego-motion seed + Z6PCWM1 archive grammar).

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: non-promotable by construction; canonical helper auto-
stamps every output as ``[macOS-MLX research-signal]``.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-height", type=int, default=48)
    parser.add_argument("--output-width", type=int, default=64)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=20)
    args = parser.parse_args(argv)

    # Canonical Z6 imports — substrate-specific (architecture, data, adapter).
    try:
        import mlx.core as mx
    except ImportError:
        print("[z6-l2-mlx-trainer] FATAL: MLX required (Apple Silicon)", file=sys.stderr)
        return 2

    from tac.data import decode_video
    from tac.substrates.time_traveler_l5_z6.architecture import Z6PredictiveCodingConfig
    from tac.substrates.time_traveler_l5_z6.long_training_adapter import Z6LongTrainingAdapter
    from tac.training.long_training_canonical import (
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # 1) Build Z6 substrate-specific config.
    cfg = Z6PredictiveCodingConfig(
        latent_dim=args.latent_dim,
        num_pairs=args.num_pairs,
        output_height=args.output_height,
        output_width=args.output_width,
        decoder_num_upsample_blocks=6 if args.output_height >= 384 else 1,
        decoder_channels=(24, 20, 16, 12, 8, 6) if args.output_height >= 384 else (6,),
        decoder_embed_dim=32 if args.output_height >= 384 else 16,
        predictor_depth=1,
        lambda_residual_entropy=args.lambda_residual,
    )

    # 2) Decode real contest video frames at the configured resolution.
    print(f"[z6-l2-mlx-trainer] Decoding {2 * args.num_pairs} real frames from {args.video_path}")
    t_decode = time.time()
    gt_frames = decode_video(
        args.video_path,
        target_h=args.output_height,
        target_w=args.output_width,
        max_frames=2 * args.num_pairs,
    )
    print(f"[z6-l2-mlx-trainer] decoded {len(gt_frames)} frames in {time.time()-t_decode:.1f}s")
    gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)
    gt_pairs = gt_arr.reshape(args.num_pairs, 2, args.output_height, args.output_width, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))

    # 3) Build canonical substrate adapter.
    adapter = Z6LongTrainingAdapter(
        config=cfg,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        lambda_residual=args.lambda_residual,
    )
    # Seed ego-motion buffer per Catalog #311 ego-motion-conditioning.
    ego = np.random.RandomState(args.seed + 100).randn(args.num_pairs, cfg.predictor_ego_motion_dim).astype(np.float32) * 0.1
    adapter.model.ego_motion_buffer = mx.array(ego)

    # 4) Build canonical LongTrainingConfig (single CurriculumStage for L2 smoke).
    canonical_config = LongTrainingConfig(
        substrate_id="time_traveler_l5_z6",
        lane_id=f"lane_path_3_d_z6_l2_long_training_canonical_proof_{datetime.now(UTC):%Y%m%d}",
        epochs=args.epochs,
        batch_pair_indices_per_step=min(args.num_pairs, 8),
        curriculum_stages=(
            CurriculumStage(name="z6_l2_recon_full", start_epoch=0, end_epoch=args.epochs),
        ),
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        device="mlx",
        checkpoint_interval_epochs=args.checkpoint_interval_epochs,
        early_stopping_patience=args.epochs + 1,  # disable early stop for L2 smoke
        evidence_grade="[macOS-MLX research-signal]",
    )

    # 5) ONE canonical helper invocation does everything else.
    artifact = run_long_training(adapter, canonical_config)

    print(f"[z6-l2-mlx-trainer] DONE epochs_completed={artifact.total_epochs_completed} wall={artifact.total_wall_clock_seconds:.1f}s promotable={artifact.promotable}")
    print(f"[z6-l2-mlx-trainer] artifact JSON: {canonical_config.output_dir/'training_artifact.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
