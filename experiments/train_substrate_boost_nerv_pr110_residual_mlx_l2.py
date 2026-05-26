# SPDX-License-Identifier: MIT
"""E=BoostNeRV-PR110-residual L2 long-training trainer — L1-PROMOTION-CASCADE entry-point.

Per Path 3 canonical-substrate-development-cascade doctrine §"L2 LONG-
TRAINING INFRASTRUCTURE" + L1-PROMOTION-CASCADE-B-C-E-G-J charter
2026-05-26: each substrate's L2 trainer is ~136 LOC of substrate-
specific config + ONE canonical helper invocation, mirroring the D=Z6
reference template at
``experiments/train_substrate_z6_predictive_coding_mlx_l2.py``.

This entry-point is the canonical L1-PROMOTION mirror for E=BoostNeRV-
PR110-residual substrate (``tac.substrates.boost_nerv_pr110_residual``).
The substrate-specific adapter at
``tac.substrates.boost_nerv_pr110_residual.long_training_adapter``
satisfies the
:class:`tac.training.long_training_canonical.SubstrateLongTrainingAdapter`
Protocol; this entry-point provides the LongTrainingConfig + invokes
``run_long_training`` once.

L0 SCAFFOLD POSTURE: invoking this trainer today fails-fast at adapter
``__init__`` with explicit L1-follow-up guidance per
``BoostNervPr110ResidualLongTrainingAdapter.__init__`` docstring. The L1
follow-up subagent removes the adapter's NotImplementedError + wraps
ResidualHeadMLX as mlx.nn.Module + wires Stage 0 PR110 base caching +
lands inflate.py; this entry-point becomes functional WITHOUT
modification when L1 lands.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: non-promotable by construction.
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--residual-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-height", type=int, default=96)
    parser.add_argument("--output-width", type=int, default=128)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--pr110-base-path", type=Path, default=None,
                        help="Path to PR110 base archive (L1 follow-up Stage 0 dependency)")
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        import mlx.core as mx
    except ImportError:
        print(
            "[e-boost-nerv-pr110-l2-mlx-trainer] FATAL: MLX required (Apple Silicon)",
            file=sys.stderr,
        )
        return 2

    from tac.data import decode_video
    from tac.substrates.boost_nerv_pr110_residual import BoostNervPr110ResidualConfig
    from tac.substrates.boost_nerv_pr110_residual.long_training_adapter import (
        BoostNervPr110ResidualLongTrainingAdapter,
    )
    from tac.training.long_training_canonical import (
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # 1) Build E substrate config (canonical dataclass at architecture.py).
    cfg = BoostNervPr110ResidualConfig(
        num_pairs=args.num_pairs,
        output_height=args.output_height,
        output_width=args.output_width,
    )

    # 2) Decode real contest video frames for substrate adapter targets.
    print(
        f"[e-boost-nerv-pr110-l2-mlx-trainer] Decoding {2 * args.num_pairs} real frames "
        f"from {args.video_path}"
    )
    t_decode = time.time()
    gt_frames = decode_video(
        args.video_path,
        target_h=args.output_height,
        target_w=args.output_width,
        max_frames=2 * args.num_pairs,
    )
    print(
        f"[e-boost-nerv-pr110-l2-mlx-trainer] decoded {len(gt_frames)} frames in "
        f"{time.time() - t_decode:.1f}s"
    )
    gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)
    gt_pairs = gt_arr.reshape(args.num_pairs, 2, args.output_height, args.output_width, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))

    # 3) Build canonical substrate adapter.
    adapter = BoostNervPr110ResidualLongTrainingAdapter(
        config=cfg,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        pr110_base_path=args.pr110_base_path,
        residual_loss_weight=args.residual_loss_weight,
    )

    # 4) Build canonical LongTrainingConfig.
    canonical_config = LongTrainingConfig(
        substrate_id="boost_nerv_pr110_residual",
        lane_id=(
            f"lane_path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_"
            f"canonical_l2_helper_{datetime.now(UTC):%Y%m%d}"
        ),
        epochs=args.epochs,
        batch_pair_indices_per_step=min(args.num_pairs, 8),
        curriculum_stages=(
            CurriculumStage(
                name="e_boost_nerv_pr110_l1_promotion_smoke",
                start_epoch=0,
                end_epoch=args.epochs,
                notes=(
                    "L1-PROMOTION-CASCADE smoke stage: E adapter Style B "
                    "train_step active after L1 wraps ResidualHeadMLX as "
                    "mlx.nn.Module + Stage 0 PR110 base cache wired."
                ),
            ),
        ),
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        device="mlx",
        checkpoint_interval_epochs=args.checkpoint_interval_epochs,
        early_stopping_patience=args.epochs + 1,
        evidence_grade="[macOS-MLX research-signal]",
        notes=(
            "E=BoostNeRV-PR110 L1-PROMOTION-CASCADE entry-point per cascade "
            "doctrine + Tier1-T3-OP7-OP8 amendment; adapter is L0 SCAFFOLD "
            "structural shell."
        ),
    )

    # 5) ONE canonical helper invocation does everything else.
    artifact = run_long_training(adapter, canonical_config)

    print(
        f"[e-boost-nerv-pr110-l2-mlx-trainer] DONE epochs_completed="
        f"{artifact.total_epochs_completed} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"promotable={artifact.promotable} "
        f"early_stopped={artifact.early_stopped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
