# SPDX-License-Identifier: MIT
"""G=NIRVANA-cascading-NeRV L2 long-training trainer — L1-PROMOTION-CASCADE entry-point.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Per Path 3 canonical-substrate-development-cascade doctrine §"L2 LONG-
TRAINING INFRASTRUCTURE" + L1-PROMOTION-CASCADE-B-C-E-G-J charter
2026-05-26: each substrate's L2 trainer is ~136 LOC of substrate-
specific config + ONE canonical helper invocation, mirroring the D=Z6
reference template at
``experiments/train_substrate_z6_predictive_coding_mlx_l2.py``.

This entry-point is the canonical L1-PROMOTION mirror for G=NIRVANA-
cascading-NeRV substrate (``tac.substrates.nirvana_cascading_nerv``).
The substrate-specific adapter at
``tac.substrates.nirvana_cascading_nerv.long_training_adapter``
satisfies the
:class:`tac.training.long_training_canonical.SubstrateLongTrainingAdapter`
Protocol; this entry-point provides the LongTrainingConfig + invokes
``run_long_training`` once.

L0 SCAFFOLD POSTURE: invoking this trainer today fails-fast at adapter
``__init__`` with explicit L1+Phase-2 follow-up guidance per
``NirvanaCascadingNervLongTrainingAdapter.__init__`` docstring. The
follow-up subagent removes the adapter's NotImplementedError + lands
``NirvanaCascadingNervRendererMLX(nn.Module)`` per Phase 2 design
memo's hierarchical-residual decoder cascade specification; this
entry-point becomes functional WITHOUT modification when Phase 2 lands.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: non-promotable by construction.

Note: sister L1 trainer already exists at
``experiments/train_substrate_nirvana_cascading_nerv_mlx.py`` (NOT
canonical L2 helper based; predates L2-INFRA-BUILD landing
`f5e4784ef`). The L1 trainer is preserved per Catalog #110/#113
APPEND-ONLY; this L2 trainer is the canonical L1-PROMOTION cascade
mirror per cascade doctrine.
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
    parser.add_argument("--output-height", type=int, default=384,
                        help="Contest resolution; G's NIRVANA1 cascade decodes to 384x512.")
    parser.add_argument("--output-width", type=int, default=512)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        import mlx.core as mx
    except ImportError:
        print(
            "[g-nirvana-l2-mlx-trainer] FATAL: MLX required (Apple Silicon)",
            file=sys.stderr,
        )
        return 2

    from tac.data import decode_video
    from tac.substrates.nirvana_cascading_nerv.long_training_adapter import (
        NirvanaCascadingNervLongTrainingAdapter,
    )
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )
    from tac.training.long_training_canonical import (
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # 1) Build G substrate config (canonical dataclass at mlx_renderer.py).
    # Default = full contest resolution (48x64 base, 4-level cascade -> 384x512).
    # Use defaults to satisfy EVAL_HW constraint; smoke uses subset of pairs.
    cfg = NirvanaCascadingNervConfig(num_pairs=args.num_pairs)

    # 2) Decode real contest video frames for substrate adapter targets.
    print(
        f"[g-nirvana-l2-mlx-trainer] Decoding {2 * args.num_pairs} real frames "
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
        f"[g-nirvana-l2-mlx-trainer] decoded {len(gt_frames)} frames in "
        f"{time.time() - t_decode:.1f}s"
    )
    gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)
    gt_pairs = gt_arr.reshape(args.num_pairs, 2, args.output_height, args.output_width, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))

    # 3) Build canonical substrate adapter.
    adapter = NirvanaCascadingNervLongTrainingAdapter(
        config=cfg,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        residual_loss_weight=args.residual_loss_weight,
    )

    # 4) Build canonical LongTrainingConfig.
    canonical_config = LongTrainingConfig(
        substrate_id="nirvana_cascading_nerv",
        lane_id=(
            f"lane_path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_"
            f"canonical_l2_helper_{datetime.now(UTC):%Y%m%d}"
        ),
        epochs=args.epochs,
        batch_pair_indices_per_step=min(args.num_pairs, 8),
        curriculum_stages=(
            CurriculumStage(
                name="g_nirvana_l1_promotion_smoke",
                start_epoch=0,
                end_epoch=args.epochs,
                notes=(
                    "L1-PROMOTION-CASCADE smoke stage: G adapter Style B "
                    "train_step active after L1+Phase-2 lands "
                    "NirvanaCascadingNervRendererMLX(nn.Module)."
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
            "G=NIRVANA L1-PROMOTION-CASCADE entry-point per cascade doctrine + "
            "Tier1-T3-OP7-OP8 amendment; adapter is L0 SCAFFOLD structural shell."
        ),
    )

    # 5) ONE canonical helper invocation does everything else.
    artifact = run_long_training(adapter, canonical_config)

    print(
        f"[g-nirvana-l2-mlx-trainer] DONE epochs_completed="
        f"{artifact.total_epochs_completed} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"promotable={artifact.promotable} "
        f"early_stopped={artifact.early_stopped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
