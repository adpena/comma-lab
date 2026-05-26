# SPDX-License-Identifier: MIT
"""C'=NSCS06-v8-chroma-LUT L2 long-training trainer — L1-PROMOTION-CASCADE entry-point.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Per Path 3 canonical-substrate-development-cascade doctrine §"L2 LONG-
TRAINING INFRASTRUCTURE" + L1-PROMOTION-CASCADE-B-C-E-G-J charter
2026-05-26.

PARADIGM ROUTING DECISION
========================
NSCS06 v8 chroma_lut is FUNDAMENTALLY a deterministic per-SegNet-class
chroma lookup-table codec (NOT gradient-trainable). The canonical L2
gradient-training helper is PRINCIPLED MISMATCH per Catalog #290. Per
``Nscs06V8ChromaLutLongTrainingAdapter.__init__`` docstring: L1 follow-
up routes to sister canonical iteration helper OR adapts the
deterministic-LUT cascade.

This entry-point exists for Protocol surface parity with sister Path
3 substrates (B'/D/E/G/J) so the L1-PROMOTION-CASCADE charter
deliverable matrix is complete; invoking it fails-fast at adapter
``__init__`` with explicit paradigm-routing guidance per Catalog #240.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: non-promotable by construction (paradigm-routing also).
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-height", type=int, default=48)
    parser.add_argument("--output-width", type=int, default=64)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        import mlx.core as mx
    except ImportError:
        print(
            "[c-prime-nscs06-v8-chroma-lut-l2-mlx-trainer] FATAL: MLX required "
            "(Apple Silicon)",
            file=sys.stderr,
        )
        return 2

    from tac.data import decode_video
    from tac.substrates.nscs06_v8_chroma_lut import SUBSTRATE_ID
    from tac.substrates.nscs06_v8_chroma_lut.architecture import (
        Nscs06V8ChromaLutConfig,
    )
    from tac.substrates.nscs06_v8_chroma_lut.long_training_adapter import (
        Nscs06V8ChromaLutLongTrainingAdapter,
    )
    from tac.training.long_training_canonical import (
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # 1) Build C' substrate config (canonical dataclass at architecture.py).
    cfg = Nscs06V8ChromaLutConfig()

    # 2) Decode real contest video frames for substrate adapter targets.
    print(
        f"[c-prime-nscs06-v8-chroma-lut-l2-mlx-trainer] Decoding "
        f"{2 * args.num_pairs} real frames from {args.video_path}"
    )
    t_decode = time.time()
    gt_frames = decode_video(
        args.video_path,
        target_h=args.output_height,
        target_w=args.output_width,
        max_frames=2 * args.num_pairs,
    )
    print(
        f"[c-prime-nscs06-v8-chroma-lut-l2-mlx-trainer] decoded "
        f"{len(gt_frames)} frames in {time.time() - t_decode:.1f}s"
    )
    gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)
    gt_pairs = gt_arr.reshape(args.num_pairs, 2, args.output_height, args.output_width, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))

    # 3) Build canonical substrate adapter (paradigm-routed shell).
    adapter = Nscs06V8ChromaLutLongTrainingAdapter(
        config=cfg,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
    )

    # 4) Build canonical LongTrainingConfig.
    canonical_config = LongTrainingConfig(
        substrate_id=SUBSTRATE_ID,
        lane_id=(
            f"lane_path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_"
            f"canonical_l2_helper_{datetime.now(UTC):%Y%m%d}"
        ),
        epochs=args.epochs,
        batch_pair_indices_per_step=min(args.num_pairs, 8),
        curriculum_stages=(
            CurriculumStage(
                name="c_prime_nscs06_v8_chroma_lut_paradigm_routed",
                start_epoch=0,
                end_epoch=args.epochs,
                notes=(
                    "L1-PROMOTION-CASCADE paradigm-routed shell stage: C' is "
                    "deterministic-LUT-codec paradigm; canonical L2 gradient-"
                    "training helper is PRINCIPLED MISMATCH per Catalog #290."
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
            "C'=NSCS06-v8-chroma-LUT L1-PROMOTION-CASCADE paradigm-routed "
            "entry-point per cascade doctrine + Tier1-T3-OP7-OP8 amendment; "
            "adapter is paradigm-mismatch shell pointing at sister canonical "
            "iteration helper."
        ),
    )

    # 5) ONE canonical helper invocation does everything else.
    artifact = run_long_training(adapter, canonical_config)

    print(
        f"[c-prime-nscs06-v8-chroma-lut-l2-mlx-trainer] DONE epochs_completed="
        f"{artifact.total_epochs_completed} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"promotable={artifact.promotable} "
        f"early_stopped={artifact.early_stopped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
