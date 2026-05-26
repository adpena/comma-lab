# SPDX-License-Identifier: MIT
"""J=MDL-IBPS L2 long-training trainer — L1-PROMOTION-CASCADE entry-point.

Per Path 3 canonical-substrate-development-cascade doctrine §"L2 LONG-
TRAINING INFRASTRUCTURE" + L1-PROMOTION-CASCADE-B-C-E-G-J charter
2026-05-26: each substrate's L2 trainer is ~136 LOC of substrate-
specific config + ONE canonical helper invocation, mirroring the D=Z6
reference template at
``experiments/train_substrate_z6_predictive_coding_mlx_l2.py``.

This entry-point is the canonical L1-PROMOTION mirror for J=MDL-IBPS
substrate (``tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid``).
The substrate-specific adapter at
``tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.long_training_adapter``
satisfies the
:class:`tac.training.long_training_canonical.SubstrateLongTrainingAdapter`
Protocol; this entry-point provides the LongTrainingConfig + invokes
``run_long_training`` once.

L0 SCAFFOLD POSTURE (Catalog #240 recipe-vs-trainer-state consistency)
===================================================================
Per ``mdl_ibps_j_discrete_categorical_mine_hybrid/__init__.py``
``RESEARCH_ONLY = True`` + ``IMPLEMENTATION_STATUS`` explicit L0
SCAFFOLD posture: invoking this trainer today fails-fast at adapter
``__init__`` with explicit L1-follow-up guidance per
``MdlIbpsJLongTrainingAdapter.__init__`` docstring. The L1 follow-up
subagent removes the adapter's NotImplementedError + wraps the
MDLIBPSJRendererMLX primitives as a trainable ``mlx.nn.Module``; this
entry-point becomes functional WITHOUT modification when L1 lands.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: fail-fast at L0 SCAFFOLD is the correct posture; the
structural shape is the L1 follow-up's drop-in target.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: non-promotable by construction; canonical helper auto-
stamps every output as ``[macOS-MLX research-signal]``.

Run (smoke at 30ep matches Z6 L1 baseline shape; fails fast at adapter
__init__ with explicit L1 follow-up guidance until L1 lands the
trainable mlx.nn.Module wrapper):

    PYTHONPATH=src:upstream:$PWD .venv/bin/python \\
        experiments/train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid_mlx_l2.py \\
        --output-dir experiments/results/path_3_j_mdl_ibps_l1_promotion_smoke_<utc>/ \\
        --epochs 30 --num-pairs 50
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
    parser.add_argument("--beta-ib", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse", type=float, default=1e-4)
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
            "[j-mdl-ibps-l2-mlx-trainer] FATAL: MLX required (Apple Silicon)",
            file=sys.stderr,
        )
        return 2

    from tac.data import decode_video
    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
        LANE_ID,
        SUBSTRATE_ID,
    )
    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.long_training_adapter import (
        MdlIbpsJLongTrainingAdapter,
    )
    from tac.training.long_training_canonical import (
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # 1) Decode real contest video frames for substrate adapter targets.
    print(
        f"[j-mdl-ibps-l2-mlx-trainer] Decoding {2 * args.num_pairs} real frames "
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
        f"[j-mdl-ibps-l2-mlx-trainer] decoded {len(gt_frames)} frames in "
        f"{time.time() - t_decode:.1f}s"
    )
    gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)
    gt_pairs = gt_arr.reshape(args.num_pairs, 2, args.output_height, args.output_width, 3)
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))

    # 2) Build canonical substrate adapter.
    # L0 SCAFFOLD posture: __init__ raises NotImplementedError with explicit
    # L1-follow-up guidance per Catalog #240; this entry-point honors the
    # exception by surfacing it to the operator BEFORE any GPU/MLX spend.
    adapter = MdlIbpsJLongTrainingAdapter(
        config=None,  # L1 follow-up adds MdlIbpsJConfig dataclass.
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        beta_ib=args.beta_ib,
        lambda_sparse=args.lambda_sparse,
    )

    # 3) Build canonical LongTrainingConfig (single CurriculumStage for L1-promotion smoke).
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
                name="j_mdl_ibps_l1_promotion_smoke",
                start_epoch=0,
                end_epoch=args.epochs,
                notes=(
                    "L1-PROMOTION-CASCADE smoke stage: J adapter Style B "
                    "train_step active after L1 wraps renderer as mlx.nn.Module."
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
            "J=MDL-IBPS L1-PROMOTION-CASCADE entry-point per cascade doctrine + "
            "Tier1-T3-OP7-OP8 amendment; adapter is L0 SCAFFOLD structural shell."
        ),
    )

    # 4) ONE canonical helper invocation does everything else.
    artifact = run_long_training(adapter, canonical_config)

    print(
        f"[j-mdl-ibps-l2-mlx-trainer] DONE epochs_completed={artifact.total_epochs_completed} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"promotable={artifact.promotable} "
        f"early_stopped={artifact.early_stopped}"
    )
    print(
        f"[j-mdl-ibps-l2-mlx-trainer] artifact JSON: "
        f"{canonical_config.output_dir / 'training_artifact.json'}"
    )
    # Reference unused canonical id for static checkers.
    _ = LANE_ID
    return 0


if __name__ == "__main__":
    sys.exit(main())
