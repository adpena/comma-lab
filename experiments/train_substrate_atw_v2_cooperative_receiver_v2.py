# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — MLX score-aware trainer (Path 3 candidate H).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_decodes_real_contest_video_via_decode_mlx_targets_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: this trainer's ``_full_main`` is
UNBLOCKED. The prior ``NotImplementedError`` (Catalog #240(c)) is replaced by a
route through the canonical substrate-AGNOSTIC harness
``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``. The
unblock required wrapping the substrate's cond-embed head + HNeRV decoder + a
learnable per-pair ego-motion ``pose_delta`` table as a single trainable
``mlx.nn.Module`` (``ATWv2CooperativeReceiverV2TrainableMLX``) — the prior
blocker was that the renderer was NOT an ``nn.Module`` and ``reconstruct_pair``
took a separate ``pose_delta`` argument the harness does not supply.

## Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL_BECAUSE_SERVES: training loop / EMA / score-aware loss /
  Provenance / posterior anchor (the harness + ``run_long_training``).
- FORK_BECAUSE_PRINCIPLED_MISMATCH (this substrate's UNIQUE primitive): the
  Atick-Redlich cooperative-receiver + ego-motion FOE conditioning
  (``mlx_renderer.ATWv2CooperativeReceiverV2TrainableMLX``; per Catalog #311).

## Dispatch gating (Catalog #325)

MLX-LOCAL ONLY ($0 M5 Max); the harness fails closed on a non-MLX host (NO
CPU/CUDA paid-dispatch leak per Catalog #1 + #317). Any recipe stays
``dispatch_enabled: false`` + ``research_only: true``; output is non-promotable
``[macOS-MLX research-signal]`` per Catalog #192/#341. Per-axis SegNet/PoseNet
decomposition + MLX->PyTorch export bridge (#1251) + Catalog #319
deliverability_proof + paired [contest-CUDA]+[contest-CPU] anchor remain
DEFERRED to the PyTorch sister L2 / paid-dispatch path.

Non-promotable canonical contract per CLAUDE.md "MLX portable-local-substrate
authority": tagged [macOS-MLX research-signal]; score_claim=False,
promotion_eligible=False.
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
# Catalog #151 manifest — EVERY flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer (ast.AnnAssign per Catalog #168).
# The L0 scaffold has NO required input files in smoke; --full requires the
# real contest video at --video-path (required_input_file=True).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "ATW_V2_CR2_OUTPUT_DIR",
        "rationale": (
            "Output directory for the substrate's MLX-local training "
            "artifacts: training_artifact JSON + EMA checkpoint + "
            "observability surface."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "ATW_V2_CR2_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. Full training pending "
            "per-substrate symposium per Catalog #325 before any paid dispatch."
        ),
        "default": "2",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "ATW_V2_CR2_VIDEO_PATH",
        "rationale": (
            "Real contest video for --full MLX-first score-aware training "
            "(Catalog #114; real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Run smoke-mode forward pass"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full MLX score-aware training via the canonical harness.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Per-pair latent dim (full default 32 per design memo).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="MLX score-aware training epochs (--full).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for --full training artifacts.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full score-aware training (Catalog #114).",
    )
    parser.add_argument(
        "--full-lr",
        type=float,
        default=1e-3,
        help="Learning rate for --full MLX score-aware training.",
    )
    parser.add_argument(
        "--distillation-weight",
        type=float,
        default=0.5,
        help="Weight on the gradient-reachable Hinton-KL T=2.0 scorer "
        "surrogate term in the --full score-aware loss (0.0 disables).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional smoke result JSON path.",
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """End-to-end MLX forward smoke for the L0 SCAFFOLD.

    Builds a small substrate config + initializes weights + runs forward
    through the cond_embed + decoder; verifies output shape + value range.
    Sister to test_basic.py end-to-end tests.

    Per Catalog #287 + #323: emits non-promotable canonical Provenance.
    """
    import numpy as np

    from tac.substrates.atw_v2_cooperative_receiver_v2.numpy_reference import (
        CooperativeReceiverConfig,
        init_numpy_weights_random,
        numpy_decode_pair_with_ego_motion_conditioning,
        numpy_ego_motion_foe_projection,
    )

    print("[atw_v2_cooperative_receiver_v2:_smoke_main] starting L0 SCAFFOLD smoke")
    print(f"[atw_v2_cooperative_receiver_v2] seed={args.seed} batch_size={args.batch_size}")

    cfg = CooperativeReceiverConfig(
        num_pairs=args.num_pairs,
        latent_dim=8,  # smoke-only; production uses 32
        ego_motion_dim=6,
        cond_embed_dim=4,
        decoder_embed_dim=8,
        decoder_initial_grid_h=3,
        decoder_initial_grid_w=4,
        decoder_channels=(8, 6),
        decoder_num_upsample_blocks=2,
        output_height=12,
        output_width=16,
    )
    weights = init_numpy_weights_random(cfg, seed=args.seed)

    B = args.batch_size
    per_pair_latent_residual = np.zeros((B, cfg.latent_dim), dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    pose_delta = rng.normal(0, 0.1, size=(B, 6)).astype(np.float32)
    ego_motion_proj = numpy_ego_motion_foe_projection(pose_delta)

    rgb_0, rgb_1 = numpy_decode_pair_with_ego_motion_conditioning(
        per_pair_latent_residual,
        ego_motion_proj,
        cfg=cfg,
        cond_embed_weight_1=weights.cond_embed_weight_1,
        cond_embed_bias_1=weights.cond_embed_bias_1,
        cond_embed_weight_2=weights.cond_embed_weight_2,
        cond_embed_bias_2=weights.cond_embed_bias_2,
        initial_proj_weight=weights.initial_proj_weight,
        initial_proj_bias=weights.initial_proj_bias,
        decoder_block_weights=weights.decoder_block_weights,
        decoder_block_biases=weights.decoder_block_biases,
        final_conv_weight=weights.final_conv_weight,
        final_conv_bias=weights.final_conv_bias,
    )

    print(f"[atw_v2_cooperative_receiver_v2] rgb_0.shape={rgb_0.shape}")
    print(f"[atw_v2_cooperative_receiver_v2] rgb_1.shape={rgb_1.shape}")
    print(f"[atw_v2_cooperative_receiver_v2] rgb_0 range=[{rgb_0.min():.4f}, {rgb_0.max():.4f}]")
    print(f"[atw_v2_cooperative_receiver_v2] rgb_1 range=[{rgb_1.min():.4f}, {rgb_1.max():.4f}]")

    if args.output_json:
        result = {
            "substrate_id": "atw_v2_cooperative_receiver_v2",
            "version": "v2_phase_3_l0_scaffold_20260526",
            "smoke_only": True,
            "lane_id": "lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526",
            "rgb_0_shape": list(rgb_0.shape),
            "rgb_1_shape": list(rgb_1.shape),
            "rgb_0_range": [float(rgb_0.min()), float(rgb_0.max())],
            "rgb_1_range": [float(rgb_1.min()), float(rgb_1.max())],
            "evidence_grade": "macOS-MLX research-signal",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "phase_3_design_memo_path": ".omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526.md",
        }
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"[atw_v2_cooperative_receiver_v2] wrote smoke JSON to {args.output_json}")

    print("[atw_v2_cooperative_receiver_v2:_smoke_main] DONE")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """MLX-first score-aware full training via the canonical MLX harness.

    Routes through the substrate-AGNOSTIC harness binding real contest-video
    targets (Catalog #114) + gradient-reachable score-aware loss (reconstruction
    MSE + Hinton-KL T=2.0 scorer surrogate; Catalog #164) + canonical EMA /
    OOM-safe / telemetry / Provenance / posterior anchor via
    ``run_long_training``.

    The UNIQUE primitive is ``ATWv2CooperativeReceiverV2TrainableMLX`` (the
    cooperative-receiver + ego-motion FOE conditioning ``nn.Module``).
    """
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        ATWv2CooperativeReceiverV2TrainableMLX,
    )
    from tac.substrates.atw_v2_cooperative_receiver_v2.numpy_reference import (
        CooperativeReceiverConfig,
    )

    if args.output_dir is None:
        raise SystemExit(
            "--output-dir is required for --full training "
            "(Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS)."
        )

    cfg = CooperativeReceiverConfig(
        num_pairs=int(args.num_pairs),
        latent_dim=int(args.latent_dim),
    )
    model = ATWv2CooperativeReceiverV2TrainableMLX(cfg)
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=float(args.distillation_weight),
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="atw_v2_cooperative_receiver_v2",
        lane_id="lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        notes=(
            "ATW V2 cooperative-receiver V2 MLX-first score-aware full training "
            "via canonical mlx_score_aware harness; real contest video + "
            "reconstruction + Hinton-KL T=2.0 scorer surrogate + ego-motion FOE "
            "conditioning (Catalog #311); non-promotable [macOS-MLX "
            "research-signal] per Catalog #192/#317/#341; per-axis + MLX->PyTorch "
            "bridge + paired CUDA/CPU anchor DEFERRED to sister L2."
        ),
    )
    print(
        f"[atw_v2_cooperative_receiver_v2:_full_main] DONE "
        f"epochs={artifact.total_epochs_completed} promotable={artifact.promotable} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
