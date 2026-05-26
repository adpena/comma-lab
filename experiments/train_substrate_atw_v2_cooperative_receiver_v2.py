# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — MLX smoke trainer (Path 3 candidate H).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Per Phase 3 design memo §10 (L0 SCAFFOLD package structure) + Catalog
#240(c) (_full_main raises NotImplementedError pre-Phase 4 council).

L0 SCAFFOLD scope
=================

- _smoke_main: end-to-end MLX forward smoke (encoder + cond_embed + decoder
  via numpy reference as parity check); ~1-3 min wall-clock on macOS.
- _full_main: raises NotImplementedError per Catalog #240(c). Phase 4
  council approval required to lift.

Non-promotable canonical contract per CLAUDE.md "MLX portable-local-substrate
authority":

- Tagged [macOS-MLX research-signal] at landing
- score_claim=False, promotion_eligible=False
- Phase 4 promotion path: MLX state_dict → PyTorch (via #1251 bridge) →
  archive → contest-equivalence gate per Catalog #1265 → operator routes
  paid CUDA dispatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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

    # Build deterministic input
    B = args.batch_size
    per_pair_latent_residual = np.zeros((B, cfg.latent_dim), dtype=np.float32)

    # Generate per-pair pose_delta + ego-motion FOE projection
    rng = np.random.default_rng(args.seed)
    pose_delta = rng.normal(0, 0.1, size=(B, 6)).astype(np.float32)
    ego_motion_proj = numpy_ego_motion_foe_projection(pose_delta)

    # Forward pass
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
    """Full training entry point — raises NotImplementedError per Catalog #240(c).

    Phase 4 council approval required to lift. Pending:
    1. D4-equivalent probe for ego-motion conditioning surface
       (`tools/probe_atw_v2_cooperative_receiver_ego_motion_conditioning_disambiguator.py`)
    2. PyTorch training loss via canonical `tac.codec.cooperative_receiver.
       atick_redlich.cooperative_receiver_loss` per Catalog #164
    3. MLX→PyTorch state_dict export bridge per #1251 pattern
    4. Catalog #1265 contest-equivalence gate verification
    5. Per-substrate symposium evidence per Catalog #325
    """
    raise NotImplementedError(
        "atw_v2_cooperative_receiver_v2._full_main requires Phase 4 council approval "
        "per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' + "
        "Catalog #240(c). L0 SCAFFOLD provides MLX-native forward + numpy reference + "
        "PyTorch parity reference + archive grammar + inflate runtime + smoke trainer; "
        "PyTorch training loss + D4-equivalent probe + Catalog #1265 gate verification "
        "are the Phase 4 deliverables."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run smoke-mode forward pass")
    parser.add_argument("--full", action="store_true", help="Run full training (NotImplementedError per Catalog #240(c))")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--output-json", type=str, default=None, help="Optional smoke result JSON path")
    args = parser.parse_args(argv)

    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
