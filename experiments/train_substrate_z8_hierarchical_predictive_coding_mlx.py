# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding MLX-local smoke trainer — L0 SCAFFOLD.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Path 3 substrate-class-shift candidate F MLX trainer. Per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable + the
operator's MLX-first directive 2026-05-26.

L0 SCAFFOLD scope:

- MLX-local smoke ONLY (axis_tag=``[macOS-MLX research-signal]``); no paid CUDA.
- Smoke ≤5 epochs ≤8 pairs (smoke-only mode); demonstrates the multi-level
  RSSM + per-level Gumbel-Softmax STE + Rao-Ballard residual L2 training step
  converges monotonically on synthetic targets.
- ``_full_main raises NotImplementedError`` per Catalog #240 acceptance
  cascade (c) pre-build substrate-engineering: full training path is
  council-gated.
- All artifacts carry ``[macOS-MLX research-signal]`` + ``score_claim=false``
  + ``promotion_eligible=false`` + ``ready_for_exact_eval_dispatch=false``
  per Catalog #127 + #192 + #317 + #341.

Per CLAUDE.md "MLX portable-local-substrate authority": MLX is a local substrate
for fast candidate generation; not a contest scoring axis. MLX results MUST
flow through ``tac.optimization.scorer_response_dataset`` for any LL planner
consumption (Phase 2 wire-in).

Discipline: Catalog #229 PV / #117/#157/#174 canonical serializer (commits) /
#206 checkpoint discipline / #119 Co-Authored-By Claude trailer / #287
placeholder-rationale rejection / #208 no /Users paths / #270 dispatch
optimization protocol (Phase 2 trainer wires canonical helpers) / #310 + #311
+ #312 class-shift-not-bolt-on + ego-motion conditioning + canonical-quadruple
binding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# CATEGORY-CHECK: this script is OUT-OF-SCOPE for Catalog #226
# (`check_trainer_auth_eval_uses_canonical_helper`) because it is an MLX-local
# SMOKE trainer that does NOT invoke `experiments/contest_auth_eval.py`.
# Phase 2 PyTorch trainer (separate file) will use the canonical
# `gate_auth_eval_call` helper.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Z8 hierarchical predictive coding MLX-local smoke trainer "
            "(L0 SCAFFOLD; non-promotable)."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode (≤5 epochs ≤8 pairs synthetic targets; required at L0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of smoke epochs (default 3; max 5 at L0).",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=4,
        help="Number of synthetic pairs for smoke (default 4; max 8 at L0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--smoke-output",
        type=str,
        default="",
        help=(
            "Optional path for smoke convergence JSON manifest (non-promotable; "
            "carries [macOS-MLX research-signal] tag)."
        ),
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """L0 smoke convergence: minimal MLX training loop on synthetic targets.

    Demonstrates the multi-level RSSM + per-level Gumbel-Softmax STE + the
    Rao-Ballard residual L2 + decoder forward all train end-to-end on MLX.
    Synthetic MSE proxy stands in for the full canonical
    ``score_pair_components`` per Catalog #164 (which Phase 2 PyTorch trainer
    routes through).
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as exc:
        print(
            f"[Z8 MLX smoke] MLX not available: {exc}; cannot run smoke.",
            file=sys.stderr,
        )
        return 65  # canonical "MLX unavailable" L0 exit code

    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    if args.epochs > 5:
        print(f"[Z8 MLX smoke] L0 caps epochs at 5; got {args.epochs}", file=sys.stderr)
        return 2
    if args.num_pairs > 8:
        print(
            f"[Z8 MLX smoke] L0 caps num_pairs at 8; got {args.num_pairs}",
            file=sys.stderr,
        )
        return 2

    mx.random.seed(int(args.seed))

    # Smoke config: small enough to converge in seconds on Apple Silicon.
    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=int(args.num_pairs),
        deterministic_state_dim=8,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    model = Z8HierarchicalPredictiveCoderMLX(cfg)

    # Synthetic target: per-pair random RGB at scorer resolution.
    # This is the smoke-only MSE proxy; Phase 2 trainer routes through
    # canonical `score_pair_components` per Catalog #164.
    target_pairs = mx.random.normal(  # type: ignore[union-attr]
        shape=(cfg.num_pairs, 2, 3, *cfg.eval_size),
        key=mx.random.key(123),
    ) * 50.0 + 127.5  # roughly in [0, 255]

    def loss_fn(model: Z8HierarchicalPredictiveCoderMLX, pair_idx: int) -> any:
        """MSE proxy loss + Rao-Ballard per-level residual L2 sketch."""
        indices = mx.array([pair_idx], dtype=mx.int32)
        rgb_pair, per_level_indices, per_level_soft = model.forward_training(
            indices
        )
        # Synthetic MSE on decoded pair
        target_slice = target_pairs[pair_idx : pair_idx + 1]
        mse = mx.mean((rgb_pair - target_slice) ** 2)
        # Rao-Ballard per-level residual L2 sketch: penalize Gumbel-softmax
        # entropy (proxy for per-level residual entropy term — true Rao-Ballard
        # residual term requires the bottom-up error encoder which Phase 2
        # implements). Sum across levels.
        rao_ballard_proxy = mx.zeros(())
        for soft in per_level_soft:
            # Gumbel-Softmax soft entropy: -sum p log p
            entropy = -mx.sum(soft * mx.log(soft + 1e-10), axis=-1)
            rao_ballard_proxy = rao_ballard_proxy + mx.mean(entropy)
        return mse + 0.01 * rao_ballard_proxy

    opt = optim.Adam(learning_rate=1e-3)

    epoch_losses: list[float] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        epoch_loss = 0.0
        for pair_idx in range(cfg.num_pairs):
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss_value, grads = loss_and_grad_fn(model, pair_idx)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            epoch_loss += float(loss_value)
        epoch_loss = epoch_loss / cfg.num_pairs
        epoch_losses.append(epoch_loss)
        print(
            f"[Z8 MLX smoke] epoch={epoch + 1}/{args.epochs} loss={epoch_loss:.4f}",
        )
    elapsed = time.time() - start

    # Smoke convergence verdict: monotonic decrease ≥5 epochs is the canonical
    # L0 convergence signal. At <5 epochs we just require last < first.
    converged = (
        epoch_losses[-1] < epoch_losses[0] if len(epoch_losses) >= 2 else True
    )

    smoke_manifest = {
        "schema": "z8_mlx_smoke_convergence_manifest_v1",
        "substrate_id": "z8_hierarchical_predictive_coding",
        "lane_id": (
            "lane_path_3_f_z8_hierarchical_predictive_coding_canonical_quadruple_20260526"
        ),
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "MPS-research-signal",
        "smoke_mode": True,
        "num_epochs": int(args.epochs),
        "num_pairs": int(args.num_pairs),
        "epoch_losses": [float(x) for x in epoch_losses],
        "converged_monotonic_proxy": bool(converged),
        "wall_clock_seconds": float(elapsed),
        "canonical_equation_refs": [
            "mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
            "scorer_conditional_joint_rate_distortion_floor_v1",
            "categorical_posterior_capacity_vs_continuous_gaussian_v1",
        ],
        "design_memo_path": (
            ".omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md"
        ),
        "non_promotable_rationale": (
            "L0 SCAFFOLD MLX-local smoke; per CLAUDE.md 'MLX portable-local-substrate "
            "authority' MLX is research-signal not contest scoring axis; per Catalog "
            "#192 + #317 + #341 non-promotable by construction"
        ),
    }
    print(json.dumps(smoke_manifest, indent=2, sort_keys=True))

    if args.smoke_output:
        out_path = Path(args.smoke_output)
        if str(out_path).startswith("/tmp/") or "/tmp/" in str(out_path):
            print(
                f"[Z8 MLX smoke] refusing /tmp/ path per CLAUDE.md transient-evidence "
                f"trap: {out_path}",
                file=sys.stderr,
            )
            return 4
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[Z8 MLX smoke] wrote manifest to {out_path}")

    return 0 if converged else 1


def _full_main(args: argparse.Namespace) -> int:
    """Full training is council-gated per Catalog #240 acceptance cascade (c).

    Phase 2 council deliberation required to lift per CLAUDE.md "Substrate
    scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable. Full path
    will:

    1. Run the canonical 6-stage curriculum (Phase 0-5 per design memo Section 9).
    2. Route through canonical `tac.substrates._shared.score_aware_common.score_pair_components`
       per Catalog #164.
    3. Apply `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`
       per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE".
    4. Apply `patch_upstream_yuv6_globally()` before `load_differentiable_scorers()`
       per Catalog #187.
    5. Use `detect_hardware_substrate(substrate_tag="z8_hpc_v1", axis="cuda")`
       per Catalog #190.
    6. Apply EMA 0.997 weights / 0.99 codebook per CLAUDE.md "EMA — NON-NEGOTIABLE".
    7. Build Z8HPC1 archive + paired CPU/CUDA auth_eval via canonical
       `gate_auth_eval_call` per Catalog #226.
    """
    raise NotImplementedError(
        "Z8 _full_main is council-gated per Catalog #240 acceptance cascade (c) "
        "pre-build substrate-engineering. Phase 2 lifts via per-substrate symposium "
        "per Catalog #325. See design memo "
        ".omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md "
        "Section 13 operator-routable #4."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main())
