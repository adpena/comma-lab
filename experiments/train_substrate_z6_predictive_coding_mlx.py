# SPDX-License-Identifier: MIT
"""Z6 predictive-coding MLX-local trainer — L0 SCAFFOLD for $0 Apple-Silicon iteration.

Per OVERNIGHT Path 3 candidate #D (operator 2026-05-26 cascade: #1251 + #1257 +
#1258 corrected + #1265 gate) — MLX-local iteration is now contest-grade at
frontier-tightening granularity. This trainer is the FREE local iteration vehicle
for the Z6 substrate's previously-blocked cascade (paid Modal dispatch blocked
2026-05-17 per #768).

Purpose
-------

Train the Z6 FiLM-conditioned next-frame predictor + encoder + decoder on Apple
Silicon (MLX) at $0, then export to a contest Z6PCWM1 archive that:

1. Loads byte-stably into the PyTorch sister substrate
   :class:`tac.substrates.time_traveler_l5_z6.architecture.Z6PredictiveCodingSubstrate`
2. Inflates via :mod:`tac.substrates.time_traveler_l5_z6.inflate` to contest-grade raw frames
3. Passes the canonical #1265 contest-equivalence gate
   (``tools/gate_mlx_candidate_contest_equivalence.py``) before any paid CUDA dispatch

L0 SCAFFOLD scope (BOUNDED per operator directive)
--------------------------------------------------

- Smoke-only synthetic training (no pyav video decode required)
- Single-layer FiLM predictor (predictor_depth=1; Wave 2 BUILD multi-layer
  deferred to follow-up subagent)
- Bounded total params (target ~75K per Z6 design memo Section 10)
- Tiny epoch count for the smoke (≤5 epochs / ≤10 pairs)
- Output: .pt + Z6PCWM1 archive (NOT a paid-dispatch trainer)

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
seed) AND threads it through the FiLM predictor's modulation pipeline. For
synthetic smoke training the ego-motion has no semantic content; the
ego-motion gradient surface IS exercised, satisfying the canonical Catalog
#311 ego-motion-conditioning requirement at the structural level. Real-video
ego-motion (PoseNet-projected) is the responsibility of the PyTorch trainer
when promotion to paid CUDA dispatch becomes appropriate.

Usage
-----

Smoke:

    .venv/bin/python experiments/train_substrate_z6_predictive_coding_mlx.py \\
        --smoke \\
        --num-pairs 8 --epochs 3 \\
        --output-dir experiments/results/z6_mlx_smoke_$(date -u +%Y%m%dT%H%M%SZ)/

Full L0 SCAFFOLD (bounded; still no paid CUDA):

    .venv/bin/python experiments/train_substrate_z6_predictive_coding_mlx.py \\
        --num-pairs 600 --epochs 100 \\
        --output-dir experiments/results/z6_mlx_full_$(date -u +%Y%m%dT%H%M%SZ)/

The trainer produces:
- ``z6_mlx_state_dict.pt`` (PyTorch state_dict via #1251 bridge)
- ``0.bin`` (Z6PCWM1 archive ready for #1265 gate)
- ``training_manifest.json`` (canonical Provenance + per-epoch metrics)
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
        help="L0 SCAFFOLD only supports depth=1 (single-layer FiLM); depth>=2 deferred.",
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
        num_pairs = min(args.num_pairs, 8)
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

    # Synthetic per-pair targets: random unit-range RGB. The MLX trainer is
    # smoke-only; real-video targets are the PyTorch trainer's responsibility.
    target_rgb_0_np = np.random.RandomState(args.seed + 200).rand(
        cfg.num_pairs, cfg.output_height, cfg.output_width, 3,
    ).astype(np.float32)
    target_rgb_1_np = np.random.RandomState(args.seed + 300).rand(
        cfg.num_pairs, cfg.output_height, cfg.output_width, 3,
    ).astype(np.float32)
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

    per_epoch_metrics: list[dict[str, float]] = []
    t_start = time.time()
    for epoch in range(epochs):
        loss_val, grads = loss_grad(renderer)
        optimizer.update(renderer, grads)
        mx.eval(renderer.parameters(), optimizer.state)
        loss_scalar = float(loss_val.item())
        per_epoch_metrics.append({
            "epoch": int(epoch),
            "loss": loss_scalar,
            "wall_clock_seconds": float(time.time() - t_start),
        })
        print(
            f"[z6-mlx-trainer] epoch {epoch+1}/{epochs} "
            f"loss={loss_scalar:.6f} "
            f"wall={time.time() - t_start:.1f}s"
        )
    total_wall = time.time() - t_start

    # Export state_dict .pt via canonical #1251 bridge
    pt_path = args.output_dir / "z6_mlx_state_dict.pt"
    print(f"[z6-mlx-trainer] Exporting .pt to {pt_path}")
    pt_manifest = build_z6_pytorch_pt_from_mlx_renderer(
        renderer, pt_path, overwrite=True,
    )

    # Build Z6PCWM1 archive (ready for #1265 gate)
    archive_path = args.output_dir / "0.bin"
    print(f"[z6-mlx-trainer] Building Z6PCWM1 archive at {archive_path}")
    arc_manifest = build_z6pcwm1_archive_from_mlx_renderer(
        renderer, archive_path, overwrite=True,
        lambda_residual_entropy=args.lambda_residual,
    )

    # Build training manifest (canonical Provenance per Catalog #287/#323)
    training_manifest = {
        "schema_version": f"{SCHEMA_VERSION}_training_manifest",
        "substrate_id": "time_traveler_l5_z6",
        "lane_id": LANE_ID,
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
        },
        "parameter_breakdown": breakdown,
        "per_epoch_metrics": per_epoch_metrics,
        "total_wall_clock_seconds": total_wall,
        "outputs": {
            "pt_manifest": pt_manifest,
            "archive_manifest": arc_manifest,
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
            "synthetic_mse_proxy_loss_non_promotable_per_catalog_287",
            "requires_pass_verdict_from_tools_gate_mlx_candidate_contest_equivalence",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_any_score_claim",
        ],
        "operator_routable_next_steps": [
            (
                f"Run gate: .venv/bin/python tools/gate_mlx_candidate_contest_equivalence.py "
                f"--archive-zip {archive_path} --candidate-label z6_predictive_coding "
                f"--output-json {args.output_dir / 'gate_verdict.json'}"
            ),
            (
                "On PASS: operator routes paired CUDA dispatch via "
                "experiments/train_substrate_time_traveler_l5_z6.py + "
                "tools/operator_authorize.py per Catalog #313"
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
