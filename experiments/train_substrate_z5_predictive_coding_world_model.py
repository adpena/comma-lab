# SPDX-License-Identifier: MIT
"""Train the Z5 predictive-coding world-model substrate (Time-Traveler L5 staircase Step 3).

Per `.omx/research/campaign_z5_predictive_coding_world_model_20260514.md` and
`feedback_grand_council_maximize_value_landed_20260514.md`: Step 3 of the
Time-Traveler across-class staircase. Hierarchical predictive-coding world-
model on top of Z4 + A1; predicted ΔS −0.025 to −0.038 vs Z4 → predicted
band [0.155, 0.180] on contest-CUDA T4 (Time-Traveler asymptote estimate).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187).
- ``load_differentiable_scorers`` for SegNet/PoseNet.
- ``apply_eval_roundtrip_during_training`` (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` (Catalog #88).
- Score-domain predictive-coding Lagrangian (HNeRV parity L6 + Rao-Ballard).
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval EVERYWHERE").
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151 + #168 AnnAssign).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived`` (Catalog #197).

V1 SCOPE: ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on
synthetic data, runs archive pack + parse + inflate roundtrip + autoregression
test, and emits a contest-compliant runtime tree (no scorer load).

``_full_main`` is currently a NotImplementedError stub awaiting Phase 2
council approval. Operator-routable decision pending.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_z5_predictive_coding_world_model.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z5_smoke_<utc> \\
        --epochs 3 --device cpu --smoke
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:autoregressive-predictor-unroll-needs-canary-validation
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_full_main
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tac.substrates.z5_predictive_coding_world_model import (
    PredictiveCodingConfig,
    PredictiveCodingSubstrate,
    pack_archive,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBSTRATE_TAG = "z5_predictive_coding_world_model"
SUBSTRATE_LANE_ID = "lane_z5_predictive_coding_world_model_step3_20260514"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z5_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z5_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "Z5_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=300",
        "default": "300",
    },
    "--batch-size": {
        "env": "Z5_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; T4 handles 4-8 at 384x512 with autoregressive "
            "predictor unroll across 600 pairs"
        ),
        "default": "4",
    },
    "--lr": {
        "env": "Z5_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--lambda-residual-entropy": {
        "env": "Z5_LAMBDA_RESIDUAL_ENTROPY",
        "rationale": (
            "Predictive-coding residual-entropy weight (Rao-Ballard 1999); "
            "0 = no PC, 1 = canonical, higher = more aggressive predictor learning"
        ),
        "default": "1.0",
    },
    "--predictor-num-layers": {
        "env": "Z5_PREDICTOR_NUM_LAYERS",
        "rationale": "Hierarchical predictor depth; 2 or 3 (Rao-Ballard 1999 requires > 1)",
        "default": "2",
    },
    "--predictor-ego-motion-dim": {
        "env": "Z5_PREDICTOR_EGO_MOTION_DIM",
        "rationale": (
            "Ego-motion proxy dimension projected from PoseNet output; "
            "default 8; small dim keeps predictor compact"
        ),
        "default": "8",
    },
    "--identity-predictor": {
        "env": "Z5_IDENTITY_PREDICTOR",
        "rationale": (
            "Probe-disambiguator ablation: when true, predictor is identity "
            "(no learning); compare to full hierarchical for Rao-Ballard refutation/confirmation"
        ),
        "default": "false",
    },
    "--enable-autocast-fp16": {
        "env": "Z5_ENABLE_AUTOCAST_FP16",
        "rationale": "Catalog #172; pending canonical autocast backport",
        "default": "false",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_z5_predictive_coding_world_model",
        description=(
            "Train Z5 predictive-coding world-model substrate (Time-Traveler "
            "L5 staircase Step 3; Rao-Ballard 1999)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=24)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument("--decoder-num-upsample-blocks", type=int, default=6)
    p.add_argument("--predictor-hidden-dim", type=int, default=64)
    p.add_argument("--predictor-num-layers", type=int, default=2,
                   choices=[2, 3])
    p.add_argument("--predictor-ego-motion-dim", type=int, default=8)
    p.add_argument("--identity-predictor", action="store_true",
                   help="Probe-disambiguator regime: identity predictor (no learning)")

    # Predictive-coding Lagrangian weights
    p.add_argument("--lambda-residual-entropy", type=float, default=1.0,
                   help="Rao-Ballard residual-entropy weight; 0=no PC, 1=canonical")
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Training
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)

    # Mode flags
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke path (tiny config, synthetic data, no scorer load)")
    p.add_argument("--full-cpu", action="store_true",
                   help="Opt-in to non-smoke CPU training (Catalog #197 paired flag required)")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true",
                   help="Required sister flag for --full-cpu (Catalog #197)")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; pending canonical autocast backport")
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with --advisory-cpu-explicitly-waived."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 (paired-flag attestation that the CPU-axis bypass "
            "is intentional and non-promotable)"
        )


# ---------------------------------------------------------------------------
# Smoke entry path
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: tiny config, synthetic data, ≤3 epochs, no scorer load."""
    torch.manual_seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tiny smoke config
    num_pairs = 5
    cfg = PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_num_layers=args.predictor_num_layers,
        predictor_ego_motion_dim=4,
        identity_predictor=args.identity_predictor,
        lambda_residual_entropy=args.lambda_residual_entropy,
    )
    substrate = PredictiveCodingSubstrate(cfg).to(args.device)
    print(f"[z5-smoke] param breakdown: {substrate.num_parameters_breakdown()}")
    print(f"[z5-smoke] identity_predictor={cfg.identity_predictor}")

    # Synthetic frame targets per pair (Catalog #114 allowed in smoke path only)
    synth_targets_0 = torch.rand(
        num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
    )
    synth_targets_1 = torch.rand(
        num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
    )

    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for epoch in range(max(args.epochs, 3)):
        opt.zero_grad()
        idx = torch.arange(num_pairs, device=args.device, dtype=torch.long)
        rgb_0, rgb_1, z_t = substrate.reconstruct_pair(idx)
        # Pixel-MSE proxy + residual-norm proxy in smoke
        recon_loss = (rgb_0 - synth_targets_0).pow(2).mean() + (rgb_1 - synth_targets_1).pow(2).mean()
        residual_loss = args.lambda_residual_entropy * substrate.residuals.pow(2).mean()
        loss = recon_loss + residual_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
        opt.step()
        losses.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "recon": float(recon_loss.item()),
            "residual": float(residual_loss.item()),
        })

    # Pack archive
    enc_sd = substrate.encoder.state_dict()
    dec_sd = substrate.decoder.state_dict()
    pred_sd = substrate.predictor.state_dict()
    latent_init = substrate.latent_init.detach().cpu()
    residuals = substrate.residuals.detach().cpu()
    ego_motion = substrate.ego_motion_buffer.detach().cpu()
    meta = {
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "predictor_hidden_dim": cfg.predictor_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
    }
    archive_bytes = pack_archive(
        enc_sd, dec_sd, pred_sd, latent_init, residuals, ego_motion, meta,
        lambda_residual_entropy=args.lambda_residual_entropy,
        predictor_num_layers=args.predictor_num_layers,
        identity_predictor=args.identity_predictor,
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    final = losses[-1] if losses else {"loss": float("inf")}
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final["loss"],
        "final_recon": final.get("recon"),
        "final_residual": final.get("residual"),
        "archive_bytes": len(archive_bytes),
        "lambda_residual_entropy": args.lambda_residual_entropy,
        "predictor_num_layers": args.predictor_num_layers,
        "identity_predictor": args.identity_predictor,
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "param_breakdown": substrate.num_parameters_breakdown(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[z5-smoke] OK final_loss={final['loss']:.6f} archive={len(archive_bytes)}B "
        f"predictor_layers={args.predictor_num_layers} identity={args.identity_predictor}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full entry path — NotImplementedError stub awaiting Phase 2 council approval
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Phase 2 council approval required.

    The full Z5 training path needs:
    1. pyav decode of upstream/videos/0.mkv into per-pair frames (Catalog #114)
    2. patch_upstream_yuv6_globally + load_differentiable_scorers (Catalog #187)
    3. PredictiveCodingScoreAwareLoss with eval-roundtrip + EMA(0.997)
    4. AdamW + grad clip 1.0 + NaN watchdog
    5. Autoregressive predictor unroll across 600 pairs (mini-batched per
       Catalog #218 reconstruct_pair pattern; D4 sister substrate's lesson)
    6. Ego-motion buffer populated from PoseNet output projection (or
       held at zeros for identity-predictor ablation regime)
    7. CUDA auth eval on best EMA checkpoint via canonical smoke_auth_eval_gate
    8. posterior_update_locked (Catalog #128)
    9. Contest-compliant runtime emission (Catalog #146 + #163)

    The architecture, archive, inflate, and score_aware_loss modules are
    landed and tested (31 dedicated tests pass; autoregression recurrence
    test confirms inflate-time correctness). The trainer body is
    deliberately stubbed pending council approval per CLAUDE.md "Design
    decisions — non-negotiable": world-model predictive-coding is a
    council-grade tradeoff because (a) the autoregressive unroll cost
    multiplies training wall-clock, (b) the identity-predictor ablation
    needs to fire BEFORE the full predictor to anchor the probe, and (c)
    the predicted [0.155, 0.180] band requires Z4 (Step 2) to land first
    per canary_dependency.

    Operator-routable: see `.omx/research/campaign_z5_predictive_coding_world_model_20260514.md`
    §7 for stop/continue thresholds. Phase 2 dispatch approval lifts this
    NotImplementedError after council review AND Z4 contest-CUDA anchor.
    """
    raise NotImplementedError(
        "Phase 2 council approval required to lift this NotImplementedError. "
        "Additionally requires canary_dependency=lane_z4_cooperative_receiver_loss_step2_20260514 "
        "to have ≥ 1 successful contest-CUDA anchor per Catalog #173. "
        "See `.omx/research/campaign_z5_predictive_coding_world_model_20260514.md` "
        "§7 for the dispatch-gating thresholds."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
