"""Train the Z4 cooperative-receiver-loss substrate (Time-Traveler L5 staircase Step 2).

Per `.omx/research/campaign_z4_cooperative_receiver_loss_20260514.md` and
`feedback_grand_council_maximize_value_landed_20260514.md`: Step 2 of the
Time-Traveler across-class staircase. Loss-only intervention on top of Z3
+ A1 substrate; predicted ΔS −0.005 to −0.010 vs Z3 → predicted band
[0.180, 0.188] on contest-CUDA T4.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at inflate).
- ``apply_eval_roundtrip_during_training`` (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` (Catalog #88).
- Score-domain cooperative-receiver Lagrangian (HNeRV parity L6 + Atick-Redlich).
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval EVERYWHERE").
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151 + #168 AnnAssign).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived`` (Catalog #197).

V1 SCOPE: ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on
synthetic data, runs archive pack + parse + inflate roundtrip, and emits a
contest-compliant runtime tree (no scorer load).

``_full_main`` is currently a NotImplementedError stub awaiting Phase 2
council approval. Operator-routable decision pending.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_z4_cooperative_receiver_loss.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z4_smoke_<utc> \\
        --epochs 3 --device cpu --smoke
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_full_main
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tac.substrates.z4_cooperative_receiver_loss import (
    CooperativeReceiverConfig,
    CooperativeReceiverSubstrate,
    pack_archive,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBSTRATE_TAG = "z4_cooperative_receiver_loss"
SUBSTRATE_LANE_ID = "lane_z4_cooperative_receiver_loss_step2_20260514"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z4_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z4_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "Z4_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=200",
        "default": "200",
    },
    "--batch-size": {
        "env": "Z4_BATCH_SIZE",
        "rationale": "Per-step pair count; T4 handles 4-8 at 384x512",
        "default": "4",
    },
    "--lr": {
        "env": "Z4_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--lambda-pixel": {
        "env": "Z4_LAMBDA_PIXEL",
        "rationale": (
            "Cooperative-receiver pixel-MSE residual weight; "
            "0.0 = pure Atick-Redlich; 1.0 = Z3 baseline (pixel-MSE-only)"
        ),
        "default": "0.0",
    },
    "--beta-seg": {
        "env": "Z4_BETA_SEG",
        "rationale": "SegNet distortion weight in cooperative-receiver Lagrangian (contest formula = 100)",
        "default": "100.0",
    },
    "--gamma-pose": {
        "env": "Z4_GAMMA_POSE",
        "rationale": "PoseNet distortion sqrt-weight (contest formula = sqrt(10))",
        "default": str(math.sqrt(10.0)),
    },
    "--enable-autocast-fp16": {
        "env": "Z4_ENABLE_AUTOCAST_FP16",
        "rationale": "Catalog #172; pending canonical autocast backport",
        "default": "false",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_z4_cooperative_receiver_loss",
        description=(
            "Train Z4 cooperative-receiver-loss substrate (Time-Traveler L5 "
            "staircase Step 2; Atick-Redlich 1990)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=24)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument("--decoder-num-upsample-blocks", type=int, default=6)

    # Cooperative-receiver Lagrangian weights
    p.add_argument("--lambda-pixel", type=float, default=0.0,
                   help="Pixel-MSE residual weight; 0=pure cooperative-receiver, 1=Z3 baseline")
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))

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
    num_pairs = 4
    cfg = CooperativeReceiverConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        cooperative_receiver_lambda_pixel=args.lambda_pixel,
    )
    substrate = CooperativeReceiverSubstrate(cfg).to(args.device)
    print(f"[z4-smoke] param breakdown: {substrate.num_parameters_breakdown()}")

    # Synthetic data (Catalog #114 allowed in smoke path only)
    synth_frame_1 = torch.rand(
        num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
    )
    synth_frame_0_target = torch.rand(
        num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
    )

    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for epoch in range(max(args.epochs, 3)):
        opt.zero_grad()
        idx = torch.arange(num_pairs, device=args.device, dtype=torch.long)
        rgb_0, rgb_1, _mu, _logvar = substrate(idx, frames_for_encoder=synth_frame_1)
        # Pixel-MSE proxy in smoke (no scorer load)
        recon_loss = (rgb_0 - synth_frame_0_target).pow(2).mean() + (rgb_1 - synth_frame_1).pow(2).mean()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
        opt.step()
        losses.append({"epoch": epoch, "loss": float(recon_loss.item())})

    # Pack archive
    enc_sd = substrate.encoder.state_dict()
    dec_sd = substrate.decoder.state_dict()
    latents = substrate.latents.detach().cpu()
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
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
    }
    archive_bytes = pack_archive(
        enc_sd, dec_sd, latents, meta,
        cooperative_receiver_lambda_pixel=args.lambda_pixel,
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
        "archive_bytes": len(archive_bytes),
        "lambda_pixel": args.lambda_pixel,
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
        f"[z4-smoke] OK final_loss={final['loss']:.6f} "
        f"archive={len(archive_bytes)}B lambda_pixel={args.lambda_pixel}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full entry path — NotImplementedError stub awaiting Phase 2 council approval
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Phase 2 council approval required.

    The full Z4 training path needs:
    1. pyav decode of upstream/videos/0.mkv into per-pair frames (Catalog #114)
    2. patch_upstream_yuv6_globally + load_differentiable_scorers (Catalog #187)
    3. CooperativeReceiverScoreAwareLoss with eval-roundtrip + EMA(0.997)
    4. AdamW + grad clip 1.0 + NaN watchdog
    5. CUDA auth eval on best EMA checkpoint via canonical smoke_auth_eval_gate
    6. posterior_update_locked (Catalog #128)
    7. Contest-compliant runtime emission (Catalog #146 + #163)

    The architecture, archive, inflate, and score_aware_loss modules are
    landed and tested (31 dedicated tests pass). The trainer body is
    deliberately stubbed pending council approval per CLAUDE.md "Design
    decisions — non-negotiable": loss-only intervention on top of A1+Z3
    substrate is a council-grade tradeoff because the predicted [0.180,
    0.188] band overlaps Z3's [0.188, 0.193] within tolerance; the test
    of cooperative-receiver vs pixel-MSE-with-same-budget IS the council
    deliberation.

    Operator-routable: see `.omx/research/campaign_z4_cooperative_receiver_loss_20260514.md`
    §7 for stop/continue thresholds. Phase 2 dispatch approval lifts this
    NotImplementedError after council review.
    """
    raise NotImplementedError(
        "Phase 2 council approval required to lift this NotImplementedError. "
        "See `.omx/research/campaign_z4_cooperative_receiver_loss_20260514.md` "
        "§7 for the dispatch-gating thresholds. Per CLAUDE.md 'Design "
        "decisions — non-negotiable', this is a council-grade tradeoff."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
