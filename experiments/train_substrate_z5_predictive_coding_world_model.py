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

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _canon_git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    torch_version_string as _canon_torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates.z5_predictive_coding_world_model import (
    PredictiveCodingConfig,
    PredictiveCodingSubstrate,
    pack_archive,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
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
    p.add_argument("--enable-torch-compile", action="store_true",
                   help="Catalog #179; pending autoregressive-predictor canary validation")
    p.add_argument("--enable-gt-scorer-cache", action="store_true",
                   help="Catalog #228; GT-scorer-output cache")
    # Full-loop flags (CLASS-SHIFT-FULL-MAIN-CLUSTER pattern).
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap decoded pairs (full default = 600 contest pairs).")
    p.add_argument("--val-pair-count", type=int, default=64,
                   help="Held-out validation pair count.")
    p.add_argument("--val-every-epochs", type=int, default=10,
                   help="Run validation + best-checkpoint every N epochs.")
    p.add_argument("--skip-archive-build", action="store_true", default=False,
                   help="Skip 0.bin + archive.zip emission (training-only diagnostic).")
    p.add_argument("--skip-auth-eval", action="store_true", default=False,
                   help="Skip CUDA auth eval (training-only diagnostic).")
    return p


def _resolve_full_device(args: argparse.Namespace):
    """Resolve the full-mode device with the CLAUDE.md "MPS is NOISE" gate.

    Non-smoke training is CUDA-required (Catalog #1 + #325) UNLESS the operator
    explicitly opts into the non-promotable CPU advisory path via the paired
    ``--full-cpu`` + ``--advisory-cpu-explicitly-waived`` flags (Catalog #197;
    already validated by ``_validate_full_cpu_flags``). MPS is rejected.
    """
    import torch

    name = str(args.device).lower()
    if name == "mps":
        raise SystemExit(
            "ERROR: --device mps is FORBIDDEN per CLAUDE.md 'MPS auth eval is "
            "NOISE'. Full training is CUDA-required; use --device cuda."
        )
    if name == "cpu":
        if not getattr(args, "full_cpu", False):
            raise SystemExit(
                "ERROR: full (non-smoke) z5 training is CUDA-required per "
                "CLAUDE.md 'MPS auth eval is NOISE' + Catalog #1/#325. Pass "
                "--device cuda, OR opt into the non-promotable CPU advisory "
                "path via --full-cpu --advisory-cpu-explicitly-waived "
                "(Catalog #197)."
            )
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: --device cuda requested but torch.cuda.is_available() is "
            "False. Full training requires a real CUDA device."
        )
    return torch.device(name)


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
    """Full score-aware training entry point — CUDA-required; paid-GPU gated.

    CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: routes the substrate-AGNOSTIC
    training loop through the canonical
    ``tac.substrates._shared.pact_nerv_full_main`` helper; the UNIQUE Z5
    distinguishing feature (Rao-Ballard hierarchical predictive coding:
    encoder + autoregressive ego-motion predictor + per-pair residual codes;
    Atick-Redlich cooperative-receiver framing) stays in this substrate's
    architecture + archive + score-aware loss. The ``reconstruct_pair`` adapter
    threads the per-pair ``residuals`` entropy term into the loss; the archive
    partitions encoder/decoder/predictor submodules + latent_init + residuals +
    ego_motion. The ``NotImplementedError`` is extinguished; PAID DISPATCH is
    still gated by ``dispatch_enabled: false`` + ``research_only: true`` on the
    recipe per Catalog #325 (code complete, trigger gated).

    Honored end-to-end: real contest video (Catalog #114); patch yuv6 BEFORE
    scorer construction (eval_roundtrip non-negotiable, Catalog #187);
    ``load_differentiable_scorers`` (no scorer at inflate); score-domain
    Lagrangian + Rao-Ballard residual-entropy term via the variant loss;
    EMA shadow (Quantizr 0.997); CUDA-required (``_resolve_full_device``
    rejects MPS per Catalog #1 + non-promotable --full-cpu opt-in per Catalog
    #197); CUDA auth-eval via canonical ``gate_auth_eval_call`` (Catalog #226);
    posterior-update via ``posterior_update_locked`` (Catalog #128);
    contest-compliant runtime (Catalog #146 + #295).

    Reactivation criteria for PAID DISPATCH (per HNeRV parity L2 + the campaign
    memo §7): per-substrate symposium operator-gated approval (Catalog #325) +
    Z4 cooperative-receiver canary anchor (canary_dependency) + identity-
    predictor ablation probe; recipe ``research_only`` flips to false +
    ``dispatch_enabled`` to true. See
    ``.omx/research/campaign_z5_predictive_coding_world_model_20260514.md`` §7.
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates._shared.pact_nerv_full_main import (
        build_archive_zip,
        closed_form_weight_byte_proxy,
        decode_pairs_for_training,
        run_pact_nerv_score_aware_training,
        write_contest_runtime,
    )
    from tac.substrates.z5_predictive_coding_world_model import (
        PredictiveCodingLossWeights,
        PredictiveCodingScoreAwareLoss,
    )

    torch.manual_seed(args.seed)
    device = _resolve_full_device(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []
    yuv6_token = patch_upstream_yuv6_globally()
    try:
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()

        print(f"[full:{SUBSTRATE_TAG}] decoding pairs from {args.video_path} ...")
        pair_tensor = decode_pairs_for_training(
            args.video_path, substrate_tag=SUBSTRATE_TAG, n_pairs=N_PAIRS_FULL,
            max_pairs=args.max_pairs, repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full:{SUBSTRATE_TAG}] decoded {n_pairs} pairs at {EVAL_HW}")

        cfg = PredictiveCodingConfig(
            latent_dim=args.latent_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_num_upsample_blocks=args.decoder_num_upsample_blocks,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_num_layers=args.predictor_num_layers,
            predictor_ego_motion_dim=args.predictor_ego_motion_dim,
            identity_predictor=args.identity_predictor,
            lambda_residual_entropy=args.lambda_residual_entropy,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0], output_width=EVAL_HW[1],
        )
        model = PredictiveCodingSubstrate(cfg).to(device)
        print(f"[full:{SUBSTRATE_TAG}] params: {model.num_parameters():,}")

        weights = PredictiveCodingLossWeights(
            alpha_rate=args.alpha_rate, beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose, pose_weight_scale=args.pose_weight_scale,
            lambda_residual_entropy=args.lambda_residual_entropy,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = PredictiveCodingScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )

        opt_ctx = _canon_build_optimized_training_context(
            args, scorers=(posenet, segnet), gt_pairs=pair_tensor,
            substrate_model=model, device=device,
        )
        gt_cache = opt_ctx.gt_cache
        # Rate term covers shipped encoder + decoder + predictor + latent_init
        # + per-pair residuals; closed-form fp16 proxy over all parameters.
        archive_bytes_proxy = closed_form_weight_byte_proxy(model)

        def _compute_loss(
            m, idx, gt_0, gt_1, abp, *, gt_pose_batch, gt_seg_batch, gt_seg_already_probs
        ):
            rgb_0, rgb_1, _z = m.reconstruct_pair(idx)
            residuals = m.residuals[idx]
            return loss_fn(
                reconstructed_rgb_0=rgb_0 * 255.0,
                reconstructed_rgb_1=rgb_1 * 255.0,
                gt_rgb_0=gt_0, gt_rgb_1=gt_1,
                archive_bytes_proxy=abp,
                residuals=residuals,
                apply_eval_roundtrip=True, noise_std=args.noise_std,
                gt_pose_batch=gt_pose_batch, gt_seg_batch=gt_seg_batch,
                gt_seg_already_probs=gt_seg_already_probs,
            )

        result = run_pact_nerv_score_aware_training(
            model=model, pair_tensor=pair_tensor, compute_loss=_compute_loss,
            archive_bytes_proxy=archive_bytes_proxy, device=device,
            output_dir=args.output_dir, substrate_tag=SUBSTRATE_TAG,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            ema_decay=args.ema_decay, val_pair_count=args.val_pair_count,
            val_every_epochs=args.val_every_epochs, gt_cache=gt_cache,
            stage_log=stage_log, config_asdict=asdict(cfg),
        )
        print(
            f"[full:{SUBSTRATE_TAG}] train done: best_val_lag="
            f"{result.best_val_lagrangian:.6f} elapsed={result.train_elapsed_sec:.1f}s"
        )

        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            # Reload EMA shadow into a fresh model to partition submodule
            # state_dicts (encoder/decoder/predictor) for the multi-component
            # Z5PCWM1 archive grammar (mirrors the smoke's pack_archive call).
            export_model = PredictiveCodingSubstrate(cfg)
            export_model.load_state_dict(result.best_ema_state_dict)
            export_model.eval()
            enc_sd = export_model.encoder.state_dict()
            dec_sd = export_model.decoder.state_dict()
            pred_sd = export_model.predictor.state_dict()
            latent_init = export_model.latent_init.detach().cpu()
            residuals = export_model.residuals.detach().cpu()
            ego_motion = export_model.ego_motion_buffer.detach().cpu()
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
            }
            bin_bytes = pack_archive(
                enc_sd, dec_sd, pred_sd, latent_init, residuals, ego_motion, meta,
                lambda_residual_entropy=args.lambda_residual_entropy,
                predictor_num_layers=args.predictor_num_layers,
                identity_predictor=args.identity_predictor,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_z5(bin_bytes)
            archive_bytes = len(bin_bytes)
            submission_dir = args.output_dir / "submission"
            write_contest_runtime(
                submission_dir,
                substrate_pkg_name="z5_predictive_coding_world_model",
                repo_root=REPO_ROOT,
            )
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir
            )
            print(f"[full:{SUBSTRATE_TAG}] wrote 0.bin ({archive_bytes} B) + archive.zip")

        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args, archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir, output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag=SUBSTRATE_TAG, device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(f"[full:{SUBSTRATE_TAG}] [contest-CUDA] score = {contest_cuda_score}")

        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                _detected = _canon_detect_hardware_substrate(
                    axis="cuda", substrate_tag=SUBSTRATE_TAG,
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("Z5_GPU", "MODAL_GPU"),
                )
                update = posterior_update_locked(
                    ContestResult(
                        axis="cuda", hardware_substrate=_detected,
                        architecture_class=SUBSTRATE_LANE_ID,
                        score_value=contest_cuda_score, evidence_tag="[contest-CUDA]",
                        archive_sha256=archive_sha, archive_bytes=archive_bytes,
                        notes=f"z5 first-anchor; epochs={args.epochs}",
                        observed_at_utc=_canon_utc_now_iso(),
                    )
                )
                print(f"[full:{SUBSTRATE_TAG}] posterior_update accepted={update.accepted}")
            except Exception as exc:
                print(f"[full:{SUBSTRATE_TAG}] posterior_update failed: {exc}", file=sys.stderr)

        provenance = {
            "schema": "z5_predictive_coding_full_provenance_v1",
            "generated_at": _canon_utc_now_iso(),
            "git_head": _canon_git_head_sha(REPO_ROOT),
            "trainer": "experiments/train_substrate_z5_predictive_coding_world_model.py",
            "lane_id": SUBSTRATE_LANE_ID,
            "substrate_tag": SUBSTRATE_TAG,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "pytorch_version": _canon_torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": result.n_pairs,
            "best_val_lagrangian": (
                result.best_val_lagrangian
                if result.best_val_lagrangian == result.best_val_lagrangian else None
            ),
            "best_epoch": result.best_epoch,
            "train_elapsed_sec": result.train_elapsed_sec,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": "[contest-CUDA]" if contest_cuda_score is not None else None,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full:{SUBSTRATE_TAG}] wrote {args.output_dir / 'provenance.json'}")
        return 0
    finally:
        unpatch_upstream_yuv6(yuv6_token)


def _sha256_z5(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
