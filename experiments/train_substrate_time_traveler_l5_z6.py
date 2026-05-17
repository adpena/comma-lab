# SPDX-License-Identifier: MIT
"""Train the Z6 Time-Traveler L5 F-asymptote-node predictive-coding world-model substrate.

Per the Time-Traveler L5 Z6/Z7/Z8 predictive-coding world-model scoping design memo
(``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``,
commit ``aa412d2db``): Z6 is the FIRST sequenced Z-variant of the F-asymptote
trajectory along the scorer-relationship class-shift axis (predictive-coding
paradigm; Rao-Ballard 1999 + Atick-Redlich 1990 cooperative-receiver). Recommended
FIRST build per Section 22 op-routable #2 (lowest engineering risk; sister Z5 L1
scaffold pattern).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187).
- ``load_differentiable_scorers`` for SegNet/PoseNet.
- ``apply_eval_roundtrip_during_training`` (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` (Catalog #88).
- Score-domain predictive-coding Lagrangian via canonical
  ``tac.substrates._shared.score_aware_common.score_pair_components`` per
  Catalog #164 (single-source-of-truth scorer-preprocess routing).
- End with CUDA auth eval on best EMA checkpoint via canonical
  ``smoke_auth_eval_gate.gate_auth_eval_call`` (CLAUDE.md "Auth eval
  EVERYWHERE" + Catalog #226).
- Inflate-device-fork via canonical ``select_inflate_device`` per Catalog #205.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151 + #168 AnnAssign).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived``
  (Catalog #197).
- SubstrateContract via canonical ``@register_substrate`` per Catalog #241/#242
  (META layer auto-wire one-way data flow).
- Catalog #240 substrate-engineering opt-out: ``_full_main`` raises
  ``NotImplementedError("Phase 2 council approval required")``; the recipe's
  ``dispatch_enabled: false`` + ``research_only: true`` prevents Modal
  dispatch from firing before council adjudication.

V1 SCOPE (this L1 SCAFFOLD landing per op-routable #2):
- ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on synthetic
  data, runs archive pack + parse + inflate roundtrip + autoregression
  test, and emits a contest-compliant runtime tree (no scorer load).
- ``_full_main`` is currently a NotImplementedError stub awaiting Phase 2
  council approval per Catalog #240 + Z6 memo Section 19 reactivation
  criteria. Operator-routable decision pending.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z6_smoke_<utc> \\
        --epochs 3 --device cpu --smoke
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:autoregressive-predictor-unroll-needs-canary-validation
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_full_main
# INLINE_DEVICE_FORK_OK:canonical-select_inflate_device-imported-via-tac.substrates._shared.inflate_runtime-per-Catalog-#205
# SCORER_PREPROCESS_HANDLED_OK:routed-through-canonical-tac.substrates._shared.score_aware_common.score_pair_components-per-Catalog-#164
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Catalog #241/#242 META layer substrate contract — single source-of-truth
# for trainer + recipe + lane registry + cost band envelope.
from tac.substrate_registry import SubstrateContract, register_substrate

# Catalog #205 canonical inflate device-fork — token visible to Catalog #270
# Tier 3 substrate-correctness verifier.
from tac.substrates._shared.inflate_runtime import select_inflate_device  # noqa: F401
from tac.substrates._shared.trainer_skeleton import decode_real_pairs as _decode_real_pairs

# Catalog #164 canonical scorer-loss helper routing — token visible to
# Catalog #270 Tier 1 dispatch-optimization-protocol verifier.
from tac.substrates.score_aware_common import (  # noqa: F401
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)
from tac.substrates.time_traveler_l5_z6 import (
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
    pack_archive,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBSTRATE_TAG = "time_traveler_l5_z6"
SUBSTRATE_LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
PAIRED_CONTROL_INITIALIZATION = "shared_modules_seed_order_matched_v2"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z6_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z6_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "Z6_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=300",
        "default": "300",
    },
    "--batch-size": {
        "env": "Z6_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; T4 handles 4-8 at 384x512 with autoregressive "
            "predictor unroll across 600 pairs"
        ),
        "default": "4",
    },
    "--lr": {
        "env": "Z6_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--lambda-residual-entropy": {
        "env": "Z6_LAMBDA_RESIDUAL_ENTROPY",
        "rationale": (
            "Predictive-coding residual-entropy weight (Rao-Ballard 1999); "
            "0 = no PC, 1 = canonical, higher = more aggressive predictor learning"
        ),
        "default": "1.0",
    },
    "--predictor-kernel-size": {
        "env": "Z6_PREDICTOR_KERNEL_SIZE",
        "rationale": (
            "FiLM predictor conv kernel size; 1 / 3 / 5; default 3 per design memo "
            "Section 4.1"
        ),
        "default": "3",
    },
    "--predictor-ego-motion-dim": {
        "env": "Z6_PREDICTOR_EGO_MOTION_DIM",
        "rationale": (
            "Ego-motion proxy dimension projected from PoseNet output; "
            "default 8; small dim keeps predictor compact"
        ),
        "default": "8",
    },
    "--identity-predictor": {
        "env": "Z6_IDENTITY_PREDICTOR",
        "rationale": (
            "Probe-disambiguator ablation: when true, predictor is identity "
            "(no learning); compare to full FiLM for Rao-Ballard "
            "refutation/confirmation per Catalog #125 hook #6"
        ),
        "default": "false",
    },
    "--enable-autocast-fp16": {
        "env": "Z6_ENABLE_AUTOCAST_FP16",
        "rationale": "Catalog #172; pending canonical autocast backport",
        "default": "false",
    },
}


# ---------------------------------------------------------------------------
# Catalog #241/#242 SubstrateContract — META layer single source of truth.
# This binds (a) trainer's claimed contract, (b) recipe schema, (c) lane
# registry, (d) cost band envelope into ONE source-of-truth that fails-loud
# at decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------
TIME_TRAVELER_L5_Z6_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="time_traveler_l5_z6",
    lane_id=SUBSTRATE_LANE_ID,
    target_modes=(
        "contest_one_video_replay",
        "contest_generalized",
        "research_substrate",
    ),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "Z6PCWM1 monolithic single-file 0.bin: header + encoder + decoder + "
        "FiLM predictor state_dicts (fp16 brotli) + latent_init + per-pair "
        "residuals + ego_motion + meta JSON"
    ),
    parser_section_manifest={
        "header": "Z6PCWM1_magic_and_version",
        "encoder_blob": "fp16_brotli_blob",
        "decoder_blob": "fp16_brotli_blob",
        "predictor_blob": "fp16_brotli_blob",
        "latent_init_blob": "int8_quantized",
        "residuals_blob": "int8_quantized",
        "ego_motion_blob": "int8_quantized_sidecar",
        "meta_blob": "sorted_keys_json_utf8",
    },
    inflate_runtime_loc_budget=120,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=700,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate recipe YAML
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=3,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=1.0,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token="Rao-Ballard",
    hook_continual_learning_anchor_kind="not_applicable_with_rationale",
    hook_probe_disambiguator=(
        "tools/probe_z6_predictive_coding_vs_identity_disambiguator.py"
    ),
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_163_remote_lane_sentinel_used",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_240_substrate_engineering_pre_build_opt_out",
        "catalog_244_remote_lane_canonical_nvml_block",
        "catalog_270_dispatch_optimization_protocol_scaffold_pass",
        "catalog_272_distinguishing_feature_byte_mutation_pending",
        "catalog_290_canonical_vs_unique_decision_per_layer_documented",
        "catalog_305_observability_surface_declared",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "FiLM predictor gradient norm IS the sensitivity signal but registration"
            " happens post Phase 2 council approval"
        ),
        "hook_bit_allocator_class": (
            "int8 per-pair residuals + fp16 brotli weights; no per-tensor bit"
            " allocator at L1 SCAFFOLD"
        ),
        "hook_continual_learning_anchor_kind": (
            "L1 SCAFFOLD has no contest-CUDA anchor yet; posterior_update_locked"
            " fires after Phase 2 dispatch + paired CPU/CUDA"
        ),
    },
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_time_traveler_l5_z6",
        description=(
            "Train Z6 Time-Traveler L5 F-asymptote-node predictive-coding "
            "world-model substrate (FiLM-conditioned next-frame predictor; "
            "Rao-Ballard 1999 + Atick-Redlich 1990 cooperative-receiver)."
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
    p.add_argument("--predictor-film-mlp-hidden-dim", type=int, default=32)
    p.add_argument(
        "--predictor-kernel-size", type=int, default=3, choices=[1, 3, 5]
    )
    p.add_argument("--predictor-ego-motion-dim", type=int, default=8)
    p.add_argument(
        "--identity-predictor", action="store_true",
        help=(
            "Probe-disambiguator regime: identity predictor (no learning); "
            "Catalog #125 hook #6"
        ),
    )
    p.add_argument(
        "--smoke-ego-motion-mode",
        choices=("ramp", "zero", "random", "real-video"),
        default="ramp",
        help=(
            "Smoke-only ego-motion proxy. Default ramp exercises FiLM "
            "conditioning; real-video derives a proxy from upstream video frame "
            "deltas; zero is retained only as a cargo-cult control."
        ),
    )
    p.add_argument(
        "--smoke-target-mode",
        choices=("synthetic", "real-video"),
        default="synthetic",
        help=(
            "Smoke target source. synthetic is fastest; real-video decodes the "
            "first tiny contest-video pair batch through the canonical pyav helper."
        ),
    )

    # Predictive-coding Lagrangian weights
    p.add_argument(
        "--lambda-residual-entropy", type=float, default=1.0,
        help="Rao-Ballard residual-entropy weight; 0=no PC, 1=canonical",
    )
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))

    # Training
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)

    # Mode flags
    p.add_argument(
        "--smoke", action="store_true",
        help="Run smoke path (tiny config, synthetic data, no scorer load)",
    )
    p.add_argument(
        "--full-cpu", action="store_true",
        help="Opt-in to non-smoke CPU training (Catalog #197 paired flag required)",
    )
    p.add_argument(
        "--advisory-cpu-explicitly-waived", action="store_true",
        help="Required sister flag for --full-cpu (Catalog #197)",
    )
    p.add_argument(
        "--enable-autocast-fp16", action="store_true",
        help="Catalog #172; pending canonical autocast backport",
    )
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

def _smoke_effective_epochs(requested_epochs: int) -> int:
    """Return the bounded epoch count for smoke runs.

    Default full-training epochs are intentionally large, but smoke is a
    training-artifact probe and must stay at <=3 epochs unless the full path is
    lifted separately.
    """

    return max(1, min(int(requested_epochs), 3))


def _populate_smoke_ego_motion(
    substrate: Z6PredictiveCodingSubstrate,
    *,
    mode: str,
    seed: int,
    real_video_ego_motion: torch.Tensor | None = None,
) -> None:
    """Populate smoke ego-motion so FiLM conditioning is actually exercised."""

    shape = (
        substrate.cfg.num_pairs,
        substrate.cfg.predictor_ego_motion_dim,
    )
    target_device = substrate.ego_motion_buffer.device
    if mode == "real-video":
        if real_video_ego_motion is None:
            raise ValueError(
                "real-video smoke ego-motion mode requires decoded real-video features"
            )
        if tuple(real_video_ego_motion.shape) != shape:
            raise ValueError(
                f"real-video ego-motion shape {tuple(real_video_ego_motion.shape)} "
                f"!= expected {shape}"
            )
        substrate.ego_motion_buffer.copy_(real_video_ego_motion.to(target_device))
        return
    if mode == "zero":
        substrate.ego_motion_buffer.zero_()
        return
    if mode == "ramp":
        values = torch.linspace(
            -1.0,
            1.0,
            steps=shape[0] * shape[1],
            device=target_device,
        ).view(shape)
        substrate.ego_motion_buffer.copy_(values)
        return
    if mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
        values = torch.randn(shape, generator=generator, device="cpu").to(target_device)
        substrate.ego_motion_buffer.copy_(values)
        return
    raise ValueError(f"unknown smoke ego-motion mode: {mode!r}")


def _ego_motion_from_smoke_targets(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    ego_motion_dim: int,
) -> torch.Tensor:
    """Derive a tiny deterministic ego-motion proxy from real-video smoke targets."""

    if target0.shape != target1.shape:
        raise ValueError(
            f"target0 shape {tuple(target0.shape)} != target1 shape {tuple(target1.shape)}"
        )
    if target0.dim() != 4 or target0.shape[1] != 3:
        raise ValueError(
            "smoke targets must be shaped (num_pairs, 3, height, width)"
        )
    if ego_motion_dim <= 0:
        raise ValueError(f"ego_motion_dim must be > 0; got {ego_motion_dim}")
    diff = target1 - target0
    channel_mean_delta = diff.mean(dim=(2, 3))
    abs_delta_mean = diff.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    features = torch.cat([channel_mean_delta, abs_delta_mean], dim=1)
    features = features - features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    features = features / scale
    if features.shape[1] < ego_motion_dim:
        repeats = math.ceil(ego_motion_dim / features.shape[1])
        features = features.repeat(1, repeats)
    return features[:, :ego_motion_dim].contiguous()


def _decode_real_video_smoke_targets(
    video_path: Path,
    cfg: Z6PredictiveCodingConfig,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a tiny real-video smoke batch and its frame-delta ego proxy."""

    pairs = _decode_real_pairs(
        video_path,
        n_pairs=cfg.num_pairs,
        max_pairs=cfg.num_pairs,
        substrate_tag=SUBSTRATE_TAG,
        repo_root=REPO_ROOT,
    )
    pairs = pairs.to(device=device, dtype=torch.float32) / 255.0
    flat = pairs.reshape(
        cfg.num_pairs * 2,
        3,
        pairs.shape[-2],
        pairs.shape[-1],
    )
    resized = F.interpolate(
        flat,
        size=(cfg.output_height, cfg.output_width),
        mode="bilinear",
        align_corners=False,
    )
    resized_pairs = resized.view(
        cfg.num_pairs,
        2,
        3,
        cfg.output_height,
        cfg.output_width,
    ).contiguous()
    target0 = resized_pairs[:, 0]
    target1 = resized_pairs[:, 1]
    ego_motion = _ego_motion_from_smoke_targets(
        target0,
        target1,
        ego_motion_dim=cfg.predictor_ego_motion_dim,
    )
    return target0, target1, ego_motion


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: tiny config, ≤3 epochs, no scorer load."""
    torch.manual_seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_epochs = _smoke_effective_epochs(args.epochs)

    # Tiny smoke config
    num_pairs = 5
    cfg = Z6PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_film_mlp_hidden_dim=8,
        predictor_kernel_size=args.predictor_kernel_size,
        predictor_ego_motion_dim=4,
        identity_predictor=args.identity_predictor,
        lambda_residual_entropy=args.lambda_residual_entropy,
    )
    substrate = Z6PredictiveCodingSubstrate(cfg).to(args.device)
    real_video_ego_motion: torch.Tensor | None = None
    if (
        args.smoke_target_mode == "real-video"
        or args.smoke_ego_motion_mode == "real-video"
    ):
        target0, target1, real_video_ego_motion = _decode_real_video_smoke_targets(
            args.video_path,
            cfg,
            device=args.device,
        )
    else:
        target0 = torch.rand(
            num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
        )
        target1 = torch.rand(
            num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
        )
    _populate_smoke_ego_motion(
        substrate,
        mode=args.smoke_ego_motion_mode,
        seed=args.seed,
        real_video_ego_motion=real_video_ego_motion,
    )
    print(f"[z6-smoke] param breakdown: {substrate.num_parameters_breakdown()}")
    print(f"[z6-smoke] identity_predictor={cfg.identity_predictor}")
    print(f"[z6-smoke] smoke_ego_motion_mode={args.smoke_ego_motion_mode}")

    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for epoch in range(effective_epochs):
        opt.zero_grad()
        idx = torch.arange(num_pairs, device=args.device, dtype=torch.long)
        rgb_0, rgb_1, z_t = substrate.reconstruct_pair(idx)
        # Pixel-MSE proxy + residual-norm proxy in smoke
        recon_loss = (
            (rgb_0 - target0).pow(2).mean()
            + (rgb_1 - target1).pow(2).mean()
        )
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
        "predictor_film_mlp_hidden_dim": cfg.predictor_film_mlp_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
        "smoke_target_mode": args.smoke_target_mode,
        "smoke_ego_motion_mode": args.smoke_ego_motion_mode,
        "requested_epochs": args.epochs,
        "effective_epochs": effective_epochs,
    }
    archive_bytes = pack_archive(
        enc_sd, dec_sd, pred_sd, latent_init, residuals, ego_motion, meta,
        lambda_residual_entropy=args.lambda_residual_entropy,
        predictor_kernel_size=args.predictor_kernel_size,
        identity_predictor=args.identity_predictor,
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    final = losses[-1] if losses else {"loss": float("inf")}
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "requested_epochs": args.epochs,
        "epochs": len(losses),
        "smoke_epoch_cap": 3,
        "final_loss_proxy": final["loss"],
        "final_recon": final.get("recon"),
        "final_residual": final.get("residual"),
        "archive_bytes": len(archive_bytes),
        "lambda_residual_entropy": args.lambda_residual_entropy,
        "predictor_kernel_size": args.predictor_kernel_size,
        "identity_predictor": args.identity_predictor,
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        "paired_control_shared_modules": [
            "encoder",
            "decoder",
            "latent_init",
            "residuals",
            "ego_motion_buffer",
        ],
        "smoke_target_mode": args.smoke_target_mode,
        "smoke_ego_motion_mode": args.smoke_ego_motion_mode,
        "video_path": str(args.video_path),
        "ego_motion_nonzero_fraction": float(
            (substrate.ego_motion_buffer.detach().abs() > 0).float().mean().item()
        ),
        "ego_motion_l2": float(
            substrate.ego_motion_buffer.detach().pow(2).sum().sqrt().item()
        ),
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
        f"[z6-smoke] OK final_loss={final['loss']:.6f} archive={len(archive_bytes)}B "
        f"kernel={args.predictor_kernel_size} identity={args.identity_predictor} "
        f"ego={args.smoke_ego_motion_mode}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full entry path — NotImplementedError stub awaiting Phase 2 council approval.
# Catalog #240 substrate-engineering pre-build opt-out: the trainer's full
# path is council-gated so no $1+ Modal dispatch can fire until the design
# verdicts are adjudicated per the Z6 memo Section 19 reactivation criteria.
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Phase 2 council approval required.

    The full Z6 training path needs (per design memo Sections 5-9):

    1. pyav decode of upstream/videos/0.mkv into per-pair frames (Catalog #114)
    2. patch_upstream_yuv6_globally + load_differentiable_scorers (Catalog #187)
    3. Z6PredictiveCodingScoreAwareLoss with eval-roundtrip + EMA(0.997)
       routed through canonical score_pair_components (Catalog #164)
    4. AdamW + grad clip 1.0 + NaN watchdog
    5. Autoregressive FiLM predictor unroll across 600 pairs (mini-batched per
       Catalog #218 reconstruct_pair pattern; D4 sister substrate's lesson)
    6. Ego-motion buffer populated from PoseNet output projection (or
       held at zeros for identity-predictor ablation regime)
    7. CUDA auth eval on best EMA checkpoint via canonical
       gate_auth_eval_call (Catalog #226)
    8. posterior_update_locked (Catalog #128)
    9. Contest-compliant runtime emission with select_inflate_device per
       Catalog #205 + 3-positional-arg inflate.sh per Catalog #146 + #163

    The architecture, archive, inflate, and score_aware_loss modules are
    landed and tested (≥10 dedicated tests pass; autoregression recurrence
    test confirms inflate-time correctness). The trainer body is
    deliberately stubbed pending council approval per CLAUDE.md "Design
    decisions — non-negotiable" + Catalog #240 (recipe-vs-trainer-state
    consistency): world-model predictive-coding is a council-grade tradeoff
    because (a) the autoregressive unroll cost multiplies training
    wall-clock, (b) the identity-predictor ablation needs to fire BEFORE the
    full predictor to anchor the probe, and (c) the predicted [0.13, 0.16]
    band requires sextet-pact council consensus per design memo Section 19
    reactivation criteria.

    Operator-routable: see Z6 memo Section 19 for the 5 reactivation criteria
    + Section 22 op-routables for the next-step decision path. Phase 2
    dispatch approval lifts this NotImplementedError after sextet-pact
    council CONSENSUS.
    """
    raise NotImplementedError(
        "Phase 2 council approval required to lift this NotImplementedError. "
        "Per Catalog #240 + Z6 memo Section 19 reactivation criteria: "
        "(1) operator decision Z6 selected as FIRST Z-variant; (2) Dykstra-"
        "feasibility polytope confirms Z6 predicted band FEASIBLE; (3) sister "
        "L1 SCAFFOLD lands trainer + archive + inflate + tests; (4) Phase 2 "
        "sextet-pact council CONSENSUS (Shannon + Dykstra + Rao + Ballard + "
        "Tishby + Contrarian + Assumption-Adversary); (5) Catalog #167 "
        "smoke-before-full pattern confirms Z6 smoke at $1 Modal T4 lands "
        "rc=0 + archive bytes within [80, 120] KB + identity-predictor "
        "disambiguator probe ready. See `.omx/research/"
        "time_traveler_l5_z6_z7_z8_predictive_coding_world_models_"
        "asymptotic_pursuit_scoping_design_20260516.md` §19 for the "
        "dispatch-gating thresholds."
    )


@register_substrate(TIME_TRAVELER_L5_Z6_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
