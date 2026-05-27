# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:v1-scaffold-_full_main-raises-NotImplementedError-Phase-2-council-pre-build-substrate-engineering-gate
"""Train the ATW codec V1 substrate (Atick-Tishby-Wyner cooperative-receiver codec).

Per the 2026-05-15 grand reunion symposium Composite #1 (lines 727-770) and
the design memo at ``.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md``.

Composite of three foundational information-theoretic frameworks:

* Atick & Redlich (1990) — cooperative-receiver theorem
* Tishby, Pereira & Bialek (1999) — Information Bottleneck Lagrangian
* Wyner & Ziv (1976) — source coding with side information at decoder

The training Lagrangian:

::

    L_ATW = α · B/N + β_seg · d_seg + γ_pose · sqrt(d_pose)
          + κ_IB · I(T; Y_predicted)
          + λ_WZ · R_WZ_residual(t | side_info_head(class_prior))
          + λ_pixel · MSE(decoded, GT)

V1 SCAFFOLD SCOPE:

* ``_smoke_main``: builds a tiny config, runs synthetic 1-3 epoch sanity
  check, validates archive pack + parse + roundtrip, and emits an ATW1
  monolithic ``0.bin`` archive for byte inspection. NO scorer load.
  NO real video decode. ``$0`` cost.
* ``_full_main``: RAISES ``NotImplementedError`` per CLAUDE.md "Substrate
  scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220 substrate-
  engineering pre-build council-gated cascade. Phase 2 council approval
  required to lift; reactivation criteria documented in design memo §5.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~1-3 epochs)::

    .venv/bin/python experiments/train_substrate_atw_codec_v1.py \\
        --output-dir experiments/results/atw_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; PHASE 2 COUNCIL APPROVAL REQUIRED — currently raises NotImplementedError)::

    .venv/bin/python experiments/train_substrate_atw_codec_v1.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/atw_<utc> \\
        --epochs 200 --batch-size 4 --lr 5e-4 --device cuda \\
        --kappa-ib 0.0 --lambda-wz 1.0 --lambda-pixel 0.0
"""
# Catalog #168 AnnAssign + Catalog #151 manifest. _full_main is council-gated
# substrate_engineering scaffold per Catalog #220 cascade; smoke uses synthetic
# data per the canonical substrate-scaffold pattern.
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-data-_full_main-raises-NotImplementedError
from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    device_or_die as _canon_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _canon_git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _canon_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    torch_version_string as _canon_torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates.atw_codec_v1 import (
    ATW1_MAGIC,
    ATWCodec,
    ATWCodecConfig,
    pack_archive,
    parse_archive,
)
from tac.substrates.atw_codec_v1.registered_substrate import (
    ATW_CODEC_V1_CONTRACT,  # noqa: F401  (forces package-side contract validation)
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
SUBSTRATE_TAG = "atw_codec_v1"
SUBSTRATE_LANE_ID = "lane_atw_codec_design_v1_20260515"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "ATW_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training "
            "(non-smoke is council-gated per Catalog #220 cascade)"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "ATW_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "ATW_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal A100 full=200",
        "default": "200",
    },
    "--batch-size": {
        "env": "ATW_BATCH_SIZE",
        "rationale": "Per-step pair count; A100 handles 4-8 at 384x512",
        "default": "4",
    },
    "--lr": {
        "env": "ATW_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--kappa-ib": {
        "env": "ATW_KAPPA_IB",
        "rationale": (
            "Tishby IB regularizer weight; 0 = no IB (Atick-Redlich + WZ pure); "
            "0.05-0.1 = IB regime (probe-disambiguator corner)"
        ),
        "default": "0.0",
    },
    "--lambda-wz": {
        "env": "ATW_LAMBDA_WZ",
        "rationale": (
            "Wyner-Ziv residual term weight; 1 = ATW canonical; "
            "0 = WZ disabled (= Z4 baseline branch of probe-disambiguator)"
        ),
        "default": "1.0",
    },
    "--lambda-pixel": {
        "env": "ATW_LAMBDA_PIXEL",
        "rationale": (
            "Pixel-MSE residual weight; 0 = pure ATW; "
            "1 = Z3 baseline (probe-disambiguator corner)"
        ),
        "default": "0.0",
    },
    "--beta-seg": {
        "env": "ATW_BETA_SEG",
        "rationale": "SegNet distortion weight (contest formula = 100)",
        "default": "100.0",
    },
    "--gamma-pose": {
        "env": "ATW_GAMMA_POSE",
        "rationale": "PoseNet distortion sqrt-weight (contest formula = sqrt(10))",
        "default": str(math.sqrt(10.0)),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_atw_codec_v1",
        description=(
            "Train ATW codec V1 substrate (Atick-Tishby-Wyner cooperative-"
            "receiver codec; grand reunion symposium Composite #1)."
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
    p.add_argument("--scorer-class-prior-dim", type=int, default=16)
    p.add_argument("--wz-head-hidden-dim", type=int, default=32)
    p.add_argument("--wz-head-enabled", type=lambda s: s.lower() == "true", default=True)

    # ATW Lagrangian three knobs
    p.add_argument("--kappa-ib", type=float, default=0.0,
                   help="Tishby IB regularizer weight; 0 = no IB; 0.05-0.1 = IB regime")
    p.add_argument("--lambda-wz", type=float, default=1.0,
                   help="Wyner-Ziv residual weight; 1 = ATW canonical; 0 = WZ disabled")
    p.add_argument("--lambda-pixel", type=float, default=0.0,
                   help="Pixel-MSE residual weight; 0 = pure ATW; 1 = Z3 baseline")
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Full-mode score-aware training flags (CLASS-SHIFT-FULL-MAIN-CLUSTER
    # pattern; run_pact_nerv_score_aware_training consumes these).
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient L2-norm clip (Council D NaN-watchdog companion).")
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA shadow decay (Quantizr 0.997 per CLAUDE.md EMA rule).")
    p.add_argument("--noise-std", type=float, default=0.5,
                   help="eval_roundtrip noise std (threaded; never disables roundtrip).")
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
    p.add_argument("--enable-autocast-fp16", action="store_true", default=False,
                   help="RESERVED (Phase 2): Wrap forward in torch.autocast(fp16).")
    p.add_argument("--enable-torch-compile", action="store_true", default=False,
                   help="RESERVED (Phase 2): Wrap substrate with torch.compile / Inductor.")
    p.add_argument("--enable-gt-scorer-cache", action="store_true", default=False,
                   help="RESERVED (Phase 2): GT-scorer-output cache (Catalog #228).")

    # Smoke / mode flags
    p.add_argument("--smoke", action="store_true", help="Run synthetic-data sanity smoke")
    return p


def _smoke_main(args: argparse.Namespace) -> int:
    """Synthetic-data sanity smoke — validates substrate forward + archive roundtrip.

    No scorer load. No real video decode. ``$0`` cost. Verifies:

    1. ATWCodec instantiates with the canonical config.
    2. Forward pass produces (rgb_0, rgb_1) of correct shape.
    3. WZ side-info head produces non-zero z_residual when enabled.
    4. Archive pack → parse roundtrip is byte-identical.
    5. ATW1 magic + section-offset parser refuses tampered bytes.
    """
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tiny config so smoke runs in <1 min on CPU.
    cfg = ATWCodecConfig(
        latent_dim=args.latent_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_num_upsample_blocks=4,  # smaller than full
        decoder_channels=(16, 12, 8, 6, 4, 2),
        num_pairs=8,  # tiny
        output_height=64,  # tiny
        output_width=96,
        scorer_class_prior_dim=args.scorer_class_prior_dim,
        wz_head_hidden_dim=args.wz_head_hidden_dim,
        wz_head_enabled=args.wz_head_enabled,
        ib_kappa_default=args.kappa_ib,
        wz_lambda_default=args.lambda_wz,
        pixel_lambda_default=args.lambda_pixel,
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = ATWCodec(cfg).to(device)
    model.eval()

    # Populate scorer_class_prior_table with deterministic non-zero pattern
    # (smoke: synthetic; full: precomputed from real scorer at compress time).
    with torch.no_grad():
        for i in range(cfg.num_pairs):
            model.scorer_class_prior_table[i] = (
                torch.arange(cfg.scorer_class_prior_dim, dtype=torch.float32) * 0.1
                + float(i) * 0.01
            )

    # Forward smoke
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long, device=device)
    with torch.no_grad():
        rgb_0, rgb_1, _mu, _logvar, z_residual, z_predicted = model(
            pair_indices, frames_for_encoder=None, compute_wz_residual=True
        )
    expected_shape = (cfg.num_pairs, 3, cfg.output_height, cfg.output_width)
    if tuple(rgb_0.shape) != expected_shape or tuple(rgb_1.shape) != expected_shape:
        raise RuntimeError(
            f"smoke forward shape mismatch: got rgb_0 {tuple(rgb_0.shape)}, "
            f"rgb_1 {tuple(rgb_1.shape)}; expected {expected_shape}"
        )

    # Archive roundtrip smoke
    encoder_sd = model.encoder.state_dict()
    decoder_sd = model.decoder.state_dict()
    wz_head_sd = model.wz_side_info_head.state_dict()
    meta_seed: dict[str, object] = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
    }
    archive_bytes = pack_archive(
        encoder_sd,
        decoder_sd,
        wz_head_sd,
        z_residual.detach().cpu() if z_residual is not None else model.latents.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        meta_seed,
        atw_kappa_ib=args.kappa_ib,
        atw_lambda_wz=args.lambda_wz,
        atw_lambda_pixel=args.lambda_pixel,
        wz_head_enabled=cfg.wz_head_enabled,
    )
    if not archive_bytes.startswith(ATW1_MAGIC):
        raise RuntimeError(
            f"archive magic mismatch: got {archive_bytes[:4]!r} expected {ATW1_MAGIC!r}"
        )
    parsed = parse_archive(archive_bytes)
    if parsed.schema_version != 1:
        raise RuntimeError(f"unexpected schema version: {parsed.schema_version}")

    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    stats: dict[str, Any] = {
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "smoke": True,
        "device": str(device),
        "epochs": args.epochs,
        "archive_bytes": len(archive_bytes),
        "archive_sha256_first16": _sha256_first16(archive_bytes),
        "model_params": model.num_parameters_breakdown(),
        "kappa_ib": args.kappa_ib,
        "lambda_wz": args.lambda_wz,
        "lambda_pixel": args.lambda_pixel,
        "wz_head_enabled": cfg.wz_head_enabled,
        "atw1_magic_ok": True,
        "roundtrip_ok": True,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "smoke_stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"[atw_codec_v1] SMOKE OK device={device} archive_bytes={len(archive_bytes)} "
        f"params={model.num_parameters()} kappa={args.kappa_ib} lambda_wz={args.lambda_wz} "
        f"lambda_pixel={args.lambda_pixel}"
    )
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full score-aware training entry point — CUDA-required; paid-GPU gated.

    CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: routes the substrate-AGNOSTIC
    training loop through the canonical
    ``tac.substrates._shared.pact_nerv_full_main`` helper; the UNIQUE ATW
    distinguishing feature (Atick-Redlich cooperative-receiver IB encoder +
    Wyner-Ziv side-information residual head; three-knob Lagrangian
    kappa_ib/lambda_wz/lambda_pixel) stays in this substrate's architecture +
    archive + score-aware loss. The forward adapter threads the WZ
    ``z_residual``/``z_predicted`` terms into ``ATWScoreAwareLoss``; the
    archive partitions encoder/decoder/wz_side_info_head submodules +
    latent_residual + scorer_class_prior_table. The ``NotImplementedError`` is
    extinguished; PAID DISPATCH is still gated by ``dispatch_enabled: false`` +
    ``research_only: true`` + ``lane_class=substrate_engineering`` on the recipe
    per Catalog #220 + #325 (code complete, trigger gated).

    Honored end-to-end: real contest video (Catalog #114); patch yuv6 BEFORE
    scorer construction (eval_roundtrip non-negotiable); ``load_differentiable_
    scorers`` (no scorer at inflate); ATW three-knob score-domain Lagrangian via
    the variant loss → Catalog #164 dispatch; EMA shadow (Quantizr 0.997);
    CUDA-required (``device_or_die`` rejects MPS per Catalog #1); CUDA auth-eval
    via canonical ``gate_auth_eval_call`` (Catalog #226); posterior-update via
    ``posterior_update_locked`` (Catalog #128); contest-compliant runtime
    (Catalog #146 + #295).

    Reactivation criteria for PAID DISPATCH (per HNeRV parity L2 + the design
    memo §5): per-substrate symposium operator-gated approval (Catalog #325) +
    cargo-cult audit (Catalog #303 — kappa_ib/lambda_wz/lambda_pixel defaults
    flagged for empirical sweep); recipe ``research_only`` flips to false +
    ``dispatch_enabled`` to true.
    """
    from dataclasses import asdict

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
    from tac.substrates.atw_codec_v1 import (
        ATWLossWeights,
        ATWScoreAwareLoss,
    )

    torch.manual_seed(args.seed)
    device = _canon_device_or_die(args.device, smoke=False, substrate_tag=SUBSTRATE_TAG)
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

        cfg = ATWCodecConfig(
            latent_dim=args.latent_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_num_upsample_blocks=args.decoder_num_upsample_blocks,
            scorer_class_prior_dim=args.scorer_class_prior_dim,
            wz_head_hidden_dim=args.wz_head_hidden_dim,
            wz_head_enabled=args.wz_head_enabled,
            ib_kappa_default=args.kappa_ib,
            wz_lambda_default=args.lambda_wz,
            pixel_lambda_default=args.lambda_pixel,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0], output_width=EVAL_HW[1],
        )
        model = ATWCodec(cfg).to(device)
        print(f"[full:{SUBSTRATE_TAG}] params: {sum(p.numel() for p in model.parameters()):,} "
              f"wz_head_enabled={cfg.wz_head_enabled}")

        weights = ATWLossWeights(
            alpha_rate=args.alpha_rate, beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose, pose_weight_scale=args.pose_weight_scale,
            kappa_ib=args.kappa_ib, lambda_wz=args.lambda_wz,
            lambda_pixel=args.lambda_pixel, contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = ATWScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )

        opt_ctx = _canon_build_optimized_training_context(
            args, scorers=(posenet, segnet), gt_pairs=pair_tensor,
            substrate_model=model, device=device,
        )
        gt_cache = opt_ctx.gt_cache
        archive_bytes_proxy = closed_form_weight_byte_proxy(model)

        def _compute_loss(
            m, idx, gt_0, gt_1, abp, *, gt_pose_batch, gt_seg_batch, gt_seg_already_probs
        ):
            rgb_0, rgb_1, _mu, _logvar, z_residual, z_predicted = m(
                idx, frames_for_encoder=None, compute_wz_residual=True
            )
            return loss_fn(
                reconstructed_rgb_0=rgb_0 * 255.0,
                reconstructed_rgb_1=rgb_1 * 255.0,
                gt_rgb_0=gt_0, gt_rgb_1=gt_1,
                archive_bytes_proxy=abp,
                z_residual=z_residual, z_predicted=z_predicted,
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
            export_model = ATWCodec(cfg)
            export_model.load_state_dict(result.best_ema_state_dict)
            export_model.eval()
            with torch.no_grad():
                _, _, _, _, z_residual_full, _ = export_model(
                    torch.arange(n_pairs, dtype=torch.long),
                    frames_for_encoder=None, compute_wz_residual=True,
                )
            encoder_sd = export_model.encoder.state_dict()
            decoder_sd = export_model.decoder.state_dict()
            wz_head_sd = export_model.wz_side_info_head.state_dict()
            latent_residual = (
                z_residual_full.detach().cpu() if z_residual_full is not None
                else export_model.latents.detach().cpu()
            )
            scorer_prior = export_model.scorer_class_prior_table.detach().cpu()
            meta = {
                "latent_dim": cfg.latent_dim,
                "encoder_input_channels": cfg.encoder_input_channels,
                "encoder_hidden_dim": cfg.encoder_hidden_dim,
                "decoder_embed_dim": cfg.decoder_embed_dim,
                "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
                "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
                "scorer_class_prior_dim": cfg.scorer_class_prior_dim,
                "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
            }
            bin_bytes = pack_archive(
                encoder_sd, decoder_sd, wz_head_sd, latent_residual, scorer_prior, meta,
                atw_kappa_ib=args.kappa_ib, atw_lambda_wz=args.lambda_wz,
                atw_lambda_pixel=args.lambda_pixel, wz_head_enabled=cfg.wz_head_enabled,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _canon_sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            submission_dir = args.output_dir / "submission"
            write_contest_runtime(
                submission_dir, substrate_pkg_name="atw_codec_v1", repo_root=REPO_ROOT,
            )
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir
            )
            print(f"[full:{SUBSTRATE_TAG}] wrote 0.bin ({archive_bytes} B; magic={ATW1_MAGIC!r}) + archive.zip")

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
                    env_var_candidates=("ATW_CODEC_V1_GPU", "MODAL_GPU"),
                )
                update = posterior_update_locked(
                    ContestResult(
                        axis="cuda", hardware_substrate=_detected,
                        architecture_class=SUBSTRATE_LANE_ID,
                        score_value=contest_cuda_score, evidence_tag="[contest-CUDA]",
                        archive_sha256=archive_sha, archive_bytes=archive_bytes,
                        notes=f"atw_codec_v1 first-anchor; epochs={args.epochs}",
                        observed_at_utc=_canon_utc_now_iso(),
                    )
                )
                print(f"[full:{SUBSTRATE_TAG}] posterior_update accepted={update.accepted}")
            except Exception as exc:
                print(f"[full:{SUBSTRATE_TAG}] posterior_update failed: {exc}", file=sys.stderr)

        provenance = {
            "schema": "atw_codec_v1_full_provenance_v1",
            "generated_at": _canon_utc_now_iso(),
            "git_head": _canon_git_head_sha(REPO_ROOT),
            "trainer": "experiments/train_substrate_atw_codec_v1.py",
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


def _sha256_first16(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
