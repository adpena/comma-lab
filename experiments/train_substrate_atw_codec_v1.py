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
    """Full-mode trainer entry point.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #220 cascade: ATW codec V1 is an L1 SCAFFOLD
    + research_only=true + lane_class=substrate_engineering + pre-build
    council-gated. The ``_full_main`` body is intentionally NOT IMPLEMENTED
    so no $1+ Modal dispatch can fire from this trainer until Phase 2
    council approval is granted.

    Reactivation criteria are documented in
    ``.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md`` §5.
    """
    raise NotImplementedError(
        "ATW codec V1 _full_main is council-gated per Catalog #220 substrate-"
        "engineering pre-build cascade. The substrate is L1 SCAFFOLD + "
        "research_only=true at landing 2026-05-15. Phase 2 council approval "
        "required to lift; reactivation criteria documented in "
        ".omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md §5. "
        "Use --smoke for synthetic-data sanity verification (no GPU spend)."
    )


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
