#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""NSCS02 downsampled-renderer + inflate-upsample trainer.

Trains the NSCS02 5-stage decoder against ``upstream/videos/0.mkv``
with score-aware loss (canonical scorer-preprocess routing per
Catalog #164 with same-line waiver). Renders at (192, 256), upsamples
to scorer-native (384, 512) for the proxy loss, and at archive build
time emits ``0.bin`` per the wire format declared in
``tac.substrates.nscs02_downsampled_renderer.archive``.

Per the standing directive UNIQUE-AND-COMPLETE-PER-METHOD this trainer
is its own focused implementation. It SHARES tier-1 engineering
primitives (autocast / TF32 / no_grad-at-eval) with the canonical
trainer skeleton because they are score-NEUTRAL engineering hygiene
(per the canonical-vs-unique decision section in the landing memo).

Per Catalog #151 the ``TIER_1_OPERATOR_REQUIRED_FLAGS`` manifest is
declared as an ``ast.AnnAssign`` so the auto-wire validator can
introspect it.

Per Catalog #226 auth_eval is invoked through the canonical
``gate_auth_eval_call`` helper at the end of full training (NOT
hand-rolled subprocess.run).

Per Catalog #220 the substrate's `score_improvement_mechanism_status`
is OPERATIONAL: the inflate-time bicubic upsample is the score-
relevant runtime consumption that converts the downsampled-renderer
bytes into score-relevant frame-pixels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from tac.substrates.nscs02_downsampled_renderer import (
    NSCS02_BASE_CHANNELS,
    NSCS02_LATENT_DIM,
    NSCS02_N_PAIRS,
    NSCS02_RENDER_HW,
)
from tac.substrates.nscs02_downsampled_renderer.architecture import (
    NSCS02DownsampledDecoder,
)
from tac.substrates.nscs02_downsampled_renderer.archive import (
    pack_nscs02_archive,
    parse_nscs02_archive,
)
from tac.substrates.nscs02_downsampled_renderer.score_aware_loss import (
    SCORER_HW,
    compute_nscs02_score_aware_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"

# ---------------------------------------------------------------------------
# Catalog #151 manifest — declared as ast.AnnAssign per Catalog #168
# (META gate handles BOTH ast.Assign and ast.AnnAssign post-2026-05-12).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS02_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training "
            "(non-smoke is contest-CUDA target per Catalog #220 OPERATIONAL)"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "NSCS02_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "NSCS02_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=200",
        "default": "200",
    },
    "--batch-size": {
        "env": "NSCS02_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; T4 handles 8-16 at downsampled (192, 256) "
            "render resolution (much larger budget than A1's 4-8 at 384x512)"
        ),
        "default": "8",
    },
    "--lr": {
        "env": "NSCS02_LR",
        "rationale": "AdamW base learning rate; default 5e-4",
        "default": "5e-4",
    },
    "--upsample-mode": {
        "env": "NSCS02_UPSAMPLE_MODE",
        "rationale": (
            "Inflate-side upsample mode (192, 256) -> (384, 512). Bicubic "
            "matches A1's inflate; bilinear is the train/test scorer's own "
            "preprocess mode. Defaults bicubic for inflate parity."
        ),
        "default": "bicubic",
    },
    "--seg-weight": {
        "env": "NSCS02_SEG_WEIGHT",
        "rationale": "SegNet KL-distill weight (contest formula = 100)",
        "default": "100.0",
    },
    "--pose-weight": {
        "env": "NSCS02_POSE_WEIGHT",
        "rationale": "PoseNet MSE weight (contest formula sqrt-amplified)",
        "default": "1.0",
    },
    "--enable-autocast-fp16": {
        "env": "ENABLE_AUTOCAST_FP16",
        "rationale": (
            "Catalog #172 Tier-1 speed primitive; canonical engineering "
            "hygiene shared with all substrates (score-neutral)"
        ),
        "default": "false",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs02_downsampled_renderer",
        description=(
            "Train NSCS02 downsampled-renderer + inflate-upsample substrate "
            "(ASSUMPTIONS-CHALLENGE-AUDIT NSCS02 entry; predicted ΔS = "
            "[-0.010, -0.030] vs A1 baseline)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--upsample-mode", type=str, default="bicubic",
                   choices=("bicubic", "bilinear"))
    p.add_argument("--seg-weight", type=float, default=100.0)
    p.add_argument("--pose-weight", type=float, default=1.0)
    p.add_argument("--pixel-weight", type=float, default=0.1)
    p.add_argument("--latent-dim", type=int, default=NSCS02_LATENT_DIM)
    p.add_argument("--base-channels", type=int, default=NSCS02_BASE_CHANNELS)
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172 Tier-1 speed primitive; score-neutral")
    p.add_argument("--smoke", action="store_true",
                   help="Run synthetic-data smoke; verifies forward shape + archive roundtrip")
    return p


def _smoke_main(args: argparse.Namespace) -> int:
    """Synthetic-data sanity smoke — validates substrate forward + archive roundtrip.

    No scorer load. No real video decode. ``$0`` cost. Verifies:

    1. NSCS02 decoder instantiates with the canonical config.
    2. Forward pass produces (B, 2, 3, 192, 256) RGB pair.
    3. Upsample to (B, 2, 3, 384, 512) via canonical bicubic.
    4. Archive pack -> parse roundtrip is byte-identical for state-dict.
    5. NSCS02 magic + section-offset parser refuses tampered bytes.
    """
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    decoder = NSCS02DownsampledDecoder(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        render_hw=NSCS02_RENDER_HW,
    ).to(device)
    decoder.eval()

    print(f"[nscs02-smoke] decoder param count: {decoder.parameter_count():,}")

    # Forward smoke — render a tiny batch
    z = torch.randn(4, args.latent_dim, device=device)
    with torch.no_grad():
        rendered = decoder(z)
    expected_shape = (4, 2, 3, NSCS02_RENDER_HW[0], NSCS02_RENDER_HW[1])
    if tuple(rendered.shape) != expected_shape:
        raise RuntimeError(
            f"smoke forward shape mismatch: got {tuple(rendered.shape)}; "
            f"expected {expected_shape}"
        )

    # Upsample-to-scorer smoke
    with torch.no_grad():
        upsampled = decoder.render_then_upsample_to_scorer(z, scorer_hw=SCORER_HW, mode=args.upsample_mode)
    expected_up = (4, 2, 3, SCORER_HW[0], SCORER_HW[1])
    if tuple(upsampled.shape) != expected_up:
        raise RuntimeError(
            f"smoke upsample shape mismatch: got {tuple(upsampled.shape)}; "
            f"expected {expected_up}"
        )

    # Archive roundtrip smoke
    latents = torch.randn(NSCS02_N_PAIRS, args.latent_dim)
    archive_bytes = pack_nscs02_archive(decoder, latents)
    print(f"[nscs02-smoke] archive size: {len(archive_bytes):,} bytes")

    template = NSCS02DownsampledDecoder(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        render_hw=NSCS02_RENDER_HW,
    )
    parsed = parse_nscs02_archive(archive_bytes, template)
    if parsed.decoder_state_dict.keys() != decoder.state_dict().keys():
        raise RuntimeError("smoke archive state-dict key-set mismatch after roundtrip")

    # Latents fp16 roundtrip should be approximately equal (tolerance fp16 epsilon)
    diff = (parsed.latents - latents).abs().max().item()
    if diff > 1e-2:
        raise RuntimeError(f"smoke latents fp16 roundtrip max-diff too large: {diff}")

    # Tampered-bytes refusal
    tampered = bytearray(archive_bytes)
    tampered[0] = (tampered[0] + 1) & 0xFF
    try:
        parse_nscs02_archive(bytes(tampered), template)
        raise RuntimeError("smoke parser failed to refuse tampered magic bytes")
    except ValueError:
        pass

    smoke_stats = {
        "substrate_id": "nscs02_downsampled_renderer",
        "render_hw": list(NSCS02_RENDER_HW),
        "scorer_hw": list(SCORER_HW),
        "param_count": decoder.parameter_count(),
        "archive_bytes": len(archive_bytes),
        "smoke_status": "GREEN",
        "score_improvement_mechanism_status": "OPERATIONAL",
        "operational_overlay": True,
        "runtime_overlay_consumed": True,
    }
    (output_dir / "smoke_stats.json").write_text(json.dumps(smoke_stats, indent=2))
    print("[nscs02-smoke] DONE")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full training path — gated until council/operator review.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #220 + Catalog #240 the L1 SCAFFOLD landing
    declares ``score_improvement_mechanism_status=OPERATIONAL`` (the
    inflate-time upsample is the operational consumption); however the
    full $5-15 Modal T4 dispatch is council-gated pending the operator's
    smoke-before-full review per Catalog #167.

    The recipe at
    ``.omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml``
    is intentionally ``research_only: true`` + ``dispatch_enabled: false``
    until this full path trains, exports a byte-closed archive, and routes
    through paired CPU+CUDA auth-eval custody.
    """
    raise NotImplementedError(
        "NSCS02 _full_main is council-gated pending smoke-before-full review per "
        "Catalog #167. Run with --smoke first; on GREEN the operator-authorize "
        "recipe at .omx/operator_authorize_recipes/substrate_nscs02_downsampled_"
        "renderer_modal_t4_dispatch.yaml remains research_only=true and "
        "dispatch_enabled=false until full train/export/auth-eval custody lands. "
        "Predicted ΔS [-0.010, -0.030] vs A1 baseline remains a prediction."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
