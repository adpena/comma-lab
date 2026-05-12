"""Train the sane_hnerv substrate end-to-end on contest video.

Operator-callable training script per the Fields-medal grand council substrate
design wave (2026-05-12). This script is NOT YET dispatched on GPU — it is
the scaffold that the operator approves for first-anchor dispatch.

Council-binding contract:
- Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic data;
  FORBIDDEN per CLAUDE.md "make_synthetic_pair_batch calls in non-smoke" rule)
- Gradient through SegNet + PoseNet via tac.differentiable_eval_roundtrip
- patch upstream rgb_to_yuv6 BEFORE scorer construction (PR #95/#106 contract)
- eval_roundtrip=True with noise_std=0.5 (Hotz STE fix)
- EMA decay=0.997 on weights
- Score-domain Lagrangian (NOT rel_err^2; CLAUDE.md L6)
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up checks

Usage (single-GPU smoke; non-dispatch)::

    .venv/bin/python experiments/train_substrate_sane_hnerv.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sane_hnerv_smoke_<utc> \\
        --epochs 5 --smoke

For real first-anchor dispatch, the operator runs this through the canonical
Modal / Vast.ai / Lightning wrapper with all TIER_1 flags threaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


# Catalog #151 manifest — every flag in TIER_1 must be threaded by any
# operator wrapper that subprocess-invokes this trainer.
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--video-path": {
        "env": "VIDEO_PATH",
        "rationale": "Score-aware substrate MUST train against contest video (NOT synthetic)",
    },
    "--output-dir": {
        "env": "OUTPUT_DIR",
        "rationale": "Custody location for checkpoints + manifest",
    },
    "--epochs": {
        "env": "EPOCHS",
        "rationale": "Substrate engineering pass; under-training silently regresses",
    },
}
"""Operator-required flags for non-smoke dispatch. See Catalog #151."""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_sane_hnerv",
        description=__doc__,
    )
    p.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to upstream/videos/0.mkv (or equivalent 1200-frame contest video)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write checkpoints + score logs + manifest",
    )
    p.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of pair indices per batch (default 8)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (Adam)",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device. 'cpu' permitted only with --smoke for local development.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke-mode forward pass (no scorer load; CPU OK)",
    )
    p.add_argument(
        "--latent-dim",
        type=int,
        default=28,
        help="Per-pair latent dimensionality (council default 28)",
    )
    p.add_argument(
        "--sin-frequency",
        type=float,
        default=30.0,
        help="SIREN sin activation frequency",
    )
    p.add_argument(
        "--pose-weight-scale",
        type=float,
        default=1.0,
        help="At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71",
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="STE noise std for eval-roundtrip simulation (Hotz fix)",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help="EMA decay for the weight shadow (CLAUDE.md non-negotiable)",
    )
    return p


def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.sane_hnerv.architecture import SaneHnervConfig, SaneHnervSubstrate

    # 2-block tiny config that fits in CPU RAM
    cfg = SaneHnervConfig(
        latent_dim=args.latent_dim,
        embed_dim=64,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(32, 16, 8),
        sin_frequency=args.sin_frequency,
        num_pairs=4,
        output_height=24,
        output_width=32,
        num_upsample_blocks=2,
    )
    device = torch.device(args.device if args.device == "cpu" else "cuda")
    model = SaneHnervSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] sane_hnerv params: {model.num_parameters():,}")
    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "smoke": True,
    }
    ckpt_path = args.output_dir / "smoke_checkpoint.pt"
    import torch

    torch.save(ckpt, ckpt_path)
    print(f"[smoke] wrote {ckpt_path}")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full training entry point — requires CUDA + score-aware scorers.

    This path is OPERATOR-GATED and requires the wrapper (Modal/Vast.ai/
    Lightning) to thread all TIER_1 flags + run the dual CPU+CUDA auth eval
    afterward per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA" rule.
    """
    if args.device != "cuda":
        raise SystemExit(
            "Full training requires --device cuda. Use --smoke for CPU smoke."
        )
    # The actual full-training loop wires:
    # 1. tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()
    # 2. construct SegNet + PoseNet from upstream/contest path
    # 3. score-aware Lagrangian SaneHnervScoreAwareLoss
    # 4. pyav decode of upstream/videos/0.mkv (NOT synthetic)
    # 5. tac.training.EMA shadow (decay=0.997)
    # 6. eval_roundtrip=True / noise_std=0.5
    # 7. auth-eval on best ckpt -> dual CPU+CUDA
    #
    # The full implementation is deferred to the operator-gated dispatch
    # subagent per the council's "minimum-viable-integration-loop first" rule.
    # The scaffold ships this skeleton; the full loop will be wired in the
    # follow-up subagent once OD-SUBSTRATE-1 is approved.
    print(
        "[full] not-yet-implemented — operator-gated; use --smoke for now",
        file=sys.stderr,
    )
    print(
        "[full] design memo: .omx/research/grand_council_fields_medal_substrate_design_20260512.md",
        file=sys.stderr,
    )
    return 3


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
