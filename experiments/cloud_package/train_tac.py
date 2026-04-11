#!/usr/bin/env python3
"""Unified training script using the tac library.

This is THE canonical training entry point. All experiments should use this.
Fully portable — all paths configurable via CLI args or env vars.

Usage (local):
    .venv/bin/python experiments/train_tac.py --tag my_run

Usage (cloud / Modal):
    python train_tac.py --tag h96_cloud \
        --archive /data/archive.zip \
        --gt-video /data/videos/0.mkv \
        --saliency /data/saliency.npy \
        --models-dir /data/models \
        --upstream-dir /data/upstream
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure tac is importable (works both local and when script is copied elsewhere)
_script_dir = Path(__file__).parent
_repo = _script_dir.parent
if (_repo / "src" / "tac").exists():
    sys.path.insert(0, str(_repo))
    sys.path.insert(0, str(_repo / "src"))

from tac.architectures import build_postfilter
from tac.data import load_raw_saliency
from tac.scorer import detect_device, load_scorers
from tac.training import TrainConfig, Trainer

# Default paths (relative to repo root, overridable via CLI)
_UPSTREAM = _repo / "workspace" / "upstream" / "comma_video_compression_challenge"
DEFAULTS = {
    "archive": str(_repo / "submissions" / "robust_current" / "archive.zip"),
    "gt_video": str(_UPSTREAM / "videos" / "0.mkv"),
    "saliency": str(_repo / "experiments" / "masks" / "posenet_saliency.npy"),
    "models_dir": str(_UPSTREAM / "models"),
    "upstream_dir": str(_UPSTREAM),
}


def main():
    parser = argparse.ArgumentParser(description="Train with tac library")

    # Named profiles (council-recommended settings)
    parser.add_argument("--profile", default=None,
                        help="Named profile from tac.profiles (e.g., council_v1, segnet_attack, smoke). "
                        "CLI args override profile values.")

    # Paths (all overridable, with sensible local defaults)
    parser.add_argument("--archive", default=os.environ.get("TAC_ARCHIVE", DEFAULTS["archive"]),
                        help="Compressed archive.zip to train on")
    parser.add_argument("--gt-video", default=os.environ.get("TAC_GT_VIDEO", DEFAULTS["gt_video"]),
                        help="Ground truth video path")
    parser.add_argument("--precomputed", default=os.environ.get("TAC_PRECOMPUTED", None),
                        help="Directory with precomputed comp_frames.pt + gt_frames.pt (skips decode)")
    parser.add_argument("--saliency", default=os.environ.get("TAC_SALIENCY", DEFAULTS["saliency"]),
                        help="Saliency map .npy path")
    parser.add_argument("--models-dir", default=os.environ.get("TAC_MODELS_DIR", DEFAULTS["models_dir"]),
                        help="Directory containing posenet.safetensors and segnet.safetensors")
    parser.add_argument("--upstream-dir", default=os.environ.get("TAC_UPSTREAM_DIR", DEFAULTS["upstream_dir"]),
                        help="Upstream repo root (for scorer imports)")

    # Architecture
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--sal-lambda", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate int8 checkpoint every N epochs (default 5)")
    parser.add_argument("--hard-frame-ratio", type=float, default=0.0,
                        help="Fraction of hard SegNet pairs to oversample (0=uniform, 0.5=half hard)")
    parser.add_argument("--error-replay-every", type=int, default=0,
                        help="Recompute hard-frame weights using model output every N epochs (0=static)")

    # Loss mode
    parser.add_argument("--loss-mode", default="standard", choices=["standard", "temperature", "focal_ste", "kl_distill"])
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--segnet-loss-weight", type=float, default=100.0)

    # Dual saliency
    parser.add_argument("--use-dual-saliency", action="store_true")
    parser.add_argument("--alpha-seg", type=float, default=200.0)

    # SegNet STE
    parser.add_argument("--use-ste", action="store_true")
    parser.add_argument("--boundary-weight", type=float, default=1.0)

    # Resume
    parser.add_argument("--resume-from", type=str, default=None)

    # Wall-clock timeout
    parser.add_argument("--wall-clock-timeout", type=int, default=0,
                        help="Max training wall-clock seconds (0=unlimited, 39600=11h for Kaggle)")

    # Output
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default="experiments/postfilter_weights")
    args = parser.parse_args()

    # Validate paths — saliency + models always needed, archive/video only without precomputed
    for name, path in [("saliency", args.saliency), ("models-dir", args.models_dir)]:
        if not Path(path).exists():
            print(f"[train_tac] ERROR: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)
    if not args.precomputed:
        for name, path in [("archive", args.archive), ("gt-video", args.gt_video)]:
            if not Path(path).exists():
                print(f"[train_tac] ERROR: {name} not found: {path}", file=sys.stderr)
                sys.exit(1)

    device = detect_device()
    print(f"[train_tac] device: {device}")
    print(f"[train_tac] config: h={args.hidden} {args.variant} epochs={args.epochs} "
          f"alpha={args.alpha} sal_lambda={args.sal_lambda} loss={args.loss_mode}")

    # Load data — precomputed tensors (instant) or video decode (slow)
    from tac.data import load_frames
    comp_frames, gt_frames = load_frames(
        archive_path=args.archive,
        gt_video_path=args.gt_video,
        precomputed_dir=args.precomputed,
    )
    print(f"[train_tac] {len(comp_frames)} compressed + {len(gt_frames)} GT frames")

    print("[train_tac] Loading saliency map...")
    raw_saliency = load_raw_saliency(args.saliency)
    print(f"[train_tac] Saliency shape: {raw_saliency.shape}")

    # Load scorers
    models_dir = Path(args.models_dir)
    print("[train_tac] Loading scorer models...")
    posenet, segnet = load_scorers(
        models_dir / "posenet.safetensors",
        models_dir / "segnet.safetensors",
        device=device,
        upstream_dir=args.upstream_dir,
    )

    # Build model
    model = build_postfilter(args.variant, hidden=args.hidden, kernel=args.kernel)
    print(f"[train_tac] Model: {args.variant} h={args.hidden} "
          f"({sum(p.numel() for p in model.parameters())} params)")

    # Build config — profile provides defaults, CLI args override
    profile_defaults = {}
    if args.profile:
        from tac.profiles import PROFILES
        if args.profile not in PROFILES:
            print(f"ERROR: unknown profile '{args.profile}'. Available: {list(PROFILES.keys())}")
            sys.exit(1)
        profile_defaults = PROFILES[args.profile]
        print(f"[train_tac] Using profile: {args.profile}")

    def _val(cli_arg, profile_key, cli_default):
        """CLI arg wins if explicitly set (differs from argparse default), else profile."""
        cli_val = getattr(args, cli_arg.replace("-", "_"))
        if cli_val != cli_default and cli_arg.replace("-", "_") in vars(args):
            return cli_val
        return profile_defaults.get(profile_key, cli_val)

    config = TrainConfig(
        hidden=_val("hidden", "hidden", 64),
        kernel=_val("kernel", "kernel", 3),
        variant=_val("variant", "variant", "standard"),
        epochs=_val("epochs", "epochs", 2500),
        alpha=_val("alpha", "alpha", 20.0),
        sal_lambda=_val("sal-lambda", "sal_lambda", 1.0),
        lr=_val("lr", "lr", 5e-4),
        ema_decay=_val("ema-decay", "ema_decay", 0.997),
        accum_steps=_val("accum-steps", "accum_steps", 4),
        eval_every=_val("eval-every", "eval_every", 5),
        hard_frame_ratio=_val("hard-frame-ratio", "hard_frame_ratio", 0.0),
        error_replay_every=_val("error-replay-every", "error_replay_every", 0),
        loss_mode=_val("loss-mode", "loss_mode", "standard"),
        temperature_start=_val("temperature-start", "temperature_start", 1.0),
        temperature_end=_val("temperature-end", "temperature_end", 0.05),
        focal_gamma=_val("focal-gamma", "focal_gamma", 2.0),
        segnet_loss_weight=_val("segnet-loss-weight", "segnet_loss_weight", 100.0),
        use_dual_saliency=args.use_dual_saliency or profile_defaults.get("use_dual_saliency", False),
        alpha_seg=_val("alpha-seg", "alpha_seg", 200.0),
        use_ste_segnet=args.use_ste or profile_defaults.get("use_ste_segnet", False),
        boundary_weight=_val("boundary-weight", "boundary_weight", 1.0),
        boundary_anneal=profile_defaults.get("boundary_anneal", False),
        resume_from=args.resume_from,
        wall_clock_timeout=_val("wall-clock-timeout", "wall_clock_timeout", 0),
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Train
    trainer = Trainer(model, config, device=device)
    best = trainer.fit_lazy(
        comp_frames, gt_frames, posenet, segnet, raw_saliency,
        subsample=args.subsample,
    )
    print(f"[train_tac] Done. Best scorer: {best:.4f}")


if __name__ == "__main__":
    main()
