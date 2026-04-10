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
from tac.data import decode_archive, decode_video, load_raw_saliency
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

    # Paths (all overridable, with sensible local defaults)
    parser.add_argument("--archive", default=os.environ.get("TAC_ARCHIVE", DEFAULTS["archive"]),
                        help="Compressed archive.zip to train on")
    parser.add_argument("--gt-video", default=os.environ.get("TAC_GT_VIDEO", DEFAULTS["gt_video"]),
                        help="Ground truth video path")
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

    # Output
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default="experiments/postfilter_weights")
    args = parser.parse_args()

    # Validate paths exist
    for name, path in [("archive", args.archive), ("gt-video", args.gt_video),
                       ("saliency", args.saliency), ("models-dir", args.models_dir)]:
        if not Path(path).exists():
            print(f"[train_tac] ERROR: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)

    device = detect_device()
    print(f"[train_tac] device: {device}")
    print(f"[train_tac] config: h={args.hidden} {args.variant} epochs={args.epochs} "
          f"alpha={args.alpha} sal_lambda={args.sal_lambda} loss={args.loss_mode}")
    print(f"[train_tac] archive: {args.archive}")
    print(f"[train_tac] gt_video: {args.gt_video}")

    # Load data
    print("[train_tac] Decoding compressed archive...")
    comp_frames = decode_archive(args.archive)
    print(f"[train_tac] {len(comp_frames)} compressed frames")

    print("[train_tac] Decoding GT video...")
    gt_frames = decode_video(args.gt_video)
    print(f"[train_tac] {len(gt_frames)} GT frames")

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

    # Build config
    config = TrainConfig(
        hidden=args.hidden,
        kernel=args.kernel,
        variant=args.variant,
        epochs=args.epochs,
        alpha=args.alpha,
        sal_lambda=args.sal_lambda,
        lr=args.lr,
        ema_decay=args.ema_decay,
        accum_steps=args.accum_steps,
        loss_mode=args.loss_mode,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        focal_gamma=args.focal_gamma,
        segnet_loss_weight=args.segnet_loss_weight,
        use_dual_saliency=args.use_dual_saliency,
        alpha_seg=args.alpha_seg,
        use_ste_segnet=args.use_ste,
        boundary_weight=args.boundary_weight,
        resume_from=args.resume_from,
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
