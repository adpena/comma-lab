#!/usr/bin/env python3
"""Unified training script using the tac library.

This is THE canonical training entry point. All experiments should use this.
It wires TrainConfig → Trainer.fit_lazy with the full tac pipeline.

Usage:
    .venv/bin/python experiments/train_tac.py --hidden 64 --epochs 2500 --tag my_run
    .venv/bin/python experiments/train_tac.py --hidden 64 --sal-lambda 0 --tag no_sal
    .venv/bin/python experiments/train_tac.py --hidden 64 --loss-mode temperature --tag temp_anneal
    .venv/bin/python experiments/train_tac.py --hidden 64 --loss-mode focal_ste --tag focal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure tac is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from tac.architectures import build_postfilter
from tac.data import decode_video, decode_archive, load_raw_saliency, pair_start_indices
from tac.scorer import load_scorers, detect_device
from tac.training import Trainer, TrainConfig

# Paths (relative to repo root)
REPO = Path(__file__).parent.parent
UPSTREAM = REPO / "workspace" / "upstream" / "comma_video_compression_challenge"
ARCHIVE_ZIP = REPO / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
GT_VIDEO = UPSTREAM / "videos" / "0.mkv"
SALIENCY = REPO / "experiments" / "masks" / "posenet_saliency.npy"
MODELS_DIR = UPSTREAM / "models"


def main():
    parser = argparse.ArgumentParser(description="Train with tac library")
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
    # Loss mode (council SegNet interventions)
    parser.add_argument("--loss-mode", default="standard", choices=["standard", "temperature", "focal_ste"])
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

    device = detect_device()
    print(f"[train_tac] device: {device}")
    print(f"[train_tac] config: h={args.hidden} {args.variant} epochs={args.epochs} "
          f"alpha={args.alpha} sal_lambda={args.sal_lambda} loss={args.loss_mode}")

    # Load data
    print("[train_tac] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[train_tac] {len(comp_frames)} compressed frames")

    print("[train_tac] Decoding GT video...")
    gt_frames = decode_video(str(GT_VIDEO))
    print(f"[train_tac] {len(gt_frames)} GT frames")

    print("[train_tac] Loading saliency map...")
    raw_saliency = load_raw_saliency(str(SALIENCY))
    print(f"[train_tac] Saliency shape: {raw_saliency.shape}")

    # Load scorers
    print("[train_tac] Loading scorer models...")
    posenet, segnet = load_scorers(
        MODELS_DIR / "posenet.safetensors",
        MODELS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=str(UPSTREAM),
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
