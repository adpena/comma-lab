#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 1 pre-training for MaskRenderer in MLX (4.7x faster than PyTorch MPS).
[MPS-research-signal]

Phase 1 uses only L1 + edge loss (no scorer models). The scorer models are
PyTorch-only, so Phase 1 runs entirely in MLX on Apple Silicon Metal.

After pre-training completes, weights are converted to PyTorch and saved as
a checkpoint that train_renderer.py can resume from for Phase 2 (scorer
fine-tuning).

Usage:
    .venv/bin/python -m tac.experiments.train_renderer_mlx --tag mlx_pretrain --epochs 100
    .venv/bin/python -m tac.experiments.train_renderer_mlx --tag mlx_pretrain --epochs 100 --lr 2e-3

    # Then continue Phase 2 in PyTorch:
    .venv/bin/python -m tac.experiments.train_renderer --resume-from experiments/postfilter_weights/mlx_pretrained_mlx_pretrain.pt --tag v1

Benchmark target:
    Phase 1 epoch: ~20-25 seconds (vs ~100 seconds in PyTorch MPS)
    Phase 1 complete (100 epochs): ~40 minutes (vs ~3 hours in PyTorch)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

# ── Path setup ──────────────────────────────────────────────────────────

_script_dir = Path(__file__).resolve().parent
_repo = _script_dir.parent.parent.parent  # src/tac/experiments -> repo root
sys.path.insert(0, str(_repo / "src"))

_upstream = _repo / "workspace" / "upstream" / "comma_video_compression_challenge"

from tac.mlx_renderer import (  # noqa: E402
    build_mlx_renderer,
    mlx_to_pytorch,
    pretrain_loss_fn,
)
from tac.utils import setup_signal_handlers  # noqa: E402


# ── Argument parsing ────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 MLX pre-training for mask renderer (L1 + edge loss)"
    )
    # Architecture (must match PyTorch for weight transfer)
    p.add_argument("--base-ch", type=int, default=36)
    p.add_argument("--mid-ch", type=int, default=60)
    p.add_argument("--embed-dim", type=int, default=6)
    p.add_argument("--motion-hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=1)

    # Training
    p.add_argument("--epochs", type=int, default=100,
                   help="Phase 1 pre-training epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--subsample", type=int, default=4,
                   help="Train on 1/N of pairs per epoch")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 = no clipping)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--compile", action="store_true", default=False,
                   help="Use mx.compile() for loss+grad (EXPERIMENTAL: "
                        "incompatible with nn.value_and_grad stateful updates)")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size (2 optimal on M5 Max, 1.46x throughput)")
    p.add_argument("--log-every", type=int, default=10,
                   help="Print loss every N epochs")

    # Data
    p.add_argument("--precomputed", type=str,
                   default=str(_repo / "experiments" / "precomputed_local"),
                   help="Dir with gt_frames.pt and masks (skip video decode)")

    # Output
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--output-dir", type=str,
                   default=str(_repo / "experiments" / "postfilter_weights"))
    return p.parse_args(argv)


# ── Data loading ────────────────────────────────────────────────────────


def load_data_as_mlx(precomputed_dir: str) -> tuple[mx.array, mx.array]:
    """Load GT frames and masks as MLX arrays.

    Expects:
        precomputed_dir/gt_frames.pt  -- (N, H, W, 3) uint8 or float tensor
        precomputed_dir/masks.pt      -- (N, H, W) long tensor

    If masks.pt doesn't exist, falls back to extracting masks via PyTorch
    SegNet (one-time cost).

    Returns:
        gt_frames: (N, H, W, 3) float32 MLX array in [0, 255]
        masks: (N, H, W) int32 MLX array
    """
    import torch

    pc = Path(precomputed_dir)

    # Load GT frames
    frames_path = pc / "gt_frames.pt"
    if not frames_path.exists():
        raise FileNotFoundError(
            f"No precomputed GT frames at {frames_path}. "
            "Run the PyTorch pipeline first to generate them."
        )
    print(f"[data] Loading GT frames from {frames_path}")
    frames_pt = torch.load(frames_path, map_location="cpu", weights_only=True)
    if isinstance(frames_pt, list):
        frames_pt = torch.stack(frames_pt)
    # Ensure HWC float
    if frames_pt.dtype == torch.uint8:
        frames_np = frames_pt.numpy().astype(np.float32)
    else:
        frames_np = frames_pt.float().numpy()
    gt_frames = mx.array(frames_np)
    print(f"[data] GT frames: {gt_frames.shape}, dtype={gt_frames.dtype}")

    # Load masks
    masks_path = pc / "masks.pt"
    if masks_path.exists():
        print(f"[data] Loading masks from {masks_path}")
        masks_pt = torch.load(masks_path, map_location="cpu", weights_only=True)
        masks_np = masks_pt.numpy().astype(np.int32)
        masks = mx.array(masks_np)
    else:
        # Extract masks via PyTorch SegNet (one-time)
        print("[data] masks.pt not found, extracting via SegNet...")
        masks = _extract_and_save_masks(frames_pt, pc)

    print(f"[data] Masks: {masks.shape}, dtype={masks.dtype}")
    return gt_frames, masks


def _extract_and_save_masks(frames_pt, precomputed_dir: Path) -> mx.array:
    """Extract masks via PyTorch SegNet and save for future reuse."""
    import torch
    sys.path.insert(0, str(_upstream))
    from tac.mask_codec import extract_masks
    from tac.scorer import detect_device, load_scorers

    device = detect_device()
    _, segnet = load_scorers(
        _upstream / "models" / "posenet.safetensors",
        _upstream / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=str(_upstream),
    )

    frames_list = [frames_pt[i] for i in range(frames_pt.shape[0])]
    masks_pt = extract_masks(frames_list, segnet, device=device, batch_size=4)
    masks_pt = masks_pt.cpu()

    # Save for reuse
    masks_path = precomputed_dir / "masks.pt"
    torch.save(masks_pt, masks_path)
    print(f"[data] Saved masks to {masks_path}")

    return mx.array(masks_pt.numpy().astype(np.int32))


# ── Pair generation ─────────────────────────────────────────────────────


def pair_start_indices(n_frames: int) -> list[int]:
    """Valid pair start indices (consecutive frames, 20-frame groups).

    NOTE: This duplicates tac.data.pair_start_indices to avoid importing
    PyTorch (tac.data depends on torch) in the MLX-only training path.
    """
    indices = []
    for i in range(n_frames - 1):
        if (i % 20) < 19:  # skip last frame in each 20-frame group
            indices.append(i)
    return indices


def get_pair(
    gt_frames: mx.array,
    masks: mx.array,
    start: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Get a single mask pair and GT frame pair.

    Returns:
        mask_t: (1, H, W) int32
        mask_t1: (1, H, W) int32
        gt_pair: (1, 2, H, W, 3) float32 in [0, 255]
    """
    mask_t = masks[start:start+1]       # (1, H, W)
    mask_t1 = masks[start+1:start+2]    # (1, H, W)

    # GT frames at mask resolution (384x512)
    # If GT frames are larger, we need to downsample
    ft = gt_frames[start:start+1]       # (1, H, W, 3)
    ft1 = gt_frames[start+1:start+2]    # (1, H, W, 3)

    _, mH, mW = mask_t.shape
    _, fH, fW, _ = ft.shape

    if fH != mH or fW != mW:
        # Nearest-neighbor downsample GT to mask resolution
        row_idx = mx.arange(mH) * fH // mH
        col_idx = mx.arange(mW) * fW // mW
        ft = ft[:, row_idx][:, :, col_idx]
        ft1 = ft1[:, row_idx][:, :, col_idx]

    gt_pair = mx.stack([ft, ft1], axis=1)  # (1, 2, H, W, 3)
    return mask_t, mask_t1, gt_pair


# ── Training loop ───────────────────────────────────────────────────────


def train(args: argparse.Namespace):
    print("[mlx_train] Phase 1 MLX pre-training")
    print(f"[mlx_train] Architecture: base_ch={args.base_ch}, mid_ch={args.mid_ch}, "
          f"embed_dim={args.embed_dim}, depth={args.depth}")
    print(f"[mlx_train] Training: epochs={args.epochs}, lr={args.lr}, "
          f"subsample=1/{args.subsample}")
    print(f"[mlx_train] Optimizations: compile={args.compile}, "
          f"batch_size={args.batch_size}")

    # Load data
    gt_frames, masks = load_data_as_mlx(args.precomputed)

    # Build model
    model = build_mlx_renderer(
        num_classes=5,
        embed_dim=args.embed_dim,
        base_ch=args.base_ch,
        mid_ch=args.mid_ch,
        motion_hidden=args.motion_hidden,
        depth=args.depth,
    )

    # NOTE: FP16 was benchmarked at 1.20x speedup but causes NaN in GroupNorm
    # due to variance overflow with 3M+ elements in fp16 accumulation.
    # bfloat16 not fully supported in MLX 0.31.1. Staying fp32.

    # Optimizer with warmup cosine schedule
    warmup_steps = args.warmup_epochs
    total_steps = args.epochs
    warmup_schedule = optim.linear_schedule(
        init=1e-7,
        end=args.lr,
        steps=warmup_steps,
    )
    cosine_schedule = optim.cosine_decay(
        init=args.lr,
        decay_steps=total_steps - warmup_steps,
        end=1e-5,
    )
    lr_schedule = optim.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)

    # Grad clipping — applied manually in the training loop (see below).
    # NOTE: optim.clip_grad_norm(optimizer, ...) crashes with
    # "AttributeError: 'AdamW' object has no attribute 'square'" because
    # the MLX wrapper tries to call .square() on optimizer state, not grads.
    max_grad_norm = args.grad_clip if args.grad_clip > 0 else 0.0

    # Pair indices
    n_frames = gt_frames.shape[0]
    all_starts = pair_start_indices(n_frames)
    n_total = len(all_starts)
    train_size = max(1, n_total // args.subsample)
    print(f"[mlx_train] {train_size}/{n_total} pairs per epoch")

    # Loss + grad function
    # NOTE: mx.compile() is incompatible with nn.value_and_grad + stateful
    # optimizer updates in MLX 0.31.1. The grad closure captures internal
    # arrays that can't be registered as compile inputs. Compile is disabled
    # by default. When MLX adds first-class compiled training support, enable.
    loss_and_grad_fn = nn.value_and_grad(model, pretrain_loss_fn)
    if args.compile:
        print("[mlx_train] WARNING: mx.compile() with value_and_grad is experimental")
        compiled_lag = mx.compile(
            lambda mt, mt1, gp: loss_and_grad_fn(model, mt, mt1, gp)
        )
    else:
        compiled_lag = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Emergency save on SIGTERM/SIGINT
    def _emergency_save():
        mlx_path = out_dir / f"mlx_renderer_{args.tag}_emergency.safetensors"
        model.save_weights(str(mlx_path))
        print(f"[mlx_train] Emergency save: {mlx_path}")

    setup_signal_handlers(_emergency_save)

    best_loss = float("inf")
    best_epoch = -1
    total_start = time.monotonic()

    for epoch in range(args.epochs):
        epoch_start = time.monotonic()

        # Shuffle and subsample pairs
        perm = list(range(n_total))
        random.shuffle(perm)
        perm = perm[:train_size]

        epoch_loss = 0.0
        n_steps = 0

        # Build mini-batches of size args.batch_size
        # Benchmarked: bs=2 gives 3.8 pairs/sec vs 2.6 at bs=1 (1.46x throughput)
        batch_starts = []
        for pair_idx in perm:
            batch_starts.append(all_starts[pair_idx])

        for bi in range(0, len(batch_starts), args.batch_size):
            batch_indices = batch_starts[bi:bi + args.batch_size]
            bs = len(batch_indices)

            # Stack batch
            mask_ts, mask_t1s, gt_pairs = [], [], []
            for start in batch_indices:
                mt, mt1, gp = get_pair(gt_frames, masks, start)
                # Horizontal flip augmentation (50% probability)
                if random.random() < 0.5:
                    mt = mt[:, :, ::-1]
                    mt1 = mt1[:, :, ::-1]
                    gp = gp[:, :, :, ::-1, :]
                mask_ts.append(mt)
                mask_t1s.append(mt1)
                gt_pairs.append(gp)

            if bs > 1:
                mask_t = mx.concatenate(mask_ts, axis=0)
                mask_t1 = mx.concatenate(mask_t1s, axis=0)
                gt_pair = mx.concatenate(gt_pairs, axis=0)
            else:
                mask_t, mask_t1, gt_pair = mask_ts[0], mask_t1s[0], gt_pairs[0]

            # Forward + backward
            if compiled_lag is not None:
                loss, grads = compiled_lag(mask_t, mask_t1, gt_pair)
            else:
                loss, grads = loss_and_grad_fn(model, mask_t, mask_t1, gt_pair)

            # Manual gradient clipping (replaces broken optim.clip_grad_norm)
            if max_grad_norm > 0:
                grads_flat = mlx.utils.tree_flatten(grads)
                total_norm = mx.sqrt(
                    sum(g.square().sum() for g in grads_flat if isinstance(g, mx.array))
                )
                clip_coef = min(1.0, max_grad_norm / (total_norm.item() + 1e-6))
                if clip_coef < 1.0:
                    grads = mlx.utils.tree_map(lambda g: g * clip_coef, grads)

            optimizer.update(model, grads)
            # Sync to avoid memory buildup
            mx.eval(model.parameters(), optimizer.state, loss)

            epoch_loss += loss.item() * bs
            n_steps += bs

        avg_loss = epoch_loss / max(n_steps, 1)
        epoch_sec = time.monotonic() - epoch_start
        try:
            lr = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else args.lr
        except Exception:
            lr = args.lr

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch

            # Save MLX checkpoint
            mlx_path = out_dir / f"mlx_renderer_{args.tag}_best.safetensors"
            model.save_weights(str(mlx_path))

        # Log
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            eta_min = epoch_sec * (args.epochs - epoch - 1) / 60
            print(f"[ep {epoch:4d}/{args.epochs}] loss={avg_loss:.6f} "
                  f"best={best_loss:.6f}@{best_epoch} "
                  f"lr={lr:.6f} {epoch_sec:.1f}s/ep ETA={eta_min:.1f}m")

    total_sec = time.monotonic() - total_start
    print(f"\n[mlx_train] Phase 1 complete in {total_sec / 60:.1f} minutes")
    print(f"[mlx_train] Best loss: {best_loss:.6f} at epoch {best_epoch}")

    # ── Convert best MLX weights to PyTorch checkpoint ────────────────

    print("[mlx_train] Converting MLX weights to PyTorch...")

    # Load best MLX weights
    best_mlx_path = out_dir / f"mlx_renderer_{args.tag}_best.safetensors"
    model.load_weights(str(best_mlx_path))

    # Convert to PyTorch state dict
    flat_params = {}
    _flatten_for_conversion(model.parameters(), "", flat_params)
    pt_state = mlx_to_pytorch(flat_params)

    # Sanity check: compare keys against a fresh PyTorch model
    from tac.renderer import build_renderer as _build_pt_renderer
    _ref_model = _build_pt_renderer(
        num_classes=5, embed_dim=args.embed_dim, base_ch=args.base_ch,
        mid_ch=args.mid_ch, motion_hidden=args.motion_hidden, depth=args.depth,
    )
    _ref_keys = set(_ref_model.state_dict().keys())
    _got_keys = set(pt_state.keys())
    _missing = _ref_keys - _got_keys
    _unexpected = _got_keys - _ref_keys
    if _missing:
        print(f"[mlx_train] WARNING: {len(_missing)} keys missing from converted state dict: "
              f"{sorted(_missing)[:5]}{'...' if len(_missing) > 5 else ''}")
    if _unexpected:
        print(f"[mlx_train] WARNING: {len(_unexpected)} unexpected keys in converted state dict: "
              f"{sorted(_unexpected)[:5]}{'...' if len(_unexpected) > 5 else ''}")
    del _ref_model

    # Save as PyTorch checkpoint compatible with train_renderer.py --resume-from
    import torch

    # Build a full training state checkpoint.
    # Intentionally omits "optimizer" and "scheduler" keys: Phase 2
    # (train_renderer.py) will detect the missing keys and create fresh
    # optimizer/scheduler appropriate for scorer fine-tuning. This avoids
    # carrying over MLX optimizer state that has no PyTorch equivalent.
    pt_checkpoint = {
        "epoch": -1,  # signals "pretrained, start from epoch 0 in Phase 2"
        "model": pt_state,
        "ema_shadow": pt_state,  # EMA starts from pretrained weights
        "best_scorer": float("inf"),
        "best_epoch": -1,
        "baseline_pose": None,
        "baseline_seg": None,
    }

    pt_path = out_dir / f"mlx_pretrained_{args.tag}.pt"
    pt_tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(pt_checkpoint, pt_tmp)
    pt_tmp.rename(pt_path)
    print(f"[mlx_train] Saved PyTorch checkpoint: {pt_path}")
    print("[mlx_train] Resume Phase 2 with:")
    print(f"  .venv/bin/python -m tac.experiments.train_renderer "
          f"--resume-from {pt_path} --tag <phase2_tag>")

    # Save metadata
    meta = {
        "phase": "mlx_pretrain",
        "epochs": args.epochs,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "total_minutes": round(total_sec / 60, 2),
        "architecture": {
            "base_ch": args.base_ch,
            "mid_ch": args.mid_ch,
            "embed_dim": args.embed_dim,
            "motion_hidden": args.motion_hidden,
            "depth": args.depth,
        },
        "pt_checkpoint": str(pt_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / f"mlx_pretrained_{args.tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[mlx_train] Metadata: {meta_path}")

    return best_loss


def _flatten_for_conversion(params, prefix: str, out: dict):
    """Flatten MLX nested params to dot-separated keys for conversion."""
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            _flatten_for_conversion(v, new_prefix, out)
    elif isinstance(params, list):
        for i, v in enumerate(params):
            _flatten_for_conversion(v, f"{prefix}.{i}", out)
    elif isinstance(params, mx.array):
        out[prefix] = params


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result is not None else 1)
