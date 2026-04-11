#!/usr/bin/env python3
"""Training script for GPU-lane mask-conditioned renderer.

Pipeline:
    GT frames -> frozen SegNet -> 5-class masks (extracted once at startup)
    Training loop: mask pairs -> PairGenerator -> scorer_loss -> backprop

Usage:
    .venv/bin/python experiments/train_renderer.py --profile mask_renderer_smoke --tag test
    .venv/bin/python experiments/train_renderer.py --profile mask_renderer_full --tag v1
    .venv/bin/python experiments/train_renderer.py --epochs 100 --lr 1e-3 --tag quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ── Path setup ──────────────────────────────────────────────────────────

_script_dir = Path(__file__).resolve().parent
_repo = _script_dir.parent
sys.path.insert(0, str(_repo / "src"))

_upstream = _repo / "workspace" / "upstream" / "comma_video_compression_challenge"
if _upstream.exists() and str(_upstream) not in sys.path:
    sys.path.insert(0, str(_upstream))

# ── Imports (after path setup) ──────────────────────────────────────────

from tac.data import decode_video, pair_from_frames, pair_start_indices  # noqa: E402
from tac.fp4_quantize import (  # noqa: E402
    QATRendererFP4,
    dequantize_fp4,
    quantize_fp4,
    save_fp4,
)
from tac.losses import eval_scorer_loss, scorer_forward_pair, scorer_loss, scorer_loss_cached  # noqa: E402
from tac.mask_codec import extract_masks, mask_pair_from_index  # noqa: E402
from tac.profiles import PROFILES  # noqa: E402
from tac.renderer import build_renderer  # noqa: E402
from tac.scorer import detect_device, load_scorers  # noqa: E402
from tac.training import EMA  # noqa: E402


# ── Argument parsing ────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mask-conditioned renderer (GPU lane)")

    # Profile
    renderer_profiles = [k for k in PROFILES if k.startswith("mask_renderer")]
    p.add_argument("--profile", type=str, default=None, choices=list(PROFILES.keys()),
                   help=f"Named profile. Renderer profiles: {renderer_profiles}")

    # Architecture
    p.add_argument("--base-ch", type=int, default=None, help="MaskRenderer base channels")
    p.add_argument("--mid-ch", type=int, default=None, help="MaskRenderer bottleneck channels")
    p.add_argument("--embed-dim", type=int, default=None, help="Per-class embedding dim")
    p.add_argument("--motion-hidden", type=int, default=None, help="MotionPredictor hidden channels")

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--subsample", type=int, default=4,
                   help="Train on 1/N of pairs per epoch")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--segnet-weight", type=float, default=None,
                   help="Weight for SegNet term in scorer_loss")
    p.add_argument("--use-qat", action="store_true", default=True,
                   help="Enable FP4 QAT (default: on)")
    p.add_argument("--no-qat", dest="use_qat", action="store_false")

    # Data
    p.add_argument("--precomputed", type=str,
                   default=str(_repo / "experiments" / "precomputed_local"),
                   help="Dir with gt_frames.pt (skip video decode)")
    p.add_argument("--video", type=str,
                   default=str(_upstream / "videos" / "0.mkv"),
                   help="GT video path (used if no precomputed)")
    p.add_argument("--mask-batch-size", type=int, default=4,
                   help="Batch size for SegNet mask extraction")

    # Output
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--output-dir", type=str,
                   default=str(_repo / "experiments" / "postfilter_weights"))
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args(argv)

    # Apply profile defaults, then CLI overrides
    profile_vals = {}
    if args.profile:
        profile_vals = dict(PROFILES[args.profile])

    def _resolve(cli_val, profile_key, default):
        if cli_val is not None:
            return cli_val
        return profile_vals.get(profile_key, default)

    args.base_ch = _resolve(args.base_ch, "hidden", 36)
    args.mid_ch = _resolve(args.mid_ch, "mid_ch", 60)
    args.embed_dim = _resolve(args.embed_dim, "embed_dim", 6)
    args.motion_hidden = _resolve(args.motion_hidden, "motion_hidden", 32)
    args.epochs = _resolve(args.epochs, "epochs", 200)
    args.lr = _resolve(args.lr, "lr", 1e-3)
    args.ema_decay = _resolve(args.ema_decay, "ema_decay", 0.997)
    args.accum_steps = _resolve(args.accum_steps, "accum_steps", 2)
    args.eval_every = _resolve(args.eval_every, "eval_every", 10)
    args.segnet_weight = _resolve(args.segnet_weight, "segnet_loss_weight", 100.0)

    return args


# ── Data loading ────────────────────────────────────────────────────────


def load_gt_frames(args: argparse.Namespace) -> list[torch.Tensor]:
    """Load GT frames from precomputed .pt or decode from video."""
    precomputed = Path(args.precomputed) / "gt_frames.pt"
    if precomputed.exists():
        print(f"[data] Loading precomputed GT frames from {precomputed}")
        frames_tensor = torch.load(precomputed, map_location="cpu", weights_only=True)
        # Convert (N, H, W, 3) tensor to list of (H, W, 3)
        if isinstance(frames_tensor, torch.Tensor):
            return [frames_tensor[i] for i in range(frames_tensor.shape[0])]
        return frames_tensor
    else:
        print(f"[data] Decoding GT video from {args.video}")
        return decode_video(args.video)


# ── FP4 evaluation ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_fp4(
    model: nn.Module,
    ema: EMA,
    all_masks: torch.Tensor,
    gt_frames: list[torch.Tensor],
    pair_starts: list[int],
    posenet,
    segnet,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the EMA model after FP4 round-trip quantization.

    Returns: (scorer, avg_pose, avg_seg)
    """
    # Build a temporary model with FP4-quantized EMA weights
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load EMA weights
    ema.apply(model)

    # FP4 round-trip: quantize then dequantize
    fp4_packed = quantize_fp4(model.state_dict())
    fp4_state = dequantize_fp4(fp4_packed)
    model.load_state_dict(fp4_state)
    model.eval()

    use_autocast = device.type == "cuda" and torch.cuda.is_available()
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_autocast)

    total_p, total_s, count = 0.0, 0.0, 0
    with autocast_ctx:
        for start in pair_starts:
            mask_t, mask_t1 = mask_pair_from_index(all_masks, start)
            mask_t = mask_t.to(device)
            mask_t1 = mask_t1.to(device)

            gt_pair = pair_from_frames(gt_frames, start).to(device)

            rendered_pair = model(mask_t, mask_t1)  # (1, 2, H, W, 3)
            # uint8 round-trip to match official scorer pipeline
            rendered_pair = rendered_pair.round().clamp(0, 255).to(torch.uint8).float()

            score, pd, sd = eval_scorer_loss(rendered_pair, gt_pair, posenet, segnet)
            total_p += pd
            total_s += sd
            count += 1

            del mask_t, mask_t1, gt_pair, rendered_pair
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_p = total_p / max(count, 1)
    avg_s = total_s / max(count, 1)
    scorer = 100.0 * avg_s + math.sqrt(10.0 * avg_p)

    # Restore original weights
    model.load_state_dict(orig_state)
    model.train()
    return scorer, avg_p, avg_s


# ── Training loop ───────────────────────────────────────────────────────


def train(args: argparse.Namespace):
    device = torch.device(args.device) if args.device else detect_device()
    print(f"[train] Device: {device}")

    # Load scorer models
    posenet_path = _upstream / "models" / "posenet.safetensors"
    segnet_path = _upstream / "models" / "segnet.safetensors"
    print(f"[train] Loading scorers from {posenet_path.parent}")
    posenet, segnet = load_scorers(
        posenet_path, segnet_path, device=device, upstream_dir=str(_upstream),
    )

    # Load GT frames
    gt_frames = load_gt_frames(args)
    n_frames = len(gt_frames)
    print(f"[data] {n_frames} GT frames loaded, shape {gt_frames[0].shape}")

    # Extract masks once at startup
    print(f"[masks] Extracting {n_frames} masks via frozen SegNet (batch_size={args.mask_batch_size})...")
    t0 = time.monotonic()
    all_masks = extract_masks(gt_frames, segnet, device=device, batch_size=args.mask_batch_size)
    print(f"[masks] Extracted {all_masks.shape} masks in {time.monotonic() - t0:.1f}s")
    # Keep masks on CPU, move per-pair to device during training
    all_masks = all_masks.cpu()

    # Build model
    model = build_renderer(
        num_classes=5,
        embed_dim=args.embed_dim,
        base_ch=args.base_ch,
        mid_ch=args.mid_ch,
        motion_hidden=args.motion_hidden,
    )
    model = model.to(device)

    # Wrap with FP4 QAT if enabled
    qat_wrapper = None
    if args.use_qat:
        qat_wrapper = QATRendererFP4(model)
        print("[train] FP4 QAT enabled via forward hooks")

    # Training infrastructure
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-5,
    )

    # Pair indices
    all_pair_starts = pair_start_indices(n_frames)
    n_total = len(all_pair_starts)
    train_size = max(1, n_total // args.subsample)
    print(f"[train] {args.epochs} epochs, {train_size}/{n_total} pairs/epoch, "
          f"accum={args.accum_steps}, lr={args.lr}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # P0: Precompute GT scorer outputs (constant — frames and scorers are frozen)
    import torch.nn.functional as _F
    from tac.losses import _hwc_to_chw  # noqa: E402
    print("[train] P0: Precomputing GT scorer cache...")
    gt_scorer_cache = {}
    with torch.no_grad():
        for start in all_pair_starts:
            gt_pair = pair_from_frames(gt_frames, start).to(device)
            gx = _hwc_to_chw(gt_pair)
            gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)
            gt_scorer_cache[start] = {
                "pose_6": gp_out["pose"][..., :6].cpu(),
                "seg_soft": _F.softmax(gs_out, dim=1).cpu(),
            }
            del gt_pair, gx, gp_out, gs_out
    cache_bytes = sum(
        v["pose_6"].numel() * v["pose_6"].element_size()
        + v["seg_soft"].numel() * v["seg_soft"].element_size()
        for v in gt_scorer_cache.values()
    )
    print(f"[train] P0: Cached {len(gt_scorer_cache)} GT scorer outputs ({cache_bytes / 1e6:.1f}MB)")

    best_scorer = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()

        # Warmup LR
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Sample pairs for this epoch
        perm = torch.randperm(n_total)[:train_size]

        total_loss, total_pose, total_seg = 0.0, 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)
        accum = args.accum_steps

        for step, pair_idx in enumerate(perm):
            start = all_pair_starts[pair_idx.item()]

            # Load mask pair and GT pair
            mask_t, mask_t1 = mask_pair_from_index(all_masks, start)
            mask_t = mask_t.to(device)
            mask_t1 = mask_t1.to(device)
            gt_pair = pair_from_frames(gt_frames, start).to(device)

            # Forward: render pair from masks
            rendered_pair = model(mask_t, mask_t1)  # (1, 2, H, W, 3)

            # Scorer loss (P0: use cached GT scorer outputs when available)
            _cached_gt = gt_scorer_cache.get(start)
            if _cached_gt is not None:
                _gt_pose_6 = _cached_gt["pose_6"].to(device)
                _gt_seg_soft = _cached_gt["seg_soft"].to(device)
                loss, pd, sd = scorer_loss_cached(
                    rendered_pair, _gt_pose_6, _gt_seg_soft, posenet, segnet,
                )
                del _gt_pose_6, _gt_seg_soft
            else:
                loss, pd, sd = scorer_loss(rendered_pair, gt_pair, posenet, segnet)
            scaled_loss = loss / accum

            try:
                scaled_loss.backward()
            except torch.cuda.OutOfMemoryError:
                print(f"[train] CUDA OOM at step {step}, skipping")
                del mask_t, mask_t1, gt_pair, rendered_pair
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss += loss.item()
            total_pose += pd
            total_seg += sd

            # Gradient accumulation step
            if (step + 1) % accum == 0 or (step + 1) == len(perm):
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            del mask_t, mask_t1, gt_pair, rendered_pair

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LR scheduler
        if epoch >= args.warmup_epochs:
            scheduler.step()

        n_steps = len(perm)
        avg_loss = total_loss / max(n_steps, 1)
        avg_pose = total_pose / max(n_steps, 1)
        avg_seg = total_seg / max(n_steps, 1)
        lr = optimizer.param_groups[0]["lr"]

        # FP4 evaluation
        is_eval_epoch = ((epoch + 1) % args.eval_every == 0
                         or epoch == args.epochs - 1
                         or epoch == 0)
        if is_eval_epoch:
            scorer_val, eval_pose, eval_seg = evaluate_fp4(
                model, ema, all_masks, gt_frames,
                all_pair_starts, posenet, segnet, device,
            )
        else:
            scorer_val = best_scorer

        # Log
        marker = ""
        if scorer_val < best_scorer:
            best_scorer = scorer_val
            best_epoch = epoch
            marker = " *BEST*"

            # Save FP4 checkpoint from EMA weights (without disturbing training model)
            # Use a temporary model to avoid replacing optimizer's parameter references
            save_state = ema.state_dict()
            fp4_path = out_dir / f"renderer_{args.tag}_best_fp4.pt"
            fp4_packed = quantize_fp4(save_state)
            fp4_packed["__meta__"] = {
                "epoch": epoch,
                "scorer": best_scorer,
                "pose": eval_pose,
                "seg": eval_seg,
                "base_ch": args.base_ch,
                "mid_ch": args.mid_ch,
                "embed_dim": args.embed_dim,
                "motion_hidden": args.motion_hidden,
            }
            fp4_tmp = fp4_path.with_suffix(".pt.tmp")
            torch.save(fp4_packed, fp4_tmp)
            fp4_tmp.rename(fp4_path)
            fp4_size = fp4_path.stat().st_size
            param_count = sum(p.numel() for p in model.parameters())
            print(f"[fp4] Saved {param_count:,} params to {fp4_path} ({fp4_size:,} bytes)")

            # Also save EMA fp32 for resuming
            fp32_path = out_dir / f"renderer_{args.tag}_best_fp32.pt"
            fp32_tmp = fp32_path.with_suffix(".pt.tmp")
            torch.save(save_state, fp32_tmp)
            fp32_tmp.rename(fp32_path)

            # Save metadata
            meta_path = out_dir / f"renderer_{args.tag}_best_meta.json"
            meta_path.write_text(json.dumps({
                "epoch": epoch,
                "scorer": best_scorer,
                "pose": eval_pose,
                "seg": eval_seg,
                "fp4_path": str(fp4_path),
                "fp4_size": fp4_size,
                "args": {
                    "base_ch": args.base_ch,
                    "mid_ch": args.mid_ch,
                    "embed_dim": args.embed_dim,
                    "motion_hidden": args.motion_hidden,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "profile": args.profile,
                    "tag": args.tag,
                },
            }, indent=2))

        if is_eval_epoch:
            print(f"[ep {epoch:4d}/{args.epochs}] loss={avg_loss:.4f} "
                  f"pose={avg_pose:.6f} seg={avg_seg:.6f} "
                  f"fp4_scorer={scorer_val:.4f} best={best_scorer:.4f} "
                  f"lr={lr:.6f}{marker}")
        elif epoch % 10 == 0:
            print(f"[ep {epoch:4d}/{args.epochs}] loss={avg_loss:.4f} "
                  f"pose={avg_pose:.6f} seg={avg_seg:.6f} lr={lr:.6f}")

    print(f"\n[train] Complete. Best FP4 scorer: {best_scorer:.4f} at epoch {best_epoch}")
    print(f"[train] Saved to: {out_dir}/renderer_{args.tag}_best_fp4.pt")

    # Clean up QAT hooks
    if qat_wrapper is not None:
        qat_wrapper.remove_hooks()

    return best_scorer


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result is not None else 1)
