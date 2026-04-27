#!/usr/bin/env python3
"""LoRA TTO: Low-Rank Adaptation at compress time for per-video specialization.

Instead of full TTO (optimizing all pixels) or full model TTO (287K params),
LoRA adapts only low-rank weight deltas (~1-4K params) on the renderer's
conv layers. This gives model-level adaptation with minimal archive cost.

Pipeline:
    1. Load frozen base renderer (distilled checkpoint)
    2. Inject LoRA adapters (rank=4, ~1100 params)
    3. Optimize only LoRA weights against scorer feedback (like Phase 2 distillation)
    4. Extract LoRA delta → fp16 → archive (2.2KB)

At inflate time:
    1. Load base renderer + LoRA delta from archive
    2. Single forward pass with adapted model
    3. No gradient computation needed

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/train_lora_tto.py \
        --device mps --smoke

    # Full run (Vast.ai 4090):
    PYTHONPATH=src:upstream python experiments/train_lora_tto.py \
        --device cuda --epochs 2000 \
        --checkpoint experiments/results/v5_lagrangian_renderer/renderer_best.pt

    # Higher rank for more capacity:
    PYTHONPATH=src:upstream python experiments/train_lora_tto.py \
        --device cuda --rank 8 --epochs 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path("/kaggle/working/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

RESULTS_DIR = (
    Path(os.environ["TAC_RESULTS_DIR"])
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "lora_tto"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA TTO: low-rank adaptation at compress time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to base renderer checkpoint")
    p.add_argument("--rank", type=int, default=4, help="LoRA rank (4=compact, 8=capacity)")
    p.add_argument("--epochs", type=int, default=2000, help="Training epochs over all pairs")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate for LoRA params")
    p.add_argument("--batch-size", type=int, default=8, help="Pairs per training batch")
    p.add_argument("--seg-weight", type=float, default=100.0, help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0, help="PoseNet loss weight")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resize chain in scorer loss. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--segnet-loss-mode", type=str, default="hinge",
                   choices=["xent", "hinge"], help="SegNet loss function")
    p.add_argument("--hinge-margin", type=float, default=0.5, help="Hinge loss margin")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N epochs")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, 100 epochs")
    return p.parse_args()


def hinge_seg_loss(
    pred_logits: torch.Tensor,
    gt_soft: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Logit-margin hinge loss for SegNet agreement.

    Only penalizes pixels where the predicted argmax might flip relative to GT.
    This focuses gradient budget on boundary pixels that are at risk.

    Args:
        pred_logits: (B, C, H, W) raw SegNet logits for predicted frames
        gt_soft: (B, C, H, W) softmax of GT SegNet logits
        margin: margin in logit space (higher = more conservative)

    Returns:
        Scalar hinge loss
    """
    gt_class = gt_soft.argmax(dim=1)  # (B, H, W)
    B, C, H, W = pred_logits.shape

    # Gather logit of GT class: (B, H, W)
    gt_logit = pred_logits.gather(1, gt_class.unsqueeze(1)).squeeze(1)

    # Max logit of non-GT classes
    mask = torch.ones_like(pred_logits, dtype=torch.bool)
    mask.scatter_(1, gt_class.unsqueeze(1), False)
    other_logits = pred_logits.masked_fill(~mask, float("-inf"))
    max_other = other_logits.max(dim=1).values  # (B, H, W)

    # Hinge: penalize when max_other + margin > gt_logit
    violation = F.relu(max_other - gt_logit + margin)
    return violation.mean()


def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner.

    2026-04-27 codex R5-4 #4: delegated to the centralised
    `tac.eval_roundtrip_gate.enforce_eval_roundtrip` helper. The previous
    per-script copies were sticky — they only printed the warning when
    `args.eval_roundtrip` was already False, so a leftover env var in a
    shell / tmux session silently relaxed later runs without acknowledgement.
    The centralised helper warns whenever the env var is present and
    records it in run provenance.
    """
    from tac.eval_roundtrip_gate import enforce_eval_roundtrip
    output_dir = getattr(args, "output_dir", None)
    enforce_eval_roundtrip(args, output_dir=output_dir, write_provenance=output_dir is not None)


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.epochs = 100
        args.save_every = 50
        n_frames = 20
    else:
        n_frames = 1200

    device = torch.device(args.device)

    # Resolve paths
    from tac.utils import find_project_root
    root = find_project_root()
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir resolution so sidecar lands.
    if args.output_dir is None:
        args.output_dir = str(output_dir)
    _enforce_eval_roundtrip(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        root / "experiments" / "results" / "v5_lagrangian_renderer" / "renderer_best.pt"
    )

    # Verify checkpoint identity
    from tac.checkpoint import verify_checkpoint_identity
    verify_checkpoint_identity(str(checkpoint))

    # Load renderer
    from tac.renderer import MaskRenderer
    print(f"[lora-tto] Loading renderer from {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        renderer_state = state["model_state_dict"]
        config = state.get("config", {})
    else:
        renderer_state = state
        config = {}

    model = MaskRenderer(
        embed_dim=config.get("embed_dim", 6),
        base_ch=config.get("base_ch", 36),
        mid_ch=config.get("mid_ch", 60),
        depth=config.get("depth", 1),
        pose_dim=config.get("pose_dim", 0),
    )
    model.load_state_dict(renderer_state, strict=False)
    model = model.to(device)

    # Apply LoRA
    from tac.lora import apply_lora, extract_lora_state, lora_archive_size_bytes
    n_lora_params = apply_lora(model, rank=args.rank, scale=1.0)
    archive_bytes = lora_archive_size_bytes(model, dtype=torch.float16)
    print(f"[lora-tto] LoRA injected: {n_lora_params} trainable params, {archive_bytes} bytes fp16")

    # Load scorers
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode video and extract masks
    from tac.data import decode_video
    from tac.scorer import extract_gt_masks
    print(f"[lora-tto] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:n_frames]
    print(f"[lora-tto] Extracting SegNet masks for {len(gt_frames)} frames...")
    gt_masks = extract_gt_masks(gt_frames, segnet, device=device)

    # Pre-cache GT scorer outputs
    from tac.losses import scorer_forward_pair
    n_pairs = len(gt_frames) // 2
    print(f"[lora-tto] Pre-caching GT scorer outputs for {n_pairs} pairs...")
    gt_pose_cache = []
    gt_seg_cache = []
    for i in range(n_pairs):
        f0 = gt_frames[i * 2].float().to(device)
        f1 = gt_frames[i * 2 + 1].float().to(device)
        pair_chw = torch.stack([f0, f1], dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()
        with torch.no_grad():
            gp_out, gs_out = scorer_forward_pair(pair_chw, posenet, segnet)
            gt_pose_cache.append(gp_out["pose"][..., :6].cpu())
            gt_seg_cache.append(F.softmax(gs_out, dim=1).cpu())

    # Training loop
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Build pair dataset: masks + indices
    pair_masks = []  # list of (2, H, W) long tensors
    for i in range(n_pairs):
        m0 = gt_masks[i * 2]
        m1 = gt_masks[i * 2 + 1]
        pair_masks.append(torch.stack([m0, m1], dim=0))

    print(f"[lora-tto] Training: {args.epochs} epochs, batch_size={args.batch_size}")
    best_loss = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        # Shuffle pairs
        perm = torch.randperm(n_pairs)
        epoch_loss = 0.0
        epoch_pose = 0.0
        epoch_seg = 0.0
        n_batches = 0

        for batch_start in range(0, n_pairs, args.batch_size):
            batch_indices = perm[batch_start:batch_start + args.batch_size]
            batch_loss = torch.tensor(0.0, device=device)

            for idx in batch_indices:
                idx_val = idx.item()
                masks = pair_masks[idx_val].to(device)  # (2, H, W)

                # Forward through LoRA-adapted renderer
                rgb = model(masks)  # (2, 3, H, W)

                # eval_roundtrip: simulate contest eval resize chain (384→874→uint8→384)
                if args.eval_roundtrip:
                    from tac.renderer import simulate_eval_roundtrip
                    from tac.camera import CAMERA_H, CAMERA_W
                    rgb = simulate_eval_roundtrip(
                        rgb, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.5,
                    )

                # Score
                pair_chw = rgb.unsqueeze(0)  # (1, 2, C, H, W)
                fp_out, fs_out = scorer_forward_pair(pair_chw, posenet, segnet)

                # Skip GT cache when eval_roundtrip is on — cached values were
                # computed without roundtrip. Force recomputation.
                if args.eval_roundtrip:
                    f0 = gt_frames[idx_val * 2].float().to(device)
                    f1 = gt_frames[idx_val * 2 + 1].float().to(device)
                    gt_chw = torch.stack([f0, f1], dim=0).permute(0, 3, 1, 2).contiguous()
                    gt_chw = simulate_eval_roundtrip(
                        gt_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0,
                    )
                    gt_pair_for_scorer = gt_chw.unsqueeze(0)
                    with torch.no_grad():
                        gp_gt, gs_gt = scorer_forward_pair(gt_pair_for_scorer, posenet, segnet)
                    gt_pose_6 = gp_gt["pose"][..., :6]
                    gt_seg_soft = F.softmax(gs_gt, dim=1)
                else:
                    gt_pose_6 = gt_pose_cache[idx_val].to(device)
                    gt_seg_soft = gt_seg_cache[idx_val].to(device)

                pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()

                if args.segnet_loss_mode == "hinge":
                    seg_loss = hinge_seg_loss(fs_out, gt_seg_soft, margin=args.hinge_margin)
                else:
                    pred_soft = F.softmax(fs_out, dim=1)
                    seg_loss = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1).mean()

                loss = args.seg_weight * seg_loss + args.pose_weight * torch.sqrt(pose_dist + 1e-8)
                batch_loss = batch_loss + loss

                epoch_pose += pose_dist.item()
                epoch_seg += seg_loss.item()

            # Backward + step
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pose = epoch_pose / n_pairs
        avg_seg = epoch_seg / n_pairs
        history.append({"epoch": epoch, "loss": avg_loss, "pose": avg_pose, "seg": avg_seg})

        if avg_loss < best_loss:
            best_loss = avg_loss
            lora_state = extract_lora_state(model)
            torch.save(lora_state, output_dir / "lora_best.pt")

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(
                f"[lora-tto] Epoch {epoch}/{args.epochs} | "
                f"loss={avg_loss:.4f} pose={avg_pose:.6f} seg={avg_seg:.4f} | "
                f"best={best_loss:.4f} | {elapsed/60:.1f}min"
            )

        if epoch > 0 and epoch % args.save_every == 0:
            lora_state = extract_lora_state(model)
            torch.save(lora_state, output_dir / f"lora_ep{epoch}.pt")

    # Final save
    total_time = time.time() - t0
    lora_state = extract_lora_state(model)
    torch.save(lora_state, output_dir / "lora_final.pt")

    results = {
        "total_time_s": total_time,
        "best_loss": best_loss,
        "n_lora_params": n_lora_params,
        "archive_bytes_fp16": archive_bytes,
        "rank": args.rank,
        "epochs": args.epochs,
        "final_pose": avg_pose,
        "final_seg": avg_seg,
        "history_len": len(history),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save full history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\n{'='*60}")
    print(f"[lora-tto] COMPLETE in {total_time/60:.1f} min")
    print(f"[lora-tto] Best loss: {best_loss:.4f}")
    print(f"[lora-tto] LoRA archive: {archive_bytes} bytes ({archive_bytes/1024:.1f}KB)")
    print(f"[lora-tto] Final PoseNet: {avg_pose:.6f}, SegNet: {avg_seg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
