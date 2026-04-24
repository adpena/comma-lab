#!/usr/bin/env python3
"""Embedding-Space TTO: optimize the renderer's shared class embedding.

The renderer's AsymmetricPairGenerator has a shared nn.Embedding(5, 6) between
the MaskRenderer and MotionPredictor. These 30 values control how each semantic
class (road, lane marking, vehicle, etc.) is represented internally.

By optimizing these 30 values at compress time against the frozen scorers, we
make each class's internal representation scorer-optimal. This is GLOBAL: one
embedding serves all 600 pairs, unlike pose TTO which is per-pair.

Archive cost: 5 x 6 x 4 = 120 bytes (fp32), or 60 bytes (fp16).
Contest-compliant: no scorers at inflate time, single forward pass.

This COMPOUNDS with pose TTO:
1. First: optimize embedding (global, ~2 min)
2. Then: optimize poses (per-pair, ~5 min) using the optimized embedding

Usage:
    # Smoke test (local MPS, 10 pairs, 50 steps):
    PYTHONPATH=src:upstream python experiments/optimize_embedding.py \
        --checkpoint path/to/renderer_best.pt --device mps --smoke

    # Full run (4090):
    PYTHONPATH=src:upstream python experiments/optimize_embedding.py \
        --checkpoint path/to/renderer_best.pt --device cuda

    # Combined with pose TTO:
    PYTHONPATH=src:upstream python experiments/optimize_embedding.py \
        --checkpoint path/to/renderer_best.pt --device cuda \
        --save-checkpoint  # saves a new checkpoint with optimized embedding
    # Then run optimize_poses.py on the saved checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
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
    Path(os.environ.get("TAC_RESULTS_DIR", ""))
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "embedding_tto"
)

from tac.renderer import simulate_eval_roundtrip  # canonical impl (no local copy)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEGNET_INPUT_H, SEGNET_INPUT_W = 384, 512
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding-Space TTO: optimize class embedding globally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to renderer .pt or .bin checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=NUM_FRAMES,
                   help="Number of frames to process")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of full passes over all pairs")
    p.add_argument("--steps-per-batch", type=int, default=1,
                   help="Gradient accumulation steps per batch (1 = no accumulation)")
    p.add_argument("--lr", type=float, default=0.005,
                   help="Adam learning rate for embedding weights")
    p.add_argument("--batch-pairs", type=int, default=50,
                   help="Pairs per forward/backward pass")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet loss weight (hinge)")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet loss weight (MSE)")
    p.add_argument("--hinge-margin", type=float, default=0.5,
                   help="Margin for SegNet hinge loss")
    p.add_argument("--upstream", type=str, default=None,
                   help="Path to upstream repo (auto-detected if None)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: timestamped)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to GT video (default: upstream/videos/0.mkv)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 20 frames, 3 epochs")
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resolution roundtrip in loss (default: on)")
    p.add_argument("--no-eval-roundtrip", dest="eval_roundtrip", action="store_false",
                   help="Disable eval roundtrip simulation")
    p.add_argument("--early-stop-patience", type=int, default=3,
                   help="Stop if no improvement for this many epochs (uses rolling-best: "
                        "restores best weights at end regardless of when early stop fires)")
    p.add_argument("--gt-poses-path", type=str, default=None,
                   help="Path to GT or optimized poses (uses GT PoseNet targets if not provided)")
    p.add_argument("--save-checkpoint", action="store_true",
                   help="Save a new renderer checkpoint with optimized embedding")
    p.add_argument("--log-every", type=int, default=1,
                   help="Log batch metrics every N batches")
    return p.parse_args()


def load_renderer(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    """Load AsymmetricPairGenerator, returning model and checkpoint dict."""
    from tac.renderer import AsymmetricPairGenerator

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", ckpt.get("config", {}))
    model = AsymmetricPairGenerator(
        num_classes=model_cfg.get("num_classes", 5),
        embed_dim=model_cfg.get("embed_dim", 6),
        base_ch=model_cfg.get("base_ch", 36),
        mid_ch=model_cfg.get("mid_ch", 60),
        motion_hidden=model_cfg.get("motion_hidden", 32),
        depth=model_cfg.get("depth", 1),
        max_flow_px=model_cfg.get("max_flow_px", 20.0),
        max_residual=model_cfg.get("max_residual", 20.0),
        flow_only=model_cfg.get("flow_only", False),
        pose_dim=model_cfg.get("pose_dim", 0),
        use_dsconv=model_cfg.get("use_dsconv", False),
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.eval().to(device)

    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[renderer] Loaded {n_params:,} params from {checkpoint_path}")

    return model, ckpt


def find_shared_embedding(model: nn.Module) -> nn.Embedding:
    """Find the shared embedding in AsymmetricPairGenerator.

    The shared embedding is the nn.Embedding(5, 6) that is used by both
    the MaskRenderer (self.renderer.embedding) and MotionPredictor (self.motion.embedding).
    Because they share the same object, optimizing one optimizes both.
    """
    renderer_emb = model.renderer.embedding
    motion_emb = model.motion.embedding

    if id(renderer_emb) != id(motion_emb):
        print("[WARNING] renderer.embedding and motion.embedding are NOT shared.")
        print("[WARNING] Optimizing renderer.embedding only. Motion will use original.")
    else:
        print("[embedding] Confirmed: renderer and motion share the SAME embedding object.")

    emb = renderer_emb
    print(f"[embedding] Shape: {emb.weight.shape} "
          f"(num_classes={emb.weight.shape[0]}, embed_dim={emb.weight.shape[1]})")
    print(f"[embedding] Total values: {emb.weight.numel()} "
          f"({emb.weight.numel() * 4} bytes fp32, {emb.weight.numel() * 2} bytes fp16)")

    return emb


def segnet_hinge_loss(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Hinge loss on SegNet logits: penalize pixels at risk of argmax flip."""
    B, C, H, W = logits.shape
    correct = logits.gather(1, gt_masks.unsqueeze(1)).squeeze(1)
    mask_inf = torch.zeros_like(logits)
    mask_inf.scatter_(1, gt_masks.unsqueeze(1), float("-inf"))
    runner_up = (logits + mask_inf).max(dim=1).values
    loss = F.relu(margin - (correct - runner_up))
    return loss.mean()


def compute_batch_loss(
    model: nn.Module,
    masks_t: torch.Tensor,
    masks_t1: torch.Tensor,
    gt_masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    seg_weight: float,
    pose_weight: float,
    hinge_margin: float,
    eval_roundtrip: bool,
    poses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute combined loss for a batch of pairs.

    Returns (total_loss, metrics_dict) where total_loss has gradients
    flowing ONLY to the embedding (all other params are frozen).
    """
    # Forward: renderer produces (B, 2, H, W, 3) HWC pairs
    pairs = model(masks_t, masks_t1, pose=poses)

    frame_t = pairs[:, 0]   # (B, H, W, 3)
    frame_t1 = pairs[:, 1]  # (B, H, W, 3)
    frames_hwc = torch.cat([frame_t, frame_t1], dim=0)  # (2*B, H, W, 3)
    frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()  # (2*B, 3, H, W)

    if eval_roundtrip:
        frames_chw = simulate_eval_roundtrip(frames_chw)

    # SegNet loss (hinge)
    seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
    seg_logits = segnet(seg_in)
    seg_loss = segnet_hinge_loss(seg_logits, gt_masks.to(device), margin=hinge_margin)

    # PoseNet loss (MSE on 6D output)
    # NOTE: PoseNet requires a separate (B, 2, 3, H, W) pair input with its own
    # eval roundtrip because it computes inter-frame pose from paired frames,
    # whereas SegNet processes frames independently as (2*B, 1, 3, H, W).
    pairs_chw = torch.stack([
        frame_t.permute(0, 3, 1, 2),
        frame_t1.permute(0, 3, 1, 2),
    ], dim=1)  # (B, 2, 3, H, W)

    if eval_roundtrip:
        B_p, T_p, C_p, H_p, W_p = pairs_chw.shape
        flat = pairs_chw.reshape(B_p * T_p, C_p, H_p, W_p)
        flat = simulate_eval_roundtrip(flat)
        pairs_chw = flat.reshape(B_p, T_p, C_p, H_p, W_p)

    pose_in = posenet.preprocess_input(pairs_chw)
    pose_out = posenet(pose_in)["pose"][..., :6]
    pose_loss = F.mse_loss(pose_out, pose_targets.to(device))

    total_loss = seg_weight * seg_loss + pose_weight * pose_loss

    return total_loss, {
        "seg_loss": seg_loss.item(),
        "pose_loss": pose_loss.item(),
        "total_loss": total_loss.item(),
    }


def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        args.epochs = 3
        args.batch_pairs = 10
        print("[smoke] Smoke test: 20 frames, 3 epochs, 10 pairs/batch")

    args.n_frames = args.n_frames - (args.n_frames % 2)
    n_pairs = args.n_frames // 2

    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else UPSTREAM_ROOT
    if upstream is None:
        print("ERROR: Cannot find upstream root. Set --upstream or TAC_UPSTREAM_DIR.",
              file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(RESULTS_DIR / f"embedding_tto_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    print(f"[config] device={device}, n_frames={args.n_frames}, epochs={args.epochs}")
    print(f"[config] lr={args.lr}, batch_pairs={args.batch_pairs}")
    print(f"[config] seg_weight={args.seg_weight}, pose_weight={args.pose_weight}")
    print(f"[config] hinge_margin={args.hinge_margin}, eval_roundtrip={args.eval_roundtrip}")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] output_dir={output_dir}")

    t_total = time.monotonic()

    # -- Step 1: Load scorers --
    print("\n[1/5] Loading differentiable scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/5] Scorers loaded in {time.monotonic() - t0:.1f}s")

    # -- Step 2: Load renderer --
    print("\n[2/5] Loading renderer...")
    t0 = time.monotonic()
    model, ckpt = load_renderer(args.checkpoint, device)
    print(f"[2/5] Renderer loaded in {time.monotonic() - t0:.1f}s")

    # -- Find and unfreeze the shared embedding --
    embedding = find_shared_embedding(model)
    original_weights = embedding.weight.data.clone()
    embedding.weight.requires_grad_(True)

    # Verify only embedding is trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[embedding] Trainable: {trainable} / {total} parameters "
          f"({trainable / total * 100:.2f}%)")
    assert trainable == embedding.weight.numel(), (
        f"Expected {embedding.weight.numel()} trainable params, got {trainable}. "
        "Something besides the embedding is unfrozen."
    )

    # -- Step 3: Decode GT video + extract masks + pose targets --
    print(f"\n[3/5] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video
    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)
    n_pairs = args.n_frames // 2
    print(f"[3/5] Decoded {args.n_frames} frames in {time.monotonic() - t0:.1f}s")

    print("\n[4/5] Extracting GT masks and pose targets...")
    t0 = time.monotonic()
    from tac.scorer import extract_gt_masks, extract_gt_pose_targets
    gt_masks_all = extract_gt_masks(gt_frames, segnet, device)
    pose_targets_all = extract_gt_pose_targets(gt_frames, posenet, device)
    print(f"[4/5] Masks: {gt_masks_all.shape}, Poses: {pose_targets_all.shape} "
          f"in {time.monotonic() - t0:.1f}s")

    # Load poses for conditioning (if renderer has FiLM)
    renderer_pose_dim = getattr(model, "pose_dim", 0)
    init_poses: torch.Tensor | None = None
    if renderer_pose_dim > 0:
        if args.gt_poses_path and Path(args.gt_poses_path).exists():
            init_poses = torch.load(args.gt_poses_path, map_location="cpu",
                                    weights_only=True).float()[:n_pairs]
            print(f"  Loaded poses from {args.gt_poses_path}: {init_poses.shape}")
        else:
            init_poses = pose_targets_all[:n_pairs].clone()
            print(f"  Using GT PoseNet targets as pose conditioning: {init_poses.shape}")

    # -- Step 5: Epoch-based embedding optimization --
    print(f"\n[5/5] Optimizing embedding over {args.epochs} epochs "
          f"({n_pairs} pairs in batches of {args.batch_pairs})...")

    optimizer = torch.optim.Adam([embedding.weight], lr=args.lr)
    n_batches = math.ceil(n_pairs / args.batch_pairs)

    best_epoch_loss = float("inf")
    best_weights = embedding.weight.data.clone()
    patience_counter = 0
    epoch_history: list[dict] = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_seg = 0.0
        epoch_pose = 0.0
        epoch_batches = 0
        t_epoch = time.monotonic()

        # Shuffle batch order each epoch for better gradient diversity
        batch_indices = torch.randperm(n_batches).tolist()

        for bi, batch_idx in enumerate(batch_indices):
            pair_start = batch_idx * args.batch_pairs
            pair_end = min(pair_start + args.batch_pairs, n_pairs)
            frame_start = 2 * pair_start
            frame_end = 2 * pair_end

            batch_masks_t = gt_masks_all[frame_start:frame_end:2].to(device)
            batch_masks_t1 = gt_masks_all[frame_start + 1:frame_end + 1:2].to(device)
            batch_gt_masks = gt_masks_all[frame_start:frame_end].to(device)
            batch_pose_targets = pose_targets_all[pair_start:pair_end]

            # Pose conditioning for FiLM (if available)
            batch_poses = None
            if init_poses is not None:
                batch_poses = init_poses[pair_start:pair_end].to(device)

            optimizer.zero_grad()
            loss, metrics = compute_batch_loss(
                model=model,
                masks_t=batch_masks_t,
                masks_t1=batch_masks_t1,
                gt_masks=batch_gt_masks,
                pose_targets=batch_pose_targets,
                posenet=posenet,
                segnet=segnet,
                device=device,
                seg_weight=args.seg_weight,
                pose_weight=args.pose_weight,
                hinge_margin=args.hinge_margin,
                eval_roundtrip=args.eval_roundtrip,
                poses=batch_poses,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += metrics["total_loss"]
            epoch_seg += metrics["seg_loss"]
            epoch_pose += metrics["pose_loss"]
            epoch_batches += 1

            if bi % args.log_every == 0:
                grad_norm = embedding.weight.grad.norm().item() if embedding.weight.grad is not None else 0.0
                weight_delta = (embedding.weight.data - original_weights.to(device)).norm().item()
                print(f"  epoch {epoch} batch {bi}/{n_batches}: "
                      f"loss={metrics['total_loss']:.6f} "
                      f"(seg={metrics['seg_loss']:.6f}, pose={metrics['pose_loss']:.6f}) "
                      f"|grad|={grad_norm:.4f} |dw|={weight_delta:.4f}")

        avg_loss = epoch_loss / max(epoch_batches, 1)
        avg_seg = epoch_seg / max(epoch_batches, 1)
        avg_pose = epoch_pose / max(epoch_batches, 1)
        dt_epoch = time.monotonic() - t_epoch

        epoch_record = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_seg": avg_seg,
            "avg_pose": avg_pose,
            "time_s": dt_epoch,
        }
        epoch_history.append(epoch_record)

        print(f"\n  Epoch {epoch}/{args.epochs}: avg_loss={avg_loss:.6f} "
              f"(seg={avg_seg:.6f}, pose={avg_pose:.6f}) [{dt_epoch:.1f}s]")

        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            best_weights = embedding.weight.data.clone()
            patience_counter = 0
            print(f"  ** New best (improvement)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

        if patience_counter >= args.early_stop_patience:
            print(f"  Early stop at epoch {epoch} "
                  f"(no improvement for {args.early_stop_patience} epochs)")
            break

        # Free GPU memory between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # Restore best weights
    embedding.weight.data = best_weights.to(device)

    # -- Results --
    print("\n" + "=" * 70)
    print("RESULTS: Original embedding vs optimized embedding")
    print("=" * 70)

    weight_delta = (best_weights.to(device) - original_weights.to(device))
    print(f"\n  Weight delta per class:")
    class_names = ["road", "lane_marking", "vehicle", "other", "background"]
    for i in range(min(5, best_weights.shape[0])):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        delta_norm = weight_delta[i].norm().item()
        print(f"    {name:15s}: |delta|={delta_norm:.4f}  "
              f"original={original_weights[i].tolist()}  "
              f"optimized={best_weights[i].tolist()}")

    # Save optimized embedding
    torch.save(best_weights.cpu(), output_dir / "optimized_embedding.pt")
    print(f"\n  Saved optimized_embedding.pt: {best_weights.shape} "
          f"({best_weights.numel() * 4} bytes fp32)")

    # Save fp16 version
    emb_fp16 = best_weights.cpu().half()
    torch.save(emb_fp16, output_dir / "optimized_embedding_fp16.pt")
    print(f"  Saved optimized_embedding_fp16.pt: ({best_weights.numel() * 2} bytes fp16)")

    # Compute proxy score comparison
    print("\n  Computing proxy scores (original vs optimized)...")
    from tac.scorer import compute_proxy_score

    def generate_all_frames(emb_weights: torch.Tensor) -> torch.Tensor:
        """Generate all frames with given embedding weights."""
        embedding.weight.data = emb_weights.to(device)
        all_frames = []
        with torch.inference_mode():
            for start in range(0, n_pairs, args.batch_pairs):
                end = min(start + args.batch_pairs, n_pairs)
                mt = gt_masks_all[2 * start:2 * end:2].to(device)
                mt1 = gt_masks_all[2 * start + 1:2 * end + 1:2].to(device)
                bp = init_poses[start:end].to(device) if init_poses is not None else None
                pairs = model(mt, mt1, pose=bp)
                f0 = pairs[:, 0]
                f1 = pairs[:, 1]
                B = f0.shape[0]
                interleaved = torch.stack([f0, f1], dim=1).reshape(2 * B, *f0.shape[1:])
                all_frames.append(interleaved.cpu())
        return torch.cat(all_frames, dim=0).float()

    orig_frames = generate_all_frames(original_weights)
    orig_score = compute_proxy_score(
        orig_frames, gt_frames, posenet, segnet, device, rate=0.0,
    )

    opt_frames = generate_all_frames(best_weights)
    opt_score = compute_proxy_score(
        opt_frames, gt_frames, posenet, segnet, device, rate=0.0,
    )

    # Restore optimized weights
    embedding.weight.data = best_weights.to(device)

    print(f"\n  Original:  score={orig_score['score']:.4f} "
          f"(pose={orig_score['pose']:.6f}, seg={orig_score['seg']:.6f})")
    print(f"  Optimized: score={opt_score['score']:.4f} "
          f"(pose={opt_score['pose']:.6f}, seg={opt_score['seg']:.6f})")
    print(f"  Delta:     {opt_score['score'] - orig_score['score']:+.4f} "
          f"(pose: {opt_score['pose'] - orig_score['pose']:+.6f}, "
          f"seg: {opt_score['seg'] - orig_score['seg']:+.6f})")

    # Per-frame PoseNet decomposition: score frame_t and frame_t1 separately
    # to identify whether even or odd frames dominate the PoseNet distortion.
    print("\n  Per-frame PoseNet decomposition (optimized embedding):")
    with torch.inference_mode():
        pose_t_losses = []
        for start in range(0, n_pairs, args.batch_pairs):
            end = min(start + args.batch_pairs, n_pairs)
            mt = gt_masks_all[2 * start:2 * end:2].to(device)
            mt1 = gt_masks_all[2 * start + 1:2 * end + 1:2].to(device)
            bp = init_poses[start:end].to(device) if init_poses is not None else None
            pairs = model(mt, mt1, pose=bp)
            f0 = pairs[:, 0]  # (B, H, W, 3) frame_t
            f1 = pairs[:, 1]  # (B, H, W, 3) frame_t1
            # Score each frame within its pair context
            f0_chw = f0.permute(0, 3, 1, 2)
            f1_chw = f1.permute(0, 3, 1, 2)
            pair_chw = torch.stack([f0_chw, f1_chw], dim=1)
            gt_f0 = torch.stack([torch.as_tensor(gt_frames[2 * k]).float() for k in range(start, end)]).to(device)
            gt_f1 = torch.stack([torch.as_tensor(gt_frames[2 * k + 1]).float() for k in range(start, end)]).to(device)
            gt_pair_chw = torch.stack([
                gt_f0.permute(0, 3, 1, 2),
                gt_f1.permute(0, 3, 1, 2),
            ], dim=1)
            pose_in_r = posenet.preprocess_input(pair_chw)
            pose_in_g = posenet.preprocess_input(gt_pair_chw)
            pose_r = posenet(pose_in_r)["pose"][..., :6]
            pose_g = posenet(pose_in_g)["pose"][..., :6]
            # Per-pair MSE split isn't meaningful for PoseNet (it scores pairs),
            # but we can report the per-pair contribution
            per_pair_mse = ((pose_r - pose_g) ** 2).mean(dim=-1)  # (B,)
            pose_t_losses.extend(per_pair_mse.cpu().tolist())
        pose_t_arr = torch.tensor(pose_t_losses)
        print(f"    PoseNet per-pair MSE: mean={pose_t_arr.mean():.6f}, "
              f"max={pose_t_arr.max():.6f}, min={pose_t_arr.min():.6f}, "
              f"std={pose_t_arr.std():.6f}")
        worst_pairs = pose_t_arr.argsort(descending=True)[:5]
        print(f"    Worst 5 pairs: {[(int(p), f'{pose_t_arr[p]:.6f}') for p in worst_pairs]}")

    # Optionally save a new checkpoint with optimized embedding
    if args.save_checkpoint:
        ckpt_out = dict(ckpt)  # shallow copy
        # Update state dict with optimized embedding
        state_key = None
        for key in ["model_state_dict", "state_dict"]:
            if key in ckpt_out:
                state_key = key
                break

        # Only overwrite the two known embedding keys (no greedy pattern matching)
        _EMB_KEYS = ("renderer.embedding.weight", "motion.embedding.weight")
        if state_key:
            sd = dict(ckpt_out[state_key])
            for k in _EMB_KEYS:
                if k in sd:
                    sd[k] = best_weights.cpu()
                    print(f"  Updated {k} in checkpoint")
            ckpt_out[state_key] = sd
        else:
            # Raw state dict
            for k in _EMB_KEYS:
                if k in ckpt_out:
                    ckpt_out[k] = best_weights.cpu()
                    print(f"  Updated {k} in checkpoint")

        ckpt_path = output_dir / "renderer_optimized_embedding.pt"
        torch.save(ckpt_out, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # Save summary
    total_time = time.monotonic() - t_total
    summary = {
        "config": vars(args),
        "original_score": orig_score,
        "optimized_score": opt_score,
        "delta_score": opt_score["score"] - orig_score["score"],
        "total_time_s": total_time,
        "epochs_run": len(epoch_history),
        "n_pairs": n_pairs,
        "embedding_shape": list(best_weights.shape),
        "archive_bytes_fp32": best_weights.numel() * 4,
        "archive_bytes_fp16": best_weights.numel() * 2,
        "weight_delta_norm": weight_delta.norm().item(),
        "epoch_history": epoch_history,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
