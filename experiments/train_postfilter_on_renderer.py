#!/usr/bin/env python3
"""Stacked Pipeline: Renderer → Postfilter (residual correction CNN).

After the renderer produces frames, a tiny residual CNN corrects remaining
scorer-relevant errors. This is the ORIGINAL approach (45KB dilated postfilter)
but applied to RENDERER output instead of codec output.

Key differences from the original postfilter:
    - Input is renderer output (smooth, semantic-aware) not codec output (blocky, noisy)
    - Errors are in scorer-space (PoseNet/SegNet disagreement with GT), not perceptual
    - The postfilter learns ONLY the residual that reduces scorer distortion
    - Much cleaner signal — renderer output is already in the right ballpark

Architecture (same as proven dilated postfilter):
    Conv(3, 64, 3) → ReLU → Conv(64, 64, 3, dilation=2) → ReLU → Conv(64, 3, 3) + skip
    ~45K params → ~45KB int8 archive

Pipeline at inflate time:
    masks → renderer → postfilter → upscale → write frames
    Total archive: renderer.bin (~195KB FP4) + postfilter.bin (~45KB int8) = ~240KB

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/train_postfilter_on_renderer.py \
        --device mps --smoke

    # Full run (Vast.ai 4090):
    PYTHONPATH=src:upstream python experiments/train_postfilter_on_renderer.py \
        --device cuda --epochs 3000 \
        --checkpoint experiments/results/v5_lagrangian_renderer/renderer_best.pt

    # With eval roundtrip (recommended for auth-faithful proxy):
    PYTHONPATH=src:upstream python experiments/train_postfilter_on_renderer.py \
        --device cuda --epochs 3000 --simulate-resize --segnet-loss-mode hinge
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
    else Path(__file__).resolve().parent / "results" / "postfilter_renderer"
)


# ---------------------------------------------------------------------------
# Postfilter architecture
# ---------------------------------------------------------------------------
class DilatedPostfilter(nn.Module):
    """Residual dilated CNN for scorer-space error correction.

    Architecture: skip + 3-layer CNN with dilation for wider receptive field.
    The network learns ONLY the residual (correction) added to the input.

    This is the same architecture that scored auth=1.33 on codec output.
    Applied to renderer output, it should push well below that.

    Args:
        hidden_ch: hidden channels (64 proven optimal)
        dilation: dilation factor for middle conv (2 = 5x5 effective RF)
    """

    def __init__(self, hidden_ch: int = 64, dilation: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=dilation, dilation=dilation, bias=True)
        self.conv3 = nn.Conv2d(hidden_ch, 3, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        # Zero-init output conv: postfilter starts as identity (skip only)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual correction.

        Args:
            x: (B, 3, H, W) float tensor in [0, 255] (renderer output)

        Returns:
            (B, 3, H, W) corrected output, clamped to [0, 255]
        """
        residual = self.conv3(self.act(self.conv2(self.act(self.conv1(x)))))
        return (x + residual).clamp(0.0, 255.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train postfilter on renderer output against scorer feedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to renderer checkpoint (frozen during postfilter training)")
    p.add_argument("--epochs", type=int, default=3000, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=4, help="Pairs per batch")
    p.add_argument("--hidden-ch", type=int, default=64, help="Postfilter hidden channels")
    p.add_argument("--dilation", type=int, default=2, help="Middle conv dilation")
    p.add_argument("--seg-weight", type=float, default=100.0, help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0, help="PoseNet loss weight")
    p.add_argument("--simulate-resize", action="store_true",
                   help="Apply eval roundtrip (384→874→384) in scoring")
    p.add_argument("--segnet-loss-mode", type=str, default="hinge",
                   choices=["xent", "hinge"], help="SegNet loss function")
    p.add_argument("--hinge-margin", type=float, default=0.5, help="Hinge loss margin")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--save-every", type=int, default=500, help="Save every N epochs")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, 200 epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.epochs = 200
        args.save_every = 100
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        root / "experiments" / "results" / "v5_lagrangian_renderer" / "renderer_best.pt"
    )

    # Verify checkpoint
    from tac.checkpoint import verify_checkpoint_identity
    verify_checkpoint_identity(str(checkpoint))

    # Load frozen renderer
    from tac.renderer import MaskRenderer
    print(f"[postfilter] Loading frozen renderer from {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        renderer_state = state["model_state_dict"]
        config = state.get("config", {})
    else:
        renderer_state = state
        config = {}

    renderer = MaskRenderer(
        embed_dim=config.get("embed_dim", 6),
        base_ch=config.get("base_ch", 36),
        mid_ch=config.get("mid_ch", 60),
        depth=config.get("depth", 1),
        pose_dim=config.get("pose_dim", 0),
    )
    renderer.load_state_dict(renderer_state, strict=False)
    renderer = renderer.to(device).eval()
    for p in renderer.parameters():
        p.requires_grad = False

    # Create postfilter
    postfilter = DilatedPostfilter(hidden_ch=args.hidden_ch, dilation=args.dilation).to(device)
    print(f"[postfilter] Architecture: {postfilter.param_count()} params "
          f"({postfilter.param_count() * 1 / 1024:.1f}KB int8)")

    # Load scorers
    from tac.scorer import load_differentiable_scorers, extract_gt_masks
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode video and extract masks
    from tac.data import decode_video
    print(f"[postfilter] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:n_frames]
    print(f"[postfilter] Extracting SegNet masks for {len(gt_frames)} frames...")
    gt_masks = extract_gt_masks(gt_frames, segnet, device=device)

    # Pre-compute renderer outputs (frozen, constant)
    n_pairs = len(gt_frames) // 2
    print(f"[postfilter] Pre-computing renderer outputs for {n_pairs} pairs...")
    renderer_outputs = []  # (2, 3, H, W) per pair
    for i in range(n_pairs):
        m0 = gt_masks[i * 2].to(device)
        m1 = gt_masks[i * 2 + 1].to(device)
        masks = torch.stack([m0, m1], dim=0)
        with torch.no_grad():
            rgb = renderer(masks)  # (2, 3, H, W)
        renderer_outputs.append(rgb.cpu())

    # Pre-cache GT scorer outputs
    from tac.losses import scorer_forward_pair
    print(f"[postfilter] Pre-caching GT scorer outputs...")
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

    # Training
    optimizer = torch.optim.Adam(postfilter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"[postfilter] Training: {args.epochs} epochs, batch_size={args.batch_size}")
    best_loss = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        perm = torch.randperm(n_pairs)
        epoch_loss = 0.0
        epoch_pose = 0.0
        epoch_seg = 0.0

        for batch_start in range(0, n_pairs, args.batch_size):
            batch_indices = perm[batch_start:batch_start + args.batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for idx in batch_indices:
                idx_val = idx.item()
                # Load pre-computed renderer output
                renderer_rgb = renderer_outputs[idx_val].to(device)  # (2, 3, H, W)

                # Apply postfilter
                corrected = postfilter(renderer_rgb)  # (2, 3, H, W)

                # Optional eval roundtrip
                if args.simulate_resize:
                    _, _, H, W = corrected.shape
                    small = F.interpolate(corrected, size=(384, 512), mode="bilinear", align_corners=False)
                    corrected = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)

                # Score
                pair_chw = corrected.unsqueeze(0)  # (1, 2, C, H, W)
                fp_out, fs_out = scorer_forward_pair(pair_chw, posenet, segnet)

                gt_pose_6 = gt_pose_cache[idx_val].to(device)
                gt_seg_soft = gt_seg_cache[idx_val].to(device)

                pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()

                if args.segnet_loss_mode == "hinge":
                    gt_class = gt_seg_soft.argmax(dim=1)
                    gt_logit = fs_out.gather(1, gt_class.unsqueeze(1)).squeeze(1)
                    mask = torch.ones_like(fs_out, dtype=torch.bool)
                    mask.scatter_(1, gt_class.unsqueeze(1), False)
                    other_logits = fs_out.masked_fill(~mask, float("-inf"))
                    max_other = other_logits.max(dim=1).values
                    seg_loss = F.relu(max_other - gt_logit + args.hinge_margin).mean()
                else:
                    pred_soft = F.softmax(fs_out, dim=1)
                    seg_loss = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1).mean()

                loss = args.seg_weight * seg_loss + args.pose_weight * torch.sqrt(pose_dist + 1e-8)
                batch_loss = batch_loss + loss

                epoch_pose += pose_dist.item()
                epoch_seg += seg_loss.item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(postfilter.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()

        scheduler.step()

        avg_loss = epoch_loss / n_pairs
        avg_pose = epoch_pose / n_pairs
        avg_seg = epoch_seg / n_pairs
        history.append({"epoch": epoch, "loss": avg_loss, "pose": avg_pose, "seg": avg_seg})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(postfilter.state_dict(), output_dir / "postfilter_best.pt")

        if epoch % 200 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(
                f"[postfilter] Epoch {epoch}/{args.epochs} | "
                f"loss={avg_loss:.4f} pose={avg_pose:.6f} seg={avg_seg:.4f} | "
                f"best={best_loss:.4f} | {elapsed/60:.1f}min"
            )

        if epoch > 0 and epoch % args.save_every == 0:
            torch.save(postfilter.state_dict(), output_dir / f"postfilter_ep{epoch}.pt")

    # Final save
    total_time = time.time() - t0
    torch.save(postfilter.state_dict(), output_dir / "postfilter_final.pt")

    # Quantize to int8 for archive
    postfilter_int8 = {}
    for k, v in postfilter.state_dict().items():
        if "weight" in k:
            scale = v.abs().max() / 127.0
            postfilter_int8[k] = (v / scale).round().clamp(-127, 127).to(torch.int8)
            postfilter_int8[k + "_scale"] = scale
        else:
            postfilter_int8[k] = v
    torch.save(postfilter_int8, output_dir / "postfilter_int8.pt")
    int8_size = sum(v.numel() * v.element_size() for v in postfilter_int8.values())

    results = {
        "total_time_s": total_time,
        "best_loss": best_loss,
        "n_params": postfilter.param_count(),
        "archive_bytes_int8": int8_size,
        "epochs": args.epochs,
        "final_pose": avg_pose,
        "final_seg": avg_seg,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\n{'='*60}")
    print(f"[postfilter] COMPLETE in {total_time/60:.1f} min")
    print(f"[postfilter] Best loss: {best_loss:.4f}")
    print(f"[postfilter] Archive: {int8_size/1024:.1f}KB int8")
    print(f"[postfilter] Final PoseNet: {avg_pose:.6f}, SegNet: {avg_seg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
