#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Warp-Gen Deterministic Baseline: pure analytical, no neural network.

Hotz's idea: establish an analytical floor using optical flow warping.
For each consecutive frame pair:
  1. Compute RAFT optical flow from frame[2k] to frame[2k+1]
  2. Warp frame[2k] toward frame[2k+1] using the flow
  3. Use the warped result as the "generated" second frame
  4. Keep the first frame as-is (GT)

The key insight: the warp preserves PoseNet-critical structure (ego-motion
is captured by the flow itself), and the remaining artifacts only affect
pixels that moved significantly (which are typically non-PoseNet-sensitive).

This requires ZERO training and establishes the analytical floor.

Usage::

    PYTHONPATH=src:upstream python experiments/warp_gen_baseline.py \
        --device mps --smoke

    PYTHONPATH=src:upstream python experiments/warp_gen_baseline.py \
        --device cuda --n-frames 1200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


from tac.scorer import compute_proxy_score
from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warp-Gen Deterministic Baseline (no neural network)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames")
    p.add_argument("--warp-mode", type=str, default="bilinear",
                   choices=["bilinear", "nearest"],
                   help="Interpolation mode for warping")
    p.add_argument("--blend-alpha", type=float, default=0.0,
                   help="Blend factor with GT: result = (1-alpha)*warped + alpha*gt. "
                        "0.0 = pure warp, 1.0 = pure GT.")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate official scorer resolution round-trip. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    return p.parse_args()


def compute_raft_flow_batched(
    gt_frames: list[torch.Tensor],
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute RAFT flow for all non-overlapping pairs.

    Args:
        gt_frames: list of (H, W, 3) uint8/float tensors.
        device: computation device.
        batch_size: pairs per batch.

    Returns:
        (P, 2, H, W) float tensor of flow for each pair.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    N = len(gt_frames)
    P = N // 2

    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()

    H, W = gt_frames[0].shape[0], gt_frames[0].shape[1]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    all_flows = []

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        B = end - start

        f1_list, f2_list = [], []
        for k in range(start, end):
            f1_list.append(gt_frames[2 * k].float().permute(2, 0, 1) / 255.0)
            f2_list.append(gt_frames[2 * k + 1].float().permute(2, 0, 1) / 255.0)

        f1_batch = torch.stack(f1_list).to(device)  # (B, 3, H, W)
        f2_batch = torch.stack(f2_list).to(device)

        if pad_h > 0 or pad_w > 0:
            f1_batch = F.pad(f1_batch, (0, pad_w, 0, pad_h), mode="replicate")
            f2_batch = F.pad(f2_batch, (0, pad_w, 0, pad_h), mode="replicate")

        f1_t, f2_t = transforms(f1_batch, f2_batch)

        with torch.no_grad():
            flow_list = model(f1_t, f2_t)
        flow = flow_list[-1]  # (B, 2, H_pad, W_pad)
        flow = flow[:, :, :H, :W]  # Remove padding
        all_flows.append(flow.cpu())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return torch.cat(all_flows, dim=0)  # (P, 2, H, W)


def warp_frame(
    frame: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp a frame using optical flow via grid_sample.

    Args:
        frame: (H, W, 3) float tensor in [0, 255].
        flow: (2, H, W) float tensor of pixel-space flow.
        mode: interpolation mode.

    Returns:
        (H, W, 3) float tensor of warped frame in [0, 255].
    """
    H, W, C = frame.shape

    # Build sampling grid: each pixel (x, y) samples from (x + flow_x, y + flow_y)
    # grid_sample expects grid in [-1, 1] normalized coordinates
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    # Add flow (flow[0] = dx along W axis, flow[1] = dy along H axis)
    new_x = xx + flow[0]  # (H, W)
    new_y = yy + flow[1]  # (H, W)

    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * new_x / (W - 1) - 1.0
    grid_y = 2.0 * new_y / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # grid_sample expects (N, C, H, W) input
    frame_chw = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    warped = F.grid_sample(
        frame_chw, grid, mode=mode, padding_mode="border", align_corners=True,
    )
    warped = warped.squeeze(0).permute(1, 2, 0)  # (H, W, 3)

    return warped.clamp(0.0, 255.0)



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


def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        print("[smoke] Smoke test: 20 frames")

    args.n_frames = args.n_frames - (args.n_frames % 2)

    root = find_project_root()
    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else root / "upstream"

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(root / "experiments" / "results" / f"warp_gen_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir resolution so sidecar lands.
    _enforce_eval_roundtrip(args)

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    print(f"[config] device={device}, n_frames={args.n_frames}, "
          f"warp_mode={args.warp_mode}, blend_alpha={args.blend_alpha}")

    t_total_start = time.monotonic()

    # ── Step 1: Load scorers ─────────────────────────────────────────────
    print("\n[1/4] Loading scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/4] Scorers loaded + patched differentiable in {time.monotonic() - t0:.1f}s")

    # ── Step 2: Decode GT video ──────────────────────────────────────────
    print(f"\n[2/4] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video

    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)
    print(f"[2/4] Decoded {args.n_frames} frames in {time.monotonic() - t0:.1f}s")

    # ── Step 3: Compute RAFT flow + warp ─────────────────────────────────
    print(f"\n[3/4] Computing RAFT flow for {args.n_frames // 2} pairs...")
    t0 = time.monotonic()
    flows = compute_raft_flow_batched(gt_frames, device, batch_size=4)
    t_flow = time.monotonic() - t0
    print(f"[3/4] Flow computed in {t_flow:.1f}s ({t_flow / (args.n_frames // 2):.2f}s/pair)")
    print(f"[3/4] Flow stats: shape={flows.shape}, "
          f"mean_mag={flows.norm(dim=1).mean():.2f}, "
          f"max_mag={flows.norm(dim=1).max():.2f}")

    # Apply warping: for each pair (2k, 2k+1), warp frame 2k toward 2k+1
    # Result: first frame = GT, second frame = warped
    print("[3/4] Warping frames...")
    warped_frames = torch.zeros(args.n_frames, *gt_frames[0].shape, dtype=torch.float32)

    P = args.n_frames // 2
    for k in range(P):
        f0 = gt_frames[2 * k].float()
        f1 = gt_frames[2 * k + 1].float()
        flow_k = flows[k]

        # First frame of pair: keep as GT
        warped_frames[2 * k] = f0

        # Second frame: warp frame 0 using flow
        warped_f1 = warp_frame(f0, flow_k, mode=args.warp_mode)

        # Optional blend with GT
        if args.blend_alpha > 0:
            warped_f1 = (1 - args.blend_alpha) * warped_f1 + args.blend_alpha * f1

        warped_frames[2 * k + 1] = warped_f1.round().clamp(0, 255)

    print("[3/4] Warping complete")

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print("\n[4/4] Computing proxy scores...")

    # GT baseline
    gt_tensor = torch.stack(gt_frames).float()
    gt_score = compute_proxy_score(gt_tensor, gt_frames, posenet, segnet, device,
                                   eval_roundtrip=args.eval_roundtrip)
    print(f"[GT baseline] score={gt_score['score']:.6f} | "
          f"pose={gt_score['pose']:.6f} | seg={gt_score['seg']:.6f}")

    # Warped score
    warped_score = compute_proxy_score(warped_frames, gt_frames, posenet, segnet, device,
                                       eval_roundtrip=args.eval_roundtrip)

    t_total = time.monotonic() - t_total_start

    print(f"\n{'=' * 72}")
    print("WARP-GEN DETERMINISTIC BASELINE RESULTS")
    print(f"{'=' * 72}")
    print(f"  GT Baseline:    score={gt_score['score']:.6f}")
    print(f"  Warp-Gen:       score={warped_score['score']:.6f}")
    print(f"    pose = {warped_score['pose']:.6f}  ({warped_score['pose_contribution']:.6f})")
    print(f"    seg  = {warped_score['seg']:.6f}  ({warped_score['seg_contribution']:.6f})")
    print(f"  Warp mode: {args.warp_mode}, blend_alpha: {args.blend_alpha}")
    print(f"  Flow magnitude: mean={flows.norm(dim=1).mean():.2f}, max={flows.norm(dim=1).max():.2f}")
    print(f"  Total time: {t_total:.1f}s (flow: {t_flow:.1f}s)")
    print(f"{'=' * 72}")

    # ── Save ─────────────────────────────────────────────────────────────
    torch.save(warped_frames.to(torch.uint8), output_dir / "warped_frames.pt")
    torch.save(flows, output_dir / "raft_flows.pt")

    results = {
        "gt_baseline": gt_score,
        "warp_gen": warped_score,
        "config": {
            "n_frames": args.n_frames,
            "warp_mode": args.warp_mode,
            "blend_alpha": args.blend_alpha,
        },
        "flow_stats": {
            "mean_magnitude": float(flows.norm(dim=1).mean()),
            "max_magnitude": float(flows.norm(dim=1).max()),
        },
        "timing": {"total_s": round(t_total, 2), "flow_s": round(t_flow, 2)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results: {output_dir}")


if __name__ == "__main__":
    main()
