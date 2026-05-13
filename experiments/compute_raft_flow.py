#!/usr/bin/env python3
"""Compute dense optical flow from GT video using RAFT-Small.

Produces per-pair flow fields that can be used as fixed warp in the
AsymmetricPairGenerator. Stored as compressed tensor file.

Usage:
    python experiments/compute_raft_flow.py \
        --gt-video workspace/upstream/comma_video_compression_challenge/videos/0.mkv \
        --output experiments/raft_flow.pt \
        --device mps

Output: torch file with {
    "flow": (N_pairs, 2, H, W) float16 tensor,
    "affine": (N_pairs, 6) float32 tensor (affine fit per pair),
    "n_pairs": int,
    "resolution": (H, W),
}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def decode_gt_frames(video_path: str, target_h: int = 384, target_w: int = 512) -> torch.Tensor:
    """Decode GT video to (N, 3, H, W) float32 [0, 255]."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        if t.shape[1] != target_h or t.shape[2] != target_w:
            t = F.interpolate(t.unsqueeze(0), size=(target_h, target_w),
                              mode="bilinear", align_corners=False).squeeze(0)
        frames.append(t)
    container.close()
    return torch.stack(frames)  # (N, 3, H, W)


def compute_raft_flow(frames: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Compute dense optical flow for consecutive even-pairs using RAFT-Small.

    Args:
        frames: (N, 3, H, W) float32 [0, 255]

    Returns:
        (N_pairs, 2, H, W) float32 flow in pixel coordinates
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights).to(device).eval()

    flows = []
    # Even pairs: (0,1), (2,3), (4,5)...
    pair_starts = list(range(0, len(frames) - 1, 2))

    with torch.inference_mode():
        for i, start in enumerate(pair_starts):
            frame_t = frames[start].unsqueeze(0).to(device)
            frame_t1 = frames[start + 1].unsqueeze(0).to(device)

            # RAFT expects [0, 255] uint8-range, transforms handle normalization
            img1, img2 = transforms(frame_t, frame_t1)
            flow_preds = model(img1, img2)
            flow = flow_preds[-1]  # last iteration, (1, 2, H, W) pixel coords

            flows.append(flow.cpu().squeeze(0))  # (2, H, W)

            if (i + 1) % 100 == 0 or i == len(pair_starts) - 1:
                print(f"  RAFT: {i + 1}/{len(pair_starts)} pairs", flush=True)

    return torch.stack(flows)  # (N_pairs, 2, H, W)


def fit_affine_from_flow(flow: torch.Tensor) -> torch.Tensor:
    """Fit per-pair 6-parameter affine from dense flow via least squares.

    Args:
        flow: (N_pairs, 2, H, W) pixel-coordinate flow

    Returns:
        (N_pairs, 6) affine parameters [a11, a12, tx, a21, a22, ty]
    """
    N, _, H, W = flow.shape
    # Create pixel coordinate grid
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                             torch.arange(W, dtype=torch.float32), indexing="ij")
    # Normalize to [-1, 1]
    xx_norm = xx / (W - 1) * 2 - 1
    yy_norm = yy / (H - 1) * 2 - 1

    # Stack [x, y, 1] for each pixel — (HW, 3)
    ones = torch.ones(H * W)
    A = torch.stack([xx_norm.reshape(-1), yy_norm.reshape(-1), ones], dim=1)  # (HW, 3)

    affines = []
    for i in range(N):
        # Flow in normalized coordinates
        fx = flow[i, 0].reshape(-1) / (W - 1) * 2  # pixel → normalized
        fy = flow[i, 1].reshape(-1) / (H - 1) * 2

        # Least squares: A @ [a11, a12, tx]^T = fx
        sol_x, _, _, _ = torch.linalg.lstsq(A, fx.unsqueeze(1))
        sol_y, _, _, _ = torch.linalg.lstsq(A, fy.unsqueeze(1))

        affines.append(torch.cat([sol_x.squeeze(), sol_y.squeeze()]))  # (6,)

    return torch.stack(affines)  # (N, 6)


def flow_to_grid_sample_coords(flow_px: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert pixel-coordinate flow DELTAS to grid_sample normalized DELTAS.

    These are displacement vectors, not absolute sampling coordinates.
    warp_with_flow adds them to the identity grid: sample_grid = grid + flow_delta.

    Args:
        flow_px: (N, 2, H, W) flow displacement in pixel coordinates
        H, W: spatial dimensions

    Returns:
        (N, 2, H, W) flow displacement in grid_sample [-1, 1] normalized space
    """
    flow_norm = flow_px.clone()
    flow_norm[:, 0] = flow_px[:, 0] / (W - 1) * 2  # x
    flow_norm[:, 1] = flow_px[:, 1] / (H - 1) * 2  # y
    return flow_norm


def main():
    parser = argparse.ArgumentParser(description="Compute RAFT optical flow from GT video")
    parser.add_argument("--gt-video", required=True, help="Path to GT video (0.mkv)")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--target-h", type=int, default=384, help="Target height")
    parser.add_argument("--target-w", type=int, default=512, help="Target width")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Decoding GT video: {args.gt_video}")
    frames = decode_gt_frames(args.gt_video, args.target_h, args.target_w)
    print(f"  {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[3]}")

    print(f"Computing RAFT flow on {args.device}...")
    flow_px = compute_raft_flow(frames, device=args.device)
    print(f"  Flow shape: {flow_px.shape}")
    print(f"  Flow magnitude: mean={flow_px.norm(dim=1).mean():.2f}, "
          f"max={flow_px.norm(dim=1).max():.2f} pixels")

    print("Fitting affine approximations...")
    affine = fit_affine_from_flow(flow_px)
    print(f"  Affine shape: {affine.shape}")

    print("Converting to grid_sample coordinates...")
    flow_norm = flow_to_grid_sample_coords(flow_px, args.target_h, args.target_w)

    output = {
        "flow": flow_norm.half(),  # (N_pairs, 2, H, W) float16 DELTAS in normalized coords
        "flow_px": flow_px.half(),  # (N_pairs, 2, H, W) float16 DELTAS in pixel coords
        "affine": affine,          # (N_pairs, 6) float32
        "n_pairs": flow_norm.shape[0],
        "resolution": (args.target_h, args.target_w),
    }

    torch.save(output, args.output)
    file_size = Path(args.output).stat().st_size
    print(f"\nSaved: {args.output} ({file_size / 1024:.1f} KB)")
    print(f"  Dense flow: {flow_norm.nelement() * 2 / 1024:.1f} KB (float16)")
    print(f"  Affine params: {affine.nelement() * 4 / 1024:.1f} KB (float32)")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
