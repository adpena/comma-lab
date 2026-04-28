#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tac.raft_pose import (
    build_pose_tensor_from_flow,
    calibrate_pose_dim0,
    compute_raft_flow,
    flow_to_pose_dim0,
)


def _load_pose_tensor(path: Path, device: str) -> torch.Tensor:
    data = torch.load(str(path), map_location=device, weights_only=True)
    if isinstance(data, dict):
        for key in ("poses", "optimized_poses", "gt_poses"):
            value = data.get(key)
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32, device=device)
        raise ValueError(f"{path} dict has no poses, optimized_poses, or gt_poses key")
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive FiLM poses from RAFT flow")
    parser.add_argument("--video", required=True, help="Path to upstream/videos/0.mkv")
    parser.add_argument("--baseline-poses", required=True, help="Lane A optimized_poses.pt for calibration")
    parser.add_argument("--output", required=True, help="Output raft_poses.pt")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--n-frames", type=int, default=1200)
    args = parser.parse_args()

    baseline = _load_pose_tensor(Path(args.baseline_poses), args.device)
    if baseline.ndim != 2 or baseline.shape[1] != 6:
        raise ValueError(f"--baseline-poses must have shape (N, 6), got {tuple(baseline.shape)}")

    flow = compute_raft_flow(args.video, n_frames=int(args.n_frames), device=args.device)
    raw_dim0_all = flow_to_pose_dim0(flow)
    raw_dim0 = raw_dim0_all[::2][: baseline.shape[0]]
    if raw_dim0.numel() != baseline.shape[0]:
        raise ValueError(
            f"RAFT produced {raw_dim0.numel()} renderer-pair flow values, "
            f"but baseline has {baseline.shape[0]} poses"
        )

    calibrated_dim0, a, b = calibrate_pose_dim0(raw_dim0, baseline[:, 0])
    poses = build_pose_tensor_from_flow(calibrated_dim0, baseline_poses=baseline)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(poses.detach(), out_path)

    delta = poses - baseline[: poses.shape[0]]
    per_dim_rmse = torch.sqrt(torch.mean(delta.square(), dim=0))
    print(f"raft_pose: wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"raft_pose: calibration a={a:.8g} b={b:.8g}")
    print(f"raft_pose: raw_dim0 mean={raw_dim0.mean().item():.6f} std={raw_dim0.std(unbiased=False).item():.6f}")
    print(
        "raft_pose: calibrated_dim0 "
        f"mean={calibrated_dim0.mean().item():.6f} "
        f"std={calibrated_dim0.std(unbiased=False).item():.6f}"
    )
    print("raft_pose: per_dim_rmse=[" + ", ".join(f"{v:.6f}" for v in per_dim_rmse.tolist()) + "]")


if __name__ == "__main__":
    main()
