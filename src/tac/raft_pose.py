# SPDX-License-Identifier: MIT
"""RAFT-derived pose-dim0 estimation for compress-time lane experiments."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def _validate_cuda_device(device: str) -> torch.device:
    if device != "cuda":
        raise ValueError(f"Lane FL requires device='cuda', got {device!r}")
    return torch.device(device)


def _pad_to_multiple_of_8(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (h, w)


def compute_raft_flow(video_path: str, n_frames: int = 1200, device: str = "cuda") -> torch.Tensor:
    """Compute RAFT-Large optical flow for consecutive video frames.

    Args:
        video_path: Path to the contest MKV.
        n_frames: Maximum number of frames to decode.
        device: Must be ``"cuda"``. No fallback is attempted.

    Returns:
        Tensor with shape ``(T - 1, H, W, 2)`` on the requested device.
    """

    dev = _validate_cuda_device(device)
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    if n_frames < 2:
        raise ValueError(f"n_frames must be at least 2, got {n_frames}")

    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError as exc:
        raise ImportError(
            "Lane FL requires torchvision with optical-flow models. "
            "Install a torchvision build that includes raft_large."
        ) from exc

    from tac.data import decode_video

    frame_list = decode_video(path, max_frames=n_frames)
    if len(frame_list) < 2:
        raise ValueError(f"video must contain at least 2 frames, got {len(frame_list)}")
    frames = torch.stack(
        [frame.permute(2, 0, 1).contiguous() for frame in frame_list],
        dim=0,
    ).to(device=dev, dtype=torch.float32) / 255.0

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(dev).eval()

    flows: list[torch.Tensor] = []
    with torch.inference_mode():
        for idx in range(frames.shape[0] - 1):
            img1 = frames[idx : idx + 1]
            img2 = frames[idx + 1 : idx + 2]
            img1, img2 = transforms(img1, img2)
            img1, original_hw = _pad_to_multiple_of_8(img1)
            img2, _ = _pad_to_multiple_of_8(img2)
            flow_predictions = model(img1, img2)
            flow = flow_predictions[-1][0, :, : original_hw[0], : original_hw[1]]
            flows.append(flow.permute(1, 2, 0).contiguous())

    return torch.stack(flows, dim=0).to(dtype=torch.float32)


def flow_to_pose_dim0(
    flow: torch.Tensor,
    fx: float = 910.0,
    road_region: tuple[float, float] = (0.5, 0.95),
) -> torch.Tensor:
    """Convert road-region horizontal flow into uncalibrated pose dim 0."""

    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"flow must have shape (N, H, W, 2), got {tuple(flow.shape)}")
    if fx <= 0:
        raise ValueError(f"fx must be positive, got {fx}")
    start_frac, end_frac = road_region
    if not (0.0 <= start_frac < end_frac <= 1.0):
        raise ValueError(f"road_region must satisfy 0 <= start < end <= 1, got {road_region}")

    h = int(flow.shape[1])
    row_start = int(h * start_frac)
    row_end = int(h * end_frac)
    row_end = max(row_start + 1, min(row_end, h))
    road_flow_x = flow[:, row_start:row_end, :, 0]
    return road_flow_x.mean(dim=(1, 2)).to(dtype=torch.float32) / float(fx)


def calibrate_pose_dim0(raw_dim0: torch.Tensor, baseline_dim0: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Fit ``calibrated = a * raw + b`` by least squares."""

    raw = torch.as_tensor(raw_dim0, dtype=torch.float32).flatten()
    baseline = torch.as_tensor(baseline_dim0, dtype=torch.float32).flatten().to(raw.device)
    if raw.numel() != baseline.numel():
        raise ValueError(f"raw and baseline lengths differ: {raw.numel()} vs {baseline.numel()}")
    if raw.numel() < 2:
        raise ValueError("at least two samples are required for calibration")

    raw_mean = raw.mean()
    baseline_mean = baseline.mean()
    centered_raw = raw - raw_mean
    centered_baseline = baseline - baseline_mean
    denom = centered_raw.square().sum()
    if denom.abs().item() < 1e-12:
        a_tensor = torch.zeros((), dtype=torch.float32, device=raw.device)
    else:
        a_tensor = (centered_raw * centered_baseline).sum() / denom
    b_tensor = baseline_mean - a_tensor * raw_mean
    calibrated = a_tensor * raw + b_tensor
    return calibrated.to(dtype=torch.float32), float(a_tensor.item()), float(b_tensor.item())


def build_pose_tensor_from_flow(
    flow_dim0: torch.Tensor,
    baseline_poses: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a full ``(N, 6)`` pose tensor from calibrated flow dim 0."""

    dim0 = torch.as_tensor(flow_dim0, dtype=torch.float32).flatten()
    poses = torch.zeros(dim0.numel(), 6, dtype=torch.float32, device=dim0.device)
    poses[:, 0] = dim0
    if baseline_poses is not None:
        baseline = torch.as_tensor(baseline_poses, dtype=torch.float32, device=dim0.device)
        if baseline.ndim != 2 or baseline.shape[1] != 6:
            raise ValueError(f"baseline_poses must have shape (N, 6), got {tuple(baseline.shape)}")
        if baseline.shape[0] < dim0.numel():
            raise ValueError(
                f"baseline_poses has {baseline.shape[0]} rows, need at least {dim0.numel()}"
            )
        poses[:, 1:] = baseline[: dim0.numel(), 1:]
    return poses
