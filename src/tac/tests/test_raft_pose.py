from __future__ import annotations

import inspect

import torch

import tac.raft_pose as raft_pose
from tac.raft_pose import (
    build_pose_tensor_from_flow,
    calibrate_pose_dim0,
    flow_to_pose_dim0,
)


def test_flow_to_pose_dim0_shape() -> None:
    flow = torch.zeros(4, 6, 8, 2, dtype=torch.float32)
    flow[..., 0] = 9.1

    dim0 = flow_to_pose_dim0(flow, fx=910.0)

    assert dim0.shape == (4,)
    assert torch.allclose(dim0, torch.full((4,), 0.01))


def test_road_region_slice() -> None:
    flow = torch.zeros(2, 10, 4, 2, dtype=torch.float32)
    for row in range(flow.shape[1]):
        flow[:, row, :, 0] = float(row)

    dim0 = flow_to_pose_dim0(flow, fx=1.0, road_region=(0.5, 0.9))

    assert torch.allclose(dim0, torch.full((2,), 6.5))


def test_calibration_identity() -> None:
    raw = torch.tensor([-0.2, 0.0, 0.3, 0.9], dtype=torch.float32)
    baseline = raw.clone()

    calibrated, a, b = calibrate_pose_dim0(raw, baseline)

    assert torch.allclose(calibrated, baseline, atol=1e-6)
    assert abs(a - 1.0) < 1e-5
    assert abs(b) < 1e-5


def test_build_pose_tensor_shape() -> None:
    flow_dim0 = torch.linspace(-0.1, 0.1, 7)
    baseline = torch.arange(42, dtype=torch.float32).reshape(7, 6)

    poses = build_pose_tensor_from_flow(flow_dim0, baseline_poses=baseline)

    assert poses.shape == (7, 6)
    assert torch.allclose(poses[:, 0], flow_dim0)
    assert torch.allclose(poses[:, 1:], baseline[:, 1:])


def test_no_device_fallback_in_compute_raft_flow() -> None:
    source = inspect.getsource(raft_pose.compute_raft_flow)
    assert "mps" not in source.lower()
    assert "cpu" not in source.lower()


def test_compute_raft_flow_pushes_frame_limit_into_decoder() -> None:
    source = inspect.getsource(raft_pose.compute_raft_flow)
    assert "read_video" not in source
    assert "decode_video(path, max_frames=n_frames)" in source
