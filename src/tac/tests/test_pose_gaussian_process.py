from __future__ import annotations

import torch

from tac.pose_gaussian_process import (
    PoseGPModel,
    fit_pose_gp,
    load_pose_gp,
    reconstruct_poses,
    save_pose_gp,
)


def _smooth_pose_trajectory(n_pairs: int) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, n_pairs, dtype=torch.float32)
    poses = torch.zeros(n_pairs, 6, dtype=torch.float32)
    poses[:, 0] = 0.15 + 0.4 * t - 0.25 * t.square() + 0.08 * t.pow(4)
    poses[:, 1] = torch.sin(t * 3.0)
    poses[:, 2] = torch.cos(t * 2.0)
    poses[:, 3] = t - t.mean()
    poses[:, 4] = 0.2
    poses[:, 5] = -0.1 * t
    return poses


def test_fit_roundtrip() -> None:
    baseline = _smooth_pose_trajectory(96)

    model = fit_pose_gp(baseline)
    reconstructed = reconstruct_poses(model, baseline.shape[0])

    assert reconstructed.shape == baseline.shape
    rmse = torch.sqrt(torch.mean((reconstructed[:, 0] - baseline[:, 0]).square()))
    assert rmse.item() < 0.05


def test_save_load(tmp_path) -> None:
    model = PoseGPModel(
        poly_coeffs=torch.linspace(-0.5, 0.5, 11, dtype=torch.float32),
        sigma=torch.linspace(0.01, 0.05, 5, dtype=torch.float32),
    )
    path = tmp_path / "pose_gp.bin"

    save_pose_gp(model, path)
    loaded = load_pose_gp(path)

    expected_coeffs = model.poly_coeffs.to(torch.float16).to(torch.float32)
    expected_sigma = model.sigma.to(torch.float16).to(torch.float32)
    assert torch.allclose(loaded.poly_coeffs, expected_coeffs, atol=1e-4, rtol=1e-3)
    assert torch.allclose(loaded.sigma, expected_sigma, atol=1e-4, rtol=1e-3)


def test_reconstruct_dims_1_5_are_zeros() -> None:
    baseline = _smooth_pose_trajectory(24)
    model = fit_pose_gp(baseline)

    reconstructed = reconstruct_poses(model, 24)

    assert torch.count_nonzero(reconstructed[:, 1:]).item() == 0


def test_file_size_under_threshold(tmp_path) -> None:
    model = fit_pose_gp(_smooth_pose_trajectory(128))
    path = tmp_path / "pose_gp.bin"

    save_pose_gp(model, path)

    assert path.stat().st_size < 512
