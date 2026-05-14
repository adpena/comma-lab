# SPDX-License-Identifier: MIT
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
    """Round 2B B1 hygiene (2026-05-06): pass baseline_poses to avoid the
    off-manifold UserWarning. The fit roundtrip semantically depends on dim 0
    only; passing baseline_poses preserves dims 1-5 and removes the noise."""
    baseline = _smooth_pose_trajectory(96)

    model = fit_pose_gp(baseline)
    reconstructed = reconstruct_poses(model, baseline.shape[0], baseline_poses=baseline)

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


def test_reconstruct_with_baseline_preserves_dims_1_5() -> None:
    """Round 2B B1 fix (2026-05-06, 88% confidence): the previous test pinned
    the catastrophic OFF-MANIFOLD path (`baseline_poses=None` returns zeros for
    dims 1-5) as if it were correct. The CORRECT call passes baseline_poses;
    dims 1-5 are then preserved from baseline (not zeroed). Per the production
    docstring, calling without `baseline_poses` warns + returns OFF-MANIFOLD
    bytes that catastrophically degrade scores. Test the correct path.
    """
    baseline = _smooth_pose_trajectory(24)
    model = fit_pose_gp(baseline)

    reconstructed = reconstruct_poses(model, 24, baseline_poses=baseline)

    # Dims 1-5 should match baseline since reconstruction only fits dim 0.
    assert torch.allclose(reconstructed[:, 1:], baseline[:, 1:], atol=1e-5), (
        "reconstruct_poses(baseline_poses=baseline) must preserve dims 1-5"
    )


def test_reconstruct_without_baseline_warns_and_zeros_dims_1_5() -> None:
    """Round 2B B1 fix (2026-05-06): explicit test for the deprecated
    no-baseline path. Asserts the UserWarning fires AND records the
    catastrophic zero-dims-1-5 behavior so future agents can reason about
    when this path triggers. The path is OFF-MANIFOLD per the docstring.
    """
    import warnings

    baseline = _smooth_pose_trajectory(24)
    model = fit_pose_gp(baseline)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reconstructed = reconstruct_poses(model, 24)

    assert any(
        "baseline" in str(w.message).lower() or "off-manifold" in str(w.message).lower()
        for w in caught
    ), f"expected UserWarning about missing baseline_poses; got {[str(w.message) for w in caught]}"
    assert torch.count_nonzero(reconstructed[:, 1:]).item() == 0


def test_file_size_under_threshold(tmp_path) -> None:
    model = fit_pose_gp(_smooth_pose_trajectory(128))
    path = tmp_path / "pose_gp.bin"

    save_pose_gp(model, path)

    assert path.stat().st_size < 512
