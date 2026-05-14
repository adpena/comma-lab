# SPDX-License-Identifier: MIT
"""Lane GE tests: degree-12 geodesic pose compression."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.geodesic_pose import (
    GEODESIC_POSE_DEGREE,
    GEODESIC_POSE_SENTINEL,
    GeodesicPoseModel,
    fit_geodesic_pose,
    load_geodesic_pose,
    save_geodesic_pose,
)


def _poses(n: int = 32) -> torch.Tensor:
    t = torch.linspace(0, 1, n)
    poses = torch.zeros(n, 6)
    poses[:, 0] = 2.0 + 0.5 * t + 0.25 * t.square()
    poses[:, 1:] = torch.randn(n, 5) * 0.01
    return poses


def test_forward_shape_and_zero_tail() -> None:
    model = fit_geodesic_pose(_poses())
    out = model(17)
    assert out.shape == (17, 6)
    assert torch.all(out[:, 1:] == 0)


def test_gradient_flow_through_coefficients() -> None:
    model = GeodesicPoseModel(torch.zeros(GEODESIC_POSE_DEGREE + 1))
    out = model(8)
    loss = out[:, 0].square().mean()
    loss.backward()
    assert model.coeffs.grad is not None
    assert torch.isfinite(model.coeffs.grad).all()


def test_edge_cases_reject_bad_shapes_and_short_inputs() -> None:
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        fit_geodesic_pose(torch.zeros(10, 5))
    with pytest.raises(ValueError, match="at least"):
        fit_geodesic_pose(torch.zeros(GEODESIC_POSE_DEGREE, 6))
    with pytest.raises(ValueError, match="positive"):
        GeodesicPoseModel(torch.zeros(GEODESIC_POSE_DEGREE + 1))(0)


def test_determinism_and_compact_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(123)
    poses = _poses(40)
    a = fit_geodesic_pose(poses)
    b = fit_geodesic_pose(poses)
    assert torch.allclose(a.coeffs, b.coeffs)

    path = tmp_path / "geodesic_pose_v1.bin"
    save_geodesic_pose(a, path)
    raw = path.read_bytes()
    assert raw.startswith(GEODESIC_POSE_SENTINEL)
    assert len(raw) <= 80
    loaded = load_geodesic_pose(path)
    assert torch.allclose(a(40), loaded(40), atol=2e-2)


def test_cuda_only_enforcement_raises_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        fit_geodesic_pose(_poses(), require_cuda=True)

