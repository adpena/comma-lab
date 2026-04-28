"""Lane GE: compact one-dimensional geodesic pose compression.

PoseNet's useful signal is effectively rank-1, so this module stores only
the road-manifold coordinate in pose dimension 0.  The coordinate is evaluated
as a Chebyshev series over normalized time; the derivative of that series is
the implied velocity profile.  The remaining five pose dimensions are fixed to
zero at inflate time.
"""
from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn

GEODESIC_POSE_DEGREE: int = 12
GEODESIC_POSE_SENTINEL: bytes = b"TACGEODE1"

CAMERA_FX: float = 910.0
CAMERA_FY: float = 910.0
CAMERA_PP: tuple[float, float] = (582.0, 437.0)

_HEADER_FMT = "<B"
_COEFF_FMT = f"<{GEODESIC_POSE_DEGREE + 1}f"

__all__ = [
    "GEODESIC_POSE_DEGREE",
    "GEODESIC_POSE_SENTINEL",
    "CAMERA_FX",
    "CAMERA_FY",
    "CAMERA_PP",
    "GeodesicPoseModel",
    "fit_geodesic_pose",
    "save_geodesic_pose",
    "load_geodesic_pose",
]


def _chebyshev_basis(
    t: torch.Tensor,
    degree: int = GEODESIC_POSE_DEGREE,
) -> torch.Tensor:
    """Return ``T_0..T_degree`` evaluated at normalized times ``t``."""
    if degree < 0:
        raise ValueError(f"degree must be non-negative; got {degree}")
    x = t.mul(2.0).sub(1.0)
    terms = [torch.ones_like(x)]
    if degree >= 1:
        terms.append(x)
    for k in range(2, degree + 1):
        terms.append(2.0 * x * terms[k - 1] - terms[k - 2])
    return torch.stack(terms, dim=-1)


class GeodesicPoseModel(nn.Module):
    """Differentiable one-DOF pose model.

    Args:
        coeffs: Chebyshev coefficients with shape ``(GEODESIC_POSE_DEGREE+1,)``.
    """

    def __init__(self, coeffs: torch.Tensor) -> None:
        super().__init__()
        coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
        expected = (GEODESIC_POSE_DEGREE + 1,)
        if tuple(coeffs.shape) != expected:
            raise ValueError(f"coeffs must have shape {expected}; got {tuple(coeffs.shape)}")
        self.coeffs = nn.Parameter(coeffs.clone())

    def forward(self, n: int) -> torch.Tensor:
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}")
        t = torch.linspace(
            0.0,
            1.0,
            int(n),
            device=self.coeffs.device,
            dtype=self.coeffs.dtype,
        )
        basis = _chebyshev_basis(t, GEODESIC_POSE_DEGREE)
        dim0 = basis @ self.coeffs
        out = torch.zeros(int(n), 6, device=self.coeffs.device, dtype=self.coeffs.dtype)
        out[:, 0] = dim0
        return out


def fit_geodesic_pose(
    poses: torch.Tensor,
    *,
    require_cuda: bool = False,
) -> GeodesicPoseModel:
    """Fit a compact Chebyshev geodesic model to PoseNet-style poses.

    Only pose dimension 0 is fitted.  Dimensions 1..5 are deliberately
    discarded because the measured scorer signal is rank-1.
    """
    if require_cuda and poses.device.type != "cuda":
        raise RuntimeError("CUDA is required for geodesic pose fitting")
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"poses must have shape (N, 6); got {tuple(poses.shape)}")
    if poses.shape[0] < GEODESIC_POSE_DEGREE + 1:
        raise ValueError(
            f"poses must contain at least {GEODESIC_POSE_DEGREE + 1} samples; "
            f"got {poses.shape[0]}"
        )

    device = poses.device
    t = torch.linspace(0.0, 1.0, poses.shape[0], device=device, dtype=torch.float64)
    basis = _chebyshev_basis(t, GEODESIC_POSE_DEGREE)
    target = poses[:, 0].to(dtype=torch.float64)
    coeffs = torch.linalg.lstsq(basis, target).solution.to(dtype=torch.float32)
    return GeodesicPoseModel(coeffs.to(device=device))


def save_geodesic_pose(model: GeodesicPoseModel, path: str | Path) -> None:
    """Write a compact binary geodesic-pose artifact."""
    path = Path(path)
    coeffs = model.coeffs.detach().cpu().float()
    if coeffs.numel() != GEODESIC_POSE_DEGREE + 1:
        raise ValueError(
            f"model has {coeffs.numel()} coeffs; expected {GEODESIC_POSE_DEGREE + 1}"
        )
    raw = (
        GEODESIC_POSE_SENTINEL
        + struct.pack(_HEADER_FMT, GEODESIC_POSE_DEGREE)
        + struct.pack(_COEFF_FMT, *[float(x) for x in coeffs])
    )
    path.write_bytes(raw)


def load_geodesic_pose(path: str | Path) -> GeodesicPoseModel:
    """Load a compact binary geodesic-pose artifact."""
    raw = Path(path).read_bytes()
    if not raw.startswith(GEODESIC_POSE_SENTINEL):
        raise ValueError("invalid geodesic pose sentinel")
    cursor = len(GEODESIC_POSE_SENTINEL)
    (degree,) = struct.unpack_from(_HEADER_FMT, raw, cursor)
    cursor += struct.calcsize(_HEADER_FMT)
    if degree != GEODESIC_POSE_DEGREE:
        raise ValueError(f"unsupported geodesic pose degree {degree}")
    coeff_bytes = struct.calcsize(_COEFF_FMT)
    if len(raw) != cursor + coeff_bytes:
        raise ValueError(f"invalid geodesic pose artifact length {len(raw)}")
    coeffs = torch.tensor(struct.unpack_from(_COEFF_FMT, raw, cursor), dtype=torch.float32)
    return GeodesicPoseModel(coeffs)
