# SPDX-License-Identifier: MIT
"""Deterministic rank-1 pose compression via a tiny polynomial model.

Lane GP targets the contest observation that PoseNet's useful pose Jacobian is
almost entirely dim 0. The archive stores a degree-10 polynomial for dim 0 and
diagnostic sigma values for dims 1-5; inflate reconstructs dims 1-5 as zeros.

LANE_GP_BASIS_FIT_KILL_ACKNOWLEDGED:
This module's lane class (smooth-basis pose-fit) was killed 2026-04-30 per
Council #271 + Lane GP v4 design verdict
(.omx/research/council_lane_gp_v4_design_20260430.md). The Lane G v3 baseline
pose trajectory is approximately white-noise in dims 1-5 (diff_std > signal_std)
with uniformly-distributed spectral support — no smooth basis (polynomial /
B-spline / DCT / natural cubic) can fit it below RMSE ≈ 1.2 (near signal std).
The Runge-phenomenon diagnosis in
project_lane_gp_v3_landed_runge_phenomenon_20260429.md was incomplete; the
trajectory is structurally incompressible by smooth bases at any K. This module
is RETAINED for archival/historical reasons (Lane GP v3 reproducer); the
underlying lane class is killed. See preflight.py Check 91.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


POSE_GP_SENTINEL = b"pose_gp_v1"
POSE_GP_DEGREE = 10
POSE_GP_NUM_COEFFS = POSE_GP_DEGREE + 1
POSE_GP_NUM_SIGMA = 5
_POSE_GP_STRUCT = "<" + "e" * (POSE_GP_NUM_COEFFS + POSE_GP_NUM_SIGMA)


@dataclass(frozen=True)
class PoseGPModel:
    """Tiny deterministic pose model.

    Attributes:
        poly_coeffs: Degree-10 polynomial coefficients for pose dim 0, ordered
            as ``numpy.polyfit`` returns them: highest power first.
        sigma: Per-dimension standard deviations for baseline dims 1-5. These
            are retained for diagnostics and round-trip integrity; reconstruction
            deliberately does not sample them.
    """

    poly_coeffs: torch.Tensor
    sigma: torch.Tensor

    def __post_init__(self) -> None:
        coeffs = torch.as_tensor(self.poly_coeffs, dtype=torch.float32).flatten()
        sigma = torch.as_tensor(self.sigma, dtype=torch.float32).flatten()
        if coeffs.numel() != POSE_GP_NUM_COEFFS:
            raise ValueError(
                f"poly_coeffs must contain {POSE_GP_NUM_COEFFS} values, got {coeffs.numel()}"
            )
        if sigma.numel() != POSE_GP_NUM_SIGMA:
            raise ValueError(f"sigma must contain {POSE_GP_NUM_SIGMA} values, got {sigma.numel()}")
        object.__setattr__(self, "poly_coeffs", coeffs)
        object.__setattr__(self, "sigma", sigma)


def fit_pose_gp(baseline_poses: torch.Tensor) -> PoseGPModel:
    """Fit a degree-10 polynomial to baseline pose dim 0.

    Args:
        baseline_poses: Tensor with shape ``(N, 6)``.

    Returns:
        A :class:`PoseGPModel` with polynomial coefficients and diagnostic
        sigma values for dims 1-5.
    """

    poses = torch.as_tensor(baseline_poses, dtype=torch.float32)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"baseline_poses must have shape (N, 6), got {tuple(poses.shape)}")
    if poses.shape[0] < POSE_GP_NUM_COEFFS:
        raise ValueError(
            f"Need at least {POSE_GP_NUM_COEFFS} pose pairs to fit degree-{POSE_GP_DEGREE}, "
            f"got {poses.shape[0]}"
        )

    t = np.linspace(0.0, 1.0, int(poses.shape[0]), dtype=np.float64)
    y = poses[:, 0].detach().cpu().numpy().astype(np.float64)
    coeffs = np.polyfit(t, y, deg=POSE_GP_DEGREE).astype(np.float32)
    sigma = poses[:, 1:].detach().float().std(dim=0, unbiased=False)
    return PoseGPModel(
        poly_coeffs=torch.from_numpy(coeffs).to(dtype=torch.float32),
        sigma=sigma.to(dtype=torch.float32),
    )


def reconstruct_poses(
    model: PoseGPModel,
    n_pairs: int,
    baseline_poses: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct deterministic FiLM pose conditioning vectors.

    Dim 0 is the polynomial value on ``linspace(0, 1, n_pairs)``.

    Dims 1-5: if ``baseline_poses`` is provided (recommended), use the
    baseline values to preserve on-manifold behavior for renderers trained
    with full 6-DOF FiLM conditioning. If ``None``, dims 1-5 default to zero
    — but that produces OFF-MANIFOLD inputs for 6-DOF-trained renderers and
    catastrophically breaks PoseNet/SegNet (Lane GP v2 audit, 2026-04-29:
    score 89.66 vs predicted [1.05, 1.20], because dims 1-5 = 0 on Lane A's
    renderer = pose 149.95, seg 0.50). See memory
    project_lane_mn_radial_zoom_negative + project_lane_gp_v2_audit_20260429.
    """

    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    t = np.linspace(0.0, 1.0, int(n_pairs), dtype=np.float64)
    coeffs = model.poly_coeffs.detach().cpu().numpy().astype(np.float64)
    dim0 = np.polyval(coeffs, t).astype(np.float32)
    if baseline_poses is None:
        # Preserve old behavior for tests; warn loudly that this is off-manifold.
        import warnings
        warnings.warn(
            "reconstruct_poses() called without baseline_poses — dims 1-5 "
            "default to ZERO, which is OFF-MANIFOLD for 6-DOF-trained "
            "renderers. This catastrophically degrades scores. Pass "
            "baseline_poses=Lane_A_optimized_poses to preserve dims 1-5.",
            stacklevel=2,
        )
        poses = torch.zeros(int(n_pairs), 6, dtype=torch.float32)
    else:
        baseline = torch.as_tensor(baseline_poses, dtype=torch.float32)
        if baseline.shape != (n_pairs, 6):
            raise ValueError(
                f"baseline_poses shape {tuple(baseline.shape)} != "
                f"({n_pairs}, 6); cannot use for on-manifold reconstruction."
            )
        poses = baseline.clone()
    poses[:, 0] = torch.from_numpy(dim0)
    return poses


def save_pose_gp(model: PoseGPModel, path: str | Path) -> None:
    """Write a compact ``pose_gp_v1`` binary with FP16 coefficients."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Move both tensors to CPU before cat — avoids "Expected all tensors on
    # same device" when poly_coeffs is on cuda but sigma on cpu (or vice versa).
    coeffs = model.poly_coeffs.detach().cpu()
    sigma = model.sigma.detach().cpu()
    values = torch.cat([coeffs, sigma]).to(torch.float16).tolist()
    payload = POSE_GP_SENTINEL + struct.pack(_POSE_GP_STRUCT, *values)
    out_path.write_bytes(payload)


def load_pose_gp(path: str | Path) -> PoseGPModel:
    """Load a ``pose_gp_v1`` binary written by :func:`save_pose_gp`."""

    raw = Path(path).read_bytes()
    if not raw.startswith(POSE_GP_SENTINEL):
        raise ValueError(f"{path} is not a pose_gp_v1 artifact")
    body = raw[len(POSE_GP_SENTINEL):]
    expected = struct.calcsize(_POSE_GP_STRUCT)
    if len(body) != expected:
        raise ValueError(f"{path} has {len(body)} payload bytes, expected {expected}")
    values = struct.unpack(_POSE_GP_STRUCT, body)
    coeffs = torch.tensor(values[:POSE_GP_NUM_COEFFS], dtype=torch.float32)
    sigma = torch.tensor(values[POSE_GP_NUM_COEFFS:], dtype=torch.float32)
    return PoseGPModel(poly_coeffs=coeffs, sigma=sigma)
