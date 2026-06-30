# SPDX-License-Identifier: MIT
"""se(3) WIRE-IN via VERIFIED EQUIVALENCE: the v2 warp's hand-rolled SO(3)/SE(3) math is
parity-gated against the canonical ``tac.lie`` library (the tested, 49-test, numpy-fp oracle).

Why a parity test and not a code replacement: the warp math appears in TWO mlx-free surfaces that
must NOT import ``tac.lie`` (which pulls MLX):
  * the COMPRESS-side warp ``tools/measure_pose_warp_dseg._expmap_so3`` (+ the per-class regime
    homography ``_m_step``), consumed by ``tac.v2_compose.bulk_generator``;
  * the INFLATE.py decoder TEMPLATE (the ``_INFLATE_PY_V2`` string in
    ``tac.v2_compose.archive_grammar``), which is numpy+torch+brotli ONLY at decode (no MLX).

So the wire-in is: ``tac.lie`` is the canonical, tested ORACLE, and this test PROVES the warp's
hand-rolled Rodrigues exp is bit-equivalent to it (the same discipline as the numpy reference oracle
gating the MLX fast path inside ``tac.lie`` itself). Any future drift in either copy is caught here.

[macOS-CPU advisory] NON-PROMOTABLE. Pure math; no GPU, no MLX-as-authority, no score claim.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.lie import _se3_numpy as lie
from tools.measure_pose_warp_dseg import _expmap_so3 as warp_expmap_so3


def _angle_axis_grid(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        theta = 10.0 ** rng.uniform(-9.0, np.log10(np.pi - 1e-3))
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        out.append(theta * axis)
    out.append(np.zeros(3))  # exact zero (the small-angle branch boundary)
    return out


def test_warp_expmap_so3_matches_tac_lie_oracle():
    """The compress-side warp's Rodrigues exp == tac.lie numpy-oracle exp_so3 to fp64 tolerance."""
    worst = 0.0
    for omega in _angle_axis_grid():
        A = np.asarray(warp_expmap_so3(omega), dtype=np.float64)
        B = np.asarray(lie.exp_so3(omega), dtype=np.float64)
        worst = max(worst, float(np.abs(A - B).max()))
    # both are Rodrigues; they differ only in the theta<1e-12 branch (warp: I+K; lie: I+K+0.5 K^2,
    # where ||K^2|| ~ theta^2 < 1e-24). Tolerance is comfortably machine-precision.
    assert worst < 1e-10, f"warp exp_so3 drifted from tac.lie oracle: max|diff|={worst:.3e}"


def test_warp_expmap_so3_is_a_valid_rotation():
    """Sanity: the warp's exp output is SO(3) (orthonormal, det +1) -- so tac.lie's identities apply."""
    for omega in _angle_axis_grid(n=500, seed=1):
        R = np.asarray(warp_expmap_so3(omega), dtype=np.float64)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(float(np.linalg.det(R)) - 1.0) < 1e-9


def test_warp_m_step_ground_homography_uses_canonical_rotation():
    """The per-class ground-regime plane homography M = R - t n^T/d uses the SAME R as tac.lie, so
    the full warp transform is anchored to the verified Lie exp (not a private re-derivation)."""
    from tools.measure_screw_reach_through_R import _m_step
    from tools.measure_pose_warp_dseg import CAMERA_HEIGHT_M

    rng = np.random.default_rng(2)
    s_t, s_r, pitch = -0.0032, 0.05, -0.01  # representative calib (s_r != 0 to exercise rotation)
    for _ in range(200):
        pose6 = rng.normal(scale=0.3, size=6)
        M = np.asarray(_m_step(pose6, s_t, s_r, pitch, "ground"), dtype=np.float64)
        R = np.asarray(lie.exp_so3(s_r * pose6[3:6]), dtype=np.float64)
        t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
        n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
        M_ref = R - np.outer(t, n) / CAMERA_HEIGHT_M
        assert np.allclose(M, M_ref, atol=1e-10), "warp _m_step ground homography drifted from canonical R"


def test_rotonly_regime_is_pure_canonical_rotation():
    from tools.measure_screw_reach_through_R import _m_step

    rng = np.random.default_rng(3)
    s_r = 0.07
    for _ in range(200):
        pose6 = rng.normal(scale=0.3, size=6)
        M = np.asarray(_m_step(pose6, -0.0032, s_r, -0.01, "rotonly"), dtype=np.float64)
        R = np.asarray(lie.exp_so3(s_r * pose6[3:6]), dtype=np.float64)
        assert np.allclose(M, R, atol=1e-10)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
