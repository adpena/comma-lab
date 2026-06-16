# SPDX-License-Identifier: MIT
"""NO-FAKE tests for Lever B — the pose-null projection hook.

Build a SYNTHETIC (but exact) ``PoseSubspaceSpectrum`` with a known orthonormal row basis, then prove the
projection is load-bearing: a residual BUILT inside the row space projects to ~0 null (sensitive_frac ~1),
and an isotropic residual projects mostly into the null (null_frac ~ (N-r)/N). Would FAIL if the hook
returned its input unchanged or fabricated the energy split.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.posenet_subspace_spectrum import PoseSubspaceSpectrum
from tac.torch_vehicle.pose_null_projection_hook import (
    pose_null_residual_fraction,
    project_residual_onto_pose_null,
)


def _synthetic_spectrum(h=4, w=4, rank=2, seed=0):
    """A spectrum with `rank` orthonormal row-basis directions in (3*h*w)-dim pixel space."""
    n = 3 * h * w
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((rank, n))
    q, _ = np.linalg.qr(a.T)  # (n, rank) orthonormal cols
    basis = q[:, :rank].T.astype(np.float64)  # (rank, n) orthonormal rows
    sv = np.array([2.0, 1.0][:rank], dtype=np.float64)
    return PoseSubspaceSpectrum(
        singular_values=sv, row_basis=basis, n_pixels=n, h=h, w=w, frame_slot=0,
        compute_path="cpu_torch", effective_dim=float(rank), rank=rank,
    ), basis, n


def test_residual_in_row_space_is_all_sensitive() -> None:
    spec, basis, n = _synthetic_spectrum()
    # build a residual entirely inside the row space → its null component is ~0.
    coeffs = np.array([0.7, -1.3])
    flat = coeffs @ basis  # (n,)
    chw = flat.reshape(3, spec.h, spec.w)
    frac = pose_null_residual_fraction(chw, spec)
    assert frac["null_energy_frac"] == pytest.approx(0.0, abs=1e-9)
    assert frac["sensitive_energy_frac"] == pytest.approx(1.0, abs=1e-9)
    # projecting it onto the null gives ~0 (it had no null component).
    res = project_residual_onto_pose_null(chw, spec)
    assert np.linalg.norm(res.null_residual) == pytest.approx(0.0, abs=1e-6)


def test_isotropic_residual_is_mostly_null() -> None:
    spec, basis, n = _synthetic_spectrum(rank=2)
    rng = np.random.default_rng(7)
    chw = rng.standard_normal((3, spec.h, spec.w))
    frac = pose_null_residual_fraction(chw, spec)
    # isotropic → expected null fraction ~ (N - r)/N; here N large vs r=2 → close to 1.
    expected_null = (n - 2) / n
    assert frac["null_energy_frac"] == pytest.approx(expected_null, abs=0.15)
    assert frac["null_energy_frac"] > frac["sensitive_energy_frac"]


def test_projection_removes_the_sensitive_component() -> None:
    spec, basis, n = _synthetic_spectrum(rank=2, seed=3)
    rng = np.random.default_rng(11)
    chw = rng.standard_normal((3, spec.h, spec.w))
    res = project_residual_onto_pose_null(chw, spec)
    # the projected (null) residual must be orthogonal to every row of the basis (~0 dot product).
    null_flat = res.null_residual.reshape(-1).astype(np.float64)
    for row in basis:
        assert abs(float(row @ null_flat)) == pytest.approx(0.0, abs=1e-4)
    # and it has strictly less energy than the input (the sensitive part was removed).
    assert np.linalg.norm(null_flat) < np.linalg.norm(chw.reshape(-1))


def test_shape_mismatch_fails_closed() -> None:
    spec, _basis, _n = _synthetic_spectrum(h=4, w=4)
    wrong = np.zeros((3, 8, 8))  # wrong resolution
    with pytest.raises(ValueError):
        project_residual_onto_pose_null(wrong, spec)


def test_summary_reports_removed_energy() -> None:
    spec, basis, n = _synthetic_spectrum(rank=2)
    coeffs = np.array([1.0, 1.0])
    flat = coeffs @ basis
    chw = flat.reshape(3, spec.h, spec.w)
    res = project_residual_onto_pose_null(chw, spec)
    s = res.to_summary()
    assert s["removed_energy_frac"] == pytest.approx(s["sensitive_energy_frac"])
    assert s["sensitive_energy_frac"] == pytest.approx(1.0, abs=1e-9)
