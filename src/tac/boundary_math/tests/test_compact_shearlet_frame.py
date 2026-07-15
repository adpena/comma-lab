# SPDX-License-Identifier: MIT
"""Tests for the genuinely compact cone-adapted shearlet frame (task #502).

The load-bearing tests are the two SWAP-TESTS: (1) LOCALIZATION -- the frame must
pass a span/concentration gate a plain oriented-Fourier basis structurally FAILS;
(2) SHEAR-SELECTIVITY -- steering must be by SHEAR (anchor-axis invariant), which a
ROTATION-steered family (curvelet) structurally FAILS. Together they are the
catalog-#351 anti-fake guard. Plus determinism, MLX parity, integer-lattice
preservation, parabolic-scaling law, config validation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.boundary_math.compact_shearlet_frame import (
    CompactShearletConfig,
    _atom_xi_eta,
    _sigma_pair,
    compact_shearlet_feats,
    mlx_parity_check,
    n_atoms,
    shearlet_atom_index,
    shearlet_certificate,
)


def test_frame_is_genuinely_localized_and_shear_steered_swap_test():
    """The decisive anti-fake test: localized (Fourier fails) AND shear-steered (rotation fails)."""
    cert = shearlet_certificate(CompactShearletConfig())
    assert cert.passes is True
    # localization: shearlet envelope is a spatial bump; Fourier envelope is constant.
    assert cert.shearlet_envelope_span >= cert.span_gate
    assert cert.fourier_envelope_span < cert.span_gate
    assert cert.shearlet_envelope_span / cert.fourier_envelope_span > 1e5
    # shear-selectivity: shear invariant on the anchor line, rotation NOT.
    assert cert.shear_anchor_dispersion < 1e-6
    assert cert.rotation_anchor_dispersion > 1e-3
    assert cert.shear_discrimination_ratio >= cert.discrimination_gate


def test_rotation_swap_flips_certificate_to_false():
    """#351-tight: if the internal steering is swapped to ROTATION, ``passes`` MUST flip False.

    This proves the shear-selectivity gate reads the REAL forward, not a re-derived
    formula -- a rotation basis wearing a shearlet label cannot falsely pass.
    """
    import tac.boundary_math.compact_shearlet_frame as mod

    orig = mod._atom_xi_eta

    def rotation_xi_eta(x, y, a):
        dx = x - a.cx
        dy = y - a.cy
        theta = math.atan(a.shear_k) if a.cone == 0 else (math.pi / 2 - math.atan(a.shear_k))
        ct, st = math.cos(theta), math.sin(theta)
        return dx * ct + dy * st, -dx * st + dy * ct

    mod._atom_xi_eta = rotation_xi_eta
    try:
        cert = shearlet_certificate(CompactShearletConfig())
        assert cert.passes is False
        # rotation is NOT anchor-invariant -> shear dispersion jumps, ratio collapses.
        assert cert.shear_anchor_dispersion > 1e-3
        assert cert.shear_discrimination_ratio < cert.discrimination_gate
    finally:
        mod._atom_xi_eta = orig


def test_shear_matrix_fixes_the_anchor_axis_pointwise():
    """The genuine shear property: S_k * (x, 0) = (x, 0). On the anchor line the xi
    coordinate (oscillation axis) is shear-invariant; a rotation would move it."""
    cfg = CompactShearletConfig()
    xs = np.linspace(-1.0, 1.0, 21)
    zeros = np.zeros_like(xs)
    base = None
    for k in (0.0, 0.5, 1.0, -0.75):
        a = type(shearlet_atom_index(cfg)[0])(cone=0, scale=0, shear_k=k, freq=2.0,
                                              sigma_n=0.5, sigma_t=0.5, cx=0.0, cy=0.0)
        xi, eta = _atom_xi_eta(xs, zeros, a)  # anchor line y=0
        assert np.allclose(xi, xs)   # xi = dx + k*0 = dx, invariant across k
        assert np.allclose(eta, 0.0)
        if base is None:
            base = xi
        else:
            assert np.allclose(xi, base)


def test_normal_direction_steers_with_shear():
    """The oscillation normal grad(xi) = (1, k) [cone 0] genuinely steers with k."""
    cfg = CompactShearletConfig()
    Atom = type(shearlet_atom_index(cfg)[0])
    # xi(dx,dy) = dx + k*dy -> partials (1, k); check numerically for two k's differ.
    for k in (0.0, 1.0):
        a = Atom(cone=0, scale=0, shear_k=k, freq=2.0, sigma_n=0.5, sigma_t=0.5, cx=0.0, cy=0.0)
        xi_dx, _ = _atom_xi_eta(np.array([1.0]), np.array([0.0]), a)
        xi_dy, _ = _atom_xi_eta(np.array([0.0]), np.array([1.0]), a)
        assert math.isclose(float(xi_dx[0]), 1.0)   # d xi / d dx = 1
        assert math.isclose(float(xi_dy[0]), k)     # d xi / d dy = k (STEERS)


def test_integer_lattice_preserving_gate():
    cert = shearlet_certificate(CompactShearletConfig())
    assert cert.integer_lattice_preserving is True


def test_two_cones_cover_wide_angular_range():
    """Cone 0 (horizontal anchor) + cone 1 (vertical anchor) -> both orientations present."""
    cfg = CompactShearletConfig(two_cones=True)
    atoms = shearlet_atom_index(cfg)
    cones = {a.cone for a in atoms}
    assert cones == {0, 1}
    cfg1 = CompactShearletConfig(two_cones=False)
    assert {a.cone for a in shearlet_atom_index(cfg1)} == {0}


def test_paired_envelope_is_spatially_varying():
    cfg = CompactShearletConfig()
    from tac.boundary_math.lever_b_levelset_generator import build_coords

    coords = build_coords(33, 33)
    feats = compact_shearlet_feats(coords, cfg)
    d = feats.shape[1] // 2
    env = feats[:, :d].astype(np.float64) ** 2 + feats[:, d:].astype(np.float64) ** 2
    assert float(np.ptp(env, axis=0).max()) > 0.1


def test_parabolic_scaling_law_and_monotone():
    cfg = CompactShearletConfig(w0=0.5, width_ratio=2.0, aniso=1.0, n_scales=4, min_sigma=1e-9)
    for j in range(cfg.n_scales):
        sn, st = _sigma_pair(cfg, j)
        assert math.isclose(sn, (st / cfg.aniso) ** 2 / cfg.w0, rel_tol=1e-9)
    cert = shearlet_certificate(cfg)
    assert cert.parabolic_scaling_monotone is True


def test_feature_shape_and_atom_count():
    cfg = CompactShearletConfig()
    from tac.boundary_math.lever_b_levelset_generator import build_coords

    coords = build_coords(9, 11)
    feats = compact_shearlet_feats(coords, cfg)
    assert feats.shape == (99, 2 * n_atoms(cfg))
    assert feats.dtype == np.float32
    assert len(shearlet_atom_index(cfg)) == n_atoms(cfg)


def test_atom_count_formula():
    # cones x scales x (2*n_shear+1) x n_trans^2
    cfg = CompactShearletConfig(n_scales=2, n_shear=2, two_cones=True, n_trans=2)
    assert n_atoms(cfg) == 2 * 2 * 5 * 4  # 80
    cfg1 = CompactShearletConfig(n_scales=3, n_shear=1, two_cones=False, n_trans=1)
    assert n_atoms(cfg1) == 1 * 3 * 3 * 1  # 9


def test_determinism_and_stable_order():
    cfg = CompactShearletConfig()
    from tac.boundary_math.lever_b_levelset_generator import build_coords

    coords = build_coords(13, 13)
    a = compact_shearlet_feats(coords, cfg)
    b = compact_shearlet_feats(coords, cfg)
    assert np.array_equal(a, b)
    assert shearlet_atom_index(cfg) == shearlet_atom_index(cfg)


def test_mlx_parity_if_available():
    res = mlx_parity_check(CompactShearletConfig())
    if not res["mlx_available"]:
        pytest.skip(f"mlx not available: {res.get('reason')}")
    assert res["within_tol"] is True
    assert res["max_abs_diff"] <= 1e-4


def test_features_bounded_by_envelope():
    cfg = CompactShearletConfig()
    from tac.boundary_math.lever_b_levelset_generator import build_coords

    coords = build_coords(17, 17)
    feats = compact_shearlet_feats(coords, cfg)
    assert np.max(np.abs(feats)) <= 1.0 + 1e-5


def test_config_rejects_bad_args():
    with pytest.raises(ValueError):
        CompactShearletConfig(n_scales=0)
    with pytest.raises(ValueError):
        CompactShearletConfig(w0=-1.0)
    with pytest.raises(ValueError):
        CompactShearletConfig(width_ratio=1.0)
    with pytest.raises(ValueError):
        CompactShearletConfig(aniso=0.5)
    with pytest.raises(ValueError):
        CompactShearletConfig(shear_step=0.0)
    with pytest.raises(ValueError):
        CompactShearletConfig(f0=float("nan"))
    with pytest.raises(ValueError):
        CompactShearletConfig(two_cones="yes")  # type: ignore[arg-type]
