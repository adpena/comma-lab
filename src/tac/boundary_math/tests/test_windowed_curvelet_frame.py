# SPDX-License-Identifier: MIT
"""Tests for the genuinely localized windowed-curvelet frame (task #502).

The load-bearing test is the LOCALIZATION SWAP-TEST: the frame must PASS a localization
certificate that a plain oriented-Fourier basis structurally FAILS -- the catalog-#351
anti-fake guard. Plus determinism, MLX parity, parabolic-scaling law, config validation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.boundary_math.lever_b_levelset_generator import (
    PolarDirectionalFourierBankConfig,
    build_coords,
    polar_directional_fourier_B,
    polar_directional_fourier_feats,
)
from tac.boundary_math.windowed_curvelet_frame import (
    WindowedCurveletConfig,
    _envelope_energy_stats,
    _sigma_pair,
    curvelet_atom_index,
    localization_certificate,
    mlx_parity_check,
    n_atoms,
    windowed_curvelet_feats,
)


def test_frame_is_genuinely_localized_swap_test():
    """The decisive anti-fake test: curvelet localized, Fourier NOT, on the same gate."""
    cert = localization_certificate(WindowedCurveletConfig())
    assert cert.passes is True
    # Curvelet envelope is a spatial bump; Fourier envelope is constant (~1e-7 span).
    assert cert.curvelet_envelope_span >= cert.span_gate
    assert cert.fourier_envelope_span < cert.span_gate
    # decisive discrimination margin: many orders of magnitude.
    assert cert.curvelet_envelope_span / cert.fourier_envelope_span > 1e5


def test_oriented_fourier_basis_fails_the_localization_gate():
    """Directly prove a plain oriented-Fourier basis FAILS the span gate the frame passes."""
    coords = build_coords(33, 33)
    ffeats = polar_directional_fourier_feats(
        coords, polar_directional_fourier_B(PolarDirectionalFourierBankConfig())
    )
    f_span, _, _ = _envelope_energy_stats(ffeats, coords)
    assert f_span < 0.10  # constant paired envelope -> not localized (would FAIL a curvelet claim)


def test_energy_concentration_and_tangent_elongation():
    cert = localization_certificate(WindowedCurveletConfig())
    assert cert.curvelet_energy_concentration > 0.5  # localized: energy in few pixels
    assert cert.tangent_over_normal_extent > 1.0  # edge-shaped: elongated along the tangent


def test_paired_envelope_is_spatially_varying():
    """real^2 + imag^2 = env^2 must be a spatial bump (large ptp), unlike Fourier's constant 1."""
    cfg = WindowedCurveletConfig()
    coords = build_coords(33, 33)
    feats = windowed_curvelet_feats(coords, cfg)
    d = feats.shape[1] // 2
    env = feats[:, :d].astype(np.float64) ** 2 + feats[:, d:].astype(np.float64) ** 2
    spans = np.ptp(env, axis=0)
    assert float(spans.max()) > 0.1


def test_parabolic_scaling_law_width_like_length_squared():
    """sigma_n = (sigma_t/aniso)^2 / w0 -- the Candes-Donoho parabolic law, per scale."""
    cfg = WindowedCurveletConfig(w0=0.5, width_ratio=2.0, aniso=1.0, n_scales=4, min_sigma=1e-9)
    for j in range(cfg.n_scales):
        sn, st = _sigma_pair(cfg, j)
        expected_sn = (st / cfg.aniso) ** 2 / cfg.w0
        assert math.isclose(sn, expected_sn, rel_tol=1e-9)


def test_parabolic_scaling_monotone():
    cert = localization_certificate(WindowedCurveletConfig())
    assert cert.parabolic_scaling_monotone is True


def test_aniso_boosts_tangent_without_breaking_parabolic():
    """aniso>1 elongates the atom (edge-shaped) while KEEPING sigma_n ~ sigma_t^2."""
    cfg = WindowedCurveletConfig(aniso=3.0, min_sigma=1e-9)
    sn0, st0 = _sigma_pair(cfg, 0)
    assert st0 > sn0  # elongated along tangent
    # parabolic identity still holds with the aniso constant folded in.
    assert math.isclose(sn0, (st0 / cfg.aniso) ** 2 / cfg.w0, rel_tol=1e-9)


def test_feature_shape_and_atom_count():
    cfg = WindowedCurveletConfig()
    coords = build_coords(9, 11)
    feats = windowed_curvelet_feats(coords, cfg)
    assert feats.shape == (99, 2 * n_atoms(cfg))
    assert feats.dtype == np.float32
    assert len(curvelet_atom_index(cfg)) == n_atoms(cfg)


def test_determinism():
    cfg = WindowedCurveletConfig()
    coords = build_coords(13, 13)
    a = windowed_curvelet_feats(coords, cfg)
    b = windowed_curvelet_feats(coords, cfg)
    assert np.array_equal(a, b)


def test_atom_order_is_stable():
    cfg = WindowedCurveletConfig()
    idx1 = curvelet_atom_index(cfg)
    idx2 = curvelet_atom_index(cfg)
    assert idx1 == idx2  # order is the counted-coefficient order at decode


def test_atom_index_covers_scales_orientations_translations():
    cfg = WindowedCurveletConfig(n_scales=2, n_orient0=3, n_trans=2)
    atoms = curvelet_atom_index(cfg)
    # j=0: L=3 orientations; j=1: L=3*2^(1//2)=3 -> 6 orient-total; x n_trans^2=4 -> 24 atoms.
    assert len(atoms) == 24
    scales = {a.scale for a in atoms}
    assert scales == {0, 1}


def test_mlx_parity_if_available():
    res = mlx_parity_check(WindowedCurveletConfig())
    if not res["mlx_available"]:
        pytest.skip(f"mlx not available: {res.get('reason')}")
    assert res["within_tol"] is True
    assert res["max_abs_diff"] <= 1e-4


def test_config_rejects_bad_args():
    with pytest.raises(ValueError):
        WindowedCurveletConfig(n_scales=0)
    with pytest.raises(ValueError):
        WindowedCurveletConfig(w0=-1.0)
    with pytest.raises(ValueError):
        WindowedCurveletConfig(width_ratio=1.0)  # must exceed 1
    with pytest.raises(ValueError):
        WindowedCurveletConfig(aniso=0.5)  # must be >= 1
    with pytest.raises(ValueError):
        WindowedCurveletConfig(f0=float("nan"))


def test_finer_scale_atoms_are_thinner_across():
    """sigma_n shrinks with scale (finer atoms resolve sharper boundaries)."""
    cfg = WindowedCurveletConfig(min_sigma=1e-9)
    sn_prev = None
    for j in range(cfg.n_scales):
        sn, _ = _sigma_pair(cfg, j)
        if sn_prev is not None:
            assert sn < sn_prev
        sn_prev = sn


def test_features_bounded_by_envelope():
    """|feature| <= env <= 1 (envelope is a normalized Gaussian; oscillation is bounded)."""
    cfg = WindowedCurveletConfig()
    coords = build_coords(17, 17)
    feats = windowed_curvelet_feats(coords, cfg)
    assert np.max(np.abs(feats)) <= 1.0 + 1e-5
