# SPDX-License-Identifier: MIT
"""Behavior tests for ``tac.boundary_math.lever_b_generator`` (tasks #55-react + #56).

These tests verify BEHAVIOR not constants (NO-FAKE class 2):
  - the numpy forward ACTUALLY computes the FiLM-modulated MLP (a stub returning zeros
    FAILS the per-pair-distinct-output test);
  - save/load roundtrips weights + cfg EXACTLY (a lossy stub FAILS the allclose);
  - generator_argmax reflects the trained weights (different mod codes -> different argmax);
  - residual_component_stats matches a CONSTRUCTED residual ground truth (a stub that
    returns constant counts FAILS the histogram assertions);
  - aggregate_residual_stats sums correctly across pairs;
  - the contiguous_fraction is the discriminating quantity (salt-and-pepper -> ~0;
    contiguous patches -> high) — the frontier-vs-lever_b decision input.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.lever_b_generator import (
    GeneratorConfig,
    aggregate_residual_stats,
    all_class_boundary_mask,
    all_class_boundary_proximity_and_tangent,
    build_coords,
    deterministic_fourier_B,
    directional_fourier_feats,
    generator_argmax,
    load_generator_npz,
    numpy_reference_forward,
    residual_component_stats,
    save_generator_npz,
)


def test_all_class_boundary_mask_marks_inter_class_edges():
    a = np.zeros((6, 6), dtype=np.uint8)
    a[3:, :] = 2
    bnd = all_class_boundary_mask(a)
    assert bnd[2].all() and bnd[3].all()
    assert not bnd[0].any() and not bnd[5].any()


def test_all_class_boundary_mask_empty_when_uniform():
    assert not all_class_boundary_mask(np.full((5, 5), 3, np.uint8)).any()


def test_all_class_boundary_proximity_peaks_on_boundary():
    a = np.zeros((10, 10), dtype=np.uint8)
    a[5:, :] = 4
    prox, tang = all_class_boundary_proximity_and_tangent(a, tau=2.0)
    assert prox.shape == (10, 10)
    assert prox[4, 5] > prox[0, 0]  # near-boundary > interior
    assert prox.min() >= 0.0 and prox.max() <= 1.0 + 1e-5
    norms = np.sqrt((tang**2).sum(-1))
    assert np.allclose(norms, 1.0, atol=1e-4)  # unit tangent everywhere


def test_directional_fourier_feats_depends_on_tangent():
    coords = np.array([[0.3, -0.2], [0.5, 0.1]], np.float32)
    t1 = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)
    t2 = np.array([[0.0, 1.0], [0.0, 1.0]], np.float32)
    f1 = directional_fourier_feats(coords, t1, n_freqs=4, freq_across=16.0, freq_along=2.0)
    f2 = directional_fourier_feats(coords, t2, n_freqs=4, freq_across=16.0, freq_along=2.0)
    assert f1.shape == (2, 16)
    assert not np.allclose(f1, f2)  # genuinely oriented, not a no-op


def _random_params(cfg: GeneratorConfig, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cf = 2 * cfg.n_fourier
    h, nh, md, nc = cfg.hidden_dim, cfg.n_hidden, cfg.mod_dim, cfg.n_classes
    # small scale so the numpy float64 forward stays finite for a "trained-like" net.
    sc = 0.3
    p = {
        "mod": (rng.standard_normal((cfg.num_pairs, md)) * sc).astype(np.float32),
        "in_proj.weight": (rng.standard_normal((h, cf)) * sc).astype(np.float32),
        "in_proj.bias": (rng.standard_normal(h) * sc).astype(np.float32),
        "film.weight": (rng.standard_normal((2 * h * nh, md)) * sc).astype(np.float32),
        "film.bias": (rng.standard_normal(2 * h * nh) * sc).astype(np.float32),
        "out.weight": (rng.standard_normal((nc, h)) * sc).astype(np.float32),
        "out.bias": (rng.standard_normal(nc) * sc).astype(np.float32),
    }
    for li in range(nh):
        p[f"hidden.{li}.weight"] = (rng.standard_normal((h, h)) * sc).astype(np.float32)
        p[f"hidden.{li}.bias"] = (rng.standard_normal(h) * sc).astype(np.float32)
    return p


# ── numpy forward behavior ────────────────────────────────────────────────
def test_numpy_forward_returns_logits_shape():
    cfg = GeneratorConfig(num_pairs=3, n_fourier=4, hidden_dim=8, n_hidden=2, mod_dim=4)
    p = _random_params(cfg)
    coords = build_coords(4, 5)
    fb = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    out = numpy_reference_forward(p, fb, coords, p["mod"][0], cfg.n_hidden, cfg.hidden_dim)
    assert out.shape == (20, cfg.n_classes)


def test_per_pair_modulation_changes_output():
    """Different mod codes MUST produce different logits (a stub ignoring mod FAILS)."""
    cfg = GeneratorConfig(num_pairs=2, n_fourier=4, hidden_dim=8, n_hidden=2, mod_dim=4)
    p = _random_params(cfg)
    # make the two pairs' mod codes very different.
    p["mod"][0] = 1.0
    p["mod"][1] = -1.0
    coords = build_coords(4, 5)
    fb = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    o0 = numpy_reference_forward(p, fb, coords, p["mod"][0], cfg.n_hidden, cfg.hidden_dim)
    o1 = numpy_reference_forward(p, fb, coords, p["mod"][1], cfg.n_hidden, cfg.hidden_dim)
    assert not np.allclose(o0, o1), "FiLM modulation had no effect — mod ignored (fake)"


def test_fourier_table_is_deterministic():
    a = deterministic_fourier_B(8, 8.0)
    b = deterministic_fourier_B(8, 8.0)
    assert np.array_equal(a, b)
    assert a.shape == (2, 8)


# ── checkpoint I/O ────────────────────────────────────────────────────────
def test_save_load_roundtrip_exact(tmp_path):
    cfg = GeneratorConfig(num_pairs=3, n_fourier=4, hidden_dim=8, n_hidden=2, mod_dim=4)
    p = _random_params(cfg)
    path = tmp_path / "g.npz"
    save_generator_npz(path, p, cfg)
    p2, cfg2 = load_generator_npz(path)
    assert cfg2.to_dict() == cfg.to_dict()
    assert set(p2) == set(p)
    for k in p:
        assert np.allclose(p[k], p2[k]), f"weight {k} not roundtripped"


def test_save_refuses_tmp_path():
    cfg = GeneratorConfig(num_pairs=1)
    with pytest.raises(ValueError, match="/tmp-class"):
        save_generator_npz("/tmp/should_refuse.npz", _random_params(cfg), cfg)


# ── generator_argmax reflects weights ─────────────────────────────────────
def test_generator_argmax_shape_and_dtype():
    cfg = GeneratorConfig(num_pairs=2, n_fourier=4, hidden_dim=8, n_hidden=2, mod_dim=4)
    p = _random_params(cfg)
    coords = build_coords(6, 7)
    am = generator_argmax(p, cfg, coords, 0, 6, 7)
    assert am.shape == (6, 7) and am.dtype == np.uint8
    assert am.max() < cfg.n_classes


def test_generator_argmax_differs_per_pair():
    cfg = GeneratorConfig(num_pairs=2, n_fourier=6, hidden_dim=12, n_hidden=2, mod_dim=6)
    p = _random_params(cfg, seed=7)
    p["mod"][0] = 2.0
    p["mod"][1] = -2.0
    coords = build_coords(8, 8)
    a0 = generator_argmax(p, cfg, coords, 0, 8, 8)
    a1 = generator_argmax(p, cfg, coords, 1, 8, 8)
    # at least SOME pixels differ (the per-pair code matters).
    assert (a0 != a1).any(), "per-pair argmax identical — mod ignored (fake)"


# ── residual_component_stats (the DECISIVE Step-1 measurement) ─────────────
def test_residual_stats_constructed_ground_truth():
    """Construct a known residual: 3 single-px flips + 1 contiguous 16px patch."""
    base = np.zeros((20, 20), np.uint8)
    tgt = np.zeros((20, 20), np.uint8)
    base[0, 0] = 1
    base[0, 5] = 2
    base[10, 10] = 3
    base[2:6, 2:6] = 4  # 16px contiguous patch
    st = residual_component_stats(base, tgt)
    assert st.n_flips == 3 + 16
    assert st.n_components == 4
    assert st.single_pixel_components == 3
    assert abs(st.single_pixel_fraction - 3 / 4) < 1e-9
    assert st.largest_component_pixels == 16
    assert st.size_histogram["1px"] == 3
    assert st.size_histogram["10-49px"] == 1
    # contiguous fraction = 16 / 19 (the 16px patch is the only >=4 component).
    assert abs(st.contiguous_fraction - 16 / 19) < 1e-9


def test_residual_stats_salt_and_pepper_low_contiguity():
    """Salt-and-pepper (all single-px) -> contiguous_fraction == 0 (the frontier finding)."""
    rng = np.random.default_rng(0)
    base = np.zeros((40, 40), np.uint8)
    tgt = np.zeros((40, 40), np.uint8)
    # scatter 30 isolated single-px flips (no two adjacent).
    placed = 0
    while placed < 30:
        r, c = int(rng.integers(0, 40)), int(rng.integers(0, 40))
        # ensure isolation: no 4-neighbour already flipped.
        if base[r, c] != 0:
            continue
        if any(base[r + dr, c + dc] != 0 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
               if 0 <= r + dr < 40 and 0 <= c + dc < 40):
            continue
        base[r, c] = 1
        placed += 1
    st = residual_component_stats(base, tgt)
    assert st.single_pixel_fraction == 1.0
    assert st.contiguous_fraction == 0.0


def test_residual_stats_no_flips():
    a = np.zeros((10, 10), np.uint8)
    st = residual_component_stats(a, a)
    assert st.n_flips == 0 and st.n_components == 0 and st.contiguous_fraction == 0.0


def test_aggregate_residual_stats_sums():
    s1 = residual_component_stats(
        np.array([[1, 0], [0, 0]], np.uint8), np.zeros((2, 2), np.uint8))  # 1 single flip
    s2 = residual_component_stats(
        np.array([[2, 2], [2, 2]], np.uint8), np.zeros((2, 2), np.uint8))  # 4px patch
    agg = aggregate_residual_stats([s1, s2])
    assert agg["n_pairs"] == 2
    assert agg["total_flips"] == 1 + 4
    assert agg["total_components"] == 2
    assert agg["single_pixel_components"] == 1
    # 4px patch is contiguous (>=4); single is not.
    assert abs(agg["contiguous_flip_fraction"] - 4 / 5) < 1e-9


def test_aggregate_empty():
    assert aggregate_residual_stats([])["n_pairs"] == 0
