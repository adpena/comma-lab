"""Tests for ``tac.optimization.ddm_bp2_blind_pose_actuator``.

These verify BEHAVIOUR, not constants (CLAUDE.md NO-FAKE class 2): the tap
decomposition is checked by reproducing an independent bilinear warp, the adjoint
by the inner-product identity, and the blind set against the real torch operator.
Every test would fail if the corresponding function body were replaced by a
canonical-looking return value.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_bp2_blind_pose_actuator import (
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    adjoint_taps,
    apply_taps,
    blind_influence_mass,
    d_column_weights,
    v4d_pair_taps,
    warp_taps,
)

SMALL_H, SMALL_W = 12, 17


def _grid(h: int, w: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([us.ravel(), vs.ravel(), np.ones(h * w)], 0).astype(np.float64)


def _reference_warp(src: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Independent re-implementation of ``pfs1_warp_receiver.warp_rgb``."""
    h, w, c = src.shape
    flat = src.reshape(-1, c).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        src_h = np.linalg.inv(homography) @ _grid(h, w)
        z = src_h[2]
        su, sv = src_h[0] / z, src_h[1] / z
    valid = (
        np.isfinite(su) & np.isfinite(sv) & (z > 0)
        & (su >= 0) & (su <= w - 1) & (sv >= 0) & (sv <= h - 1)
    )
    su_c, sv_c = np.clip(su, 0.0, w - 1), np.clip(sv, 0.0, h - 1)
    x0, y0 = np.floor(su_c).astype(int), np.floor(sv_c).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (su_c - x0)[:, None], (sv_c - y0)[:, None]
    top = flat[y0 * w + x0] * (1 - wx) + flat[y0 * w + x1] * wx
    bot = flat[y1 * w + x0] * (1 - wx) + flat[y1 * w + x1] * wx
    return np.where(valid[:, None], top * (1 - wy) + bot * wy, flat).reshape(h, w, c)


def _shift_homography(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _perspective_homography() -> np.ndarray:
    return np.array([[1.03, 0.02, -1.7], [-0.01, 0.98, 2.3], [1e-4, 2e-4, 1.0]])


@pytest.fixture
def src() -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(0, 256, size=(SMALL_H, SMALL_W, 3)).astype(np.float64)


@pytest.mark.parametrize(
    "homography",
    [np.eye(3), _shift_homography(2.5, -1.25), _perspective_homography()],
)
def test_taps_reproduce_independent_bilinear_warp(src, homography):
    idx, w, _ = warp_taps(homography, _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W)
    got = apply_taps(idx, w, src)
    assert np.abs(got - _reference_warp(src, homography)).max() < 1e-12


def test_tap_rows_sum_to_one(src):
    _, w, _ = warp_taps(
        _perspective_homography(), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    assert np.abs(w.sum(0) - 1.0).max() < 1e-12


def test_invalid_region_is_the_identity_not_a_warp_read(src):
    """A large shift pushes most of the frame out of bounds; there the receiver
    returns the SOURCE pixel at the same index, which must show up as a unit tap."""
    homography = _shift_homography(500.0, 500.0)
    idx, w, valid = warp_taps(
        homography, _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    assert not valid.all(), "fixture must actually exercise the invalid branch"
    q = np.arange(SMALL_H * SMALL_W)
    assert np.array_equal(idx[0][~valid], q[~valid])
    assert np.allclose(w[0][~valid], 1.0)
    assert np.allclose(w[1:, ~valid], 0.0)
    assert np.allclose(apply_taps(idx, w, src).reshape(-1, 3)[~valid], src.reshape(-1, 3)[~valid])


def test_constant_image_is_preserved_by_the_warp(src):
    idx, w, _ = warp_taps(
        _perspective_homography(), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    const = np.full((SMALL_H, SMALL_W, 3), 37.0)
    assert np.abs(apply_taps(idx, w, const) - 37.0).max() < 1e-12


def test_adjoint_satisfies_the_inner_product_identity(src):
    """<M x, v> == <x, M^T v> — the definition of the transpose."""
    rng = np.random.default_rng(9)
    idx, w, _ = warp_taps(
        _perspective_homography(), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    v = rng.normal(size=(SMALL_H, SMALL_W, 3))
    lhs = float((apply_taps(idx, w, src) * v).sum())
    rhs = float((src * adjoint_taps(idx, w, v, height=SMALL_H, width=SMALL_W)).sum())
    assert abs(lhs - rhs) < 1e-9 * max(abs(lhs), 1.0)


def test_adjoint_is_channel_independent():
    rng = np.random.default_rng(11)
    idx, w, _ = warp_taps(
        _perspective_homography(), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    v = np.repeat(rng.normal(size=(SMALL_H, SMALL_W, 1)), 3, axis=2)
    three = adjoint_taps(idx, w, v, height=SMALL_H, width=SMALL_W)
    assert np.abs(three[..., 0] - three[..., 1]).max() < 1e-12
    assert np.abs(three[..., 0] - three[..., 2]).max() < 1e-12


def test_adjoint_defaults_to_the_cotangent_shape_not_the_camera_constants():
    """Regression: the default used to be (874,1164), so a small cotangent produced
    a full-size, mostly-zero result with no error -- a wrong answer that never
    raises.  The shape must now come from the cotangent."""
    idx, w, _ = warp_taps(np.eye(3), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W)
    out = adjoint_taps(idx, w, np.ones((SMALL_H, SMALL_W, 3)))
    assert out.shape == (SMALL_H, SMALL_W, 3)
    assert (CAMERA_H, CAMERA_W) != (SMALL_H, SMALL_W)


def test_adjoint_refuses_a_shape_that_disagrees_with_its_taps():
    idx, w, _ = warp_taps(np.eye(3), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W)
    with pytest.raises(ValueError, match="disagrees"):
        adjoint_taps(idx, w, np.ones((SMALL_H, SMALL_W, 3)), height=SMALL_H + 1)
    with pytest.raises(ValueError, match="taps cover"):
        adjoint_taps(idx, w, np.ones((SMALL_H + 1, SMALL_W, 3)))
    with pytest.raises(ValueError, match=r"\(H,W,C\)"):
        adjoint_taps(idx, w, np.ones((SMALL_H, SMALL_W)))


def test_adjoint_of_ones_recovers_the_column_mass():
    idx, w, _ = warp_taps(
        _perspective_homography(), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    col = adjoint_taps(
        idx, w, np.ones((SMALL_H, SMALL_W, 1)), height=SMALL_H, width=SMALL_W
    )
    assert abs(float(col.sum()) - SMALL_H * SMALL_W) < 1e-9


def test_d_column_weights_blind_count_is_230904():
    weights = d_column_weights()
    assert weights.shape == (CAMERA_H, CAMERA_W)
    assert int((weights == 0.0).sum()) == 230904


def test_d_column_weights_read_count_is_exactly_four_per_scorer_pixel():
    weights = d_column_weights()
    assert int((weights > 0.0).sum()) == 4 * SEG_H * SEG_W


def test_no_camera_pixel_is_read_twice():
    """Stride 2.276 > 2 means windows are disjoint: every column sum is <= 1."""
    assert d_column_weights().max() <= 1.0 + 1e-9


def test_d_column_weights_total_equals_scorer_pixel_count():
    assert abs(float(d_column_weights().sum()) - SEG_H * SEG_W) < 1e-6


def test_d_column_weights_agrees_with_ll1_blind_mask():
    from tac.optimization.ddm_ll1_window_solve import blind_mask

    assert np.array_equal(d_column_weights() == 0.0, blind_mask())


def test_blind_influence_mass_closure_and_photometric_scaling():
    rng = np.random.default_rng(2)
    weights = rng.random((SMALL_H, SMALL_W))
    blind = np.zeros((SMALL_H, SMALL_W), dtype=bool)
    blind[::3, ::4] = True
    idx, w, _ = warp_taps(
        _perspective_homography(), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W
    )
    one = blind_influence_mass(idx, w, weights, blind)
    two = blind_influence_mass(idx, w, weights, blind, photometric_a=-2.0)
    assert abs(one["total_mass"] - float(weights.sum())) < 1e-9
    assert abs(two["total_mass"] - 2.0 * float(weights.sum())) < 1e-9
    assert abs(two["blind_mass"] - 2.0 * one["blind_mass"]) < 1e-9
    assert abs(two["blind_mass_frac"] - one["blind_mass_frac"]) < 1e-12


def test_blind_influence_mass_is_zero_when_nothing_is_blind():
    weights = np.ones((SMALL_H, SMALL_W))
    idx, w, _ = warp_taps(np.eye(3), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W)
    row = blind_influence_mass(idx, w, weights, np.zeros((SMALL_H, SMALL_W), dtype=bool))
    assert row["blind_mass"] == 0.0
    assert row["blind_mass_frac"] == 0.0


def test_blind_influence_under_identity_warp_equals_the_blind_column_mass():
    """With M = I the composed operator is just D, so blind mass must be 0 for a
    blind set defined by zero D weight — the sanity anchor for the whole metric."""
    rng = np.random.default_rng(6)
    weights = rng.random((SMALL_H, SMALL_W))
    blind = weights < 0.25
    weights = np.where(blind, 0.0, weights)
    idx, w, _ = warp_taps(np.eye(3), _grid(SMALL_H, SMALL_W), height=SMALL_H, width=SMALL_W)
    assert blind_influence_mass(idx, w, weights, blind)["blind_mass"] == 0.0


class _StubDecoder:
    """Minimal stand-in for ``inflate_runner_v4d.Decoder`` (the receiver is vendored)."""

    def __init__(self, sel: int, beta: float, height: int, width: int) -> None:
        self.p_best = np.array([[1.0, 0.1, -0.1, 1e-3, 1e-4, 5e-4]])
        self.st_vals = np.array([0.08])
        self.st_idx = np.array([0])
        self.sel = np.array([sel])
        self.beta_mags = (0.0, beta)
        self.beta_idx = np.array([1 if beta else 0])
        self.K = np.array([[910.0, 0.0, 582.0], [0.0, 910.0, 437.0], [0.0, 0.0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        self.grid = _grid(height, width)
        self._far = (np.arange(height)[:, None] < height // 2) & np.ones((1, width), bool)
        self._alpha = (np.arange(height) / (height - 1.0))[:, None, None]


@pytest.fixture
def vendored_receiver(monkeypatch):
    """Provide ``pfs1_warp_receiver.pose_to_homography`` without the vendored tree.

    Byte-faithful copy of the receiver's own body (it is 6 lines of fixed geometry
    with no video-derived content), so the composition logic under test is exercised
    against the real homography, not a stub that could hide a wiring error.
    """
    import sys
    import types

    module = types.ModuleType("pfs1_warp_receiver")

    def _expmap_so3(omega):
        theta = float(np.linalg.norm(omega))
        k = np.array(
            [[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]]
        )
        if theta < 1e-12:
            return np.eye(3) + k
        return (
            np.eye(3)
            + (np.sin(theta) / theta) * k
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (k @ k)
        )

    def pose_to_homography(pose6, k_mat, k_inv, s_t, s_r, pitch):
        t = s_t * np.array([pose6[2], pose6[1], pose6[0]])
        rot = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]]))
        n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
        return k_mat @ (rot - np.outer(t, n) / 1.22) @ k_inv

    module.pose_to_homography = pose_to_homography
    monkeypatch.setitem(sys.modules, "pfs1_warp_receiver", module)
    return module


@pytest.mark.parametrize(
    ("sel", "beta", "n_taps"), [(0, 0.0, 4), (1, 0.0, 4), (0, 1.0, 8), (1, 1.0, 8)]
)
def test_v4d_pair_taps_shape_and_row_sums(sel, beta, n_taps, vendored_receiver):
    dec = _StubDecoder(sel, beta, SMALL_H, SMALL_W)
    idx, w, _ = v4d_pair_taps(dec, 0)
    assert idx.shape == (n_taps, SMALL_H * SMALL_W)
    assert np.abs(w.sum(0) - 1.0).max() < 1e-12


def test_v4d_pair_taps_selector_one_actually_splits_far_from_ground(vendored_receiver):
    """selector=1 composes TWO homographies on a row split; if the far/ground
    branch were dropped the two operators would be identical."""
    single = v4d_pair_taps(_StubDecoder(0, 0.0, SMALL_H, SMALL_W), 0)
    two_plane = v4d_pair_taps(_StubDecoder(1, 0.0, SMALL_H, SMALL_W), 0)
    far_rows = slice(0, (SMALL_H // 2) * SMALL_W)
    assert not np.allclose(single[1][:, far_rows], two_plane[1][:, far_rows])
    ground = slice((SMALL_H // 2) * SMALL_W, None)
    assert np.allclose(single[1][:, ground], two_plane[1][:, ground])


def test_v4d_pair_taps_beta_blend_weights_are_the_row_ramp(vendored_receiver):
    """The rung-A blend must be (1-alpha) on the first warp and alpha on the second,
    with alpha ramping 0->1 down the rows."""
    idx, w, _ = v4d_pair_taps(_StubDecoder(0, 1.0, SMALL_H, SMALL_W), 0)
    first = w[:4].sum(0).reshape(SMALL_H, SMALL_W)
    second = w[4:].sum(0).reshape(SMALL_H, SMALL_W)
    alpha = (np.arange(SMALL_H) / (SMALL_H - 1.0))[:, None]
    assert np.abs(second - alpha).max() < 1e-12
    assert np.abs(first - (1.0 - alpha)).max() < 1e-12
    assert idx.shape[0] == 8


def test_module_ships_nothing_video_derived():
    """rule-118 guard: the module must carry no learned/video-derived table."""
    import tac.optimization.ddm_bp2_blind_pose_actuator as mod

    for name in dir(mod):
        value = getattr(mod, name)
        if isinstance(value, np.ndarray):
            pytest.fail(f"module-level array {name} would be a shipped table")
    assert (CAMERA_H, CAMERA_W, SEG_H, SEG_W) == (874, 1164, 384, 512)
