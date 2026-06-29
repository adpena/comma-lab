# SPDX-License-Identifier: MIT
"""Tests for the Wyner-Ziv lane-centerline head-start (tac.boundary_math.lane_headstart).

NO-FAKE discipline: these tests verify BEHAVIOUR (the functions do the work their
names claim on real-shaped masks), not constants.  The residual round-trip and the
byte-serialization round-trip are bit-exact property checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import lane_headstart as lh


# ───────────────────────── LaneCenterline ─────────────────────────
def test_centerline_eff_degree():
    cl = lh.LaneCenterline("col_of_row", 0, 5, (0.0, 10.0), 1)  # col = 10 (constant+linear)
    assert cl.eff_degree == 1


def test_centerline_samples_rounds_poly():
    # col = 0.5*row + 2.0 over rows 0..3 -> [2, 2(2.5->2), 3, 4(3.5->4)] (banker's round)
    cl = lh.LaneCenterline("col_of_row", 0, 3, (0.5, 2.0), 0)
    s = cl.samples()
    assert s.tolist() == np.rint([2.0, 2.5, 3.0, 3.5]).astype(int).tolist()
    assert s.dtype == np.int64


# ───────────────────────── fit_centerlines ─────────────────────────
def _vertical_line_mask(h=80, w=60, col=30, r0=10, r1=70):
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, col] = True
    return m


def test_fit_empty_mask_returns_empty():
    assert lh.fit_centerlines(np.zeros((40, 40), dtype=bool)) == []


def test_fit_vertical_line_one_component_col_of_row():
    cls = lh.fit_centerlines(_vertical_line_mask(), degree=2, min_component_pixels=5)
    assert len(cls) == 1
    assert cls[0].axis == "col_of_row"  # row extent > col extent


def test_fit_min_component_pixels_filters_tiny():
    m = _vertical_line_mask(r0=10, r1=13)  # only 3 px
    assert lh.fit_centerlines(m, min_component_pixels=12) == []


def test_fit_horizontal_line_is_row_of_col():
    m = np.zeros((80, 120), dtype=bool)
    m[40, 10:100] = True
    cls = lh.fit_centerlines(m, min_component_pixels=5)
    assert len(cls) == 1
    assert cls[0].axis == "row_of_col"


# ───────────────────────── rasterize ─────────────────────────
def test_rasterize_reproduces_clean_vertical_line():
    m = _vertical_line_mask(col=30)
    cls = lh.fit_centerlines(m, degree=1, min_component_pixels=5, max_half_width=0)
    base = lh.rasterize_centerlines(cls, *m.shape)
    # a clean 1px vertical line at half_width 0 should be reproduced exactly
    assert np.array_equal(base, m)


def test_rasterize_half_width_paints_band():
    cl = lh.LaneCenterline("col_of_row", 0, 9, (0.0, 5.0), 2)  # col=5, hw=2
    base = lh.rasterize_centerlines([cl], 10, 10)
    assert base[0, 3:8].all()  # cols 3..7 painted (5±2)
    assert not base[0, 0:3].any()


def test_rasterize_empty_centerlines_is_blank():
    base = lh.rasterize_centerlines([], 10, 10)
    assert base.shape == (10, 10) and not base.any()


# ───────────────────────── residual pipeline (Wyner-Ziv X - Y) ─────────────────────────
def test_residual_signs():
    base = np.array([[1, 0, 1, 0]], dtype=bool)
    target = np.array([[1, 1, 0, 0]], dtype=bool)
    res = lh.compute_lane_residual(base, target)
    # col0 both 1 -> 0; col1 FN(target1,base0) -> +1; col2 FP(target0,base1) -> -1; col3 -> 0
    assert res.tolist() == [[0, 1, -1, 0]]
    assert res.dtype == np.int8


def test_residual_values_in_minus1_0_plus1():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, (30, 40)).astype(bool)
    target = rng.integers(0, 2, (30, 40)).astype(bool)
    res = lh.compute_lane_residual(base, target)
    assert set(np.unique(res).tolist()).issubset({-1, 0, 1})


def test_apply_residual_is_exact_inverse_random():
    rng = np.random.default_rng(1)
    for _ in range(20):
        base = rng.integers(0, 2, (24, 32)).astype(bool)
        target = rng.integers(0, 2, (24, 32)).astype(bool)
        res = lh.compute_lane_residual(base, target)
        recon = lh.apply_lane_residual(base, res)
        assert np.array_equal(recon, target)  # base + (target-base) == target, bit-exact


def test_residual_support_frac_equals_nonzero_over_size():
    base = np.zeros((10, 10), dtype=bool)
    target = np.zeros((10, 10), dtype=bool)
    target[0, :5] = True  # 5 FN flips out of 100
    res = lh.compute_lane_residual(base, target)
    assert lh.residual_support_frac(res) == pytest.approx(0.05)


def test_residual_support_equals_base_dseg_by_construction():
    # the nonzero residual fraction is EXACTLY the XOR(base,target) = base lane d_seg
    base = _vertical_line_mask(col=30)
    target = _vertical_line_mask(col=31)  # shifted 1px -> all flips
    res = lh.compute_lane_residual(base, target)
    xor_frac = float(np.logical_xor(base, target).sum()) / base.size
    assert lh.residual_support_frac(res) == pytest.approx(xor_frac)


# ───────────────────────── entropy + bytes ─────────────────────────
def test_shannon_entropy_constant_is_zero():
    assert lh._shannon_entropy_bits(np.zeros(50, dtype=np.int64)) == 0.0


def test_shannon_entropy_uniform_two_symbols_is_one_bit():
    v = np.array([0, 1] * 50, dtype=np.int64)
    assert lh._shannon_entropy_bits(v) == pytest.approx(1.0)


def test_estimate_base_bytes_keys_and_positive():
    m = _vertical_line_mask()
    cls = lh.fit_centerlines(m, min_component_pixels=5)
    est = lh.estimate_base_bytes([cls, cls, cls], n_target_frames=600)
    for k in (
        "parametric_bytes_600",
        "delta_entropy_bytes_600",
        "zlib_temporal_bytes_600",
        "achievable_iid_bytes_600",
        "achievable_iid_rate_term_600",
    ):
        assert k in est and est[k] >= 0.0
    # achievable is the min of the three concrete estimates
    assert est["achievable_iid_bytes_600"] <= est["parametric_bytes_600"] + 1e-6


# ───────────────────────── serialization round-trip (exact-preserving) ─────────────────────────
def test_serialize_is_deterministic():
    m = _vertical_line_mask()
    cls = lh.fit_centerlines(m, min_component_pixels=5)
    a = lh.serialize_centerlines_delta([cls, cls])
    b = lh.serialize_centerlines_delta([cls, cls])
    assert a == b and len(a) > 0


def test_serialize_deserialize_reproduces_base_exactly():
    rng = np.random.default_rng(2)
    h, w = 60, 50
    frames = []
    expected = []
    for _ in range(4):
        m = np.zeros((h, w), dtype=bool)
        col = int(rng.integers(10, 40))
        m[5:55, col] = True
        m[20:30, col + 1] = True  # a small bump so deltas are non-trivial
        cls = lh.fit_centerlines(m, min_component_pixels=5)
        frames.append(cls)
        expected.append(lh.rasterize_centerlines(cls, h, w))
    raw = lh.serialize_centerlines_delta(frames)
    decoded = lh.deserialize_and_rasterize(raw, h, w)
    assert len(decoded) == len(expected)
    for dec, exp in zip(decoded, expected):
        assert np.array_equal(dec, exp)  # exact-preserving claim is TESTED


# ───────────────────────── full driver ─────────────────────────
def test_build_lane_headstart_synthetic_roundtrip_and_recovery():
    # synthetic 3-frame argmax with a lane line we can mostly recover
    h, w = 80, 64
    lstars = np.full((3, h, w), 2, dtype=np.int64)  # Undrivable background
    for i in range(3):
        lstars[i, 10:70, 30 + i] = lh.LANE_CLASS  # a near-vertical lane line, drifting
    r = lh.build_lane_headstart(lstars, degree=2, min_component_pixels=5)
    assert r.n_frames == 3
    assert r.roundtrip_exact is True  # base + residual == lane on every frame
    assert 0.0 <= r.base_lane_dseg <= r.from_scratch_lane_dseg + 1e-9
    # a clean line is well recovered -> recovered_frac should be high
    assert r.recovered_frac > 0.5


def test_gauge_cost_cell_shape_and_pending_gpu():
    h, w = 60, 60
    lstars = np.full((2, h, w), 2, dtype=np.int64)
    lstars[:, 10:50, 30] = lh.LANE_CLASS
    r = lh.build_lane_headstart(lstars, degree=1, min_component_pixels=5)
    cell = lh.gauge_cost_cell(r)
    assert cell["gauge"] == "CONDITIONAL_ON_LANE_PRIOR"
    assert cell["learned_residual_cost"] == "PENDING-GPU"  # never fabricated
    assert cell["residual_dseg_target_sub015"] == lh.SUB015_LANE_DSEG
    assert "Wyner-Ziv" in cell["lens"]
