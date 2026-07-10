# SPDX-License-Identifier: MIT
"""Tests for the curve-relative signed-offset residual coder (v8 T2 Lever-2).

Proves the NO-FAKE contract: the coder REPRODUCES the boundary it claims (decode(encode)==input,
bit-for-bit) across col-param + row-param + multi-segment junction + off-support-exception cases,
and that the absolute-2-D baseline coder is itself bit-exact.  All synthetic (fast, deterministic);
a real-cache smoke is gated.  ``[macOS-CPU advisory · NON-PROMOTABLE]`` -- a coder moves no pointer.
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math import curve_relative_offset_coder as C

H, W = 64, 80


def _flat(rc: list[tuple[int, int]]) -> np.ndarray:
    return np.array([r * W + c for r, c in rc], dtype=np.int64)


# --------------------------------------------------------------------------- baseline coder
def test_absolute_2d_roundtrip_bit_exact():
    frames = [
        _flat([(3, 4), (3, 5), (10, 20), (11, 20)]),
        np.zeros(0, dtype=np.int64),
        _flat([(0, 0), (63, 79)]),
    ]
    blob = C.encode_absolute_2d(frames, H, W)
    gh, gw, dec = C.decode_absolute_2d(blob)
    assert (gh, gw) == (H, W)
    for want, got in zip(frames, dec):
        assert np.array_equal(np.unique(want), got)


# --------------------------------------------------------------------------- col-param curve (horizon)
def test_curve_relative_col_param_bit_exact():
    # horizon at row 30, residual pixels a few px above/below across columns
    y = np.full(W, 30, dtype=np.int64)
    curve = C.curve_from_column_function(y, seg_id=0)
    rc = [(30 + d, c) for c, d in zip(range(5, 40, 3), [-2, -1, 0, 1, 2, -1, 3, -2, 1, 0, -1, 2])]
    resid = _flat(rc)
    ch = C.chart_transform(resid, [curve], H, W)
    assert ch.exceptions.size == 0  # all columns on support
    assert int(np.abs(ch.n).max()) <= 3
    recon = C.reconstruct_from_chart(ch, [curve])
    assert np.array_equal(np.unique(resid), recon)
    gh, gw, dec = C.decode_curve_relative(C.encode_curve_relative([ch]), [curve])
    assert np.array_equal(np.unique(resid), dec[0])


# --------------------------------------------------------------------------- row-param + off-support
def test_curve_relative_row_param_and_exceptions():
    # vertical lane: column ~40 over rows 20..50
    cov = np.zeros((H, W), dtype=bool)
    for r in range(20, 51):
        cov[r, 39:42] = True
    curves = C.curves_from_coverage_mask(cov, axis="row", min_len=8)
    assert len(curves) >= 1 and curves[0].axis == "row"
    # residual near the lane (row 25..45) + one far-off px (off-support param) -> exception
    rc = [(r, 40 + d) for r, d in zip(range(25, 46, 2), [-1, 0, 1, 2, -2, 0, 1, -1, 3, 0, -1])]
    rc.append((5, 5))  # row 5 has no lane segment -> exception
    resid = _flat(rc)
    ch = C.chart_transform(resid, curves, H, W)
    assert ch.exceptions.size == 1
    assert _flat([(5, 5)])[0] in ch.exceptions
    gh, gw, dec = C.decode_curve_relative(C.encode_curve_relative([ch]), curves)
    assert np.array_equal(np.unique(resid), dec[0])


# --------------------------------------------------------------------------- junction / multi-valued
def test_junction_multi_segment_bit_exact():
    # two crossing lanes (X shape) -> curves_from_coverage must segment into monotone pieces
    cov = np.zeros((H, W), dtype=bool)
    for r in range(10, 50):
        c1 = 20 + (r - 10)  # rightward
        c2 = 60 - (r - 10)  # leftward
        cov[r, max(0, c1 - 1):c1 + 2] = True
        cov[r, max(0, c2 - 1):c2 + 2] = True
    curves = C.curves_from_coverage_mask(cov, axis="row", min_len=6)
    assert len(curves) >= 2  # the X is split into (at least) two monotone segments
    # residual straddling both arms
    rc = []
    for r in range(12, 48, 2):
        rc.append((r, 20 + (r - 10) + 1))
        rc.append((r, 60 - (r - 10) - 1))
    resid = _flat(rc)
    ch = C.chart_transform(resid, curves, H, W)
    recon = C.reconstruct_from_chart(ch, curves)
    assert np.array_equal(np.unique(resid), recon)
    gh, gw, dec = C.decode_curve_relative(C.encode_curve_relative([ch]), curves)
    assert np.array_equal(np.unique(resid), dec[0])


# --------------------------------------------------------------------------- multi-frame stream
def test_multi_frame_stream_bit_exact():
    y = np.full(W, 25, dtype=np.int64)
    curve = C.curve_from_column_function(y, seg_id=0)
    frames_resid = [
        _flat([(25 + d, c) for c, d in zip(range(0, 60, 4), [-1, 0, 1, 2, -2, 0, 1, -1, 3, 0, -1, 2, 1, -1, 0])]),
        np.zeros(0, dtype=np.int64),
        _flat([(24, 10), (26, 11), (25, 12)]),
    ]
    charts = [C.chart_transform(fr, [curve], H, W) for fr in frames_resid]
    blob = C.encode_curve_relative(charts)
    gh, gw, dec = C.decode_curve_relative(blob, [curve])
    for want, got in zip(frames_resid, dec):
        assert np.array_equal(np.unique(want), got)


# --------------------------------------------------------------------------- spectrum + savings sanity
def test_delta_s_spectrum_and_savings():
    # a smooth offset signal n(s) = small sinusoid -> low entropy, sparse Haar, curve-rel beats abs
    y = np.full(W, 30, dtype=np.int64)
    curve = C.curve_from_column_function(y, seg_id=0)
    cols = np.arange(0, 78)
    off = np.round(2.0 * np.sin(cols / 6.0)).astype(int)
    rc = [(30 + int(o), int(c)) for c, o in zip(cols, off)]
    resid = _flat(rc)
    res = C.measure_curve_relative([resid], [[curve]], H, W, edge_name="synthetic")
    assert res["curve_relative_bit_exact"] is True
    assert res["absolute_bit_exact"] is True
    spec = res["delta_s_spectrum"]
    assert spec["offset_abs_max_px"] <= 2
    # smooth bounded offset must code smaller than the absolute 2-D coords
    assert res["bytes_curve_relative"] < res["bytes_absolute_2d_baseline"]
    assert res["savings_ratio"] > 1.0


def test_haar_nterm_sparse_for_smooth_signal():
    smooth = np.round(3 * np.sin(np.arange(128) / 10.0)).astype(float)
    noisy = np.random.default_rng(0).integers(-3, 4, size=128).astype(float)
    assert C._haar_nterm_fraction(smooth) < C._haar_nterm_fraction(noisy)


def test_empty_and_single_curve_edge_cases():
    curve = C.curve_from_column_function(np.full(W, 20, dtype=np.int64))
    ch = C.chart_transform(np.zeros(0, dtype=np.int64), [curve], H, W)
    assert ch.seg_id.size == 0 and ch.exceptions.size == 0
    blob = C.encode_curve_relative([ch])
    gh, gw, dec = C.decode_curve_relative(blob, [curve])
    assert dec[0].size == 0
