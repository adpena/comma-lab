# SPDX-License-Identifier: MIT
"""Tests for the #401 blind-coordinate rate lever (tac.through_r.blind_coordinate).

Covers: mask correctness (count, subgrid, inclusion-exclusion, edge rows/cols, exact-zero
weights), bit-identity-through-R (random/max/zero fill), generic fill determinism +
data-independence + non-blind preservation, retained sub-grid extraction, real byte delta,
error handling, and the rule-118 boundary comment guard (hide-data-in-code NO-FAKE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.through_r.blind_coordinate import (
    BlindCoordinateError,
    apply_blind_fill,
    bit_identity_report,
    blind_fraction,
    build_blind_mask,
    extract_retained_subgrid,
    generic_inpaint_fill,
    measure_byte_delta,
)
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W, RGB_CHANNELS

# The DERIVED constants (verified against the real bilinear resize kernel).
EXPECTED_BLIND_PX = 230904
EXPECTED_BLIND_ROWS = 106
EXPECTED_BLIND_COLS = 140
RETAINED_HW = (768, 1024)


def _synthetic_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.uint8)


# ---------------------------------------------------------------- mask correctness
def test_blind_pixel_count_matches_derivation():
    bm = build_blind_mask()
    assert bm.n_blind == EXPECTED_BLIND_PX
    assert bm.n_total == CAMERA_H * CAMERA_W


def test_blind_rows_and_cols_counts():
    bm = build_blind_mask()
    assert int(bm.blind_rows.sum()) == EXPECTED_BLIND_ROWS
    assert int(bm.blind_cols.sum()) == EXPECTED_BLIND_COLS


def test_retained_subgrid_is_regular_product():
    bm = build_blind_mask()
    assert bm.retained_hw == RETAINED_HW
    # retained = (non-blind rows) x (non-blind cols), exact complement of blind
    assert RETAINED_HW[0] * RETAINED_HW[1] == bm.n_retained
    assert bm.n_retained + bm.n_blind == bm.n_total


def test_inclusion_exclusion_identity():
    bm = build_blind_mask()
    r = int(bm.blind_rows.sum())
    c = int(bm.blind_cols.sum())
    assert r * CAMERA_W + c * CAMERA_H - r * c == bm.n_blind


def test_edge_rows_and_cols_not_blind():
    # first/last camera row+col ARE read (interior samples get skipped, not the edges).
    bm = build_blind_mask()
    assert not bm.blind_rows[0]
    assert not bm.blind_rows[CAMERA_H - 1]
    assert not bm.blind_cols[0]
    assert not bm.blind_cols[CAMERA_W - 1]


def test_mask_is_row_or_col_structure():
    bm = build_blind_mask()
    expected = bm.blind_rows[:, None] | bm.blind_cols[None, :]
    assert np.array_equal(bm.mask, expected)


def test_blind_weights_are_exactly_zero_not_epsilon():
    # a blind camera index must have EXACTLY 0.0 kernel weight (bilinear), not a tiny epsilon.
    from tac.through_r.flip_inverse import resize_matrix_1d
    from tac.through_r.resolution_chain import SEG_H

    down_col = resize_matrix_1d(CAMERA_H, SEG_H, "bilinear", align_corners=False)
    maxabs = np.abs(down_col).max(axis=0)
    blind = maxabs == 0.0
    assert blind.sum() == EXPECTED_BLIND_ROWS
    # the smallest NONZERO weight is far from zero (clean separation, no epsilon blindness)
    assert maxabs[maxabs > 0].min() > 1e-6


def test_mask_is_cached_same_object():
    assert build_blind_mask() is build_blind_mask()


def test_blind_fraction_report_schema():
    frac = blind_fraction()
    assert frac["n_blind_px"] == EXPECTED_BLIND_PX
    assert frac["retained_subgrid_hw"] == list(RETAINED_HW)
    assert abs(frac["blind_fraction"] - EXPECTED_BLIND_PX / (CAMERA_H * CAMERA_W)) < 1e-12
    assert frac["inclusion_exclusion_check"] == EXPECTED_BLIND_PX


# ---------------------------------------------------------------- bit-identity through R
@pytest.mark.parametrize("fill_mode", ["random", "max", "zero"])
def test_bit_identity_through_R_arbitrary_fill(fill_mode):
    # ARBITRARY blind content leaves BOTH scorer inputs bit-for-bit identical.
    f0 = np.stack([_synthetic_frame(1), _synthetic_frame(2)])
    f1 = np.stack([_synthetic_frame(3), _synthetic_frame(4)])
    r = bit_identity_report(f0, f1, seed=7, fill_mode=fill_mode)
    assert r.all_bit_identical
    assert r.max_abs_diff_pose == 0.0
    assert r.max_abs_diff_seg == 0.0
    assert r.n_failures == 0


def test_bit_identity_only_blind_pixels_matter():
    # sanity: changing a NON-blind pixel DOES change the scorer input (mask is not vacuous).
    from tac.through_r.blind_coordinate import _scorer_inputs

    bm = build_blind_mask()
    f0 = _synthetic_frame(11)
    f1 = _synthetic_frame(12)
    p0, s0 = _scorer_inputs(f0, f1)
    f1b = f1.copy()
    # pick a definitely-non-blind pixel (edge, always read)
    assert not bm.mask[0, 0]
    f1b[0, 0] = (255 - f1b[0, 0].astype(int)).astype(np.uint8)
    import torch

    p1, s1 = _scorer_inputs(f0, f1b)
    assert not (torch.equal(s0, s1))  # SegNet reads frame1 -> must differ


# ---------------------------------------------------------------- generic fill
def test_inpaint_preserves_nonblind_pixels():
    bm = build_blind_mask()
    frame = _synthetic_frame(21)
    recon = generic_inpaint_fill(extract_retained_subgrid(frame, bm), bm)
    assert recon.shape == frame.shape
    assert np.array_equal(recon[~bm.mask], frame[~bm.mask])


def test_inpaint_is_deterministic():
    bm = build_blind_mask()
    sub = extract_retained_subgrid(_synthetic_frame(22), bm)
    assert np.array_equal(generic_inpaint_fill(sub, bm), generic_inpaint_fill(sub, bm))


def test_inpaint_is_data_independent_of_blind_pixels():
    # the receiver fill depends ONLY on the retained sub-grid: two frames that agree on the
    # retained set but differ arbitrarily on blind pixels produce the SAME reconstruction.
    bm = build_blind_mask()
    base = _synthetic_frame(23)
    other = base.copy()
    rng = np.random.default_rng(99)
    other[bm.mask] = rng.integers(0, 256, size=(bm.n_blind, RGB_CHANNELS), dtype=np.uint8)
    rec_a = generic_inpaint_fill(extract_retained_subgrid(base, bm), bm)
    rec_b = generic_inpaint_fill(extract_retained_subgrid(other, bm), bm)
    assert np.array_equal(rec_a, rec_b)


def test_apply_blind_fill_modes():
    bm = build_blind_mask()
    frame = _synthetic_frame(24)
    # scalar fill
    filled = apply_blind_fill(frame, 200, bm)
    assert np.all(filled[bm.mask] == 200)
    assert np.array_equal(filled[~bm.mask], frame[~bm.mask])
    # None => generic reconstruction; non-blind untouched
    recon = apply_blind_fill(frame, None, bm)
    assert np.array_equal(recon[~bm.mask], frame[~bm.mask])


def test_extract_retained_subgrid_shape_and_values():
    bm = build_blind_mask()
    frame = _synthetic_frame(25)
    sub = extract_retained_subgrid(frame, bm)
    assert sub.shape == (RETAINED_HW[0], RETAINED_HW[1], RGB_CHANNELS)
    rr, cc = bm.retained_rows, bm.retained_cols
    assert np.array_equal(sub, frame[np.ix_(rr, cc)])


# ---------------------------------------------------------------- byte delta
def test_measure_byte_delta_positive_and_consistent():
    frames = np.stack([_synthetic_frame(31), _synthetic_frame(32)])
    bd = measure_byte_delta(frames)
    assert bd.n_frames == 2
    assert bd.bytes_retained_mean < bd.bytes_full_mean
    assert bd.byte_delta_mean > 0
    assert 0.0 < bd.delta_fraction_mean < 1.0
    assert abs(bd.blind_fraction - EXPECTED_BLIND_PX / (CAMERA_H * CAMERA_W)) < 1e-12


# ---------------------------------------------------------------- error handling
def test_wrong_shape_frame_raises():
    with pytest.raises(BlindCoordinateError):
        extract_retained_subgrid(np.zeros((10, 10, 3), np.uint8))


def test_bit_identity_empty_refused():
    empty = np.zeros((0, CAMERA_H, CAMERA_W, RGB_CHANNELS), np.uint8)
    with pytest.raises(BlindCoordinateError):
        bit_identity_report(empty, empty)


def test_byte_delta_wrong_shape_refused():
    with pytest.raises(BlindCoordinateError):
        measure_byte_delta(np.zeros((2, 10, 10, 3), np.uint8))


def test_generic_inpaint_wrong_subgrid_shape_refused():
    with pytest.raises(BlindCoordinateError):
        generic_inpaint_fill(np.zeros((5, 5, 3), np.uint8))


# ---------------------------------------------------------------- rule-118 boundary guard
def test_rule_118_boundary_comment_present():
    # NO-FAKE #6/#7: the module MUST document the rule-118 free/counted boundary so a future
    # edit cannot silently smuggle a video-derived blind table into the "free" algorithm.
    src = (Path(__file__).resolve().parents[1] / "blind_coordinate.py").read_text()
    assert "rule-118 boundary" in src
    assert "hide-data-in-code" in src
    assert "COUNTED in archive.zip" in src
