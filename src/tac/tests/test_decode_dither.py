# SPDX-License-Identifier: MIT
"""Tests for tac.witness_control.decode_dither (B19 decode-side seeded dither, P-DITHER).

Band provenance: DRAFT_OPTIMAL_STACK_v6 SS0.3/SS5/SS7c; gate driver
tools/witness_dither_decode_ab.py; probe memo probe_tau2_dither_20260708.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.decode_dither import (
    DEFAULT_AMP,
    DEFAULT_MODE,
    DEFAULT_SEED,
    bayer_matrix,
    dither_offset,
    dither_unit_field,
)


def test_bayer_matrix_is_permutation_and_recursive():
    b8 = bayer_matrix(8)
    assert b8.shape == (8, 8)
    assert sorted(b8.ravel().tolist()) == list(range(64))
    b2 = bayer_matrix(2)
    assert b2.tolist() == [[0, 2], [3, 1]]
    assert bayer_matrix(1).tolist() == [[0]]


def test_bayer_matrix_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        bayer_matrix(6)
    with pytest.raises(ValueError):
        bayer_matrix(0)


def test_dither_unit_field_deterministic_and_keyed():
    a = dither_unit_field(16, 24, "bayer8", 0xB19, 3, 1)
    b = dither_unit_field(16, 24, "bayer8", 0xB19, 3, 1)
    assert np.array_equal(a, b)  # same key -> identical
    c = dither_unit_field(16, 24, "bayer8", 0xB19, 4, 1)
    d = dither_unit_field(16, 24, "bayer8", 0xB19, 3, 0)
    assert not np.array_equal(a, c)  # pair key changes the field
    assert not np.array_equal(a, d)  # frame key changes the field
    assert a.shape == (16, 24, 3) and a.dtype == np.float32
    assert float(a.min()) >= 0.0 and float(a.max()) < 1.0


def test_dither_unit_field_bayer_values_are_cell_centered():
    f = dither_unit_field(8, 8, "bayer8", 1, 0, 0)
    # rolled tiling of (B+0.5)/64: each channel is a permutation of the 64 cell centers.
    for c in range(3):
        vals = sorted(f[..., c].ravel().tolist())
        expect = [(k + 0.5) / 64.0 for k in range(64)]
        assert np.allclose(vals, expect)


def test_dither_unit_field_white_mode_and_bad_mode():
    a = dither_unit_field(8, 8, "white", 2, 0, 0)
    b = dither_unit_field(8, 8, "white", 2, 0, 0)
    assert np.array_equal(a, b)
    assert float(a.min()) >= 0.0 and float(a.max()) < 1.0
    with pytest.raises(ValueError):
        dither_unit_field(8, 8, "pink", 2, 0, 0)


def test_dither_offset_amp_zero_is_identity_and_amp_bounds():
    z = dither_offset(8, 8, amp=0.0)
    assert not z.any() and z.dtype == np.float32
    o = dither_offset(8, 8, amp=1.0, mode="bayer8", seed=0xB19, pair_idx=0, frame_idx=1)
    assert float(np.abs(o).max()) <= 0.5  # amp=1 spans +/-0.5 = the rounding deadzone
    assert abs(float(o.mean())) < 0.05  # ~zero-mean
    half = dither_offset(8, 8, amp=0.5, mode="bayer8", seed=0xB19, pair_idx=0, frame_idx=1)
    assert np.allclose(half, 0.5 * o)


def test_defaults_are_the_preregistered_ab_config():
    assert DEFAULT_MODE == "bayer8"
    assert DEFAULT_AMP == 1.0
    assert DEFAULT_SEED == 0xB19
