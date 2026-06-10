# SPDX-License-Identifier: MIT
"""Behavior tests for the representation-audit probe helpers (Task #83).

These pin the NO-FAKE behavior of the audit's measurement primitives:
``_luma`` (BT.601), ``_delta_entropy_bytes`` (real zlib coded length),
``boundary_fraction`` (O(boundary) seg geometry), and
``partition_change_mask_bytes`` (the cross-frame partition-delta codec — the seg
analog of optical flow). Every assertion checks an actual computed quantity, not a
constant; a stub that returned a fixed value would FAIL.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[4]
for _p in (_REPO / "src", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_probe():
    path = _REPO / "tools" / "representation_audit_probe.py"
    spec = importlib.util.spec_from_file_location("representation_audit_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_probe()


# ---------------------------------------------------------------- _luma (BT.601)
def test_luma_pure_red_matches_bt601_coefficient():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[..., 0] = 255  # pure red
    y = PROBE._luma(frame)
    assert np.allclose(y, 0.299 * 255), y


def test_luma_pure_green_matches_bt601_coefficient():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[..., 1] = 255
    y = PROBE._luma(frame)
    assert np.allclose(y, 0.587 * 255), y


def test_luma_pure_blue_matches_bt601_coefficient():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[..., 2] = 255
    y = PROBE._luma(frame)
    assert np.allclose(y, 0.114 * 255), y


def test_luma_white_is_full_scale():
    frame = np.full((3, 4, 3), 255, dtype=np.uint8)
    y = PROBE._luma(frame)
    assert np.allclose(y, 255.0, atol=1e-9)


def test_luma_preserves_spatial_shape():
    frame = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    y = PROBE._luma(frame)
    assert y.shape == (5, 7)


# ----------------------------------------------- _delta_entropy_bytes (real zlib)
def test_delta_entropy_zero_delta_is_tiny():
    """A zero (perfectly redundant) delta compresses to near-nothing."""
    zeros = np.zeros((100, 100), dtype=np.int64)
    n = PROBE._delta_entropy_bytes(zeros)
    assert n < 200, n  # a constant-zero buffer is highly compressible


def test_delta_entropy_random_delta_is_large():
    """A high-entropy delta compresses poorly — larger than the redundant case."""
    rng = np.random.default_rng(1)
    noise = rng.integers(-100, 100, (100, 100)).astype(np.int64)
    n_noise = PROBE._delta_entropy_bytes(noise)
    n_zero = PROBE._delta_entropy_bytes(np.zeros((100, 100), dtype=np.int64))
    assert n_noise > n_zero * 10, (n_noise, n_zero)


def test_delta_entropy_returns_int_byte_count():
    n = PROBE._delta_entropy_bytes(np.zeros((4, 4), dtype=np.int64))
    assert isinstance(n, int) and n >= 0


def test_delta_entropy_clips_out_of_range():
    """Values outside [-127,127] are clipped (int8 payload) — does not raise."""
    big = np.full((10, 10), 500, dtype=np.int64)
    n = PROBE._delta_entropy_bytes(big)
    assert n > 0


# --------------------------------------------------- boundary_fraction (geometry)
def test_boundary_fraction_uniform_is_zero():
    a = np.zeros((20, 20), dtype=np.int64)
    assert PROBE.boundary_fraction(a) == 0.0


def test_boundary_fraction_checkerboard_is_high():
    """A checkerboard has nearly every pixel on a boundary."""
    a = np.indices((10, 10)).sum(axis=0) % 2
    bf = PROBE.boundary_fraction(a.astype(np.int64))
    assert bf > 0.8, bf


def test_boundary_fraction_half_split_is_low():
    """A single vertical split: only the seam column pixels are boundary."""
    a = np.zeros((10, 100), dtype=np.int64)
    a[:, 50:] = 1
    bf = PROBE.boundary_fraction(a)
    # one boundary column out of 100 -> ~1% boundary fraction
    assert 0.005 < bf < 0.05, bf


def test_boundary_fraction_rejects_non_2d():
    with pytest.raises(ValueError):
        PROBE.boundary_fraction(np.zeros((3, 3, 3), dtype=np.int64))


def test_boundary_fraction_real_partition_scale_is_small():
    """Real SegNet partitions have ~1.25% boundary fraction (measured); a synthetic
    few-region partition should likewise be O(1%), confirming O(boundary) << O(area)."""
    a = np.zeros((384, 512), dtype=np.int64)
    a[100:200, 100:300] = 1
    a[250:350, 50:200] = 2
    bf = PROBE.boundary_fraction(a)
    assert bf < 0.05, bf  # boundary is a sliver of the area


# ------------------------------------ partition_change_mask_bytes (seg flow codec)
def test_change_mask_static_partition_is_cheap():
    """No change between frames -> the change-mask payload is small."""
    a = np.zeros((384, 512), dtype=np.int64)
    a[100:200, 100:300] = 1
    n, frac = PROBE.partition_change_mask_bytes(a, a)
    assert frac == 0.0
    assert n < 300, n  # an all-zero change mask compresses away


def test_change_mask_changed_fraction_is_measured():
    a = np.zeros((100, 100), dtype=np.int64)
    b = a.copy()
    b[:10, :] = 1  # change 10% of pixels
    n, frac = PROBE.partition_change_mask_bytes(b, a)
    assert abs(frac - 0.10) < 1e-9, frac
    assert n > 0


def test_change_mask_scattered_costs_more_than_static():
    """The audit's key finding: a SCATTERED boundary change-mask is high-entropy and
    costs MORE than a static (no-change) partition. Pins the naive-delta-loses result."""
    a = np.zeros((200, 200), dtype=np.int64)
    rng = np.random.default_rng(3)
    b = a.copy()
    # scatter ~1% changed pixels (mimics the real ~1% cross-frame boundary motion)
    idx = rng.choice(a.size, size=a.size // 100, replace=False)
    flat = b.ravel()
    flat[idx] = 1
    b = flat.reshape(a.shape)
    n_scatter, _ = PROBE.partition_change_mask_bytes(b, a)
    n_static, _ = PROBE.partition_change_mask_bytes(a, a)
    assert n_scatter > n_static, (n_scatter, n_static)


def test_change_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        PROBE.partition_change_mask_bytes(
            np.zeros((4, 4), dtype=np.int64), np.zeros((4, 5), dtype=np.int64)
        )


def test_change_mask_returns_int_and_float():
    n, frac = PROBE.partition_change_mask_bytes(
        np.zeros((8, 8), dtype=np.int64), np.zeros((8, 8), dtype=np.int64)
    )
    assert isinstance(n, int) and isinstance(frac, float)
