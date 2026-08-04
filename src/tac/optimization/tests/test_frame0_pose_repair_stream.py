# SPDX-License-Identifier: MIT
"""Tests for the base-agnostic frame_0 pose-repair stream (ddm_bz1's reusable general-mechanism
section).  These verify carriage correctness in ISOLATION -- no scorer, no seg base -- exactly the
property that makes the section swappable across seg bases (ep854 / #827 / phase field)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.frame0_pose_repair_stream import (
    MAGIC,
    apply_pose_repair_scorer,
    dct_atoms,
    decode_pose_repair_stream,
    encode_pose_repair_stream,
    section_ledger,
)

K = 4
H, W = 384, 512


def _rng_coefs(pairs, k=K, seed=0):
    rng = np.random.default_rng(seed)
    # magnitudes in the measured range (ddm_bz1 cmax 158-282); DC largest
    out = {}
    for p in pairs:
        c = rng.integers(-300, 301, size=(3, k * k)).astype(np.int16)
        out[p] = c
    return out


def test_round_trip_exact():
    coefs = _rng_coefs([0, 20, 48, 261, 471])
    blob = encode_pose_repair_stream(coefs, k=K, seg_h=H, seg_w=W)
    assert blob[:8] == MAGIC
    sec = decode_pose_repair_stream(blob)
    assert sec.k == K and sec.seg_h == H and sec.seg_w == W
    assert set(sec.coefs) == set(coefs)
    for p in coefs:
        assert np.array_equal(sec.coefs[p], coefs[p])


def test_ledger_closes_and_counts_bytes():
    coefs = _rng_coefs([1, 2, 3])
    blob = encode_pose_repair_stream(coefs, k=K, seg_h=H, seg_w=W)
    led = section_ledger(blob)
    assert led["counted_bytes"] == len(blob)          # closes exactly on the whole section
    assert led["n_pairs_repaired"] == 3
    assert led["k"] == K
    assert led["bytes_per_pair_raw"] == K * K * 3 * 2  # 96 B/pair at k=4


def test_apply_matches_reference_float_pipeline():
    """apply_pose_repair_scorer must be byte-identical to round(clamp(base + coef@A))."""
    atoms = dct_atoms(K, H, W)
    A = atoms.reshape(K * K, -1).astype(np.float32)
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(3, H, W)).astype(np.float32)
    coef = rng.integers(-50, 51, size=(3, K * K)).astype(np.int16)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # macOS Accelerate quirk
        delta = (coef.astype(np.float32) @ A).reshape(3, H, W)
    ref = np.rint(np.clip(base + delta, 0, 255)).astype(np.uint8).transpose(1, 2, 0)
    got = apply_pose_repair_scorer(base, coef, atoms)
    assert np.array_equal(got, ref)
    # accepts HWC base too
    got_hwc = apply_pose_repair_scorer(base.transpose(1, 2, 0), coef, atoms)
    assert np.array_equal(got_hwc, ref)


def test_all_zero_pair_omission_is_zero_rate():
    """A pair with no damage carries 0 bytes by being omitted -- the correct rate for no repair."""
    coefs = _rng_coefs([5, 9])
    blob_two = encode_pose_repair_stream(coefs, k=K, seg_h=H, seg_w=W)
    blob_one = encode_pose_repair_stream({5: coefs[5]}, k=K, seg_h=H, seg_w=W)
    assert len(blob_one) < len(blob_two)
    assert section_ledger(blob_one)["n_pairs_repaired"] == 1


def test_int16_range_enforced():
    with pytest.raises(ValueError):
        encode_pose_repair_stream({0: np.full((3, K * K), 40000, np.int32)},
                                  k=K, seg_h=H, seg_w=W)


def test_coder_never_over_counts():
    """The section chooses the smallest supported lossless coder, so it never over-counts vs raw."""
    coefs = _rng_coefs(list(range(30)), seed=3)
    blob = encode_pose_repair_stream(coefs, k=K, seg_h=H, seg_w=W)
    raw = 8 + 9 + 4 * 30 + 30 * K * K * 3 * 2
    assert len(blob) <= raw


def test_forced_lossless_coders_round_trip_exact():
    coefs = _rng_coefs([3, 7, 11], seed=11)
    for coder in ("raw_int16", "lzma1_raw", "brotli_q11", "pair_bitpack"):
        blob = encode_pose_repair_stream(coefs, k=K, seg_h=H, seg_w=W, coder=coder)
        sec = decode_pose_repair_stream(blob)
        assert sec.coder_name == coder
        for p in coefs:
            assert np.array_equal(sec.coefs[p], coefs[p])
