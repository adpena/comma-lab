# SPDX-License-Identifier: MIT
"""Round-trip + accounting tests for the #307 contour-string flip coder
(tools/measure_contour_string_flip_coding.py).

These are LOSSLESS-CODER correctness tests (encode -> decode -> bit-exact), not score claims —
constructed flip maps are the right fixture for coder correctness (the n600 measurement itself
runs on the real byte-close render + frozen SegNet flips and decode-verifies every frame).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

mcs = pytest.importorskip("measure_contour_string_flip_coding")


def _roundtrip(flips, classes):
    enc = mcs.contour_encode_frames(flips, classes)
    dec_f, dec_c = mcs.contour_decode_frames(
        enc["streams"], len(flips), flips[0].shape[0], flips[0].shape[1])
    for i in range(len(flips)):
        assert np.array_equal(dec_f[i], flips[i]), f"frame {i} flip mismatch"
        assert np.array_equal(dec_c[i][flips[i]], classes[i][flips[i]]), f"frame {i} class mismatch"
    return enc


def test_roundtrip_empty_frame():
    f = [np.zeros((16, 24), dtype=bool)]
    c = [np.zeros((16, 24), dtype=np.int64)]
    enc = _roundtrip(f, c)
    assert enc["n_flips"] == 0 and enc["n_components"] == 0


def test_roundtrip_singletons_and_blobs():
    rng = np.random.default_rng(0)
    fm = np.zeros((32, 48), dtype=bool)
    fm[3, 5] = True                      # singleton
    fm[10, 10:20] = True                 # horizontal string
    fm[20:24, 30:34] = True              # 4x4 blob
    fm[rng.integers(0, 32, 10), rng.integers(0, 48, 10)] = True  # scattered
    cm = rng.integers(0, 5, size=fm.shape).astype(np.int64)
    enc = _roundtrip([fm], [cm])
    assert enc["n_flips"] == int(fm.sum())
    assert enc["total_bytes"] == sum(enc["stream_bytes"].values())


def test_roundtrip_multiframe_random():
    rng = np.random.default_rng(7)
    flips, classes = [], []
    for _ in range(4):
        fm = rng.random((24, 40)) < 0.08
        flips.append(fm)
        classes.append(rng.integers(0, 5, size=fm.shape).astype(np.int64))
    enc = _roundtrip(flips, classes)
    assert enc["n_flips"] == int(sum(f.sum() for f in flips))


def test_diagonal_string_is_cheap_per_flip():
    """A digitally-straight diagonal string must code far below 8 bits/flip once the
    straightness context adapts (the whole point of the chain-code coder)."""
    n = 200
    fm = np.zeros((256, 256), dtype=bool)
    for i in range(n):
        fm[10 + i, 10 + i] = True
    cm = np.ones(fm.shape, dtype=np.int64)
    enc = _roundtrip([fm], [cm])
    assert enc["n_components"] == 1
    # chain stream alone must be well under 1 bit/px-equivalent of the bz2 floor;
    # total (incl anchor+counts flush overheads) must beat 1 B/flip on this fixture.
    assert enc["b_contour"] < 0.5, enc


def test_component_size_histogram_partitions_flips():
    fm = np.zeros((16, 16), dtype=bool)
    fm[0, 0] = True          # size 1
    fm[4, 0:3] = True        # size 3
    fm[8, 0:6] = True        # size 6
    cm = np.zeros(fm.shape, dtype=np.int64)
    enc = _roundtrip([fm], [cm])
    assert enc["flips_by_comp_size"]["1"] == 1
    assert enc["flips_by_comp_size"]["2-3"] == 3
    assert enc["flips_by_comp_size"]["4-7"] == 6
    assert sum(enc["flips_by_comp_size"].values()) == enc["n_flips"]
    assert abs(enc["singleton_flip_frac"] - 1 / 10) < 1e-9


def test_varint_roundtrip():
    class _Buf:
        def __init__(self):
            self.stream = mcs.AdaptiveStream(256)

    vals = [0, 1, 127, 128, 300, 196607, 2**21 - 1]
    b = _Buf()
    for v in vals:
        mcs._write_varint(b.stream, v)
    data = b.stream.finish()
    dec = mcs.AdaptiveStreamDecoder(data, 256)
    got = [mcs._read_varint(dec) for _ in vals]
    assert got == vals
