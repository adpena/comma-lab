# SPDX-License-Identifier: MIT
"""Band-local operators must reproduce the producer byte-for-byte on real input.

research_only=true, score_claim=false. No n600 verdict is drawn here.
"""
import numpy as np
import pytest

from experiments.ddm_rbf1_band_local import axis_map, band_of, treat_band
from experiments.ddm_rbf1_boundary_probe import FIELD, RAW
from experiments.ddm_rbf1_boundary_treatments import camera_geometry, treat


def _real_frame(pair: int):
    if not RAW.is_file() or not FIELD.is_file():
        pytest.skip("pinned move-44 files unavailable")
    rgb = np.array(np.memmap(RAW, mode="r", dtype=np.uint8, shape=(600, 2, 874, 1164, 3))[pair, 1])
    tokens = np.array(np.memmap(FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))[pair])
    return rgb, tokens


@pytest.mark.parametrize("mode", ["guided", "ssaa", "sdf", "composition"])
def test_band_local_is_byte_identical_to_producer(mode):
    rgb, tokens = _real_frame(0)
    assert np.array_equal(treat(rgb, tokens, mode), treat_band(rgb, tokens, mode))


def test_band_index_matches_producer_geometry():
    rgb, tokens = _real_frame(0)
    _, _, _, _, guide, band = camera_geometry(tokens, rgb.shape[:2])
    by, bx, band_guide, ty, tx = band_of(tokens, rgb.shape[:2])
    assert np.array_equal(np.flatnonzero(band.ravel()), by * rgb.shape[1] + bx)
    assert np.array_equal(band_guide, guide[by, bx])
    assert by.size > 0


def test_axis_map_is_monotone_and_covers_every_token_row():
    rows = axis_map(874, 384)
    assert rows.min() == 0 and rows.max() == 383
    assert np.all(np.diff(rows) >= 0)
    assert np.unique(rows).size == 384


def test_class_renaming_leaves_band_local_output_invariant():
    rgb, tokens = _real_frame(0)
    permuted = np.array([3, 4, 0, 2, 1], dtype=np.uint8)[tokens]
    for mode in ("guided", "ssaa", "sdf"):
        assert np.array_equal(treat_band(rgb, tokens, mode), treat_band(rgb, permuted, mode))
