# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.measure_g1_worldsheet_g3_cellcode import (
    _bit_costs_for_flip,
    extract_interclass_edges,
    npz_member_memmap,
    symmetric_chamfer_row,
)


def test_npz_member_memmap_reads_only_named_stored_member(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    expected = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    np.savez(path, wanted=expected, unrelated=np.ones((100, 100), dtype=np.float64))

    loaded = npz_member_memmap(path, "wanted")

    assert isinstance(loaded, np.memmap)
    assert loaded.dtype == expected.dtype
    assert np.array_equal(loaded, expected)


def test_interclass_edges_are_stratified_and_include_both_sides() -> None:
    labels = np.zeros((384, 512), dtype=np.uint8)
    labels[:, 256:] = 1

    edges = extract_interclass_edges(labels)

    road_lane = edges[(0, 1)]
    assert road_lane.shape == (384 * 2, 2)
    assert set(road_lane[:, 1].astype(int)) == {255, 256}
    assert all(len(edges[pair]) == 0 for pair in edges if pair != (0, 1))


def test_identity_transport_has_zero_symmetric_chamfer() -> None:
    points = np.array([[10.0, 20.0], [11.0, 20.0], [12.0, 21.0]])

    row = symmetric_chamfer_row(points, points, np.eye(3))

    assert row["presence_state"] == "both_present"
    assert row["symmetric_chamfer_px_finite_mean"] == 0.0
    assert row["median_residual_px_finite"] == 0.0
    assert row["event_fraction_gt_px"] == {"1": 0.0, "2": 0.0, "4": 0.0}


def test_death_counts_each_source_observation_once() -> None:
    source = np.array([[10.0, 20.0], [11.0, 20.0], [12.0, 21.0]])

    row = symmetric_chamfer_row(source, np.empty((0, 2)), np.eye(3))

    assert row["presence_state"] == "death"
    assert row["infinite_event_count"] == 3
    assert row["event_fraction_gt_px"] == {"1": 1.0, "2": 1.0, "4": 1.0}


def test_causal_cell_priors_are_finite_and_use_only_predecessors() -> None:
    labels = np.zeros((2, 384, 512), dtype=np.uint8)
    labels[1, 10, 9] = 2
    labels[1, 9, 10] = 2
    labels[0, 10, 10] = 2

    costs = _bit_costs_for_flip(
        labels,
        pair=1,
        row=10,
        col=10,
        target=2,
        baseline=0,
    )

    assert costs["uniform_5ary"] == np.log2(5.0)
    assert costs["uniform_4ary_excluding_baseline"] == 2.0
    assert costs["spatial_temporal_laplace"] < costs["spatial_potts_laplace"] < 2.0
    assert costs["temporal_same_site_laplace"] < 2.0
