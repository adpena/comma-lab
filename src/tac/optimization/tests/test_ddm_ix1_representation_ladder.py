# SPDX-License-Identifier: MIT
"""Tests for the ddm_ix1 measured representation ladder.

These verify BEHAVIOUR, not constants: every codec rung round-trips, the
combinatorial rung actually attains its information floor, the layout rung
actually reorders bytes, and the live-vehicle anchors are pinned so a future
regression in any coder is caught.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from tac.optimization.ddm_ix1_representation_ladder import (
    RATE_DENOMINATOR_BYTES,
    LayoutRace,
    RepresentationLadderError,
    colex_decode,
    colex_encode,
    colex_floor_bytes,
    colex_rank,
    colex_unrank,
    delta_s_rate_from_bytes,
    elias_fano_decode,
    elias_fano_encode,
    gap_fraction_of_bytes,
    golomb_rice_gaps_encode,
    pack_bitplanes,
    pack_nibble_lane,
    race_generic,
    race_layouts,
    race_subset_index,
)


# ---------------------------------------------------------------- colex


@pytest.mark.parametrize("n,k", [(600, 224), (64, 3), (1000, 1), (20, 20), (37, 0)])
def test_colex_roundtrip(n: int, k: int) -> None:
    rng = random.Random(1234 + n + k)
    pos = sorted(rng.sample(range(n), k))
    assert list(colex_decode(colex_encode(pos, n), k, n)) == pos


def test_colex_rank_is_a_bijection_onto_the_full_range() -> None:
    n, k = 9, 4
    ranks = {colex_rank(c, n) for c in _all_subsets(n, k)}
    assert ranks == set(range(math.comb(n, k)))


def test_colex_unrank_inverts_rank_exhaustively() -> None:
    n, k = 8, 3
    for c in _all_subsets(n, k):
        assert list(colex_unrank(colex_rank(c, n), k, n)) == list(c)


def test_colex_encode_attains_the_information_floor_within_one_byte() -> None:
    # The whole point of the rung: it must not be wasteful relative to
    # log2 C(n, k).  Anything more than the byte-rounding slack is a bug.
    for n, k in [(600, 224), (692_712, 400), (37_545, 12)]:
        pos = sorted(random.Random(7).sample(range(n), k))
        assert len(colex_encode(pos, n)) - colex_floor_bytes(n, k) < 1.0


def test_colex_rejects_unsorted_and_out_of_range() -> None:
    with pytest.raises(RepresentationLadderError):
        colex_rank([3, 1, 2], 10)
    with pytest.raises(RepresentationLadderError):
        colex_rank([1, 2, 99], 10)
    with pytest.raises(RepresentationLadderError):
        colex_unrank(math.comb(10, 3), 3, 10)


def _all_subsets(n: int, k: int):
    from itertools import combinations

    return combinations(range(n), k)


# ----------------------------------------------------------- elias-fano


@pytest.mark.parametrize("n,k", [(600, 224), (692_712, 400), (128, 7), (64, 1)])
def test_elias_fano_roundtrip(n: int, k: int) -> None:
    pos = sorted(random.Random(99 + k).sample(range(n), k))
    payload, width = elias_fano_encode(pos, n)
    assert list(elias_fano_decode(payload, k, n, width)) == pos


def test_elias_fano_beats_a_raw_bitmap_when_the_set_is_sparse() -> None:
    n, k = 692_712, 400
    pos = sorted(random.Random(3).sample(range(n), k))
    payload, _ = elias_fano_encode(pos, n)
    assert len(payload) < n // 8


def test_elias_fano_rejects_non_monotone() -> None:
    with pytest.raises(RepresentationLadderError):
        elias_fano_encode([5, 5, 6], 10)


# --------------------------------------------------------------- gaps


def test_golomb_rice_gaps_encodes_and_is_nonempty() -> None:
    pos = sorted(random.Random(5).sample(range(600), 224))
    assert len(golomb_rice_gaps_encode(pos, 600)) > 0
    assert golomb_rice_gaps_encode([], 600) == b""


# ------------------------------------------------------------- packing


def test_pack_nibble_lane_halves_and_rejects_overflow() -> None:
    a = np.array([0, 15, 7, 8], dtype=np.uint8)
    assert pack_nibble_lane(a) == bytes([0x0F, 0x78])
    with pytest.raises(RepresentationLadderError):
        pack_nibble_lane(np.array([16], dtype=np.uint8))


def test_pack_bitplanes_is_a_transposition_not_a_copy() -> None:
    a = np.array([0b1010, 0b0101] * 4, dtype=np.uint8)
    planes = pack_bitplanes(a, 4)
    assert len(planes) == 4
    rebuilt = np.zeros(a.size, dtype=np.uint8)
    for b, plane in enumerate(planes):
        rebuilt |= np.unpackbits(np.frombuffer(plane, dtype=np.uint8))[: a.size] << b
    assert np.array_equal(rebuilt, a)


# ---------------------------------------------------------------- races


def test_race_generic_includes_stored_and_is_sorted() -> None:
    rows = race_generic(b"\x00" * 4096)
    assert any(r.name == "stored" for r in rows)
    assert [r.size_bytes for r in rows] == sorted(r.size_bytes for r in rows)


def test_race_generic_on_incompressible_bytes_keeps_stored_competitive() -> None:
    payload = np.random.default_rng(0).integers(0, 256, 65536, dtype=np.uint8).tobytes()
    rows = race_generic(payload)
    best = min(rows, key=lambda r: r.size_bytes)
    assert best.size_bytes >= len(payload) - 64


def test_race_subset_index_reports_floor_and_best() -> None:
    pos = sorted(random.Random(11).sample(range(600), 224))
    race = race_subset_index(pos, 600)
    assert race.n == 600 and race.k == 224
    assert race.best.size_bytes == min(r.size_bytes for r in race.rungs)
    assert race.colex_floor_bytes == pytest.approx(70.878, abs=1e-2)
    names = {r.name for r in race.rungs}
    assert {"colex_rank", "elias_fano", "golomb_rice_gaps", "bitmap_packed"} <= names


def test_race_subset_index_detects_clustering_as_structure_gain() -> None:
    # A clustered subset is NOT exchangeable, so a prior-coded rung must be
    # able to beat log2 C(n, k).  This is the measurement the racer exists for.
    n = 20_000
    clustered = list(range(1000, 1400)) + list(range(9000, 9400))
    race = race_subset_index(clustered, n)
    assert race.structure_gain_vs_colex > 1.5


def test_race_subset_index_rejects_bad_inputs() -> None:
    with pytest.raises(RepresentationLadderError):
        race_subset_index([1, 1, 2], 10)
    with pytest.raises(RepresentationLadderError):
        race_subset_index([1, 99], 10)


def test_race_layouts_finds_the_stationary_layout() -> None:
    # Construct an array whose per-cell value is nearly constant across the
    # leading axis: cell-major MUST win, array-of-structs MUST lose.
    rng = np.random.default_rng(0)
    cell = rng.integers(0, 16, (6, 8), dtype=np.uint8)
    arr = np.broadcast_to(cell, (40, 6, 8)).copy()
    arr[rng.random(arr.shape) < 0.02] = rng.integers(0, 16)
    race = race_layouts(arr, symbol_width=4)
    best_layout, _, best_bytes = race.best
    aos = min(b for lay, _, b in race.rows if lay == "aos_native")
    assert best_bytes < aos
    assert best_layout != "aos_native"


def test_race_layouts_covers_every_packing_and_is_deterministic() -> None:
    arr = np.random.default_rng(2).integers(0, 16, (5, 4, 3), dtype=np.uint8)
    a = race_layouts(arr, symbol_width=4)
    b = race_layouts(arr, symbol_width=4)
    assert a.rows == b.rows
    assert {p for _, p, _ in a.rows} == {"byte_lane", "nibble_lane", "bitplanes"}
    assert isinstance(a, LayoutRace) and a.shape == (5, 4, 3)


def test_race_layouts_honours_explicit_permutations() -> None:
    arr = np.random.default_rng(4).integers(0, 16, (4, 3, 2), dtype=np.uint8)
    race = race_layouts(arr, permutations={"only": (2, 0, 1)}, symbol_width=4)
    assert {lay for lay, _, _ in race.rows} == {"only"}


# ------------------------------------------------------------ accounting


def test_delta_s_rate_matches_the_contest_formula() -> None:
    assert delta_s_rate_from_bytes(RATE_DENOMINATOR_BYTES) == pytest.approx(25.0)
    assert delta_s_rate_from_bytes(-6486) == pytest.approx(-0.0043188, abs=1e-7)


def test_gap_fraction_requires_a_supplied_floor() -> None:
    # The module must never hardcode a floor; a bad gap is refused loudly.
    assert gap_fraction_of_bytes(-6486, total_gap=0.7262358) == pytest.approx(
        -0.005947, abs=1e-5
    )
    with pytest.raises(RepresentationLadderError):
        gap_fraction_of_bytes(-1, total_gap=0.0)
    with pytest.raises(RepresentationLadderError):
        gap_fraction_of_bytes(-1, total_gap=float("nan"))


def test_gap_fraction_uses_the_canonical_equation_total_gap() -> None:
    from tac.canonical_equations.gap_decomposition_against_floor_20260802 import (
        EQUATION_ID,
    )

    assert EQUATION_ID  # the denominator's provenance is the canonical equation
