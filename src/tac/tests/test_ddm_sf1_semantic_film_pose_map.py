"""Structural tests for the DDM-SF1 semantic/FiLM row-group pose map.

These pin the parts that decide a verdict and that a future edit could silently break: the row
partition, the reproduction of the shipped selector's own ordering, the never-a-prefix sampling
rule, and the exact upstream score arithmetic.  They deliberately do not touch the 3.6 GB raws or
the scorer -- those are covered by the tool's own in-run controls (bit-identical re-render,
frame_0 invariance at n600, and the null zero-perturbation group).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "experiments" / "ddm_sf1_semantic_film_pose_map.py"


def _load():
    spec = importlib.util.spec_from_file_location("_sf1_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sf1_under_test"] = module
    spec.loader.exec_module(module)
    return module


sf1 = _load()


@pytest.fixture
def norms():
    rng = np.random.default_rng(7)
    return {name: rng.random(sf1.FILM_ROWS) for name in sf1.PRUNE_NAMES}


def test_prune_order_matches_the_shipped_selector(norms):
    """The shipped packer sorts by descending row norm with the index as the tie-break."""
    for name in sf1.PRUNE_NAMES:
        values = norms[name]
        expected = sorted(range(len(values)),
                          key=lambda index: (-float(values[index]), index))
        assert sf1.prune_order(values) == expected


def test_prune_order_tie_break_is_the_index():
    flat = np.ones(5)
    assert sf1.prune_order(flat) == [0, 1, 2, 3, 4]


def test_keep87_selection_drops_25_rows_per_tensor(norms):
    selection = sf1.keep87_selection(norms)
    assert set(selection) == set(sf1.PRUNE_NAMES)
    for name, rows in selection.items():
        assert len(rows) == sf1.FILM_ROWS - round(sf1.FILM_ROWS * 87 / 100.0)
        assert len(rows) == 25
        # the dropped rows are exactly the smallest-norm ones
        kept = set(range(sf1.FILM_ROWS)) - set(rows)
        assert min(norms[name][list(kept)]) >= max(norms[name][rows])


def test_mechanism_groups_partition_every_film_row_exactly_once(norms):
    groups = sf1.mechanism_groups(norms)
    assert len(groups) == 18
    seen: dict[str, list[int]] = {name: [] for name in sf1.PRUNE_NAMES}
    for group in groups:
        assert group["row_count"] == 32
        assert len(group["rows"]) == 1, "a mechanism group touches exactly one tensor"
        for name, rows in group["rows"].items():
            seen[name].extend(rows)
    for name in sf1.PRUNE_NAMES:
        assert sorted(seen[name]) == list(range(sf1.FILM_ROWS))


def test_mechanism_groups_split_scale_from_shift(norms):
    for group in sf1.mechanism_groups(norms):
        rows = next(iter(group["rows"].values()))
        in_scale = [r < sf1.FILM_SCALE_ROWS for r in rows]
        assert all(in_scale) or not any(in_scale), "a group must not straddle scale and shift"
        assert ("_scale_" in group["group_id"]) == all(in_scale)


def test_selector_alternatives_all_have_the_incumbent_cardinality(norms):
    groups = sf1.selector_groups(norms, (11, 22))
    ids = [g["group_id"] for g in groups]
    assert "sel_mp2_keep87_lowest_norm" in ids
    assert "sel_highest_norm_anticontrol" in ids
    assert "sel_random_seed11" in ids and "sel_random_seed22" in ids
    counts = {g["row_count"] for g in groups}
    assert counts == {75}, "every alternative must buy the same bytes as the incumbent"


def test_anticontrol_is_disjoint_from_the_incumbent(norms):
    groups = {g["group_id"]: g for g in sf1.selector_groups(norms, ())}
    incumbent = groups["sel_mp2_keep87_lowest_norm"]["rows"]
    anti = groups["sel_highest_norm_anticontrol"]["rows"]
    for name in sf1.PRUNE_NAMES:
        assert not set(incumbent[name]) & set(anti[name])


def test_pair_subset_is_seeded_random_and_never_a_prefix():
    subset = sf1._pair_subset(20260817, 120)
    assert subset.shape == (120,)
    assert len(set(subset.tolist())) == 120
    assert list(subset) == sorted(subset)
    assert list(subset) != list(range(120)), "a contiguous prefix is forbidden for pose"
    assert subset.max() > 400, "the sample must reach the far end of the population"
    # deterministic for a fixed seed, different for a different seed
    assert np.array_equal(subset, sf1._pair_subset(20260817, 120))
    assert not np.array_equal(subset, sf1._pair_subset(20260818, 120))


def test_d_pose_matches_upstream_compute_distortion_semantics():
    rng = np.random.default_rng(3)
    generated = rng.normal(size=(17, sf1.POSE_DIMS))
    target = rng.normal(size=(17, sf1.POSE_DIMS))
    expected = float(((generated - target) ** 2).mean(axis=1).mean())
    assert sf1.d_pose(generated, target) == pytest.approx(expected, rel=0, abs=0)
    assert sf1.d_pose(generated, generated) == 0.0


def test_score_pose_is_the_contest_sqrt_term():
    assert sf1.score_pose(0.0) == 0.0
    assert sf1.score_pose(1e-5) == pytest.approx(np.sqrt(10.0 * 1e-5))


def test_rate_constant_reproduces_the_frontier_rate_contribution():
    assert pytest.approx(0.12169171641365491, rel=1e-12) == 182_759 * sf1.S_PER_BYTE


def test_gap_to_target_in_bytes_is_the_live_bar():
    """fb1's live bar on hv1 is -14,413.4 B; the -15,157 B figure is off a superseded base."""
    gap_bytes = (sf1.FRONTIER_S - sf1.TARGET_S) / sf1.S_PER_BYTE
    assert gap_bytes == pytest.approx(14_413.4, abs=0.5)


def test_film_row_norms_uses_squared_row_sum_like_the_packer():
    state = {name: np.arange(sf1.FILM_ROWS * 8, dtype=np.float64).reshape(sf1.FILM_ROWS, 8)
             for name in sf1.PRUNE_NAMES}
    norms = sf1.film_row_norms(state)
    for name in sf1.PRUNE_NAMES:
        assert norms[name].shape == (sf1.FILM_ROWS,)
        assert norms[name][0] == pytest.approx(float((np.arange(8.0) ** 2).sum()))


def test_global_selection_ranks_across_tensors_not_within_them():
    """The shipped packer ranks within each tensor; the global selection must not."""
    norms = {
        "blocks.1.film.weight": np.full(sf1.FILM_ROWS, 100.0),
        "blocks.2.film.weight": np.full(sf1.FILM_ROWS, 1.0),
        "blocks.3.film.weight": np.full(sf1.FILM_ROWS, 2.0),
    }
    chosen = sf1.global_lowest_norm_selection(norms, 75)
    assert sum(len(v) for v in chosen.values()) == 75
    assert "blocks.1.film.weight" not in chosen, "the loudest tensor must be skipped entirely"
    assert len(chosen["blocks.2.film.weight"]) == 75


def test_global_selection_spills_into_the_next_tensor_when_one_runs_out():
    norms = {
        "blocks.1.film.weight": np.full(sf1.FILM_ROWS, 100.0),
        "blocks.2.film.weight": np.full(sf1.FILM_ROWS, 1.0),
        "blocks.3.film.weight": np.full(sf1.FILM_ROWS, 2.0),
    }
    chosen = sf1.global_lowest_norm_selection(norms, 250)
    assert len(chosen["blocks.2.film.weight"]) == sf1.FILM_ROWS
    assert len(chosen["blocks.3.film.weight"]) == 250 - sf1.FILM_ROWS
    assert "blocks.1.film.weight" not in chosen


def test_global_selection_never_costs_more_energy_than_the_per_tensor_one(norms):
    """The whole point: at equal cardinality the global pick is weakly cheaper in energy."""
    count = 75
    per_tensor = sf1.keep87_selection(norms)
    global_pick = sf1.global_lowest_norm_selection(norms, count)
    assert sum(len(v) for v in per_tensor.values()) == count
    def energy(selection):
        return sum(float(norms[name][row])
                   for name, rows in selection.items() for row in rows)

    assert energy(global_pick) <= energy(per_tensor) + 1e-12


def test_global_selection_groups_match_the_two_mp2_prune_counts(norms):
    groups = {g["group_id"]: g for g in sf1.global_selection_groups(norms)}
    assert groups["glob_lowest_norm_keep87count"]["row_count"] == 75
    assert groups["glob_lowest_norm_keep25count"]["row_count"] == 432
    for group in groups.values():
        assert group["family"] == "global_reselection"


def _synthetic_state():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    state = {name: torch.from_numpy(
        rng.normal(size=(sf1.FILM_ROWS, 8)).astype(np.float32)) for name in sf1.PRUNE_NAMES}
    state["head.weight"] = torch.from_numpy(rng.normal(size=(3, 4)).astype(np.float32))
    state["head.bias"] = torch.from_numpy(rng.normal(size=(3,)).astype(np.float32))
    return state


def test_sm3r_roundtrip_reproduces_the_shipped_packer_expected_state():
    """The reconstruction of an SM3R candidate must equal what the shipped packer declares."""
    torch = pytest.importorskip("torch")
    sm3 = pytest.importorskip("experiments.ddm_sm3_semantic_representation")
    state = _synthetic_state()
    norms = sf1.film_row_norms({k: v.numpy() for k, v in state.items()
                                if k in sf1.PRUNE_NAMES})
    dropped = sf1.keep87_selection(norms)
    mine = sf1.sm3r_roundtrip_state(state, dropped)
    _, expected, _ = sm3.pack_prune_candidate(state, 87)
    assert set(mine) == set(expected)
    for name in expected:
        assert torch.equal(mine[name], expected[name]), name


def test_sm3r_roundtrip_zeroes_exactly_the_dropped_rows():
    torch = pytest.importorskip("torch")
    state = _synthetic_state()
    dropped = {sf1.PRUNE_NAMES[0]: [3, 17, 191]}
    restored = sf1.sm3r_roundtrip_state(state, dropped)
    flat = restored[sf1.PRUNE_NAMES[0]].reshape(sf1.FILM_ROWS, -1)
    for row in range(sf1.FILM_ROWS):
        if row in dropped[sf1.PRUNE_NAMES[0]]:
            assert torch.count_nonzero(flat[row]) == 0
        else:
            assert torch.count_nonzero(flat[row]) > 0
    # a tensor with no dropped rows is re-encoded, never zeroed
    assert torch.count_nonzero(restored[sf1.PRUNE_NAMES[1]]) > 0


def test_attribution_groups_separate_the_reencode_from_the_prune():
    rng = np.random.default_rng(9)
    norms = {name: rng.random(sf1.FILM_ROWS) for name in sf1.PRUNE_NAMES}
    groups = {g["group_id"]: g for g in sf1.attribution_groups(norms)}
    assert groups["attr_sm3r_q4_reencode_only"]["row_count"] == 0
    assert groups["attr_sm3r_q4_reencode_only"]["builder"] == "sm3r_roundtrip"
    assert groups["attr_sm3r_q4_plus_keep87_rows"]["row_count"] == 75
    assert groups["attr_zero_all_film_rows"]["row_count"] == 3 * sf1.FILM_ROWS
    assert groups["attr_zero_all_film_rows"].get("builder") is None


def test_group_dir_tag_separates_repeat_samplings():
    plain = sf1._group_dir("g1")
    tagged = sf1._group_dir("g1", "seed2")
    assert plain != tagged
    assert tagged.name == "g1__seed2"
