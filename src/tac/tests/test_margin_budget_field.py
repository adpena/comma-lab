# SPDX-License-Identifier: MIT
"""n4-fixture tests for the QA80 margin-budget producer (scorer-free)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.margin_budget_field import (
    CLASS_ORDER,
    MAX_PAIR_NORM_PER_CLASS,
    PAIR_NORM_MATRIX,
    MarginBudgetError,
    MarginBudgetField,
    budget_field_from_gt_cache,
    conservative_budget_field,
    exact_flip_distance_field,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    HEAD_PAIR_NORMS,
    head_flip_distance_feature_space,
)


def test_pair_norm_matrix_symmetric_and_matches_measured():
    assert len(CLASS_ORDER) == 5
    assert PAIR_NORM_MATRIX.shape == (5, 5)
    assert np.allclose(PAIR_NORM_MATRIX, PAIR_NORM_MATRIX.T, equal_nan=True)
    assert np.all(np.isinf(np.diag(PAIR_NORM_MATRIX)))
    idx = {n: i for i, n in enumerate(CLASS_ORDER)}
    a, b = "Road", "Lane"
    assert PAIR_NORM_MATRIX[idx[a], idx[b]] == pytest.approx(HEAD_PAIR_NORMS["Road-Lane"])


def test_exact_field_matches_rank4_law():
    margin = np.array([[2.0, -1.5], [0.5, 3.0]])
    winner = np.array([[0, 1], [2, 3]])
    runner = np.array([[1, 2], [0, 4]])
    field = exact_flip_distance_field(margin, winner, runner)
    for m, c, cp, got in zip(margin.ravel(), winner.ravel(), runner.ravel(), field.ravel()):
        assert got == pytest.approx(head_flip_distance_feature_space(m, PAIR_NORM_MATRIX[c, cp]))


def test_conservative_is_a_sound_lower_bound_on_exact():
    rng = np.random.default_rng(1)
    margin = rng.uniform(0.1, 5.0, size=(4, 8))
    winner = rng.integers(0, 5, size=(4, 8))
    runner = (winner + rng.integers(1, 5, size=(4, 8))) % 5  # != winner
    exact = exact_flip_distance_field(margin, winner, runner)
    cons = conservative_budget_field(margin, winner)
    # max pair-norm denominator => conservative <= exact for every runner-up
    assert np.all(cons <= exact + 1e-12)
    # conservative uses the per-class MAX pair-norm
    assert MAX_PAIR_NORM_PER_CLASS.shape == (5,)


def test_fail_closed_on_bad_inputs():
    with pytest.raises(MarginBudgetError):
        exact_flip_distance_field(np.array([1.0]), np.array([0]), np.array([0]))  # winner == runner
    with pytest.raises(MarginBudgetError):
        exact_flip_distance_field(np.array([1.0]), np.array([9]), np.array([0]))  # class OOR
    with pytest.raises(MarginBudgetError):
        conservative_budget_field(np.array([1.0, 2.0]), np.array([0]))  # shape mismatch


def _write_cache(path, *, with_margins=True, n=2, h=4, w=4):
    rng = np.random.default_rng(2)
    arrays = {
        "lstars": rng.integers(0, 5, size=(n, h, w)).astype(np.int64),
        "gt_poses": rng.standard_normal((n, 6)).astype(np.float32),
    }
    if with_margins:
        arrays["margins"] = rng.uniform(0.0, 4.0, size=(n, h, w)).astype(np.float32)
    np.savez(path, **arrays)


def test_budget_field_from_gt_cache_conservative(tmp_path):
    cache = tmp_path / "gt_n2.npz"
    _write_cache(cache)
    mb = budget_field_from_gt_cache(cache, mode="conservative")
    assert isinstance(mb, MarginBudgetField)
    assert mb.mode == "conservative"
    assert mb.field.shape == (2, 4, 4)
    with np.load(cache) as z:
        expect = conservative_budget_field(z["margins"], z["lstars"])
    assert np.allclose(mb.field, expect)
    # single-pair selection
    mb0 = budget_field_from_gt_cache(cache, pair_index=0)
    assert mb0.field.shape == (4, 4)


def test_budget_field_from_gt_cache_fail_closed(tmp_path):
    cache = tmp_path / "gt_n2.npz"
    _write_cache(cache)
    with pytest.raises(MarginBudgetError):
        budget_field_from_gt_cache(cache, mode="exact")  # runner-up unavailable => post-burn
    nomargin = tmp_path / "gt_nomargin.npz"
    _write_cache(nomargin, with_margins=False)
    with pytest.raises(MarginBudgetError):
        budget_field_from_gt_cache(nomargin)


def test_margin_budget_field_summary_and_save(tmp_path):
    cache = tmp_path / "gt_n2.npz"
    _write_cache(cache)
    mb = budget_field_from_gt_cache(cache)
    summary = mb.summary()
    assert summary["mode"] == "conservative"
    assert len(summary["quantiles_0_5_25_50_75_95_100"]) == 7
    manifest = mb.save(tmp_path / "field.npy")
    assert (tmp_path / "field.npy").is_file()
    assert manifest["field_sha256"]
    assert manifest["schema"] == "qa80_margin_budget_field.v1"
