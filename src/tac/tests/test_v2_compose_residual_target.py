# SPDX-License-Identifier: MIT
"""Tests for tac.v2_compose.residual_target — the keystone S3 (residual = GT - bulk_through_R)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.v2_compose.residual_target import (
    compute_residual_target,
    load_residual_target,
    save_residual_target,
)


def _stacks():
    rng = np.random.default_rng(0)
    n, H, W = 4, 12, 16
    gt = rng.integers(0, 5, size=(n, H, W)).astype(np.int64)
    bulk = gt.copy()
    # make the bulk WRONG on a known set of cells (the residual the INR must flip)
    bulk[:, 0, 0] = (gt[:, 0, 0] + 1) % 5
    bulk[:, 1, 1] = (gt[:, 1, 1] + 2) % 5
    return bulk, gt


def test_residual_is_bulk_subtracted_exact_cells():
    """residual_mask = EXACTLY the cells where bulk != gt (the bulk is SUBTRACTED)."""
    bulk, gt = _stacks()
    rt = compute_residual_target(bulk, gt)
    assert rt.residual_mask.shape == gt.shape
    assert np.array_equal(rt.residual_mask, bulk != gt)
    # 2 wrong cells per frame, 4 frames -> 8 wrong cells of n*H*W = 4*12*16 = 768
    assert int(rt.residual_mask.sum()) == 8
    assert rt.bulk_dseg == pytest.approx(8 / (4 * 12 * 16))


def test_perfect_bulk_has_zero_residual():
    _bulk, gt = _stacks()
    rt = compute_residual_target(gt.copy(), gt)  # bulk == gt
    assert int(rt.residual_mask.sum()) == 0
    assert rt.bulk_dseg == 0.0


def test_per_class_residual_sums_to_total():
    bulk, gt = _stacks()
    rt = compute_residual_target(bulk, gt)
    total = float(rt.residual_mask.mean())
    assert sum(rt.per_class_residual.values()) == pytest.approx(total, abs=1e-9)
    # ranked is descending by residual fraction
    vals = [rt.per_class_residual[c] for c in rt.residual_classes_ranked]
    assert vals == sorted(vals, reverse=True)


def test_save_load_roundtrip(tmp_path):
    bulk, gt = _stacks()
    rt = compute_residual_target(bulk, gt)
    path = tmp_path / "residual_target.npz"
    nbytes = save_residual_target(rt, path)
    assert nbytes > 0
    assert path.exists()
    assert path.with_suffix(".summary.json").exists()
    rt2 = load_residual_target(path)
    assert np.array_equal(rt2.residual_mask, rt.residual_mask)
    assert np.array_equal(rt2.bulk_argmax_through_R, rt.bulk_argmax_through_R)
    assert np.array_equal(rt2.gt_lstars, rt.gt_lstars)
    assert rt2.bulk_dseg == pytest.approx(rt.bulk_dseg)


def test_shape_mismatch_raises():
    bulk, gt = _stacks()
    with pytest.raises(ValueError):
        compute_residual_target(bulk[:, :, :-1], gt)
    with pytest.raises(ValueError):
        compute_residual_target(bulk[0], gt[0])  # 2D, not (n,H,W)


def test_summary_is_advisory_nonpromotable():
    bulk, gt = _stacks()
    rt = compute_residual_target(bulk, gt)
    s = rt.to_summary()
    assert s["score_claim"] is False
    assert s["promotable"] is False
    assert s["bulk_dseg_floor"] == pytest.approx(rt.bulk_dseg)
