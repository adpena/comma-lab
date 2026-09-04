"""Tests for the ddm_gm1 gradient-mass-at-m_safe instrument.

These tests verify BEHAVIOUR, not constants.  Every one of them would fail if the function under
test were replaced by a body that returned canonical-looking markers (NO-FAKE forbidden class 2):
the partition tests mutate the inputs and assert the outputs move, the surrogate test calls the
TRAINER'S OWN loss as the oracle, and the allocator tests assert the mean-1 / monotonicity /
mask-fraction properties the MLX original guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_gm1_gradient_mass_msafe as gm1
from experiments import ddm_qbt1_qbflow_trainer as qbt


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260904)


# ---------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------
def test_group_of_partition_is_exhaustive_and_disjoint():
    rng = _rng()
    margin = rng.normal(0.0, 0.5, size=4096)
    flip = margin < 0.0
    group = gm1.group_of(margin, flip, 0.05)
    assert set(np.unique(group)).issubset({0, 1, 2})
    assert group.shape == margin.shape
    counts = np.bincount(group, minlength=3)
    assert counts.sum() == margin.size


def test_group_of_places_wrong_pixels_in_group_two_regardless_of_band():
    margin = np.array([-10.0, -0.001, 0.001, 10.0])
    flip = np.array([True, True, False, False])
    group = gm1.group_of(margin, flip, 0.05)
    assert group.tolist() == [2, 2, 1, 0]


def test_group_of_band_boundary_is_inclusive_inside():
    m_safe = 0.04376363754272461
    margin = np.array([m_safe, m_safe * (1.0 + 1e-9)])
    flip = np.zeros(2, dtype=bool)
    assert gm1.group_of(margin, flip, m_safe).tolist() == [1, 0]


def test_group_of_moves_pixels_when_m_safe_moves():
    """A larger cap must move correct pixels from OUTSIDE into INSIDE -- never the reverse."""

    rng = _rng()
    margin = np.abs(rng.normal(0.0, 0.2, size=8192))
    flip = np.zeros(margin.size, dtype=bool)
    narrow = np.bincount(gm1.group_of(margin, flip, 0.02), minlength=3)
    wide = np.bincount(gm1.group_of(margin, flip, 0.20), minlength=3)
    assert wide[1] > narrow[1]
    assert wide[0] < narrow[0]


def test_per_class_group_uses_each_pixels_own_threshold():
    margin = np.array([0.03, 0.03, 0.03])
    flip = np.zeros(3, dtype=bool)
    gt = np.array([0, 1, 2])
    thresholds = [0.05, 0.01, 0.05, 0.05, 0.05]
    assert gm1.per_class_group(margin, flip, gt, thresholds).tolist() == [1, 0, 1]


def test_per_class_group_rejects_wrong_threshold_geometry():
    with pytest.raises(gm1.GM1Error):
        gm1.per_class_group(np.zeros(3), np.zeros(3, bool), np.zeros(3, int), [0.1, 0.2])


# ---------------------------------------------------------------------------
# bin algebra
# ---------------------------------------------------------------------------
def test_bin_index_and_decode_roundtrip_every_cell():
    for gt in range(gm1.N_CLASSES):
        for gg in range(gm1.N_GROUPS):
            for gc in range(gm1.N_GROUPS):
                vector = np.zeros(gm1.N_BINS)
                index = int(
                    gm1.bin_index(np.array([gt]), np.array([gg]), np.array([gc]))[0]
                )
                vector[index] = 7.0
                table = gm1.decode_bins(vector)
                assert table[gt, gg, gc] == 7.0
                assert table.sum() == 7.0


def test_decode_bins_shape():
    assert gm1.decode_bins(np.zeros(gm1.N_BINS)).shape == (
        gm1.N_CLASSES,
        gm1.N_GROUPS,
        gm1.N_GROUPS,
    )


# ---------------------------------------------------------------------------
# the two masses
# ---------------------------------------------------------------------------
def test_grad_magnitude_is_even_in_margin():
    """|d sigmoid(-m/tau)/dm| depends only on the DISTANCE to the boundary."""

    rng = _rng()
    margin = rng.normal(0.0, 1.0, size=512)
    left = gm1.grad_magnitude(margin, 0.07)
    right = gm1.grad_magnitude(-margin, 0.07)
    assert np.allclose(left, right, rtol=0.0, atol=1e-15)


def test_grad_magnitude_matches_finite_difference():
    margin = np.array([-0.3, -0.02, 0.0, 0.02, 0.3])
    tau = 0.05
    step = 1e-7
    numeric = np.abs(
        (
            gm1.sd1.stable_sigmoid(-(margin + step) / tau)
            - gm1.sd1.stable_sigmoid(-(margin - step) / tau)
        )
        / (2.0 * step)
    )
    assert np.allclose(gm1.grad_magnitude(margin, tau), numeric, rtol=1e-4)


def test_grad_magnitude_peaks_at_zero_margin_and_scales_as_one_over_tau():
    tau = 0.05
    peak = gm1.grad_magnitude(np.array([0.0]), tau)[0]
    assert peak == pytest.approx(0.25 / tau, rel=1e-12)
    assert gm1.grad_magnitude(np.array([0.0]), tau / 2)[0] == pytest.approx(2.0 * peak, rel=1e-12)


def test_grad_magnitude_rejects_non_positive_tau():
    with pytest.raises(gm1.GM1Error):
        gm1.grad_magnitude(np.zeros(4), 0.0)


def test_surrogate_matches_the_trainers_own_loss_exactly():
    """Differential test against ``qbt.expected_flip_margin_loss`` -- the objective itself.

    This instrument cannot drift from the loss whose gradient it decomposes.
    """

    import torch

    rng = _rng()
    logits = rng.normal(0.0, 3.0, size=(1, gm1.N_CLASSES, 12, 16)).astype(np.float32)
    target = rng.integers(0, gm1.N_CLASSES, size=(1, 12, 16)).astype(np.int64)
    for tau in (0.15, 0.043763637542724609, 0.010940909385681152):
        oracle = float(
            qbt.expected_flip_margin_loss(
                torch.from_numpy(logits), torch.from_numpy(target), tau
            )
        )
        margin, _competitor, _argmax = gm1.sd1.margin_and_competitor(logits[0], target[0])
        mine = float(gm1.sd1.stable_sigmoid(-margin.reshape(-1).astype(np.float64) / tau).mean())
        assert mine == pytest.approx(oracle, rel=1e-6)


# ---------------------------------------------------------------------------
# vr1 row 1 -- _live_margin_weight semantics
# ---------------------------------------------------------------------------
def test_top1_top2_gap_is_non_negative_and_matches_abs_margin_when_gt_is_top_two():
    rng = _rng()
    logits = rng.normal(0.0, 2.0, size=(gm1.N_CLASSES, 24, 32)).astype(np.float32)
    order = np.argsort(logits, axis=0)
    gap = gm1.top1_top2_gap(logits)
    assert (gap >= 0.0).all()
    # GT := the runner-up everywhere -> |signed margin| == gap exactly
    target = order[-2].astype(np.uint8)
    margin, _c, _a = gm1.sd1.margin_and_competitor(logits, target)
    assert np.allclose(np.abs(margin.astype(np.float64)), gap, rtol=0.0, atol=1e-5)


def test_top1_top2_gap_differs_from_abs_margin_when_gt_is_not_the_runner_up():
    """The divergence row 1 inherits: a flipped pixel whose GT is 3rd, not 2nd."""

    logits = np.array([5.0, 4.0, 1.0, 0.0, -1.0], dtype=np.float32).reshape(5, 1, 1)
    target = np.array([[2]], dtype=np.uint8)  # GT is the THIRD-placed class
    gap = gm1.top1_top2_gap(logits)[0, 0]
    margin, _c, _a = gm1.sd1.margin_and_competitor(logits, target)
    assert gap == pytest.approx(1.0, rel=1e-6)
    assert abs(float(margin[0, 0])) == pytest.approx(4.0, rel=1e-6)
    assert gap != pytest.approx(abs(float(margin[0, 0])), rel=1e-3)


def test_top1_top2_gap_rejects_wrong_geometry():
    with pytest.raises(gm1.GM1Error):
        gm1.top1_top2_gap(np.zeros((3, 4, 4), dtype=np.float32))


@pytest.mark.parametrize("fn,temp", list(gm1.ROW1_CONFIGS))
def test_row1_weight_is_mean_one(fn, temp):
    rng = _rng()
    gap = np.abs(rng.normal(0.0, 2.0, size=(64, 64)))
    weight = gm1.row1_weight(gap, fn, temp)
    assert float(weight.mean()) == pytest.approx(1.0, rel=1e-6)
    assert (weight >= 0.0).all()


@pytest.mark.parametrize("fn", ["inverse", "exp"])
def test_row1_smooth_allocators_are_monotone_decreasing_in_the_gap(fn):
    gap = np.array([0.0, 0.1, 0.5, 1.0, 5.0])
    weight = gm1.row1_weight(gap, fn, 0.3)
    assert np.all(np.diff(weight) < 0.0)


def test_row1_bottom_k_masks_the_requested_fraction():
    rng = _rng()
    gap = rng.uniform(0.0, 10.0, size=10_000)
    weight = gm1.row1_weight(gap, "bottom-k", 0.05)
    selected = weight > 0.0
    assert selected.sum() == pytest.approx(500, abs=2)
    assert gap[selected].max() <= gap[~selected].min()


def test_row1_smaller_temp_concentrates_more_mass_on_the_smallest_gaps():
    rng = _rng()
    gap = np.abs(rng.normal(0.0, 2.0, size=20_000))
    smallest = gap <= np.quantile(gap, 0.05)
    sharp = gm1.row1_weight(gap, "inverse", 0.05)[smallest].sum() / gap.size
    blunt = gm1.row1_weight(gap, "inverse", 1.0)[smallest].sum() / gap.size
    assert sharp > blunt


def test_row1_weight_rejects_unknown_allocator():
    with pytest.raises(gm1.GM1Error):
        gm1.row1_weight(np.zeros(4), "not-an-allocator", 0.3)


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------
def _accumulate_fixture(m_safe: float = 0.05):
    rng = _rng()
    logits = rng.normal(0.0, 2.0, size=(gm1.N_CLASSES, 16, 16)).astype(np.float32)
    target = rng.integers(0, gm1.N_CLASSES, size=(16, 16)).astype(np.uint8)
    margin, _c, argmax = gm1.sd1.margin_and_competitor(logits, target)
    flip = argmax != target
    gap = gm1.top1_top2_gap(logits)
    taus = [0.15, 0.05]
    binned = gm1.accumulate_pair(
        margin, target, flip, gap, taus, taus, m_safe, [m_safe] * gm1.N_CLASSES
    )
    return binned, margin, target, flip, taus


def test_accumulate_pair_pixel_counts_sum_to_the_frame():
    binned, margin, _t, _f, _taus = _accumulate_fixture()
    assert binned["pixels"].sum() == margin.size
    assert binned["total_pixels"] == margin.size


def test_accumulate_pair_grad_mass_equals_a_direct_recomputation():
    binned, margin, _t, _f, taus = _accumulate_fixture()
    for tau in taus:
        direct = float(gm1.grad_magnitude(margin.reshape(-1), tau).sum())
        assert float(binned["grad"][f"{tau:.9f}"].sum()) == pytest.approx(direct, rel=1e-10)


def test_accumulate_pair_surrogate_mass_equals_the_loss_numerator():
    binned, margin, _t, _f, taus = _accumulate_fixture()
    for tau in taus:
        direct = float(
            gm1.sd1.stable_sigmoid(-margin.reshape(-1).astype(np.float64) / tau).sum()
        )
        assert float(binned["surrogate"][f"{tau:.9f}"].sum()) == pytest.approx(direct, rel=1e-10)


def test_accumulate_pair_row1_reweighting_preserves_the_total_budget_only_on_average():
    """mean-1 conserves the PIXEL budget, so a uniform gradient field is left untouched."""

    rng = _rng()
    size = 4096
    gap = np.abs(rng.normal(0.0, 2.0, size=size))
    weight = gm1.row1_weight(gap, "inverse", 0.3)
    uniform_grad = np.ones(size)
    assert float((weight * uniform_grad).sum()) == pytest.approx(float(uniform_grad.sum()), rel=1e-6)


def test_accumulate_pair_flip_indicator_is_taken_from_the_caller_not_from_margin_sign():
    """sd1's float16 cure: the caller supplies the exact flip, and the bins must honour it."""

    margin = np.array([[0.5]])
    gt = np.array([[0]], dtype=np.uint8)
    gap = np.array([[0.5]])
    binned = gm1.accumulate_pair(
        margin, gt, np.array([[True]]), gap, [0.05], [], 0.05, [0.05] * gm1.N_CLASSES
    )
    assert gm1.decode_bins(binned["pixels"])[0, 2, 2] == 1.0


# ---------------------------------------------------------------------------
# shares and crossings
# ---------------------------------------------------------------------------
def test_group_shares_sum_to_one():
    rng = _rng()
    vector = rng.uniform(0.0, 1.0, size=gm1.N_BINS)
    shares = gm1.group_shares(vector)
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-12)


def test_group_shares_of_an_empty_field_are_nan_not_zero():
    shares = gm1.group_shares(np.zeros(gm1.N_BINS))
    assert all(np.isnan(value) for value in shares.values())


def test_per_class_group_shares_class_shares_sum_to_one():
    rng = _rng()
    vector = rng.uniform(0.0, 1.0, size=gm1.N_BINS)
    table = gm1.per_class_group_shares(vector)
    total = sum(entry["class_share_of_total_grad"] for entry in table.values())
    assert total == pytest.approx(1.0, rel=1e-12)
    for entry in table.values():
        assert sum(entry[name] for name in gm1.GROUP_NAMES) == pytest.approx(1.0, rel=1e-12)


def test_over_push_share_selects_the_global_inside_class_outside_cell():
    vector = np.zeros(gm1.N_BINS)
    lane = gm1.CLASS_NAMES.index("Lane")
    vector[int(gm1.bin_index(np.array([lane]), np.array([1]), np.array([0]))[0])] = 3.0
    vector[int(gm1.bin_index(np.array([lane]), np.array([0]), np.array([1]))[0])] = 1.0
    result = gm1.over_push_share(vector, "Lane")
    assert result["over_push_grad_mass"] == 3.0
    assert result["under_protect_grad_mass"] == 1.0
    assert result["over_push_share_of_class"] == pytest.approx(0.75, rel=1e-12)


def test_first_crossing_interpolates_in_log_tau():
    taus = [0.01, 0.02, 0.04, 0.08]
    shares = [0.10, 0.30, 0.60, 0.90]
    crossing = gm1.first_crossing(taus, shares, 0.50)
    assert crossing is not None
    assert 0.02 < crossing < 0.04


def test_first_crossing_returns_none_when_the_level_is_never_crossed():
    assert gm1.first_crossing([0.01, 0.1], [0.6, 0.9], 0.25) is None
    assert gm1.first_crossing([0.01, 0.1], [0.05, 0.10], 0.50) is None


def test_first_crossing_finds_the_highest_tau_crossing_on_a_non_monotone_curve():
    taus = [0.01, 0.02, 0.04, 0.08, 0.16]
    shares = [0.10, 0.60, 0.20, 0.40, 0.90]
    crossing = gm1.first_crossing(taus, shares, 0.50)
    assert crossing is not None
    assert 0.08 < crossing < 0.16


# ---------------------------------------------------------------------------
# thresholds resolve through the LAW, never as a literal
# ---------------------------------------------------------------------------
def test_thresholds_resolve_through_the_canonical_law():
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        resolve_margin_band_threshold,
    )

    law = resolve_margin_band_threshold()
    resolved = gm1.resolve_thresholds()
    assert resolved["m_safe"] == float(law.m_safe)
    assert resolved["delta_r"] == float(law.delta_r)
    assert resolved["headroom"] == float(law.headroom)
    assert resolved["n_frames"] == int(law.n_frames)


def test_module_carries_no_hardcoded_m_safe_literal():
    """[[m107]] split-banks cure: the cap must come from the law, not from a copied decimal."""

    from pathlib import Path

    source = Path(gm1.__file__).read_text()
    for literal in ("0.04376363754272461", "0.039180326461791926", "0.021881818771362305"):
        assert literal not in source


def test_charter_taus_are_derived_from_delta_r():
    labelled = gm1.charter_taus(0.02)
    assert labelled["2dR"] == pytest.approx(0.04, rel=1e-12)
    assert labelled["1dR"] == pytest.approx(0.02, rel=1e-12)
    assert labelled["0.5dR"] == pytest.approx(0.01, rel=1e-12)


def test_tau_grid_contains_every_charter_tau_and_any_extra():
    grid = gm1.tau_grid(0.02, extra=[0.1234])
    for value in gm1.charter_taus(0.02).values():
        assert value in grid
    assert 0.1234 in grid
    assert grid == sorted(grid)


def test_grad_magnitude_wrong_tail_is_free_of_the_p_times_one_minus_p_cancellation():
    """Regression guard for the fix this instrument had to make, in its TRACED direction.

    The loss forms ``p = sigmoid(-m/tau)``, so on a confidently-WRONG pixel (``m << 0``) ``p -> 1``
    and ``p*(1-p)`` computes ``1 - (1 - e^-|z|)``, losing the result to one ulp of 1.0.  MEASURED
    relative error of the naive form: 3.6e-08 at z = 20, 4.2e-06 at z = 25, 1.0e-03 at z = 30.  The
    already-CORRECT tail is exact under the same branch split -- my first reading of this had the
    sign backwards and the trace corrected it.  The ``a/(1+a)^2`` form is exact on BOTH tails.

    Aggregate significance on this arm's numbers is below 1e-12 of the total mass (those pixels
    carry ~e^-z of it), so this is hygiene on an exact group share, not a finding.
    """

    tau = 0.05
    for z in (20.0, 25.0, 30.0):
        analytic = float(np.exp(-z) / (1.0 + np.exp(-z)) ** 2 / tau)
        wrong_margin = np.array([-z * tau])
        probability = gm1.sd1.stable_sigmoid(-wrong_margin / tau)
        naive = float((probability * (1.0 - probability) / tau)[0])
        even = float(gm1.grad_magnitude(wrong_margin, tau)[0])
        assert abs(naive - analytic) / analytic > 1e-9  # the defect is real, not hypothetical
        assert abs(even - analytic) / analytic < 1e-14
        # and the correct-side mirror is exact under the even form too
        assert float(gm1.grad_magnitude(np.array([z * tau]), tau)[0]) == pytest.approx(
            analytic, rel=1e-14
        )


def test_accumulate_pair_uses_the_cancellation_free_gradient():
    """The fix must reach the MEASUREMENT path, not only the helper (the dangling-helper trap)."""

    margin = np.array([[-1.25]])  # z = -25 at tau = 0.05: the tail that carries the cancellation
    gt = np.zeros((1, 1), dtype=np.uint8)
    binned = gm1.accumulate_pair(
        margin, gt, np.ones((1, 1), bool), np.array([[1.25]]), [0.05], [], 0.05,
        [0.05] * gm1.N_CLASSES,
    )
    analytic = np.exp(-25.0) / (1.0 + np.exp(-25.0)) ** 2 / 0.05
    assert float(binned["grad"]["0.050000000"].sum()) == pytest.approx(analytic, rel=1e-14)


# ---------------------------------------------------------------------------
# re-analysis from the stored payload
# ---------------------------------------------------------------------------
def _synthetic_report(shift: float = 0.0):
    """A minimal report with the same shape ``run()`` writes, built from known bins."""

    taus = [0.05, 0.15]
    grad = {}
    for index, tau in enumerate(taus):
        vector = np.zeros(gm1.N_BINS)
        # put all mass on Road, global group varying so the shares are predictable
        road = 0
        vector[int(gm1.bin_index(np.array([road]), np.array([0]), np.array([0]))[0])] = 6.0 + shift
        vector[int(gm1.bin_index(np.array([road]), np.array([1]), np.array([1]))[0])] = 3.0
        vector[int(gm1.bin_index(np.array([road]), np.array([2]), np.array([2]))[0])] = 1.0 + index
        grad[f"{tau:.9f}"] = vector.tolist()
    bins = {
        "pixels": np.ones(gm1.N_BINS).tolist(),
        "grad": grad,
        "surrogate": grad,
        "row1_grad": {f"{fn}@{temp:g}": dict(grad) for fn, temp in gm1.ROW1_CONFIGS},
    }
    milestone = {
        "step": 0,
        "tau_eval": 0.15,
        "milestone_recorded": {"d_seg_hat": 0.005, "d_pose_hat": 1.0, "S_hat": 1.0},
        "calibration": {"d_seg_ht_recomputed_vehicle": 0.005},
        "lineages": {
            lineage: dict.fromkeys(("ht", "raw"), bins) for lineage in gm1.LINEAGES
        },
    }
    return {
        "thresholds": {
            "delta_r": 0.02,
            "m_safe": 0.04,
            "headroom": 2.0,
            "per_class_m_safe": dict.fromkeys(gm1.CLASS_NAMES, 0.04),
        },
        "taus": taus,
        "row1_taus": taus,
        "milestones": [milestone],
    }


def test_reanalyse_reproduces_shares_from_the_stored_bins():
    view = gm1.reanalyse(_synthetic_report())["authority_dali:ht"]
    shares = view["milestones"]["0"]["grad_group_share"]["0.150000000"]
    assert shares["correct_outside"] == pytest.approx(6.0 / 11.0, rel=1e-12)
    assert shares["correct_inside"] == pytest.approx(3.0 / 11.0, rel=1e-12)
    assert shares["wrong"] == pytest.approx(2.0 / 11.0, rel=1e-12)


def test_reanalyse_covers_every_lineage_and_mode():
    views = gm1.reanalyse(_synthetic_report())
    assert set(views) == {
        f"{lineage}:{mode}" for lineage in gm1.LINEAGES for mode in ("ht", "raw")
    }


def test_cross_cell_agreement_reports_the_worst_disagreement():
    reports = {"a": _synthetic_report(0.0), "b": _synthetic_report(1.0)}
    agreement = gm1.cross_cell_agreement(reports)
    assert agreement["cells"] == ["a", "b"]
    # cell a: 6/11 outside; cell b: 7/12 outside -> |6/11 - 7/12|
    expected = abs(6.0 / 11.0 - 7.0 / 12.0)
    assert agreement["per_group_max_abs_delta"]["correct_outside"] == pytest.approx(
        expected, rel=1e-9
    )
    assert agreement["argmax"]["correct_outside"]["step"] == "0"


def test_cross_cell_agreement_says_so_when_there_is_only_one_cell():
    result = gm1.cross_cell_agreement({"only": _synthetic_report()})
    assert result["cells"] == ["only"]
    assert "note" in result


def test_run_reanalysis_refuses_an_empty_store(tmp_path):
    with pytest.raises(gm1.GM1Error):
        gm1.run_reanalysis(tmp_path)


def test_main_refuses_measurement_without_run_root(tmp_path):
    with pytest.raises(gm1.GM1Error):
        gm1.main(["--store", str(tmp_path)])


def test_cross_cell_agreement_fails_closed_on_more_than_two_cells():
    """A silently-dropped third cell would be a silently-wrong agreement bound."""

    reports = {name: _synthetic_report(float(i)) for i, name in enumerate("abc")}
    with pytest.raises(gm1.GM1Error):
        gm1.cross_cell_agreement(reports)
