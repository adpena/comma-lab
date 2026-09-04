"""Tests for the ddm_sd1 surrogate-vs-exact mis-pricing instrument.

The load-bearing test is :func:`test_surrogate_matches_the_trainers_own_loss_exactly` -- a
DIFFERENTIAL against ``ddm_qbt1_qbflow_trainer.expected_flip_margin_loss`` itself, so the
instrument cannot drift from the objective it claims to decompose.  Every other test is written so
it would FAIL if the quantity under test were mis-signed, mis-normalised, or silently vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments import ddm_qbt1_qbflow_trainer as qbt
from experiments import ddm_sd1_surrogate_exact_map as sd1


def _random_case(seed: int, height: int = 24, width: int = 32, classes: int = 5):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=3.0, size=(classes, height, width)).astype(np.float32)
    target = rng.integers(0, classes, size=(height, width)).astype(np.uint8)
    return logits, target


# ---------------------------------------------------------------------------
# the differential: this instrument must compute the trainer's own loss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tau", [0.15, 0.11, 0.05])
def test_surrogate_matches_the_trainers_own_loss_exactly(tau: float) -> None:
    logits, target = _random_case(seed=20260904)
    reference = float(
        qbt.expected_flip_margin_loss(
            torch.from_numpy(logits)[None],
            torch.from_numpy(target.astype(np.int64))[None],
            tau,
        )
    )
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    binned = sd1.accumulate_pair(margin, target, competitor, [tau])
    measured = binned["surrogate"][f"{tau:.6f}"].sum() / binned["total_pixels"]
    assert measured == pytest.approx(reference, rel=1e-6)


def test_margin_matches_the_trainers_masked_amax_construction() -> None:
    logits, target = _random_case(seed=7)
    work = torch.from_numpy(logits)[None].clone()
    index = torch.from_numpy(target.astype(np.int64))[None][:, None]
    target_logit = work.gather(1, index).squeeze(1)
    work.scatter_(1, index, -1.0e9)
    reference = (target_logit - work.amax(dim=1))[0].numpy()
    margin, _competitor, _argmax = sd1.margin_and_competitor(logits, target)
    assert np.allclose(margin, reference, rtol=0, atol=1e-5)


def test_margin_sign_is_exactly_the_exact_flip_indicator_in_float32() -> None:
    """``argmax != target`` and ``margin < 0`` agree when no exact tie exists."""

    logits, target = _random_case(seed=11)
    margin, _competitor, argmax = sd1.margin_and_competitor(logits, target)
    assert not np.any(margin == 0.0), "float32 random case should carry no exact ties"
    assert np.array_equal(argmax != target, margin < 0.0)


def test_competitor_is_never_the_target_class() -> None:
    logits, target = _random_case(seed=13)
    _margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    assert not np.any(competitor == target)


def test_margin_and_competitor_refuse_wrong_geometry() -> None:
    logits, target = _random_case(seed=3)
    with pytest.raises(sd1.SD1Error):
        sd1.margin_and_competitor(logits[:3], target)
    with pytest.raises(sd1.SD1Error):
        sd1.margin_and_competitor(logits, target[:5])


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------
def test_stable_sigmoid_matches_the_naive_form_in_range() -> None:
    z = np.linspace(-30.0, 30.0, 4001)
    assert np.allclose(sd1.stable_sigmoid(z), 1.0 / (1.0 + np.exp(-z)), rtol=1e-12, atol=1e-15)


def test_stable_sigmoid_does_not_overflow_at_the_extremes() -> None:
    z = np.array([-1.0e6, -800.0, 0.0, 800.0, 1.0e6])
    with np.errstate(over="raise", invalid="raise"):
        out = sd1.stable_sigmoid(z)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[-1] == 1.0
    assert out[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------
def test_bin_index_and_decode_are_inverse() -> None:
    gt = np.array([0, 1, 4, 2], dtype=np.uint8)
    competitor = np.array([1, 0, 2, 4], dtype=np.uint8)
    annulus = np.array([0, 1, 0, 1], dtype=bool)
    flip = np.array([1, 1, 0, 0], dtype=bool)
    idx = sd1.bin_index(gt, competitor, annulus, flip)
    counts = np.bincount(idx, minlength=sd1.N_BINS)
    cube = sd1.decode_bins(counts)
    for g, c, a, f in zip(gt, competitor, annulus, flip, strict=True):
        assert cube[g, c, int(a), int(f)] == 1
    assert cube.sum() == 4


def test_accumulate_pair_conserves_pixels_and_flip_counts() -> None:
    logits, target = _random_case(seed=101)
    margin, competitor, argmax = sd1.margin_and_competitor(logits, target)
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1])
    assert binned["pixels"].sum() == target.size
    assert binned["total_pixels"] == target.size
    assert binned["flips"] == int((argmax != target).sum())


def test_accumulate_pair_honours_an_explicit_flip_override() -> None:
    """The float16-tie repair: the caller's flip indicator wins over ``margin < 0``."""

    logits, target = _random_case(seed=202)
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    forced = np.zeros(target.shape, dtype=bool)
    forced[0, 0] = True
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1], flip=forced)
    assert binned["flips"] == 1
    assert binned["margin_negative"] == int((margin < 0).sum())
    assert binned["flip_vs_margin_negative_sites"] == int((forced != (margin < 0)).sum())
    cube = sd1.decode_bins(binned["pixels"])
    assert cube[:, :, :, 1].sum() == 1


def test_annulus_band_uses_delta_r_on_the_absolute_margin() -> None:
    logits, target = _random_case(seed=303)
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    delta_r = 0.5
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1], delta_r=delta_r)
    cube = sd1.decode_bins(binned["pixels"])
    assert cube[:, :, 1, :].sum() == int((np.abs(margin.astype(np.float64)) < delta_r).sum())


def test_delta_r_constant_is_the_dr1_n600_value() -> None:
    assert sd1.DELTA_R_N600 == 0.021881818771362305


# ---------------------------------------------------------------------------
# tau
# ---------------------------------------------------------------------------
def test_tau_for_milestone_matches_the_sealed_schedule() -> None:
    for step in (0, 1, 1000, 2000, 4999):
        assert sd1.tau_for_milestone(step) == qbt.tau_for_step(step, sd1.TOTAL_STEPS)


def test_tau_for_milestone_clamps_the_out_of_range_terminal_step() -> None:
    assert sd1.tau_for_milestone(5000) == qbt.tau_for_step(4999, sd1.TOTAL_STEPS)
    assert sd1.tau_for_milestone(5000) == pytest.approx(0.05)
    assert sd1.tau_for_milestone(0) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
def _bins_from_case(seed: int, taus=(0.1,)) -> dict:
    logits, target = _random_case(seed=seed)
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    binned = sd1.accumulate_pair(margin, target, competitor, list(taus))
    return {
        "pixels": binned["pixels"].tolist(),
        "surrogate": {k: v.tolist() for k, v in binned["surrogate"].items()},
        "grad": {k: v.tolist() for k, v in binned["grad"].items()},
    }


def test_global_split_conserves_the_surrogate_and_bounds_recovered_mass() -> None:
    bins = _bins_from_case(seed=404)
    split = sd1.global_split(bins, "0.100000")
    assert split["phantom_mass"] + split["recovered_mass"] == pytest.approx(split["surrogate_total"])
    # every flip carries surrogate mass > 0.5 and every non-flip < 0.5 (sigmoid crosses at m == 0)
    assert split["recovered_mass"] <= split["exact_flips"]
    assert split["recovered_mass"] >= 0.5 * split["exact_flips"]
    assert split["unpriced_flip_mass"] == pytest.approx(
        split["exact_flips"] - split["recovered_mass"]
    )
    assert split["price_ratio"] == pytest.approx(split["surrogate_total"] / split["exact_flips"])


def test_edge_table_exact_flips_sum_to_the_global_flip_count() -> None:
    bins = _bins_from_case(seed=505)
    rows = sd1.edge_table(bins, "0.100000")
    split = sd1.global_split(bins, "0.100000")
    assert sum(row["exact_flips"] for row in rows) == pytest.approx(split["exact_flips"])
    assert sum(row["surrogate_mass"] for row in rows) == pytest.approx(split["surrogate_total"])
    assert len(rows) == sd1.N_CLASSES * (sd1.N_CLASSES - 1)
    assert all(row["gt_class"] != row["competitor_class"] for row in rows)


def test_edge_table_price_ratio_is_surrogate_over_exact() -> None:
    bins = _bins_from_case(seed=606)
    for row in sd1.edge_table(bins, "0.100000"):
        if row["exact_flips"] > 0:
            assert row["price_ratio"] == pytest.approx(
                row["surrogate_mass"] / row["exact_flips"]
            )
        else:
            assert row["price_ratio"] is None
        assert row["phantom_mass"] + row["recovered_mass"] == pytest.approx(row["surrogate_mass"])


def test_price_ratio_spread_is_max_over_min_and_respects_the_support_floor() -> None:
    rows = [
        {"edge": "a", "exact_flips": 100.0, "price_ratio": 2.0},
        {"edge": "b", "exact_flips": 100.0, "price_ratio": 1.0},
        {"edge": "c", "exact_flips": 1.0, "price_ratio": 50.0},
        {"edge": "d", "exact_flips": 0.0, "price_ratio": None},
    ]
    spread = sd1.price_ratio_spread(rows, min_flips=10.0)
    assert spread["spread"] == pytest.approx(2.0)
    assert spread["n_edges"] == 2
    assert spread["max_edge"] == "a" and spread["min_edge"] == "b"
    # the unsupported edge would have manufactured a 50x spread; the floor must exclude it
    assert sd1.price_ratio_spread(rows, min_flips=0.5)["spread"] == pytest.approx(50.0)


def test_price_ratio_spread_reports_none_when_support_is_absent() -> None:
    rows = [{"edge": "a", "exact_flips": 1.0, "price_ratio": 2.0}]
    spread = sd1.price_ratio_spread(rows, min_flips=10.0)
    assert spread["spread"] is None and spread["n_edges"] == 0


# ---------------------------------------------------------------------------
# the vr1 ranking
# ---------------------------------------------------------------------------
def _edge_row(edge, gt, competitor, exact, surrogate, annulus_flips=0.0, interior_flips=0.0):
    return {
        "edge": edge,
        "gt_class": gt,
        "competitor_class": competitor,
        "exact_flips": exact,
        "surrogate_mass": surrogate,
        "annulus_flips": annulus_flips,
        "interior_flips": interior_flips,
    }


def test_excursion_attribution_scores_sign_agreement_not_magnitude() -> None:
    start = [
        _edge_row("Road->Lane", "Road", "Lane", 100.0, 200.0),
        _edge_row("Undrivable->Movable", "Undrivable", "Movable", 100.0, 200.0),
    ]
    end = [
        # exact rises, surrogate rises  -> sign agrees, row 4 could rescale it
        _edge_row("Road->Lane", "Road", "Lane", 150.0, 260.0),
        # exact rises, surrogate FALLS  -> sign disagrees, no positive scale repairs it
        _edge_row("Undrivable->Movable", "Undrivable", "Movable", 150.0, 140.0),
    ]
    out = sd1.excursion_attribution(start, end)
    assert out["total_abs_delta_exact"] == pytest.approx(100.0)
    assert out["net_delta_exact"] == pytest.approx(100.0)
    assert out["coverage"]["vr1_row4_per_edge_tau_sign_agreeing_fraction"] == pytest.approx(0.5)
    # only Undrivable->Movable has a rare competitor here (Movable); Lane is rare too
    assert out["coverage"]["vr1_row3_area_cap_rare_competitor_fraction"] == pytest.approx(1.0)


def test_excursion_attribution_row1_coverage_is_the_annulus_share() -> None:
    start = [_edge_row("Road->Lane", "Road", "Lane", 100.0, 200.0, annulus_flips=10.0)]
    end = [_edge_row("Road->Lane", "Road", "Lane", 140.0, 260.0, annulus_flips=40.0)]
    out = sd1.excursion_attribution(start, end)
    assert out["coverage"]["vr1_row1_margin_weight_annulus_fraction"] == pytest.approx(30.0 / 40.0)


def test_excursion_attribution_row3_coverage_excludes_majority_competitors() -> None:
    start = [
        _edge_row("Road->MyCar", "Road", "MyCar", 100.0, 200.0),
        _edge_row("Road->Movable", "Road", "Movable", 100.0, 200.0),
    ]
    end = [
        _edge_row("Road->MyCar", "Road", "MyCar", 130.0, 240.0),
        _edge_row("Road->Movable", "Road", "Movable", 110.0, 220.0),
    ]
    out = sd1.excursion_attribution(start, end)
    assert out["coverage"]["vr1_row3_area_cap_rare_competitor_fraction"] == pytest.approx(10.0 / 40.0)


def test_excursion_attribution_is_empty_when_nothing_moved() -> None:
    rows = [_edge_row("Road->Lane", "Road", "Lane", 100.0, 200.0)]
    out = sd1.excursion_attribution(rows, rows)
    assert out["total_abs_delta_exact"] == 0.0
    assert out["coverage"] == {}


def test_rare_classes_are_the_two_dual_ascent_constrained_classes() -> None:
    """Row 3 targets exactly the classes qbt1's one-sided dual already pressures upward."""

    assert set(sd1.RARE_CLASSES) == set(
        qbt.MARGIN_CONSTRAINT_MODE_PINS[qbt.MARGIN_CONSTRAINT_LANE_MOVABLE]["bounds"]
    )


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def test_ht_weighted_d_seg_uses_the_sealed_selection_weights() -> None:
    rows = [
        {"pair_id": 4, "sample_weight": 15.0, "lineages": {
            "vehicle_pyav": {"d_seg_exact": 0.001}, "authority_dali": {"d_seg_exact": 0.002}}},
        {"pair_id": 5, "sample_weight": 30.0, "lineages": {
            "vehicle_pyav": {"d_seg_exact": 0.004}, "authority_dali": {"d_seg_exact": 0.005}}},
    ]
    out = sd1.ht_weighted_d_seg(rows)
    assert out["vehicle_pyav"] == pytest.approx((15.0 * 0.001 + 30.0 * 0.004) / 45.0)
    assert out["vehicle_pyav_unweighted"] == pytest.approx(0.0025)
    assert out["vehicle_pyav"] != pytest.approx(out["vehicle_pyav_unweighted"])


def test_sample_weight_lookup_covers_the_whole_sealed_selection() -> None:
    lookup = sd1.sample_weight_lookup()
    assert set(lookup) == set(qbt.SELECTION_IDS)
    assert sorted(set(lookup.values())) == [15.0, 30.0]
    assert sum(1 for v in lookup.values() if v == 30.0) == 8


def test_class_names_are_the_canonical_comma10k_order() -> None:
    assert sd1.CLASS_NAMES == ("Road", "Lane", "Undrivable", "Movable", "MyCar")
    assert sd1.N_CLASSES == 5
    # the luma-sorted order is the wrong one that has bitten this campaign three times
    assert sd1.CLASS_NAMES != ("Road", "Lane", "MyCar", "Undrivable", "Movable")


def test_read_pair_arrays_fails_closed_on_a_missing_payload(tmp_path) -> None:
    with pytest.raises(sd1.SD1Error):
        sd1.read_pair_arrays(tmp_path, 0, 4)
    with pytest.raises(sd1.SD1Error):
        sd1.read_milestone_json(tmp_path, 0)


# ---------------------------------------------------------------------------
# margin capture curve (the fair reading of vr1 row 1's reach)
# ---------------------------------------------------------------------------
def test_margin_capture_is_monotone_and_bounded_by_the_totals() -> None:
    logits, target = _random_case(seed=808)
    margin, competitor, argmax = sd1.margin_and_competitor(logits, target)
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1])
    capture = binned["margin_capture"]
    assert capture["multiples"] == list(sd1.MARGIN_CAPTURE_MULTIPLES)
    assert capture["pixels"] == sorted(capture["pixels"])
    assert capture["flips"] == sorted(capture["flips"])
    assert capture["pixels"][-1] <= target.size
    assert capture["flips"][-1] <= int((argmax != target).sum())
    for pixels, flips in zip(capture["pixels"], capture["flips"], strict=True):
        assert flips <= pixels


def test_margin_capture_counts_the_band_exactly() -> None:
    logits, target = _random_case(seed=909)
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    delta_r = 0.25
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1], delta_r=delta_r)
    absolute = np.abs(margin.astype(np.float64))
    for index, multiple in enumerate(sd1.MARGIN_CAPTURE_MULTIPLES):
        expected = int((absolute < multiple * delta_r).sum())
        assert binned["margin_capture"]["pixels"][index] == float(expected)


def test_margin_capture_at_one_delta_r_matches_the_annulus_bin() -> None:
    """The capture curve at k == 1 must be the same band the edge tables split on."""

    logits, target = _random_case(seed=1010)
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1], delta_r=0.4)
    index = list(sd1.MARGIN_CAPTURE_MULTIPLES).index(1.0)
    cube = sd1.decode_bins(binned["pixels"])
    assert binned["margin_capture"]["pixels"][index] == float(cube[:, :, 1, :].sum())
    assert binned["margin_capture"]["flips"][index] == float(cube[:, :, 1, 1].sum())


def test_global_split_bound_is_exact_only_under_the_margin_flip_indicator() -> None:
    """The docstring's bound: exact with ``margin < 0``, approximate with an override."""

    logits, target = _random_case(seed=1111)
    margin, competitor, _argmax = sd1.margin_and_competitor(logits, target)
    binned = sd1.accumulate_pair(margin, target, competitor, [0.1])
    bins = {
        "pixels": binned["pixels"].tolist(),
        "surrogate": {k: v.tolist() for k, v in binned["surrogate"].items()},
        "grad": {k: v.tolist() for k, v in binned["grad"].items()},
    }
    split = sd1.global_split(bins, "0.100000")
    assert 0.5 * split["exact_flips"] <= split["recovered_mass"] <= split["exact_flips"]

    # Force one non-flip pixel to count as a flip: it carries < 0.5 surrogate mass, so the lower
    # half of the bound must now be violable -- which is exactly why the docstring carries a caveat.
    forced = (margin < 0.0).copy()
    safest = np.unravel_index(int(np.argmax(margin)), margin.shape)
    forced[safest] = True
    binned2 = sd1.accumulate_pair(margin, target, competitor, [0.1], flip=forced)
    bins2 = {
        "pixels": binned2["pixels"].tolist(),
        "surrogate": {k: v.tolist() for k, v in binned2["surrogate"].items()},
        "grad": {k: v.tolist() for k, v in binned2["grad"].items()},
    }
    split2 = sd1.global_split(bins2, "0.100000")
    assert split2["exact_flips"] == split["exact_flips"] + 1
    assert split2["recovered_mass"] < split["recovered_mass"] + 0.5
