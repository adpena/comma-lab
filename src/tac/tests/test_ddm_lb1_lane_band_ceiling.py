# SPDX-License-Identifier: MIT
"""Tests for ``experiments.ddm_lb1_lane_band_ceiling`` -- the lane-band carrier ceiling price.

Every test exercises BEHAVIOUR (what the composition/scoring/detection actually does to real
arrays), never a constant or a marker, per the CLAUDE.md NO-FAKE class 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_lb1_lane_band_ceiling as lb1


# ---------------------------------------------------------------------------
# lane class SELF-DETECTION (the CLAUDE.md class-order law)
# ---------------------------------------------------------------------------
def _synthetic_labels(lane_cls: int = 1, n: int = 3, h: int = 32, w: int = 40) -> np.ndarray:
    """Label maps whose thinnest, smallest class is ``lane_cls`` -- a 1-px vertical stripe."""

    a = np.zeros((n, h, w), dtype=np.uint8)
    a[:, : h // 3, :] = 2          # a big top block
    a[:, 2 * h // 3 :, :] = 4      # a big bottom block
    a[:, h // 3 : 2 * h // 3, : w // 2] = 0
    a[:, h // 3 : 2 * h // 3, w // 2 :] = 3
    a[:, h // 3 : 2 * h // 3, w // 2 - 1] = lane_cls  # one thin column
    return a


def test_detect_lane_class_finds_the_thin_small_class() -> None:
    labels = _synthetic_labels(lane_cls=1)
    detected, geom = lb1.detect_lane_class(labels)
    assert detected == 1
    assert geom[1]["area_fraction"] < geom[0]["area_fraction"]
    assert geom[1]["thinness"] > geom[0]["thinness"]


def test_detect_lane_class_is_not_index_hardcoded() -> None:
    """Move the thin class to index 3 and the detector must follow it."""

    labels = _synthetic_labels(lane_cls=1)
    remap = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
    moved = np.vectorize(remap.get)(labels).astype(np.uint8)
    assert lb1.detect_lane_class(moved)[0] == 3


def test_detect_lane_class_refuses_when_ambiguous() -> None:
    """Smallest-area and thinnest disagree -> refuse rather than guess."""

    a = np.zeros((1, 20, 20), dtype=np.uint8)
    a[0, :10, :] = 1          # a large but very compact block (low thinness)
    a[0, 15, ::4] = 2         # a tiny, maximally scattered class (high thinness, small area)
    a[0, 16, ::4] = 3         # another scattered class, even smaller
    a[0, 17, 0] = 4
    with pytest.raises(lb1.LB1Error, match="ambiguous"):
        lb1.detect_lane_class(a)


def test_class_geometry_rejects_wrong_rank() -> None:
    with pytest.raises(lb1.LB1Error, match=r"\[P,H,W\]"):
        lb1.class_geometry(np.zeros((4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# exact integer HT scoring
# ---------------------------------------------------------------------------
def test_ht_numerator_is_the_weighted_wrong_site_count() -> None:
    wrong = np.zeros((2, 3, 4), dtype=bool)
    wrong[0, 0, :2] = True      # 2 sites in pair 0
    wrong[1, 1, :3] = True      # 3 sites in pair 1
    weights = np.asarray([15, 30], dtype=np.int64)
    assert lb1.ht_numerator(wrong, weights) == 2 * 15 + 3 * 30


def test_ht_numerator_masked_restricts_to_the_mask() -> None:
    wrong = np.ones((1, 2, 2), dtype=bool)
    mask = np.zeros((1, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    assert lb1.ht_numerator_masked(wrong, mask, np.asarray([15], dtype=np.int64)) == 15


def test_ht_numerator_rejects_shape_mismatch() -> None:
    with pytest.raises(lb1.LB1Error):
        lb1.ht_numerator(np.zeros((2, 2, 2), dtype=bool), np.asarray([15], dtype=np.int64))


def test_denominator_is_population_times_sites() -> None:
    assert lb1.denominator(600, 384 * 512) == 117_964_800.0


# ---------------------------------------------------------------------------
# composition rules
# ---------------------------------------------------------------------------
def _tiny_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    born = np.asarray([[[0, 1, 0, 1]]], dtype=np.uint8)       # sites 1 and 3 predicted Lane
    runner = np.asarray([[[2, 2, 2, 2]]], dtype=np.uint8)     # every runner-up is class 2
    claim = np.asarray([[[True, True, False, False]]])        # carrier claims sites 0 and 1
    return born, runner, claim


def test_union_only_adds_lane_and_never_demotes() -> None:
    born, runner, claim = _tiny_case()
    out = lb1.compose(born, runner, claim, 1, "union")
    assert out.tolist() == [[[1, 1, 0, 1]]]


def test_replace_demotes_unclaimed_born_lane_to_the_runner_up() -> None:
    born, runner, claim = _tiny_case()
    out = lb1.compose(born, runner, claim, 1, "replace")
    # site 3 was born-Lane and unclaimed -> becomes the runner-up class 2.
    assert out.tolist() == [[[1, 1, 0, 2]]]


def test_band_rule_leaves_born_lane_untouched_outside_the_band() -> None:
    born, runner, claim = _tiny_case()
    band = np.asarray([[[True, True, True, False]]])
    out = lb1.compose(born, runner, claim, 1, "band", band=band)
    # site 3 is outside the band, so its born-Lane prediction survives.
    assert out.tolist() == [[[1, 1, 0, 1]]]


def test_band_rule_requires_a_band() -> None:
    born, runner, claim = _tiny_case()
    with pytest.raises(lb1.LB1Error, match="band"):
        lb1.compose(born, runner, claim, 1, "band")


def test_unknown_rule_refuses() -> None:
    born, runner, claim = _tiny_case()
    with pytest.raises(lb1.LB1Error, match="unknown composition rule"):
        lb1.compose(born, runner, claim, 1, "sideways")


def test_compose_does_not_mutate_the_born_field() -> None:
    born, runner, claim = _tiny_case()
    before = born.copy()
    lb1.compose(born, runner, claim, 1, "replace")
    assert np.array_equal(born, before)


# ---------------------------------------------------------------------------
# the perfect-Lane oracle
# ---------------------------------------------------------------------------
def test_oracle_makes_every_gt_lane_site_correct() -> None:
    born = np.asarray([[[0, 0, 1, 3]]], dtype=np.uint8)
    runner = np.asarray([[[2, 2, 0, 2]]], dtype=np.uint8)
    gt = np.asarray([[[0, 1, 0, 3]]], dtype=np.uint8)
    out = lb1.oracle_lane_compose(born, runner, gt, 1)
    assert out[gt == 1].tolist() == [1]
    # the born-Lane false positive at site 2 falls back to its runner-up (class 0, correct here)
    assert out.tolist() == [[[0, 1, 0, 3]]]


def test_oracle_never_breaks_a_correct_site() -> None:
    rng = np.random.default_rng(20260904)
    born = rng.integers(0, 5, size=(4, 8, 8), dtype=np.uint8)
    runner = rng.integers(0, 5, size=(4, 8, 8), dtype=np.uint8)
    gt = rng.integers(0, 5, size=(4, 8, 8), dtype=np.uint8)
    out = lb1.oracle_lane_compose(born, runner, gt, 1)
    broke = (born == gt) & (out != gt)
    assert int(broke.sum()) == 0


def test_oracle_is_monotone_on_the_wrong_site_count() -> None:
    rng = np.random.default_rng(7)
    born = rng.integers(0, 5, size=(3, 16, 16), dtype=np.uint8)
    runner = rng.integers(0, 5, size=(3, 16, 16), dtype=np.uint8)
    gt = rng.integers(0, 5, size=(3, 16, 16), dtype=np.uint8)
    out = lb1.oracle_lane_compose(born, runner, gt, 1)
    assert int((out != gt).sum()) <= int((born != gt).sum())


# ---------------------------------------------------------------------------
# dilation
# ---------------------------------------------------------------------------
def test_dilate_bool_radius_zero_is_identity() -> None:
    mask = np.zeros((1, 5, 5), dtype=bool)
    mask[0, 2, 2] = True
    assert np.array_equal(lb1.dilate_bool(mask, 0), mask)


def test_dilate_bool_grows_a_diamond() -> None:
    mask = np.zeros((1, 5, 5), dtype=bool)
    mask[0, 2, 2] = True
    grown = lb1.dilate_bool(mask, 1)
    assert int(grown.sum()) == 5  # centre + 4 cardinal neighbours
    assert grown[0, 1, 2] and grown[0, 3, 2] and grown[0, 2, 1] and grown[0, 2, 3]
    assert not grown[0, 1, 1]


def test_dilate_bool_does_not_mutate_its_input() -> None:
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, 1, 1] = True
    before = mask.copy()
    lb1.dilate_bool(mask, 2)
    assert np.array_equal(mask, before)


# ---------------------------------------------------------------------------
# scoring bookkeeping
# ---------------------------------------------------------------------------
def _score_fixture() -> dict[str, object]:
    born = np.asarray([[[0, 1, 2, 1]]], dtype=np.uint8)
    gt = np.asarray([[[0, 1, 3, 3]]], dtype=np.uint8)
    composed = np.asarray([[[0, 1, 3, 0]]], dtype=np.uint8)
    site_class = np.zeros((1, 1, 4), dtype=np.uint8)
    site_class[0, 0, 2] = 2  # one site in class code 2
    class_code = {"ALWAYS_CORRECT": 0, "CHURN": 1, "PERSISTENT": 2}
    return lb1.score_composition(
        born, composed, gt, np.asarray([15], dtype=np.int64), site_class, class_code, den=60.0
    )


def test_score_composition_partition_gate_is_exact() -> None:
    scored = _score_fixture()
    assert scored["partition_gate_exact"] is True
    assert scored["partition_gate_sites"] == 4


def test_score_composition_counts_healed_broken_and_still_wrong() -> None:
    scored = _score_fixture()
    # site 2: wrong (2 vs 3) -> correct (3)  => healed
    # site 3: wrong (1 vs 3) -> wrong (0)    => still wrong
    # sites 0,1: correct before and after     => neither
    assert scored["H_healed_sites"] == 1
    assert scored["B_broken_sites"] == 0
    assert scored["W_still_wrong_sites"] == 1
    assert scored["H_healed_numerator"] == 15
    assert scored["numerator_before"] == 30
    assert scored["numerator_after"] == 15


def test_score_composition_reports_the_persistent_class_removal() -> None:
    scored = _score_fixture()
    persistent = scored["by_site_class"]["PERSISTENT"]
    assert persistent["numerator_before"] == 15
    assert persistent["numerator_after"] == 0
    assert persistent["removed_fraction"] == pytest.approx(1.0)


def test_score_composition_delta_d_seg_matches_the_numerators() -> None:
    scored = _score_fixture()
    expected = (scored["numerator_after"] - scored["numerator_before"]) / 60.0
    assert scored["delta_d_seg"] == pytest.approx(expected)


def test_score_composition_collateral_is_split_by_gt_class() -> None:
    scored = _score_fixture()
    # the one healed site has GT class 3.
    assert scored["collateral_by_gt_class"]["3"]["healed_sites"] == 1
    assert scored["collateral_by_gt_class"]["3"]["broken_sites"] == 0


# ---------------------------------------------------------------------------
# the module's declared axis discipline
# ---------------------------------------------------------------------------
def test_axis_declares_the_label_space_ceiling_and_non_promotability() -> None:
    assert "LABEL-SPACE CEILING" in lb1.AXIS
    assert "NON-PROMOTABLE" in lb1.AXIS
    assert "no score claim" in lb1.AXIS


def test_rate_constant_matches_the_contest_rate_term() -> None:
    assert lb1.RATE_S_PER_BYTE == pytest.approx(25.0 / 37_545_489.0)


def test_parser_exposes_every_measured_mode() -> None:
    parser = lb1.build_parser()
    args = parser.parse_args(["--mode", "summary"])
    assert args.mode == "summary"
    for mode in ("forward", "price", "optimal", "fitsweep"):
        assert parser.parse_args(["--mode", mode]).mode == mode


# ---------------------------------------------------------------------------
# the LBND2 slot-schema guard (ddm_lb1 two-landing self-protection)
# ---------------------------------------------------------------------------
def test_slot_vec_pads_a_lower_degree_centerline_losslessly() -> None:
    """Padding with LEADING zeros is polyval-identity, so a deg-2 fit must survive exactly."""

    from tac.boundary_math.analytic_lane_render_band import (
        LaneLine,
        _line_to_slot_vec,
        _slot_vec_to_line,
    )

    line = LaneLine(
        centerline_coeffs=np.asarray([0.5, -2.0, 3.0], np.float64),
        halfwidth_coeffs=np.asarray([0.1, 4.0], np.float64),
        dash_period_m=0.0, dash_phase_m=0.0, dash_duty=0.0,
        forward_range=(5.0, 60.0),
    )
    back = _slot_vec_to_line(_line_to_slot_vec(line))
    forward = np.linspace(5.0, 60.0, 128)
    original = np.polyval(np.asarray(line.centerline_coeffs, np.float64), forward)
    restored = np.polyval(np.asarray(back.centerline_coeffs, np.float64), forward)
    assert np.allclose(original, restored, atol=0.0, rtol=0.0)


def test_slot_vec_refuses_a_centerline_the_schema_cannot_carry() -> None:
    """A deg-4 centerline used to be SILENTLY truncated (measured 23.33 m lateral error)."""

    from tac.boundary_math.analytic_lane_render_band import LaneLine, _line_to_slot_vec

    line = LaneLine(
        centerline_coeffs=np.asarray([1e-4, 0.5, -2.0, 3.0, 1.0], np.float64),
        halfwidth_coeffs=np.asarray([0.1, 4.0], np.float64),
        dash_period_m=0.0, dash_phase_m=0.0, dash_duty=0.0,
        forward_range=(5.0, 60.0),
    )
    with pytest.raises(ValueError, match="refusing to silently drop"):
        _line_to_slot_vec(line)


def test_slot_vec_refuses_an_overlong_halfwidth() -> None:
    from tac.boundary_math.analytic_lane_render_band import LaneLine, _line_to_slot_vec

    line = LaneLine(
        centerline_coeffs=np.asarray([0.5, -2.0, 3.0], np.float64),
        halfwidth_coeffs=np.asarray([0.01, 0.1, 4.0], np.float64),
        dash_period_m=0.0, dash_phase_m=0.0, dash_duty=0.0,
        forward_range=(5.0, 60.0),
    )
    with pytest.raises(ValueError, match="refusing to silently drop"):
        _line_to_slot_vec(line)
