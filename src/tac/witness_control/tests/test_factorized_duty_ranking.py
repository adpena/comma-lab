# SPDX-License-Identifier: MIT
"""Tests for the exact-factorized duty ranking (organ upgrade A).

NO-FAKE: tests verify the MATH — a ker(A)-supported camera actuation yields a marginal of
EXACTLY zero (the theorem, via the real mask), crossing counts equal hand-computed exact
counts on constructed margin fields, rankings respond to the margin field (not constants),
and the persisted-row recompute agrees with the direct compute."""
from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS
from tac.witness_control.factorized_duty_ranking import (
    LeverMarginal,
    alignment,
    camera_map_survival_scale,
    format_factorized_duty_line,
    lever_class_direction,
    lever_marginal_from_snapshot,
    rank_levers_from_snapshot,
    rank_levers_from_summary_row,
    self_calibrated_eps,
)
from tac.witness_control.factorized_features import (
    CAMERA_HW,
    SCORER_HW,
    MarginSnapshot,
    ker_a_zero_weight_mask,
    parse_oriented_key,
)


def _snap(margins_by: dict[str, list[float]], total_px: int = 2 * 384 * 512) -> MarginSnapshot:
    wrongs, gts, ms = [], [], []
    for key, vals in margins_by.items():
        w, g = parse_oriented_key(key)
        for v in vals:
            wrongs.append(w)
            gts.append(g)
            ms.append(v)
    n = len(ms)
    return MarginSnapshot(
        run_ref="synthetic", ema_epoch=7, generated_at="t", pair_indices=(0, 1),
        scorer_hw=SCORER_HW, total_px=total_px, d_seg_sample=n / total_px,
        flip_pair_idx=np.zeros(n, np.int32), flip_y=np.zeros(n, np.int32),
        flip_x=np.arange(n, dtype=np.int32),
        flip_wrong=np.asarray(wrongs, np.int8), flip_gt=np.asarray(gts, np.int8),
        flip_margin=np.asarray(ms, np.float64),
    )


def test_alignment_perfectly_aimed_is_one():
    u = np.zeros(5)
    u[1] = 1.0   # gt = Lane
    u[0] = -1.0  # wrong = Road
    assert alignment(u, 0, 1) == pytest.approx(1.0)


def test_alignment_zero_direction_and_wrong_direction_are_zero():
    assert alignment(np.zeros(5), 0, 1) == 0.0
    u = np.zeros(5)
    u[0] = 1.0  # pushes the WRONG class up
    assert alignment(u, 0, 1) == 0.0  # clamped


def test_lever_class_direction_matches_dsl_trunk():
    u = lever_class_direction("thin_lane")
    assert np.allclose(u, [0, 1, 0, 0, 0])
    assert np.allclose(lever_class_direction("rankfloor"), np.zeros(5))


def test_camera_map_pure_ker_support_scale_is_exactly_zero():
    k = ker_a_zero_weight_mask()
    m = np.zeros(CAMERA_HW)
    m[k] = 7.5
    assert camera_map_survival_scale(m) == 0.0


def test_camera_map_fully_visible_scale_is_one():
    k = ker_a_zero_weight_mask()
    m = np.zeros(CAMERA_HW)
    m[~k] = 1.0
    assert camera_map_survival_scale(m) == pytest.approx(1.0)


def test_ker_supported_lever_marginal_is_provably_zero():
    snap = _snap({"Road->Lane": [1e-6] * 50})  # trivially crossable margins
    k = ker_a_zero_weight_mask()
    blind_map = np.zeros(CAMERA_HW)
    blind_map[k] = 100.0
    r = lever_marginal_from_snapshot("thin_lane", snap, eps_feat=1e9, camera_map=blind_map)
    assert r.marginal_d_seg == 0.0 and r.kappa == 0.0


def test_crossed_count_exact_on_constructed_margins():
    # thin_lane: u = e_Lane; Road->Lane align = 1/sqrt(2); norm = 3.953
    margins = [0.05, 0.10, 0.20, 0.40, 0.80]
    snap = _snap({"Road->Lane": margins})
    eps = 0.1
    thr = eps * HEAD_PAIR_NORMS["Road-Lane"] * (1.0 / np.sqrt(2.0))  # ~0.2795
    expect = sum(1 for m in margins if m <= thr)  # 0.05, 0.10, 0.20 -> 3
    r = lever_marginal_from_snapshot("thin_lane", snap, eps_feat=eps)
    assert r.crossed_flips == expect == 3
    assert r.marginal_d_seg == pytest.approx(expect / snap.total_px)


def test_marginal_monotone_in_eps():
    snap = _snap({"Road->Lane": list(np.linspace(0.01, 1.0, 100))})
    vals = [lever_marginal_from_snapshot("thin_lane", snap, eps_feat=e).crossed_flips
            for e in (0.01, 0.05, 0.1, 0.3)]
    assert vals == sorted(vals) and vals[-1] > vals[0]


def test_no_class_direction_lever_has_zero_marginal():
    snap = _snap({"Road->Lane": [1e-6] * 10})
    r = lever_marginal_from_snapshot("rankfloor", snap, eps_feat=10.0)
    assert r.marginal_d_seg == 0.0 and r.align_mass_weighted == 0.0


def test_ranking_tracks_the_margin_field_not_constants():
    """Margins chosen NEAR the discrimination boundary: at eps=0.1, thin_lane's Road->Lane
    threshold is 0.2795 (align 1/sqrt2) while island_amplify's is 0.110 (align 0.279) —
    margins of 0.2 separate them; the Movable case separates the other way (thin_lane
    cannot aim Road->Movable at all).  The ranking must follow the FIELD."""
    lane_snap = _snap({"Road->Lane": [0.2] * 100, "Road->Movable": [0.2] * 2})
    mov_snap = _snap({"Road->Movable": [0.15] * 100, "Road->Lane": [0.5] * 2})
    names = ["thin_lane", "island_amplify"]
    top_lane = rank_levers_from_snapshot(lane_snap, names, eps_feat=0.1)[0].lever
    top_mov = rank_levers_from_snapshot(mov_snap, names, eps_feat=0.1)[0].lever
    assert top_lane == "thin_lane" and top_mov == "island_amplify"


def test_self_calibrated_eps_is_median_feature_flipdist():
    snap = _snap({"Road->Lane": [0.3953, 0.3953 * 3]})
    # flip distances: 0.1, 0.3 -> median 0.2
    assert self_calibrated_eps(snap) == pytest.approx(0.2, rel=1e-3)
    assert self_calibrated_eps(_snap({})) == 0.0


def test_summary_row_recompute_matches_direct_compute():
    rng = np.random.default_rng(5)
    snap = _snap({
        "Road->Lane": list(rng.uniform(0.01, 2.0, 400)),
        "Lane->Road": list(rng.uniform(0.01, 2.0, 300)),
        "Undrivable->Road": list(rng.uniform(0.01, 2.0, 100)),
    })
    eps = 0.08
    direct = {r.lever: r.marginal_d_seg
              for r in rank_levers_from_snapshot(snap, ["thin_lane", "horizon_margin"], eps_feat=eps)}
    from_row = {r["lever"]: r["marginal_d_seg"]
                for r in rank_levers_from_summary_row(snap.summary_row(),
                                                      ["thin_lane", "horizon_margin"], eps_feat=eps)}
    for k in direct:
        # histogram-resolution approximation: agree within a few % of the flip mass
        assert from_row[k] == pytest.approx(direct[k], rel=0.08, abs=2e-7)


def test_rank_from_summary_row_empty_inputs():
    assert rank_levers_from_summary_row({}) == []
    assert rank_levers_from_summary_row({"total_px": 0, "by_oriented_pair": {}}) == []


def test_format_line_pure_formatter():
    line = format_factorized_duty_line(
        [{"lever": "thin_lane", "marginal_d_seg": 1.2e-4}], ema_epoch=900, age_s=120.0)
    assert "thin_lane" in line and "ep900" in line and "NON-PROMOTABLE" in line
    assert "no snapshot rows" in format_factorized_duty_line([])


def test_lever_marginal_to_dict_carries_non_promotable_markers():
    snap = _snap({"Road->Lane": [0.05]})
    d = lever_marginal_from_snapshot("thin_lane", snap, eps_feat=0.1).to_dict()
    assert d["score_claim"] is False and "advisory" in d["axis_tag"]
    assert isinstance(LeverMarginal(**{k: d[k] for k in (
        "lever", "marginal_d_seg", "crossed_flips", "eps_feat", "kappa",
        "align_mass_weighted", "adjoint_response_l1")}), LeverMarginal)
