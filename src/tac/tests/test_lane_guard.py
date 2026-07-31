"""ddm_lg1 (#808) tests for the CONSTRAIN-AND-PROTECT layer.

Covers: schema/derived-value provenance, OFF byte-identity (addend None), the
primal-dual constraint FIRING on a synthetic violation, caps-law bounds, budget
match, born-mask correctness, and the gate_update integration.  Pure numpy — no
scorer, no trainer, no RNG.
"""

from __future__ import annotations

import numpy as np

from tac.optimization import lane_guard as lg


# ------------------------------------------------------------------- derived values
def test_lane_head_sensitivity_ratio_from_measured_normals():
    r = lg.derive_lane_head_sensitivity_ratio()
    # mean(4 Lane pairs)/mean(10 pairs) = 3.8925/3.2544
    assert abs(r - (3.8925 / 3.2544)) < 1e-6
    assert 1.19 < r < 1.20  # Lane ~1.2x more head-sensitive than average
    assert r == lg.LANE_HEAD_SENSITIVITY_RATIO


def test_budget_constants_match_xp1():
    assert lg.LANE_BUDGET_S_UNITS == 0.12589
    assert abs(lg.LANE_BUDGET_DSEG - 0.0012589) < 1e-12


def test_derive_eta_lambda_formula_and_provenance():
    eta, prov = lg.derive_eta_lambda()
    assert abs(eta - 1.0 / (10 * 0.00151)) < 1e-9  # ~66.225
    assert prov["formula"] == "lambda_target / (n_gates_to_engage * erosion_s)"
    assert prov["erosion_s"] == lg.EROSION_S_MEASURED
    # non-default inputs propagate
    eta2, _ = lg.derive_eta_lambda(lambda_target=2.0, n_gates_to_engage=5, erosion_s=0.001)
    assert abs(eta2 - 2.0 / (5 * 0.001)) < 1e-9


def test_derive_lambda_step_cap():
    assert lg.derive_lambda_step_cap() == 0.1
    assert lg.derive_lambda_step_cap(lambda_target=2.0, n_gates_to_engage=4) == 0.5


def test_derive_margin_floor_is_percentile_of_lane_field():
    field = np.linspace(0.0, 1.0, 101, dtype=np.float32)  # p10 == 0.10
    floor, prov = lg.derive_margin_floor(field, pct=10.0)
    assert abs(floor - 0.10) < 1e-6
    assert prov["n_samples"] == 101
    # empty is safe
    f0, _ = lg.derive_margin_floor(np.array([], dtype=np.float32))
    assert f0 == 0.0


# ------------------------------------------------------------------- budget match
def test_per_class_lane_flip_S_matches_qa92_definition():
    # 1 pair, 4x4; make 2 GT-Lane pixels flip.
    gt = np.zeros((1, 4, 4), dtype=np.int64)
    gt[0, 0, 0] = lg.LANE_CLASS
    gt[0, 0, 1] = lg.LANE_CLASS
    gt[0, 1, 1] = lg.LANE_CLASS
    realized = gt.copy()
    realized[0, 0, 0] = 0  # flip one Lane pixel
    realized[0, 0, 1] = 2  # flip another Lane pixel
    s = lg.per_class_lane_flip_S(realized, gt)
    # 2 lane flips over 16 px => 100*2/16 = 12.5
    assert abs(s - 12.5) < 1e-9


def test_per_class_lane_flip_S_zero_when_perfect():
    gt = np.random.default_rng(0).integers(0, 5, size=(2, 8, 8)).astype(np.int64)
    assert lg.per_class_lane_flip_S(gt.copy(), gt) == 0.0


# ------------------------------------------------------------------- born mask
def test_born_lane_support_mask_is_won_lane_only():
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[0, :] = lg.LANE_CLASS  # a lane row
    realized = np.zeros((4, 4), dtype=np.int64)
    realized[0, :2] = lg.LANE_CLASS  # only half the lane row is won
    m = lg.born_lane_support_mask(realized, gt)
    assert m.dtype == np.float32
    assert m.sum() == 2.0  # only the won pixels
    assert m[0, 0] == 1.0 and m[0, 2] == 0.0
    assert m[1, 0] == 0.0  # non-lane untouched


# ------------------------------------------------------------------- OFF identity
def test_addend_none_when_disabled_off_identity():
    cfg = lg.LaneGuardConfig(enabled=False).resolved()
    st = lg.LaneGuardState()
    lstar = np.zeros((8, 8), dtype=np.int64)
    lstar[0, :] = lg.LANE_CLASS
    # lambda 0, no born, no margin-floor => None (seg_pixel_w stays None => byte path)
    assert lg.pixel_weight_addend(lstar, None, st, cfg, 0) is None


def test_addend_none_when_lambda_pinned_zero_control():
    # ON but budget=+inf => g<=0 => lambda stays 0; born/margin off => None (tp1 control).
    cfg = lg.LaneGuardConfig(enabled=True, budget_s=float("inf")).resolved()
    st = lg.LaneGuardState()
    realized = np.zeros((1, 4, 4), dtype=np.int64)
    gts = np.zeros((1, 4, 4), dtype=np.int64)
    gts[0, 0, 0] = lg.LANE_CLASS  # a lane pixel, correctly realized
    realized[0, 0, 0] = lg.LANE_CLASS
    lg.gate_update(st, cfg, realized, gts, (0,))
    assert st.lambda_lane == 0.0
    lstar = gts[0]
    assert lg.pixel_weight_addend(lstar, None, st, cfg, 0) is None


# ------------------------------------------------------------------- constraint fires
def test_dual_ascent_rises_on_violation_and_is_bounded():
    cfg = lg.LaneGuardConfig(enabled=True).resolved()
    st = lg.LaneGuardState()
    # realized Lane WELL above budget => sustained violation.
    over = cfg.budget_s + 0.05
    prev = -1.0
    for _ in range(3):
        lam = lg.dual_ascent(st, cfg, over)
        # each gate step is capped
        assert lam - max(prev, 0.0) <= cfg.lambda_step_cap + 1e-9
        assert lam >= 0.0
        prev = lam
    assert st.lambda_lane > 0.0  # constraint FIRED
    assert st.last_g_s > 0.0


def test_dual_ascent_stays_zero_under_budget():
    cfg = lg.LaneGuardConfig(enabled=True).resolved()
    st = lg.LaneGuardState()
    under = cfg.budget_s - 0.02
    for _ in range(5):
        lg.dual_ascent(st, cfg, under)
    assert st.lambda_lane == 0.0  # never rises when satisfied


def test_dual_ascent_respects_lambda_max_ceiling():
    cfg = lg.LaneGuardConfig(enabled=True, lambda_max=0.3).resolved()
    st = lg.LaneGuardState()
    for _ in range(100):
        lg.dual_ascent(st, cfg, cfg.budget_s + 10.0)  # huge violation
    assert st.lambda_lane == 0.3  # clipped at the ceiling, never exceeds


def test_dual_ascent_projects_nonnegative_and_decays():
    cfg = lg.LaneGuardConfig(enabled=True).resolved()
    st = lg.LaneGuardState(lambda_lane=0.25)
    # now satisfied (under budget): lambda decays but never below 0.
    for _ in range(100):
        lg.dual_ascent(st, cfg, cfg.budget_s - 0.01)
    assert st.lambda_lane == 0.0


# ------------------------------------------------------------------- gate_update
def test_gate_update_telemetry_and_born_refresh():
    cfg = lg.LaneGuardConfig(enabled=True, born_protect_weight=0.5).resolved()
    st = lg.LaneGuardState()
    # 2 gate pairs, make Lane heavily eroded so g>0.
    gts = np.zeros((2, 4, 4), dtype=np.int64)
    gts[:, 0, :] = lg.LANE_CLASS
    realized = gts.copy()
    realized[:, 0, :] = 0  # erase all lane => big violation
    row = lg.gate_update(st, cfg, realized, gts, (7, 9))
    assert row["event"] == "lane_guard"
    assert row["realized_lane_s_units"] > 0.0
    assert row["g_s_units"] > 0.0
    assert row["lambda_lane"] > 0.0
    assert row["score_claim"] is False
    assert row["born_mask_pairs"] == 2
    assert set(st.born_masks.keys()) == {7, 9}


def test_gate_update_derives_margin_floor_once():
    cfg = lg.LaneGuardConfig(enabled=True, margin_floor_weight=1.0, margin_floor_pct=10.0).resolved()
    st = lg.LaneGuardState()
    gts = np.zeros((1, 4, 4), dtype=np.int64)
    gts[0, 0, :] = lg.LANE_CLASS
    realized = gts.copy()
    margins = {5: np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)}
    lg.gate_update(st, cfg, realized, gts, (5,), lane_margins_by_id=margins)
    assert st.margin_floor is not None
    prev = st.margin_floor
    # second gate does NOT re-derive
    lg.gate_update(st, cfg, realized, gts, (5,), lane_margins_by_id=margins)
    assert st.margin_floor == prev


# ------------------------------------------------------------------- addend active
def test_addend_active_lambda_on_lane_pixels():
    cfg = lg.LaneGuardConfig(enabled=True).resolved()
    st = lg.LaneGuardState(lambda_lane=2.0)
    lstar = np.zeros((4, 4), dtype=np.int64)
    lstar[0, :] = lg.LANE_CLASS
    add = lg.pixel_weight_addend(lstar, None, st, cfg, 0)
    assert add is not None
    assert np.all(add[0, :] == 2.0)      # lane pixels get +lambda
    assert np.all(add[1:, :] == 0.0)     # non-lane untouched


def test_addend_born_scaled_by_sensitivity():
    cfg = lg.LaneGuardConfig(enabled=True, born_protect_weight=0.5).resolved()
    st = lg.LaneGuardState()
    st.born_masks[3] = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    lstar = np.zeros((2, 2), dtype=np.int64)  # lambda 0, only born term
    add = lg.pixel_weight_addend(lstar, None, st, cfg, 3)
    assert add is not None
    assert abs(add[0, 0] - 0.5 * cfg.lane_sensitivity_ratio) < 1e-6
    assert add[1, 1] == 0.0


def test_addend_margin_floor_emphasizes_low_margin_lane():
    cfg = lg.LaneGuardConfig(enabled=True, margin_floor_weight=1.0).resolved()
    st = lg.LaneGuardState(margin_floor=0.5)
    lstar = np.full((2, 2), lg.LANE_CLASS, dtype=np.int64)
    margin = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    add = lg.pixel_weight_addend(lstar, margin, st, cfg, 0)
    assert add is not None
    # relu(1 - m/0.5): m=0 -> 1.0 ; m=0.5 -> 0.0 ; m=1.0 -> 0(clamped) ; m=0.25 -> 0.5
    assert abs(add[0, 0] - 1.0) < 1e-6
    assert abs(add[0, 1] - 0.0) < 1e-6
    assert abs(add[1, 0] - 0.0) < 1e-6
    assert abs(add[1, 1] - 0.5) < 1e-6


# ------------------------------------------------------------------- piece-3 helper
def test_per_component_min_flip_distance_closed_form():
    born = np.zeros((6, 6), dtype=np.float32)
    born[0, 0:2] = 1.0          # component A (2 px)
    born[4:6, 4:6] = 1.0        # component B (4 px)
    margin = np.full((6, 6), 9.0, dtype=np.float32)
    margin[0, 1] = 0.4          # A's min margin
    margin[5, 5] = 2.0          # B's min margin
    labels, min_d = lg.per_component_min_flip_distance(margin, born, dw_norm=4.0)
    assert labels.max() == 2 and min_d.shape == (2,)
    a_lab = labels[0, 0]
    assert abs(min_d[a_lab - 1] - 0.4 / 4.0) < 1e-6   # d = |m|/||dw|| (fp32 input)
    b_lab = labels[5, 5]
    assert abs(min_d[b_lab - 1] - 2.0 / 4.0) < 1e-6


def test_per_component_min_flip_distance_default_norm_is_conservative():
    born = np.ones((2, 2), dtype=np.float32)
    margin = np.full((2, 2), 1.0, dtype=np.float32)
    _, d_def = lg.per_component_min_flip_distance(margin, born)
    # default dw_norm = 4.007 (Lane-Movable, largest measured Lane-pair normal)
    assert abs(d_def[0] - 1.0 / 4.007) < 1e-6
    # empty mask safe
    labels, d0 = lg.per_component_min_flip_distance(margin, np.zeros((2, 2)))
    assert d0.shape == (0,) and labels.max() == 0


def test_complementarity_in_gate_row():
    cfg = lg.LaneGuardConfig(enabled=True).resolved()
    st = lg.LaneGuardState()
    gts = np.zeros((1, 4, 4), dtype=np.int64)
    gts[0, 0, :] = lg.LANE_CLASS
    realized = np.zeros((1, 4, 4), dtype=np.int64)  # lane fully erased
    row = lg.gate_update(st, cfg, realized, gts, (0,))
    assert abs(row["complementarity"] - row["lambda_lane"] * row["g_s_units"]) < 1e-12


# ------------------------------------------------------------------- DSL levers
def test_dsl_lever_factories_flags_match_trainer_argparse():
    from tac.witness_dsl.spec_tr1_renderer_20260728 import (
        lever_lane_guard_born,
        lever_lane_guard_lambda,
        lever_lane_guard_margin_floor,
    )

    levers = [lever_lane_guard_lambda(), lever_lane_guard_born(0.5),
              lever_lane_guard_margin_floor(1.0)]
    flags: set[str] = set()
    for lv in levers:
        flags |= set(lv.flag_dict() if hasattr(lv, "flag_dict") else lv.overrides)
    import experiments.train_tr1_partition_renderer_mlx as tr1

    known: set[str] = set()
    for a in tr1.build_argparser()._actions:
        known |= set(a.option_strings)
    assert flags <= known, f"invented flags: {flags - known}"
    # budget LawRef custody present on the lambda lever
    assert "--lane-guard-budget-s" in levers[0].constant_refs


def test_resolved_fail_closed_on_sign_inverting_values():
    import pytest

    with pytest.raises(ValueError):
        lg.LaneGuardConfig(enabled=True, eta_lambda=-1.0).resolved()
    with pytest.raises(ValueError):
        lg.LaneGuardConfig(enabled=True, lambda_step_cap=-0.1).resolved()
    with pytest.raises(ValueError):
        lg.LaneGuardConfig(enabled=True, lambda_max=0.0).resolved()
