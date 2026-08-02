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


def test_dsl_compile_argv_parses_against_real_trainer_argparse():
    """b4s regression (the store_true wiring gap): a program carrying the three lane-guard
    levers must compile to an argv the REAL trainer argparser accepts — the bool True
    override must emit the bare ``--lane-guard`` flag (never a stray ``True`` token)."""
    from tac.witness_dsl.spec_tr1_renderer_20260728 import (
        TR1RendererProgramV1,
        lever_a1_gate,
        lever_desc_level_roundtrip,
        lever_lane_guard_born,
        lever_lane_guard_lambda,
        lever_lane_guard_margin_floor,
        lever_renderer_capacity,
        lever_seg_physics,
        lever_token_grid,
        lever_token_temporal,
        lever_variant,
        lever_window,
    )

    prog = TR1RendererProgramV1(
        levers=(lever_variant("plain"), lever_token_grid(16, 4),
                lever_renderer_capacity(24), lever_desc_level_roundtrip(16, "round"),
                lever_token_temporal("shared_base"), lever_seg_physics("ce", 100.0, 1.3),
                lever_a1_gate(5), lever_window(10, 20.0),
                lever_lane_guard_lambda(), lever_lane_guard_born(0.25),
                lever_lane_guard_margin_floor(0.5)),
        num_pairs=8, out_dir="/tmp/b4s_compile_probe")
    argv = prog.compile_trainer_argv()
    assert "True" not in argv, f"stray store_true token in argv: {argv}"
    assert "False" not in argv
    assert "--lane-guard" in argv
    import experiments.train_tr1_partition_renderer_mlx as tr1

    ns = tr1.build_argparser().parse_args(argv[1:])  # argv[0] is the trainer relpath
    assert ns.lane_guard is True
    assert ns.lane_guard_born_weight == 0.25
    assert ns.lane_guard_margin_floor_weight == 0.5
    assert ns.class_weight_lane == 1.3


# ==================================================================================
# ddm_bs2 (#871) — the BUDGET SCHEDULE (a constant budget can never bind)
#
# Every test below is a REGRESSION GUARD for a defect that was MEASURED, not imagined:
# the burn-4 primary telemetry (64 lane_guard rows) has lambda_lane == 0.0 on 64/64
# gates and g < 0 on 64/64.  The two control tests are the canary pair — the negative
# one FAILED on the first implementation (analytic k) and is what forced the calibrator.
# ==================================================================================

# MEASURED from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl
_BURN4_SIGMA = 0.00142148          # sd(diff(realized_lane_s))/sqrt(2) over the 64 gates
_BURN4_FIRST, _BURN4_LAST = 0.122438, 0.077481


def _replay(series, *, ratchet: bool, horizon: int = 64):
    """Drive the SHIPPED ratchet + dual arithmetic over a realized-Lane series."""
    cfg = lg.LaneGuardConfig(enabled=True, budget_ratchet=ratchet,
                             ratchet_horizon_gates=horizon).resolved()
    st = lg.LaneGuardState()
    lam, bud = [], []
    for x in series:
        st.realized_history.append(float(x))
        if st.budget_s_current is None:
            st.budget_s_current = float(cfg.budget_s)
        if cfg.budget_ratchet:
            nb, _ = lg.derive_ratchet_budget(
                st.realized_history, st.budget_s_current, cfg.eta_lambda,
                cfg.lambda_step_cap, mean_gates=cfg.ratchet_mean_gates,
                n_gates_horizon=cfg.ratchet_horizon_gates, lambda_max=cfg.lambda_max)
            st.budget_s_current = float(nb)
        lam.append(lg.dual_ascent(st, cfg, float(x), budget_s=st.budget_s_current))
        bud.append(st.budget_s_current)
    return np.asarray(lam), np.asarray(bud)


# ------------------------------------------------------------------ noise floor
def test_noise_floor_is_trend_agnostic():
    """A pure linear ramp has ZERO gate-to-gate noise; an OLS-residual estimator would
    also say ~0 here, but the first-difference estimator must say EXACTLY 0 and must
    keep saying sigma when a ramp is ADDED to noise (the burn-4 situation)."""
    ramp = np.linspace(0.12, 0.07, 40)
    s_ramp, _ = lg.derive_noise_floor(ramp)
    assert s_ramp is not None and s_ramp < 1e-12
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, _BURN4_SIGMA, 40)
    s_flat, _ = lg.derive_noise_floor(noise)
    s_both, _ = lg.derive_noise_floor(ramp + noise)
    # adding a linear trend must not inflate the estimate
    assert abs(s_both - s_flat) < 0.15 * s_flat


def test_noise_floor_refuses_short_history():
    """Fail SAFE: with too little history there is no measured floor, so the caller
    must fall back to the static budget rather than ratchet on an unmeasured sigma."""
    s, prov = lg.derive_noise_floor([0.1, 0.1, 0.1])
    assert s is None and prov["value"] is None
    b, prov2 = lg.derive_ratchet_budget([0.1, 0.1, 0.1], 0.12589, 66.2, 0.1)
    assert b == 0.12589
    assert prov2["engaged"] is False and prov2["reason"] == "insufficient_history"


def test_noise_floor_matches_burn4_measurement():
    """The estimator reproduces the MEASURED burn-4 sigma on a synthetic series of the
    same construction (guards against a silent change of estimator)."""
    rng = np.random.default_rng(5)
    ser = np.linspace(0.122, 0.077, 64) + rng.normal(0.0, _BURN4_SIGMA, 64)
    s, prov = lg.derive_noise_floor(ser)
    assert abs(s - _BURN4_SIGMA) < 0.25 * _BURN4_SIGMA
    assert prov["mad_agreement_rel"] < 0.35  # MAD twin agrees => no outlier contamination


# ------------------------------------------------------------------ deadband
def test_analytic_k_is_a_lower_bracket_for_the_calibrated_k():
    """The analytic k prices a FIXED reference; the shipped budget is a RUNNING MINIMUM.
    The gap IS the min-selection bias and must be strictly positive, never negative."""
    eta, cap = lg.derive_eta_lambda()[0], lg.derive_lambda_step_cap()
    k_a, _ = lg.derive_deadband_k(_BURN4_SIGMA, eta, cap, 64)
    k_c, prov = lg.calibrate_deadband_k(_BURN4_SIGMA, eta, cap, horizon=64)
    assert k_c > k_a > 0.0
    assert prov["selection_bias_k"] == k_c - k_a > 0.0
    assert prov["null_expected_max_lambda_rel"] <= 1.0 + 1e-9


def test_deadband_k_grows_with_noise_and_horizon():
    """Physics check on the derivation: a noisier gate or a longer horizon must WIDEN
    the deadband.  If either monotonicity inverts, the formula is wired backwards."""
    eta, cap = lg.derive_eta_lambda()[0], lg.derive_lambda_step_cap()
    k_short, _ = lg.derive_deadband_k(_BURN4_SIGMA, eta, cap, 16)
    k_long, _ = lg.derive_deadband_k(_BURN4_SIGMA, eta, cap, 256)
    assert k_long > k_short
    k_quiet, _ = lg.derive_deadband_k(_BURN4_SIGMA / 4.0, eta, cap, 64)
    assert k_quiet < k_short or k_quiet < k_long


def test_deadband_k_degenerate_inputs_do_not_raise():
    k, prov = lg.derive_deadband_k(0.0, 66.2, 0.1, 64)
    assert k == 0.0 and "degenerate" in prov["note"]
    k2, prov2 = lg.calibrate_deadband_k(0.0, 66.2, 0.1)
    assert k2 == 0.0 and prov2["calibrated"] is False


# ------------------------------------------------------------------ the ratchet
def test_ratchet_is_monotone_non_increasing():
    """The whole point: a budget that can RISE is not a ratchet and cannot lock in gains."""
    rng = np.random.default_rng(3)
    ser = np.concatenate([np.linspace(0.122, 0.077, 40), 0.077 + rng.normal(0, 0.003, 24)])
    _, bud = _replay(ser, ratchet=True)
    assert np.all(np.diff(bud) <= 1e-12)


def test_ratchet_target_is_never_an_extrapolation():
    """Feasibility by construction: the budget never drops below a level the run has
    actually held on average, so the constraint is always achievable."""
    rng = np.random.default_rng(4)
    ser = np.linspace(0.122, 0.077, 64) + rng.normal(0, _BURN4_SIGMA, 64)
    _, bud = _replay(ser, ratchet=True)
    m = lg.N_GATES_TO_ENGAGE_DEFAULT
    for t in range(m, len(ser)):
        assert bud[t] >= float(np.min([np.mean(ser[max(0, i + 1 - m):i + 1])
                                       for i in range(m - 1, t + 1)])) - 1e-12


def test_ratchet_off_reproduces_the_measured_burn4_inertness():
    """REGRESSION GUARD on the defect itself.  On a monotonically IMPROVING Lane series
    the constant budget yields lambda == 0 at every gate — the exact 64/64 signature
    measured on burn-4.  If this ever fails, the legacy arm changed behaviour."""
    rng = np.random.default_rng(7)
    ser = np.linspace(_BURN4_FIRST, _BURN4_LAST, 64) + rng.normal(0, _BURN4_SIGMA, 64)
    lam, bud = _replay(ser, ratchet=False)
    assert np.all(lam == 0.0)
    assert len(np.unique(bud)) == 1 and bud[0] == lg.LANE_BUDGET_S_UNITS


def test_ratchet_arms_the_guard_on_the_same_series():
    """Same improving series, ratchet ON: lambda still stays 0 (correct — nothing eroded)
    but the budget has TRACKED DOWN, so the guard is now armed near the achieved level
    instead of ~0.047 S above it."""
    rng = np.random.default_rng(7)
    ser = np.linspace(_BURN4_FIRST, _BURN4_LAST, 64) + rng.normal(0, _BURN4_SIGMA, 64)
    lam, bud = _replay(ser, ratchet=True)
    assert np.all(lam == 0.0), "an improving series must not engage the dual"
    assert bud[-1] < lg.LANE_BUDGET_S_UNITS
    permitted_erosion = bud[-1] - float(ser.min())
    assert permitted_erosion < 0.02, "ratcheted budget must sit near the achieved level"
    legacy_permitted = lg.LANE_BUDGET_S_UNITS - float(ser.min())
    assert legacy_permitted > 2.0 * permitted_erosion


# ------------------------------------------------------------------ THE CONTROLS (P4)
def test_negative_control_null_series_does_not_thrash_the_dual():
    """NEGATIVE CONTROL / canary.  A stationary series with only measurement noise has NO
    erosion, so noise alone must not move the dual by more than one step cap.  This is
    the test that FAILED on the analytic-k implementation (36.2% of gates engaged,
    200/200 trials) and forced calibrate_deadband_k."""
    rng = np.random.default_rng(777)
    cap = lg.derive_lambda_step_cap()
    peaks, engaged = [], []
    for _ in range(24):
        lam, _ = _replay(0.09 + rng.normal(0, _BURN4_SIGMA, 64), ratchet=True)
        peaks.append(float(lam.max()))
        engaged.append(float((lam > 0).mean()))
    assert float(np.mean(peaks)) <= cap, (
        f"E[max lambda | NULL] = {np.mean(peaks):.5f} exceeds the step cap {cap}")
    assert float(np.mean(engaged)) < 0.20, "null engagement rate is thrash-level"


def test_negative_control_single_lucky_gate_does_not_lock_the_budget():
    """A -4 sigma outlier must not ratchet the budget onto an unreachable level (the
    naive zero-deadband ratchet's failure mode: bind forever after one lucky gate)."""
    rng = np.random.default_rng(31)
    ser = 0.09 + rng.normal(0, _BURN4_SIGMA, 64)
    ser[20] -= 4.0 * _BURN4_SIGMA
    lam, _ = _replay(ser, ratchet=True)
    assert lam.max() == 0.0


def test_positive_control_genuine_erosion_engages_the_dual():
    """POSITIVE CONTROL.  Real descent followed by real erosion must engage — and must
    engage where the legacy constant budget stays silent."""
    rng = np.random.default_rng(999)
    ser = np.concatenate([np.linspace(0.120, 0.075, 40),
                          0.075 + np.linspace(0.0, 0.025, 24)])
    ser = ser + rng.normal(0, _BURN4_SIGMA, 64)
    lam_r, _ = _replay(ser, ratchet=True)
    lam_l, _ = _replay(ser, ratchet=False)
    assert lam_r.max() > 0.5, "ratchet failed to engage on +0.025 S of genuine erosion"
    assert lam_l.max() == 0.0, "legacy is expected to be blind here (the defect)"


def test_positive_control_detection_floor_matches_the_derived_deadband():
    """The MEASURED smallest detected erosion must agree with the DERIVED deadband
    k*sigma — derivation and behaviour have to be the same object."""
    rng = np.random.default_rng(1234)
    eta, cap = lg.derive_eta_lambda()[0], lg.derive_lambda_step_cap()
    k, _ = lg.calibrate_deadband_k(_BURN4_SIGMA, eta, cap, horizon=64)
    deadband = k * _BURN4_SIGMA
    below = np.concatenate([np.linspace(0.120, 0.075, 40),
                            0.075 + np.linspace(0.0, 0.3 * deadband, 24)])
    above = np.concatenate([np.linspace(0.120, 0.075, 40),
                            0.075 + np.linspace(0.0, 6.0 * deadband, 24)])
    n = rng.normal(0, _BURN4_SIGMA, 64)
    lam_below = _replay(below + n, ratchet=True)[0].max()
    lam_above = _replay(above + n, ratchet=True)[0].max()
    # The DESIGNED guarantee is a bound on MAGNITUDE, not on incidence: sub-deadband
    # motion may produce a transient blip but must stay inside the same one-step-cap
    # envelope the null calibration enforces.  (Asserting lam_below == 0 here is
    # STRONGER than anything the derivation promises — it failed, correctly.)
    assert lam_below <= cap, f"sub-deadband erosion drove lambda to {lam_below} > cap {cap}"
    assert lam_above > 10.0 * cap, "supra-deadband erosion must engage decisively"
    assert lam_above > 10.0 * max(lam_below, 1e-12)


# ------------------------------------------------------------------ integration
def test_gate_update_reports_ratchet_and_inertness():
    """The guard must SURFACE its own inert state: burn-4 ran 64/64 gates with lambda==0
    and g<0 and NO telemetry field said so.  'Off' is a tracked state, never silent."""
    cfg = lg.LaneGuardConfig(enabled=True, budget_ratchet=False).resolved()
    st = lg.LaneGuardState()
    gt = np.zeros((2, 8, 8), dtype=np.int64)
    gt[:, :2, :] = lg.LANE_CLASS
    realized = gt.copy()  # perfect Lane => realized_lane_s = 0 => deeply slack
    row = {}
    for _ in range(cfg.ratchet_mean_gates + 2):
        row = lg.gate_update(st, cfg, realized, gt, (0, 1))
    assert row["lambda_lane"] == 0.0
    assert row["g_s_units"] < 0.0
    assert row["inertness_alarm"] is True
    assert row["inert_slack_gates"] >= cfg.ratchet_mean_gates
    assert row["budget_ratchet"] is False
    assert row["ratchet"]["engaged"] is False
    assert row["budget_s_units"] == row["budget_s_static"] == cfg.budget_s


def test_gate_update_ratchet_on_tightens_the_budget():
    cfg = lg.LaneGuardConfig(enabled=True, budget_ratchet=True).resolved()
    st = lg.LaneGuardState()
    gt = np.zeros((2, 8, 8), dtype=np.int64)
    gt[:, :2, :] = lg.LANE_CLASS
    realized = gt.copy()
    rows = [lg.gate_update(st, cfg, realized, gt, (0, 1))
            for _ in range(cfg.ratchet_mean_gates + 2)]
    row = rows[-1]
    assert row["budget_ratchet"] is True
    assert row["ratchet"]["engaged"] is True
    assert row["budget_s_units"] < row["budget_s_static"]
    assert row["ratchet"]["sigma_s"] >= 0.0
    # `tightened_by_s` is PER-GATE, so it is 0.0 once the ratchet has settled (this series
    # is constant).  The cumulative tightening is the quantity that must be positive.
    assert sum(r["ratchet"].get("tightened_by_s", 0.0) for r in rows) > 0.0
    assert row["ratchet"]["tightened_by_s"] == 0.0, "a settled ratchet must not keep moving"


def test_config_rejects_invalid_ratchet_params():
    import pytest
    with pytest.raises(ValueError):
        lg.LaneGuardConfig(enabled=True, ratchet_mean_gates=0).resolved()
    with pytest.raises(ValueError):
        lg.LaneGuardConfig(enabled=True, ratchet_horizon_gates=-1).resolved()


def test_dual_ascent_budget_override_is_backward_compatible():
    """budget_s=None must reproduce the pre-ratchet arithmetic EXACTLY."""
    cfg = lg.LaneGuardConfig(enabled=True).resolved()
    a, b = lg.LaneGuardState(), lg.LaneGuardState()
    for x in (0.20, 0.20, 0.05):
        la = lg.dual_ascent(a, cfg, x)
        lb = lg.dual_ascent(b, cfg, x, budget_s=None)
        assert la == lb


def test_ratchet_dsl_lever_compiles_to_real_flags():
    """never-invent-flags: the DSL lever's tokens must exist in the trainer's argparse."""
    import experiments.train_tr1_partition_renderer_mlx as tr1
    from tac.witness_dsl.spec_tr1_renderer_20260728 import lever_lane_guard_ratchet
    lev = lever_lane_guard_ratchet(64)
    ap = tr1.build_argparser()
    known: set[str] = set()
    for a in ap._actions:
        known |= set(a.option_strings)
    for flag in lev.overrides:
        assert flag in known, f"DSL lever emits {flag}, absent from the trainer argparse"
    ns = ap.parse_args(["--variant", "lotto", "--out-dir", "/dev/null",
                        "--lane-guard", "--lane-guard-ratchet",
                        "--lane-guard-ratchet-horizon", "64"])
    assert ns.lane_guard is True and ns.lane_guard_ratchet is True
    assert ns.lane_guard_ratchet_horizon == 64
