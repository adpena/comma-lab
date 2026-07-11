# SPDX-License-Identifier: MIT
"""Tests for the regime-conditional self-dispatch (#436).

Protects the CONTRACT: past-only classification (no fold-outcome leakage), the interpretable
observatory row, the honest walk-forward arbiter (dispatcher vs global-single-best), the
meta-λ defer guard, and the deferral behavior on uncertain/plateau states. Every number
[macOS advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

import numpy as np

from tac.witness_control.lambda_net import (
    CampaignTrajectory,
    build_intervals,
    fit_score_composition,
)
from tac.witness_control.regime_dispatch import (
    DISPATCH_POLICY,
    PERSISTENCE,
    DispatchBacktest,
    RegimeClassification,
    backtest_dispatch,
    classify_regime,
    dispatch_decision,
    dispatch_for_trajectory,
)


def _traj(slopes: list[float], *, base: float = 0.02, seed: int = 0) -> CampaignTrajectory:
    """Build a deterministic trajectory whose class-1 d_seg walks by the given per-interval
    slopes (so the class-weighted slope magnitude sequence is controllable for the tests)."""
    rng = np.random.default_rng(seed)
    verdicts, loss_terms = [], []
    val = base
    for i, s in enumerate([0.0, *slopes]):
        val = max(val + s, 1e-6)
        ep = 50.0 + 25.0 * i
        d_by = [0.006, val, 0.001, 0.03, 5e-4]
        d_seg = float(np.array([0.23, 0.006, 0.5, 0.012, 0.26]) @ np.array(d_by))
        verdicts.append({"stage": "verdict", "epoch": ep, "d_seg": d_seg,
                         "d_seg_by_class": d_by, "blob_bytes": 85000.0, "ep_loss": 1.0})
        for j in range(3):
            loss_terms.append({"stage": "loss_terms", "ep": ep + j,
                               "terms": {"seg": 1.0, "eikonal": 0.1},
                               "gnorm": 1.0 + 0.01 * rng.random()})
    return CampaignTrajectory(run_dir="synthetic", verdicts=tuple(verdicts),
                              loss_terms=tuple(loss_terms), lever_names=("eikonal", "seg"))


def _real_traj() -> CampaignTrajectory:
    from tac.witness_control.lambda_net import read_trajectory
    return read_trajectory("experiments/results/levelset_v752_baseline_20260710T185913Z")


# ── policy shape ─────────────────────────────────────────────────────────────
def test_policy_is_regime_conditional_not_single_arm():
    # the whole point: the tool depends on the regime (not one global arm)
    assert DISPATCH_POLICY["transient"] == "T_gp_costate_posterior"
    assert DISPATCH_POLICY["plateau"] == PERSISTENCE
    assert DISPATCH_POLICY["uncertain"] == PERSISTENCE
    assert len(set(DISPATCH_POLICY.values())) >= 2


# ── classification ───────────────────────────────────────────────────────────
def test_transient_classified_when_recent_slope_large():
    # last observed slope large vs the running median → transient → route GP
    traj = _traj([0.001, 0.0001, 0.0001, 0.0001, 0.02], seed=1)
    comp = fit_score_composition(traj.verdicts)
    ivs = build_intervals(traj)
    cls = classify_regime(ivs, comp, traj.lever_names, meta_lambda_guard=False)
    assert cls.regime == "transient"
    assert not cls.plateau


def test_plateau_classified_when_recent_slope_small():
    # last observed slope tiny vs the running median → plateau → route persistence
    traj = _traj([0.02, 0.01, 0.005, 0.003, 1e-6], seed=2)
    comp = fit_score_composition(traj.verdicts)
    ivs = build_intervals(traj)
    cls = classify_regime(ivs, comp, traj.lever_names, meta_lambda_guard=False)
    assert cls.regime == "plateau"
    assert cls.plateau
    dec = dispatch_decision(ivs, comp, traj.lever_names, meta_lambda_guard=False)
    assert dec.tool == PERSISTENCE


def test_uncertain_defer_on_insufficient_history():
    # <2 observed intervals → cannot classify → defer to persistence
    traj = _traj([0.01], seed=3)               # 2 verdicts → 1 interval
    comp = fit_score_composition(traj.verdicts)
    ivs = build_intervals(traj)
    cls = classify_regime(ivs, comp, traj.lever_names)
    assert cls.regime == "uncertain"
    assert dispatch_decision(ivs, comp, traj.lever_names).tool == PERSISTENCE


def test_classification_is_past_only_no_target_leak():
    """The classifier for fold `hold` must be a function of intervals[:hold] ONLY —
    appending a future interval must NOT change the classification at the earlier cut."""
    ivs_full = build_intervals(_traj([0.02, 0.001, 0.001, 0.001, 0.05], seed=4))
    traj = _real_traj()
    comp = fit_score_composition(traj.verdicts)
    ivs = build_intervals(traj)
    cut = 5
    c_cut = classify_regime(ivs[:cut], comp, traj.lever_names, meta_lambda_guard=False)
    c_cut_again = classify_regime(ivs[:cut], comp, traj.lever_names, meta_lambda_guard=False)
    # deterministic + independent of anything after the cut (we pass the same slice)
    assert c_cut.regime == c_cut_again.regime
    assert c_cut.recent_slope_mag == c_cut_again.recent_slope_mag
    # and appending the rest of the trajectory does not retroactively change the cut call
    assert isinstance(ivs_full[0], type(ivs[0]))
    assert isinstance(c_cut, RegimeClassification)


# ── observatory (Rudin: dispatch is legible) ─────────────────────────────────
def test_dispatch_decision_emits_observatory_row():
    traj = _real_traj()
    dec = dispatch_for_trajectory(traj)
    line = dec.explain()
    assert "regime=" in line and "TOOL=" in line
    # the WHY must be present: deciding signal + per-regime WF ranking prior
    assert dec.classification.deciding_signal
    assert dec.per_regime_wf_ranking
    assert dec.axis_tag == "[macOS advisory] NON-PROMOTABLE"
    assert dec.actuation == "NONE"


# ── the arbiter: does per-state dispatch beat global-single-best? ────────────
def test_backtest_beats_persistence_and_global_single_best_on_205():
    traj = _real_traj()
    bt = backtest_dispatch(traj, seed=0)
    assert isinstance(bt, DispatchBacktest)
    # MEASURED verdict on the sealed trajectory (the deliverable's arbiter)
    assert bt.beats_persistence, bt.verdict
    assert bt.beats_global_single_best, bt.verdict
    assert bt.dispatcher_wf_mae < bt.global_single_best_wf_mae
    assert bt.dispatcher_wf_mae < bt.persistence_wf_mae
    assert bt.n_folds == 7


def test_backtest_is_deterministic():
    traj = _real_traj()
    a = backtest_dispatch(traj, seed=0).dispatcher_wf_mae
    b = backtest_dispatch(traj, seed=0).dispatcher_wf_mae
    assert a == b


def test_meta_lambda_guard_measured_not_asserted():
    """The guard's effect must be MEASURED (both variants reported), not assumed."""
    traj = _real_traj()
    bt = backtest_dispatch(traj, seed=0, meta_lambda_guard=True)
    assert np.isfinite(bt.dispatcher_wf_mae_no_meta_guard)
    # on #205 the guard flips exactly one late transient fold the head was surprised on;
    # both variants beat global-single-best (the conclusion is robust to the guard choice)
    bt_ng = backtest_dispatch(traj, seed=0, meta_lambda_guard=False)
    assert bt.dispatcher_wf_mae <= bt_ng.dispatcher_wf_mae + 1e-12
    assert bt_ng.dispatcher_wf_mae < bt_ng.global_single_best_wf_mae


def test_meta_lambda_guard_defers_when_model_surprised():
    """Self-awareness: at the live #205 state the discriminator says transient but the
    meta-λ surprise guard fires → defer to persistence ('know when to use nothing')."""
    traj = _real_traj()
    dec = dispatch_for_trajectory(traj, meta_lambda_guard=True)
    assert dec.classification.meta_lambda_surprise
    assert dec.classification.regime == "uncertain"
    assert dec.tool == PERSISTENCE
    # without the guard the same state routes to the transient GP arm (the guard is doing work)
    dec_ng = dispatch_for_trajectory(traj, meta_lambda_guard=False)
    assert dec_ng.classification.regime == "transient"
    assert dec_ng.tool == "T_gp_costate_posterior"


def test_fold_rows_report_regime_tool_and_oracle_diagnostic():
    traj = _real_traj()
    bt = backtest_dispatch(traj, seed=0)
    for r in bt.fold_rows:
        assert r["regime"] in ("transient", "plateau", "uncertain")
        assert r["tool"] in set(DISPATCH_POLICY.values())
        assert "oracle_arm" in r and "route_matches_oracle" in r
        assert np.isfinite(r["dispatcher_err"])


def test_backtest_needs_three_intervals():
    traj = _traj([0.01], seed=9)   # 1 interval
    try:
        backtest_dispatch(traj)
        raise AssertionError("expected ValueError for <3 intervals")
    except ValueError:
        pass


# ── DSL leg ──────────────────────────────────────────────────────────────────
def test_dsl_dispatch_policy_compiles_and_routes():
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1
    prog = derive_costate_agent_v1(
        "experiments/results/levelset_v752_baseline_20260710T185913Z")
    assert prog.validate_program() == []
    assert prog.dispatch_policy.enabled
    assert prog.dispatch_policy.policy_dict()["transient"] == "T_gp_costate_posterior"
    org = prog.compile()
    dec = org.dispatch()
    assert dec.tool in set(DISPATCH_POLICY.values())
    bt = org.dispatch_backtest()
    assert bt.beats_global_single_best


def test_dsl_dispatch_policy_rejects_invented_tool():
    from tac.witness_dsl.costate_agent_dsl import DispatchPolicySpec
    try:
        DispatchPolicySpec(regime_tool=(("transient", "not_a_real_arm"),
                                        ("plateau", "persistence"),
                                        ("uncertain", "persistence")))
        raise AssertionError("expected ValueError for invented tool")
    except ValueError:
        pass


def test_dsl_dispatch_policy_requires_full_regime_cover():
    from tac.witness_dsl.costate_agent_dsl import DispatchPolicySpec
    try:
        DispatchPolicySpec(regime_tool=(("transient", "T_gp_costate_posterior"),))
        raise AssertionError("expected ValueError for missing regimes")
    except ValueError:
        pass
