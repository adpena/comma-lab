# SPDX-License-Identifier: MIT
"""Contract tests for deterministic costate-organ router stability."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tac.witness_control.lambda_net import read_trajectory
from tac.witness_control.regime_dispatch import (
    DISPATCH_POLICY,
    backtest_dispatch,
    dispatch_for_trajectory,
)
from tac.witness_control.router_stability import (
    BLOCKED_DISTRIBUTION_CUSTODY,
    DistributionCustodyError,
    RegimeDensityCustody,
    RouterReplayError,
    RouterReplayMismatchError,
    append_decide_record,
    calibrate_router_forecast,
    certify_fp32_gate,
    importance_weighted_architecture_eval,
    load_replay_events,
    make_decision_record,
    replay_apply,
    self_normalized_clipped_masked_weights,
)

_REAL_RUN = "experiments/results/levelset_v752_baseline_20260710T185913Z"
_HASH_A = hashlib.sha256(b"backtest").hexdigest()
_HASH_B = hashlib.sha256(b"live").hexdigest()
_HASH_SCHEMA = hashlib.sha256(b"transient|plateau|uncertain").hexdigest()


def _custody() -> RegimeDensityCustody:
    return RegimeDensityCustody(
        backtest_density={"transient": 0.5, "plateau": 0.5},
        live_density={"transient": 0.75, "plateau": 0.25},
        backtest_source_sha256=_HASH_A,
        live_source_sha256=_HASH_B,
        regime_schema_sha256=_HASH_SCHEMA,
        backtest_source="backtest.jsonl",
        live_source="live.jsonl",
    )


def test_fp32_gate_is_deterministic_and_certifies_margin():
    kwargs = {
        "recent_slope_mag": 2.0e-4,
        "median_slope_mag": 1.0e-4,
        "n_past_intervals": 9,
        "surprise_ratio": 1.2,
        "meta_lambda_guard": True,
        "policy": DISPATCH_POLICY,
    }
    a = certify_fp32_gate(**kwargs)
    b = certify_fp32_gate(**kwargs)
    assert a == b
    assert a.gate_dtype == "numpy.float32"
    assert a.selected_regime == "transient"
    assert a.selected_tool == "T_gp_costate_posterior"
    assert a.slope_margin_ulps is not None and a.slope_margin_ulps > 1.0
    assert a.stable_beyond_float32_roundoff


def test_fp32_gate_tie_rule_is_explicit_and_marks_boundary_unstable():
    cert = certify_fp32_gate(
        recent_slope_mag=1.0,
        median_slope_mag=1.0,
        n_past_intervals=2,
        surprise_ratio=float("nan"),
        meta_lambda_guard=False,
        policy=DISPATCH_POLICY,
    )
    assert cert.selected_regime == "transient"
    assert cert.slope_abs_margin == 0.0
    assert not cert.stable_beyond_float32_roundoff
    assert "tie" in cert.tie_break_rule


def test_decide_apply_replays_exact_tool_and_is_restart_idempotent(tmp_path):
    cert = certify_fp32_gate(
        recent_slope_mag=2.0,
        median_slope_mag=1.0,
        n_past_intervals=4,
        surprise_ratio=1.0,
        meta_lambda_guard=True,
        policy=DISPATCH_POLICY,
    )
    rec = make_decision_record(
        run_ref="run", decision_epoch=225.0, selected_regime=cert.selected_regime,
        selected_tool=cert.selected_tool, certificate=cert)
    ledger = tmp_path / "router.jsonl"
    append_decide_record(ledger, rec)
    append_decide_record(ledger, rec)
    outcome = replay_apply(ledger, rec.decision_id)
    events = load_replay_events(ledger)
    assert [e["event"] for e in events] == ["DECIDE", "REPLAY_MATCH"]
    assert outcome.selected_tool == rec.selected_tool
    assert not outcome.router_learning_frozen
    assert outcome.actuation == "NONE"


def test_decision_record_rejects_a_tool_not_selected_by_certificate():
    cert = certify_fp32_gate(
        recent_slope_mag=2.0,
        median_slope_mag=1.0,
        n_past_intervals=4,
        surprise_ratio=1.0,
        meta_lambda_guard=True,
        policy=DISPATCH_POLICY,
    )
    with pytest.raises(ValueError, match="selected_tool"):
        make_decision_record(
            run_ref="run", decision_epoch=225.0,
            selected_regime=cert.selected_regime,
            selected_tool="persistence",
            certificate=cert,
        )


def test_decide_apply_mismatch_appends_alarm_and_fails_closed(tmp_path):
    cert = certify_fp32_gate(
        recent_slope_mag=0.5,
        median_slope_mag=1.0,
        n_past_intervals=4,
        surprise_ratio=float("nan"),
        meta_lambda_guard=True,
        policy=DISPATCH_POLICY,
    )
    rec = make_decision_record(
        run_ref="run", decision_epoch=100.0, selected_regime=cert.selected_regime,
        selected_tool=cert.selected_tool, certificate=cert)
    ledger = tmp_path / "router.jsonl"
    append_decide_record(ledger, rec)
    with pytest.raises(RouterReplayMismatchError, match="router replay mismatch"):
        replay_apply(ledger, rec.decision_id, requested_tool="T_gp_costate_posterior")
    assert load_replay_events(ledger)[-1]["event"] == "MISMATCH_ALARM"


def test_old_decision_replays_after_router_learns_a_new_selection(tmp_path):
    transient = certify_fp32_gate(
        recent_slope_mag=2.0, median_slope_mag=1.0, n_past_intervals=4,
        surprise_ratio=1.0, meta_lambda_guard=True, policy=DISPATCH_POLICY)
    plateau = certify_fp32_gate(
        recent_slope_mag=0.5, median_slope_mag=1.0, n_past_intervals=5,
        surprise_ratio=1.0, meta_lambda_guard=True, policy=DISPATCH_POLICY)
    first = make_decision_record(
        run_ref="run", decision_epoch=100.0, selected_regime=transient.selected_regime,
        selected_tool=transient.selected_tool, certificate=transient)
    second = make_decision_record(
        run_ref="run", decision_epoch=125.0, selected_regime=plateau.selected_regime,
        selected_tool=plateau.selected_tool, certificate=plateau)
    ledger = tmp_path / "router.jsonl"
    append_decide_record(ledger, first)
    append_decide_record(ledger, second)
    replay = replay_apply(ledger, first.decision_id)
    assert replay.selected_tool == first.selected_tool
    assert replay.selected_tool != second.selected_tool
    assert not replay.router_learning_frozen


def test_replay_rejects_tampered_decide_payload(tmp_path):
    cert = certify_fp32_gate(
        recent_slope_mag=2.0, median_slope_mag=1.0, n_past_intervals=4,
        surprise_ratio=1.0, meta_lambda_guard=True, policy=DISPATCH_POLICY)
    record = make_decision_record(
        run_ref="run", decision_epoch=100.0, selected_regime=cert.selected_regime,
        selected_tool=cert.selected_tool, certificate=cert)
    ledger = tmp_path / "router.jsonl"
    append_decide_record(ledger, record)
    row = json.loads(ledger.read_text())
    row["payload"]["selected_tool"] = "persistence"
    ledger.write_text(json.dumps(row) + "\n")
    with pytest.raises(RouterReplayError, match="content-address verification"):
        replay_apply(ledger, record.decision_id)


def test_self_normalized_clipped_masked_density_ratio():
    diag = self_normalized_clipped_masked_weights(
        ["transient", "transient", "plateau", "plateau"],
        custody=_custody(),
        clip_bounds=(0.75, 1.25),
        support_mask=[True, False, True, True],
    )
    weights = np.asarray(diag.normalized_weights)
    assert weights[1] == 0.0
    assert weights.sum() == pytest.approx(3.0)
    assert diag.raw_ratios == pytest.approx((1.5, 1.5, 0.5, 0.5))
    assert diag.clipped_ratios == pytest.approx((1.25, 1.25, 0.75, 0.75))
    assert diag.n_retained == 3
    assert 0.0 < diag.effective_sample_size <= 3.0


def test_importance_weighting_refuses_missing_or_zero_support_custody():
    with pytest.raises(DistributionCustodyError, match=BLOCKED_DISTRIBUTION_CUSTODY):
        self_normalized_clipped_masked_weights(
            ["transient"], custody=None, clip_bounds=(0.5, 2.0))
    zero_support = RegimeDensityCustody(
        backtest_density={"transient": 0.0, "plateau": 1.0},
        live_density={"transient": 1.0, "plateau": 0.0},
        backtest_source_sha256=_HASH_A,
        live_source_sha256=_HASH_B,
        regime_schema_sha256=_HASH_SCHEMA,
        backtest_source="backtest.jsonl",
        live_source="live.jsonl",
    )
    with pytest.raises(DistributionCustodyError, match="zero backtest support"):
        self_normalized_clipped_masked_weights(
            ["transient"], custody=zero_support, clip_bounds=(0.5, 2.0))


def test_importance_weighted_arch_eval_selects_with_lexical_tie_rule():
    rows = [
        {"regime": "transient", "per_arm_err": {"arm_a": 1.0, "arm_b": 0.0}},
        {"regime": "plateau", "per_arm_err": {"arm_a": 0.0, "arm_b": 1.0}},
    ]
    report = importance_weighted_architecture_eval(
        rows, custody=_custody(), clip_bounds=(0.5, 2.0))
    assert report.status == "IS_WEIGHTED"
    assert report.selected_arch == "arm_b"
    assert report.per_arch_weighted_mae["arm_b"] < report.per_arch_weighted_mae["arm_a"]


def test_forecast_calibration_is_sequential_and_allocates_shadow_compute():
    rows = [
        {
            "epoch": 1.0,
            "tool": "persistence",
            "route_matches_oracle": False,
            "dispatcher_err": 2.0,
            "gate_certificate": {"stable_beyond_float32_roundoff": True},
        },
        {
            "epoch": 2.0,
            "tool": "T_gp_costate_posterior",
            "route_matches_oracle": True,
            "dispatcher_err": 1.0,
            "gate_certificate": {"stable_beyond_float32_roundoff": False},
        },
    ]
    report = calibrate_router_forecast(rows)
    assert report.status == "MIS_CALIBRATED_INSTANCE"
    assert report.terminal_posterior_alpha == 2.0
    assert report.terminal_posterior_beta == 2.0
    assert report.terminal_posterior_match_probability == pytest.approx(0.5)
    assert [step["posterior_match_probability"]
            for step in report.sequential_posterior] == pytest.approx((1.0 / 3.0, 0.5))
    assert all(row["requested_k"] == 2 for row in report.compute_allocations)
    assert all("A_ridge_solve" in row["candidate_tools"]
               for row in report.compute_allocations)
    assert report.actuation == "NONE"


def test_forecast_calibration_rejects_unproven_prior_or_missing_certificate():
    with pytest.raises(ValueError, match="prior parameters"):
        calibrate_router_forecast([
            {"epoch": 1.0, "tool": "persistence", "route_matches_oracle": True,
             "dispatcher_err": 0.0,
             "gate_certificate": {"stable_beyond_float32_roundoff": True}},
        ], prior_alpha=0.0)
    with pytest.raises(ValueError, match="explicit prior_provenance"):
        calibrate_router_forecast([
            {"epoch": 1.0, "tool": "persistence", "route_matches_oracle": True,
             "dispatcher_err": 0.0,
             "gate_certificate": {"stable_beyond_float32_roundoff": True}},
        ], prior_alpha=2.0, prior_beta=1.0)
    with pytest.raises(ValueError, match="gate certificate"):
        calibrate_router_forecast([
            {"epoch": 1.0, "tool": "persistence", "route_matches_oracle": True,
             "dispatcher_err": 0.0},
        ])


def test_real_205_gate_certificates_and_distribution_blocker():
    traj = read_trajectory(_REAL_RUN)
    backtest = backtest_dispatch(traj, seed=0)
    assert backtest.n_folds == 7
    # MEASURED pathology: ep75 and ep125 sit exactly on the slope==median tie.
    # The fp32/tie contract makes them reproducible, but their margin is honestly zero.
    assert backtest.gate_unstable_fold_count == 2
    assert backtest.gate_min_boundary_margin_ulps == 0.0
    assert all(row["gate_certificate"]["gate_dtype"] == "numpy.float32"
               for row in backtest.fold_rows)
    calibration = backtest.forecast_calibration
    assert calibration["status"] == "MIS_CALIBRATED_INSTANCE"
    assert calibration["stable_bin"]["matches"] == 3
    assert calibration["stable_bin"]["n_folds"] == 5
    assert calibration["roundoff_unstable_bin"]["matches"] == 2
    assert calibration["roundoff_unstable_bin"]["n_folds"] == 2
    assert calibration["terminal_posterior_match_probability"] == pytest.approx(2.0 / 3.0)
    assert calibration["high_minus_low_match_rate"] == pytest.approx(-0.4)
    assert all(row["requested_k"] == 2
               for row in calibration["compute_allocations"])
    report = importance_weighted_architecture_eval(
        backtest.fold_rows, custody=None, clip_bounds=None)
    assert report.status == BLOCKED_DISTRIBUTION_CUSTODY
    assert report.selected_arch is None
    assert "no live/backtest density manifest" in report.blocker
    live = dispatch_for_trajectory(traj, seed=0)
    assert live.classification.meta_lambda_surprise
    assert "meta-λ surprise defer" in live.per_regime_wf_ranking


def test_compiled_costate_controller_owns_replay_and_is_gate(tmp_path):
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1

    organ = derive_costate_agent_v1(_REAL_RUN).compile()
    ledger = tmp_path / "controller_router.jsonl"
    record = organ.dispatch_record(str(ledger), seed=0)
    replay = organ.dispatch_apply(str(ledger), record.decision_id)
    blocked = organ.dispatch_is_backtest(seed=0)
    calibration = organ.dispatch_forecast_calibration(seed=0)
    assert replay.status == "REPLAY_MATCH"
    assert blocked.status == BLOCKED_DISTRIBUTION_CUSTODY
    assert calibration.status == "MIS_CALIBRATED_INSTANCE"
    assert calibration.allocation_verdict.startswith("K2_SHADOW_A_RIDGE_SOLVE")
    assert organ.program.router_stability.is_clip_bounds is None
    assert organ.program.containment.heavy_requires_operator_go is True
