# SPDX-License-Identifier: MIT
"""Tests for the #515 constants+telemetry build wave (DERIVED levers + B0 instruments)."""
from __future__ import annotations

import math

import pytest

from tac.witness_dsl.constants_telemetry_build_wave_20260715 import (
    BANKED_R1_DPOSE,
    BUILD_WAVE_BATTERY,
    BatteryArm,
    BuildWaveManifest,
    CLASS4_WAIVERS,
    DerivedAdamBeta2,
    DerivedEmaDecay,
    DerivedEvalEvery,
    DerivedWPoseAtEngage,
    HardcodedWaiverCustody,
    ModDimDynamicsOn,
    TRAINER_WIREIN_QUEUE,
    TrainerWireInQueued,
    VerdictBatch64,
    WeightNormTelemetryRow,
    derived_eval_every,
    derived_w_pose_at_engage,
    log_midpoint_decay,
    populate_build_wave_evaluators,
    stationarity_window_steps,
)


# ---------------------------------------------------------------------- laws
def test_stationarity_window_matches_beta2_law():
    n_lo, n_hi = stationarity_window_steps(75, 100.0)
    assert n_lo == pytest.approx(75.0)          # one data cycle (floor)
    assert n_hi == pytest.approx(2500.0)        # T_c*S/3 (ceiling)


def test_log_midpoint_decay_value():
    # N* = sqrt(75*2500) ~= 433 => decay ~= 0.99769
    assert log_midpoint_decay(75, 100.0) == pytest.approx(1.0 - 1.0 / math.sqrt(187500.0))
    assert log_midpoint_decay(75, 100.0) == pytest.approx(0.99769, abs=1e-5)


def test_log_midpoint_inside_window():
    n_lo, n_hi = stationarity_window_steps(75, 100.0)
    n_star = 1.0 / (1.0 - log_midpoint_decay(75, 100.0))
    assert n_lo < n_star < n_hi


def test_derived_w_pose_matches_costate_law():
    # lambda_pose = 5/sqrt(10*0.001610) = 39.405 => ratio 0.394
    expected = (5.0 / math.sqrt(10.0 * BANKED_R1_DPOSE)) / 100.0
    assert derived_w_pose_at_engage() == pytest.approx(expected)
    assert derived_w_pose_at_engage() == pytest.approx(0.394, abs=1e-3)


def test_derived_w_pose_rejects_nonpositive_dpose():
    with pytest.raises(ValueError):
        derived_w_pose_at_engage(0.0)


def test_derived_eval_every_conditional_on_vpw():
    # WITH VPW(8): amortization derives 19 < the 25 information floor => 25.
    assert derived_eval_every(verdict_parallel_workers=8) == 25
    # WITHOUT VPW: 900/(0.10*325) = 27.7 => 28 (the incumbent 25 is over-budget).
    assert derived_eval_every(verdict_parallel_workers=0) == 28


def test_derived_eval_every_rejects_bad_economics():
    with pytest.raises(ValueError):
        derived_eval_every(verdict_inflation_s=0.0)


# -------------------------------------------------------------------- levers
def test_derived_adam_beta2_lever_custody():
    lv = DerivedAdamBeta2()
    assert lv.overrides["--adam-beta2"] == pytest.approx(0.997691, abs=1e-6)
    ref = lv.lawrefs["--adam-beta2"]
    assert ref.equation_id == "stationarity_window_log_midpoint_v1"
    assert ref.ladder_class == "derived_at_config"
    manifest = lv.constant_manifest["--adam-beta2"]
    assert manifest["equation_id"] == "stationarity_window_log_midpoint_v1"
    assert manifest["fallback_used"] is False


def test_derived_ema_decay_lever_emits_derived_not_ancestor():
    lv = DerivedEmaDecay()
    # emits the DERIVED value, never the inherited Quantizr 0.997 literal
    assert lv.overrides["--ema-decay"] != 0.997
    assert lv.overrides["--ema-decay"] == pytest.approx(0.997691, abs=1e-6)
    assert lv.lawrefs["--ema-decay"].ladder_class == "derived_at_config"


def test_derived_w_pose_lever():
    lv = DerivedWPoseAtEngage()
    assert lv.overrides["--w-pose"] == pytest.approx(0.3941, abs=1e-4)
    assert lv.lawrefs["--w-pose"].equation_id == "costate_w_pose_engage_ratio_v1"


def test_derived_eval_every_lever_confirms_incumbent():
    lv = DerivedEvalEvery()
    assert lv.overrides["--eval-every"] == 25
    assert "conditional" in lv.notes.lower() or "CONFIRMED" in lv.notes


def test_verdict_batch_64_lever_measured_anchor():
    lv = VerdictBatch64()
    assert lv.overrides["--verdict-batch"] == 64
    assert lv.lawrefs["--verdict-batch"].ladder_class == "measured_anchor"


def test_mod_dim_dynamics_lever_explicit_true():
    lv = ModDimDynamicsOn()
    assert lv.overrides["--mod-dim-dynamics"] is True


def test_weight_norm_lever_fail_closed_until_trainer_flag_lands():
    # The trainer flag has not landed (queued behind the live dry-start):
    # composing the lever must FAIL CLOSED with the insertion point named.
    with pytest.raises(TrainerWireInQueued, match="insertion point"):
        WeightNormTelemetryRow()


def test_lever_flags_exist_in_live_trainer_parser():
    """never-invent-flags: every composable lever's flags are real trainer flags."""
    from tac.witness_dsl.curriculum_dsl import real_trainer_flags

    trainer = real_trainer_flags(None)
    for factory in (DerivedAdamBeta2, DerivedEmaDecay, DerivedWPoseAtEngage,
                    DerivedEvalEvery, VerdictBatch64, ModDimDynamicsOn):
        for flag in factory().overrides:
            assert flag in trainer, f"{factory.__name__} emits unknown flag {flag}"


# ----------------------------------------------------------- waivers/battery
def test_class4_waivers_typed_and_complete():
    assert len(CLASS4_WAIVERS) == 5
    for w in CLASS4_WAIVERS:
        assert isinstance(w, HardcodedWaiverCustody)
        assert w.reason and w.owner and w.rederivation_trigger and w.battery_arm


def test_waiver_rejects_placeholder_rationale():
    with pytest.raises(ValueError, match="non-placeholder"):
        HardcodedWaiverCustody(
            constant="--x", value="1", reason="<reason>", owner="o",
            rederivation_trigger="t", battery_arm="B9",
        )


def test_battery_arm_ids_unique_and_derived_arms_present():
    ids = [a.arm_id for a in BUILD_WAVE_BATTERY]
    assert len(ids) == len(set(ids))
    joined = " ".join(" ".join(a.arms) for a in BUILD_WAVE_BATTERY)
    for lever_name in ("DerivedEmaDecay", "DerivedAdamBeta2", "DerivedWPoseAtEngage",
                       "VerdictBatch64", "WeightNormTelemetryRow"):
        assert lever_name in joined, f"{lever_name} not folded into the battery"


def test_battery_arm_validation():
    with pytest.raises(ValueError, match="non-empty"):
        BatteryArm(arm_id="Bx", dimension="d", arms=("a",), metric="m",
                   falsification="f", scale_cost="c", order_gate=" ")
    with pytest.raises(ValueError, match="arms"):
        BatteryArm(arm_id="Bx", dimension="d", arms=(), metric="m",
                   falsification="f", scale_cost="c", order_gate="g")


def test_manifest_containment_and_contract():
    m = BuildWaveManifest()
    assert m.research_only and not m.score_claim and not m.promotion_eligible
    contract = m.compile_contract()
    assert contract["schema"] == "constants_telemetry_build_wave.v1"
    assert len(contract["waivers"]) == 5
    assert len(contract["battery"]) == len(BUILD_WAVE_BATTERY)
    with pytest.raises(ValueError, match="actuation"):
        BuildWaveManifest(live_training_enabled=True)
    with pytest.raises(ValueError, match="MEANS"):
        BuildWaveManifest(score_claim=True)


def test_trainer_wirein_queue_names_pid_and_insertion_points():
    assert len(TRAINER_WIREIN_QUEUE) == 4
    wnt = TRAINER_WIREIN_QUEUE[0]
    # the weight_norm telemetry row remains queued behind the live dry-start.
    assert "31576" in wnt["status"] or "dry-start" in wnt["status"]
    for row in TRAINER_WIREIN_QUEUE:
        assert row["insertion_point"].strip()
        assert row["producer"].strip()
    # the rate-rolling telemetry producer (#408/#404 FEED-ratetelemetry) is now LANDED on the
    # p0_328_408 merge-window branch (trainer flag + emission + resume wired; merges post-v9c2).
    rate = TRAINER_WIREIN_QUEUE[-1]
    assert "rate_rolling" in rate["producer"]
    assert "landed" in rate["status"].lower()
    assert "p0_328_408" in rate["status"]


def test_evaluator_registration_idempotent():
    first = populate_build_wave_evaluators()
    second = populate_build_wave_evaluators()
    assert first == second
    assert "stationarity_window_log_midpoint_v1" in first
