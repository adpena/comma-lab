from __future__ import annotations

import pytest

from tac.scorer_surrogate.onpolicy_costate import FULL_BUILD_BLOCKER
from tac.witness_dsl.onpolicy_scorer_surrogate_policy import OnPolicyScorerSurrogatePolicy


def test_policy_derives_95_percent_skip_cadence_and_emits_no_trainer_flag() -> None:
    policy = OnPolicyScorerSurrogatePolicy()
    assert policy.derived_target_cadence == 20
    assert policy.measurement_horizon == 40
    assert policy.measured_cadences == (1, 4, 20)
    contract = policy.compile_measurement_contract()
    assert contract["live_trainer_argv"] == []
    assert contract["admission_predicate"] == {
        "exact_cycle_ce_descent": True,
        "nonnegative_costate_cosine": True,
        "sequence_endpoint_dseg_nonworsening": True,
        "sequence_endpoint_dpose_nonworsening": True,
    }
    assert contract["capacity_control_law"] == "constant first-probe defaults: hidden=8; fit_steps=8"
    assert contract["capacity_default_status"] == "ASSUMED_NOT_DERIVED"
    assert contract["capacity_recess_measurement"] == "shared_horizon_width_fit_grid_hidden_4_8_16_steps_4_8_16"
    assert contract["cadence_recess_measurement"] == "K4_cadence_interpolation_canary"
    assert contract["research_only"] is True
    assert contract["full_build_blocker"] == FULL_BUILD_BLOCKER


def test_policy_cannot_launder_partial_scaffold_as_nonresearch() -> None:
    with pytest.raises(ValueError, match="remains research-only"):
        OnPolicyScorerSurrogatePolicy(research_only=False)


def test_corrected_contract_self_derives_dense_matched_window_and_emits_no_live_argv() -> None:
    policy = OnPolicyScorerSurrogatePolicy()
    contract = policy.compile_corrected_measurement_contract()

    assert policy.matched_window_steps == 5
    assert policy.decisive_window_steps == 20
    assert policy.frame_channels == 3
    assert policy.amortized_hidden_channels == 16
    assert policy.branch_kernel_sizes == (3, 5)
    assert policy.dense_optimizer_steps_per_observation == 2
    assert policy.dense_ema_decay == pytest.approx(0.8)
    assert contract["target_anchor_cadence"] == 20
    assert contract["matched_window_steps"] == 5
    assert contract["decisive_window_steps"] == 20
    assert contract["admissible_window_steps"] == [5, 20]
    assert contract["dense_collection_steps"] == 5
    assert contract["architecture"] == {
        "frame_channels": 3,
        "input_channels": 9,
        "hidden_channels": 16,
        "branch_kernel_sizes": [3, 5],
        "frame_value_scale": 255.0,
        "normalization_floor": pytest.approx(2.0**-23),
        "mse_weight": 1.0,
        "cosine_weight": 1.0,
        "ema_decay": pytest.approx(0.8),
        "admission_min_relative_improvement": 0.0,
    }
    assert contract["common_control_schedule_required"] is True
    assert contract["exact_metric_trace_every_step"] is True
    assert contract["ema_provider_is_admission_authority"] is True
    assert contract["resume_preserves_anchor_frame_and_costate"] is True
    assert contract["live_trainer_argv"] == []
    assert contract["research_only"] is True
    assert contract["score_claim"] is False
