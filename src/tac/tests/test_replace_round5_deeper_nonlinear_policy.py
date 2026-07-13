# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from tac.witness_dsl.replace_round5_deeper_nonlinear_policy import (
    REALIZED_AREA_FRACTION,
    ReplaceRound5DeeperNonlinearPolicy,
)


def test_round5_contract_closes_counts_and_preregistered_gates() -> None:
    policy = ReplaceRound5DeeperNonlinearPolicy()
    contract = policy.compile_measurement_contract()
    assert policy.train_state_count == 480
    assert policy.heldout_state_count == 120
    assert policy.nonlinear_core_state_count == 420
    assert policy.nonlinear_dev_state_count == 60
    assert contract["realized_area_fraction"] == REALIZED_AREA_FRACTION
    assert contract["branch_horizon_ticket"]["current_fixed_replay_status"] == (
        "blocked-not-identified"
    )
    assert contract["query_real_ticket"]["total_fraction"] == 0.05
    assert contract["live_trainer_argv"] == []


def test_round5_policy_refuses_unpreregistered_changes() -> None:
    with pytest.raises(ValueError, match="seed"):
        ReplaceRound5DeeperNonlinearPolicy(nonlinear_seeds=(455, 456, 999))
    with pytest.raises(ValueError, match="call budget"):
        ReplaceRound5DeeperNonlinearPolicy(teacher_started_call_budget=601)
    with pytest.raises(ValueError, match="FORE"):
        ReplaceRound5DeeperNonlinearPolicy(fore_weighting_enabled=True)
