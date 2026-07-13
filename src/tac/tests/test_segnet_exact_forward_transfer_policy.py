# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from tac.witness_dsl.segnet_exact_forward_transfer_policy import (
    SegNetExactForwardTransferPolicy,
)


def _policy(**overrides: object) -> SegNetExactForwardTransferPolicy:
    values: dict[str, object] = {
        "physical_core_count": 18,
        "torch_default_intraop_threads": 6,
        "logical_core_count": 24,
        "batch_size": 1,
        "channels": 3,
        "height": 384,
        "width": 512,
    }
    values.update(overrides)
    return SegNetExactForwardTransferPolicy(**values)  # type: ignore[arg-type]


def test_thread_candidates_are_derived_and_do_not_encode_a_winner() -> None:
    policy = _policy()
    assert policy.effective_thread_ceiling == 6
    assert policy.candidate_threads == tuple(range(1, 7))
    contract = policy.compile_measurement_contract()
    assert "selected_threads" not in contract
    assert contract["fallback"] == "torch_default_intraop_threads"


def test_forward_size_and_canary_count_are_derived() -> None:
    policy = _policy()
    assert policy.forward_work_elements == 1 * 3 * 384 * 512
    assert policy.heuristic_canary_count == 3
    contract = policy.compile_measurement_contract()
    assert contract["canary_count"] == 3
    assert contract["canary_count_authority"] == "ASSUMED_HEURISTIC_SCREEN_ONLY"
    assert contract["verdict_pair_cardinality"] == 600
    assert contract["live_trainer_argv"] == []


def test_runtime_capacity_intersection_adapts_to_host() -> None:
    policy = _policy(
        physical_core_count=4,
        logical_core_count=8,
        torch_default_intraop_threads=12,
    )
    assert policy.effective_thread_ceiling == 4
    assert policy.candidate_threads == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "field",
    [
        "physical_core_count",
        "torch_default_intraop_threads",
        "batch_size",
        "channels",
        "height",
        "width",
    ],
)
def test_positive_integer_contract(field: str) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _policy(**{field: 0})


def test_authority_escalation_fails_closed() -> None:
    with pytest.raises(ValueError, match="research-only"):
        _policy(score_claim=True)

    with pytest.raises(ValueError, match="advisory authority"):
        _policy(contest_cpu_measured=True)


def test_supported_strategy_domain_is_runtime_consumable() -> None:
    assert SegNetExactForwardTransferPolicy.supported_strategies() == (
        "eager_nchw_autograd",
        "eager_channels_last_autograd",
    )


def test_static_process_abba_dual_replay_contract_is_explicit() -> None:
    contract = _policy().compile_measurement_contract()
    lifecycle = contract["process_lifecycle"]
    assert lifecycle == {
        "method": "fresh_child_process_static_threads",
        "parent_canary_is_terminal_evidence": False,
        "stage_order": ["baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1"],
        "measurement_children_per_stage": 1,
        "independent_replay_children_per_stage": 1,
        "bind_intraop_before_model_load": True,
        "bind_interop_before_model_load": True,
        "mid_pass_thread_mutation_forbidden": True,
        "four_way_measurement_sequence_sha_equality_required": True,
        "measurement_replay_sequence_sha_equality_required": True,
        "n600_only_go": True,
    }
    assert contract["checkpoint_interval_pairs"] == 25
    assert contract["checkpoint_interval_provenance"].startswith("ASSUMED_")
    assert contract["matched_sign_alpha_provenance"].startswith("OPERATOR_SEALED_")
    assert contract["pointer_moved"] is False
    assert contract["contest_cpu_measured"] is False
    assert contract["mps_used"] is False
    assert contract["cuda_used"] is False


def test_logical_capacity_is_part_of_the_derived_intersection() -> None:
    policy = _policy(physical_core_count=18, logical_core_count=4, torch_default_intraop_threads=6)
    assert policy.effective_thread_ceiling == 4
    assert policy.compile_measurement_contract()["candidate_threads"] == [1, 2, 3, 4]
