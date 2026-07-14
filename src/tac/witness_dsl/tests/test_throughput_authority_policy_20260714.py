from __future__ import annotations

import pytest

from tac.witness_dsl.throughput_authority_policy_20260714 import (
    AssignmentState,
    Operation,
    Substrate,
    compile_throughput_authority_policy,
)


def _qdq(
    bits: int = 22, *, activation_scale_mode: str = "dynamic_exact_absmax"
) -> dict[str, object]:
    arm = f"w{bits}a{bits}"
    indices_hash = "a" * 64
    return {
        "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
        "fingerprint": "b" * 64,
        "contract": {
            "native_integer_speed_claim": False,
            "activation_scale_mode": activation_scale_mode,
        },
        "summary": {
            "full_real_n600": True,
            "minimum_argmax_exact_arm": arm,
            "arms": {arm: {"argmax_exact_admitted": True}},
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
        },
    }


def _metal(bits: int = 22, **overrides: object) -> dict[str, object]:
    summary = {
        "complete": True,
        "full_real_n600": True,
        "cross_process_argmax_identical": True,
        "argmax_exact": True,
        "strict_interval_certified": True,
        "positive_speed": True,
        "admitted_candidate_authority_filter": True,
    }
    summary.update(overrides)
    return {
        "schema": "metal_fixedpoint_segnet_n600.v1",
        "contract": {
            "bits": bits,
            "activation_scale_mode": "dynamic_exact_absmax",
            "qdq_receipt_fingerprint": "b" * 64,
        },
        "summary": summary,
    }


def _integer_r(admitted: bool = True) -> dict[str, object]:
    return {
        "schema": "integer_r_adjoint_backend_benchmark.v1",
        "admission": {"admitted_for_training": admitted},
    }


def _find(policy, operation: Operation, substrate: Substrate):
    return next(
        row
        for row in policy.assignments
        if row.operation is operation and row.substrate is substrate
    )


def test_empty_receipts_fail_closed_but_keep_cpu_authority() -> None:
    policy = compile_throughput_authority_policy()
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    cpu = _find(policy, Operation.SEGNET_VERDICT, Substrate.TORCH_CPU_ONE_THREAD)
    integer_r = _find(policy, Operation.R_ADJOINT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.HELD_OWED
    assert cpu.state is AssignmentState.ACTIVE
    assert integer_r.state is AssignmentState.HELD_OWED


def test_full_conjunction_unlocks_default_off_candidates() -> None:
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=_qdq(),
        metal_fixedpoint_receipt=_metal(),
        integer_r_receipt=_integer_r(),
    )
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    integer_r = _find(policy, Operation.R_ADJOINT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.DEFAULT_OFF_CANDIDATE
    assert metal.selected_bits == 22
    assert metal.activation_scale_mode == "dynamic_exact_absmax"
    assert integer_r.state is AssignmentState.DEFAULT_OFF_CANDIDATE


@pytest.mark.parametrize(
    "failed_gate",
    [
        "complete",
        "full_real_n600",
        "cross_process_argmax_identical",
        "argmax_exact",
        "strict_interval_certified",
        "positive_speed",
        "admitted_candidate_authority_filter",
    ],
)
def test_each_metal_gate_is_load_bearing(failed_gate: str) -> None:
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=_qdq(),
        metal_fixedpoint_receipt=_metal(**{failed_gate: False}),
    )
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.HELD_OWED
    assert failed_gate in metal.evidence


def test_qdq_never_claims_native_integer_speed() -> None:
    receipt = _qdq()
    receipt["contract"]["native_integer_speed_claim"] = True  # type: ignore[index]
    policy = compile_throughput_authority_policy(fixedpoint_qdq_receipt=receipt)
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.HELD_OWED
    assert "disclaim" in metal.evidence


def test_ane_w8a8_is_settled_formulation_refusal() -> None:
    policy = compile_throughput_authority_policy()
    ane = _find(policy, Operation.SEGNET_VERDICT, Substrate.COREML_ANE)
    assert ane.state is AssignmentState.FORBIDDEN
    assert "45.836809%" in ane.evidence


def test_pose_gate_and_inputs_are_typed() -> None:
    policy = compile_throughput_authority_policy(
        pose_gate_enabled=False, pose_canary_every=3, banked_r1_dpose=0.2
    )
    row = _find(policy, Operation.POSENET_VERDICT_PRE_FINISH, Substrate.BANKED_TELEMETRY)
    assert row.state is AssignmentState.DEFAULT_OFF_CANDIDATE
    assert policy.pose_canary_every == 3
    with pytest.raises(ValueError, match="positive integer"):
        compile_throughput_authority_policy(pose_canary_every=0)
    with pytest.raises(ValueError, match="finite"):
        compile_throughput_authority_policy(banked_r1_dpose=float("nan"))


def test_policy_serialization_is_explicitly_non_authority_for_score() -> None:
    payload = compile_throughput_authority_policy().to_dict()
    assert payload["schema"] == "throughput_authority_policy.v1"
    assert payload["research_only"] is True
    assert payload["score_claim"] is False
    assert payload["pointer_moved"] is False
