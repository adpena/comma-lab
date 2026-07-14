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
            "precision_assignment": f"uniform_W{bits}A{bits}",
            "activation_scale_mode": "dynamic_exact_absmax",
            "qdq_receipt_fingerprint": "b" * 64,
        },
        "summary": summary,
    }


def _mixed_integer_scorer() -> dict[str, object]:
    indices_hash = "d" * 64
    return {
        "schema": "mixed_int64_fixedpoint_scorer_n600.v1",
        "fingerprint": "c" * 64,
        "contract": {
            "native_integer_speed_claim": True,
            "activation_scale_mode": "dynamic_exact_absmax",
        },
        "custody": {"qdq_precursor_fingerprint": "b" * 64},
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 30,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": "largest_geometry_safe_bits_with_signed_int64_static_bound",
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "minimum_bits": 26,
            "maximum_bits": 30,
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
        },
    }


def _mixed_metal(**overrides: object) -> dict[str, object]:
    receipt = _metal(bits=26, **overrides)
    receipt["contract"]["precision_assignment"] = "geometry_safe_W26_to_W30"  # type: ignore[index]
    receipt["contract"]["exact_int64_cpu_precursor_fingerprint"] = "c" * 64  # type: ignore[index]
    return receipt


def _weight_l1_integer_scorer() -> dict[str, object]:
    receipt = _mixed_integer_scorer()
    receipt["schema"] = "weight_l1_int64_fixedpoint_scorer_n600.v1"
    receipt["summary"]["maximum_bits"] = 31  # type: ignore[index]
    receipt["model_manifest"]["maximum_bits"] = 31  # type: ignore[index]
    receipt["model_manifest"].update(  # type: ignore[union-attr]
        {
            "assignment_rule": "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound",
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
            "precision_histogram": {"27": 4, "28": 28, "29": 32, "30": 41, "31": 20},
        }
    )
    return receipt


def _weight_l1_metal(**overrides: object) -> dict[str, object]:
    receipt = _mixed_metal(**overrides)
    receipt["contract"]["bits"] = 27  # type: ignore[index]
    receipt["contract"]["precision_assignment"] = (  # type: ignore[index]
        "frozen_weight_l1_safe_W27_to_W31"
    )
    return receipt


def _tie_snap_integer_scorer() -> dict[str, object]:
    receipt = _weight_l1_integer_scorer()
    epsilon = 2.0**-19
    receipt["schema"] = "weight_l1_tie_snap_scorer_n600.v1"
    receipt["contract"].update(  # type: ignore[union-attr]
        {
            "decision_rule": "lowest class index within epsilon of candidate maximum",
            "epsilon_selection": (
                "minimum calibration-exact epsilon; no heldout reselection"
            ),
            "runtime_label_or_frame_dependent": False,
        }
    )
    receipt["summary"].update(  # type: ignore[union-attr]
        {
            "minimum_calibration_exact_arm": "epsilon_2m19",
            "minimum_calibration_exact_epsilon": epsilon,
            "selected_heldout_exact": True,
            "selected_full_exact": True,
        }
    )
    return receipt


def _tie_snap_metal(**overrides: object) -> dict[str, object]:
    receipt = _weight_l1_metal(**overrides)
    receipt["contract"]["precision_assignment"] = (  # type: ignore[index]
        f"frozen_weight_l1_safe_W27_to_W31_tie_snap_{float(2.0**-19).hex()}"
    )
    return receipt


def _class_pair_tie_snap_integer_scorer() -> dict[str, object]:
    receipt = _weight_l1_integer_scorer()
    receipt["schema"] = "weight_l1_class_pair_tie_snap_scorer_n600.v1"
    receipt["contract"].update(  # type: ignore[union-attr]
        {
            "design_split": [0, 264],
            "second_validation_split": [264, 600],
            "epsilon": 2.0**-19,
            "candidate_winner_class": 4,
            "candidate_runner_class": 0,
            "replacement_class": 0,
            "rule_frozen_before_second_validation_access": True,
            "second_validation_reselection": False,
            "runtime_label_or_frame_dependent": False,
        }
    )
    receipt["summary"].update(  # type: ignore[union-attr]
        {"design_exact": True, "second_validation_exact": True}
    )
    return receipt


def _class_pair_tie_snap_metal(**overrides: object) -> dict[str, object]:
    receipt = _weight_l1_metal(**overrides)
    receipt["contract"]["precision_assignment"] = (  # type: ignore[index]
        "frozen_weight_l1_safe_W27_to_W31_class_pair_tie_snap_w4_r0_to0_eps_"
        f"{float(2.0**-19).hex()}"
    )
    return receipt


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
    assert "actual evolving witness frames" in str(metal.required_next_gate)
    assert integer_r.state is AssignmentState.DEFAULT_OFF_CANDIDATE


def test_exact_mixed_integer_precursor_can_replace_nonexact_qdq_gate() -> None:
    qdq = _qdq(bits=26)
    qdq["summary"]["minimum_argmax_exact_arm"] = None  # type: ignore[index]
    qdq["summary"]["arms"]["w26a26"]["argmax_exact_admitted"] = False  # type: ignore[index]
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        integer_scorer_receipt=_mixed_integer_scorer(),
        metal_fixedpoint_receipt=_mixed_metal(strict_interval_certified=False),
    )
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.DEFAULT_OFF_CANDIDATE
    assert metal.selected_bits == 26
    assert "exact-int64 scorer admits geometry_safe_W26_to_W30" in metal.evidence


def test_weight_l1_integer_precursor_is_typed_and_label_free() -> None:
    qdq = _qdq(bits=26)
    qdq["summary"]["minimum_argmax_exact_arm"] = None  # type: ignore[index]
    qdq["summary"]["arms"]["w26a26"]["argmax_exact_admitted"] = False  # type: ignore[index]
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        integer_scorer_receipt=_weight_l1_integer_scorer(),
        metal_fixedpoint_receipt=_weight_l1_metal(strict_interval_certified=False),
    )
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.DEFAULT_OFF_CANDIDATE
    assert metal.selected_bits == 27
    assert "frozen_weight_l1_safe_W27_to_W31" in metal.evidence


def test_calibration_selected_tie_snap_precursor_binds_metal_decision_head() -> None:
    qdq = _qdq(bits=26)
    qdq["summary"]["minimum_argmax_exact_arm"] = None  # type: ignore[index]
    qdq["summary"]["arms"]["w26a26"]["argmax_exact_admitted"] = False  # type: ignore[index]
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        integer_scorer_receipt=_tie_snap_integer_scorer(),
        metal_fixedpoint_receipt=_tie_snap_metal(),
    )
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.DEFAULT_OFF_CANDIDATE
    assert "tie_snap_0x1.0000000000000p-19" in metal.evidence


def test_frozen_class_pair_rule_requires_disjoint_second_validation() -> None:
    qdq = _qdq(bits=26)
    qdq["summary"]["minimum_argmax_exact_arm"] = None  # type: ignore[index]
    qdq["summary"]["arms"]["w26a26"]["argmax_exact_admitted"] = False  # type: ignore[index]
    scorer = _class_pair_tie_snap_integer_scorer()
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        integer_scorer_receipt=scorer,
        metal_fixedpoint_receipt=_class_pair_tie_snap_metal(),
    )
    metal = _find(policy, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL)
    assert metal.state is AssignmentState.DEFAULT_OFF_CANDIDATE
    assert metal.selected_bits == 27
    assert "class_pair_tie_snap_w4_r0_to0" in metal.evidence
    scorer["summary"]["second_validation_exact"] = False  # type: ignore[index]
    held = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        integer_scorer_receipt=scorer,
        metal_fixedpoint_receipt=_class_pair_tie_snap_metal(),
    )
    assert (
        _find(held, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL).state
        is AssignmentState.HELD_OWED
    )
    scorer["summary"]["second_validation_exact"] = True  # type: ignore[index]
    scorer["model_manifest"]["precision_histogram"] = {"25": 125}  # type: ignore[index]
    out_of_range = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        integer_scorer_receipt=scorer,
        metal_fixedpoint_receipt=_class_pair_tie_snap_metal(),
    )
    assert (
        _find(out_of_range, Operation.SEGNET_VERDICT, Substrate.CUSTOM_METAL).state
        is AssignmentState.HELD_OWED
    )


@pytest.mark.parametrize(
    "failed_gate",
    [
        "complete",
        "full_real_n600",
        "cross_process_argmax_identical",
        "argmax_exact",
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
