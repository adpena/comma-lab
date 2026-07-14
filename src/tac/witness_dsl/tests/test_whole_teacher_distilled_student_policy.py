from __future__ import annotations

import hashlib

import pytest

from tac.witness_dsl.whole_teacher_distilled_student_policy import (
    MEASUREMENT_AXIS,
    AdmissionState,
    EvidenceTier,
    FidelityGates,
    StudentAdmissionEvidence,
    StudentSize,
    WholeTeacherDistilledStudentPolicy,
    whole_teacher_distilled_student_lever,
)


def _green_evidence(
    policy: WholeTeacherDistilledStudentPolicy | None = None,
    **overrides: object,
) -> StudentAdmissionEvidence:
    selected = policy or WholeTeacherDistilledStudentPolicy()

    def sha(label: str) -> str:
        return hashlib.sha256(label.encode()).hexdigest()

    values: dict[str, object] = {
        "n_pairs": 600,
        "cache_manifest_valid": True,
        "real_rendered_states": True,
        "exact_teacher_quotient_custody": True,
        "exact_teacher_input_vjp_custody": True,
        "actual_r_custody_verified": True,
        "frozen_teacher_custody_verified": True,
        "scalar_objective_custody_verified": True,
        "deterministic_repeat_verified": True,
        "replay_states": ("ep150", "ep251", "ep275"),
        "measured_tier": selected.tier,
        "measured_student_size": selected.selected_size,
        "measured_student_anchor_cadence": selected.student_anchor_cadence,
        "backend": "mlx",
        "teacher_calls": 0,
        "cache_manifest_sha256": sha("cache manifest semantic"),
        "cache_manifest_file_sha256": sha("cache manifest bytes"),
        "cache_validated_sha256": sha("validated cache"),
        "source_custody_sha256": sha("source custody"),
        "teacher_source_custody_sha256": sha("teacher source custody"),
        "actual_r_operator_sha256": sha("actual R"),
        "post_r_input_surface_sha256": sha("post R input surface"),
        "frozen_teacher_weights_sha256": sha("frozen teacher weights"),
        "quotient_basis_sha256": sha("Helmert quotient basis"),
        "scalar_objective_sha256": sha("scalar objective"),
        "fit_policy_sha256": sha("fit policy"),
        "parameter_layout_sha256": sha("parameter layout"),
        "student_parameters_sha256": sha("student parameters"),
        "deterministic_repeat_sha256": sha("deterministic repeat"),
        "measurement_contract_sha256": selected.measurement_contract_sha256(),
        "teacher_timing_receipt_sha256": sha("teacher timing receipt"),
        "forward_worst_pair_cosine": 0.999,
        "forward_worst_pair_relative_l2": 0.01,
        "forward_worst_pair_argmax_disagreement": 0.001,
        "vjp_worst_pair_cosine": 0.98,
        "vjp_worst_pair_relative_l2": 0.10,
        "numpy_framework_forward_worst_pair_cosine": 0.9999,
        "numpy_framework_vjp_worst_pair_cosine": 0.9999,
        "charged_timing_measured": True,
        "student_forward_cost_ms": 5.0,
        "student_forward_vjp_cost_ms": 10.0,
        "exact_teacher_forward_cost_ms": 100.0,
        "exact_teacher_forward_vjp_cost_ms": 200.0,
        "anchor_update_cost_ms": 2.0,
        "student_timing_axis": MEASUREMENT_AXIS,
        "teacher_timing_axis": MEASUREMENT_AXIS,
        "measurement_axis": MEASUREMENT_AXIS,
    }
    values.update(overrides)
    return StudentAdmissionEvidence(**values)  # type: ignore[arg-type]


def test_default_policy_is_research_only_off_and_argv_inert() -> None:
    policy = WholeTeacherDistilledStudentPolicy()
    contract = policy.compile_activation_contract()
    assert policy.enabled is False
    assert policy.research_only is True
    assert contract["target"] == "centered_logit_quotient"
    assert contract["live_trainer_argv"] == []
    assert contract["trainer_activation_admitted"] is False
    assert contract["offline_admission"]["state"] == "BLOCKED_DATA_CUSTODY"
    assert "INPUT-CACHE" in contract["verdict_scope"]
    assert "actual R" in contract["req_R"]
    assert not any(contract["containment"].values())

    lever = whole_teacher_distilled_student_lever(policy)
    assert lever.name == "whole_teacher_distilled_student"
    assert lever.overrides == {}
    assert lever.epochs_delta == 0
    with pytest.raises(ValueError, match="cached-only local measurement"):
        WholeTeacherDistilledStudentPolicy(teacher_recomputation_enabled=True)


def test_student_anchor_cadence_is_independent_of_inner_k2_reuse() -> None:
    policy = WholeTeacherDistilledStudentPolicy(
        student_anchor_cadence=64,
        exact_costate_reuse_kmax=2,
    )
    contract = policy.compile_measurement_contract()
    assert contract["student_anchor_cadence"] == 64
    assert contract["exact_costate_reuse_kmax"] == 2
    assert 20 in contract["student_anchor_cadence_candidates"]
    assert 128 in contract["student_anchor_cadence_candidates"]
    assert "no inherited speed claim" in contract["cadence_composition_law"]

    without_inner_reuse = WholeTeacherDistilledStudentPolicy(
        student_anchor_cadence=64,
        exact_costate_reuse_kmax=None,
    )
    assert without_inner_reuse.student_anchor_cadence == 64
    with pytest.raises(ValueError, match="sealed to #487"):
        WholeTeacherDistilledStudentPolicy(exact_costate_reuse_kmax=3)
    with pytest.raises(ValueError, match="preregistered candidate"):
        WholeTeacherDistilledStudentPolicy(student_anchor_cadence=21)


def test_forward_advisory_tier_does_not_borrow_vjp_authority() -> None:
    policy = WholeTeacherDistilledStudentPolicy(tier=EvidenceTier.FORWARD_ADVISORY)
    evidence = _green_evidence(
        policy,
        exact_teacher_input_vjp_custody=False,
        vjp_worst_pair_cosine=None,
        vjp_worst_pair_relative_l2=None,
    )
    decision = policy.evaluate_evidence(evidence)
    assert decision.state is AdmissionState.GO_READY_NOT_FIRED
    assert decision.admitted is True
    assert decision.tier is EvidenceTier.FORWARD_ADVISORY
    assert policy.compile_activation_contract(evidence)["trainer_activation_admitted"] is False


def test_false_authority_axis_and_duplicate_replay_states_are_refused() -> None:
    policy = WholeTeacherDistilledStudentPolicy(tier=EvidenceTier.FORWARD_ADVISORY)
    mps = _green_evidence(
        policy,
        measurement_axis="[MPS; n600; no score authority]",
        student_timing_axis="[MPS; n600; no score authority]",
        teacher_timing_axis="[MPS; n600; no score authority]",
    )
    assert policy.evaluate_evidence(mps).state is AdmissionState.BLOCKED_DATA_CUSTODY
    duplicate = _green_evidence(policy, replay_states=("ep150", "ep251", "ep275", "ep275"))
    assert policy.evaluate_evidence(duplicate).state is AdmissionState.BLOCKED_DATA_CUSTODY

    plausible_but_unregistered = _green_evidence(
        policy,
        measurement_axis="[n600 other-MLX advisory; no score authority]",
        student_timing_axis="[n600 other-MLX advisory; no score authority]",
        teacher_timing_axis="[n600 other-MLX advisory; no score authority]",
    )
    assert (
        policy.evaluate_evidence(plausible_but_unregistered).state
        is AdmissionState.BLOCKED_DATA_CUSTODY
    )


def test_training_gradient_tier_requires_the_decisive_vjp_gate() -> None:
    policy = WholeTeacherDistilledStudentPolicy(tier=EvidenceTier.TRAINING_GRADIENT)
    no_vjp_custody = _green_evidence(policy, exact_teacher_input_vjp_custody=False)
    assert policy.evaluate_evidence(no_vjp_custody).state is AdmissionState.BLOCKED_DATA_CUSTODY

    wrong_vjp = _green_evidence(
        policy,
        vjp_worst_pair_cosine=0.949,
        vjp_worst_pair_relative_l2=0.251,
    )
    decision = policy.evaluate_evidence(wrong_vjp)
    assert decision.state is AdmissionState.BLOCKED_VJP_FIDELITY
    assert decision.admitted is False
    assert "one Jacobian failure" in decision.verdict_scope
    assert "Sobolev fit" in decision.req_R


def test_worst_pair_forward_and_economics_gates_fail_closed() -> None:
    policy = WholeTeacherDistilledStudentPolicy()
    bad_forward = _green_evidence(policy, forward_worst_pair_relative_l2=0.051)
    forward_decision = policy.evaluate_evidence(bad_forward)
    assert forward_decision.state is AdmissionState.BLOCKED_FORWARD_FIDELITY
    assert "one value-fidelity failure" in forward_decision.verdict_scope
    assert "student architecture" in forward_decision.req_R
    unmeasured_cost = _green_evidence(policy, charged_timing_measured=False)
    assert policy.evaluate_evidence(unmeasured_cost).state is AdmissionState.BLOCKED_ECONOMICS
    nonpaying = _green_evidence(policy, student_forward_vjp_cost_ms=201.0)
    economics_decision = policy.evaluate_evidence(nonpaying)
    assert economics_decision.state is AdmissionState.BLOCKED_ECONOMICS
    assert "one non-paying operating point" in economics_decision.verdict_scope
    assert "matched hardware" in economics_decision.req_R


def test_green_offline_evidence_only_prepares_a_go_packet() -> None:
    policy = WholeTeacherDistilledStudentPolicy(
        enabled=True,
        operator_go_recorded=True,
        selected_size=StudentSize.MEDIUM,
        student_anchor_cadence=32,
    )
    contract = policy.compile_activation_contract(_green_evidence(policy))
    assert contract["offline_admission"]["admitted"] is True
    assert contract["go_packet_state"] == "GO_READY_NOT_FIRED"
    assert contract["trainer_activation_admitted"] is False
    assert contract["trainer_activation_authority"] == "REFUSED_NO_PROVIDER_OR_ARGV"
    assert "live student provider is not integrated" in contract["trainer_activation_errors"]


def test_gate_values_are_validated_and_provenance_is_compiled() -> None:
    with pytest.raises(ValueError, match="argmax disagreement"):
        FidelityGates(forward_worst_pair_max_argmax_disagreement=1.1)
    with pytest.raises(ValueError, match="non-empty provenance"):
        FidelityGates(provenance="")
    contract = WholeTeacherDistilledStudentPolicy().compile_measurement_contract()
    assert contract["gates"]["vjp_worst_pair_min_cosine"] == 0.95
    assert "ASSUMED_AWAITING_VERIFICATION" in contract["gates"]["provenance"]
    assert contract["required_cache_fields"] == {
        "all_tiers": [
            "rendered_frame",
            "teacher_quotient4",
            "labels",
            "per_tensor_sha256",
        ],
        "training_gradient_additional": ["teacher_input_costate"],
    }
    assert contract["semantic_custody_requirements"]["measurement_axis"] == MEASUREMENT_AXIS
    assert "decided by the NumPy-fp32 reference" in contract["numerical_authority"]
    assert "one Jacobian failure" in contract["negative_dispositions"]["BLOCKED_VJP_FIDELITY"][
        "verdict_scope"
    ]


def test_numpy_primary_fidelity_cannot_bypass_separate_mlx_parity_gates() -> None:
    policy = WholeTeacherDistilledStudentPolicy(tier=EvidenceTier.TRAINING_GRADIENT)
    bad_forward_parity = _green_evidence(
        policy,
        forward_worst_pair_cosine=0.999,
        vjp_worst_pair_cosine=0.98,
        numpy_framework_forward_worst_pair_cosine=0.9996,
        numpy_framework_vjp_worst_pair_cosine=0.9999,
    )
    assert (
        policy.evaluate_evidence(bad_forward_parity).state
        is AdmissionState.BLOCKED_FORWARD_FIDELITY
    )

    bad_vjp_parity = _green_evidence(
        policy,
        forward_worst_pair_cosine=0.999,
        vjp_worst_pair_cosine=0.98,
        numpy_framework_forward_worst_pair_cosine=0.9999,
        numpy_framework_vjp_worst_pair_cosine=0.9996,
    )
    assert policy.evaluate_evidence(bad_vjp_parity).state is AdmissionState.BLOCKED_VJP_FIDELITY

    advisory = WholeTeacherDistilledStudentPolicy(tier=EvidenceTier.FORWARD_ADVISORY)
    forward_only = _green_evidence(
        advisory,
        exact_teacher_input_vjp_custody=False,
        vjp_worst_pair_cosine=None,
        vjp_worst_pair_relative_l2=None,
        numpy_framework_vjp_worst_pair_cosine=None,
    )
    assert advisory.evaluate_evidence(forward_only).state is AdmissionState.GO_READY_NOT_FIRED


def test_receipt_binding_and_raw_charged_economics_are_policy_specific() -> None:
    policy = WholeTeacherDistilledStudentPolicy(
        selected_size=StudentSize.MEDIUM,
        student_anchor_cadence=32,
    )
    evidence = _green_evidence(policy)
    economics = evidence.derived_economics_for(policy)
    assert economics["charged_cost_ms"] == pytest.approx(10.0 + (200.0 + 2.0) / 32)
    assert economics["strict_pays"] is True
    assert economics["inclusive_95"] is False
    assert policy.evaluate_evidence(evidence).state is AdmissionState.GO_READY_NOT_FIRED

    wrong_k = _green_evidence(policy, measured_student_anchor_cadence=64)
    assert policy.evaluate_evidence(wrong_k).state is AdmissionState.BLOCKED_DATA_CUSTODY
    wrong_policy_hash = _green_evidence(
        policy,
        measurement_contract_sha256=hashlib.sha256(b"wrong policy").hexdigest(),
    )
    assert policy.evaluate_evidence(wrong_policy_hash).state is AdmissionState.BLOCKED_DATA_CUSTODY
    missing_semantics = _green_evidence(policy, actual_r_custody_verified=False)
    assert policy.evaluate_evidence(missing_semantics).state is AdmissionState.BLOCKED_DATA_CUSTODY
    mismatched_axis = _green_evidence(
        policy,
        teacher_timing_axis="[different MLX device; n600; no score authority]",
    )
    assert policy.evaluate_evidence(mismatched_axis).state is AdmissionState.BLOCKED_ECONOMICS

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        StudentAdmissionEvidence(cache_manifest_sha256="a" * 64)
