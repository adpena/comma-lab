from __future__ import annotations

import math

import pytest

from tac.canonical_equations.ddm_cn5_arc_consolidation_20260813 import (
    LATTICE_EXPONENT_EQUATION_ID,
    POSE_BUDGET_EQUATION_ID,
    POSE_SEMANTIC_EQUATION_ID,
    STEP_RESOLVABILITY_EQUATION_ID,
    build_ddm_cn5_arc_equations,
    build_local_model_step_resolvability_equation,
    build_pose_stack_exact_budget_equation,
    build_receiver_lattice_leakage_exponent_equation,
    build_receiver_pose_semantic_preservation_equation,
    populate_ddm_cn5_arc_equations,
)
from tac.canonical_equations.evaluators import (
    EvaluatorError,
    eval_local_model_step_resolvability_ratio,
    eval_pose_stack_exact_budget,
    eval_receiver_lattice_leakage_exponent,
    eval_receiver_pose_semantic_preservation_ratio,
    populate_lawref_evaluators,
    resolve_equation_value,
)
from tac.canonical_equations.registry import query_equations


def _sidecar(tmp_path, name: str = "receipt.md"):
    path = tmp_path / name
    path.write_text("receipt\n", encoding="utf-8")
    return path


def test_step_resolvability_ratio_matches_po1_most_charitable_step() -> None:
    assert eval_local_model_step_resolvability_ratio(
        {"predicted_step_magnitude": -1.0e-8, "forward_mismatch_floor": 9.36e-6}
    ) == pytest.approx(1.0e-8 / 9.36e-6)


@pytest.mark.parametrize("floor", [0.0, -1.0, math.inf])
def test_step_resolvability_refuses_invalid_floor(floor: float) -> None:
    with pytest.raises(EvaluatorError, match="forward_mismatch_floor"):
        eval_local_model_step_resolvability_ratio({"predicted_step_magnitude": 1.0, "forward_mismatch_floor": floor})


def test_pose_semantic_ratio_matches_pz4r_collapse() -> None:
    assert eval_receiver_pose_semantic_preservation_ratio(
        {
            "base_d_pose": 0.00014746535453014076,
            "candidate_d_pose": 0.6310142278671265,
        }
    ) == pytest.approx(4279.06764865338)


@pytest.mark.parametrize("base", [0.0, -1.0, math.inf])
def test_pose_semantic_ratio_refuses_invalid_base(base: float) -> None:
    with pytest.raises(EvaluatorError, match="base_d_pose"):
        eval_receiver_pose_semantic_preservation_ratio({"base_d_pose": base, "candidate_d_pose": 0.0})


def test_exact_pose_budget_matches_hv1_corrected_js7_budget() -> None:
    assert eval_pose_stack_exact_budget(
        {
            "base_d_pose": 6.885642960696714e-6,
            "seg_credit_s": 0.000960,
            "archive_delta_bytes": 323,
        }
    ) == pytest.approx(1.291770121176113e-6)


def test_exact_pose_budget_includes_rate_saving_as_allowance() -> None:
    without_saving = eval_pose_stack_exact_budget(
        {"base_d_pose": 1.0e-5, "seg_credit_s": 0.001, "archive_delta_bytes": 0}
    )
    with_saving = eval_pose_stack_exact_budget(
        {"base_d_pose": 1.0e-5, "seg_credit_s": 0.001, "archive_delta_bytes": -100}
    )
    assert with_saving > without_saving


def test_exact_pose_budget_refuses_when_bytes_consume_all_credit() -> None:
    with pytest.raises(EvaluatorError, match="byte cost exceeds Seg credit"):
        eval_pose_stack_exact_budget({"base_d_pose": 1.0e-5, "seg_credit_s": 0.0, "archive_delta_bytes": 1})


def test_lattice_exponent_recovers_exact_power_law() -> None:
    amplitudes = [1.0, 0.5, 0.25, 0.125]
    leakages = [3.0 * amplitude**2.5 for amplitude in amplitudes]
    assert eval_receiver_lattice_leakage_exponent({"amplitudes": amplitudes, "leakages": leakages}) == pytest.approx(
        2.5
    )


def test_lattice_exponent_matches_js5_receiver_fit() -> None:
    assert eval_receiver_lattice_leakage_exponent(
        {
            "amplitudes": [1.0, 0.5, 0.25, 0.125, 0.0625],
            "leakages": [
                8.5741907e-4,
                1.3687468e-4,
                6.6803035e-5,
                4.2786291e-5,
                5.5702489e-5,
            ],
        }
    ) == pytest.approx(0.9566008675707367)


@pytest.mark.parametrize(
    ("amplitudes", "leakages"),
    [([1.0], [1.0]), ([1.0, 0.5], [1.0]), ([1.0, 0.0], [1.0, 1.0])],
)
def test_lattice_exponent_refuses_invalid_series(amplitudes, leakages) -> None:
    with pytest.raises(EvaluatorError):
        eval_receiver_lattice_leakage_exponent({"amplitudes": amplitudes, "leakages": leakages})


def test_po1_equation_keeps_below_floor_claim_narrow(tmp_path) -> None:
    equation = build_local_model_step_resolvability_equation(source_receipt=_sidecar(tmp_path))
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == STEP_RESOLVABILITY_EQUATION_ID
    assert anchor.predicted_output["instrument_resolved"] is False
    assert equation.domain_of_validity["score_claim"] is False
    assert "representation-changing learned models" in equation.domain_of_validity["excluded"]


def test_pz4r_equation_refuses_hash_identity_as_semantic_proof(tmp_path) -> None:
    equation = build_receiver_pose_semantic_preservation_equation(source_receipt=_sidecar(tmp_path))
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == POSE_SEMANTIC_EQUATION_ID
    assert anchor.empirical_output["repeat_identity"] is True
    assert anchor.empirical_output["semantic_pose_gate_passed"] is False
    assert anchor.empirical_output["candidate_over_base_d_pose"] > 4_000


def test_pose_budget_equation_marks_js7_seg_credit_inferred(tmp_path) -> None:
    equation = build_pose_stack_exact_budget_equation(source_receipt=_sidecar(tmp_path))
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == POSE_BUDGET_EQUATION_ID
    assert "inferred" in anchor.inputs["seg_credit_provenance"].lower()
    assert anchor.empirical_output["safety_factor_included"] is False
    assert anchor.empirical_output["exact_pose_budget"] == pytest.approx(1.291770121176113e-6)


def test_js5_equation_keeps_n32_scope_and_both_exponents(tmp_path) -> None:
    equation = build_receiver_lattice_leakage_exponent_equation(source_receipt=_sidecar(tmp_path))
    empirical = equation.empirical_anchors[0].empirical_output
    assert equation.equation_id == LATTICE_EXPONENT_EQUATION_ID
    assert empirical["continuous_exponent"] == pytest.approx(2.5110854339243196)
    assert empirical["receiver_exponent"] == pytest.approx(0.9566008675707367)
    assert "n600 or contest-axis authority" in equation.domain_of_validity["excluded"]


def test_all_cn5_equations_have_anchors_routes_and_no_score_claim() -> None:
    equations = build_ddm_cn5_arc_equations()
    assert {equation.equation_id for equation in equations} == {
        STEP_RESOLVABILITY_EQUATION_ID,
        POSE_SEMANTIC_EQUATION_ID,
        POSE_BUDGET_EQUATION_ID,
        LATTICE_EXPONENT_EQUATION_ID,
    }
    for equation in equations:
        assert equation.empirical_anchors
        assert equation.canonical_consumers
        assert equation.canonical_producers
        assert equation.domain_of_validity["score_claim"] is False


def test_lawref_population_registers_all_cn5_evaluators() -> None:
    ids = populate_lawref_evaluators()
    assert {
        STEP_RESOLVABILITY_EQUATION_ID,
        POSE_SEMANTIC_EQUATION_ID,
        POSE_BUDGET_EQUATION_ID,
        LATTICE_EXPONENT_EQUATION_ID,
    }.issubset(ids)


def test_lawref_resolves_cn5_pose_budget() -> None:
    populate_lawref_evaluators()
    value = resolve_equation_value(
        POSE_BUDGET_EQUATION_ID,
        {"base_d_pose": 1.0e-5, "seg_credit_s": 0.001, "archive_delta_bytes": 0},
    )
    assert value == pytest.approx(((math.sqrt(1.0e-4) + 0.001) ** 2) / 10 - 1.0e-5)


def test_locked_population_registers_exactly_four_equations(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    ids = populate_ddm_cn5_arc_equations(path=registry, lock_path=lock)
    assert len(ids) == 4
    assert {equation.equation_id for equation in query_equations(path=registry)} == set(ids)
