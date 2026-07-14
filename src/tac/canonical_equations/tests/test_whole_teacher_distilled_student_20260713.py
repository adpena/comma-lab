from __future__ import annotations

import json

import numpy as np
import pytest

from tac.canonical_equations.whole_teacher_distilled_student_20260713 import (
    EQUATION_ID,
    HELMERT_BASIS_5X4,
    amortized_student_teacher_cost_ms,
    build_whole_teacher_distilled_student_fidelity_economics_v1,
    lift_centered_quotient_numpy,
    minimum_student_anchor_cadence_for_remaining_fraction,
    pairwise_fidelity_summary,
    populate_whole_teacher_distilled_student_fidelity_economics_v1,
    project_centered_quotient_numpy,
    surrogate_economics,
    vjp_fidelity_summary,
)


def test_helmert_quotient_is_orthonormal_zero_sum_and_gauge_invariant() -> None:
    basis = np.asarray(HELMERT_BASIS_5X4, dtype=np.float32)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4, dtype=np.float32), atol=2e-7)
    np.testing.assert_allclose(basis.T @ np.ones(5, dtype=np.float32), 0.0, atol=2e-7)

    logits = np.asarray(
        [[[2.0, -1.0, 0.5, 3.0, -4.0], [1.0, 2.0, 3.0, 4.0, 5.0]]],
        dtype=np.float32,
    )
    gauge = np.asarray([[[7.0], [-11.0]]], dtype=np.float32)
    quotient = project_centered_quotient_numpy(logits, class_axis=-1)
    shifted = project_centered_quotient_numpy(logits + gauge, class_axis=-1)
    np.testing.assert_allclose(quotient, shifted, atol=2e-6)

    lifted = lift_centered_quotient_numpy(quotient, class_axis=-1)
    centered = logits - logits.mean(axis=-1, keepdims=True, dtype=np.float32)
    np.testing.assert_allclose(lifted, centered, atol=2e-6)
    np.testing.assert_allclose(lifted.sum(axis=-1), 0.0, atol=2e-6)
    np.testing.assert_array_equal(np.argmax(lifted, axis=-1), np.argmax(logits, axis=-1))


def test_pairwise_summary_exposes_a_single_bad_worst_pair() -> None:
    reference = np.ones((600, 2, 4), dtype=np.float32)
    candidate = reference.copy()
    candidate[417] *= -1.0
    summary = pairwise_fidelity_summary(reference, candidate)
    assert summary["n_pairs"] == 600
    assert summary["worst_cosine_pair"] == 417
    assert summary["worst_relative_l2_pair"] == 417
    assert summary["worst_cosine"] == pytest.approx(-1.0)
    assert summary["worst_relative_l2"] == pytest.approx(2.0)


def test_vjp_summary_is_full_vector_and_any_zero_vector_fails_closed() -> None:
    exact = np.arange(1, 1 + 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2)
    same = vjp_fidelity_summary(exact, exact)
    assert same["worst_cosine"] == pytest.approx(1.0)
    assert same["worst_relative_l2"] == pytest.approx(0.0)
    assert same["authority_surface"] == "full_exact_teacher_input_vjp"

    zero_teacher = np.zeros((1, 3), dtype=np.float32)
    nonzero_student = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="undefined for a zero"):
        vjp_fidelity_summary(zero_teacher, nonzero_student)
    with pytest.raises(ValueError, match="undefined for a zero"):
        vjp_fidelity_summary(zero_teacher, zero_teacher)
    with pytest.raises(ValueError, match="undefined for a zero"):
        vjp_fidelity_summary(nonzero_student, zero_teacher)


def test_charged_cost_and_k20_inclusive_95_impossibility() -> None:
    charged_k20 = amortized_student_teacher_cost_ms(
        student_cost_ms=1.0,
        exact_teacher_cost_ms=100.0,
        anchor_update_cost_ms=1.0,
        student_anchor_cadence=20,
    )
    assert charged_k20 == pytest.approx(6.05)
    assert charged_k20 > 5.0

    boundary = minimum_student_anchor_cadence_for_remaining_fraction(
        student_cost_ms=1.0,
        exact_teacher_cost_ms=100.0,
        anchor_update_cost_ms=1.0,
        target_remaining_fraction=0.05,
    )
    assert boundary["finite_cadence_feasible"] is True
    assert boundary["minimum_cadence_real"] == pytest.approx(25.25)
    assert boundary["minimum_cadence_integer"] == 26

    impossible = minimum_student_anchor_cadence_for_remaining_fraction(
        student_cost_ms=5.0,
        exact_teacher_cost_ms=100.0,
        anchor_update_cost_ms=0.0,
        target_remaining_fraction=0.05,
    )
    assert impossible["finite_cadence_feasible"] is False
    assert impossible["minimum_cadence_integer"] is None


def test_economics_separates_cost_math_fidelity_and_measurement_authority() -> None:
    scenario = surrogate_economics(
        tier="training_gradient",
        student_cost_ms=2.0,
        exact_teacher_cost_ms=100.0,
        anchor_update_cost_ms=1.0,
        student_anchor_cadence=32,
        fidelity_gate_passed=True,
        charged_timing_measured=False,
        exact_costate_reuse_kmax=2,
    )
    assert scenario["cost_pays"] is True
    assert scenario["strict_pays"] is False
    assert scenario["inclusive_95"] is False
    assert scenario["pays"] is False
    assert scenario["status"] == "NO_PAY_AUTHORITY"
    assert scenario["student_anchor_cadence"] == 32
    assert scenario["exact_costate_reuse_kmax"] == 2
    assert scenario["exact_costate_reuse_speed_claim_imported"] is False

    measured = surrogate_economics(
        tier="training_gradient",
        student_cost_ms=2.0,
        exact_teacher_cost_ms=100.0,
        anchor_update_cost_ms=1.0,
        student_anchor_cadence=32,
        fidelity_gate_passed=True,
        charged_timing_measured=True,
        exact_costate_reuse_kmax=None,
    )
    assert measured["pays"] is True
    assert measured["strict_pays"] is True
    assert measured["inclusive_95"] is False
    assert measured["status"] == "STUDENT_PAYS"
    with pytest.raises(ValueError, match="sealed to K_max=2"):
        surrogate_economics(
            tier="training_gradient",
            student_cost_ms=2.0,
            exact_teacher_cost_ms=100.0,
            anchor_update_cost_ms=1.0,
            student_anchor_cadence=32,
            fidelity_gate_passed=True,
            charged_timing_measured=True,
            exact_costate_reuse_kmax=3,
        )


def test_equation_is_honest_about_missing_empirical_authority() -> None:
    equation = build_whole_teacher_distilled_student_fidelity_economics_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.empirical_anchors == ()
    assert equation.domain_of_validity["research_only"] is True
    assert equation.domain_of_validity["empirical_status"] == ("UNMEASURED_BLOCKED_INPUT_CACHE")
    assert "n600" in equation.domain_of_validity["req_R"]
    assert "closes" in equation.domain_of_validity["verdict_scope"]
    assert equation.provenance.score_claim_valid is False


def test_equation_populates_only_an_explicit_temporary_registry(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    equation = populate_whole_teacher_distilled_student_fidelity_economics_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.jsonl.lock",
        agent="codex",
        subagent_id="whole_teacher_distilled_student",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert equation.equation_id == EQUATION_ID
    assert [row["equation_id"] for row in rows] == [EQUATION_ID]
