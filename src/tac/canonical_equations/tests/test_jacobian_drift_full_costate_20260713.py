from __future__ import annotations

from tac.canonical_equations.jacobian_drift_full_costate_20260713 import (
    BASELINE_VALIDATIONS_PER_TEACHER_CALL,
    CURRENT_DESCENT_PREFIX_BY_REGIME,
    EQUATION_ID,
    HVP_LOWER_BOUND_MATCHED_VALIDATION_EQUIVALENTS,
    ORACLE_SAFE_PREFIX_BY_REGIME,
    RIGOROUS_CERTIFIED_REUSES,
    build_jacobian_drift_full_costate_v1,
    populate_jacobian_drift_full_costate_v1,
)


def test_equation_encodes_cubic_bound_and_authority_boundary() -> None:
    equation = build_jacobian_drift_full_costate_v1()
    assert equation.equation_id == EQUATION_ID
    assert "B_JL_q+L_c" in equation.latex_form
    assert "\\tfrac12L_HL_qr^3" in equation.latex_form
    assert "gamma_\\theta/B_R" in equation.latex_form
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["rigorous_certified_reuses"] == RIGOROUS_CERTIFIED_REUSES == 0
    assert tuple(anchor.empirical_output["oracle_safe_prefix_by_regime"]) == (
        ORACLE_SAFE_PREFIX_BY_REGIME
    )
    assert BASELINE_VALIDATIONS_PER_TEACHER_CALL == 8.375
    assert CURRENT_DESCENT_PREFIX_BY_REGIME == (17, 10, 17)
    assert anchor.empirical_output["current_ce_dseg_descent_radius_l2_by_regime"] == [
        103.02526092529297,
        5.644960403442383,
        147.25413513183594,
    ]
    assert HVP_LOWER_BOUND_MATCHED_VALIDATION_EQUIVALENTS > 1.0
    assert anchor.inputs["receipt_sha256"] == (
        "c1a2431ebe9df21a370748f864f2da81a5f242544051986ed341d59fe1518d48"
    )
    excluded = equation.domain_of_validity["excluded"]
    assert any("point HVP" in item for item in excluded)
    assert any("activation boundary" in item for item in excluded)
    assert equation.predicted_vs_empirical_residual["missing_rigorous_bound_custody"] == 1.0


def test_equation_populates_locked_temporary_registry(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    equation = populate_jacobian_drift_full_costate_v1(
        path=registry,
        lock_path=lock,
        agent="test",
        subagent_id="test_jacobian_drift",
    )
    assert equation.equation_id == EQUATION_ID
    assert registry.is_file()
    assert EQUATION_ID in registry.read_text()
