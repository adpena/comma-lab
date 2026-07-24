from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_rg1_bounded_receiver_composition_20260724 import (
    EQUATION_ID,
    build_ddm_rg1_bounded_receiver_composition_v1,
    rg1_bounded_center_projection,
)
from tac.canonical_equations.evaluators import resolve_equation_value
from tac.optimization.direct_description_minimizer import DirectDescriptionError


def test_projection_matches_closed_integer_interval() -> None:
    assert rg1_bounded_center_projection(
        center=-9,
        relative_coordinates=(-3, 0, 4),
        extent=20,
    ) == 3
    assert rg1_bounded_center_projection(
        center=30,
        relative_coordinates=(-3, 0, 4),
        extent=20,
    ) == 15


def test_uniform_evaluator_and_fail_closed_geometry() -> None:
    assert resolve_equation_value(
        EQUATION_ID,
        {
            "center": 9,
            "relative_coordinates": [-2, 3],
            "extent": 16,
        },
    ) == 9
    with pytest.raises(DirectDescriptionError, match="cannot fit"):
        rg1_bounded_center_projection(
            center=0,
            relative_coordinates=(-8, 8),
            extent=16,
        )


def test_equation_builds_from_rg1_receipt_with_scoped_authority() -> None:
    equation = build_ddm_rg1_bounded_receiver_composition_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["inactive_identity"].startswith("P_0=C_0=I")
    assert equation.domain_of_validity["score_claim"] is False
    assert equation.domain_of_validity["verdict_scope"] == "INSTANCE_EXTENDED_GRAMMAR_RG1"
