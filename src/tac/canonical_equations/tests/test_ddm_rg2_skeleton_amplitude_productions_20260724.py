from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_rg2_skeleton_amplitude_productions_20260724 import (
    EQUATION_ID,
    build_ddm_rg2_skeleton_amplitude_productions_v1,
    select_skeleton_amplitude_row_band,
)
from tac.canonical_equations.evaluators import resolve_equation_value


def test_row_band_selects_earliest_max_without_scorer_input() -> None:
    rows = [0] * 384
    rows[128:192] = [2] * 64
    rows[256:320] = [2] * 64
    assert select_skeleton_amplitude_row_band(rows) == 2


def test_uniform_evaluator_and_zero_support_fail_closed() -> None:
    assert resolve_equation_value(
        EQUATION_ID,
        {
            "support_mass_by_row": [0, 1, 4, 0],
            "band_height": 2,
        },
    ) == 1
    with pytest.raises(ValueError, match="no receiver support"):
        select_skeleton_amplitude_row_band([0] * 384)


def test_equation_builds_from_rg2_receipt_with_scoped_authority() -> None:
    equation = build_ddm_rg2_skeleton_amplitude_productions_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["inactive_identity"].startswith("A_empty=I")
    assert equation.domain_of_validity["score_claim"] is False
    assert equation.domain_of_validity["verdict_scope"] == "INSTANCE_EXTENDED_GRAMMAR_RG2"
