# SPDX-License-Identifier: MIT
"""Regression tests for the permanently retired inflate.py LOC formula."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.inflate_py_loc_budget_derivation import (
    COMPLEXITY_BUDGET_MCCABE,
    DEPENDENCY_BUDGET_HNERV,
    LOC_BUDGET_AT_30SEC,
    LOCBudgetInput,
    canonical_inflate_py_loc_budget,
)


@pytest.mark.parametrize("loc", [0, 100, 200, 500, 50_000])
def test_every_source_length_is_a_no_op(loc: int) -> None:
    result = canonical_inflate_py_loc_budget(
        LOCBudgetInput(loc=loc, cyclomatic_complexity=20, external_dependencies=4),
        emit_arbitrariness_atom=True,
    )
    assert result.solved_value == 0.0
    assert result.intermediate_values["restriction_active"] is False
    assert result.intermediate_values["loc"] == loc
    assert "atom" not in result.coupled_adjustments


def test_historical_constants_remain_importable() -> None:
    assert LOC_BUDGET_AT_30SEC == 200
    assert COMPLEXITY_BUDGET_MCCABE == 10
    assert DEPENDENCY_BUDGET_HNERV == 2


def test_invalid_telemetry_inputs_raise() -> None:
    with pytest.raises(ValueError, match="loc"):
        LOCBudgetInput(loc=-1, cyclomatic_complexity=5, external_dependencies=1)
    with pytest.raises(ValueError, match="cyclomatic_complexity"):
        LOCBudgetInput(loc=100, cyclomatic_complexity=-1, external_dependencies=1)
    with pytest.raises(ValueError, match="external_dependencies"):
        LOCBudgetInput(loc=100, cyclomatic_complexity=5, external_dependencies=-1)


def test_citation_records_permanent_removal() -> None:
    result = canonical_inflate_py_loc_budget(
        LOCBudgetInput(loc=500, cyclomatic_complexity=5, external_dependencies=1)
    )
    assert "permanently removed" in result.literature_citation
