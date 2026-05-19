# SPDX-License-Identifier: MIT
"""Tests for Row #8 — HNeRV L4 inflate.py LOC budget derivation."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.inflate_py_loc_budget_derivation import (
    LOCBudgetInput,
    LOC_BUDGET_AT_30SEC,
    COMPLEXITY_BUDGET_MCCABE,
    DEPENDENCY_BUDGET_HNERV,
    canonical_inflate_py_loc_budget,
)


def test_small_inflate_passes():
    """100 LOC, complexity 5, 1 dep -> reviewability <= 1.0."""
    r = canonical_inflate_py_loc_budget(LOCBudgetInput(
        loc=100, cyclomatic_complexity=5, external_dependencies=1,
    ))
    assert r.solved_value <= 1.0
    assert r.intermediate_values["passes_30_sec_criterion"] is True


def test_oversize_inflate_fails():
    """400 LOC, complexity 20, 4 deps -> reviewability > 1.0."""
    r = canonical_inflate_py_loc_budget(LOCBudgetInput(
        loc=400, cyclomatic_complexity=20, external_dependencies=4,
    ))
    assert r.solved_value > 1.0
    assert r.intermediate_values["passes_30_sec_criterion"] is False


def test_canonical_budgets_pinned():
    """LOC 200 + McCabe 10 + deps 2 budgets pinned."""
    assert LOC_BUDGET_AT_30SEC == 200
    assert COMPLEXITY_BUDGET_MCCABE == 10
    assert DEPENDENCY_BUDGET_HNERV == 2


def test_dominant_factor_identified():
    """Largest normalized factor surfaced in coupled_adjustments."""
    r = canonical_inflate_py_loc_budget(LOCBudgetInput(
        loc=50, cyclomatic_complexity=20, external_dependencies=1,
    ))
    assert r.coupled_adjustments["dominant_factor"] == "complexity"


def test_invalid_inputs_raise():
    """Negative inputs raise."""
    with pytest.raises(ValueError, match="loc"):
        LOCBudgetInput(loc=-1, cyclomatic_complexity=5, external_dependencies=1)
    with pytest.raises(ValueError, match="cyclomatic_complexity"):
        LOCBudgetInput(loc=100, cyclomatic_complexity=-1, external_dependencies=1)
    with pytest.raises(ValueError, match="external_dependencies"):
        LOCBudgetInput(loc=100, cyclomatic_complexity=5, external_dependencies=-1)


def test_citation_hnerv_l4():
    """HNeRV parity L4 + McCabe + Catalog #328 citations."""
    r = canonical_inflate_py_loc_budget(LOCBudgetInput(
        loc=100, cyclomatic_complexity=5, external_dependencies=1,
    ))
    assert "HNeRV parity" in r.literature_citation
    assert "McCabe" in r.literature_citation
