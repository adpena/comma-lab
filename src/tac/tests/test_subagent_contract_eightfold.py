"""Tests for the eightfold design-philosophies clause in the subagent contract + integrity gate."""
from __future__ import annotations

from tac import subagent_contract as sc


def test_eightfold_clause_present_and_registered():
    assert "one fact, one store, one key" in sc.EIGHTFOLD_CLAUSE
    assert "P8 floor-first" in sc.EIGHTFOLD_CLAUSE
    assert "fmtools" in sc.EIGHTFOLD_CLAUSE and "#259" in sc.EIGHTFOLD_CLAUSE
    assert "EIGHTFOLD_CLAUSE" in sc.CONTRACT_CONSTANT_NAMES
    assert sc.KEY_PHRASES["EIGHTFOLD_CLAUSE"] in sc.EIGHTFOLD_CLAUSE


def test_standard_contract_composes_eightfold():
    out = sc.standard_contract()
    assert sc.KEY_PHRASES["EIGHTFOLD_CLAUSE"] in out
    assert "fmtools" in out
    # still composes the highest-value harvest block
    assert sc.KEY_PHRASES["GROUNDED_PROGRESS"] in out


def test_integrity_gate_passes_with_eightfold():
    from tac.preflight import check_subagent_contract_module_integrity
    violations = check_subagent_contract_module_integrity()
    assert violations == [], violations
    from tac.preflight import _SUBAGENT_CONTRACT_REQUIRED_CONSTANTS
    assert "EIGHTFOLD_CLAUSE" in _SUBAGENT_CONTRACT_REQUIRED_CONSTANTS
