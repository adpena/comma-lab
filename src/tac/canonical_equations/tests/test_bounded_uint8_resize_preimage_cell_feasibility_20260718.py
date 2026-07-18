# SPDX-License-Identifier: MIT
"""Tests for the ratified factor-2 uint8-lattice feasibility canonical equation
(Task #540). The law's evaluator is the deterministic Diophantine IFF predicate;
these tests exercise it on known-feasible / known-infeasible cells and confirm
the equation builds + registers into a temp registry with honest scope flags."""
from __future__ import annotations

from pathlib import Path

from tac.canonical_equations.bounded_uint8_resize_preimage_cell_feasibility_20260718 import (
    EQUATION_ID,
    bounded_cell_feasibility_certificate,
    build_bounded_uint8_resize_preimage_cell_feasibility_v1,
    populate_bounded_uint8_resize_preimage_cell_feasibility_equation,
)


def test_feasible_cell_yields_exact_uint8_witness() -> None:
    # c=(1,1), denom=1, target=300 -> exists z in [0,255]^2 with z0+z1=300
    # (e.g. 45+255). FEASIBLE_EXACT with an exact integer witness.
    cert = bounded_cell_feasibility_certificate([1, 1], 1, 300.0)
    assert cert["feasible_exact"] is True
    assert cert["exact_uint8_witness"] is not None
    z = cert["exact_uint8_witness"]
    assert sum(z) == 300
    assert all(0 <= v <= 255 for v in z)
    assert cert["score_claim"] is False
    assert cert["promotion_eligible"] is False


def test_infeasible_cell_is_proven_not_guessed() -> None:
    # c=(2,2), denom=1, target=301 -> 2*(z0+z1)=301 has NO integer solution
    # (odd target, even coefficient span) -> proven exhaustively infeasible.
    cert = bounded_cell_feasibility_certificate([2, 2], 1, 301.0)
    assert cert["feasible_exact"] is False
    assert cert["proven_lattice_infeasible"] is True
    assert cert["exact_uint8_witness"] is None


def test_out_of_range_target_infeasible() -> None:
    # c=(1,1), max reachable is 510; target 600 is out of the bounded lattice.
    cert = bounded_cell_feasibility_certificate([1, 1], 1, 600.0)
    assert cert["feasible_exact"] is False
    assert cert["exact_uint8_witness"] is None


def test_equation_builds_with_honest_scope() -> None:
    eq = build_bounded_uint8_resize_preimage_cell_feasibility_v1()
    assert eq.equation_id == EQUATION_ID
    # No fabricated empirical anchor; the score claims stay byte-close-gated.
    assert eq.empirical_anchors == ()
    dov = eq.domain_of_validity
    assert dov["score_claim"] is False
    assert dov["promotion_eligible"] is False
    assert "BYTE_CLOSE_GATED" in dov["score_authority"]
    assert "VERIFIED_VIA_SOURCE_INSPECTION" in dov["feasibility_predicate_authority"]
    # The callable path resolves to the evaluator in THIS module.
    assert eq.python_callable_module_path.endswith(
        ":bounded_cell_feasibility_certificate"
    )


def test_registers_into_temp_registry(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    eq = populate_bounded_uint8_resize_preimage_cell_feasibility_equation(
        path=reg, lock_path=lock, agent="test", subagent_id="pytest",
    )
    assert eq.equation_id == EQUATION_ID
    assert reg.exists()
    assert EQUATION_ID in reg.read_text()
