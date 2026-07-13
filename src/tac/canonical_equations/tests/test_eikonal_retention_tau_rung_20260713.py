from __future__ import annotations

import pytest

from tac.canonical_equations.eikonal_retention_tau_rung_20260713 import (
    EQUATION_ID,
    build_eikonal_retention_couples_to_tau_rung_v1,
    eikonal_retention_for_rung,
)


def test_rung_law_endpoints_and_progression() -> None:
    got = [eikonal_retention_for_rung(0.01, 0.05, k, 4) for k in range(5)]
    assert got == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])


@pytest.mark.parametrize("rung,n", [(-1, 4), (5, 4), (0, 0)])
def test_rung_law_fails_closed_on_invalid_state(rung: int, n: int) -> None:
    with pytest.raises(ValueError):
        eikonal_retention_for_rung(0.01, 0.05, rung, n)


def test_equation_custody_and_scope_are_honest() -> None:
    eq = build_eikonal_retention_couples_to_tau_rung_v1()
    assert eq.equation_id == EQUATION_ID
    assert "UNMEASURED" in eq.domain_of_validity["verdict_scope"]
    assert "train_levelset_witness_realized_through_R_mlx" in " ".join(eq.canonical_consumers)
