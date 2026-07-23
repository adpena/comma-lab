# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.optimization.ddm_column_generation import PricedColumn
from tools.run_ddm_v18b_common_master_pricing import (
    FIXED_BUDGETS,
    _formulation_falsified,
    _largest_prefix_not_exceeding,
    _load_bundle_rows,
    _miqp_diagonal_proposal,
)


def test_bound_v12_inventory_is_exactly_4096_atoms_in_353_bundles() -> None:
    rows = _load_bundle_rows()
    assert len(rows) == 353
    assert sum(int(row["atomic_obligation_count"]) for row in rows) == 4096


def test_conflict_miqp_respects_exact_byte_cap_and_conflicts() -> None:
    columns = (
        PricedColumn("a", "fixture", 4, -4.0, ("same-site",)),
        PricedColumn("b", "fixture", 4, -3.0, ("same-site",)),
        PricedColumn("c", "fixture", 3, -2.0),
    )
    assert _miqp_diagonal_proposal(columns, added_byte_budget=7) == ("a", "c")


def test_equal_byte_prefix_uses_realized_control_cap_not_nominal_rung() -> None:
    assert (
        _largest_prefix_not_exceeding(
            (100, 110, 130, 119),
            base_archive_bytes=100,
            realized_added_byte_cap=20,
        )
        == 3
    )


def test_falsifier_requires_three_clean_rounds_and_no_equal_byte_win() -> None:
    history = [
        {
            "round": index,
            "complete": True,
            "exact_pricing": True,
            "negative_reduced_cost_count": 0,
        }
        for index in range(1, 4)
    ]
    equal = [{"added_byte_budget": budget, "beats_v12": False} for budget in FIXED_BUDGETS]
    assert _formulation_falsified(history, equal) == (True, True, False)

    equal[-1]["beats_v12"] = True
    assert _formulation_falsified(history, equal) == (False, True, True)

    equal[-1]["beats_v12"] = False
    history[-1]["negative_reduced_cost_count"] = 1
    assert _formulation_falsified(history, equal) == (False, False, False)
