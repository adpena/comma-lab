# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import pytest

from tac.canonical_equations.ddm_v18_column_pricing_law_20260723 import (
    ddm_column_reduced_cost,
)
from tac.optimization.ddm_column_generation import (
    DDMColumnGenerationError,
    ExactReplay,
    PricedColumn,
    exact_replay_beam_select,
    generated_vocabulary_falsifier,
    price_columns,
    solve_restricted_master_lp,
)


def _column(
    column_id: str,
    *,
    cost: float,
    size: int,
    conflicts: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
) -> PricedColumn:
    return PricedColumn(column_id, "fixture", size, cost, conflicts, dependencies)


def test_restricted_master_duals_and_pricing_use_real_byte_constraint() -> None:
    columns = (
        _column("a", cost=-4.0, size=4),
        _column("b", cost=-3.0, size=4),
    )
    duals = solve_restricted_master_lp(columns, added_byte_budget=4)
    assert sum(duals.selected_fraction_by_id.values()) == pytest.approx(1.0)
    assert duals.byte_marginal < 0.0
    rows = {row.column_id: row for row in price_columns(columns, duals=duals)}
    assert rows["a"].reduced_cost < 0.0  # selected at its x<=1 upper bound
    assert rows["b"].reduced_cost == pytest.approx(0.0)


def test_exact_byte_fields_reject_noninteger_values() -> None:
    with pytest.raises(DDMColumnGenerationError, match="positive integer"):
        _column("fractional", cost=-1.0, size=1.5)  # type: ignore[arg-type]


def test_conflict_dual_changes_reduced_cost() -> None:
    columns = (
        _column("a", cost=-4.0, size=1, conflicts=("same-pixel",)),
        _column("b", cost=-3.0, size=1, conflicts=("same-pixel",)),
    )
    duals = solve_restricted_master_lp(columns, added_byte_budget=10)
    assert duals.conflict_marginals["same-pixel"] < 0.0
    priced = {row.column_id: row for row in price_columns(columns, duals=duals)}
    assert priced["a"].reduced_cost < 0.0  # selected at its x<=1 upper bound
    assert priced["b"].reduced_cost == pytest.approx(0.0)


def test_canonical_equation_matches_pricer() -> None:
    value = ddm_column_reduced_cost(
        singleton_objective_delta=-2.0,
        exact_coder_bytes=10,
        byte_dual_marginal=-0.1,
        conflict_dual_marginals={"x": -0.25},
        active_conflict_keys=("x",),
    )
    assert value == pytest.approx(-0.75)


def test_beam_replays_every_explored_set_and_finds_non_greedy_pair() -> None:
    columns = (
        _column("a", cost=-4.0, size=4, conflicts=("x",)),
        _column("b", cost=-3.0, size=3),
        _column("c", cost=-2.0, size=3),
    )
    objective = {
        (): 10.0,
        ("a",): 6.0,
        ("b",): 7.0,
        ("c",): 8.0,
        ("a", "b"): 6.5,
        ("a", "c"): 6.2,
        ("b", "c"): 4.0,
        ("a", "b", "c"): 8.0,
    }
    calls: list[tuple[str, ...]] = []

    def replay(ids: tuple[str, ...]) -> ExactReplay:
        calls.append(ids)
        size = 100 + sum(next(row.real_coder_bytes for row in columns if row.column_id == value) for value in ids)
        digest = hashlib.sha256(",".join(ids).encode()).hexdigest()
        return ExactReplay(ids, size, 0.1, 1.0, objective[ids], digest, True, True)

    best, rows = exact_replay_beam_select(
        columns,
        base_archive_bytes=100,
        added_byte_budget=6,
        replay=replay,
        beam_width=32,
    )
    assert best.column_ids == ("b", "c")
    assert len(calls) == len(set(calls)) == len(rows)


def test_beam_enforces_dependencies_before_replay() -> None:
    columns = (
        _column("base", cost=-1.0, size=2),
        _column("template", cost=-2.0, size=2, dependencies=("base",)),
    )
    seen: list[tuple[str, ...]] = []

    def replay(ids: tuple[str, ...]) -> ExactReplay:
        seen.append(ids)
        digest = hashlib.sha256(str(ids).encode()).hexdigest()
        return ExactReplay(ids, 100 + 2 * len(ids), 0.1, 1.0, 10.0 - len(ids), digest, True, True)

    best, _rows = exact_replay_beam_select(
        columns,
        base_archive_bytes=100,
        added_byte_budget=4,
        replay=replay,
    )
    assert ("template",) not in seen
    assert best.column_ids == ("base", "template")


def test_beam_refuses_dependency_cycles() -> None:
    columns = (
        _column("a", cost=-1.0, size=1, dependencies=("b",)),
        _column("b", cost=-1.0, size=1, dependencies=("a",)),
    )

    def replay(ids: tuple[str, ...]) -> ExactReplay:
        digest = hashlib.sha256(str(ids).encode()).hexdigest()
        return ExactReplay(ids, 100, 0.1, 1.0, 1.0, digest, True, True)

    with pytest.raises(DDMColumnGenerationError, match="cycle"):
        exact_replay_beam_select(
            columns,
            base_archive_bytes=100,
            added_byte_budget=2,
            replay=replay,
        )


def test_beam_refuses_non_exact_replay() -> None:
    column = _column("a", cost=-1.0, size=1)

    def replay(ids: tuple[str, ...]) -> ExactReplay:
        return ExactReplay(
            ids,
            101,
            0.1,
            1.0,
            1.0,
            "0" * 64,
            receiver_closed=False,
            scorer_replayed=True,
        )

    with pytest.raises(DDMColumnGenerationError, match="receiver"):
        exact_replay_beam_select(
            (column,),
            base_archive_bytes=100,
            added_byte_budget=1,
            replay=replay,
        )


def test_falsifier_requires_exact_three_round_conjunction() -> None:
    rounds = [
        {
            "round": index,
            "complete": True,
            "exact_pricing": True,
            "negative_reduced_cost_count": 0,
        }
        for index in range(1, 4)
    ]
    rows = [
        {
            "added_byte_budget": budget,
            "exact_replay_complete": True,
            "global_selector": "beam_width_32",
            "generated_vocabulary": {"d_seg": 0.034003668891},
        }
        for budget in (16_384, 49_152, 98_304, 147_456)
    ]
    assert generated_vocabulary_falsifier(
        pricing_rounds=rounds,
        equal_byte_rows=rows,
        v12_d_seg=0.034003668891,
    )
    rounds[2]["complete"] = False
    assert not generated_vocabulary_falsifier(
        pricing_rounds=rounds,
        equal_byte_rows=rows,
        v12_d_seg=0.034003668891,
    )


def test_falsifier_stays_open_if_any_global_row_beats_v12() -> None:
    rounds = [
        {
            "round": index,
            "complete": True,
            "exact_pricing": True,
            "negative_reduced_cost_count": 0,
        }
        for index in range(1, 4)
    ]
    rows = [
        {
            "added_byte_budget": budget,
            "exact_replay_complete": True,
            "global_selector": "beam_width_32",
            "generated_vocabulary": {"d_seg": 0.034003668891},
        }
        for budget in (16_384, 49_152, 98_304, 147_456)
    ]
    rows[-1]["generated_vocabulary"]["d_seg"] = 0.033
    assert not generated_vocabulary_falsifier(
        pricing_rounds=rounds,
        equal_byte_rows=rows,
        v12_d_seg=0.034003668891,
    )


def test_falsifier_refuses_nan_and_duplicate_budget_rows() -> None:
    rounds = [
        {
            "round": index,
            "complete": True,
            "exact_pricing": True,
            "negative_reduced_cost_count": 0,
        }
        for index in range(1, 4)
    ]
    rows = [
        {
            "added_byte_budget": budget,
            "exact_replay_complete": True,
            "global_selector": "conflict_miqp",
            "generated_vocabulary": {"d_seg": 0.034003668891},
        }
        for budget in (16_384, 49_152, 98_304, 147_456)
    ]
    rows[0]["generated_vocabulary"]["d_seg"] = float("nan")
    assert not generated_vocabulary_falsifier(
        pricing_rounds=rounds,
        equal_byte_rows=rows,
        v12_d_seg=0.034003668891,
    )
    rows[0]["generated_vocabulary"]["d_seg"] = 0.034003668891
    rows[-1]["added_byte_budget"] = 98_304
    assert not generated_vocabulary_falsifier(
        pricing_rounds=rounds,
        equal_byte_rows=rows,
        v12_d_seg=0.034003668891,
    )
