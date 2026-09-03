from __future__ import annotations

import json

import pytest

from tac.canonical_equations.custom_sparse_adjoint_achieved_ceiling_20260713 import (
    DERIVED_FLAGSHIP_CEILING_X,
    EQUATION_ID,
    achieved_vs_ceiling_law,
    basis_reuse_law,
    build_custom_sparse_adjoint_achieved_vs_ceiling_v1,
    populate_custom_sparse_adjoint_achieved_vs_ceiling_v1,
)


def test_achieved_vs_ceiling_separates_arithmetic_and_wall() -> None:
    result = achieved_vs_ceiling_law(
        dense_flops=220.0,
        sparse_flops=100.0,
        dense_time=22.0,
        sparse_time=15.0,
    )
    assert result["arithmetic_ceiling_x"] == 2.2
    assert result["achieved_wall_speedup_x"] == pytest.approx(22.0 / 15.0)
    assert result["achieved_to_ceiling_ratio"] == pytest.approx((22.0 / 15.0) / 2.2)
    assert result["ideal_sparse_time"] == 10.0
    assert result["residual_sparse_time_above_flop_scaled_ideal"] == 5.0
    with pytest.raises(ValueError):
        achieved_vs_ceiling_law(
            dense_flops=1.0, sparse_flops=2.0, dense_time=1.0, sparse_time=1.0
        )


def test_basis_reuse_requires_state_stable_amortization() -> None:
    no_reuse = basis_reuse_law(
        dense_vjp_time_per_step=1.0, basis_vjp_time=1.6, reuse_steps=1
    )
    k2 = basis_reuse_law(dense_vjp_time_per_step=1.0, basis_vjp_time=1.6, reuse_steps=2)
    assert no_reuse["wins"] is False
    assert k2["wins"] is True
    assert k2["amortized_speedup_x"] == 1.25
    assert k2["minimum_reuse_steps_strict"] == 2


def test_equation_records_measured_metal_wall_with_history() -> None:
    # Stale until 2026-09-04: this test still asserted the pre-2026-07-14 state
    # ("no Metal anchor") after the D43 whole-network Metal-wall replay was
    # measured and appended (anchor metal_wall_125conv_replay_20260714). The
    # honest invariant is that the measured status REPLACES the blocked one
    # while the blocked one survives as history (append-only provenance).
    equation = build_custom_sparse_adjoint_achieved_vs_ceiling_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["flagship_derived_ceiling_x"] == (
        DERIVED_FLAGSHIP_CEILING_X
    )
    assert equation.domain_of_validity["empirical_status"] == (
        "METAL_WALL_MEASURED_20260714_WHOLE_NETWORK_SLOWDOWN_0p7078x"
    )
    assert equation.domain_of_validity["empirical_status_history"] == (
        "BLOCKED_NO_METAL_IN_CURRENT_SANDBOX",
    )
    assert [a.anchor_id for a in equation.empirical_anchors] == [
        "metal_wall_125conv_replay_20260714"
    ]
    assert equation.provenance.score_claim_valid is False


def test_equation_populates_only_explicit_temporary_registry(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    populated = populate_custom_sparse_adjoint_achieved_vs_ceiling_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.jsonl.lock",
        agent="codex",
        subagent_id="custom_sparse_adjoint_kernel",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert populated.equation_id == EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["equation_id"] == EQUATION_ID
