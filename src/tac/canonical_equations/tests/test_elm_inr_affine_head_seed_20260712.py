# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.boundary_math.elm_inr_head_solve import solve_partitioned_affine_head_with_fold
from tac.canonical_equations.elm_inr_affine_head_seed_20260712 import (
    EQUATION_ID,
    build_elm_inr_affine_sdf_head_seed_v1,
    populate_elm_inr_affine_sdf_head_seed_equation,
)
from tac.canonical_equations.registry import query_equations


def test_equation_is_honest_about_scope_and_score_authority() -> None:
    equation = build_elm_inr_affine_sdf_head_seed_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.python_callable_module_path.endswith(":solve_partitioned_affine_head_with_fold")
    assert equation.empirical_anchors == ()
    assert equation.predicted_vs_empirical_residual == {}
    assert equation.domain_of_validity["score_authority"].startswith("none")
    assert "fold-to-local RMSE" in equation.domain_of_validity["pou_custody"]
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False


def test_canonical_callable_solves_the_declared_affine_head() -> None:
    rng = np.random.default_rng(41)
    hidden = rng.normal(size=(128, 5))
    expected_weight = rng.normal(size=(3, 5)).astype(np.float32)
    expected_bias = rng.normal(size=3).astype(np.float32)
    targets = hidden @ expected_weight.T + expected_bias

    coords = rng.uniform(-1.0, 1.0, size=(hidden.shape[0], 2))
    solution = solve_partitioned_affine_head_with_fold(
        hidden,
        targets,
        coords,
        grid_shape=(1, 1),
        ridge=0.0,
    )
    np.testing.assert_allclose(
        solution.direct_global_beta[:-1].T,
        expected_weight,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        solution.direct_global_beta[-1],
        expected_bias,
        rtol=2e-6,
        atol=2e-6,
    )
    assert solution.direct_global_diagnostics.rank == solution.direct_global_diagnostics.dimension
    assert solution.fold_second_solve_applied is False


def test_population_uses_isolated_registry_and_round_trips(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.jsonl.lock"
    populated = populate_elm_inr_affine_sdf_head_seed_equation(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="elm_head_seed",
    )
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [equation.equation_id for equation in loaded] == [EQUATION_ID]
