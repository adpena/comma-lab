# SPDX-License-Identifier: MIT
"""Tests for META-LIFT-2 Pareto polytope unified solver.

Per Catalog #229 premise verification + #294 9-dim checklist + #303
cargo-cult audit per assumption: every solver invariant has an
explicit test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.pareto_polytope_unified_solver import (
    CANONICAL_EQUATION_ID,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    PREDICTED_AXIS_TAG,
    SCHEMA_VERSION,
    VALID_AXIS_LABELS,
    PareDLPProblemSpec,
    PareDLPSolution,
    PareDLPSolutionCorruptError,
    UnifiedBitBudgetAllocation,
    append_solution_locked,
    build_problem_spec_from_meta_lift_1_analysis,
    load_solutions_strict,
    solve_pareto_polytope_via_dykstra_projections,
)
from tac.pareto_polytope_unified_solver.solver import (
    _project_onto_cauchy_schwarz_bound,
    _project_onto_non_negativity,
    _project_onto_per_axis_box,
    _project_onto_per_substrate_aggregate_budget,
)


# ---------------------------------------------------------------------------
# Canonical synthetic problem fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_3sub_3axis_spec() -> PareDLPProblemSpec:
    """Synthetic 3-substrate × 3-axis problem with known feasible point.

    The all-zeros allocation is trivially feasible (intersection is
    non-empty per Boyd 2004 §7.2 Theorem 7.1). The Cauchy-Schwarz
    bound of 50.0 is below the sum of per-axis caps × gradient L2
    norms (so the C-S constraint is binding).
    """
    return PareDLPProblemSpec(
        substrate_archive_sha256s=(
            "sha_a" * 13 + "x",  # 65-char synthetic sha
            "sha_b" * 13 + "x",
            "sha_c" * 13 + "x",
        ),
        per_axis_gradient_l2_norms=(
            (0.5, 1.0, 0.3),  # substrate 0: pose dominant
            (2.0, 0.5, 0.1),  # substrate 1: seg dominant
            (0.1, 0.1, 3.0),  # substrate 2: rate dominant
        ),
        per_substrate_per_axis_byte_budget_caps=(
            (100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0),
        ),
        per_substrate_aggregate_byte_budget_caps=(200.0, 200.0, 200.0),
        cauchy_schwarz_aggregate_upper_bound=50.0,
    )


# ---------------------------------------------------------------------------
# Module constants pinned (Catalog #299 quota brake + canonical contract)
# ---------------------------------------------------------------------------


def test_schema_version_pinned() -> None:
    assert SCHEMA_VERSION == "pareto_polytope_unified_solution_v1"


def test_canonical_equation_id_pinned() -> None:
    assert CANONICAL_EQUATION_ID == (
        "pareto_polytope_dykstra_unified_bit_budget_allocation_savings_v1"
    )


def test_default_max_iterations_pinned() -> None:
    assert DEFAULT_MAX_ITERATIONS == 100


def test_default_tolerance_pinned() -> None:
    assert DEFAULT_TOLERANCE == 1e-6


def test_predicted_axis_tag_canonical() -> None:
    assert PREDICTED_AXIS_TAG == "[predicted]"


def test_valid_axis_labels_canonical() -> None:
    assert VALID_AXIS_LABELS == frozenset({"seg", "pose", "rate"})


# ---------------------------------------------------------------------------
# Dataclass invariants (Catalog #323 canonical contract validation)
# ---------------------------------------------------------------------------


def test_problem_spec_invariants_reject_empty_substrates() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        PareDLPProblemSpec(
            substrate_archive_sha256s=(),
            per_axis_gradient_l2_norms=(),
            per_substrate_per_axis_byte_budget_caps=(),
            per_substrate_aggregate_byte_budget_caps=(),
            cauchy_schwarz_aggregate_upper_bound=0.0,
        )


def test_problem_spec_invariants_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        PareDLPProblemSpec(
            substrate_archive_sha256s=("a", "b"),
            per_axis_gradient_l2_norms=((1.0, 1.0, 1.0),),  # only 1, need 2
            per_substrate_per_axis_byte_budget_caps=(
                (10.0, 10.0, 10.0),
                (10.0, 10.0, 10.0),
            ),
            per_substrate_aggregate_byte_budget_caps=(10.0, 10.0),
            cauchy_schwarz_aggregate_upper_bound=1.0,
        )


def test_problem_spec_invariants_reject_negative_gradient_norm() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PareDLPProblemSpec(
            substrate_archive_sha256s=("a",),
            per_axis_gradient_l2_norms=((-1.0, 1.0, 1.0),),
            per_substrate_per_axis_byte_budget_caps=((10.0, 10.0, 10.0),),
            per_substrate_aggregate_byte_budget_caps=(10.0,),
            cauchy_schwarz_aggregate_upper_bound=1.0,
        )


def test_problem_spec_invariants_reject_negative_cs_bound() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PareDLPProblemSpec(
            substrate_archive_sha256s=("a",),
            per_axis_gradient_l2_norms=((1.0, 1.0, 1.0),),
            per_substrate_per_axis_byte_budget_caps=((10.0, 10.0, 10.0),),
            per_substrate_aggregate_byte_budget_caps=(10.0,),
            cauchy_schwarz_aggregate_upper_bound=-0.1,
        )


def test_problem_spec_m_property(synthetic_3sub_3axis_spec: PareDLPProblemSpec) -> None:
    assert synthetic_3sub_3axis_spec.m == 3


def test_problem_spec_as_dict_round_trip_keys(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    d = synthetic_3sub_3axis_spec.as_dict()
    assert "substrate_archive_sha256s" in d
    assert "per_axis_gradient_l2_norms" in d
    assert "cauchy_schwarz_aggregate_upper_bound" in d
    # JSON-serializable
    json.dumps(d, allow_nan=False)


def test_allocation_invariants_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        UnifiedBitBudgetAllocation(
            substrate_archive_sha256s=("a", "b"),
            per_substrate_per_axis_allocations=((1.0, 1.0, 1.0),),  # only 1
            per_substrate_aggregate_allocations=(3.0, 3.0),
            aggregate_total_bytes_allocated=6.0,
            aggregate_predicted_delta_s=0.0,
            feasible=True,
            feasibility_residual=0.0,
        )


def test_allocation_tolerates_small_negative_noise() -> None:
    """Per Boyd 2004 §7.2 numerical noise: allow small negative residuals."""
    alloc = UnifiedBitBudgetAllocation(
        substrate_archive_sha256s=("a",),
        per_substrate_per_axis_allocations=((-1e-10, 1.0, 1.0),),
        per_substrate_aggregate_allocations=(2.0,),
        aggregate_total_bytes_allocated=2.0,
        aggregate_predicted_delta_s=0.0,
        feasible=True,
        feasibility_residual=0.0,
    )
    assert alloc.per_substrate_per_axis_allocations[0][0] == pytest.approx(-1e-10)


def test_allocation_rejects_large_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        UnifiedBitBudgetAllocation(
            substrate_archive_sha256s=("a",),
            per_substrate_per_axis_allocations=((-1.0, 1.0, 1.0),),
            per_substrate_aggregate_allocations=(1.0,),
            aggregate_total_bytes_allocated=1.0,
            aggregate_predicted_delta_s=0.0,
            feasible=True,
            feasibility_residual=0.0,
        )


# ---------------------------------------------------------------------------
# Projection operator correctness (canonical Boyd 2004 §6.4 closed-forms)
# ---------------------------------------------------------------------------


def test_project_onto_per_axis_box_clips_to_bounds() -> None:
    x = np.array([[1.5, -0.5, 0.5]])
    caps = np.array([[1.0, 1.0, 1.0]])
    out = _project_onto_per_axis_box(x, caps)
    np.testing.assert_array_almost_equal(out, [[1.0, 0.0, 0.5]])


def test_project_onto_non_negativity_zeros_negatives() -> None:
    x = np.array([[1.0, -2.0, 3.0]])
    out = _project_onto_non_negativity(x)
    np.testing.assert_array_almost_equal(out, [[1.0, 0.0, 3.0]])


def test_project_onto_cs_bound_identity_when_feasible() -> None:
    """When <a, x> <= bound, projection is identity."""
    x = np.array([[1.0, 1.0, 1.0]])
    norms = np.array([[1.0, 1.0, 1.0]])  # <a, x> = 3.0
    out = _project_onto_cauchy_schwarz_bound(x, norms, cauchy_schwarz_bound=10.0)
    np.testing.assert_array_almost_equal(out, x)


def test_project_onto_cs_bound_active_when_violated() -> None:
    """When <a, x> > bound, projection moves x onto the half-space boundary."""
    x = np.array([[10.0, 10.0, 10.0]])
    norms = np.array([[1.0, 1.0, 1.0]])  # <a, x> = 30.0; bound = 6.0
    out = _project_onto_cauchy_schwarz_bound(x, norms, cauchy_schwarz_bound=6.0)
    # After projection, <a, out> should equal 6.0 (active constraint).
    cs_val = float(np.sum(norms * out))
    assert cs_val == pytest.approx(6.0, abs=1e-9)


def test_project_onto_aggregate_budget_scales_down_when_over() -> None:
    """When Σ_axis x_i_axis > B_i, scale down proportionally."""
    x = np.array([[10.0, 10.0, 10.0]])  # sum = 30
    caps = np.array([12.0])  # cap = 12
    out = _project_onto_per_substrate_aggregate_budget(x, caps)
    # Sum should equal cap (active constraint).
    assert float(np.sum(out)) == pytest.approx(12.0, abs=1e-9)


def test_project_onto_aggregate_budget_identity_when_under() -> None:
    """When Σ_axis x_i_axis <= B_i, projection is identity."""
    x = np.array([[1.0, 2.0, 3.0]])  # sum = 6
    caps = np.array([10.0])  # cap = 10
    out = _project_onto_per_substrate_aggregate_budget(x, caps)
    np.testing.assert_array_almost_equal(out, x)


# ---------------------------------------------------------------------------
# Solver convergence + canonical output contract
# ---------------------------------------------------------------------------


def test_solver_converges_on_synthetic_problem(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    sol = solve_pareto_polytope_via_dykstra_projections(
        synthetic_3sub_3axis_spec, max_iterations=2000, tol=1e-8
    )
    assert sol.converged
    assert sol.n_iterations_to_convergence > 0
    assert sol.n_iterations_to_convergence < 2000


def test_solver_respects_cauchy_schwarz_bound(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    """No allocation may violate META-LIFT-1's Cauchy-Schwarz upper bound."""
    sol = solve_pareto_polytope_via_dykstra_projections(
        synthetic_3sub_3axis_spec, max_iterations=2000, tol=1e-8
    )
    # Aggregate predicted ΔS = <a, x> must be <= CS bound.
    cs_bound = synthetic_3sub_3axis_spec.cauchy_schwarz_aggregate_upper_bound
    assert sol.allocation.aggregate_predicted_delta_s <= cs_bound + 1e-5


def test_solver_respects_per_axis_box_caps(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    sol = solve_pareto_polytope_via_dykstra_projections(
        synthetic_3sub_3axis_spec, max_iterations=2000, tol=1e-8
    )
    for i, alloc in enumerate(sol.allocation.per_substrate_per_axis_allocations):
        caps = synthetic_3sub_3axis_spec.per_substrate_per_axis_byte_budget_caps[i]
        for j in range(3):
            assert alloc[j] >= -1e-5
            assert alloc[j] <= caps[j] + 1e-5


def test_solver_respects_per_substrate_aggregate_caps(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    sol = solve_pareto_polytope_via_dykstra_projections(
        synthetic_3sub_3axis_spec, max_iterations=2000, tol=1e-8
    )
    for i, agg in enumerate(sol.allocation.per_substrate_aggregate_allocations):
        cap = synthetic_3sub_3axis_spec.per_substrate_aggregate_byte_budget_caps[i]
        assert agg <= cap + 1e-5


def test_solver_determinism_same_inputs_same_output(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    """Catalog #294 Dim 7 deterministic reproducibility regression guard."""
    sol1 = solve_pareto_polytope_via_dykstra_projections(
        synthetic_3sub_3axis_spec, max_iterations=500, tol=1e-7
    )
    sol2 = solve_pareto_polytope_via_dykstra_projections(
        synthetic_3sub_3axis_spec, max_iterations=500, tol=1e-7
    )
    np.testing.assert_array_almost_equal(
        np.asarray(sol1.allocation.per_substrate_per_axis_allocations),
        np.asarray(sol2.allocation.per_substrate_per_axis_allocations),
    )


def test_solver_emits_canonical_provenance(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    """Catalog #323 + #341: every solution carries canonical Provenance markers."""
    sol = solve_pareto_polytope_via_dykstra_projections(synthetic_3sub_3axis_spec)
    assert sol.axis_tag == PREDICTED_AXIS_TAG
    assert sol.score_claim is False
    assert sol.promotable is False
    assert sol.evidence_grade.startswith("[predicted;")
    assert sol.canonical_equation_id == CANONICAL_EQUATION_ID
    assert sol.canonical_equation_status == "FORMALIZATION_PENDING"


def test_solver_emits_per_axis_decomposition_per_catalog_356(
    synthetic_3sub_3axis_spec: PareDLPProblemSpec,
) -> None:
    """Per Catalog #356: per-axis decomposition must be present in output."""
    sol = solve_pareto_polytope_via_dykstra_projections(synthetic_3sub_3axis_spec)
    for alloc in sol.allocation.per_substrate_per_axis_allocations:
        assert len(alloc) == 3  # seg, pose, rate per Catalog #356


def test_solver_handles_zero_init() -> None:
    """All-zeros init must produce all-zeros output (boundary of intersection)."""
    spec = PareDLPProblemSpec(
        substrate_archive_sha256s=("a",),
        per_axis_gradient_l2_norms=((1.0, 1.0, 1.0),),
        per_substrate_per_axis_byte_budget_caps=((10.0, 10.0, 10.0),),
        per_substrate_aggregate_byte_budget_caps=(30.0,),
        cauchy_schwarz_aggregate_upper_bound=100.0,  # very loose
    )
    sol = solve_pareto_polytope_via_dykstra_projections(
        spec, initial_allocation=np.zeros((1, 3))
    )
    np.testing.assert_array_almost_equal(
        np.asarray(sol.allocation.per_substrate_per_axis_allocations), [[0.0, 0.0, 0.0]]
    )


def test_solver_raises_on_invalid_max_iterations() -> None:
    spec = PareDLPProblemSpec(
        substrate_archive_sha256s=("a",),
        per_axis_gradient_l2_norms=((1.0, 1.0, 1.0),),
        per_substrate_per_axis_byte_budget_caps=((1.0, 1.0, 1.0),),
        per_substrate_aggregate_byte_budget_caps=(1.0,),
        cauchy_schwarz_aggregate_upper_bound=1.0,
    )
    with pytest.raises(ValueError, match="max_iterations"):
        solve_pareto_polytope_via_dykstra_projections(spec, max_iterations=0)


def test_solver_raises_on_invalid_tol() -> None:
    spec = PareDLPProblemSpec(
        substrate_archive_sha256s=("a",),
        per_axis_gradient_l2_norms=((1.0, 1.0, 1.0),),
        per_substrate_per_axis_byte_budget_caps=((1.0, 1.0, 1.0),),
        per_substrate_aggregate_byte_budget_caps=(1.0,),
        cauchy_schwarz_aggregate_upper_bound=1.0,
    )
    with pytest.raises(ValueError, match="tol"):
        solve_pareto_polytope_via_dykstra_projections(spec, tol=-1.0)


# ---------------------------------------------------------------------------
# META-LIFT-1 integration bridge tests
# ---------------------------------------------------------------------------


def test_build_problem_spec_from_meta_lift_1_analysis_dict_form() -> None:
    """Bridge accepts META-LIFT-1 analysis dict (load_analyses_strict output)."""
    analysis_dict = {
        "substrate_rows": [
            {
                "archive_sha256": "a" * 64,
                "measurement_axis": "[contest-CUDA]",
                "n_bytes": 1000,
                "per_axis_projections": [
                    {"axis": "seg", "gradient_l2_norm": 1.5},
                    {"axis": "pose", "gradient_l2_norm": 2.5},
                    {"axis": "rate", "gradient_l2_norm": 0.5},
                ],
            }
        ],
        "cauchy_schwarz_aggregate_upper_bound": 10.0,
    }
    spec = build_problem_spec_from_meta_lift_1_analysis(analysis_dict)
    assert spec.m == 1
    assert spec.per_axis_gradient_l2_norms[0] == (1.5, 2.5, 0.5)
    assert spec.cauchy_schwarz_aggregate_upper_bound == 10.0


def test_build_problem_spec_from_meta_lift_1_analysis_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one substrate row"):
        build_problem_spec_from_meta_lift_1_analysis(
            {"substrate_rows": [], "cauchy_schwarz_aggregate_upper_bound": 0.0}
        )


def test_build_problem_spec_from_meta_lift_1_analysis_default_cap_fractions() -> None:
    """Default per-axis cap = 10% / 3 axes; aggregate cap = 20%."""
    analysis_dict = {
        "substrate_rows": [
            {
                "archive_sha256": "a" * 64,
                "n_bytes": 3000,
                "per_axis_projections": [
                    {"axis": "seg", "gradient_l2_norm": 1.0},
                    {"axis": "pose", "gradient_l2_norm": 1.0},
                    {"axis": "rate", "gradient_l2_norm": 1.0},
                ],
            }
        ],
        "cauchy_schwarz_aggregate_upper_bound": 100.0,
    }
    spec = build_problem_spec_from_meta_lift_1_analysis(analysis_dict)
    # Per-axis cap = (3000 * 0.10) / 3 = 100.0
    assert spec.per_substrate_per_axis_byte_budget_caps[0] == (100.0, 100.0, 100.0)
    # Aggregate cap = 3000 * 0.20 = 600.0
    assert spec.per_substrate_aggregate_byte_budget_caps[0] == 600.0


# ---------------------------------------------------------------------------
# Canonical ledger persistence (Catalog #131/#138/#245 sister discipline)
# ---------------------------------------------------------------------------


def test_append_solution_locked_writes_canonical_row(
    tmp_path: Path, synthetic_3sub_3axis_spec: PareDLPProblemSpec
) -> None:
    sol = solve_pareto_polytope_via_dykstra_projections(synthetic_3sub_3axis_spec)
    ledger = tmp_path / "test_solutions.jsonl"
    written = append_solution_locked(sol, path=ledger)
    assert "solution_id" in written
    assert "written_at_utc" in written
    assert ledger.exists()
    # Lenient re-read
    text = ledger.read_text(encoding="utf-8").strip()
    assert text
    row = json.loads(text.splitlines()[-1])
    assert row["canonical_equation_id"] == CANONICAL_EQUATION_ID


def test_load_solutions_strict_raises_on_corruption(tmp_path: Path) -> None:
    ledger = tmp_path / "corrupt.jsonl"
    ledger.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(PareDLPSolutionCorruptError):
        load_solutions_strict(ledger)


def test_load_solutions_strict_returns_empty_when_missing(tmp_path: Path) -> None:
    rows = load_solutions_strict(tmp_path / "missing.jsonl")
    assert rows == []


def test_load_solutions_strict_round_trip(
    tmp_path: Path, synthetic_3sub_3axis_spec: PareDLPProblemSpec
) -> None:
    sol = solve_pareto_polytope_via_dykstra_projections(synthetic_3sub_3axis_spec)
    ledger = tmp_path / "round_trip.jsonl"
    append_solution_locked(sol, path=ledger)
    rows = load_solutions_strict(ledger)
    assert len(rows) == 1
    assert rows[0]["solution_id"] == sol.solution_id


# ---------------------------------------------------------------------------
# Cathedral consumer contract compliance (Catalog #335)
# ---------------------------------------------------------------------------


def test_cathedral_consumer_protocol_satisfied() -> None:
    """Per Catalog #335: the cathedral consumer satisfies the canonical contract."""
    from tac.cathedral.consumer_contract import validate_consumer_module
    from tac.cathedral_consumers import pareto_polytope_unified_solver_consumer

    # Should not raise.
    validate_consumer_module(pareto_polytope_unified_solver_consumer)


def test_cathedral_consumer_emits_canonical_routing_markers() -> None:
    """Per Catalog #341: routing markers must be present in every return."""
    from tac.cathedral_consumers import pareto_polytope_unified_solver_consumer

    candidate = {"archive_sha256": "synthetic_sha_no_solution_yet"}
    out = pareto_polytope_unified_solver_consumer.consume_candidate(candidate)
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"


def test_cathedral_consumer_handles_missing_solution_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ledger must NOT crash the cathedral autopilot loop."""
    from tac.cathedral_consumers import pareto_polytope_unified_solver_consumer

    # Monkeypatch _state_dir to a non-existent path.
    monkeypatch.setattr(
        pareto_polytope_unified_solver_consumer,
        "_state_dir",
        lambda: Path("/nonexistent/path/.omx/state"),
    )
    out = pareto_polytope_unified_solver_consumer.consume_candidate(
        {"archive_sha256": "test"}
    )
    assert out["annotation"]["pareto_polytope_solution_status"] == "MISSING_SOLUTION_LEDGER"


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


CLI_PATH = Path(__file__).resolve().parents[3] / "tools" / "pareto_polytope_solver_cli.py"


def test_cli_exits_2_on_invalid_arg() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--max-iterations", "-1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "max-iterations" in result.stderr


def test_cli_exits_2_when_no_meta_lift_1_ledger(tmp_path: Path) -> None:
    fake_ledger = tmp_path / "missing.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--meta-lift-1-ledger-path",
            str(fake_ledger),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "META-LIFT-1" in result.stderr or "ledger" in result.stderr


# ---------------------------------------------------------------------------
# Integration test: consume META-LIFT-1 analyzer output end-to-end
# ---------------------------------------------------------------------------


def test_end_to_end_meta_lift_1_then_meta_lift_2() -> None:
    """Per the 11th standing directive ORDER: META-LIFT-1 → META-LIFT-2."""
    from tac.cross_substrate_master_gradient_analyzer import (
        analyze_cross_substrate_master_gradients,
    )

    # Synthetic 2-substrate inputs (avoid hitting real ledger).
    rng = np.random.default_rng(42)
    substrate_inputs = [
        {
            "gradient_array": rng.standard_normal((100, 3)) * 0.01,
            "archive_sha256": "a" * 64,
            "measurement_axis": "[contest-CUDA]",
            "measurement_hardware": "linux_x86_64_t4",
            "measurement_call_id": "fc-synthetic-a",
            "is_authoritative": True,
        },
        {
            "gradient_array": rng.standard_normal((200, 3)) * 0.02,
            "archive_sha256": "b" * 64,
            "measurement_axis": "[contest-CUDA]",
            "measurement_hardware": "linux_x86_64_t4",
            "measurement_call_id": "fc-synthetic-b",
            "is_authoritative": True,
        },
    ]

    analysis = analyze_cross_substrate_master_gradients(substrate_inputs)
    assert len(analysis.substrate_rows) == 2

    spec = build_problem_spec_from_meta_lift_1_analysis(analysis)
    assert spec.m == 2

    solution = solve_pareto_polytope_via_dykstra_projections(
        spec, max_iterations=500, tol=1e-7
    )
    assert solution.allocation.feasible
    # Cauchy-Schwarz bound respected.
    assert (
        solution.allocation.aggregate_predicted_delta_s
        <= spec.cauchy_schwarz_aggregate_upper_bound + 1e-5
    )


# ---------------------------------------------------------------------------
# Live-repo regression guard (no STRICT preflight gates added in this lane)
# ---------------------------------------------------------------------------


def test_canonical_helper_invocation_string_canonical() -> None:
    """Per Catalog #323 canonical Provenance: helper invocation token pinned."""
    from tac.pareto_polytope_unified_solver.solver import solve_pareto_polytope_via_dykstra_projections as fn

    assert fn.__module__ == "tac.pareto_polytope_unified_solver.solver"
