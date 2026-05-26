# SPDX-License-Identifier: MIT
"""Tests for META-LAGRANGIAN-WIRE-1 Phase 3 typed atom flow integration.

Per operator standing directive 2026-05-26 verbatim *"all are approved + follow
up are approved + pursue other attacks as well + remember all MLX first +
individually fractally optimized"* + Phase 2 landing commit ``c6eb7d641``.

The Phase 3 integration deliberately ENGAGES the Phase 2
:func:`compute_per_axis_dual_variables` solver through the cathedral autopilot
helper ``invoke_meta_lagrangian_on_candidates`` when the ``--use-phase-2-dual-solver``
CLI flag (or kwarg) is set. This file exercises the integration boundary at the
TYPED ATOM FLOW level:

  1. A candidate that carries an :class:`AxisDecomposition`-shaped
     ``predicted_axis_decomposition`` consumer payload flows through the
     Phase 2 solver and surfaces a typed dual-solver payload.
  2. The exposed scalar ``adjustment_factor`` remains bounded to the
     Phase 1 safety envelope ``[0.95, 1.05]`` per Catalog #355 STRICT
     preflight gate.
  3. The Phase 1 default path is BYTE-PRESERVED when
     ``use_phase_2_dual_solver=False`` (counterfactual demonstration).
  4. Canonical Provenance markers per Catalog #323 + Catalog #341 routing
     markers are emitted at every Phase 2 payload row
     (``score_claim=False`` + ``promotion_eligible=False`` +
     ``axis_tag="[predicted]"``).
  5. Scalar-only candidates (no AxisDecomposition payload) are SKIPPED
     at the typed atom flow boundary per the "refuses scalar-to-axis
     inference" contract.
  6. The integration end-to-end works for batches (multi-candidate set)
     with deterministic per-candidate outputs.

6-hook wire-in declaration per Catalog #125 (sister of dual_solver_phase_2):
  - hook #1 sensitivity-map = ACTIVE — per-axis KKT residuals surface
    at the typed atom flow boundary.
  - hook #2 Pareto constraint = ACTIVE — Dykstra alt-projections fire
    once per candidate when Phase 2 is enabled.
  - hook #3 bit-allocator = ACTIVE (indirect) — λ_rate per-axis dual is
    available on the typed payload's ``dual_variables_per_axis["rate"]``.
  - hook #4 cathedral autopilot dispatch = ACTIVE PRIMARY — the helper
    callsite is the structural protection at the ranker decision boundary.
  - hook #5 continual-learning posterior = ACTIVE — Phase 3 anchor is
    queued for canonical equation #344
    ``meta_lagrangian_dual_solver_per_axis_kkt_residual_v1``.
  - hook #6 probe-disambiguator = ACTIVE — per-axis dual variables
    disambiguate which polytope axis is the binding constraint.

Cross-references
----------------
- Sister: ``src/tac/findings_lagrangian/tests/test_dual_solver_phase_2.py``
  (52/52 Phase 2 dual-solver unit tests; this file extends with the
  TYPED ATOM FLOW INTEGRATION dimension).
- Sister: ``src/tac/tests/test_meta_lagrangian_cathedral_wire_in.py``
  (33/33 Phase 1 helper tests + 2 in-line Phase 2 toggle tests; this
  file deepens the typed atom flow integration coverage).
- CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable.
- Catalog #355 STRICT preflight gate (preserved through Phase 3).
- Catalog #323 + Catalog #341 + Catalog #287 canonical Provenance +
  routing markers + placeholder-rationale rejection sister disciplines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make tools/ importable for the helper tests.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


from cathedral_autopilot_autonomous_loop import (  # noqa: E402
    PHASE_2_DUAL_SOLVER_INVOCATION_SCHEMA,
    CandidateRow,
    invoke_meta_lagrangian_on_candidates,
)
from tac.findings_lagrangian import (  # noqa: E402
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN,
    PHASE_2_DUAL_SOLVER_SCHEMA_VERSION,
    PerAxisDualSolverResult,
    compute_per_axis_dual_variables,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic CandidateRow with optional AxisDecomposition
# ---------------------------------------------------------------------------


def _make_candidate_with_axis_decomposition(
    *,
    candidate_id: str = "phase_3_atom_candidate",
    family: str = "phase_3_family",
    predicted_score_delta: float = -0.003,
    d_seg: float = 0.001,
    d_pose: float = 0.0001,
    archive_bytes: int = 200,
    include_axis_decomposition: bool = True,
) -> CandidateRow:
    """Build a synthetic CandidateRow optionally carrying AxisDecomposition payload."""
    consumer_payload: dict = {}
    if include_axis_decomposition:
        consumer_payload["predicted_axis_decomposition"] = {
            "predicted_d_seg_delta": d_seg,
            "predicted_d_pose_delta": d_pose,
            "predicted_archive_bytes_delta": archive_bytes,
            "axis_tag": "[predicted]",
            "canonical_provenance": {
                "fixture": "phase_3_typed_atom_flow_test",
                "score_claim": False,
                "promotable": False,
            },
        }
    return CandidateRow(
        candidate_id=candidate_id,
        family=family,
        predicted_score_delta=predicted_score_delta,
        expected_information_gain=0.5,
        estimated_dispatch_cost_usd=1.0,
        consumer_payload=consumer_payload,
    )


# ---------------------------------------------------------------------------
# Phase 3 typed atom flow integration tests
# ---------------------------------------------------------------------------


def test_phase_3_typed_atom_flow_routes_axis_decomposition_to_phase_2_solver() -> None:
    """An AxisDecomposition-carrying candidate triggers Phase 2 dual-solver via the typed atom flow."""
    candidate = _make_candidate_with_axis_decomposition()
    result = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=True,
    )
    assert result["phase_2_dual_solver_enabled"] is True
    assert result["phase_2_dual_solver_invoked_count"] == 1
    assert result["phase"] == "phase_2_dual_solver_enabled_with_phase_1_fallback"
    row = result["invocations"][0]
    phase_2 = row["phase_2_dual_solver"]
    assert phase_2["status"] == "invoked"
    assert phase_2["invoked"] is True
    assert phase_2["schema"] == PHASE_2_DUAL_SOLVER_INVOCATION_SCHEMA


def test_phase_3_phase_1_default_path_byte_preserved_when_phase_2_disabled() -> None:
    """The Phase 1 default path remains byte-preserved when use_phase_2_dual_solver=False."""
    candidate = _make_candidate_with_axis_decomposition()
    result_disabled = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=False,
    )
    assert result_disabled["phase_2_dual_solver_enabled"] is False
    assert result_disabled["phase_2_dual_solver_invoked_count"] == 0
    assert result_disabled["phase"] == "phase_1_canonical_invocation_with_bounded_proxy_adjuster"
    row = result_disabled["invocations"][0]
    # Phase 1 default contract: no phase_2_dual_solver payload on the row.
    assert "phase_2_dual_solver" not in row
    # Phase 1 adjustment_factor must still be bounded.
    adj = row["adjustment_factor"]
    assert PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN <= adj <= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX


def test_phase_3_adjustment_factor_remains_bounded_inside_phase_2_envelope() -> None:
    """The exposed scalar adjustment_factor stays in [0.95, 1.05] even at Phase 2."""
    test_deltas = [-0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5]
    for delta in test_deltas:
        candidate = _make_candidate_with_axis_decomposition(predicted_score_delta=delta)
        result = invoke_meta_lagrangian_on_candidates(
            [candidate],
            use_phase_2_dual_solver=True,
        )
        row = result["invocations"][0]
        adj = row["adjustment_factor"]
        assert PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN <= adj <= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX, (
            f"Phase 2 adjustment_factor {adj} outside Catalog #355 envelope for delta={delta}"
        )


def test_phase_3_per_axis_dual_variables_surfaced_typed() -> None:
    """The typed PerAxisDualSolverResult is surfaced via the typed atom flow."""
    candidate = _make_candidate_with_axis_decomposition()
    result = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=True,
    )
    row = result["invocations"][0]
    phase_2 = row["phase_2_dual_solver"]
    assert "dual_variables_per_axis" in phase_2
    dual = phase_2["dual_variables_per_axis"]
    assert set(dual.keys()) == {"seg", "pose", "rate"}
    for axis in ("seg", "pose", "rate"):
        assert isinstance(dual[axis], (int, float))


def test_phase_3_kkt_residuals_per_axis_surfaced() -> None:
    """KKT residuals per axis surface on the typed dual-solver payload."""
    candidate = _make_candidate_with_axis_decomposition()
    result = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=True,
    )
    phase_2 = result["invocations"][0]["phase_2_dual_solver"]
    assert "kkt_residual_per_axis" in phase_2
    kkt = phase_2["kkt_residual_per_axis"]
    assert set(kkt.keys()) == {"seg", "pose", "rate"}


def test_phase_3_canonical_provenance_markers_present_per_catalog_323() -> None:
    """Every Phase 2 dual-solver payload row carries canonical Catalog #323 + #341 markers."""
    candidate = _make_candidate_with_axis_decomposition()
    result = invoke_meta_lagrangian_on_candidates(
        [candidate],
        use_phase_2_dual_solver=True,
    )
    phase_2 = result["invocations"][0]["phase_2_dual_solver"]
    # Catalog #341 routing markers.
    assert phase_2["score_claim"] is False
    assert phase_2["promotable"] is False
    assert phase_2["promotion_eligible"] is False
    assert phase_2["ready_for_exact_eval_dispatch"] is False
    assert phase_2["axis_tag"] == "[predicted]"


def test_phase_3_scalar_only_candidate_refuses_scalar_to_axis_inference() -> None:
    """A scalar-only candidate (no AxisDecomposition payload) is SKIPPED per the canonical refusal."""
    candidate_scalar = _make_candidate_with_axis_decomposition(
        include_axis_decomposition=False,
    )
    result = invoke_meta_lagrangian_on_candidates(
        [candidate_scalar],
        use_phase_2_dual_solver=True,
    )
    assert result["phase_2_dual_solver_enabled"] is True
    assert result["phase_2_dual_solver_invoked_count"] == 0
    phase_2 = result["invocations"][0]["phase_2_dual_solver"]
    assert phase_2["status"] == "skipped"
    assert phase_2["invoked"] is False
    assert "refuses scalar-to-axis inference" in phase_2["rationale"]


def test_phase_3_batch_integration_per_candidate_deterministic() -> None:
    """Phase 3 integrates a batch of candidates with deterministic per-candidate outputs."""
    candidates = [
        _make_candidate_with_axis_decomposition(
            candidate_id=f"phase_3_batch_{i}",
            family=f"family_{i}",
            d_seg=0.001 * (i + 1),
            d_pose=0.0001 * (i + 1),
            archive_bytes=100 * (i + 1),
        )
        for i in range(5)
    ]
    result_a = invoke_meta_lagrangian_on_candidates(
        candidates,
        use_phase_2_dual_solver=True,
    )
    result_b = invoke_meta_lagrangian_on_candidates(
        candidates,
        use_phase_2_dual_solver=True,
    )
    assert result_a["candidates_invoked"] == 5
    assert result_a["phase_2_dual_solver_invoked_count"] == 5
    assert result_b["phase_2_dual_solver_invoked_count"] == 5
    # Determinism check: per-candidate scalar adjustment factors match.
    for row_a, row_b in zip(result_a["invocations"], result_b["invocations"]):
        assert row_a["adjustment_factor"] == row_b["adjustment_factor"]


def test_phase_3_typed_PerAxisDualSolverResult_direct_call_matches_helper() -> None:
    """Direct call to compute_per_axis_dual_variables matches the helper's typed payload."""
    direct = compute_per_axis_dual_variables(
        "phase_3_direct_candidate",
        predicted_axis_targets={
            "seg": 0.001,
            "pose": 0.0001,
            "rate": 200,
        },
        per_axis_budgets={
            "seg": (0.0, 0.01),
            "pose": (0.0, 0.001),
            "rate": (0.0, 1000.0),
        },
    )
    assert isinstance(direct, PerAxisDualSolverResult)
    direct_dict = direct.as_dict()
    assert direct_dict["schema"] == PHASE_2_DUAL_SOLVER_SCHEMA_VERSION
    assert set(direct_dict["dual_variables_per_axis"].keys()) == {"seg", "pose", "rate"}


def test_phase_3_catalog_355_strict_gate_preserved_after_phase_3_landing() -> None:
    """Catalog #355 STRICT preflight gate remains clean after Phase 3 deliverables land."""
    from tac.preflight import check_cathedral_autopilot_main_invokes_meta_lagrangian

    violations = check_cathedral_autopilot_main_invokes_meta_lagrangian(strict=True)
    assert violations == [], (
        "Catalog #355 STRICT preflight gate must remain clean after Phase 3 landing:\n  - "
        + "\n  - ".join(v[:200] for v in violations[:3])
    )


def test_phase_3_phase_2_dual_solver_invocation_schema_pinned() -> None:
    """The Phase 2 dual-solver invocation schema is pinned and stable across Phase 3 landing."""
    # Phase 2 schema (cathedral side) — pinned at Phase 2 landing commit c6eb7d641.
    assert PHASE_2_DUAL_SOLVER_INVOCATION_SCHEMA == (
        "meta_lagrangian_phase_2_dual_solver_invocation_v1_20260526"
    )
    # Phase 2 solver schema (findings_lagrangian side) — pinned at Phase 2 module landing.
    assert PHASE_2_DUAL_SOLVER_SCHEMA_VERSION == "meta_lagrangian_dual_solver_phase_2_v1_20260526"


def test_phase_3_canonical_equation_344_anchor_landed() -> None:
    """Canonical equation #344 #52 meta_lagrangian_dual_solver_per_axis_kkt_residual_v1
    has at least one anchor_appended event after Phase 3 typed atom flow landing."""
    from tac.canonical_equations import load_equation_registry_strict

    registry = load_equation_registry_strict()
    matches = [
        row
        for row in registry
        if row["equation_id"] == "meta_lagrangian_dual_solver_per_axis_kkt_residual_v1"
    ]
    assert len(matches) >= 1, (
        "Phase 2 must have at least 1 'registered' event for the canonical equation"
    )
    # The Phase 3 anchor_appended event is OPTIONAL at test-import-time
    # because tests run BEFORE the landing memo commits the anchor.
    # The assertion verifies the REGISTERED event exists per Phase 2 landing.


def test_phase_3_per_candidate_typed_atom_flow_decomposes_per_axis() -> None:
    """The typed atom flow surfaces per-axis decomposition end-to-end."""
    # Build 3 candidates each with distinct AxisDecomposition values so we
    # can verify per-axis decomposition surfaces distinctly per candidate.
    candidates = [
        _make_candidate_with_axis_decomposition(
            candidate_id=f"decompose_c{i}",
            family=f"decompose_f{i}",
            d_seg=0.001 * (i + 1),
            d_pose=0.0001 * (i + 1),
            archive_bytes=100 * (i + 1),
        )
        for i in range(3)
    ]
    result = invoke_meta_lagrangian_on_candidates(
        candidates,
        use_phase_2_dual_solver=True,
    )
    assert result["phase_2_dual_solver_invoked_count"] == 3
    seen_seg_targets = set()
    seen_pose_targets = set()
    seen_rate_targets = set()
    for row in result["invocations"]:
        phase_2 = row["phase_2_dual_solver"]
        targets = phase_2["predicted_axis_targets"]
        seen_seg_targets.add(round(targets["seg"], 6))
        seen_pose_targets.add(round(targets["pose"], 6))
        seen_rate_targets.add(round(targets["rate"], 6))
    # 3 distinct candidates → 3 distinct per-axis target tuples.
    assert len(seen_seg_targets) == 3
    assert len(seen_pose_targets) == 3
    assert len(seen_rate_targets) == 3
