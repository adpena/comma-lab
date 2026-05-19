# SPDX-License-Identifier: MIT
"""Tests for the 6 initial canonical equations (Catalog #344 anchor)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tac.canonical_equations import (
    CanonicalEquation,
    build_all_initial_equations,
    populate_initial_equations,
    query_equations,
)


EXPECTED_EQUATION_IDS = (
    "brotli_cascade_bounded_per_stream_v1",
    "mps_drift_architecture_class_dependent_v1",
    "per_byte_leverage_uniformly_distributed_v1",
    "per_pair_master_gradient_score_impact_taylor_v1",
    "master_gradient_locality_violation_by_codec_v1",
    "canonical_frontier_pointer_v1",
)


def test_six_initial_equations_built():
    eqs = build_all_initial_equations()
    assert len(eqs) == 6
    ids = {e.equation_id for e in eqs}
    assert ids == set(EXPECTED_EQUATION_IDS)


def test_each_equation_has_non_empty_anchors():
    eqs = build_all_initial_equations()
    for eq in eqs:
        assert len(eq.empirical_anchors) >= 1, (
            f"{eq.equation_id} must have >=1 empirical anchor"
        )


def test_each_equation_has_non_empty_consumers_or_producers():
    """Orphan equations are rejected by __post_init__; this guards the population."""
    eqs = build_all_initial_equations()
    for eq in eqs:
        assert eq.canonical_consumers or eq.canonical_producers


def test_each_equation_callable_module_path_well_formed():
    """Smoke-check the dotted module path is parseable; not asserting symbol exists."""
    eqs = build_all_initial_equations()
    for eq in eqs:
        path = eq.python_callable_module_path
        assert ":" in path or "." in path, (
            f"{eq.equation_id} callable path={path!r} must be dotted form"
        )


def test_each_equation_has_provenance():
    eqs = build_all_initial_equations()
    for eq in eqs:
        assert eq.provenance is not None
        # Predicted grade since registration is a design event.
        assert eq.provenance.evidence_grade.value == "predicted"


def test_mps_drift_anchor_reflects_falsification_residual():
    """Slot 16 empirical 30x miss must be recorded."""
    eqs = {e.equation_id: e for e in build_all_initial_equations()}
    mps = eqs["mps_drift_architecture_class_dependent_v1"]
    assert "tinyrenderer_phase_b_paired_mps_cuda" in mps.predicted_vs_empirical_residual
    assert mps.predicted_vs_empirical_residual["tinyrenderer_phase_b_paired_mps_cuda"] == 30.0


def test_brotli_cascade_anchor_reflects_bounded_observation():
    eqs = {e.equation_id: e for e in build_all_initial_equations()}
    brotli = eqs["brotli_cascade_bounded_per_stream_v1"]
    assert brotli.predicted_vs_empirical_residual["pr101_op7_diff"] == 0.0


def test_populate_writes_to_registry(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    populate_initial_equations(path=path, lock_path=lock)
    loaded = query_equations(path=path)
    assert len(loaded) == 6
    ids = {e.equation_id for e in loaded}
    assert ids == set(EXPECTED_EQUATION_IDS)


def test_populate_idempotent_appends_new_events(tmp_path: Path):
    """Re-populating appends a new 'registered' event; query still returns 6."""
    from tac.canonical_equations.registry import load_registry_events_lenient

    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    populate_initial_equations(path=path, lock_path=lock)
    populate_initial_equations(path=path, lock_path=lock)
    events = load_registry_events_lenient(path)
    assert len(events) == 12  # 6 from each invocation
    loaded = query_equations(path=path)
    assert len(loaded) == 6


def test_uniform_leverage_predictor_callable():
    """Equation #3's canonical callable is importable and returns a sensible value."""
    from tac.canonical_equations.builtins import uniform_leverage_predictor

    assert uniform_leverage_predictor(1.0) == 0.01
    assert uniform_leverage_predictor(50.0) == 0.5
    with pytest.raises(ValueError):
        uniform_leverage_predictor(-1.0)
    with pytest.raises(ValueError):
        uniform_leverage_predictor(101.0)


def test_each_equation_callable_module_path_importable_or_documented():
    """Verify the dotted module-path for each equation is at least importable
    (catches typos in dotted paths)."""
    eqs = build_all_initial_equations()
    skipped = []
    for eq in eqs:
        path = eq.python_callable_module_path
        module_part = path.split(":")[0]
        try:
            importlib.import_module(module_part)
        except ImportError as exc:
            skipped.append((eq.equation_id, module_part, str(exc)))
    # All 6 modules should be importable as of landing.
    assert not skipped, f"Unimportable canonical callable modules: {skipped}"


def test_live_registry_has_six_equations():
    """Live-repo regression guard: the persisted registry has the 6 equations."""
    from tac.canonical_equations import query_equations

    # Use the live registry path.
    loaded = query_equations()
    ids = {e.equation_id for e in loaded}
    # Subset check: live registry must contain at least the 6 expected.
    missing = set(EXPECTED_EQUATION_IDS) - ids
    assert not missing, f"Live registry missing equations: {missing}"
