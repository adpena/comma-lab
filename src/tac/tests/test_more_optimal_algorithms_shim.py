# SPDX-License-Identifier: MIT
"""Tests for the canonical ``tac.solvers.more_optimal_algorithms`` shim package.

Per grand council T3 finding #6 PROCEED op-routables #4 + #5
(2026-05-18; lane ``lane_more_optimal_algorithms_wire_in_20260518``).

Covers:

* Shim package re-exports the 3 PROCEED algorithms (FISTA / Frank-Wolfe / RN).
* :data:`CANONICAL_ALGORITHM_REGISTRY` carries paper-citations + wall-clock
  multipliers + invariant contracts per Catalog #305 observability surface.
* :func:`build_more_optimal_algorithm_wall_clock_atom` mints valid Atoms with:
    - PROCEED verdict iff speedup>=1.0 AND invariant_check_passed
    - OPERATOR_REVIEW_REQUIRED verdict iff regression (per CLAUDE.md
      "Forbidden premature KILL" — research-defer never kill)
    - ValueError on unknown algorithm_id
    - ValueError on non-positive wall-clock
* :func:`append_more_optimal_algorithm_wall_clock_event` round-trips through
  the fcntl-locked atom ledger.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tac.solvers.more_optimal_algorithms import (
    ALGORITHM_FISTA,
    ALGORITHM_FRANK_WOLFE_KCARD,
    ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL,
    CANONICAL_ALGORITHM_REGISTRY,
    AlgorithmCanonicalMetadata,
    append_more_optimal_algorithm_wall_clock_event,
    build_more_optimal_algorithm_wall_clock_atom,
)


class TestShimReExports:
    """Verify the shim package re-exports the 3 PROCEED algorithms."""

    def test_fista_re_exports_present(self):
        from tac.solvers.more_optimal_algorithms import (
            FistaResult,
            fista_proximal_gradient,
            project_simplex,
            soft_threshold,
        )

        assert callable(fista_proximal_gradient)
        assert callable(soft_threshold)
        assert callable(project_simplex)
        assert FistaResult.__module__ == "tac.solvers.fista"

    def test_frank_wolfe_re_exports_present(self):
        from tac.solvers.more_optimal_algorithms import (
            FrankWolfeResult,
            frank_wolfe_kcard,
            lmo_kcardinality,
        )

        assert callable(frank_wolfe_kcard)
        assert callable(lmo_kcardinality)
        assert FrankWolfeResult.__module__ == "tac.solvers.frank_wolfe"

    def test_riemannian_newton_re_exports_present(self):
        from tac.solvers.more_optimal_algorithms import (
            RiemannianNewtonResult,
            StiefelManifold,
            project_to_stiefel,
            retract_qr,
            riemannian_newton_step,
        )

        assert callable(riemannian_newton_step)
        assert callable(retract_qr)
        assert callable(project_to_stiefel)
        assert StiefelManifold.__module__ == "tac.solvers.riemannian_newton_stiefel"

    def test_sinkhorn_paired_sister_re_exported(self):
        from tac.solvers.more_optimal_algorithms import (
            sinkhorn_ensemble_select_k,
            sinkhorn_knopp,
        )

        assert callable(sinkhorn_knopp)
        assert callable(sinkhorn_ensemble_select_k)


class TestCanonicalAlgorithmRegistry:
    """Verify the registry contract per Catalog #305 observability surface."""

    def test_registry_has_three_proceed_algorithms(self):
        assert len(CANONICAL_ALGORITHM_REGISTRY) == 3
        assert "fista_beck_teboulle_2009" in CANONICAL_ALGORITHM_REGISTRY
        assert "frank_wolfe_kcardinality_jaggi_2013" in CANONICAL_ALGORITHM_REGISTRY
        assert (
            "riemannian_newton_stiefel_edelman_1998" in CANONICAL_ALGORITHM_REGISTRY
        )

    def test_fista_metadata_has_paired_comparison_anchors(self):
        meta = ALGORITHM_FISTA
        assert meta.wall_clock_multiplier_macos_cpu_advisory == 1.25
        assert "Beck & Teboulle 2009" in meta.paper_citation
        assert "DOI 10.1137/080716542" in meta.paper_citation
        assert meta.canonical_module_path == "tac.solvers.fista"
        assert "byte-identical" in meta.invariant_contract
        assert "O(1/k^2)" in meta.convergence_rate_theoretical
        assert len(meta.consumer_callsites) >= 1

    def test_frank_wolfe_metadata_has_paired_comparison_anchors(self):
        meta = ALGORITHM_FRANK_WOLFE_KCARD
        assert meta.wall_clock_multiplier_macos_cpu_advisory == 1.9
        assert "Jaggi 2013" in meta.paper_citation
        assert "Frank & Wolfe 1956" in meta.paper_citation
        assert meta.canonical_module_path == "tac.solvers.frank_wolfe"
        assert "sparsity invariant" in meta.invariant_contract
        assert "K-sparsity" in meta.invariant_contract or "exactly K" in meta.invariant_contract

    def test_riemannian_newton_metadata_has_paired_comparison_anchors(self):
        meta = ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL
        assert meta.wall_clock_multiplier_macos_cpu_advisory == 1.88
        assert "Edelman" in meta.paper_citation
        assert "DOI 10.1137/S0895479895290954" in meta.paper_citation
        assert meta.canonical_module_path == "tac.solvers.riemannian_newton_stiefel"
        assert "orthogonality" in meta.invariant_contract

    def test_all_registry_entries_are_frozen_dataclass(self):
        for meta in CANONICAL_ALGORITHM_REGISTRY.values():
            assert isinstance(meta, AlgorithmCanonicalMetadata)
            with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
                meta.algorithm_id = "mutated"

    def test_all_registry_entries_cite_macos_cpu_advisory_per_catalog_192_317(self):
        # Every multiplier is paired-comparison-validated on local M5 Max CPU
        # per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #317.
        # Non-positive multiplier is invalid (would mean canonical is faster).
        for meta in CANONICAL_ALGORITHM_REGISTRY.values():
            assert meta.wall_clock_multiplier_macos_cpu_advisory > 1.0, (
                f"{meta.algorithm_id} must beat canonical (multiplier > 1.0); "
                f"got {meta.wall_clock_multiplier_macos_cpu_advisory}"
            )

    def test_registry_consumer_callsites_are_real_paths(self):
        """Sister regression guard: callsites should look like py paths."""
        for meta in CANONICAL_ALGORITHM_REGISTRY.values():
            for cs in meta.consumer_callsites:
                assert cs.startswith("src/tac/"), f"non-canonical callsite: {cs}"
                assert ".py" in cs, f"missing .py extension: {cs}"


class TestBuildMoreOptimalAlgorithmWallClockAtom:
    """Op-routable #5: wall-clock atom builder."""

    def test_proceed_verdict_when_speedup_and_invariant_preserved(self):
        atom = build_more_optimal_algorithm_wall_clock_atom(
            atom_id="test_proceed",
            algorithm_id="fista_beck_teboulle_2009",
            consumer_callsite="src/tac/bit_allocator.py::allocate_bits",
            wall_clock_seconds_canonical=1.25,
            wall_clock_seconds_more_optimal=1.0,
            n_iterations_canonical=64,
            n_iterations_more_optimal=42,
            invariant_check_passed=True,
            invariant_description="byte-identical bit allocation",
        )
        assert atom.metadata["verdict"] == "PROCEED"
        assert atom.metadata["wall_clock_speedup_multiplier"] == 1.25
        assert atom.metadata["invariant_check_passed"] is True
        assert atom.metadata["substrate"] == (
            "src/tac/bit_allocator.py::allocate_bits"
        )
        assert "wire" in atom.metadata["next_action"]

    def test_operator_review_required_when_regression(self):
        # Slower → regression should NOT silently downgrade to PROCEED
        atom = build_more_optimal_algorithm_wall_clock_atom(
            atom_id="test_regression",
            algorithm_id="fista_beck_teboulle_2009",
            consumer_callsite="src/tac/bit_allocator.py::allocate_bits",
            wall_clock_seconds_canonical=1.0,
            wall_clock_seconds_more_optimal=2.0,  # 2x slower
            n_iterations_canonical=64,
            n_iterations_more_optimal=200,
            invariant_check_passed=True,
            invariant_description="byte-identical bit allocation",
        )
        assert atom.metadata["verdict"] == "OPERATOR_REVIEW_REQUIRED"
        assert atom.metadata["wall_clock_speedup_multiplier"] == 0.5
        assert "research-defer" in atom.metadata["next_action"]
        assert "Forbidden premature KILL" in atom.metadata["next_action"]

    def test_operator_review_required_when_invariant_broken(self):
        # Even if faster, if invariant broken → OPERATOR_REVIEW (no auto-PROCEED)
        atom = build_more_optimal_algorithm_wall_clock_atom(
            atom_id="test_invariant_broken",
            algorithm_id="fista_beck_teboulle_2009",
            consumer_callsite="src/tac/bit_allocator.py::allocate_bits",
            wall_clock_seconds_canonical=1.0,
            wall_clock_seconds_more_optimal=0.5,  # 2x faster
            n_iterations_canonical=64,
            n_iterations_more_optimal=20,
            invariant_check_passed=False,  # but invariant broken
            invariant_description="byte-identical bit allocation",
        )
        assert atom.metadata["verdict"] == "OPERATOR_REVIEW_REQUIRED"
        assert atom.metadata["wall_clock_speedup_multiplier"] == 2.0
        assert atom.metadata["invariant_check_passed"] is False

    def test_value_error_on_unknown_algorithm_id(self):
        with pytest.raises(ValueError, match="not in CANONICAL_ALGORITHM_REGISTRY"):
            build_more_optimal_algorithm_wall_clock_atom(
                atom_id="bad",
                algorithm_id="nonexistent_algorithm",
                consumer_callsite="src/tac/x.py",
                wall_clock_seconds_canonical=1.0,
                wall_clock_seconds_more_optimal=1.0,
                n_iterations_canonical=1,
                n_iterations_more_optimal=1,
                invariant_check_passed=True,
                invariant_description="x",
            )

    def test_value_error_on_zero_wall_clock(self):
        with pytest.raises(ValueError, match="wall-clock seconds must be > 0"):
            build_more_optimal_algorithm_wall_clock_atom(
                atom_id="bad",
                algorithm_id="fista_beck_teboulle_2009",
                consumer_callsite="x",
                wall_clock_seconds_canonical=0.0,
                wall_clock_seconds_more_optimal=1.0,
                n_iterations_canonical=1,
                n_iterations_more_optimal=1,
                invariant_check_passed=True,
                invariant_description="x",
            )

    def test_value_error_on_negative_wall_clock(self):
        with pytest.raises(ValueError, match="wall-clock seconds must be > 0"):
            build_more_optimal_algorithm_wall_clock_atom(
                atom_id="bad",
                algorithm_id="fista_beck_teboulle_2009",
                consumer_callsite="x",
                wall_clock_seconds_canonical=1.0,
                wall_clock_seconds_more_optimal=-0.5,
                n_iterations_canonical=1,
                n_iterations_more_optimal=1,
                invariant_check_passed=True,
                invariant_description="x",
            )

    def test_metadata_includes_paper_citation_per_catalog_305(self):
        atom = build_more_optimal_algorithm_wall_clock_atom(
            atom_id="test_citation",
            algorithm_id="riemannian_newton_stiefel_edelman_1998",
            consumer_callsite="src/tac/pr101_split_brotli_codec_derivers.py",
            wall_clock_seconds_canonical=1.88,
            wall_clock_seconds_more_optimal=1.0,
            n_iterations_canonical=100,
            n_iterations_more_optimal=10,
            invariant_check_passed=True,
            invariant_description="machine-eps orthogonality",
        )
        assert "Edelman" in atom.metadata["paper_citation"]
        assert (
            atom.metadata["canonical_module_path"]
            == "tac.solvers.riemannian_newton_stiefel"
        )

    def test_extra_metadata_merged_without_clobbering_required_keys(self):
        atom = build_more_optimal_algorithm_wall_clock_atom(
            atom_id="test_extra",
            algorithm_id="fista_beck_teboulle_2009",
            consumer_callsite="src/tac/bit_allocator.py",
            wall_clock_seconds_canonical=1.0,
            wall_clock_seconds_more_optimal=0.5,
            n_iterations_canonical=10,
            n_iterations_more_optimal=5,
            invariant_check_passed=True,
            invariant_description="x",
            extra_metadata={"dispatch_label": "smoke_50ep", "host": "M5_Max"},
        )
        assert atom.metadata["dispatch_label"] == "smoke_50ep"
        assert atom.metadata["host"] == "M5_Max"
        # Required keys remain intact
        assert atom.metadata["algorithm_id"] == "fista_beck_teboulle_2009"
        assert atom.metadata["wall_clock_speedup_multiplier"] == 2.0


class TestAppendMoreOptimalAlgorithmWallClockEvent:
    """Op-routable #5: end-to-end ledger append via canonical helper."""

    def test_append_roundtrip_via_tmp_ledger(self, tmp_path: Path):
        # Use isolated ledger path via patch to avoid touching real .omx/state
        ledger = tmp_path / "atom_ledger.jsonl"
        lock = tmp_path / ".atom_ledger.lock"
        with patch("tac.atom.ledger.ATOM_LEDGER_PATH", ledger), patch(
            "tac.atom.ledger.ATOM_LEDGER_LOCK", lock
        ):
            row = append_more_optimal_algorithm_wall_clock_event(
                atom_id="wall_clock_test_roundtrip",
                algorithm_id="frank_wolfe_kcardinality_jaggi_2013",
                consumer_callsite="src/tac/losses/core.py::sinkhorn_w2_mask_distortion_per_pixel",
                wall_clock_seconds_canonical=1.9,
                wall_clock_seconds_more_optimal=1.0,
                n_iterations_canonical=100,
                n_iterations_more_optimal=42,
                invariant_check_passed=True,
                invariant_description="K=8 sparsity preserved",
            )
            assert ledger.exists()
            content = ledger.read_text()
            lines = [json.loads(line) for line in content.strip().split("\n")]
            assert len(lines) == 1
            # append_atom row schema: {"atom": {...}, "event_type": ..., ...}
            atom_row = lines[0]["atom"]
            assert atom_row["atom_id"] == "wall_clock_test_roundtrip"
            assert atom_row["kind"] == "probe_outcome"
            assert (
                atom_row["metadata"]["algorithm_id"]
                == "frank_wolfe_kcardinality_jaggi_2013"
            )
            assert atom_row["metadata"]["verdict"] == "PROCEED"
            assert isinstance(row, dict)
            # The returned dict mirrors the canonical row shape
            assert row["atom"]["atom_id"] == "wall_clock_test_roundtrip"


class TestShimPackageDocAndExports:
    """Verify the shim package's surface matches its docstring contract."""

    def test_all_includes_canonical_metadata_and_builders(self):
        from tac.solvers import more_optimal_algorithms as mod

        for required in (
            "CANONICAL_ALGORITHM_REGISTRY",
            "AlgorithmCanonicalMetadata",
            "ALGORITHM_FISTA",
            "ALGORITHM_FRANK_WOLFE_KCARD",
            "ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL",
            "build_more_optimal_algorithm_wall_clock_atom",
            "append_more_optimal_algorithm_wall_clock_event",
            "FistaResult",
            "FrankWolfeResult",
            "RiemannianNewtonResult",
            "SinkhornResult",
        ):
            assert required in mod.__all__, f"{required} missing from __all__"

    def test_solvers_package_lists_shim_module(self):
        from tac import solvers

        assert "more_optimal_algorithms" in solvers.__all__
