# SPDX-License-Identifier: MIT
"""Integration tests: xray primitives wire into existing solver-stack surfaces.

These tests verify the wire-in CONTRACT, not the consumer's internal
behavior. Each test checks that the bundle returned by
:func:`wire_in_for_hook` carries enough information for the named
downstream surface to use it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.xray import (
    aggregate_hook_evidence_grade,
    canonical_xray_primitive_inventory,
    discover_primitives_by_hook,
    wire_in_for_hook,
)


def test_sensitivity_map_hook_yields_at_least_eight_primitives():
    """Per the registry, ``sensitivity_map`` is wired into 8+ primitives."""
    bundle = wire_in_for_hook("sensitivity_map", targets={})
    assert bundle.n_primitives >= 8
    assert bundle.hook == "sensitivity_map"


def test_pareto_constraint_hook_yields_at_least_three():
    bundle = wire_in_for_hook("pareto_constraint", targets={})
    assert bundle.n_primitives >= 3


def test_bit_allocator_hook_yields_at_least_seven():
    bundle = wire_in_for_hook("bit_allocator", targets={})
    assert bundle.n_primitives >= 7


def test_cathedral_autopilot_hook_yields_at_least_three():
    bundle = wire_in_for_hook("cathedral_autopilot", targets={})
    assert bundle.n_primitives >= 3


def test_continual_learning_hook_engaged_by_mdl():
    """At minimum F1 (mdl_scorer_conditional) engages continual_learning."""
    discovered = discover_primitives_by_hook()
    assert "mdl_scorer_conditional" in discovered["continual_learning"]


def test_probe_disambiguator_hook_majority_engaged():
    """Most primitives engage probe_disambiguator (the 'extra coverage' hook)."""
    discovered = discover_primitives_by_hook()
    # 9+ primitives engage probe_disambiguator.
    assert len(discovered["probe_disambiguator"]) >= 9


def test_pareto_bundle_results_carry_confidence_bands():
    """Pareto consumer needs confidence_band on every result for trust-region
    construction."""
    bundle = wire_in_for_hook(
        "pareto_constraint",
        targets={
            "shannon_vector_r_d": {
                "target": None,
                "d_seg_target": 0.067,
                "d_pose_target": 0.018,
            },
            "score_lipschitz": {
                "target": b"\x00" * 16,
            },
        },
    )
    for r in bundle.results:
        assert r.confidence_band is not None, (
            f"{r.primitive_name} missing confidence_band; Pareto consumer "
            "requires it for trust-region construction"
        )


def test_sensitivity_map_bundle_results_carry_primitive_value():
    """Sensitivity-map consumer needs primitive_value (tensor / dict) on
    every result."""
    bundle = wire_in_for_hook(
        "sensitivity_map",
        targets={
            "shannon_vector_r_d": {"target": None},
            "vq_codebook_coverage": {
                "target": torch.randn(4, 8),
                "codebook": torch.randn(4, 8),
            },
        },
    )
    for r in bundle.results:
        assert r.primitive_value is not None


def test_cathedral_autopilot_bundle_engaged_by_per_pair_decomposition():
    """The cathedral autopilot's top-K-pair selector is driven by F9."""
    bundle = wire_in_for_hook(
        "cathedral_autopilot",
        targets={
            "per_pair_score_decomposition": {
                "target": torch.tensor([[0.1, 0.01], [0.5, 0.02], [0.05, 0.005]]),
                "top_k": 2,
            },
        },
    )
    names = {r.primitive_name for r in bundle.results}
    assert "per_pair_score_decomposition" in names


def test_continual_learning_bundle_engaged_by_mdl_and_pairset_component():
    """F1 plus the pairset component-marginal primitive engage continual learning."""
    bundle = wire_in_for_hook("continual_learning", targets={})
    assert bundle.n_primitives == 2
    spec_names = {
        "mdl_scorer_conditional",
        "pairset_component_marginal",
    }
    discovered = discover_primitives_by_hook()
    assert set(discovered["continual_learning"]) == spec_names


def test_aggregate_evidence_grade_consistent_for_strong_bundle():
    """A bundle of only mathematical-derivation results aggregates to
    mathematical-derivation."""
    bundle = wire_in_for_hook(
        "probe_disambiguator",
        targets={
            "shannon_vector_r_d": {"target": None},
            "score_lipschitz": {"target": b"\x00" * 32},
        },
    )
    grade = aggregate_hook_evidence_grade(bundle)
    # Both are mathematical-derivation OR first-principles-bound; the
    # weakest of those is first-principles-bound (council-deliberation
    # is the weakest of the entire taxonomy but not relevant here).
    assert grade in (
        "mathematical-derivation",
        "first-principles-bound",
    )


def test_wire_in_skipped_lifecycle_recorded():
    """When targets are missing, primitives are skipped with a reason."""
    bundle = wire_in_for_hook(
        "sensitivity_map",
        targets={
            "shannon_vector_r_d": {"target": None},
        },
    )
    # Some primitives provided, others skipped.
    skipped_names = {n for n, _ in bundle.skipped_primitives}
    discovered = discover_primitives_by_hook()
    for name in discovered["sensitivity_map"]:
        if name == "shannon_vector_r_d":
            continue
        assert name in skipped_names, (
            f"primitive {name!r} not in skipped_primitives — wire-in lifecycle "
            "log incomplete"
        )


def test_bundle_results_serializable_to_dict():
    """Solver-stack consumers may JSON-serialize bundle results; verify
    each result can be converted to a dict."""
    import dataclasses

    bundle = wire_in_for_hook(
        "sensitivity_map",
        targets={
            "shannon_vector_r_d": {"target": None},
        },
    )
    for r in bundle.results:
        d = dataclasses.asdict(r)
        assert "primitive_name" in d
        assert "wire_in_hooks_engaged" in d
        assert "evidence_grade" in d


def test_full_six_hook_coverage_via_discover():
    """All 6 canonical hooks are discoverable from the inventory."""
    discovered = discover_primitives_by_hook()
    for hook in (
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
        "cathedral_autopilot",
        "continual_learning",
        "probe_disambiguator",
    ):
        assert hook in discovered
        assert len(discovered[hook]) >= 1


def test_inventory_matches_fourteen_primitives_after_pairset_component_signal():
    """Lock the 14-primitive inventory cardinality after pairset component wiring so future
    regressions surface immediately."""
    assert len(canonical_xray_primitive_inventory()) == 14


def test_canonical_a1_archive_mdl_density_close_to_one_via_wire_in():
    """Integration: A1 archive (sha 87ec7ca5...) should produce
    mdl_density ~0.999 via the wire-in surface."""
    a1 = Path("submissions/a1/archive.zip")
    if not a1.exists():
        pytest.skip("A1 archive not present in this checkout")
    bundle = wire_in_for_hook(
        "continual_learning",
        targets={"mdl_scorer_conditional": {"target": a1}},
    )
    assert len(bundle.results) == 1
    result = bundle.results[0]
    assert result.archive_sha256 == (
        "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
    )
    assert result.primitive_value.mdl_density > 0.8
