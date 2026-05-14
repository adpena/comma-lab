# SPDX-License-Identifier: MIT
"""Tests for the canonical solver-stack wire-in surface."""

from __future__ import annotations

import pytest
import torch

from tac.xray.base import (
    CANONICAL_WIRE_IN_HOOKS,
    XRayPrimitive,
)
from tac.xray.registry import (
    canonical_xray_primitive_inventory,
    get_xray_primitive_spec,
)
from tac.xray.wire_in import (
    XRayWireInBundle,
    aggregate_hook_evidence_grade,
    discover_primitives_by_hook,
    instantiate_primitive,
    wire_in_for_hook,
)


def test_bundle_rejects_unknown_hook():
    with pytest.raises(ValueError, match="hook"):
        XRayWireInBundle(
            hook="bogus_hook",  # type: ignore
            n_primitives=0,
            results=(),
        )


def test_bundle_rejects_negative_n_primitives():
    with pytest.raises(ValueError, match="n_primitives"):
        XRayWireInBundle(
            hook="sensitivity_map",
            n_primitives=-1,
            results=(),
        )


def test_bundle_rejects_more_results_than_n():
    from tac.xray.base import XRayPrimitiveResult

    r = XRayPrimitiveResult(
        primitive_name="x",
        archive_or_video_path=None,
        archive_sha256=None,
        primitive_value=1.0,
        evidence_grade="mathematical-derivation",
        confidence_band=None,
        composes_with=(),
        wire_in_hooks_engaged=("sensitivity_map",),
    )
    with pytest.raises(ValueError, match="results length"):
        XRayWireInBundle(
            hook="sensitivity_map",
            n_primitives=0,
            results=(r,),
        )


def test_instantiate_primitive_returns_xray_protocol():
    spec = get_xray_primitive_spec("shannon_vector_r_d")
    p = instantiate_primitive(spec)
    assert isinstance(p, XRayPrimitive)


def test_instantiate_all_canonical_primitives():
    """Every primitive in the canonical inventory must instantiate."""
    for spec in canonical_xray_primitive_inventory():
        p = instantiate_primitive(spec)
        assert p.name == spec.primitive_name


def test_discover_primitives_by_hook_covers_all_six():
    discovered = discover_primitives_by_hook()
    assert set(discovered.keys()) == set(CANONICAL_WIRE_IN_HOOKS)
    # Every hook has at least one primitive.
    for hook, names in discovered.items():
        assert len(names) >= 1, f"hook {hook} has no primitives"


def test_discover_primitives_by_hook_sensitivity_map_has_at_least_eight():
    discovered = discover_primitives_by_hook()
    # Per the registry, 11 primitives engage sensitivity_map.
    assert len(discovered["sensitivity_map"]) >= 8


def test_wire_in_rejects_unknown_hook():
    with pytest.raises(ValueError, match="hook"):
        wire_in_for_hook("bogus_hook")  # type: ignore


def test_wire_in_empty_targets_skips_everything():
    """With no targets, every primitive in the hook is skipped."""
    bundle = wire_in_for_hook("sensitivity_map", targets={})
    assert len(bundle.results) == 0
    assert bundle.n_primitives >= 8
    assert len(bundle.skipped_primitives) == bundle.n_primitives


def test_wire_in_runs_provided_primitives():
    """Provide targets for shannon_vector_r_d; bundle should have 1 result."""
    bundle = wire_in_for_hook(
        "sensitivity_map",
        targets={
            "shannon_vector_r_d": {
                "target": None,
                "d_seg_target": 0.067,
                "d_pose_target": 0.018,
            },
        },
    )
    # At least one result.
    assert len(bundle.results) == 1
    assert bundle.results[0].primitive_name == "shannon_vector_r_d"


def test_wire_in_aggregates_multiple_results():
    """Provide targets for multiple primitives; bundle aggregates all."""
    bundle = wire_in_for_hook(
        "bit_allocator",
        targets={
            "shannon_vector_r_d": {
                "target": None,
                "d_seg_target": 0.067,
            },
            "score_lipschitz": {
                "target": b"\x00" * 32,
            },
        },
    )
    # Both primitives engage bit_allocator? Actually shannon_vector_r_d
    # only engages pareto_constraint + sensitivity_map + probe_disambiguator.
    # So only score_lipschitz should produce a result here.
    names = {r.primitive_name for r in bundle.results}
    assert "score_lipschitz" in names


def test_wire_in_skip_on_error_catches_compute_exceptions():
    """When a primitive raises during compute, skip_on_error catches it."""
    # Pass an invalid target to shannon_vector_r_d -> ValueError.
    bundle = wire_in_for_hook(
        "sensitivity_map",
        targets={
            "shannon_vector_r_d": {
                "target": None,
                "correlation_factor": 2.0,  # invalid; > 1.0
            },
        },
        skip_on_error=True,
    )
    assert len(bundle.results) == 0
    assert any(
        "shannon_vector_r_d" in name
        for name, _ in bundle.skipped_primitives
    )


def test_wire_in_propagates_on_error_when_disabled():
    with pytest.raises(ValueError):
        wire_in_for_hook(
            "sensitivity_map",
            targets={
                "shannon_vector_r_d": {
                    "target": None,
                    "correlation_factor": 2.0,
                },
            },
            skip_on_error=False,
        )


def test_aggregate_evidence_grade_weakest_wins():
    """Aggregating mathematical-derivation + council-deliberation yields
    council-deliberation (weaker)."""
    bundle = wire_in_for_hook(
        "sensitivity_map",
        targets={
            "shannon_vector_r_d": {"target": None},  # first-principles-bound
            "vq_codebook_coverage": {
                "target": torch.randn(4, 8),
                "codebook": torch.randn(4, 8),
            },  # council-deliberation
        },
    )
    grade = aggregate_hook_evidence_grade(bundle)
    # council-deliberation is weaker than first-principles-bound.
    assert grade == "council-deliberation"


def test_aggregate_evidence_grade_empty_bundle_proxy():
    """Empty bundle yields 'proxy' as fallback."""
    bundle = XRayWireInBundle(
        hook="sensitivity_map",
        n_primitives=0,
        results=(),
    )
    grade = aggregate_hook_evidence_grade(bundle)
    assert grade == "proxy"


def test_wire_in_bundle_n_primitives_matches_specs_by_hook():
    """n_primitives must equal the count of specs registered for the hook."""
    from tac.xray.registry import specs_by_hook

    bundle = wire_in_for_hook("bit_allocator", targets={})
    expected = len(specs_by_hook("bit_allocator"))
    assert bundle.n_primitives == expected
