# SPDX-License-Identifier: MIT
"""Tests for the canonical xray primitive registry."""

from __future__ import annotations

import pytest

from tac.xray.base import CANONICAL_WIRE_IN_HOOKS
from tac.xray.registry import (
    XRAY_REGISTRY_SCHEMA_VERSION,
    XRayPrimitiveSpec,
    canonical_xray_primitive_inventory,
    get_xray_primitive_spec,
    serialize_xray_inventory,
    specs_by_category,
    specs_by_hook,
)


def test_schema_version_pinned():
    assert XRAY_REGISTRY_SCHEMA_VERSION == "tac_xray_registry_v1"


def test_inventory_returns_fourteen_primitives():
    specs = canonical_xray_primitive_inventory()
    assert len(specs) == 14


def test_inventory_names_unique():
    specs = canonical_xray_primitive_inventory()
    names = [s.primitive_name for s in specs]
    assert len(names) == len(set(names))


def test_inventory_all_have_non_empty_hooks():
    for spec in canonical_xray_primitive_inventory():
        assert len(spec.wire_in_hooks) >= 1, (
            f"primitive {spec.primitive_name} has no wire-in hooks; "
            "orphan-work failure mode"
        )


def test_inventory_all_hooks_canonical():
    for spec in canonical_xray_primitive_inventory():
        for hook in spec.wire_in_hooks:
            assert hook in CANONICAL_WIRE_IN_HOOKS


def test_inventory_required_primitives_present():
    """All 13 F-primitives from the wire-in directive are registered."""
    names = {s.primitive_name for s in canonical_xray_primitive_inventory()}
    required = {
        "mdl_scorer_conditional",  # F1
        "shannon_vector_r_d",  # F2
        "bilinear_resize_nullspace",  # F3
        "score_lipschitz",  # F4
        "vq_codebook_coverage",  # F5
        "wavelet_hf_energy",  # F6
        "segnet_margin_polytope",  # F7
        "posenet_se3_lie_algebra",  # F8
        "per_pair_score_decomposition",  # F9
        "pairset_component_marginal",  # F14
        "yuv6_sublattice_geometry",  # F10
        "unified_action_principle",  # F11
        "predictive_coding_hierarchy",  # F12a
        "foveation_ego_motion",  # F12b
    }
    assert required.issubset(names)


def test_spec_rejects_empty_hooks():
    with pytest.raises(ValueError, match="zero wire-in"):
        XRayPrimitiveSpec(
            primitive_name="x",
            canonical_module="tac.xray.foo",
            canonical_symbol="Foo",
            category="information-geometric",
            description="...",
            primary_finding="...",
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(),
        )


def test_spec_rejects_empty_name():
    with pytest.raises(ValueError, match="primitive_name must be non-empty"):
        XRayPrimitiveSpec(
            primitive_name="",
            canonical_module="tac.xray.foo",
            canonical_symbol="Foo",
            category="information-geometric",
            description="...",
            primary_finding="...",
            evidence_grade="mathematical-derivation",
            wire_in_hooks=("sensitivity_map",),
        )


def test_spec_rejects_unknown_hook():
    with pytest.raises(ValueError, match="unknown hook"):
        XRayPrimitiveSpec(
            primitive_name="x",
            canonical_module="tac.xray.foo",
            canonical_symbol="Foo",
            category="information-geometric",
            description="...",
            primary_finding="...",
            evidence_grade="mathematical-derivation",
            wire_in_hooks=("bogus_hook",),  # type: ignore
        )


def test_get_spec_returns_matching_row():
    spec = get_xray_primitive_spec("mdl_scorer_conditional")
    assert spec.primitive_name == "mdl_scorer_conditional"
    assert spec.canonical_module == "tac.xray.mdl_scorer_conditional"


def test_get_spec_raises_on_unknown():
    with pytest.raises(ValueError, match="unknown xray primitive"):
        get_xray_primitive_spec("not_a_real_primitive_xyz")


def test_specs_by_hook_returns_only_matching():
    specs = specs_by_hook("sensitivity_map")
    for spec in specs:
        assert "sensitivity_map" in spec.wire_in_hooks


def test_specs_by_hook_raises_on_unknown():
    with pytest.raises(ValueError, match="unknown hook"):
        specs_by_hook("bogus_hook")  # type: ignore


def test_specs_by_category_returns_only_matching():
    info_geo = specs_by_category("information-geometric")
    for spec in info_geo:
        assert spec.category == "information-geometric"
    # We declared 3 information-geometric primitives (F1, F2, F4).
    assert len(info_geo) == 3


def test_specs_by_category_codec_axis_count():
    codec_axis = specs_by_category("codec-axis")
    # F3, F5, F6, F10 = 4
    assert len(codec_axis) == 4


def test_specs_by_category_scorer_internal_count():
    scorer_internal = specs_by_category("scorer-internal")
    # F7, F8, F9, F14 = 4
    assert len(scorer_internal) == 4


def test_specs_by_category_unified_action_count():
    unified = specs_by_category("unified-action")
    assert len(unified) == 1
    assert unified[0].primitive_name == "unified_action_principle"


def test_specs_by_category_codec_primitive_count():
    codec_prim = specs_by_category("codec-primitive")
    # F12a + F12b = 2
    assert len(codec_prim) == 2


def test_serialize_inventory_returns_lists_not_tuples():
    """JSON-friendly: tuples become lists."""
    out = serialize_xray_inventory()
    assert len(out) == 14
    for d in out:
        assert isinstance(d["wire_in_hooks"], list)
        assert isinstance(d["composes_with"], list)


def test_serialize_inventory_round_trip_names_match():
    out = serialize_xray_inventory()
    names_in_serialized = {d["primitive_name"] for d in out}
    names_in_specs = {
        s.primitive_name for s in canonical_xray_primitive_inventory()
    }
    assert names_in_serialized == names_in_specs


def test_every_canonical_module_path_resolves():
    """The canonical_module of each primitive must be importable
    (or at least the parent package). This is a soft test: we check
    that the leading prefix is ``tac.xray.`` and the symbol name is
    valid Python identifier."""
    for spec in canonical_xray_primitive_inventory():
        assert spec.canonical_module.startswith("tac.xray."), (
            f"primitive {spec.primitive_name} module "
            f"{spec.canonical_module!r} doesn't start with tac.xray."
        )
        assert spec.canonical_symbol.isidentifier(), (
            f"primitive {spec.primitive_name} symbol "
            f"{spec.canonical_symbol!r} is not a valid identifier"
        )


def test_categories_are_one_of_canonical_set():
    canonical_cats = {
        "information-geometric",
        "scorer-internal",
        "codec-axis",
        "unified-action",
        "codec-primitive",
    }
    for spec in canonical_xray_primitive_inventory():
        assert spec.category in canonical_cats


def test_inventory_composes_with_refs_are_valid():
    """Every primitive in ``composes_with`` must itself be in the inventory."""
    names = {s.primitive_name for s in canonical_xray_primitive_inventory()}
    for spec in canonical_xray_primitive_inventory():
        for compose_target in spec.composes_with:
            assert compose_target in names, (
                f"primitive {spec.primitive_name!r} composes_with "
                f"{compose_target!r} which is not in the inventory"
            )


def test_every_hook_has_at_least_one_primitive():
    """All 6 canonical hooks must have at least one primitive engaging them
    — otherwise the hook is orphan-consumer."""
    for hook in CANONICAL_WIRE_IN_HOOKS:
        specs = specs_by_hook(hook)
        assert len(specs) >= 1, (
            f"hook {hook!r} has zero xray-primitive consumers; "
            "this is an orphan-hook failure mode"
        )


def test_inventory_mdl_compose_chain():
    """Sanity check the F1+F2 compose chain declared in mdl_scorer_conditional."""
    mdl_spec = get_xray_primitive_spec("mdl_scorer_conditional")
    assert "shannon_vector_r_d" in mdl_spec.composes_with
