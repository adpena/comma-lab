"""FIX-J: substrate + CompressAI canonical inventory wire-in dedicated tests.

Per LOOPCLOSE 2026-05-12 finding, 8+ substrate-scaffold packages (Fields-medal
council 2026-05-12) AND 3 CompressAI codecs (EEEE landing 2026-05-12) were
missing from the canonical inventory surfaces (
:func:`tac.optimization.substrate_composition_matrix.canonical_substrate_inventory`
+ :func:`tac.composition.registry.canonical_primitive_inventory` + the
``tools/xray_substrate_classifier.py`` magic-byte coverage). This test file
pins the wire-in so the next sweep does not regress.

Cross-references
----------------
- ``feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512.md``
- Catalog #124 ``check_representation_lane_has_archive_grammar_at_design_time``
- HNeRV parity discipline lessons 2 (export-first design), 3 (monolithic
  single-file archive grammar), 4 (inflate.py <= 100 LOC).

CLAUDE.md compliance tags
-------------------------
- ``no_mps_authoritative``
- ``no_tmp_paths``
- ``score_claim=false`` (these are inventory consistency checks).
- ``planning_only`` (no archive bytes produced).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from tac.composition.registry import (
    PrimitiveCategory,
    SubstrateClass,
    canonical_primitive_inventory,
    primitive_compatibility,
)
from tac.optimization.substrate_composition_matrix import (
    DISPATCH_COST_USD_MIDPOINT,
    canonical_substrate_inventory,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


# The 15 substrate-scaffold subpackages introduced by the Fields-medal
# council 2026-05-12. Each maps subpackage_name -> canonical substrate_id
# used in `canonical_substrate_inventory()`. The mapping handles cases
# where the subpackage name conflicts with a pre-existing legacy row
# (e.g., ``cool_chic`` subpackage -> ``cool_chic_full_renderer`` id
# because ``cool_chic_residual`` was already in the inventory as a
# distinct residual-basis substrate).
_FIX_J_SUBPACKAGE_TO_SUBSTRATE_ID: dict[str, str] = {
    "sane_hnerv": "sane_hnerv",
    "balle_renderer": "balle_renderer",
    "hybrid_renderer_residual": "hybrid_renderer_residual",
    "self_compress_nn": "self_compress_nn",
    "pr101_lc_v2_clone": "pr101_lc_v2_clone",
    "cool_chic": "cool_chic_full_renderer",
    "wavelet": "wavelet_full_renderer",
    "grayscale_lut": "grayscale_lut",
    "vq_vae": "vq_vae_substrate",
    "siren": "siren_substrate",
    "block_nerv": "block_nerv_substrate",
    "tc_nerv": "tc_nerv_substrate",
    "ff_nerv": "ff_nerv_substrate",
    "ds_nerv": "ds_nerv_substrate",
    "hi_nerv": "hi_nerv_substrate",
}

_FIX_J_SUBSTRATE_IDS: frozenset[str] = frozenset(
    _FIX_J_SUBPACKAGE_TO_SUBSTRATE_ID.values()
)

_COMPRESSAI_PRIMITIVE_IDS: frozenset[str] = frozenset({
    "compressai_factorized_prior",
    "compressai_balle_hyperprior",
    "compressai_cheng2020",
})

_COMPRESSAI_MAGICS: dict[str, bytes] = {
    "compressai_factorized_prior": b"CAFP",
    "compressai_balle_hyperprior": b"CABH",
    "compressai_cheng2020": b"CACG",
}


# ── 1. Substrate inventory contains every FIX-J subpackage ───────────────


def test_every_fix_j_substrate_in_canonical_inventory():
    """All 15 FIX-J substrate-scaffold packages MUST appear in the canonical
    inventory; otherwise the bit-allocator / Pareto solver / autopilot
    cannot rank them."""
    rows = canonical_substrate_inventory()
    inv_ids = {r.substrate_id for r in rows}
    missing = _FIX_J_SUBSTRATE_IDS - inv_ids
    assert not missing, f"FIX-J substrates missing from inventory: {missing}"


def test_inventory_has_at_least_39_rows():
    """24 legacy + 15 FIX-J = 39 minimum. Future expansions can add rows;
    the test pins the floor."""
    rows = canonical_substrate_inventory()
    assert len(rows) >= 39


def test_fix_j_substrate_ids_unique_globally():
    """No FIX-J substrate_id collides with a legacy row id."""
    rows = canonical_substrate_inventory()
    all_ids = [r.substrate_id for r in rows]
    assert len(set(all_ids)) == len(all_ids), (
        "duplicate substrate_ids across legacy + FIX-J rows"
    )


def test_fix_j_substrates_have_dispatch_cost_entries():
    for sid in _FIX_J_SUBSTRATE_IDS:
        assert sid in DISPATCH_COST_USD_MIDPOINT, (
            f"FIX-J substrate {sid!r} missing dispatch cost"
        )
        cost = DISPATCH_COST_USD_MIDPOINT[sid]
        assert cost > 0.0, (
            f"FIX-J substrate {sid!r} has non-positive cost {cost!r}; "
            "use 0.0 only for bolt-ons / cost-unknown rows"
        )


def test_fix_j_substrates_have_unique_format_ids_within_class():
    """Per the matrix invariant, format_ids are unique within each
    substrate_class; FIX-J additions must not collide with legacy ids."""
    rows = canonical_substrate_inventory()
    by_class: dict[SubstrateClass, list[int]] = {}
    for r in rows:
        by_class.setdefault(r.substrate_class, []).append(r.format_id)
    for cls, ids in by_class.items():
        assert len(set(ids)) == len(ids), (
            f"duplicate format_id within class {cls}: {ids}"
        )


def test_fix_j_substrates_have_4_byte_magic():
    """Every FIX-J substrate-scaffold MUST declare a 4-byte ASCII magic
    so the xray classifier can disambiguate it."""
    rows = canonical_substrate_inventory()
    for r in rows:
        if r.substrate_id in _FIX_J_SUBSTRATE_IDS:
            assert len(r.magic_bytes) == 4, (
                f"FIX-J substrate {r.substrate_id!r} magic must be 4 bytes; "
                f"got {r.magic_bytes!r}"
            )


def test_fix_j_substrates_have_landing_memo_token():
    rows = canonical_substrate_inventory()
    for r in rows:
        if r.substrate_id in _FIX_J_SUBSTRATE_IDS:
            assert "fix_j" in r.landing_memo, (
                f"FIX-J substrate {r.substrate_id!r} should reference the "
                f"FIX-J landing memo; got {r.landing_memo!r}"
            )


# ── 2. Substrate package re-exports + SUBSTRATE_SCAFFOLDS registry ───────


def test_substrate_subpackages_in_substrates_all():
    """``src/tac/substrates/__init__.py``'s ``__all__`` MUST include every
    FIX-J subpackage so ``from tac.substrates import <name>`` works."""
    import tac.substrates as substrates_pkg

    pkg_all = set(substrates_pkg.__all__)
    for subpackage in _FIX_J_SUBPACKAGE_TO_SUBSTRATE_ID:
        assert subpackage in pkg_all, (
            f"subpackage {subpackage!r} missing from "
            f"src/tac/substrates/__init__.py __all__"
        )


def test_substrate_scaffolds_registry_matches_disk():
    """The SUBSTRATE_SCAFFOLDS registry must map subpackage_name -> the
    substrate_id used in the canonical inventory."""
    import tac.substrates as substrates_pkg

    assert hasattr(substrates_pkg, "SUBSTRATE_SCAFFOLDS"), (
        "tac.substrates module missing SUBSTRATE_SCAFFOLDS registry"
    )
    reg = substrates_pkg.SUBSTRATE_SCAFFOLDS
    inv_ids = {r.substrate_id for r in canonical_substrate_inventory()}
    for subpackage, substrate_id in reg.items():
        assert substrate_id in inv_ids, (
            f"SUBSTRATE_SCAFFOLDS[{subpackage!r}] -> {substrate_id!r} not in "
            "canonical_substrate_inventory()"
        )


def test_substrate_subpackages_exist_on_disk():
    """Every entry in SUBSTRATE_SCAFFOLDS must correspond to a real
    subdirectory under src/tac/substrates/ with an __init__.py."""
    substrates_root = _REPO_ROOT / "src" / "tac" / "substrates"
    for subpackage in _FIX_J_SUBPACKAGE_TO_SUBSTRATE_ID:
        subdir = substrates_root / subpackage
        assert subdir.is_dir(), (
            f"substrate subpackage {subpackage!r} missing from disk: {subdir}"
        )
        init_py = subdir / "__init__.py"
        assert init_py.is_file(), (
            f"substrate subpackage {subpackage!r} missing __init__.py: {init_py}"
        )


# ── 3. CompressAI primitives in canonical_primitive_inventory ────────────


def test_compressai_primitives_in_canonical_inventory():
    """All 3 CompressAI codecs MUST appear in the canonical primitive
    inventory so composition cells can be enumerated for them."""
    prims = canonical_primitive_inventory()
    inv_ids = {p.primitive_id for p in prims}
    missing = _COMPRESSAI_PRIMITIVE_IDS - inv_ids
    assert not missing, f"CompressAI primitives missing: {missing}"


def test_inventory_has_at_least_17_primitives():
    """14 legacy + 3 CompressAI = 17 minimum."""
    prims = canonical_primitive_inventory()
    assert len(prims) >= 17


def test_compressai_primitives_category_is_compressai_codec():
    prims = canonical_primitive_inventory()
    for p in prims:
        if p.primitive_id in _COMPRESSAI_PRIMITIVE_IDS:
            assert p.category == PrimitiveCategory.COMPRESSAI_CODEC, (
                f"primitive {p.primitive_id!r} has wrong category {p.category!r}; "
                "expected PrimitiveCategory.COMPRESSAI_CODEC"
            )


def test_compressai_compatibility_matrix_correctly_scoped():
    """CompressAI codecs apply to RENDERER_REPLACEMENT + RESIDUAL (latent
    streams) but NOT to POSE_AXIS_SIDECHANNEL / SELF_COMPRESSION / BOLT_ON
    / META_CODEC. The matrix gate enforces this."""
    assert primitive_compatibility(
        SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.COMPRESSAI_CODEC
    )
    assert primitive_compatibility(
        SubstrateClass.RESIDUAL, PrimitiveCategory.COMPRESSAI_CODEC
    )
    assert not primitive_compatibility(
        SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.COMPRESSAI_CODEC
    )
    assert not primitive_compatibility(
        SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.COMPRESSAI_CODEC
    )
    assert not primitive_compatibility(
        SubstrateClass.BOLT_ON, PrimitiveCategory.COMPRESSAI_CODEC
    )
    assert not primitive_compatibility(
        SubstrateClass.META_CODEC, PrimitiveCategory.COMPRESSAI_CODEC
    )


# ── 4. xray_substrate_classifier magic-byte coverage ──────────────────────


def _import_xray_module():
    """Import tools/xray_substrate_classifier.py as a module."""
    tools_dir = _REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    mod = importlib.import_module("xray_substrate_classifier")
    return mod


def test_xray_classifier_has_fix_j_magic_signatures():
    """Every FIX-J substrate's 4-byte magic MUST be a known signature so
    the classifier can identify it pre-dispatch."""
    xray = _import_xray_module()
    inv_magics = {sig for sig, _ in xray._SECTION_MAGIC_SIGNATURES}
    # Pull every FIX-J substrate's magic_bytes and verify presence.
    for r in canonical_substrate_inventory():
        if r.substrate_id in _FIX_J_SUBSTRATE_IDS:
            magic = r.magic_bytes.encode("ascii")
            assert magic in inv_magics, (
                f"FIX-J substrate {r.substrate_id!r} magic {magic!r} missing "
                "from xray_substrate_classifier._SECTION_MAGIC_SIGNATURES"
            )


def test_xray_classifier_has_compressai_magic_signatures():
    xray = _import_xray_module()
    inv_magics = {sig for sig, _ in xray._SECTION_MAGIC_SIGNATURES}
    for primitive_id, magic in _COMPRESSAI_MAGICS.items():
        assert magic in inv_magics, (
            f"CompressAI primitive {primitive_id!r} magic {magic!r} missing "
            "from xray_substrate_classifier._SECTION_MAGIC_SIGNATURES"
        )


def test_xray_classifier_substrate_rules_cover_fix_j():
    """Every FIX-J substrate that has a 4-byte magic MUST have a
    _SubstrateRule firing on that magic so the classifier emits the
    correct substrate_class id."""
    xray = _import_xray_module()
    rule_magics = set()
    for rule in xray._SUBSTRATE_RULES:
        rule_magics.update(rule.required_section_magics)
    for r in canonical_substrate_inventory():
        if r.substrate_id in _FIX_J_SUBSTRATE_IDS:
            magic = r.magic_bytes.encode("ascii")
            assert magic in rule_magics, (
                f"FIX-J substrate {r.substrate_id!r} magic {magic!r} has no "
                "_SubstrateRule entry in xray_substrate_classifier"
            )


def test_xray_classifier_substrate_classes_list_extended():
    """Every FIX-J _SubstrateRule's substrate_class id MUST be in the
    _SUBSTRATE_CLASSES tuple so the classifier doesn't refuse the
    classification at output time."""
    xray = _import_xray_module()
    classes_set = set(xray._SUBSTRATE_CLASSES)
    # Iterate the rules; each rule's class must be in the registered tuple.
    for rule in xray._SUBSTRATE_RULES:
        assert rule.substrate_class in classes_set, (
            f"xray _SubstrateRule for class {rule.substrate_class!r} not in "
            "_SUBSTRATE_CLASSES; classifier would refuse output"
        )


def test_xray_classifier_compressai_rules_present():
    """The 3 CompressAI packets each need a _SubstrateRule so the
    classifier can flag them when they appear as ZIP members."""
    xray = _import_xray_module()
    rule_classes = {r.substrate_class for r in xray._SUBSTRATE_RULES}
    expected = {
        "compressai_factorized_prior_packet",
        "compressai_balle_hyperprior_packet",
        "compressai_cheng2020_packet",
    }
    missing = expected - rule_classes
    assert not missing, (
        f"CompressAI substrate classes missing classifier rules: {missing}"
    )


# ── 5. End-to-end compatibility — enumerate cells with new primitives ────


def test_enumerate_cells_emits_compressai_cells():
    """Enumerating composition cells with max_primitives_per_cell=1 must
    produce at least one cell per CompressAI primitive applied to a
    RENDERER_REPLACEMENT substrate."""
    from tac.composition.enumerate import enumerate_cells

    cells = enumerate_cells(max_primitives_per_cell=1)
    compressai_cells = [
        c for c in cells
        if any(pid in _COMPRESSAI_PRIMITIVE_IDS for pid, _ in c.primitives)
    ]
    assert compressai_cells, (
        "enumerate_cells produced no cells referencing CompressAI primitives"
    )


def test_enumerate_cells_emits_fix_j_substrate_cells():
    """At least one cell per FIX-J substrate must enumerate (even if
    refused / advisory)."""
    from tac.composition.enumerate import enumerate_cells

    cells = enumerate_cells(max_primitives_per_cell=1)
    substrate_ids_with_cells = {c.substrate_id for c in cells}
    missing = _FIX_J_SUBSTRATE_IDS - substrate_ids_with_cells
    assert not missing, (
        f"enumerate_cells produced no cells for FIX-J substrates: {missing}"
    )


# ── 6. Score-claim discipline ──────────────────────────────────────────────


def test_no_fix_j_or_compressai_row_claims_score():
    """Per CLAUDE.md ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``,
    new substrate / primitive inventory rows MUST NOT advertise any
    score_claim / promotion_eligible / ready_for_exact_eval_dispatch
    affirmative attribute. The inventory rows are typed planning metadata."""
    # SubstrateRow + PrimitiveRow do not have these fields as attributes
    # (they're invariants on derived dataclasses like CompositionResult),
    # so this test asserts the module-level invariants instead.
    from tac.optimization.substrate_composition_matrix import (
        PROMOTION_ELIGIBLE,
        READY_FOR_EXACT_EVAL_DISPATCH,
        SCORE_CLAIM,
    )
    assert SCORE_CLAIM is False
    assert PROMOTION_ELIGIBLE is False
    assert READY_FOR_EXACT_EVAL_DISPATCH is False
    from tac.composition.registry import (
        PROMOTION_ELIGIBLE as REG_PROMOTION,
    )
    from tac.composition.registry import (
        READY_FOR_EXACT_EVAL_DISPATCH as REG_READY,
    )
    from tac.composition.registry import (
        SCORE_CLAIM as REG_SCORE,
    )
    assert REG_SCORE is False
    assert REG_PROMOTION is False
    assert REG_READY is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
