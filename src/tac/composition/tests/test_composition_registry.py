"""Tests for :mod:`tac.composition.registry` + :mod:`tac.composition.enumerate`.

Per operator directive 2026-05-12 ("stacking and composition on
everything"), this test file pins:

1. The 14-primitive canonical inventory shape + integrity.
2. The substrate x primitive compatibility matrix (every entry checked).
3. The pipeline-ordering / mutually-exclusive validator.
4. The enumerator's compatibility-matrix gate.
5. The enumerator's mutual-exclusion enforcement (one sign-encoding only,
   one schema-elision only).
6. The enumerator's ordered-pipeline within-category ordering.
7. The enumerator's cross-category order_index monotonicity.
8. The enumerator's bare-substrate baseline emission.
9. The score-claim discipline (every cell has score_claim=False, etc.).
10. The autopilot ranking-input shape (CandidateRow-compatible).
11. The serialization round-trip preserves planning-only invariants.
12. The non-HNeRV-substrate soft blocker for PR101 GOLD primitives.
13. Stable sort order across runs.
14. Edge cases: empty inventory raises; max_primitives_per_cell=0;
    unknown primitive_id raises.

Per CLAUDE.md "Adversarial council review of design decisions" — every
guard surfaces an explicit assertion error on misuse.
"""

from __future__ import annotations

import pytest

from tac.composition.enumerate import (
    ENUMERATION_SCHEMA,
    autopilot_ranking_input,
    enumerate_cells,
    enumerate_substrate_incompatible_cells,
    serialize_enumeration,
)
from tac.composition.registry import (
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCHEMA_VERSION,
    SCORE_CLAIM,
    PrimitiveCategory,
    RefusedReason,
    SemanticConstraint,
    SubstrateClass,
    canonical_primitive_inventory,
    canonical_substrate_inventory,
    classify_pipeline_violation,
    compute_semantic_warning,
    detect_dependency_violation,
    detect_substrate_semantic_incompatibility,
    primitive_compatibility,
    serialize_primitive_inventory,
    validate_pipeline_ordering,
)

# ── 1. Primitive inventory shape ──────────────────────────────────────────


def test_primitive_inventory_has_14_rows():
    prims = canonical_primitive_inventory()
    assert len(prims) == 14, f"expected 14 primitives, got {len(prims)}"


def test_primitive_inventory_ids_unique():
    prims = canonical_primitive_inventory()
    ids = [p.primitive_id for p in prims]
    assert len(set(ids)) == len(ids), f"duplicate primitive_ids: {ids}"


def test_primitive_inventory_categories():
    prims = canonical_primitive_inventory()
    by_cat: dict[PrimitiveCategory, list[str]] = {}
    for p in prims:
        by_cat.setdefault(p.category, []).append(p.primitive_id)
    # PR101 GOLD trio:
    assert len(by_cat[PrimitiveCategory.PR101_GOLD_STORAGE]) == 3
    # Sign-encoding x5:
    assert len(by_cat[PrimitiveCategory.SIGN_ENCODING]) == 5
    # Schema-elision x3:
    assert len(by_cat[PrimitiveCategory.SCHEMA_ELISION]) == 3
    # Magic-codec dense-streams x1:
    assert len(by_cat[PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS]) == 1
    # Brotli x1, LZMA x1:
    assert len(by_cat[PrimitiveCategory.BROTLI]) == 1
    assert len(by_cat[PrimitiveCategory.LZMA]) == 1


def test_primitive_score_claim_invariants_planning_only():
    # PrimitiveRow doesn't directly carry score_claim; the discipline is
    # enforced at CompositionCell + module-level constants.
    assert SCORE_CLAIM is False
    assert PROMOTION_ELIGIBLE is False
    assert READY_FOR_EXACT_EVAL_DISPATCH is False


def test_pr101_gold_within_category_order():
    prims = canonical_primitive_inventory()
    pr101 = [
        p for p in prims if p.category == PrimitiveCategory.PR101_GOLD_STORAGE
    ]
    # All three must declare the SAME within_category_order tuple.
    orders = {p.within_category_order for p in pr101}
    assert len(orders) == 1, f"PR101 GOLD within_category_order differs: {orders}"
    canonical = pr101[0].within_category_order
    assert canonical == (
        "pr101_decoder_storage_order",
        "pr101_conv4_storage_perms",
        "pr101_decoder_byte_maps",
    )
    # And their order_index must be monotone.
    pr101.sort(key=lambda x: x.order_index)
    assert [p.primitive_id for p in pr101] == list(canonical)


def test_sign_encoding_five_strategies():
    prims = canonical_primitive_inventory()
    sign = {
        p.primitive_id for p in prims
        if p.category == PrimitiveCategory.SIGN_ENCODING
    }
    assert sign == {
        "sign_encoding_negzig",
        "sign_encoding_zig",
        "sign_encoding_twos",
        "sign_encoding_off",
        "sign_encoding_raw_uint8",
    }


# ── 2. Compatibility matrix ───────────────────────────────────────────────


def test_compatibility_matrix_pr101_gold_renderer_replacement_only():
    # PR101 GOLD applies ONLY to RENDERER_REPLACEMENT.
    for sub_class in SubstrateClass:
        ok = primitive_compatibility(sub_class, PrimitiveCategory.PR101_GOLD_STORAGE)
        if sub_class == SubstrateClass.RENDERER_REPLACEMENT:
            assert ok is True, f"PR101 GOLD must apply to {sub_class}"
        else:
            assert ok is False, f"PR101 GOLD must NOT apply to {sub_class}"


def test_compatibility_matrix_sign_encoding_universal_except_meta_codec():
    # Sign-encoding applies to everything except META_CODEC.
    for sub_class in SubstrateClass:
        ok = primitive_compatibility(sub_class, PrimitiveCategory.SIGN_ENCODING)
        if sub_class == SubstrateClass.META_CODEC:
            assert ok is False
        else:
            assert ok is True, f"sign-encoding must apply to {sub_class}"


def test_compatibility_matrix_schema_elision_renderer_replacement_only():
    for sub_class in SubstrateClass:
        ok = primitive_compatibility(sub_class, PrimitiveCategory.SCHEMA_ELISION)
        if sub_class == SubstrateClass.RENDERER_REPLACEMENT:
            assert ok is True
        else:
            assert ok is False


def test_compatibility_matrix_brotli_lzma_universal():
    for sub_class in SubstrateClass:
        assert primitive_compatibility(sub_class, PrimitiveCategory.BROTLI) is True
        assert primitive_compatibility(sub_class, PrimitiveCategory.LZMA) is True


def test_compatibility_matrix_magic_codec_excludes_meta_codec():
    # MAGIC_CODEC_DENSE_STREAMS does NOT apply to META_CODEC substrate
    # class (self-referential).
    for sub_class in SubstrateClass:
        ok = primitive_compatibility(
            sub_class, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS
        )
        if sub_class == SubstrateClass.META_CODEC:
            assert ok is False
        else:
            assert ok is True


# ── 3. Pipeline-ordering validator ────────────────────────────────────────


def test_validate_pipeline_empty_ok():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering([], by_id)
    assert ok is True


def test_validate_pipeline_unknown_primitive_id_rejected():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(["nonexistent_primitive"], by_id)
    assert ok is False
    assert "unknown primitive_id" in msg


def test_validate_pipeline_duplicate_primitive_rejected():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(["brotli", "brotli"], by_id)
    assert ok is False
    assert "duplicate primitive_ids" in msg


def test_validate_pipeline_two_sign_encodings_rejected():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(
        ["sign_encoding_negzig", "sign_encoding_zig"], by_id
    )
    assert ok is False
    assert "mutually-exclusive" in msg
    assert "sign_encoding" in msg


def test_validate_pipeline_two_schema_elisions_rejected():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(
        ["pr98_cd1_compact_format", "pr100_schema_driven_decoder"], by_id
    )
    assert ok is False
    assert "mutually-exclusive" in msg
    assert "schema_elision" in msg


def test_validate_pipeline_pr101_gold_in_declared_order_ok():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(
        [
            "pr101_decoder_storage_order",
            "pr101_conv4_storage_perms",
            "pr101_decoder_byte_maps",
        ],
        by_id,
    )
    assert ok is True


def test_validate_pipeline_cross_category_order_index_must_increase():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    # brotli (idx=4) before pr101_decoder_storage_order (idx=0) — bad.
    ok, msg = validate_pipeline_ordering(
        ["brotli", "pr101_decoder_storage_order"], by_id
    )
    assert ok is False
    assert "order_index not monotonic" in msg


def test_validate_pipeline_brotli_and_lzma_stackable_ok():
    # brotli + lzma both order_index=4 but in different categories
    # (BROTLI vs LZMA) — they are STACKABLE; no MX collision.
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(["brotli", "lzma"], by_id)
    assert ok is True


# ── 4 + 5. Enumerator compatibility-matrix + MX gates ─────────────────────


def test_enumerate_default_returns_cells():
    cells = enumerate_cells()
    assert len(cells) > 0
    # All cells must have cell_id, substrate_id non-empty.
    for c in cells:
        assert c.cell_id
        assert c.substrate_id


def test_enumerate_cell_ids_unique():
    cells = enumerate_cells(max_primitives_per_cell=4)
    ids = [c.cell_id for c in cells]
    assert len(set(ids)) == len(ids), "cell_ids must be unique"


def test_enumerate_skips_pr101_on_non_renderer_replacement():
    # No cell with a RESIDUAL substrate should contain a PR101 GOLD primitive
    # (the compatibility matrix should gate them out).
    cells = enumerate_cells(max_primitives_per_cell=4)
    for c in cells:
        if c.substrate_class != SubstrateClass.RENDERER_REPLACEMENT:
            for pid, _ in c.primitives:
                assert not pid.startswith("pr101_"), (
                    f"PR101 GOLD primitive {pid} applied to non-RENDERER_REPLACEMENT "
                    f"substrate {c.substrate_id} (class={c.substrate_class}); "
                    f"compatibility-matrix gate failed"
                )


def test_enumerate_skips_schema_elision_on_non_renderer_replacement():
    cells = enumerate_cells(max_primitives_per_cell=4)
    schema_elision_ids = {
        "pr98_cd1_compact_format",
        "pr100_schema_driven_decoder",
        "pr105_packed_state_schema",
    }
    for c in cells:
        if c.substrate_class != SubstrateClass.RENDERER_REPLACEMENT:
            for pid, _ in c.primitives:
                assert pid not in schema_elision_ids, (
                    f"schema-elision primitive {pid} applied to non-RENDERER_REPLACEMENT "
                    f"substrate {c.substrate_id}"
                )


def test_enumerate_no_compatible_cell_has_two_sign_encodings():
    cells = enumerate_cells(max_primitives_per_cell=4)
    sign_ids = {
        "sign_encoding_negzig",
        "sign_encoding_zig",
        "sign_encoding_twos",
        "sign_encoding_off",
        "sign_encoding_raw_uint8",
    }
    for c in cells:
        if c.compatibility_verdict != "compatible":
            continue
        signs_in_cell = [pid for pid, _ in c.primitives if pid in sign_ids]
        assert len(signs_in_cell) <= 1, (
            f"compatible cell {c.cell_id} has multiple sign-encodings: "
            f"{signs_in_cell}"
        )


def test_enumerate_no_compatible_cell_has_two_schema_elisions():
    cells = enumerate_cells(max_primitives_per_cell=4)
    schema_elision_ids = {
        "pr98_cd1_compact_format",
        "pr100_schema_driven_decoder",
        "pr105_packed_state_schema",
    }
    for c in cells:
        if c.compatibility_verdict != "compatible":
            continue
        schemas_in_cell = [pid for pid, _ in c.primitives if pid in schema_elision_ids]
        assert len(schemas_in_cell) <= 1, (
            f"compatible cell {c.cell_id} has multiple schema-elisions: "
            f"{schemas_in_cell}"
        )


# ── 6. Ordered-pipeline within-category ordering ─────────────────────────


def test_enumerate_pr101_gold_in_compatible_cells_in_declared_order():
    cells = enumerate_cells(max_primitives_per_cell=4)
    declared = [
        "pr101_decoder_storage_order",
        "pr101_conv4_storage_perms",
        "pr101_decoder_byte_maps",
    ]
    for c in cells:
        if c.compatibility_verdict != "compatible":
            continue
        # Extract PR101 GOLD primitives from this cell in observed order.
        observed = [pid for pid, _ in c.primitives if pid.startswith("pr101_")]
        if len(observed) <= 1:
            continue
        # The observed sequence must be a subsequence of the declared order.
        # i.e. their declared-indices must be increasing.
        declared_indices = [declared.index(pid) for pid in observed]
        assert declared_indices == sorted(declared_indices), (
            f"cell {c.cell_id} has PR101 GOLD out of declared order: {observed}"
        )


# ── 7. Cross-category order_index monotonicity ───────────────────────────


def test_enumerate_compatible_cells_have_monotonic_order_index():
    cells = enumerate_cells(max_primitives_per_cell=4)
    prims = {p.primitive_id: p for p in canonical_primitive_inventory()}
    for c in cells:
        if c.compatibility_verdict != "compatible":
            continue
        indices = [prims[pid].order_index for pid, _ in c.primitives]
        assert indices == sorted(indices), (
            f"cell {c.cell_id} has non-monotonic order_index: {indices}"
        )


# ── 8. Bare-substrate baseline ───────────────────────────────────────────


def test_enumerate_bare_substrate_one_per_substrate():
    cells = enumerate_cells(max_primitives_per_cell=4, include_bare_substrate=True)
    bare = [
        c for c in cells
        if c.compatibility_verdict == "compatible_bare_substrate"
    ]
    subs = canonical_substrate_inventory()
    assert len(bare) == len(subs), (
        f"expected one bare-substrate cell per substrate "
        f"({len(subs)}), got {len(bare)}"
    )


def test_enumerate_bare_substrate_can_be_disabled():
    cells = enumerate_cells(max_primitives_per_cell=0, include_bare_substrate=False)
    assert cells == [], "max_primitives=0 + no bare → empty"


def test_enumerate_max_primitives_zero_gives_bare_only():
    cells = enumerate_cells(max_primitives_per_cell=0, include_bare_substrate=True)
    subs = canonical_substrate_inventory()
    assert len(cells) == len(subs)
    for c in cells:
        assert c.primitives == ()
        assert c.compatibility_verdict == "compatible_bare_substrate"


# ── 9. Score-claim discipline ────────────────────────────────────────────


def test_enumerate_every_cell_has_score_claim_false():
    cells = enumerate_cells(max_primitives_per_cell=4)
    for c in cells:
        assert c.score_claim is False, f"cell {c.cell_id} has score_claim=True"
        assert c.promotion_eligible is False
        assert c.ready_for_exact_eval_dispatch is False


def test_enumerate_violating_cells_have_zero_deltas():
    cells = enumerate_cells(max_primitives_per_cell=4)
    for c in cells:
        if c.compatibility_verdict == "violates_ordering_or_mutually_exclusive":
            assert c.predicted_score_delta == 0.0
            assert c.predicted_bytes_delta == 0
            assert c.blockers, f"violating cell {c.cell_id} missing blockers"


# ── 10. Autopilot ranking-input shape ────────────────────────────────────


def test_autopilot_ranking_input_returns_dicts():
    rows = autopilot_ranking_input(only_compatible=True)
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)


def test_autopilot_ranking_input_required_keys_present():
    rows = autopilot_ranking_input(only_compatible=True, only_with_primitives=True)
    required = {
        "candidate_id",
        "family",
        "predicted_score_delta",
        "expected_information_gain",
        "estimated_dispatch_cost_usd",
        "blockers",
        "composition_notes",
        "substrate_ids",
        "fits_per_dispatch_cap",
        "fits_cumulative_envelope",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"row {r.get('candidate_id')} missing keys: {missing}"


def test_autopilot_ranking_input_only_compatible_excludes_violating():
    rows = autopilot_ranking_input(only_compatible=True)
    # No row should have an MX-violation blocker.
    for r in rows:
        for blocker in r["blockers"]:
            assert "mutually-exclusive" not in blocker.lower(), r["candidate_id"]
            assert "duplicate primitive_ids" not in blocker, r["candidate_id"]


def test_autopilot_ranking_input_only_with_primitives_excludes_bare():
    rows = autopilot_ranking_input(only_compatible=True, only_with_primitives=True)
    # No bare-substrate cell should remain.
    for r in rows:
        assert not r["candidate_id"].endswith("__bare"), r["candidate_id"]


def test_autopilot_ranking_input_score_claim_invariants():
    rows = autopilot_ranking_input(only_compatible=True)
    for r in rows:
        assert r["score_claim"] is False
        assert r["promotion_eligible"] is False
        assert r["ready_for_exact_eval_dispatch"] is False


# ── 11. Serialization round-trip ─────────────────────────────────────────


def test_serialize_enumeration_schema_keys():
    cells = enumerate_cells(max_primitives_per_cell=2)
    payload = serialize_enumeration(cells)
    for key in (
        "schema",
        "registry_schema",
        "n_cells",
        "n_compatible",
        "n_bare",
        "n_violating",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "evidence_grade",
        "cells",
        "claude_md_compliance_tags",
    ):
        assert key in payload, f"serialize output missing {key}"
    assert payload["schema"] == ENUMERATION_SCHEMA
    assert payload["registry_schema"] == SCHEMA_VERSION
    assert payload["score_claim"] is False


def test_serialize_primitive_inventory_returns_list_of_dicts():
    out = serialize_primitive_inventory()
    assert isinstance(out, list)
    assert len(out) == 14
    for d in out:
        assert isinstance(d, dict)
        assert "primitive_id" in d
        assert "category" in d
        assert "order_sensitivity" in d
        assert "target_axis" in d
        assert "predicted_bytes_delta_band" in d


def test_serialize_enumeration_planning_only_compliance_tag():
    cells = enumerate_cells(max_primitives_per_cell=1)
    payload = serialize_enumeration(cells)
    tags = payload["claude_md_compliance_tags"]
    assert "planning_only_no_score_claim" in tags
    assert "no_tmp_paths" in tags
    assert "substrate_primitive_composition_cell_registry_v1" in tags


# ── 12. Non-HNeRV PR101 GOLD soft blocker ────────────────────────────────


def test_pr101_gold_on_non_hnerv_renderer_replacement_carries_soft_blocker():
    cells = enumerate_cells(max_primitives_per_cell=4)
    # nerv_as_renderer is RENDERER_REPLACEMENT but not in the canonical
    # HNeRV-family set; cells that apply PR101 GOLD should carry the
    # `pr101_gold_applied_to_non_hnerv_substrate` blocker.
    candidate = None
    for c in cells:
        if (
            c.substrate_id == "nerv_as_renderer"
            and any(pid.startswith("pr101_") for pid, _ in c.primitives)
            and c.compatibility_verdict == "compatible"
        ):
            candidate = c
            break
    assert candidate is not None, "no PR101 GOLD cell for nerv_as_renderer"
    assert any(
        "pr101_gold_applied_to_non_hnerv_substrate" in b
        for b in candidate.blockers
    ), candidate.blockers


# ── 13. Stable sort order ────────────────────────────────────────────────


def test_enumerate_stable_sort_across_runs():
    cells_a = enumerate_cells(max_primitives_per_cell=3)
    cells_b = enumerate_cells(max_primitives_per_cell=3)
    assert [c.cell_id for c in cells_a] == [c.cell_id for c in cells_b]


def test_enumerate_stable_sort_within_substrate_by_pipeline_length():
    cells = enumerate_cells(max_primitives_per_cell=4)
    # Group by substrate; within group, len(primitives) should be
    # non-decreasing.
    from itertools import groupby
    for sub, grp in groupby(cells, key=lambda c: c.substrate_id):
        lengths = [len(c.primitives) for c in grp]
        assert lengths == sorted(lengths), (
            f"substrate {sub} pipeline lengths not non-decreasing: {lengths}"
        )


# ── 14. Edge cases ───────────────────────────────────────────────────────


def test_enumerate_empty_substrate_inventory_raises():
    with pytest.raises(ValueError, match="substrate inventory empty"):
        enumerate_cells(substrates=[])


def test_enumerate_negative_max_primitives_raises():
    with pytest.raises(ValueError, match="max_primitives_per_cell"):
        enumerate_cells(max_primitives_per_cell=-1)


def test_cell_autopilot_candidate_kwargs_shape():
    cells = enumerate_cells(max_primitives_per_cell=2, include_bare_substrate=False)
    compatible = [c for c in cells if c.compatibility_verdict == "compatible"]
    assert compatible, "expected at least one compatible cell with primitives"
    sample = compatible[0]
    kw = sample.autopilot_candidate_kwargs()
    assert kw["candidate_id"] == sample.cell_id
    assert kw["family"] == sample.substrate_class.value
    assert kw["predicted_score_delta"] == sample.predicted_score_delta
    assert kw["expected_information_gain"] == abs(sample.predicted_score_delta)


def test_compositioncell_primitive_ids_helper():
    cells = enumerate_cells(max_primitives_per_cell=2, include_bare_substrate=False)
    for c in cells[:5]:
        assert c.primitive_ids() == tuple(pid for pid, _ in c.primitives)


def test_compatibility_matrix_no_missing_class_category_combo():
    # Every (SubstrateClass, PrimitiveCategory) pair should be answerable.
    for sc in SubstrateClass:
        for cat in PrimitiveCategory:
            # The function never raises — it returns False for un-set keys.
            result = primitive_compatibility(sc, cat)
            assert isinstance(result, bool)


# ── FIX-D (2026-05-12) — semantic_compatibility_warning + RefusedReason ──
#
# Per ZZZZZ medium polish audit 2026-05-12. Adds:
# - SemanticConstraint declaration on PrimitiveRow.
# - semantic_compatibility_warning field on CompositionCell (populated
#   for compatible cells whose pipeline has a semantically no-op
#   primitive on the substrate).
# - RefusedReason enum + refused_reason field on CompositionCell
#   (populated for every refused cell so consumers can branch on the
#   refusal class without parsing prose).


# ── FIX-D.1: RefusedReason enum coverage ─────────────────────────────────


def test_refused_reason_enum_has_all_taxonomy_classes():
    expected = {
        "ordering_violation",
        "mutual_exclusion",
        "substrate_incompatible",
        "semantic_incompatible",
        "dependency_missing",
        "duplicate_primitive",
        "unknown_primitive",
    }
    actual = {r.value for r in RefusedReason}
    assert actual == expected, f"refused_reason taxonomy drift: {actual ^ expected}"


def test_refused_reason_values_are_strings():
    for r in RefusedReason:
        assert isinstance(r.value, str)
        assert r.value == r.value.lower(), "values must be lowercase snake"


# ── FIX-D.2: SemanticConstraint defaults + serialization ─────────────────


def test_semantic_constraint_default_empty_tuples():
    sc = SemanticConstraint()
    assert sc.redundant_with_substrate_ids == ()
    assert sc.expects_substrate_property == ()
    assert sc.incompatible_with_substrate_ids == ()
    assert sc.requires_primitive_ids == ()


def test_primitive_inventory_carries_semantic_constraints():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    # PR101 GOLD step 2 requires step 1.
    step2 = by_id["pr101_conv4_storage_perms"]
    assert (
        "pr101_decoder_storage_order"
        in step2.semantic_constraint.requires_primitive_ids
    )
    # PR101 GOLD step 3 requires both step 1 and step 2.
    step3 = by_id["pr101_decoder_byte_maps"]
    assert (
        "pr101_decoder_storage_order"
        in step3.semantic_constraint.requires_primitive_ids
    )
    assert (
        "pr101_conv4_storage_perms"
        in step3.semantic_constraint.requires_primitive_ids
    )
    # magic_codec_dense_streams is redundant on cool_chic and c3.
    magic = by_id["magic_codec_dense_streams"]
    assert (
        "cool_chic_residual"
        in magic.semantic_constraint.redundant_with_substrate_ids
    )
    assert (
        "c3_residual"
        in magic.semantic_constraint.redundant_with_substrate_ids
    )


def test_primitive_inventory_sign_encoding_expects_int_tensor_weights():
    prims = canonical_primitive_inventory()
    for p in prims:
        if p.category == PrimitiveCategory.SIGN_ENCODING and p.primitive_id != "sign_encoding_off":
            assert (
                "int_tensor_weights"
                in p.semantic_constraint.expects_substrate_property
                or "categorical_substrate"
                in p.semantic_constraint.redundant_with_substrate_ids
            ), (
                f"sign-encoding primitive {p.primitive_id} should declare "
                "either int_tensor_weights property or categorical redundancy"
            )


# ── FIX-D.3: classify_pipeline_violation returns typed RefusedReason ────


def test_classify_pipeline_violation_duplicate_primitive():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, reason, msg = classify_pipeline_violation(["brotli", "brotli"], by_id)
    assert ok is False
    assert reason is RefusedReason.DUPLICATE_PRIMITIVE
    assert "duplicate primitive_ids" in msg


def test_classify_pipeline_violation_unknown_primitive():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, reason, msg = classify_pipeline_violation(["does_not_exist"], by_id)
    assert ok is False
    assert reason is RefusedReason.UNKNOWN_PRIMITIVE


def test_classify_pipeline_violation_mutual_exclusion():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, reason, msg = classify_pipeline_violation(
        ["sign_encoding_negzig", "sign_encoding_zig"], by_id
    )
    assert ok is False
    assert reason is RefusedReason.MUTUAL_EXCLUSION
    assert "mutually-exclusive" in msg


def test_classify_pipeline_violation_ordering_violation_cross_category():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    # brotli (idx=4) before pr101_storage_order (idx=0) — bad ordering.
    ok, reason, msg = classify_pipeline_violation(
        ["brotli", "pr101_decoder_storage_order"], by_id
    )
    assert ok is False
    assert reason is RefusedReason.ORDERING_VIOLATION


def test_classify_pipeline_violation_valid_returns_none_reason():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, reason, msg = classify_pipeline_violation(["brotli"], by_id)
    assert ok is True
    assert reason is None


def test_validate_pipeline_ordering_legacy_signature_preserved():
    # Back-compat: the legacy 2-tuple validator remains intact.
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = validate_pipeline_ordering(["brotli", "brotli"], by_id)
    assert ok is False
    assert "duplicate primitive_ids" in msg


# ── FIX-D.4: detect_dependency_violation ─────────────────────────────────


def test_detect_dependency_violation_step3_missing_step1():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    # PR101 GOLD step 3 requires step 1 AND step 2.
    ok, msg = detect_dependency_violation(
        ["pr101_decoder_byte_maps"], by_id
    )
    assert ok is False
    assert msg is not None
    assert "pr101_decoder_storage_order" in msg or "pr101_conv4_storage_perms" in msg


def test_detect_dependency_violation_full_trio_ok():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    ok, msg = detect_dependency_violation(
        [
            "pr101_decoder_storage_order",
            "pr101_conv4_storage_perms",
            "pr101_decoder_byte_maps",
        ],
        by_id,
    )
    assert ok is True
    assert msg is None


# ── FIX-D.5: compute_semantic_warning ────────────────────────────────────


def test_compute_semantic_warning_redundant_returns_string():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    magic = by_id["magic_codec_dense_streams"]
    warning = compute_semantic_warning("cool_chic_residual", [magic])
    assert warning is not None
    assert "magic_codec_dense_streams" in warning
    assert "cool_chic_residual" in warning
    assert "redundant" in warning.lower()


def test_compute_semantic_warning_no_match_returns_none():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    magic = by_id["magic_codec_dense_streams"]
    # blocknerv is RENDERER_REPLACEMENT but not in magic's redundant list.
    warning = compute_semantic_warning("blocknerv", [magic])
    assert warning is None


def test_compute_semantic_warning_empty_pipeline_returns_none():
    assert compute_semantic_warning("blocknerv", []) is None


def test_compute_semantic_warning_joins_multiple_with_pipe():
    prims = canonical_primitive_inventory()
    by_id = {p.primitive_id: p for p in prims}
    magic = by_id["magic_codec_dense_streams"]
    brot = by_id["brotli"]  # also declares cool_chic_residual redundancy.
    warning = compute_semantic_warning("cool_chic_residual", [magic, brot])
    assert warning is not None
    assert " | " in warning  # joined.
    assert "magic_codec_dense_streams" in warning
    assert "brotli" in warning


# ── FIX-D.6: enumerate_cells populates refused_reason on refused cells ───


def test_enumerate_cells_refused_cells_carry_refused_reason():
    cells = enumerate_cells(max_primitives_per_cell=4)
    for c in cells:
        if c.compatibility_verdict == "compatible":
            assert c.refused_reason is None, (
                f"compatible cell {c.cell_id} unexpectedly has "
                f"refused_reason={c.refused_reason}"
            )
            continue
        if c.compatibility_verdict == "compatible_bare_substrate":
            assert c.refused_reason is None
            continue
        # Refused cell — must have a typed reason.
        assert c.refused_reason is not None, (
            f"refused cell {c.cell_id} (verdict={c.compatibility_verdict}) "
            "lacks refused_reason"
        )
        assert isinstance(c.refused_reason, RefusedReason)


def test_enumerate_cells_mutual_exclusion_refused_reason():
    cells = enumerate_cells(max_primitives_per_cell=4)
    mx_cells = [
        c for c in cells
        if c.refused_reason == RefusedReason.MUTUAL_EXCLUSION
    ]
    assert mx_cells, "expected at least one MUTUAL_EXCLUSION cell"
    # Every such cell has two primitives in the same MX category.
    for c in mx_cells[:5]:
        sign_ids = {
            "sign_encoding_negzig",
            "sign_encoding_zig",
            "sign_encoding_twos",
            "sign_encoding_off",
            "sign_encoding_raw_uint8",
        }
        schema_ids = {
            "pr98_cd1_compact_format",
            "pr100_schema_driven_decoder",
            "pr105_packed_state_schema",
        }
        cell_ids = {pid for pid, _ in c.primitives}
        # Either >=2 sign-encodings or >=2 schema-elisions.
        n_sign = len(cell_ids & sign_ids)
        n_schema = len(cell_ids & schema_ids)
        assert n_sign >= 2 or n_schema >= 2, c.primitives


def test_enumerate_cells_dependency_missing_refused_reason():
    cells = enumerate_cells(max_primitives_per_cell=4)
    dep_cells = [
        c for c in cells
        if c.refused_reason == RefusedReason.DEPENDENCY_MISSING
    ]
    assert dep_cells, "expected at least one DEPENDENCY_MISSING cell"
    # Every such cell has a PR101 step without its prerequisite.
    for c in dep_cells[:5]:
        cell_pids = {pid for pid, _ in c.primitives}
        has_step3 = "pr101_decoder_byte_maps" in cell_pids
        has_step2 = "pr101_conv4_storage_perms" in cell_pids
        has_step1 = "pr101_decoder_storage_order" in cell_pids
        if has_step3:
            assert not (has_step1 and has_step2), (
                f"DEPENDENCY_MISSING cell {c.cell_id} has full trio: {cell_pids}"
            )
        elif has_step2:
            assert not has_step1


# ── FIX-D.7: enumerate_cells populates semantic_compatibility_warning ──


def test_enumerate_cells_semantic_warning_on_magic_codec_cool_chic():
    cells = enumerate_cells(max_primitives_per_cell=4)
    # Find the compatible cell that pairs cool_chic_residual with
    # magic_codec_dense_streams as the only primitive.
    target = None
    for c in cells:
        if (
            c.substrate_id == "cool_chic_residual"
            and c.compatibility_verdict == "compatible"
            and tuple(p for p, _ in c.primitives)
            == ("magic_codec_dense_streams",)
        ):
            target = c
            break
    assert target is not None, "no compatible (cool_chic, magic_codec) cell"
    assert target.semantic_compatibility_warning is not None
    assert "redundant" in target.semantic_compatibility_warning.lower()


def test_enumerate_cells_compatible_cells_warning_is_none_or_str():
    cells = enumerate_cells(max_primitives_per_cell=4)
    for c in cells:
        if c.compatibility_verdict != "compatible":
            continue
        # type contract: either None or str.
        assert (
            c.semantic_compatibility_warning is None
            or isinstance(c.semantic_compatibility_warning, str)
        )


def test_enumerate_cells_warning_count_is_positive():
    cells = enumerate_cells(max_primitives_per_cell=4)
    n_warned = sum(
        1 for c in cells if c.semantic_compatibility_warning is not None
    )
    assert n_warned > 0, "expected at least one semantic warning on enumeration"


# ── FIX-D.8: substrate-incompatible cells helper ─────────────────────────


def test_enumerate_substrate_incompatible_cells_emits_typed_refused():
    cells = enumerate_substrate_incompatible_cells()
    assert cells, "expected at least one substrate-incompatible cell"
    for c in cells:
        assert c.refused_reason == RefusedReason.SUBSTRATE_INCOMPATIBLE
        assert c.compatibility_verdict == "violates_substrate_class"
        # By construction the (substrate, primitive) pair fails the matrix.
        assert len(c.primitives) == 1, c.primitives


def test_enumerate_substrate_incompatible_cells_skips_compatible_pairs():
    cells = enumerate_substrate_incompatible_cells()
    # PR101 GOLD on a RENDERER_REPLACEMENT substrate is compatible — should
    # NOT appear in this output.
    for c in cells:
        if c.substrate_class == SubstrateClass.RENDERER_REPLACEMENT:
            for pid, _ in c.primitives:
                assert not pid.startswith("pr101_"), (
                    f"pair ({c.substrate_id}, {pid}) is compatible — should "
                    "not appear in substrate-incompatible output"
                )


def test_enumerate_substrate_incompatible_cells_empty_inv_raises():
    with pytest.raises(ValueError, match="substrate inventory empty"):
        enumerate_substrate_incompatible_cells(substrates=[])


# ── FIX-D.9: serialization captures new fields ───────────────────────────


def test_serialize_enumeration_has_refused_reason_histogram():
    cells = enumerate_cells(max_primitives_per_cell=4)
    payload = serialize_enumeration(cells)
    assert "n_refused_by_reason" in payload
    counts = payload["n_refused_by_reason"]
    assert isinstance(counts, dict)
    assert "mutual_exclusion" in counts
    assert counts["mutual_exclusion"] > 0
    # n_with_semantic_warning is also reported.
    assert "n_with_semantic_warning" in payload
    assert payload["n_with_semantic_warning"] > 0


def test_serialize_enumeration_cells_carry_refused_reason_string():
    cells = enumerate_cells(max_primitives_per_cell=4)
    payload = serialize_enumeration(cells)
    sample = next(
        c for c in payload["cells"]
        if c["refused_reason"] is not None
    )
    # Must serialize as a string value (not the enum object).
    assert isinstance(sample["refused_reason"], str)
    assert sample["refused_reason"] in {r.value for r in RefusedReason}


def test_serialize_primitive_inventory_carries_semantic_constraint():
    out = serialize_primitive_inventory()
    by_id = {d["primitive_id"]: d for d in out}
    # PR101 GOLD step 3 declares semantic_constraint.requires_primitive_ids.
    step3 = by_id["pr101_decoder_byte_maps"]
    assert "semantic_constraint" in step3
    sc = step3["semantic_constraint"]
    assert "requires_primitive_ids" in sc
    assert "pr101_decoder_storage_order" in sc["requires_primitive_ids"]


# ── FIX-D.10: autopilot_candidate_kwargs propagates new fields ──────────


def test_autopilot_candidate_kwargs_includes_new_fields():
    cells = enumerate_cells(max_primitives_per_cell=2, include_bare_substrate=False)
    compatible = [c for c in cells if c.compatibility_verdict == "compatible"]
    assert compatible
    sample = compatible[0]
    kw = sample.autopilot_candidate_kwargs()
    assert "semantic_compatibility_warning" in kw
    assert "refused_reason" in kw
    # refused_reason is a string-or-None (already enum-value-extracted).
    assert kw["refused_reason"] is None or isinstance(kw["refused_reason"], str)


def test_autopilot_candidate_kwargs_refused_cell_reports_typed_reason():
    cells = enumerate_cells(max_primitives_per_cell=4)
    refused = next(c for c in cells if c.refused_reason is not None)
    kw = refused.autopilot_candidate_kwargs()
    assert kw["refused_reason"] is not None
    assert kw["refused_reason"] == refused.refused_reason.value


# ── FIX-D.11: backward-compat invariants ─────────────────────────────────


def test_existing_compatible_cells_still_emit_after_fix_d():
    # The pre-FIX-D enumeration had 7858 compatible cells (7834 non-bare +
    # 24 bare). FIX-D dependency-missing classification subtracts the
    # PR101 GOLD subset combos that lacked their prerequisites. The
    # remaining compatible count is still substantial.
    cells = enumerate_cells(max_primitives_per_cell=4)
    n_compat = sum(1 for c in cells if c.compatibility_verdict == "compatible")
    n_bare = sum(
        1 for c in cells
        if c.compatibility_verdict == "compatible_bare_substrate"
    )
    assert n_compat > 4000  # post-FIX-D compatible (was 7834).
    assert n_bare == len(canonical_substrate_inventory())


def test_existing_violating_cells_still_emit_after_fix_d():
    # The 8975 MX violations remain (now typed via refused_reason).
    cells = enumerate_cells(max_primitives_per_cell=4)
    n_mx = sum(
        1 for c in cells
        if c.refused_reason == RefusedReason.MUTUAL_EXCLUSION
    )
    assert n_mx > 0


def test_detect_substrate_semantic_incompatibility_empty_returns_ok():
    # No primitives declare incompatible_with_substrate_ids by default;
    # any substrate passes.
    prims = canonical_primitive_inventory()
    ok, msg = detect_substrate_semantic_incompatibility(
        "blocknerv", prims
    )
    assert ok is True
    assert msg is None
