# SPDX-License-Identifier: MIT
"""Tests for ``tac.optimization.bit_allocator_end_to_end``.

Per CLAUDE.md "Recursive adversarial review protocol" these tests cover the
3-clean-pass adversarial greenup for the bit-allocator end-to-end wire
landing 2026-05-11.

Coverage:
- EndToEndBitAllocator construction + invariants
- Per-substrate mvpb + Hinton-surrogate boost + pose-axis multiplier
- Water-filling allocation invariants
- ORTHOGONAL pair preference
- Replacement constraint enforcement
- Whitelist filtering
- Negative budget refused
- Operating-point switching (PR106 r2 vs old 1x)
- AllocationPlan invariants (score_claim=False permanent)
- Compose archive manifest (planning-only + emit_payload_bytes)
- Serialization invariants
- /tmp path refusal
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.bit_allocator_end_to_end import (
    DEFAULT_GLOBAL_BYTE_BUDGET,
    PR106_R2_POSE_MARGINAL_MULTIPLIER,
    SCHEMA_VERSION,
    EndToEndBitAllocator,
    serialize_allocation_plan,
    write_allocation_plan_json,
)
from tac.optimization.substrate_composition_matrix import (
    Composability,
    ScoreAxis,
    SubstrateClass,
    canonical_substrate_inventory,
)

# ── Construction + invariants ─────────────────────────────────────────────


def test_default_constructor():
    allocator = EndToEndBitAllocator()
    # 39 = 24 legacy + 15 FIX-J substrate-scaffold rows (LOOPCLOSE 2026-05-12).
    assert allocator.matrix.n_substrates() == 39
    assert allocator.operating_point == "pr106_r2_frontier"


def test_constructor_refuses_unknown_operating_point():
    with pytest.raises(ValueError, match="unknown operating_point"):
        EndToEndBitAllocator(operating_point="some_other_op")


def test_constructor_accepts_old_1x_operating_point():
    allocator = EndToEndBitAllocator(operating_point="old_1x_score")
    assert allocator.operating_point == "old_1x_score"
    assert allocator._pose_axis_multiplier() == 1.0


def test_constructor_pr106_r2_pose_multiplier_is_2_79():
    allocator = EndToEndBitAllocator(operating_point="pr106_r2_frontier")
    assert allocator._pose_axis_multiplier() == PR106_R2_POSE_MARGINAL_MULTIPLIER
    assert PR106_R2_POSE_MARGINAL_MULTIPLIER == 2.79


def test_priors_disable_individually():
    allocator = EndToEndBitAllocator(
        enable_magic_codec_priors=False,
        enable_sparse_packet_ir_priors=False,
        enable_hinton_surrogate_priors=False,
        enable_autopilot_ranking_priors=False,
    )
    plan = allocator.allocate_bits_across_substrates()
    assert not plan.magic_codec_priors_consumed
    assert not plan.sparse_packet_ir_priors_consumed
    assert not plan.hinton_surrogate_priors_consumed
    assert not plan.autopilot_ranking_consumed


# ── Per-substrate mvpb + boosts ───────────────────────────────────────────


def test_pose_axis_multiplier_applied_to_pose_substrates():
    allocator = EndToEndBitAllocator(operating_point="pr106_r2_frontier")
    rows = canonical_substrate_inventory()
    pose_rows = [r for r in rows if r.target_axis == ScoreAxis.POSE]
    assert pose_rows, "expected at least one pose-axis substrate in inventory"
    plan = allocator.allocate_bits_across_substrates()
    # Find pose-axis allocations and check multiplier.
    pose_allocs = [a for a in plan.allocations if a.target_axis == ScoreAxis.POSE]
    assert pose_allocs
    for a in pose_allocs:
        assert a.pose_axis_multiplier_applied == PR106_R2_POSE_MARGINAL_MULTIPLIER


def test_pose_axis_multiplier_not_applied_at_old_1x():
    allocator = EndToEndBitAllocator(operating_point="old_1x_score")
    plan = allocator.allocate_bits_across_substrates()
    pose_allocs = [a for a in plan.allocations if a.target_axis == ScoreAxis.POSE]
    for a in pose_allocs:
        assert a.pose_axis_multiplier_applied == 1.0


def test_hinton_surrogate_boost_increases_residual_mvpb():
    rows = canonical_substrate_inventory()
    residual_rows = [r for r in rows if r.substrate_class == SubstrateClass.RESIDUAL]
    assert residual_rows
    a_with = EndToEndBitAllocator(enable_hinton_surrogate_priors=True)
    a_without = EndToEndBitAllocator(enable_hinton_surrogate_priors=False)
    sample = residual_rows[0]
    assert a_with._per_substrate_mvpb(sample) > a_without._per_substrate_mvpb(sample)


# ── Water-filling allocation invariants ──────────────────────────────────


def test_total_allocated_does_not_exceed_budget():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(total_byte_budget=100_000)
    assert plan.total_allocated_bytes <= 100_000


def test_zero_budget_allocates_zero_bytes():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(total_byte_budget=0)
    assert plan.total_allocated_bytes == 0
    # All allocations should be 0 (budget exhausted before any floor).
    for a in plan.allocations:
        assert a.allocated_bytes == 0


def test_negative_budget_refused():
    allocator = EndToEndBitAllocator()
    with pytest.raises(ValueError, match="must be >= 0"):
        allocator.allocate_bits_across_substrates(total_byte_budget=-1)


def test_default_budget_uses_pr106_frontier_value():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    assert plan.global_byte_budget == DEFAULT_GLOBAL_BYTE_BUDGET == 200_000


def test_allocations_have_nonempty_rationale():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    for a in plan.allocations:
        assert "[predicted; bit allocator end-to-end wire v1]" in a.rationale


# ── ORTHOGONAL pair preference ────────────────────────────────────────────


def test_orthogonal_pairs_emit_paired_allocations():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(prefer_orthogonal_pairs=True)
    paired = [a for a in plan.allocations if a.paired_substrate_id is not None]
    assert paired, "expected at least one orthogonal pair allocation"
    for a in paired:
        assert a.composability_with_pair == "orthogonal"


def test_no_pair_allocations_when_disabled():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(prefer_orthogonal_pairs=False)
    paired = [a for a in plan.allocations if a.paired_substrate_id is not None]
    assert paired == [], "expected no paired allocations when prefer_orthogonal_pairs=False"


def test_paired_allocations_reference_each_other():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    paired = [a for a in plan.allocations if a.paired_substrate_id is not None]
    by_id = {a.substrate_id: a for a in plan.allocations}
    for a in paired:
        partner = by_id.get(a.paired_substrate_id)
        assert partner is not None
        assert partner.paired_substrate_id == a.substrate_id


# ── Replacement constraint enforcement ───────────────────────────────────


def test_replacement_constraint_keeps_at_most_one_renderer():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(
        respect_replacement_constraint=True
    )
    renderer_allocs = [
        a
        for a in plan.allocations
        if a.substrate_class == SubstrateClass.RENDERER_REPLACEMENT
    ]
    assert len(renderer_allocs) <= 1


def test_replacement_constraint_disabled_admits_multiple_renderers():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(
        total_byte_budget=2_000_000,
        respect_replacement_constraint=False,
    )
    renderer_allocs = [
        a
        for a in plan.allocations
        if a.substrate_class == SubstrateClass.RENDERER_REPLACEMENT
    ]
    # 10 renderer-replacement substrates in canonical inventory.
    assert len(renderer_allocs) > 1


# ── Whitelist filtering ──────────────────────────────────────────────────


def test_whitelist_keeps_only_named_substrates():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(
        allowed_substrate_ids=("c3_residual", "wavelet_residual"),
    )
    ids = {a.substrate_id for a in plan.allocations}
    assert ids == {"c3_residual", "wavelet_residual"}


def test_whitelist_unknown_substrate_refused():
    allocator = EndToEndBitAllocator()
    with pytest.raises(ValueError, match="unknown substrate_ids"):
        allocator.allocate_bits_across_substrates(
            allowed_substrate_ids=("nonexistent_substrate",),
        )


def test_whitelist_unknown_id_refused_even_when_inventory_overridden():
    """Whitelist enforcement runs against the (possibly overridden) inventory."""
    allocator = EndToEndBitAllocator()
    rows = [r for r in canonical_substrate_inventory() if r.substrate_id == "c3_residual"]
    with pytest.raises(ValueError, match="unknown substrate_ids"):
        allocator.allocate_bits_across_substrates(
            substrate_inventory=rows,
            allowed_substrate_ids=("wavelet_residual",),
        )


# ── Plan invariants ──────────────────────────────────────────────────────


def test_plan_invariants_score_claim_false_permanent():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    assert plan.score_claim is False
    assert plan.promotion_eligible is False
    assert plan.ready_for_exact_eval_dispatch is False
    for a in plan.allocations:
        assert a.score_claim is False
        assert a.promotion_eligible is False
        assert a.ready_for_exact_eval_dispatch is False


def test_plan_schema_pinned_v1():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    assert plan.schema == SCHEMA_VERSION == "tac_bit_allocator_end_to_end_v1"


def test_plan_records_operating_point():
    allocator = EndToEndBitAllocator(operating_point="old_1x_score")
    plan = allocator.allocate_bits_across_substrates()
    assert plan.operating_point == "old_1x_score"
    assert plan.pose_axis_multiplier == 1.0


# ── Compose archive manifest ─────────────────────────────────────────────


def test_compose_archive_planning_only_no_payload_bytes():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    manifest = allocator.compose_archive_with_allocations(plan)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    for row in manifest["rows"]:
        assert "payload_bytes_hex" not in row


def test_compose_archive_with_payload_bytes_emits_zero_filled():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    manifest = allocator.compose_archive_with_allocations(
        plan, emit_payload_bytes=True
    )
    for row in manifest["rows"]:
        if row["allocated_bytes"] > 0:
            assert "payload_bytes_hex" in row
            assert set(row["payload_bytes_hex"]) <= {"0"}
            assert len(row["payload_bytes_hex"]) == row["allocated_bytes"] * 2


def test_compose_archive_includes_format_id_and_magic():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    manifest = allocator.compose_archive_with_allocations(plan)
    for row in manifest["rows"]:
        assert isinstance(row["format_id"], int)
        assert isinstance(row["magic_bytes"], str)
        assert len(row["magic_bytes"]) == 4


# ── Serialization ────────────────────────────────────────────────────────


def test_serialize_allocation_plan_jsonable():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    payload = serialize_allocation_plan(plan)
    s = json.dumps(payload)
    parsed = json.loads(s)
    assert parsed["schema"] == SCHEMA_VERSION
    assert parsed["score_claim"] is False


def test_serialize_includes_claude_md_compliance_tags():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    payload = serialize_allocation_plan(plan)
    tags = payload["claude_md_compliance_tags"]
    assert "planning_only_no_score_claim" in tags
    assert "no_mps_authoritative" in tags
    assert "no_tmp_paths" in tags
    assert "shared_byte_allocation_water_filling_v1" in tags


def test_write_allocation_plan_json_roundtrips(tmp_path: Path):
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    out = tmp_path / "plan.json"
    write_allocation_plan_json(plan, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == SCHEMA_VERSION
    assert loaded["operating_point"] == plan.operating_point


def test_write_allocation_plan_refuses_tmp_path():
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    with pytest.raises(ValueError, match="forbidden /tmp path"):
        write_allocation_plan_json(plan, "/tmp/should_not_be_allowed.json")
    with pytest.raises(ValueError, match="forbidden /tmp path"):
        write_allocation_plan_json(plan, "/var/tmp/should_not_be_allowed.json")


# ── Composite delta accumulates per substrate ────────────────────────────


def test_total_predicted_delta_is_finite_and_bounded():
    """Total predicted delta accumulates per-substrate; finite + small magnitude.

    Per the canonical substrate inventory some substrates have predicted-band
    midpoints with mixed sign (e.g., NeRV-family at (-0.005, +0.015) -> +0.005
    midpoint), so the accumulated total can be either sign. We pin only
    finiteness and order-of-magnitude bound.
    """
    import math
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates()
    assert math.isfinite(plan.total_predicted_delta)
    # Per CLAUDE.md "theoretical floor v2 refresh" the per-substrate
    # predicted bands sum to no more than ~0.05 in magnitude across the
    # 24 substrates at full allocation.
    assert abs(plan.total_predicted_delta) < 0.5


def test_pair_membership_consistent_with_matrix():
    """Every paired allocation must reference an ORTHOGONAL pair in the matrix."""
    allocator = EndToEndBitAllocator()
    plan = allocator.allocate_bits_across_substrates(prefer_orthogonal_pairs=True)
    matrix = allocator.matrix
    paired = [a for a in plan.allocations if a.paired_substrate_id is not None]
    for a in paired:
        cell = matrix.get(a.substrate_id, a.paired_substrate_id)
        assert cell.composability == Composability.ORTHOGONAL
