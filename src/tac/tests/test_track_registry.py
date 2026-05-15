# SPDX-License-Identifier: MIT
"""Tests for tac.track_registry — single source of truth for landed tracks.

Covers:
  - Every TrackKind in unified_action has a registry entry
  - Visibility lookups return correct subsets
  - Pareto-axis grouping matches the audit table
  - Promotion eligibility flags match expected initial state
"""
from __future__ import annotations

import pytest

from tac.track_registry import (
    TRACK_REGISTRY,
    TRACK_REGISTRY_SCHEMA_VERSION,
    ParetoAxis,
    TrackEntry,
    TrackPhase,
    get_track,
    list_promotable_tracks,
    list_tracks_by_pareto_axis,
    list_tracks_by_phase,
    list_tracks_visible_to,
)
from tac.unified_action import TrackKind


# ── Schema + completeness ──────────────────────────────────────────────────


def test_schema_version_exposed():
    assert TRACK_REGISTRY_SCHEMA_VERSION.startswith("tac_track_registry_v")


def test_every_unified_action_track_has_registry_entry():
    """Every TrackKind enum value (except numerical-solver-only T19 + baselines)
    must have a registry entry."""
    # Baselines are not refinement tracks and don't appear in registry.
    baselines = {
        TrackKind.SEG_BASELINE.value,
        TrackKind.POSE_BASELINE.value,
        TrackKind.RATE_BASELINE.value,
    }
    for kind in TrackKind:
        if kind.value in baselines:
            continue
        # T19 is in registry too (numerical solver step).
        # Lane 12-v2 is in registry.
        assert kind.value in TRACK_REGISTRY, f"missing registry entry for {kind.value}"


def test_registry_keys_match_audit_table():
    """Audit table track_ids match registry keys."""
    expected = {
        "t7_fisher_rao", "t8_sinkhorn_w2", "t11_lovasz_hinge",
        "t13_joint_source_rd", "t19_adaptive_rho",
        "t20_kl_pose_distill", "t22_temporal_consistency",
        "lane_12_v2_nerv_as_renderer", "a1_substrate",
        "frame0_postdecode_selector",
    }
    assert set(TRACK_REGISTRY.keys()) == expected


def test_get_track_returns_entry():
    entry = get_track("t13_joint_source_rd")
    assert isinstance(entry, TrackEntry)
    assert entry.track_id == "t13_joint_source_rd"


def test_get_track_unknown_id_raises_with_canonical_list():
    with pytest.raises(KeyError, match="valid ids"):
        get_track("t99_does_not_exist")


# ── Visibility ─────────────────────────────────────────────────────────────


def test_visibility_unified_action_includes_loss_terms():
    visible = list_tracks_visible_to("unified_action")
    visible_ids = {t.track_id for t in visible}
    assert "t8_sinkhorn_w2" in visible_ids
    assert "t11_lovasz_hinge" in visible_ids
    assert "t20_kl_pose_distill" in visible_ids


def test_visibility_joint_admm_coordinator_sees_t19_and_t13():
    visible = list_tracks_visible_to("joint_admm_coordinator")
    visible_ids = {t.track_id for t in visible}
    assert "t19_adaptive_rho" in visible_ids
    assert "t13_joint_source_rd" in visible_ids


def test_visibility_continual_learning_sees_lane_12_v2_and_a1():
    visible = list_tracks_visible_to("continual_learning_posterior")
    visible_ids = {t.track_id for t in visible}
    assert "lane_12_v2_nerv_as_renderer" in visible_ids
    assert "a1_substrate" in visible_ids
    assert "frame0_postdecode_selector" in visible_ids


def test_visibility_packet_compiler_sees_frame0_postdecode_selector():
    visible = list_tracks_visible_to("packet_compiler")
    visible_ids = {t.track_id for t in visible}
    assert visible_ids == {"frame0_postdecode_selector"}


def test_visibility_unknown_component_returns_empty_list():
    visible = list_tracks_visible_to("does_not_exist")
    assert visible == []


# ── Promotion eligibility ─────────────────────────────────────────────────


def test_promotable_tracks_includes_t8_t13_a1():
    """T8 (Phase 1 winner), T13 (already integrated), A1 (empirical anchor) are eligible."""
    promotable = list_promotable_tracks()
    promotable_ids = {t.track_id for t in promotable}
    assert "t8_sinkhorn_w2" in promotable_ids
    assert "t13_joint_source_rd" in promotable_ids
    assert "a1_substrate" in promotable_ids


def test_promotable_excludes_t19_and_lane_12_v2():
    """T19 (orphan helper), Lane 12-v2 (Phase A scaffold), T7/T11/T20/T22 (gated)
    are NOT yet eligible."""
    promotable = list_promotable_tracks()
    promotable_ids = {t.track_id for t in promotable}
    assert "t19_adaptive_rho" not in promotable_ids
    assert "lane_12_v2_nerv_as_renderer" not in promotable_ids
    assert "t7_fisher_rao" not in promotable_ids
    assert "t11_lovasz_hinge" not in promotable_ids
    assert "frame0_postdecode_selector" not in promotable_ids


# ── Pareto axis grouping ──────────────────────────────────────────────────


def test_seg_axis_groups_t7_t8_t11():
    seg_tracks = list_tracks_by_pareto_axis(ParetoAxis.SEG)
    seg_ids = {t.track_id for t in seg_tracks}
    assert seg_ids == {"t7_fisher_rao", "t8_sinkhorn_w2", "t11_lovasz_hinge"}


def test_rate_axis_groups_t13_only():
    rate_tracks = list_tracks_by_pareto_axis(ParetoAxis.RATE)
    rate_ids = {t.track_id for t in rate_tracks}
    assert rate_ids == {"t13_joint_source_rd"}


def test_pose_axis_groups_t20_only():
    pose_tracks = list_tracks_by_pareto_axis(ParetoAxis.POSE)
    pose_ids = {t.track_id for t in pose_tracks}
    assert pose_ids == {"t20_kl_pose_distill"}


def test_temporal_axis_groups_t22_only():
    temporal_tracks = list_tracks_by_pareto_axis(ParetoAxis.TEMPORAL)
    temporal_ids = {t.track_id for t in temporal_tracks}
    assert temporal_ids == {"t22_temporal_consistency"}


def test_multi_axis_groups_lane_12_v2_and_a1():
    multi_tracks = list_tracks_by_pareto_axis(ParetoAxis.MULTI)
    multi_ids = {t.track_id for t in multi_tracks}
    assert "lane_12_v2_nerv_as_renderer" in multi_ids
    assert "a1_substrate" in multi_ids
    assert "frame0_postdecode_selector" in multi_ids


def test_none_axis_groups_t19_only():
    """T19 is a numerical-solver step, not a Pareto-axis contributor."""
    none_tracks = list_tracks_by_pareto_axis(ParetoAxis.NONE)
    none_ids = {t.track_id for t in none_tracks}
    assert none_ids == {"t19_adaptive_rho"}


# ── Phase grouping ────────────────────────────────────────────────────────


def test_loss_term_phase_groups_correct_tracks():
    loss_tracks = list_tracks_by_phase(TrackPhase.LOSS_TERM)
    loss_ids = {t.track_id for t in loss_tracks}
    assert loss_ids == {"t7_fisher_rao", "t8_sinkhorn_w2", "t11_lovasz_hinge", "t20_kl_pose_distill"}


def test_rate_bound_phase_groups_t13_only():
    rate_bound_tracks = list_tracks_by_phase(TrackPhase.RATE_BOUND)
    rate_bound_ids = {t.track_id for t in rate_bound_tracks}
    assert rate_bound_ids == {"t13_joint_source_rd"}


def test_numerical_solver_phase_groups_t19_only():
    ns_tracks = list_tracks_by_phase(TrackPhase.NUMERICAL_SOLVER)
    ns_ids = {t.track_id for t in ns_tracks}
    assert ns_ids == {"t19_adaptive_rho"}


def test_architecture_phase_groups_lane_12_v2():
    arch_tracks = list_tracks_by_phase(TrackPhase.ARCHITECTURE)
    arch_ids = {t.track_id for t in arch_tracks}
    assert arch_ids == {"lane_12_v2_nerv_as_renderer"}


def test_postdecode_atom_phase_groups_frame0_selector():
    postdecode_tracks = list_tracks_by_phase(TrackPhase.POSTDECODE_ATOM)
    postdecode_ids = {t.track_id for t in postdecode_tracks}
    assert postdecode_ids == {"frame0_postdecode_selector"}


def test_substrate_phase_groups_a1():
    sub_tracks = list_tracks_by_phase(TrackPhase.SUBSTRATE)
    sub_ids = {t.track_id for t in sub_tracks}
    assert sub_ids == {"a1_substrate"}


# ── Evidence-tag discipline ───────────────────────────────────────────────


def test_every_entry_has_evidence_tag_in_canonical_form():
    """Per CLAUDE.md FORBIDDEN_PATTERNS: empirical-claim-without-evidence-tag.
    Every entry must carry an evidence_grade matching the canonical bracket form."""
    for entry in TRACK_REGISTRY.values():
        eg = entry.evidence_grade
        assert eg.startswith("["), f"{entry.track_id}: evidence_grade missing leading '['"
        assert eg.endswith("]"), f"{entry.track_id}: evidence_grade missing trailing ']'"


def test_landed_commit_or_memo_present():
    """Every entry must reference a landed commit OR memory file."""
    for entry in TRACK_REGISTRY.values():
        assert entry.landed_commit_or_memo, f"{entry.track_id}: missing landed_commit_or_memo"


def test_orphan_tracks_have_explicit_entry_conditions():
    """A non-promotable track MUST list its entry_conditions so a future
    operator can audit what's blocking promotion."""
    for entry in TRACK_REGISTRY.values():
        if entry.promotion_eligible:
            continue
        assert entry.entry_conditions, (
            f"{entry.track_id}: not promotable but no entry_conditions listed; "
            "operator can't tell what unblocks promotion"
        )
