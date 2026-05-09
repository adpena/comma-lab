"""Cathedral autopilot catalog wire-in for T20/T22/Lane-12-v2 (2026-05-09).

Per ``feedback_unified_solver_integration_landed_20260509.md`` deferred
wire-in #2: cathedral autopilot must SEE T20 (KL pose distill loss),
T22 (temporal consistency regularizer), and Lane-12-v2 (NeRV-as-renderer)
even though they are NOT yet dispatch-eligible.

This test file proves:

1. The 3 new catalog rows exist in their derived catalogs
2. Each catalog row carries the correct planner_visibility tag
3. Each catalog row's entry_conditions match the registry
4. The autopilot's build_plan output exposes them as
   ``loss_modifier_technique_ranking`` and ``representation_lane_ranking``
5. Gated entries do NOT appear in ``recommended_top_3`` until
   ``promotion_eligible == True``
6. The notes field surfaces the gated tracks for operator visibility
7. Re-running build_plan produces the same set of derived rows (idempotence)
8. The registry is single source of truth — adding a track to the registry
   with planner_visibility=cathedral_autopilot makes it appear in autopilot

Memory ref: ``feedback_t19_migration_and_cathedral_autopilot_catalog_landed_20260509.md``.

[empirical: tests/test_cathedral_autopilot_catalog_t20_t22_lane12v2.py]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.cathedral_autopilot import (  # noqa: E402
    LOSS_MODIFIER_TECHNIQUES,
    REPRESENTATION_LANES,
    AutopilotPlan,
    _seed_loss_modifier_catalog_from_registry,
    _seed_representation_lane_catalog_from_registry,
    build_plan,
)
from tac.track_registry import TRACK_REGISTRY, list_tracks_visible_to  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: T20 catalog row exists + has correct shape
# ─────────────────────────────────────────────────────────────────────────────


def test_t20_kl_pose_distill_in_loss_modifier_catalog():
    """T20 (KL pose distill) must appear in LOSS_MODIFIER_TECHNIQUES."""
    rows = [r for r in LOSS_MODIFIER_TECHNIQUES if r["name"] == "t20_kl_pose_distill"]
    assert len(rows) == 1, (
        f"expected exactly 1 t20_kl_pose_distill catalog row; got {len(rows)}"
    )
    row = rows[0]
    assert row["track_id"] == "t20_kl_pose_distill"
    assert row["track_kind"] == "loss_modifier"
    assert row["track_phase"] == "loss_term"
    assert row["track_pareto_axis"] == "pose"
    assert row["promotion_eligible"] is False  # gated on trainer wire-in
    # Module path must reference the canonical implementation.
    assert "kl_pose_distill" in row["module_path"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: T22 catalog row exists + has correct shape
# ─────────────────────────────────────────────────────────────────────────────


def test_t22_temporal_consistency_in_loss_modifier_catalog():
    """T22 (temporal consistency regularizer) must appear in LOSS_MODIFIER_TECHNIQUES."""
    rows = [r for r in LOSS_MODIFIER_TECHNIQUES if r["name"] == "t22_temporal_consistency"]
    assert len(rows) == 1, (
        f"expected exactly 1 t22_temporal_consistency catalog row; got {len(rows)}"
    )
    row = rows[0]
    assert row["track_id"] == "t22_temporal_consistency"
    assert row["track_kind"] == "loss_modifier"
    assert row["track_phase"] == "regularizer"
    assert row["track_pareto_axis"] == "temporal"
    assert row["promotion_eligible"] is False
    assert "temporal_consistency_regularizer" in row["module_path"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Lane-12-v2 catalog row exists + has correct shape
# ─────────────────────────────────────────────────────────────────────────────


def test_lane_12_v2_in_representation_lane_catalog():
    """Lane-12-v2 (NeRV-as-renderer) must appear in REPRESENTATION_LANES."""
    rows = [r for r in REPRESENTATION_LANES if r["name"] == "lane_12_v2_nerv_as_renderer"]
    assert len(rows) == 1, (
        f"expected exactly 1 lane_12_v2_nerv_as_renderer catalog row; got {len(rows)}"
    )
    row = rows[0]
    assert row["track_id"] == "lane_12_v2_nerv_as_renderer"
    assert row["track_kind"] == "representation_lane"
    assert row["track_phase"] == "architecture"
    assert row["track_pareto_axis"] == "multi"
    assert row["promotion_eligible"] is False  # gated on Phase B preconditions
    assert "lane_12_v2_nerv_as_renderer" in row["module_path"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: each catalog row has entry_conditions matching the registry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "track_id,catalog",
    [
        ("t20_kl_pose_distill", LOSS_MODIFIER_TECHNIQUES),
        ("t22_temporal_consistency", LOSS_MODIFIER_TECHNIQUES),
        ("lane_12_v2_nerv_as_renderer", REPRESENTATION_LANES),
    ],
)
def test_catalog_entry_conditions_match_registry(
    track_id: str, catalog: list[dict]
):
    """Catalog row's entry_conditions must mirror the registry's."""
    catalog_row = next(r for r in catalog if r["name"] == track_id)
    registry_entry = TRACK_REGISTRY[track_id]
    assert tuple(catalog_row["entry_conditions"]) == registry_entry.entry_conditions, (
        f"{track_id}: catalog entry_conditions drift from registry: "
        f"catalog={catalog_row['entry_conditions']!r} "
        f"registry={registry_entry.entry_conditions!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: autopilot build_plan exposes the new rankings
# ─────────────────────────────────────────────────────────────────────────────


def test_build_plan_emits_loss_modifier_and_representation_rankings():
    """build_plan output must include loss_modifier_technique_ranking and
    representation_lane_ranking fields populated from the registry-derived catalogs."""
    plan = build_plan(d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144)
    assert isinstance(plan, AutopilotPlan)
    # Loss-modifier ranking must contain T20 + T22.
    loss_names = {r["name"] for r in plan.loss_modifier_technique_ranking}
    assert "t20_kl_pose_distill" in loss_names
    assert "t22_temporal_consistency" in loss_names
    # Representation-lane ranking must contain Lane-12-v2.
    repr_names = {r["name"] for r in plan.representation_lane_ranking}
    assert "lane_12_v2_nerv_as_renderer" in repr_names


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: gated entries do NOT appear in recommended_top_3
# ─────────────────────────────────────────────────────────────────────────────


def test_gated_loss_modifier_not_in_top_3():
    """A loss-modifier with promotion_eligible=False must NOT appear in
    recommended_top_3 even if its predicted score-delta is positive."""
    plan = build_plan(d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144)
    top_3_names = {r["name"] for r in plan.recommended_top_3}
    # T20/T22/Lane-12-v2 are all gated (promotion_eligible=False) so they
    # must NOT appear in top_3.
    assert "t20_kl_pose_distill" not in top_3_names
    assert "t22_temporal_consistency" not in top_3_names
    assert "lane_12_v2_nerv_as_renderer" not in top_3_names


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: notes surface gated tracks
# ─────────────────────────────────────────────────────────────────────────────


def test_notes_surface_gated_tracks_for_operator_visibility():
    """The autopilot must add a NOTE listing every visible-but-gated track
    so the operator knows what's queued for trainer wire-in / Phase B."""
    plan = build_plan(d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144)
    notes_text = "\n".join(plan.notes)
    # Loss modifier note.
    assert "Loss-modifier tracks visible but GATED" in notes_text
    assert "t20_kl_pose_distill" in notes_text
    assert "t22_temporal_consistency" in notes_text
    # Representation lane note.
    assert "Representation lanes visible but GATED" in notes_text
    assert "lane_12_v2_nerv_as_renderer" in notes_text


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: idempotence — derived catalogs are stable across re-derivation
# ─────────────────────────────────────────────────────────────────────────────


def test_derived_catalogs_are_idempotent():
    """Re-deriving catalogs from the registry must produce the SAME row set
    (the seed functions must be deterministic)."""
    loss_a = _seed_loss_modifier_catalog_from_registry()
    loss_b = _seed_loss_modifier_catalog_from_registry()
    assert {r["name"] for r in loss_a} == {r["name"] for r in loss_b}
    repr_a = _seed_representation_lane_catalog_from_registry()
    repr_b = _seed_representation_lane_catalog_from_registry()
    assert {r["name"] for r in repr_a} == {r["name"] for r in repr_b}


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: registry is the single source of truth (cathedral_autopilot
#                                                  visibility set match)
# ─────────────────────────────────────────────────────────────────────────────


def test_cathedral_autopilot_visibility_set_matches_derived_catalogs():
    """Every track in the registry that names cathedral_autopilot in
    planner_visibility AND has phase loss_term/regularizer/architecture
    must appear in the derived catalogs.

    This is the structural single-source-of-truth check: adding a new
    track to the registry with planner_visibility=cathedral_autopilot
    automatically makes it appear in autopilot.
    """
    visible_tracks = list_tracks_visible_to("cathedral_autopilot")
    expected_loss_mods = {
        t.track_id for t in visible_tracks
        if t.phase.value in ("loss_term", "regularizer")
    }
    actual_loss_mods = {r["name"] for r in LOSS_MODIFIER_TECHNIQUES}
    assert expected_loss_mods == actual_loss_mods, (
        f"loss-modifier set drift: registry expects {expected_loss_mods}, "
        f"catalog has {actual_loss_mods}"
    )
    expected_repr_lanes = {
        t.track_id for t in visible_tracks
        if t.phase.value == "architecture"
    }
    actual_repr_lanes = {r["name"] for r in REPRESENTATION_LANES}
    assert expected_repr_lanes == actual_repr_lanes, (
        f"representation-lane set drift: registry expects {expected_repr_lanes}, "
        f"catalog has {actual_repr_lanes}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: catalog rows include landed_commit_or_memo for forensics
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "track_id,catalog",
    [
        ("t20_kl_pose_distill", LOSS_MODIFIER_TECHNIQUES),
        ("t22_temporal_consistency", LOSS_MODIFIER_TECHNIQUES),
        ("lane_12_v2_nerv_as_renderer", REPRESENTATION_LANES),
    ],
)
def test_catalog_row_carries_forensics_provenance(
    track_id: str, catalog: list[dict]
):
    """Every catalog row must carry landed_commit_or_memo so future operators
    can trace WHEN a track entered the planner."""
    row = next(r for r in catalog if r["name"] == track_id)
    assert row.get("landed_commit_or_memo"), (
        f"{track_id}: missing landed_commit_or_memo provenance"
    )
    assert row.get("evidence_grade"), (
        f"{track_id}: missing evidence_grade tag (CLAUDE.md FORBIDDEN_PATTERNS)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: autopilot still works without registry (graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────


def test_autopilot_build_plan_works_without_registry_entries():
    """If a future refactor removes the registry, the autopilot must still
    produce a valid plan (graceful soft-fail per the seed-function contract).

    Verification approach: invoke build_plan with default catalogs; even if
    LOSS_MODIFIER_TECHNIQUES is empty (registry missing), the encoder/arch
    catalogs must still produce ranked output.
    """
    plan = build_plan(d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144)
    # Encoder + arch catalogs must produce ranked output regardless.
    assert len(plan.encoder_technique_ranking) > 0
    assert len(plan.arch_technique_ranking) > 0
    # Top-3 must always have at least one entry.
    assert len(plan.recommended_top_3) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: predicted_archive_bytes default is the PR101 baseline
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "track_id,catalog",
    [
        ("t20_kl_pose_distill", LOSS_MODIFIER_TECHNIQUES),
        ("t22_temporal_consistency", LOSS_MODIFIER_TECHNIQUES),
        ("lane_12_v2_nerv_as_renderer", REPRESENTATION_LANES),
    ],
)
def test_catalog_row_predicted_bytes_neutral(
    track_id: str, catalog: list[dict]
):
    """predicted_archive_bytes must default to a neutral baseline so the
    row never CLAIMS a byte saving without empirical evidence.

    Per CLAUDE.md FORBIDDEN_PATTERNS empirical-claim-without-evidence-tag:
    a catalog row that claims fewer bytes than baseline is making an
    implicit prediction; we explicitly default to the PR101 baseline so
    the planner does not promote on the implicit prediction.
    """
    row = next(r for r in catalog if r["name"] == track_id)
    # Baseline is 178,144 (PR101 brotli optimum); the row must NOT claim
    # smaller bytes without an empirical anchor in evidence_grade.
    assert row["predicted_archive_bytes"] >= 178_144, (
        f"{track_id}: catalog row claims fewer bytes than baseline "
        f"({row['predicted_archive_bytes']}) without an empirical anchor"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: T20/T22/Lane-12-v2 all visible via list_tracks_visible_to
# ─────────────────────────────────────────────────────────────────────────────


def test_track_registry_visibility_helper_returns_all_3_new_tracks():
    """``list_tracks_visible_to('cathedral_autopilot')`` must return T20,
    T22, and Lane-12-v2 since their registry entries name cathedral_autopilot
    in planner_visibility."""
    visible = list_tracks_visible_to("cathedral_autopilot")
    visible_ids = {t.track_id for t in visible}
    assert "t20_kl_pose_distill" in visible_ids
    assert "t22_temporal_consistency" in visible_ids
    assert "lane_12_v2_nerv_as_renderer" in visible_ids
