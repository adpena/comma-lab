# SPDX-License-Identifier: MIT
"""Tests for Catalog #315 — substrate at OPTIMAL FORM before paid
empirical dispatch.

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical
dispatch" non-negotiable. The structural failure this gate prevents:
empirically dispatching substrates at LIFTED-TRAINER form (basic
implementation that passes tests + has PR95-paradigm tokens) when the
sextet / grand council returned PROCEED_WITH_REVISIONS and the substrate
has not been iterated to OPTIMAL FORM.

Empirical anchors (this session 2026-05-17):
    - NSCS06 v6 -> v7 = 44% improvement (105.15 -> 58.89) via cargo-
      cult-unwind methodology (the ONLY substrate that got iterated to
      optimal form before next paid dispatch wave).
    - 4-of-5 distinguishing-feature dispatch failures (Wunderkind G1
      v2 reducer, ATW v2 D4 cooperative-receiver, Z6 FiLM ego-motion,
      NSCS01 nullspace-split, NSCS06 v8 Path B) tested at lifted-
      trainer form.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_315_build_council_verdict_map,
    _check_315_collect_lane_text,
    _check_315_extract_family_tokens,
    _check_315_lane_in_scope,
    _check_315_lane_opt_out,
    _check_315_lookup_latest_verdict,
    _check_315_parse_iso_utc,
    _check_315_substrate_ids_for_lane,
    _check_315_waiver_present,
    check_substrate_at_optimal_form_before_paid_dispatch as check_315,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_registry(
    tmp_path: Path, lanes: list[dict], schema_version: int = 1
) -> Path:
    repo_root = tmp_path
    state_dir = repo_root / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "lane_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "updated_at": "2026-05-17T00:00:00Z",
                "description": "test fixture",
                "gate_definitions": {},
                "level_rules": {},
                "lanes": lanes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return repo_root


def _write_posterior(
    repo_root: Path, rows: list[dict]
) -> Path:
    """Write a synthetic council_deliberation_posterior.jsonl."""
    state_dir = repo_root / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = state_dir / "council_deliberation_posterior.jsonl"
    posterior_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    return posterior_path


def _make_lane_l1_impl_complete(
    lane_id: str,
    notes: str = "",
    research_only: bool | None = None,
    lane_class: str | None = None,
    archived: bool | None = None,
    target_modes: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    lane: dict = {
        "id": lane_id,
        "name": lane_id,
        "phase": 2,
        "level": 1,
        "notes": notes,
        "gates": {
            "impl_complete": {"status": True, "evidence": "ok"},
            "real_archive_empirical": {"status": False, "evidence": ""},
            "contest_cuda": {"status": False, "evidence": ""},
            "strict_preflight": {"status": False, "evidence": ""},
            "three_clean_review": {"status": False, "evidence": ""},
            "memory_entry": {"status": False, "evidence": ""},
            "deploy_runbook": {"status": False, "evidence": ""},
        },
    }
    if research_only is not None:
        lane["research_only"] = research_only
    if lane_class is not None:
        lane["lane_class"] = lane_class
    if archived is not None:
        lane["archived"] = archived
    if target_modes is not None:
        lane["target_modes"] = target_modes
    if extra:
        lane.update(extra)
    return lane


def _make_council_row(
    deliberation_id: str,
    substrate_id: str,
    verdict: str,
    written_at_utc: str,
) -> dict:
    return {
        "deliberation_id": deliberation_id,
        "deferred_substrate_id": substrate_id,
        "council_verdict": verdict,
        "council_tier": "T2",
        "council_attendees": ["Shannon", "Dykstra", "Yousfi"],
        "council_quorum_met": True,
        "topic": f"Test deliberation {deliberation_id}",
        "written_at_utc": written_at_utc,
        "memory_path": f"feedback_{deliberation_id}.md",
        "event_type": "council_deliberation",
        "related_deliberation_ids": [],
    }


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lane_id,expected",
    [
        ("lane_substrate_pr101_lc_v2_clone_20260512", True),
        ("lane_d1_segnet_margin_polytope_encoder_20260514", True),
        ("lane_yucr_yousfi_uniward_cooperative_receiver_20260514", True),
        ("lane_a1_plus_lapose_20260512", True),
        ("lane_pr106_siren_residual_sidecar_20260513", True),
        ("lane_pr106_wavelet_residual_sidecar_20260513", True),
        ("lane_nscs03_end_to_end_balle_joint_codec_20260515", True),
        ("lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516", True),
        ("lane_wunderkind_g1_v2_pivot_validation_20260516", True),
        ("lane_atw_codec_v2_substrate_build_20260516", True),
        ("lane_c6_e4_mdl_ibps_substrate_20260514", True),
        ("lane_c6_ibps_4_recipe_fixes_dispatch_unlock_20260517", True),
        ("lane_z6_l1_scaffold", True),
        ("lane_z7_predictive_coding_world_model", True),
        ("lane_z8_hierarchical_pc_substrate_20260516", True),
        # Out-of-scope (infrastructure / META / catalog gates).
        ("lane_nscs01_phase_2_sextet_council_20260516", False),
        ("lane_fix_wave_r1_post_provenance_z6_c6_wave_20260517", False),
        ("lane_lane_registry_consistent", False),
        ("lane_meta_lagrangian_atom_emitter", False),
        ("lane_random_unrelated_lane", False),
        ("", False),
    ],
)
def test_in_scope_classifier(lane_id, expected):
    assert _check_315_lane_in_scope(lane_id) is expected


def test_parse_iso_utc_handles_z_suffix():
    ts = _check_315_parse_iso_utc("2026-05-17T12:00:00Z")
    assert ts is not None
    # Round-trip sanity check.
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.year == 2026 and dt.month == 5 and dt.day == 17


def test_parse_iso_utc_handles_offset():
    ts = _check_315_parse_iso_utc("2026-05-17T12:00:00+00:00")
    assert ts is not None


def test_parse_iso_utc_returns_none_on_garbage():
    assert _check_315_parse_iso_utc("not-a-date") is None
    assert _check_315_parse_iso_utc("") is None
    assert _check_315_parse_iso_utc(None) is None  # type: ignore[arg-type]
    assert _check_315_parse_iso_utc(12345) is None  # type: ignore[arg-type]


def test_collect_lane_text_includes_notes_and_evidence():
    lane = _make_lane_l1_impl_complete("lane_substrate_a", notes="ALPHA-NOTE")
    lane["gates"]["impl_complete"]["evidence"] = "BETA-EVIDENCE"
    text = _check_315_collect_lane_text(lane)
    assert "alpha-note" in text
    assert "beta-evidence" in text


def test_lane_opt_out_research_only_top_level():
    lane = _make_lane_l1_impl_complete("lane_substrate_a", research_only=True)
    text = _check_315_collect_lane_text(lane)
    assert _check_315_lane_opt_out(lane, text) == "research_only=true"


def test_lane_opt_out_substrate_engineering_top_level():
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_a", lane_class="substrate_engineering",
    )
    text = _check_315_collect_lane_text(lane)
    assert _check_315_lane_opt_out(lane, text) == (
        "lane_class=substrate_engineering"
    )


def test_lane_opt_out_archived_top_level():
    lane = _make_lane_l1_impl_complete("lane_substrate_a", archived=True)
    text = _check_315_collect_lane_text(lane)
    assert _check_315_lane_opt_out(lane, text) == "archived=true"


def test_lane_opt_out_target_modes_research_substrate():
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_a", target_modes=["research_substrate"],
    )
    text = _check_315_collect_lane_text(lane)
    assert _check_315_lane_opt_out(lane, text) == "target_modes=research_substrate"


def test_lane_opt_out_research_only_in_notes():
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_a", notes="research_only=true pending iteration"
    )
    text = _check_315_collect_lane_text(lane)
    assert _check_315_lane_opt_out(lane, text) == "research_only=true"


def test_lane_no_opt_out_returns_none():
    lane = _make_lane_l1_impl_complete("lane_substrate_a")
    text = _check_315_collect_lane_text(lane)
    assert _check_315_lane_opt_out(lane, text) is None


def test_waiver_present_with_real_rationale():
    text = (
        "lane notes... # OPTIMAL_FORM_DISPATCH_OK:operator-routable cited"
    )
    assert _check_315_waiver_present(text) is True


def test_waiver_placeholder_rationale_rejected():
    for placeholder in ("<rationale>", "<reason>", ""):
        text = f"notes... # OPTIMAL_FORM_DISPATCH_OK:{placeholder}"
        assert _check_315_waiver_present(text) is False


def test_waiver_no_marker_returns_false():
    assert _check_315_waiver_present("just plain notes") is False


def test_substrate_ids_for_lane_includes_lane_id_and_alias():
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6",
        extra={"substrate_alias": "z6_v1_ego_conditioning_surface"},
    )
    ids = _check_315_substrate_ids_for_lane(lane)
    assert "lane_substrate_z6" in ids
    assert "z6_v1_ego_conditioning_surface" in ids


def test_substrate_ids_for_lane_supports_aliases_list():
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_a",
        extra={"substrate_aliases": ["alias_a", "alias_b"]},
    )
    ids = _check_315_substrate_ids_for_lane(lane)
    assert "alias_a" in ids
    assert "alias_b" in ids


def test_extract_family_tokens_is_separator_aware():
    assert "z3_g1" in _check_315_extract_family_tokens("lane_z3_g1_entropy")
    assert "z3_g1" not in _check_315_extract_family_tokens("lane_z30_future")
    assert "nscs03" in _check_315_extract_family_tokens(
        "nscs03_end_to_end_balle_joint_codec_v1_surface"
    )
    assert "z6" not in _check_315_extract_family_tokens(
        "lane_time_traveler_l5_autonomy_substrate_20260513"
    )


def test_build_council_verdict_map_latest_wins(tmp_path):
    repo_root = _write_registry(tmp_path, [])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "older_deliberation",
                "substrate_a",
                "PROCEED_WITH_REVISIONS",
                "2026-05-15T10:00:00Z",
            ),
            _make_council_row(
                "newer_deliberation",
                "substrate_a",
                "PROCEED",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)
    assert "substrate_a" in vmap
    assert vmap["substrate_a"]["deliberation_id"] == "newer_deliberation"
    assert vmap["substrate_a"]["council_verdict"] == "PROCEED"


def test_build_council_verdict_map_skips_rows_without_substrate_id(tmp_path):
    repo_root = _write_registry(tmp_path, [])
    posterior_path = _write_posterior(
        repo_root,
        [
            {
                "deliberation_id": "no_substrate",
                "council_verdict": "PROCEED_WITH_REVISIONS",
                "written_at_utc": "2026-05-15T10:00:00Z",
            },
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)
    assert vmap == {}


def test_lookup_latest_verdict_returns_none_when_no_anchor(tmp_path):
    lane = _make_lane_l1_impl_complete("lane_substrate_a")
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "other_deliberation",
                "different_substrate",
                "PROCEED",
                "2026-05-15T10:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)
    assert _check_315_lookup_latest_verdict(lane, vmap) is None


def test_lookup_latest_verdict_matches_family_token_surface_id(tmp_path):
    lane = _make_lane_l1_impl_complete("lane_nscs01_nullspace_split_renderer_20260515")
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "nscs01_surface_deliberation",
                "nscs01_nullspace_split_renderer_v1_head0_capacity_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)

    verdict = _check_315_lookup_latest_verdict(lane, vmap)

    assert verdict is not None
    assert verdict["deliberation_id"] == "nscs01_surface_deliberation"


def test_lookup_latest_verdict_does_not_false_match_prefix_family_token(tmp_path):
    lane = _make_lane_l1_impl_complete("lane_z30_future_surface_20260517")
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "z3_deliberation",
                "z3_g1_entropy_coded_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)

    assert _check_315_lookup_latest_verdict(lane, vmap) is None


def test_lookup_latest_verdict_does_not_false_match_z3_balle_to_z3_g1(tmp_path):
    lane = _make_lane_l1_impl_complete(
        "lane_z3_balle_hyperprior_bolton_recover_20260514"
    )
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "z3_g1_deliberation",
                "lane_z3_g1_entropy_coded_v2_20260515",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)

    assert _check_315_lookup_latest_verdict(lane, vmap) is None


def test_lookup_latest_verdict_does_not_false_match_generic_time_traveler_to_z6(
    tmp_path,
):
    lane = _make_lane_l1_impl_complete(
        "lane_time_traveler_l5_autonomy_substrate_20260513"
    )
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "z6_deliberation",
                "time_traveler_l5_z6_v1_ego_conditioning_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)

    assert _check_315_lookup_latest_verdict(lane, vmap) is None


# ---------------------------------------------------------------------------
# End-to-end gate behavior
# ---------------------------------------------------------------------------


def test_gate_passes_when_no_registry(tmp_path):
    """No lane_registry.json -> gate returns no violations."""
    repo_root = tmp_path
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_passes_when_no_posterior(tmp_path):
    """Registry present but no council posterior -> gate returns []."""
    repo_root = _write_registry(
        tmp_path,
        [_make_lane_l1_impl_complete("lane_substrate_a")],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_flags_lifted_trainer_with_proceed_with_revisions(tmp_path):
    """Canonical bug-class anchor: a lane with PROCEED_WITH_REVISIONS
    as the latest verdict and no opt-out MUST be flagged."""
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_council_z6_phase_2_consensus_20260516",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1
    assert "lane_substrate_z6" in violations[0]
    assert "PROCEED_WITH_REVISIONS" in violations[0]
    assert "OPTIMAL FORM" in violations[0]


def test_gate_passes_with_proceed_unconditional_after_revisions(tmp_path):
    """Iteration anchor: PROCEED-unconditional chronologically AFTER
    PROCEED_WITH_REVISIONS supersedes it."""
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "older_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
            _make_council_row(
                "newer_unconditional",
                "lane_substrate_z6",
                "PROCEED",
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_flags_when_revisions_chronologically_later(tmp_path):
    """If PROCEED_WITH_REVISIONS is the LATEST anchor (e.g., a re-
    review surfaced new issues), the gate fires even if an older
    PROCEED-unconditional exists."""
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "older_unconditional",
                "lane_substrate_z6",
                "PROCEED",
                "2026-05-15T10:00:00Z",
            ),
            _make_council_row(
                "newer_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1


def test_gate_accepts_research_only_opt_out(tmp_path):
    """research_only=true opts the lane out of dispatch discipline."""
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", research_only=True,
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_accepts_substrate_engineering_opt_out(tmp_path):
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", lane_class="substrate_engineering",
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_accepts_archived_opt_out(tmp_path):
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", archived=True,
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_accepts_target_modes_research_substrate(tmp_path):
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", target_modes=["research_substrate"],
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_accepts_waiver_with_real_rationale(tmp_path):
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6",
        notes=(
            "# OPTIMAL_FORM_DISPATCH_OK:operator-approved-2026-05-17 "
            "smoke-only $1 disambiguator dispatch"
        ),
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_rejects_placeholder_waiver(tmp_path):
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6",
        notes="# OPTIMAL_FORM_DISPATCH_OK:<rationale>",
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1


def test_gate_passes_when_no_council_anchor_for_lane(tmp_path):
    """Lanes WITHOUT a council deliberation entry are out of scope for
    this specific gate (covered by sister gates #233 / #294 / #298)."""
    lane = _make_lane_l1_impl_complete("lane_substrate_no_council")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "different_substrate_deliberation",
                "lane_substrate_different",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_strict_mode_does_not_turn_missing_anchor_into_revision_blocker(
    tmp_path,
):
    """Strict Catalog #315 binds known council verdicts; it does not
    retroactively require council rows for every historical substrate lane.
    """
    lane = _make_lane_l1_impl_complete("lane_substrate_no_council")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "different_substrate_deliberation",
                "lane_substrate_different",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=True, verbose=False)
    assert violations == []


def test_gate_skips_l0_lanes(tmp_path):
    """L0 lanes have no impl yet; out of scope."""
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    lane["level"] = 0
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_skips_lanes_without_impl_complete(tmp_path):
    """Lanes without impl_complete=true have not yet landed code."""
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    lane["gates"]["impl_complete"]["status"] = False
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_skips_out_of_scope_lanes(tmp_path):
    """Non-substrate lanes are out of scope."""
    lane = _make_lane_l1_impl_complete("lane_lane_registry_consistent")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "any_deliberation",
                "lane_lane_registry_consistent",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_strict_mode_raises_preflight_error(tmp_path):
    """Strict mode raises PreflightError with Catalog #315 message."""
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    with pytest.raises(PreflightError) as exc_info:
        check_315(repo_root=repo_root, strict=True, verbose=False)
    assert "OPTIMAL FORM" in str(exc_info.value)
    assert "PROCEED_WITH_REVISIONS" in str(exc_info.value) or "315" in str(
        exc_info.value
    )


def test_gate_strict_mode_silent_on_clean(tmp_path):
    """Strict mode does NOT raise when there are no violations."""
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", research_only=True,
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    # Should not raise.
    violations = check_315(repo_root=repo_root, strict=True, verbose=False)
    assert violations == []


def test_gate_aggregates_multiple_violations(tmp_path):
    """Multiple lanes with PROCEED_WITH_REVISIONS each produce one
    violation."""
    lanes = [
        _make_lane_l1_impl_complete("lane_substrate_a"),
        _make_lane_l1_impl_complete("lane_substrate_b"),
        _make_lane_l1_impl_complete("lane_substrate_c"),
    ]
    repo_root = _write_registry(tmp_path, lanes)
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                f"deliberation_{x}",
                f"lane_substrate_{x}",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            )
            for x in ("a", "b", "c")
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 3


def test_gate_handles_corrupt_registry(tmp_path):
    """Corrupt registry returns empty list (does not raise)."""
    repo_root = tmp_path
    state_dir = repo_root / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "lane_registry.json").write_text(
        "this is not json", encoding="utf-8"
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == []


def test_gate_handles_corrupt_posterior(tmp_path):
    """Corrupt posterior lines are skipped without crashing."""
    repo_root = _write_registry(
        tmp_path,
        [_make_lane_l1_impl_complete("lane_substrate_z6")],
    )
    posterior_path = (
        repo_root / ".omx" / "state" / "council_deliberation_posterior.jsonl"
    )
    posterior_path.write_text(
        "this is not json\n"
        + json.dumps(
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    # Even with the corrupt first line, the valid second line still
    # registers the verdict and the gate fires on the lane.
    assert len(violations) == 1


def test_gate_accepts_string_repo_root(tmp_path):
    """repo_root may be passed as a string; should be coerced to Path."""
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", research_only=True,
    )
    _write_registry(tmp_path, [lane])
    violations = check_315(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert violations == []


def test_gate_verbose_output_runs_cleanly(tmp_path, capsys):
    """Verbose mode prints per-lane diagnostic without crashing."""
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6", research_only=True,
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "lane_substrate_z6",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    check_315(repo_root=repo_root, strict=False, verbose=True)
    captured = capsys.readouterr()
    assert "catalog-315" in captured.out
    assert "lane_substrate_z6" in captured.out


def test_gate_alias_lookup_resolves_substrate_via_alias_field(tmp_path):
    """A lane that declares ``substrate_alias`` should resolve council
    rows that use the alias as the substrate_id (matching the canonical
    sextet anchors which use a v1-surface name)."""
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_z6",
        extra={"substrate_alias": "z6_v1_ego_conditioning_surface"},
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "sextet_revisions",
                "z6_v1_ego_conditioning_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1
    assert "lane_substrate_z6" in violations[0]


def test_gate_defer_verdict_treated_as_no_positive_proceed_blocker(tmp_path):
    """DEFER / REFUSE / ESCALATE are not paid-dispatch authorization.

    Catalog #315 v2 requires a positive PROCEED anchor, not merely the
    absence of PROCEED_WITH_REVISIONS.
    """
    lane = _make_lane_l1_impl_complete("lane_substrate_z6")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "defer_deliberation",
                "lane_substrate_z6",
                "DEFER_PENDING_EVIDENCE",
                "2026-05-16T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1
    assert "DEFER_PENDING_EVIDENCE" in violations[0]
    assert "dormant_pending_redeliberation" in violations[0]


def test_gate_live_repo_regression_guard():
    """Live-repo regression: unexpected Catalog #315 blockers fail.

    The current dirty-tree snapshot intentionally carries one Z6-v2
    PROCEED_WITH_REVISIONS blocker. That is allowed here so this test
    can still catch the FIX-WAVE-R1 C6 false-authority regression and
    any accidental broad family-token bleed into older time_traveler
    lanes.
    """
    violations = check_315(strict=False, verbose=False)
    allowed = ("lane_z6_v2_redesign_cargo_cult_unwind_path_b_20260517",)
    unexpected = [v for v in violations if not any(tok in v for tok in allowed)]
    assert unexpected == [], (
        f"Catalog #315 has unexpected live blocker(s): {unexpected[:3]}"
    )
    assert not any("c6_e4_mdl_ibps" in v for v in violations), (
        "FIX-WAVE-R1 F1 regression: C6 should resolve to its positive "
        "council row instead of surfacing as a Catalog #315 blocker"
    )


def test_orchestrator_wire_in_uses_strict_true():
    """preflight_all() must call check_315 with strict=True because the
    live count is 0 at landing.
    """
    preflight_path = Path(__file__).resolve().parents[1] / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")
    # Find the orchestrator-callsite block.
    needle = "check_substrate_at_optimal_form_before_paid_dispatch("
    idx = source.find(needle)
    assert idx > 0, "orchestrator callsite for Catalog #315 not found"
    # The call signature spans the next ~3 lines; look for strict=True.
    window = source[idx : idx + 200]
    assert "strict=True" in window, (
        "Catalog #315 must stay strict once live count is verified at 0; "
        f"got window: {window}"
    )


# ---------------------------------------------------------------------------
# v2 (codex review fix 2026-05-17) regression tests using LIVE deferred IDs
#
# Per codex's explicit recommendation:
#     "Add regression tests using the live deferred ids, not synthetic
#      ids equal to lane ids."
#
# The 5 substrate IDs below are taken VERBATIM from
# ``.omx/state/council_deliberation_posterior.jsonl`` as of 2026-05-17.
# Each ``deferred_substrate_id`` differs from its registry lane ID,
# making them the canonical witness set that the family-token fuzzy
# fallback MUST resolve. Synthetic IDs equal to lane IDs accidentally
# pass via the Pass-1 exact-ID join and silently hide regressions in
# Pass-2.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lane_id,live_surface_id,expected_token",
    [
        (
            "lane_nscs01_nullspace_split_renderer_20260515",
            "nscs01_nullspace_split_renderer_v1_head0_capacity_surface",
            "nscs01",
        ),
        (
            "lane_nscs03_end_to_end_balle_joint_codec_20260515",
            "nscs03_end_to_end_balle_joint_codec_v1_default_hyperparameter_configuration",
            "nscs03",
        ),
        (
            "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516",
            "time_traveler_l5_z6_v1_ego_conditioning_surface",
            "z6",
        ),
        # NOTE: z3-family lanes are intentionally NOT in Catalog #315's
        # in-scope id-substring set today (no `z3_` / `z4_` / `z5_`
        # entry; only `z6_` / `z7_` / `z8_`). The Z3 G1 council row
        # exists in the live posterior but the gate scope filter is
        # ground truth.
        (
            "lane_tishby_ib_pure_substrate_20260515",
            "tishby_ib_pure_substrate",
            "tishby_ib_pure",
        ),
        (
            "lane_c6_e4_mdl_ibps_substrate_20260514",
            "c6_e4_mdl_ibps_substrate",
            "c6_e4_mdl_ibps",
        ),
    ],
)
def test_family_join_resolves_live_deferred_substrate_ids(
    tmp_path, lane_id, live_surface_id, expected_token
):
    """v2 regression: every LIVE deferred_substrate_id from the council
    posterior MUST resolve to its registry lane via family-token fuzzy
    match (Pass 2). If this test fails, the family-token table has
    drifted out of sync with the live posterior."""
    lane = _make_lane_l1_impl_complete(lane_id)
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                f"live_council_deliberation_{expected_token}",
                live_surface_id,
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T12:00:00Z",
            ),
        ],
    )
    vmap = _check_315_build_council_verdict_map(posterior_path)
    verdict = _check_315_lookup_latest_verdict(lane, vmap)

    assert verdict is not None, (
        f"live deferred substrate id {live_surface_id!r} did NOT resolve "
        f"to lane {lane_id!r}; family token {expected_token!r} should "
        "be in both ID strings — check _CHECK_315_SUBSTRATE_FAMILY_TOKENS"
    )
    assert verdict["deferred_substrate_id"] == live_surface_id
    assert verdict["council_verdict"] == "PROCEED_WITH_REVISIONS"

    # End-to-end: gate must FLAG (not silently pass) when council
    # returned PROCEED_WITH_REVISIONS on the lane.
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1
    assert lane_id in violations[0]
    assert "PROCEED_WITH_REVISIONS" in violations[0]


def test_null_deferred_substrate_id_backfill_resolves_c6_council_row(tmp_path):
    """FIX-WAVE-R1 F1 regression: the live C6 IBPS council row was
    emitted with deferred_substrate_id=None. Catalog #315 must still
    join that immutable deliberation_id back to the C6 lane instead of
    treating the active dispatch lane as no_council_anchor."""
    lane = _make_lane_l1_impl_complete(
        "lane_c6_e4_mdl_ibps_substrate_20260514"
    )
    repo_root = _write_registry(tmp_path, [lane])
    posterior_path = _write_posterior(
        repo_root,
        [
            _make_council_row(
                "council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517",
                None,
                "PROCEED",
                "2026-05-17T22:53:53Z",
            ),
        ],
    )

    vmap = _check_315_build_council_verdict_map(posterior_path)
    assert "c6_e4_mdl_ibps_substrate" in vmap
    verdict = _check_315_lookup_latest_verdict(lane, vmap)
    assert verdict is not None
    assert verdict["deliberation_id"] == (
        "council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517"
    )
    assert verdict["council_verdict"] == "PROCEED"

    assert check_315(repo_root=repo_root, strict=False, verbose=False) == []


def test_null_deferred_substrate_id_backfill_blocks_c6_revisions(tmp_path):
    """The C6 backfill is not just a positive-pass shim: if the same
    malformed row carries a non-PROCEED verdict, Catalog #315 must
    block the C6 lane exactly like a normal deferred_substrate_id row."""
    lane = _make_lane_l1_impl_complete(
        "lane_c6_e4_mdl_ibps_substrate_20260514"
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517",
                None,
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T22:53:53Z",
            ),
        ],
    )

    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1
    assert "lane_c6_e4_mdl_ibps_substrate_20260514" in violations[0]
    assert "PROCEED_WITH_REVISIONS" in violations[0]


def test_family_join_resolves_all_live_deferred_ids_simultaneously(tmp_path):
    """v2 regression: synthesize the multi-lane scenario from one
    snapshot of the live posterior. Verifies cross-lane interference
    (no family token bleeds across families) AND that all in-scope
    lanes flag together when none have opt-out."""
    lanes = [
        _make_lane_l1_impl_complete("lane_nscs01_nullspace_split_renderer_20260515"),
        _make_lane_l1_impl_complete(
            "lane_nscs03_end_to_end_balle_joint_codec_20260515"
        ),
        _make_lane_l1_impl_complete(
            "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
        ),
        _make_lane_l1_impl_complete("lane_tishby_ib_pure_substrate_20260515"),
    ]
    repo_root = _write_registry(tmp_path, lanes)
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "live_nscs01",
                "nscs01_nullspace_split_renderer_v1_head0_capacity_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T11:50:25Z",
            ),
            _make_council_row(
                "live_nscs03",
                "nscs03_end_to_end_balle_joint_codec_v1_default_hyperparameter_configuration",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T11:49:16Z",
            ),
            _make_council_row(
                "live_z6",
                "time_traveler_l5_z6_v1_ego_conditioning_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T02:49:25Z",
            ),
            _make_council_row(
                "live_tishby",
                "tishby_ib_pure_substrate",
                "PROCEED_WITH_REVISIONS",
                "2026-05-16T22:30:13Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 4, (
        f"expected 4 violations (one per in-scope lifted-trainer-form "
        f"substrate) but got {len(violations)}: {violations}"
    )
    flagged_lanes = {v.split("'")[1] for v in violations}
    assert flagged_lanes == {
        "lane_nscs01_nullspace_split_renderer_20260515",
        "lane_nscs03_end_to_end_balle_joint_codec_20260515",
        "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516",
        "lane_tishby_ib_pure_substrate_20260515",
    }


def test_iteration_to_optimal_form_supersedes_live_revisions(tmp_path):
    """v2 regression: chronologically LATER PROCEED anchor MUST
    supersede the live PROCEED_WITH_REVISIONS — even when the later
    anchor uses a different surface form than the earlier one. Models
    the canonical iteration path: lifted form deferred -> cargo-cult
    unwind landed -> sextet re-deliberates PROCEED."""
    lane = _make_lane_l1_impl_complete(
        "lane_nscs01_nullspace_split_renderer_20260515"
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "older_revisions_live",
                "nscs01_nullspace_split_renderer_v1_head0_capacity_surface",
                "PROCEED_WITH_REVISIONS",
                "2026-05-17T11:50:25Z",
            ),
            _make_council_row(
                "newer_optimal_form_v2",
                "nscs01_nullspace_split_renderer_v2_optimal_form_surface",
                "PROCEED",
                "2026-05-17T20:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert violations == [], (
        "iteration to optimal form (later PROCEED) must supersede the "
        f"earlier PROCEED_WITH_REVISIONS; got: {violations}"
    )


def test_strict_mode_fails_closed_on_missing_registry(tmp_path):
    """v2 fail-closed regression: strict mode MUST raise PreflightError
    when lane_registry.json is missing. Codex CRITICAL: v1 silently
    returned [] (fail-open) on missing control-plane state."""
    with pytest.raises(PreflightError, match="lane_registry.json is MISSING"):
        check_315(repo_root=tmp_path, strict=True, verbose=False)


def test_strict_mode_fails_closed_on_missing_posterior(tmp_path):
    """v2 fail-closed regression: strict mode MUST raise PreflightError
    when posterior is missing (registry present)."""
    _write_registry(
        tmp_path,
        [_make_lane_l1_impl_complete("lane_substrate_test")],
    )
    with pytest.raises(
        PreflightError, match="council_deliberation_posterior.jsonl is MISSING"
    ):
        check_315(repo_root=tmp_path, strict=True, verbose=False)


def test_strict_mode_fails_closed_on_unreadable_registry(tmp_path):
    """v2 fail-closed regression: strict mode MUST raise on corrupt
    registry JSON. Recovery / partial checkout / JSON corruption are
    the empirical attack surfaces this catches."""
    repo_root = tmp_path
    state_dir = repo_root / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "lane_registry.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    with pytest.raises(PreflightError, match="UNREADABLE/CORRUPT"):
        check_315(repo_root=repo_root, strict=True, verbose=False)


def test_strict_mode_fails_closed_on_empty_verdict_map(tmp_path):
    """v2 fail-closed regression: strict mode MUST raise when the
    posterior file exists but contains no rows with
    deferred_substrate_id set (so verdict_map is empty). v1 silently
    fail-opened here."""
    repo_root = _write_registry(
        tmp_path,
        [_make_lane_l1_impl_complete("lane_substrate_test")],
    )
    state_dir = repo_root / ".omx" / "state"
    posterior_path = state_dir / "council_deliberation_posterior.jsonl"
    posterior_path.write_text(
        json.dumps(
            {
                "deliberation_id": "no_substrate_id_set",
                "deferred_substrate_id": None,
                "council_verdict": "PROCEED_WITH_REVISIONS",
                "council_tier": "T3",
                "council_attendees": ["Shannon"],
                "council_quorum_met": True,
                "topic": "no substrate id",
                "written_at_utc": "2026-05-16T15:35:37+00:00",
                "memory_path": "feedback_test.md",
                "event_type": "council_deliberation",
                "related_deliberation_ids": [],
            },
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="no deferred_substrate_id anchors"):
        check_315(repo_root=repo_root, strict=True, verbose=False)


def test_strict_mode_does_not_turn_missing_anchor_into_revision_blocker_v2(
    tmp_path,
):
    """Catalog #315 is the council-verdict-binding surface, not a broad
    migration gate for historical lanes that have no joinable council row.
    """
    lane = _make_lane_l1_impl_complete(
        "lane_substrate_unknown_no_anchor"
    )
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "unrelated_council",
                "different_family_unrelated_substrate",
                "PROCEED",
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=True, verbose=False)
    assert violations == []


@pytest.mark.parametrize(
    "non_proceed_verdict",
    [
        "DEFER_PENDING_EVIDENCE",
        "REFUSE",
        "ESCALATE_TO_HIGHER_TIER",
        "ESCALATE_TO_OPERATOR",
        "UNKNOWN_NEW_VERDICT_FORM",
        "",
    ],
)
def test_positive_proceed_requirement_rejects_non_proceed_verdicts(
    tmp_path, non_proceed_verdict
):
    """v2 POSITIVE-PROCEED regression: only PROCEED / PROCEED_UNCONDITIONAL
    pass. Codex HIGH: v1 used negative test
    ``verdict != 'PROCEED_WITH_REVISIONS'`` which silently accepted
    DEFER / REFUSE / ESCALATE / missing / unknown as dispatch-safe.
    The acceptance criterion is positive authorization."""
    lane = _make_lane_l1_impl_complete("lane_substrate_test")
    repo_root = _write_registry(tmp_path, [lane])
    _write_posterior(
        repo_root,
        [
            _make_council_row(
                "non_proceed_test",
                "lane_substrate_test",
                non_proceed_verdict,
                "2026-05-17T10:00:00Z",
            ),
        ],
    )
    violations = check_315(repo_root=repo_root, strict=False, verbose=False)
    assert len(violations) == 1, (
        f"non-PROCEED verdict {non_proceed_verdict!r} silently accepted; "
        f"violations={violations}"
    )
    assert "lane_substrate_test" in violations[0]
