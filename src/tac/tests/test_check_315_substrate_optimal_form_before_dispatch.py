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
        ("lane_z6_l1_scaffold", True),
        ("lane_z7_predictive_coding_world_model", True),
        ("lane_z8_hierarchical_pc_substrate_20260516", True),
        # Out-of-scope (infrastructure / META / catalog gates).
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


def test_gate_defer_verdict_treated_as_dormant_not_flagged(tmp_path):
    """DEFER / REFUSE / ESCALATE verdicts are NOT flagged (the lane is
    dormant pending re-deliberation; sister gates cover this)."""
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
    assert violations == []


def test_gate_live_repo_regression_guard():
    """Live-repo regression: this gate's live count must stay 0.

    If this test ever fails, a new lifted-trainer-form substrate landed
    without the corresponding iteration / opt-out / waiver. The
    remediation is documented in the gate's violation message.
    """
    violations = check_315(strict=False, verbose=False)
    assert violations == [], (
        f"Catalog #315 live count regressed to {len(violations)}; "
        "first violation: " + (violations[0] if violations else "")
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
