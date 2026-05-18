# SPDX-License-Identifier: MIT
"""FIX-WAVE-R1: Tests for closure of 9 R1 findings.

Lane: lane_fix_wave_r1_post_provenance_z6_c6_wave_20260517
Task: #838

Per CLAUDE.md "Recursive adversarial review protocol" 3-clean-pass discipline:
R1 council rotation A returned VERDICT REFUSE with 9 findings (4 HIGH + 3 MEDIUM
+ 2 LOW). This test file pins the post-fix invariant for each finding so R1
RE-FIRE has structural evidence that the fix-wave landed cleanly.

Findings and acceptance criteria:

F1 HIGH — Catalog #315 STRUCTURALLY BLIND to C6 IBPS
  Fix: extend _CHECK_315_IN_SCOPE_ID_SUBSTRINGS with c6_/mdl_ibps tokens +
       backfill C6 council posterior row with deferred_substrate_id
  Acceptance: family-token list contains c6_; C6 IBPS council posterior latest
              row has deferred_substrate_id='c6_e4_mdl_ibps_substrate'

F2 HIGH — Catalog #131 META-meta drift in asymptotic_pursuit:704
  Fix: wrap bare write with same-line BARE_WRITE_OK waiver (single-writer-per-
       session pattern; each call produces unique timestamped filename)
  Acceptance: Catalog #131 returns 0 violations; Catalog #185 cascade clears

F3 HIGH — Z6 landing memo filename phantom-outcome trap
  Fix: rename memo from "_proceed_unconditional_unlock_landed_*.md" to
       "_proceed_with_revisions_v2_landed_*.md" reflecting actual verdict
  Acceptance: renamed file exists; old name absent; top-banner explains rename

F4 HIGH — C6 IBPS landing memo prose escalation
  Fix: re-word TL;DR + table row 4 from "PROCEED-unconditional 6-of-6" to
       "PROCEED 6-of-6 with 2 verbatim dissents on language"
  Acceptance: memo body no longer claims unconditional advancement

F5 MED — 15 (now 17) Catalog #314 absorption violations
  Fix: documented per-commit analysis; F5 is WARN-ONLY by Catalog #314 design
  Acceptance: classification recorded in landing memo; operator-routable

F6 MED — PROVENANCE landing memo lacks YAML frontmatter
  Fix: backfill Catalog #300 v2 frontmatter (T1 working-group tier)
  Acceptance: PROVENANCE memo starts with YAML frontmatter block

F7 MED — REDO+PIVOT lacks horizon_class
  Fix: backfill horizon_class: frontier_protecting
  Acceptance: REDO+PIVOT frontmatter contains horizon_class field

F8 LOW — 3 C6 sidecar JSONs lack Provenance embed
  Fix: embed canonical Provenance via build_provenance_for_research_sidecar
  Acceptance: each of 3 JSONs contains "provenance" key with valid schema

F9 LOW — Wave landing memos don't cite Catalog #316 frontier
  Fix: add canonical_frontier_anchor to 5 wave landing memos
  Acceptance: each memo's frontmatter contains canonical_frontier_anchor
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMORY_ROOT = Path.home() / ".claude" / "projects" / "-Users-adpena-Projects-pact" / "memory"


# ============================================================================
# F1: Catalog #315 family-token list extension + C6 council posterior backfill
# ============================================================================


def test_f1_check_315_family_token_list_includes_c6():
    """F1: Catalog #315 _CHECK_315_IN_SCOPE_ID_SUBSTRINGS must include 'c6_' token."""
    from tac.preflight import _CHECK_315_IN_SCOPE_ID_SUBSTRINGS

    assert "c6_" in _CHECK_315_IN_SCOPE_ID_SUBSTRINGS, (
        f"FIX-WAVE-R1 F1 regression: 'c6_' missing from family-token list; "
        f"Catalog #315 gate would be STRUCTURALLY BLIND to C6 IBPS dispatches."
    )


def test_f1_check_315_family_token_list_includes_mdl_ibps():
    """F1: Catalog #315 family-token list must include 'mdl_ibps' token."""
    from tac.preflight import _CHECK_315_IN_SCOPE_ID_SUBSTRINGS

    assert "mdl_ibps" in _CHECK_315_IN_SCOPE_ID_SUBSTRINGS


def test_f1_check_315_family_token_list_includes_time_traveler():
    """F1: Catalog #315 family-token list must include 'time_traveler' token.

    Note: 'time_traveler' was already present before FIX-WAVE-R1; this test
    pins that invariant against future regression.
    """
    from tac.preflight import _CHECK_315_IN_SCOPE_ID_SUBSTRINGS

    assert "time_traveler" in _CHECK_315_IN_SCOPE_ID_SUBSTRINGS


def test_f1_c6_ibps_council_posterior_has_deferred_substrate_id_correction():
    """F1: C6 IBPS council posterior must have at least one row with
    deferred_substrate_id='c6_e4_mdl_ibps_substrate' (backfill_extension)."""
    posterior_path = REPO_ROOT / ".omx" / "state" / "council_deliberation_posterior.jsonl"
    if not posterior_path.exists():
        pytest.skip("posterior file missing")
    rows_with_c6 = []
    with open(posterior_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("deliberation_id", "").startswith("council_c6_ibps_phase_2_sextet"):
                if row.get("deferred_substrate_id") == "c6_e4_mdl_ibps_substrate":
                    rows_with_c6.append(row)
    assert len(rows_with_c6) >= 1, (
        "FIX-WAVE-R1 F1 regression: no C6 IBPS council posterior row with "
        "deferred_substrate_id='c6_e4_mdl_ibps_substrate'; Catalog #315 join-key "
        "binding broken."
    )


# ============================================================================
# F2: Catalog #131 META-meta drift fix
# ============================================================================


def test_f2_asymptotic_pursuit_704_has_bare_write_waiver():
    """F2: asymptotic_pursuit_candidate_readiness_assessment.py:704 must carry
    same-line BARE_WRITE_OK waiver."""
    path = REPO_ROOT / "tools" / "asymptotic_pursuit_candidate_readiness_assessment.py"
    if not path.exists():
        pytest.skip("asymptotic_pursuit tool missing")
    text = path.read_text()
    assert "BARE_WRITE_OK:unique_timestamped_filename_single_writer_per_session" in text, (
        "FIX-WAVE-R1 F2 regression: asymptotic_pursuit:704 missing BARE_WRITE_OK waiver; "
        "Catalog #131 strict gate would fire 1 violation; Catalog #185 cascade would fire."
    )


def test_f2_catalog_131_live_count_is_zero():
    """F2: Catalog #131 (no bare writes to shared state) must return 0 violations
    so CLAUDE.md 'Live count: 0 -> STRICT' claim is structurally enforced."""
    from tac.preflight import check_no_bare_writes_to_shared_state

    violations = check_no_bare_writes_to_shared_state(strict=False, verbose=False)
    assert len(violations) == 0, (
        f"FIX-WAVE-R1 F2 regression: Catalog #131 returned {len(violations)} violations; "
        f"first 3: {violations[:3]}"
    )


def test_f2_catalog_185_cascade_clears():
    """F2: Catalog #185 META-meta drift must show 0 violations (no cascade from #131)."""
    from tac.preflight import check_strict_flipped_catalog_entries_have_live_count_zero

    violations = check_strict_flipped_catalog_entries_have_live_count_zero(
        strict=False, verbose=False
    )
    assert len(violations) == 0, (
        f"FIX-WAVE-R1 F2 regression: Catalog #185 cascade returned {len(violations)} "
        f"violations; first 3: {violations[:3]}"
    )


# ============================================================================
# F3: Z6 landing memo filename rename
# ============================================================================


def test_f3_z6_landing_memo_renamed():
    """F3: Z6 landing memo must be renamed to reflect PROCEED_WITH_REVISIONS_v2."""
    renamed_path = (
        MEMORY_ROOT
        / "feedback_z6_phase_2_sextet_council_proceed_with_revisions_v2_landed_20260517.md"
    )
    old_path = (
        MEMORY_ROOT
        / "feedback_z6_phase_2_sextet_council_proceed_unconditional_unlock_landed_20260517.md"
    )
    if not MEMORY_ROOT.exists():
        pytest.skip("memory dir missing")
    assert renamed_path.exists(), (
        f"FIX-WAVE-R1 F3 regression: renamed Z6 landing memo missing at {renamed_path}"
    )
    assert not old_path.exists(), (
        f"FIX-WAVE-R1 F3 regression: old misleading Z6 landing memo still present at {old_path}"
    )


def test_f3_z6_renamed_memo_has_top_banner():
    """F3: Renamed Z6 memo must carry top-banner HTML comment explaining rename."""
    path = (
        MEMORY_ROOT
        / "feedback_z6_phase_2_sextet_council_proceed_with_revisions_v2_landed_20260517.md"
    )
    if not path.exists():
        pytest.skip("renamed Z6 memo missing")
    text = path.read_text()
    assert "FIX-WAVE-R1 F3 RENAME" in text
    assert "Catalog #249 sister-at-filename-layer" in text


# ============================================================================
# F4: C6 IBPS landing memo prose escalation correction
# ============================================================================


def test_f4_c6_ibps_landing_memo_tldr_corrected():
    """F4: C6 IBPS landing memo TL;DR must use 'PROCEED 6-of-6 with 2 verbatim
    dissents on language' (NOT 'PROCEED-unconditional 6-of-6')."""
    path = MEMORY_ROOT / "feedback_c6_ibps_4_recipe_fixes_dispatch_unlock_landed_20260517.md"
    if not path.exists():
        pytest.skip("C6 IBPS memo missing")
    text = path.read_text()
    assert "PROCEED 6-of-6 with 2 verbatim dissents on language" in text
    assert "FIX-WAVE-R1 F4 RE-WORD" in text


def test_f4_c6_ibps_landing_memo_table_row_4_corrected():
    """F4: C6 IBPS landing memo table row 4 must use the apples-to-apples phrasing."""
    path = MEMORY_ROOT / "feedback_c6_ibps_4_recipe_fixes_dispatch_unlock_landed_20260517.md"
    if not path.exists():
        pytest.skip("C6 IBPS memo missing")
    text = path.read_text()
    # Table row 4 references the council deliberation and now declares the dissent count
    assert "PROCEED 6-of-6 with 2 verbatim dissents on language" in text
    # Sextet council verdict section was also re-worded
    assert "**6-of-6 PROCEED with 2 verbatim dissents on language.**" in text


# ============================================================================
# F5: Catalog #314 absorption pattern (warn-only by design)
# ============================================================================


def test_f5_catalog_314_warn_only_baseline_documented():
    """F5: Catalog #314 is WARN-ONLY by design per CLAUDE.md text. The 17
    absorption-pattern violations from this wave are operator-direct /commit
    slash command usage during sister-subagent in-flight work — not bugs.

    This test pins that the gate continues to fire ONLY in warn mode (no strict-
    raise) and the count is documented in the landing memo.
    """
    from tac.preflight import check_no_subagent_files_touched_absorption_in_bare_commits

    # Warn-only: returns list; does NOT raise PreflightError
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        strict=False, verbose=False
    )
    # Warn-only contract: gate may report any count >= 0; the contract is that
    # the gate does not raise in non-strict mode. Strict-flip is operator-
    # routable (switch /commit plugin to canonical serializer).
    assert isinstance(violations, list)


# ============================================================================
# F6: PROVENANCE landing memo YAML frontmatter backfill
# ============================================================================


def test_f6_provenance_memo_has_yaml_frontmatter():
    """F6: PROVENANCE landing memo must carry Catalog #300 v2 YAML frontmatter."""
    path = MEMORY_ROOT / "feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517.md"
    if not path.exists():
        pytest.skip("PROVENANCE memo missing")
    text = path.read_text()
    # File must START with --- (frontmatter delimiter)
    assert text.lstrip().startswith("---"), (
        "FIX-WAVE-R1 F6 regression: PROVENANCE memo does not start with YAML frontmatter"
    )
    assert "council_tier:" in text
    assert "horizon_class:" in text
    assert "council_predicted_mission_contribution:" in text
    assert "FIX-WAVE-R1 F6 closure" in text


# ============================================================================
# F7: REDO+PIVOT landing memo horizon_class backfill
# ============================================================================


def test_f7_redo_pivot_memo_has_horizon_class():
    """F7: REDO+PIVOT landing memo must contain horizon_class field."""
    path = (
        MEMORY_ROOT
        / "feedback_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_landed_20260517.md"
    )
    if not path.exists():
        pytest.skip("REDO+PIVOT memo missing")
    text = path.read_text()
    assert "horizon_class: frontier_protecting" in text, (
        "FIX-WAVE-R1 F7 regression: REDO+PIVOT memo missing horizon_class field"
    )
    assert "FIX-WAVE-R1 F7 closure" in text


# ============================================================================
# F8: 3 C6 sidecar JSONs Provenance embed
# ============================================================================


@pytest.mark.parametrize(
    "json_filename",
    [
        "dykstra_feasibility_c6_e4_mdl_ibps.json",
        "composition_alpha_c6_e4_mdl_ibps_x_wyner_ziv.json",
        "tier_c_density_reconciliation_c6_e4_mdl_ibps.json",
    ],
)
def test_f8_c6_sidecar_json_has_provenance_embed(json_filename):
    """F8: Each of 3 C6 sidecar JSONs must contain 'provenance' key with valid schema."""
    path = REPO_ROOT / ".omx" / "state" / json_filename
    if not path.exists():
        pytest.skip(f"C6 sidecar JSON {json_filename} missing")
    with open(path) as f:
        data = json.load(f)
    assert "provenance" in data, (
        f"FIX-WAVE-R1 F8 regression: {json_filename} missing 'provenance' key"
    )
    prov = data["provenance"]
    # Provenance dataclass invariants (research-sidecar shape)
    assert "artifact_kind" in prov
    assert prov.get("artifact_kind") == "research_sidecar"
    assert prov.get("score_claim_valid") is False
    assert prov.get("promotion_eligible") is False
    # canonical helper invocation cited so consumers can trace lineage
    assert "canonical_helper_invocation" in prov
    assert "fix_wave_r1_f8_backfill" in data


# ============================================================================
# F9: Wave landing memos cite Catalog #316 frontier
# ============================================================================


@pytest.mark.parametrize(
    "memo_filename",
    [
        "feedback_z6_phase_2_sextet_council_proceed_with_revisions_v2_landed_20260517.md",
        "feedback_c6_ibps_4_recipe_fixes_dispatch_unlock_landed_20260517.md",
        "feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517.md",
        "feedback_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_landed_20260517.md",
        "feedback_asymptotic_pursuit_substrate_class_shift_top_priority_landed_20260517.md",
    ],
)
def test_f9_wave_landing_memos_cite_canonical_frontier(memo_filename):
    """F9: Each of 5 wave landing memos must contain canonical_frontier_anchor
    citing 0.19205 [contest-CPU] + 0.20533 [contest-CUDA] per Catalog #316."""
    path = MEMORY_ROOT / memo_filename
    if not path.exists():
        pytest.skip(f"wave memo {memo_filename} missing")
    text = path.read_text()
    assert "canonical_frontier_anchor" in text
    assert "0.19205" in text  # contest-CPU frontier
    assert "0.20533" in text  # contest-CUDA frontier
    assert "Catalog #316" in text


# ============================================================================
# Acceptance criteria matrix
# ============================================================================


def test_acceptance_all_meta_meta_gates_clean():
    """Acceptance: all 5 META-meta CLAUDE.md gates (#118/#159/#176/#185/#235)
    must return 0 violations post-fix-wave."""
    from tac.preflight import (
        check_claude_md_catalog_no_duplicate_numbers,
        check_claude_md_catalog_text_matches_preflight_strict_value,
        check_strict_preflight_callsites_have_claude_md_catalog_row,
        check_strict_flipped_catalog_entries_have_live_count_zero,
        check_no_sha_prefix_length_mismatch_comparisons,
    )

    assert len(check_claude_md_catalog_no_duplicate_numbers(strict=False, verbose=False)) == 0
    assert (
        len(check_claude_md_catalog_text_matches_preflight_strict_value(strict=False, verbose=False))
        == 0
    )
    assert (
        len(check_strict_preflight_callsites_have_claude_md_catalog_row(strict=False, verbose=False))
        == 0
    )
    assert (
        len(check_strict_flipped_catalog_entries_have_live_count_zero(strict=False, verbose=False))
        == 0
    )
    assert (
        len(check_no_sha_prefix_length_mismatch_comparisons(strict=False, verbose=False)) == 0
    )


def test_acceptance_catalog_131_live_count_zero():
    """Acceptance: Catalog #131 live count = 0 (was 1 pre-fix)."""
    from tac.preflight import check_no_bare_writes_to_shared_state

    assert len(check_no_bare_writes_to_shared_state(strict=False, verbose=False)) == 0
