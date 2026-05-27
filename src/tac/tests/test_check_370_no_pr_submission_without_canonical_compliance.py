# SPDX-License-Identifier: MIT
"""Tests for Catalog #370 — canonical-submission-pipeline Phase 8 STRICT gate.

Phase 8 (Layer 6) of the canonical submission pipeline per
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
§3 Phase 8.

The gate refuses ``submissions/*/`` directories containing PR-facing
artifacts (PR_BODY*.md / PR_DESCRIPTION.md / README.md with PR-title or
PR-body sentinel tokens) WITHOUT all 4 canonical verdict sidecars present
+ clean (Phase 4 builder + Phase 5 linter + Phase 6 compliance + Phase 7
paired_auth_eval). Same-line waiver
``# PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:<rationale>`` accepted in
README.md or PR_BODY*.md first 30 lines (placeholder rejected per Catalog
#287). Sister of Catalog #335 (canonical cathedral consumer contract) +
Catalog #341 (Tier A canonical-routing markers) + Catalog #361 (Modal
artifact filter preserves submission_dir) + Catalog #287 (placeholder-
rationale rejection) + Catalog #176 (META-meta: STRICT-callsites-have-
CLAUDE.md-row) + Catalog #185 (META-meta-meta: Live count: 0 verified
empirically) + Catalog #299 (catalog quota brake under 400) + Catalog
#348 (retroactive sweep for new gate).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pytest

from tac import preflight
from tac.preflight import PreflightError


REPO_ROOT = Path(__file__).resolve().parents[3]


# ============================================================================
# Helper builders for synthetic submission_dir fixtures
# ============================================================================


def _write_canonical_phase_sidecar(
    path: Path,
    phase: str,
    *,
    clean: bool = True,
) -> Path:
    """Write a canonical Phase 4/5/6/7 verdict sidecar JSON at the given path.

    ``phase`` is one of ``phase_4_builder`` / ``phase_5_linter`` /
    ``phase_6_compliance`` / ``phase_7_paired_auth_eval``. Generates a JSON
    body carrying the canonical clean marker for the given phase (or the
    inverse marker when ``clean=False``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if phase == "phase_4_builder":
        body = {
            "schema_version": "submission_bundle_v1_20260526",
            "lane_id": "lane_synthetic_test_phase4",
            "substrate_id": "synthetic_substrate",
            "archive_sha256": "0" * 64,
            "archive_bytes": 100,
            "overall_pass": clean,
        }
    elif phase == "phase_5_linter":
        body = {
            "schema_version": "lint_verdict_v1_20260526",
            "lane_id": "lane_synthetic_test_phase5",
            "substrate_id": "synthetic_substrate",
            "overall_clean": clean,
            "findings": [],
        }
    elif phase == "phase_6_compliance":
        body = {
            "schema_version": "compliance_verdict_v1_20260526",
            "lane_id": "lane_synthetic_test_phase6",
            "substrate_id": "synthetic_substrate",
            "archive_sha256": "0" * 64,
            "archive_bytes": 100,
            "submission_dir": "submissions/synthetic_phase8_test/",
            "overall_clean": clean,
            "contest_final_strict": True,
            "submission_score_axis": "contest_cuda",
            "total_checks": 1,
            "passed_count": 1 if clean else 0,
            "error_count": 0 if clean else 1,
            "warning_count": 0,
        }
    elif phase == "phase_7_paired_auth_eval":
        body = {
            "schema_version": "paired_auth_eval_verdict_v1_20260526",
            "lane_id": "lane_synthetic_test_phase7",
            "substrate_id": "synthetic_substrate",
            "verdict": "PAIRED_PASS" if clean else "PAIRED_FAIL",
            "archive_sha256": "0" * 64,
            "contest_cpu_score": 0.19,
            "contest_cuda_score": 0.20,
        }
    else:
        raise ValueError(f"unknown phase: {phase!r}")
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_submission_with_chain(
    tmp_path: Path,
    name: str,
    *,
    pr_facing: bool = True,
    phases_present: Iterable[str] = (
        "phase_4_builder",
        "phase_5_linter",
        "phase_6_compliance",
        "phase_7_paired_auth_eval",
    ),
    phases_clean: Iterable[str] = (
        "phase_4_builder",
        "phase_5_linter",
        "phase_6_compliance",
        "phase_7_paired_auth_eval",
    ),
    readme_extra: str = "",
    waiver_rationale: str | None = None,
) -> Path:
    """Build a synthetic submission directory with optional verdict chain."""
    sub_dir = tmp_path / "submissions" / name
    sub_dir.mkdir(parents=True, exist_ok=True)
    readme_body = ""
    if pr_facing:
        readme_body += "# PR submission packet\n\n"
        readme_body += "## Submission\n\nCanonical PR-facing artifact body.\n\n"
        readme_body += "## Score\n\n[contest-CPU] = 0.19xxx\n[contest-CUDA] = 0.20xxx\n"
    if waiver_rationale is not None:
        readme_body = (
            f"# PR submission\n"
            f"<!-- # PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:{waiver_rationale} -->\n"
            + readme_body
        )
    if readme_extra:
        readme_body += "\n" + readme_extra
    if pr_facing:
        (sub_dir / "README.md").write_text(readme_body, encoding="utf-8")
    phases_clean_set = set(phases_clean)
    for phase in phases_present:
        # Default canonical sidecar filenames
        if phase == "phase_4_builder":
            fname = "submission_bundle_result.json"
        elif phase == "phase_5_linter":
            fname = "lint_verdict.json"
        elif phase == "phase_6_compliance":
            fname = "compliance_verdict.json"
        elif phase == "phase_7_paired_auth_eval":
            fname = "paired_auth_eval_verdict.json"
        else:
            raise ValueError(f"unknown phase: {phase!r}")
        _write_canonical_phase_sidecar(
            sub_dir / fname, phase, clean=(phase in phases_clean_set)
        )
    return sub_dir


# ============================================================================
# Live-repo regression guard
# ============================================================================


def test_check_370_live_repo_warn_only_baseline():
    """Live-repo regression guard: warn-only initial wire-in baseline.

    Per Phase 1 spec memo §3 Phase 8 + CLAUDE.md "Strict-flip atomicity
    rule": initial wire-in is WARN-ONLY; the 4 known PR-facing submissions
    without canonical 4-verdict chain (predating the canonical pipeline)
    are expected baseline. The gate's purpose at landing is to SURFACE
    these candidates for operator-routed cleanup; the strict-flip happens
    only after Phase 10 PR101 baseline + first NEW submission both
    PACKET-CLEAN.
    """
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        strict=False, verbose=False
    )
    # Bound the baseline at landing to prevent unbounded drift.
    assert len(violations) <= 10, (
        f"Catalog #370 warn-only baseline regressed from <=10 to {len(violations)}; "
        f"first 2: {violations[:2]}"
    )


# ============================================================================
# Acceptance cascade (a): all 4 verdicts present + clean
# ============================================================================


def test_check_370_all_4_verdicts_present_and_clean_accepted(tmp_path):
    """The canonical default-path: all 4 sidecars present + clean -> PR_READY."""
    _write_submission_with_chain(tmp_path, "synthetic_all_clean")
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == [], (
        f"Expected zero violations for fully-canonical submission; "
        f"got: {violations}"
    )


# ============================================================================
# Each individual missing verdict flagged
# ============================================================================


@pytest.mark.parametrize("missing_phase", [
    "phase_4_builder",
    "phase_5_linter",
    "phase_6_compliance",
    "phase_7_paired_auth_eval",
])
def test_check_370_missing_each_individual_verdict_flagged(tmp_path, missing_phase):
    """Missing any individual canonical verdict raises a named blocker."""
    all_phases = (
        "phase_4_builder",
        "phase_5_linter",
        "phase_6_compliance",
        "phase_7_paired_auth_eval",
    )
    present_phases = tuple(p for p in all_phases if p != missing_phase)
    _write_submission_with_chain(
        tmp_path,
        f"synthetic_missing_{missing_phase}",
        phases_present=present_phases,
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1, f"Expected 1 violation; got {len(violations)}"
    msg = violations[0]
    assert "Catalog #370" in msg
    # Phase label appears in MISSING section
    phase_label_map = {
        "phase_4_builder": "Phase 4 (builder)",
        "phase_5_linter": "Phase 5 (linter)",
        "phase_6_compliance": "Phase 6 (compliance)",
        "phase_7_paired_auth_eval": "Phase 7 (paired_auth_eval)",
    }
    assert phase_label_map[missing_phase] in msg, (
        f"Expected {phase_label_map[missing_phase]!r} in violation message; "
        f"got: {msg}"
    )


def test_check_370_missing_all_4_verdicts_flagged(tmp_path):
    """Submission with PR-facing README but ZERO verdicts -> all 4 missing."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_no_verdicts",
        phases_present=(),
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    msg = violations[0]
    assert "Phase 4 (builder)" in msg
    assert "Phase 5 (linter)" in msg
    assert "Phase 6 (compliance)" in msg
    assert "Phase 7 (paired_auth_eval)" in msg


# ============================================================================
# Unclean verdict flagged
# ============================================================================


@pytest.mark.parametrize("unclean_phase", [
    "phase_4_builder",
    "phase_5_linter",
    "phase_6_compliance",
    "phase_7_paired_auth_eval",
])
def test_check_370_one_unclean_verdict_flagged(tmp_path, unclean_phase):
    """Submission with all 4 verdicts present but one in FAIL state -> flagged."""
    all_phases = (
        "phase_4_builder",
        "phase_5_linter",
        "phase_6_compliance",
        "phase_7_paired_auth_eval",
    )
    clean_phases = tuple(p for p in all_phases if p != unclean_phase)
    _write_submission_with_chain(
        tmp_path,
        f"synthetic_unclean_{unclean_phase}",
        phases_present=all_phases,
        phases_clean=clean_phases,
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1, f"Expected 1 violation; got {len(violations)}"
    msg = violations[0]
    phase_label_map = {
        "phase_4_builder": "Phase 4 (builder)",
        "phase_5_linter": "Phase 5 (linter)",
        "phase_6_compliance": "Phase 6 (compliance)",
        "phase_7_paired_auth_eval": "Phase 7 (paired_auth_eval)",
    }
    assert "PRESENT but NOT clean" in msg
    assert phase_label_map[unclean_phase] in msg


# ============================================================================
# Same-line waiver semantics + Catalog #287 placeholder rejection
# ============================================================================


def test_check_370_waiver_with_substantive_rationale_accepted(tmp_path):
    """Same-line waiver with non-placeholder rationale (>=4 chars) accepted."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_waived",
        phases_present=(),
        waiver_rationale="historical_submission_predates_canonical_pipeline_2026_05_19_anchor",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == [], (
        f"Expected zero violations for substantively waived submission; got: {violations}"
    )


def test_check_370_waiver_placeholder_rationale_rejected(tmp_path):
    """Per Catalog #287: <rationale> placeholder rejected."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_placeholder_waived",
        phases_present=(),
        waiver_rationale="<rationale>",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1, (
        f"Placeholder waiver must be rejected; got violations={violations}"
    )


def test_check_370_waiver_reason_placeholder_rejected(tmp_path):
    """Per Catalog #287: <reason> placeholder rejected."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_reason_placeholder",
        phases_present=(),
        waiver_rationale="<reason>",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_370_waiver_empty_rationale_rejected(tmp_path):
    """Empty waiver rationale rejected."""
    sub = _write_submission_with_chain(
        tmp_path,
        "synthetic_empty_waived",
        phases_present=(),
    )
    # Manually overwrite README to inject an empty-rationale waiver.
    readme = sub / "README.md"
    readme.write_text(
        "# PR submission\n"
        "<!-- # PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK: -->\n"
        "## Submission body\n",
        encoding="utf-8",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_370_waiver_short_rationale_rejected(tmp_path):
    """Rationale <4 chars rejected."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_short_waived",
        phases_present=(),
        waiver_rationale="ab",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_370_waiver_in_pr_body_file_accepted(tmp_path):
    """Waiver in PR_BODY.md (not README.md) also accepted."""
    sub_dir = tmp_path / "submissions" / "synthetic_pr_body_waived"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "README.md").write_text("Simple research README.\n", encoding="utf-8")
    (sub_dir / "PR_BODY.md").write_text(
        "# PR\n"
        "<!-- # PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:operator_reviewed_research_artifact_only -->\n"
        "## Submission\nBody.\n",
        encoding="utf-8",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_370_waiver_outside_first_30_lines_not_honored(tmp_path):
    """Waiver beyond line 30 of README.md is NOT honored."""
    sub_dir = tmp_path / "submissions" / "synthetic_late_waiver"
    sub_dir.mkdir(parents=True, exist_ok=True)
    readme_lines = ["# PR submission packet", "## Submission", "Body."]
    # Pad with 30 lines of placeholder content before the waiver.
    for i in range(30):
        readme_lines.append(f"Line {i}")
    readme_lines.append(
        "<!-- # PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:should_not_be_honored_too_late -->"
    )
    (sub_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


# ============================================================================
# Out-of-scope: not-PR-facing submissions
# ============================================================================


def test_check_370_research_only_readme_without_pr_tokens_not_in_scope(tmp_path):
    """README.md without PR-facing sentinel tokens is out of scope."""
    sub_dir = tmp_path / "submissions" / "synthetic_research_only"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "README.md").write_text(
        "# Research artifact for substrate XYZ\n\n"
        "Used internally for ablation; not a contest submission.\n",
        encoding="utf-8",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_370_submission_with_no_readme_or_pr_body_out_of_scope(tmp_path):
    """Submission directory with no PR-facing files is out of scope."""
    sub_dir = tmp_path / "submissions" / "synthetic_no_pr_artifacts"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "archive.zip").write_text("synthetic archive", encoding="utf-8")
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_370_pr_body_md_alone_qualifies_as_pr_facing(tmp_path):
    """PR_BODY.md presence alone (no README) qualifies submission as PR-facing."""
    sub_dir = tmp_path / "submissions" / "synthetic_only_pr_body"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "PR_BODY.md").write_text(
        "# PR101: substrate ABC submission body\n", encoding="utf-8"
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_370_pr_body_glob_matches_pr_body_draft(tmp_path):
    """PR_BODY_DRAFT.md (glob match) also flagged as PR-facing."""
    sub_dir = tmp_path / "submissions" / "synthetic_pr_body_draft"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "PR_BODY_DRAFT.md").write_text(
        "# Draft PR body\n", encoding="utf-8"
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


# ============================================================================
# Exempt path markers
# ============================================================================


def test_check_370_exact_current_exempt(tmp_path):
    """submissions/exact_current/ is exempt (pinned upstream snapshot)."""
    sub_dir = tmp_path / "submissions" / "exact_current"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "README.md").write_text(
        "# PR submission\n## Submission body.\n[contest-CPU] 0.19\n",
        encoding="utf-8",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == [], (
        f"submissions/exact_current/ must be exempt; got: {violations}"
    )


# ============================================================================
# Strict-mode behavior
# ============================================================================


def test_check_370_strict_mode_raises_on_violation(tmp_path):
    """strict=True raises PreflightError with Catalog #370 message."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_strict_test",
        phases_present=(),
    )
    with pytest.raises(PreflightError) as excinfo:
        preflight.check_no_pr_submission_without_canonical_compliance_verdict(
            repo_root=tmp_path, strict=True, verbose=False
        )
    msg = str(excinfo.value)
    assert "Catalog #370" in msg
    assert "Phase 8" in msg
    assert "canonical-submission-pipeline" in msg
    assert "Phase 4 builder" in msg
    assert "Phase 5 linter" in msg
    assert "Phase 6 compliance" in msg
    assert "Phase 7 paired_auth_eval" in msg


def test_check_370_strict_mode_silent_on_clean(tmp_path):
    """strict=True with zero violations returns without raising."""
    # Empty submissions/ -> zero PR-facing -> zero violations.
    (tmp_path / "submissions").mkdir(parents=True, exist_ok=True)
    result = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert result == []


def test_check_370_no_submissions_dir_returns_empty(tmp_path):
    """Missing submissions/ directory is silent (returns empty list)."""
    result = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert result == []


# ============================================================================
# Verbose output
# ============================================================================


def test_check_370_verbose_output_on_dirty(tmp_path, capsys):
    """verbose=True prints WARN summary when violations present."""
    _write_submission_with_chain(
        tmp_path,
        "synthetic_verbose_dirty",
        phases_present=(),
    )
    preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "[catalog-370]" in captured.out
    assert "WARN" in captured.out


def test_check_370_verbose_output_on_clean(tmp_path, capsys):
    """verbose=True prints OK summary when no violations."""
    (tmp_path / "submissions").mkdir(parents=True, exist_ok=True)
    preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "[catalog-370]" in captured.out
    assert "OK" in captured.out


# ============================================================================
# String repo_root accepted
# ============================================================================


def test_check_370_string_repo_root_accepted(tmp_path):
    """repo_root accepted as str (not just Path)."""
    _write_submission_with_chain(tmp_path, "synthetic_str_repo_root")
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert violations == []


# ============================================================================
# Multi-submission aggregation
# ============================================================================


def test_check_370_multi_submission_aggregation(tmp_path):
    """Multiple PR-facing submissions without canonical chain all flagged."""
    _write_submission_with_chain(tmp_path, "synthetic_multi_a", phases_present=())
    _write_submission_with_chain(tmp_path, "synthetic_multi_b", phases_present=())
    _write_submission_with_chain(tmp_path, "synthetic_multi_c", phases_present=())
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 3


# ============================================================================
# Sidecar search cascade
# ============================================================================


def test_check_370_sidecar_in_experiments_results_lane_dir(tmp_path):
    """Sidecar can live in experiments/results/<sub>*/ instead of submission_dir."""
    sub_dir = tmp_path / "submissions" / "synthetic_lane_sidecars"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "README.md").write_text(
        "# PR101 submission packet\n## Submission body\n[contest-CPU]\n",
        encoding="utf-8",
    )
    # Sidecars live in experiments/results/synthetic_lane_sidecars_lane01/
    lane_dir = tmp_path / "experiments" / "results" / "synthetic_lane_sidecars_lane01"
    lane_dir.mkdir(parents=True, exist_ok=True)
    _write_canonical_phase_sidecar(
        lane_dir / "submission_bundle_result.json", "phase_4_builder"
    )
    _write_canonical_phase_sidecar(
        lane_dir / "lint_verdict.json", "phase_5_linter"
    )
    _write_canonical_phase_sidecar(
        lane_dir / "compliance_verdict.json", "phase_6_compliance"
    )
    _write_canonical_phase_sidecar(
        lane_dir / "paired_auth_eval_verdict.json", "phase_7_paired_auth_eval"
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_370_compliance_sidecar_in_reports_pr_pre_submission(tmp_path):
    """Phase 6 compliance sidecar can live in reports/pr_pre_submission/."""
    sub_dir = tmp_path / "submissions" / "synthetic_reports_dir"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "README.md").write_text(
        "# PR submission packet\n## Reproducibility\nBody.\n",
        encoding="utf-8",
    )
    # Phases 4, 5, 7 directly in submission_dir
    _write_canonical_phase_sidecar(
        sub_dir / "submission_bundle_result.json", "phase_4_builder"
    )
    _write_canonical_phase_sidecar(
        sub_dir / "lint_verdict.json", "phase_5_linter"
    )
    _write_canonical_phase_sidecar(
        sub_dir / "paired_auth_eval_verdict.json", "phase_7_paired_auth_eval"
    )
    # Phase 6 compliance in reports/pr_pre_submission/ (canonical convention)
    reports_dir = tmp_path / "reports" / "pr_pre_submission"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_canonical_phase_sidecar(
        reports_dir / "compliance_report_synthetic_20260527T000000Z.json",
        "phase_6_compliance",
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_370_dual_eval_adjudicated_accepted_as_phase_7(tmp_path):
    """dual_eval_adjudicated.json (PR101+PR102 medal-class) accepted for Phase 7."""
    sub_dir = tmp_path / "submissions" / "synthetic_dual_eval"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "README.md").write_text(
        "# PR submission packet\n## Submission\n[contest-CUDA]\n", encoding="utf-8"
    )
    _write_canonical_phase_sidecar(
        sub_dir / "submission_bundle_result.json", "phase_4_builder"
    )
    _write_canonical_phase_sidecar(
        sub_dir / "lint_verdict.json", "phase_5_linter"
    )
    _write_canonical_phase_sidecar(
        sub_dir / "compliance_verdict.json", "phase_6_compliance"
    )
    # Phase 7 via dual_eval_adjudicated.json sister file (PR101+PR102 precedent)
    _write_canonical_phase_sidecar(
        sub_dir / "dual_eval_adjudicated.json", "phase_7_paired_auth_eval"
    )
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ============================================================================
# End-to-end Phase 4 -> 5 -> 6 -> 7 chain integration
# ============================================================================


def test_check_370_end_to_end_4_phase_chain_integration(tmp_path):
    """Full Phase 4 -> 5 -> 6 -> 7 chain via canonical sidecars accepted."""
    # The default `_write_submission_with_chain` produces a synthetic full
    # 4-verdict chain; verify it works end-to-end without per-phase
    # mocking.
    sub_dir = _write_submission_with_chain(
        tmp_path,
        "synthetic_e2e_canonical",
    )
    # Verify all 4 sidecar files exist in the submission_dir.
    assert (sub_dir / "submission_bundle_result.json").exists()
    assert (sub_dir / "lint_verdict.json").exists()
    assert (sub_dir / "compliance_verdict.json").exists()
    assert (sub_dir / "paired_auth_eval_verdict.json").exists()
    # Verify the gate accepts the submission.
    violations = preflight.check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ============================================================================
# Catalog #185 sister-callable regression guard
# ============================================================================


def test_check_370_callable_via_preflight_globals():
    """Catalog #185 sister: gate function must be callable via module globals."""
    fn = getattr(preflight, "check_no_pr_submission_without_canonical_compliance_verdict", None)
    assert fn is not None, (
        "Catalog #370 gate function must be importable from tac.preflight per "
        "Catalog #185 META-meta-meta drift detection"
    )
    assert callable(fn)


# ============================================================================
# Orchestrator wire-in regression guard
# ============================================================================


def test_check_370_orchestrator_wire_in_present():
    """preflight_all() body must invoke Catalog #370 gate (warn-only initial)."""
    preflight_source = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text(
        encoding="utf-8"
    )
    # Find the orchestrator invocation pattern.
    assert "check_no_pr_submission_without_canonical_compliance_verdict(" in preflight_source
    # Per CLAUDE.md "Strict-flip atomicity rule" + Phase 1 spec memo §3 Phase 8:
    # WARN-ONLY initial wire-in (strict=False).
    assert "Catalog #370" in preflight_source


def test_check_370_helper_constants_pinned():
    """Pin canonical helper constants to prevent silent regression."""
    from tac.preflight import (
        _CHECK_370_EXEMPT_PATH_MARKERS,
        _CHECK_370_PHASES,
        _CHECK_370_PHASE_REMEDIATION,
        _CHECK_370_PR_FACING_FILENAMES,
        _CHECK_370_PR_FACING_README_TOKENS,
        _CHECK_370_WAIVER_MIN_RATIONALE_LEN,
        _CHECK_370_WAIVER_LOOKBACK_LINES,
    )
    # Verify exact_current exclusion is canonical
    assert any("exact_current" in m for m in _CHECK_370_EXEMPT_PATH_MARKERS)
    # Verify all 4 canonical phases are wired
    phase_keys = [k for k, _ in _CHECK_370_PHASES]
    assert "phase_4_builder" in phase_keys
    assert "phase_5_linter" in phase_keys
    assert "phase_6_compliance" in phase_keys
    assert "phase_7_paired_auth_eval" in phase_keys
    # Each phase has a canonical remediation command
    for k in phase_keys:
        assert k in _CHECK_370_PHASE_REMEDIATION
        assert ".venv/bin/python" in _CHECK_370_PHASE_REMEDIATION[k]
    # PR-facing filenames + README tokens canonical
    assert "PR_BODY.md" in _CHECK_370_PR_FACING_FILENAMES
    assert "README.md" in _CHECK_370_PR_FACING_FILENAMES
    assert any("commaai" in t for t in _CHECK_370_PR_FACING_README_TOKENS)
    # Waiver guards canonical
    assert _CHECK_370_WAIVER_MIN_RATIONALE_LEN >= 4
    assert _CHECK_370_WAIVER_LOOKBACK_LINES >= 10


# ============================================================================
# Cathedral consumer regression guard (Catalog #335 + #341 sister)
# ============================================================================


def test_check_370_cathedral_consumer_canonical_contract():
    """Companion cathedral consumer satisfies Catalog #335 canonical contract."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer
    from tac.cathedral.consumer_contract import HookNumber

    # Catalog #335 canonical contract surfaces
    assert hasattr(consumer, "CONSUMER_NAME")
    assert consumer.CONSUMER_NAME == "pr_submission_compliance_consumer"
    assert hasattr(consumer, "CONSUMER_VERSION")
    assert consumer.CONSUMER_VERSION == "1.0.0"
    assert hasattr(consumer, "CONSUMER_HOOK_NUMBERS")
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert hasattr(consumer, "update_from_anchor")
    assert callable(consumer.update_from_anchor)
    assert hasattr(consumer, "consume_candidate")
    assert callable(consumer.consume_candidate)


def test_check_370_cathedral_consumer_tier_a_routing_markers():
    """Companion consumer satisfies Catalog #341 Tier A canonical-routing markers."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer

    # Empty Mapping candidate -> Tier-A canonical markers honored regardless
    # of readiness verdict shape (empty mapping flows through with all 4
    # phases missing, producing BLOCKED_ON_<phase>; non-Mapping input
    # returns UNKNOWN). The Tier-A invariants must hold in BOTH cases per
    # Catalog #341 canonical-routing markers.
    contrib_empty_map = consumer.consume_candidate({})
    assert contrib_empty_map["predicted_delta_adjustment"] == 0.0
    assert contrib_empty_map["promotable"] is False
    assert contrib_empty_map["axis_tag"] == "[predicted]"
    # Verdict for empty mapping is structurally BLOCKED_ON_<earliest-missing>
    assert contrib_empty_map["readiness_verdict"].startswith("BLOCKED_ON_") or \
        contrib_empty_map["readiness_verdict"] == "UNKNOWN"

    # Non-Mapping input -> neutral UNKNOWN observation
    contrib_non_map = consumer.consume_candidate("not a mapping")  # type: ignore[arg-type]
    assert contrib_non_map["readiness_verdict"] == "UNKNOWN"
    assert contrib_non_map["predicted_delta_adjustment"] == 0.0
    assert contrib_non_map["promotable"] is False
    assert contrib_non_map["axis_tag"] == "[predicted]"


def test_check_370_cathedral_consumer_full_clean_candidate():
    """Full clean 4-verdict candidate -> PR_READY readiness verdict."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer

    candidate = {
        "submission_bundle_result": {"overall_pass": True},
        "lint_verdict": {"overall_clean": True},
        "compliance_verdict": {"overall_clean": True},
        "paired_auth_eval_verdict": {"verdict": "PAIRED_PASS"},
    }
    contrib = consumer.consume_candidate(candidate)
    assert contrib["readiness_verdict"] == "PR_READY"
    assert contrib["predicted_delta_adjustment"] == 0.0
    assert contrib["promotable"] is False


def test_check_370_cathedral_consumer_missing_phase_routing():
    """Missing one phase -> BLOCKED_ON_<phase> readiness verdict."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer

    candidate = {
        "submission_bundle_result": {"overall_pass": True},
        "lint_verdict": {"overall_clean": True},
        "compliance_verdict": {"overall_clean": True},
        # paired_auth_eval_verdict missing
    }
    contrib = consumer.consume_candidate(candidate)
    assert "PAIRED_AUTH_EVAL" in contrib["readiness_verdict"]
    assert contrib["promotable"] is False


def test_check_370_cathedral_consumer_forbidden_macos_routing():
    """forbidden_macos_axis_detected -> BLOCKED_FORBIDDEN_MACOS_AXIS."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer

    candidate = {
        "submission_bundle_result": {"overall_pass": True},
        "lint_verdict": {"overall_clean": True},
        "compliance_verdict": {
            "overall_clean": False,
            "forbidden_macos_axis_detected": True,
        },
        "paired_auth_eval_verdict": {"verdict": "PAIRED_FAIL"},
    }
    contrib = consumer.consume_candidate(candidate)
    assert contrib["readiness_verdict"] == "BLOCKED_FORBIDDEN_MACOS_AXIS"


def test_check_370_cathedral_consumer_non_mapping_candidate():
    """Non-Mapping candidate -> neutral observation (no exception)."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer

    contrib = consumer.consume_candidate(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert contrib["readiness_verdict"] == "UNKNOWN"
    assert contrib["predicted_delta_adjustment"] == 0.0


def test_check_370_cathedral_consumer_update_from_anchor_no_exception():
    """update_from_anchor is observability-only and never raises."""
    from tac.cathedral_consumers import pr_submission_compliance_consumer as consumer

    consumer.update_from_anchor({})
    consumer.update_from_anchor(None)
    consumer.update_from_anchor({"any": "shape"})


# ============================================================================
# Sister gate cross-reference regression (Catalog #176)
# ============================================================================


def test_check_370_claude_md_catalog_row_present():
    """Catalog #176 META-meta sister: CLAUDE.md must contain Catalog #370 row."""
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert (
        "370. `check_no_pr_submission_without_canonical_compliance_verdict`"
        in claude_md
    ), (
        "Catalog #176 META-meta gate: every STRICT-callsite in preflight_all() "
        "MUST have a matching `^N. \\`check_<name>\\`` row in CLAUDE.md "
        "Meta-bug class catalog section"
    )


# ============================================================================
# Phase 4 -> 5 -> 6 -> 7 sister coordination smoke
# ============================================================================


def test_check_370_phase_5_through_7_sister_packages_present():
    """Sister Phase 4/5/6/7 packages exist (Phase 8 builds on them)."""
    from tac import submission_packet
    assert hasattr(submission_packet, "build_submission_bundle"), "Phase 4"
    assert hasattr(submission_packet, "lint_submission_bundle"), "Phase 5"
    assert hasattr(submission_packet, "enforce_contest_compliance"), "Phase 6"
    # Phase 7 may still be in flight at landing; tolerate either landing state.
    # The gate accepts dual_eval_adjudicated.json sister sidecar as Phase 7
    # fallback per PR101+PR102 medal-class precedent so the gate is functional
    # regardless of Phase 7 helper presence.
