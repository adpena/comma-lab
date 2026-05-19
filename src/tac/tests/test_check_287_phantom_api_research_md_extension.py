"""Tests for Catalog #287-B SCOPE-EXTENSION 2026-05-18 — phantom-API in
research memos.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
+ operator decision E.1 2026-05-18 + grand council T3 finding #5 PROCEED
+ design memo `.omx/research/catalog_287_scope_extension_to_research_md_
phantom_api_design_<utc>.md`.

These tests cover the NEW sub-scope B behavior added to
`check_no_docstring_overstatement_without_evidence_tag`. They scan
`.omx/research/**/*.md` for `tac.X[.Y...]` citations and refuse any whose
2-component prefix `tac.X` is not importable.

NOTE: this test file contains VERBATIM phantom-API samples for fixture
purposes (`tac.definitely_not_a_real_module`, etc.). The self-exempt
mechanism in the gate ensures this test file is NOT flagged when the gate
runs against the live repo.

Lane: lane_meta_phantom_api_structural_extinction_catalog_287_scope_extend_20260518
"""

from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
    _check_287b_extract_two_component_prefix,
    _check_287b_file_waived,
    _check_287b_line_waived,
    _check_287b_module_is_importable,
    _check_287b_strip_code_fences_and_html_comments,
    check_no_docstring_overstatement_without_evidence_tag,
)


# =============================================================================
# Helper unit tests
# =============================================================================


class TestExtractTwoComponentPrefix:
    def test_returns_tac_x_for_tac_x_y_z(self):
        assert _check_287b_extract_two_component_prefix("tac.atom.ledger.append_atom") == "tac.atom"

    def test_returns_tac_x_for_bare_tac_x(self):
        assert _check_287b_extract_two_component_prefix("tac.unified_action") == "tac.unified_action"

    def test_returns_none_for_single_component_tac(self):
        assert _check_287b_extract_two_component_prefix("tac") is None

    def test_returns_none_for_non_tac(self):
        assert _check_287b_extract_two_component_prefix("numpy.array") is None

    def test_returns_none_for_empty(self):
        assert _check_287b_extract_two_component_prefix("") is None


class TestModuleIsImportable:
    def test_real_canonical_importable(self):
        # tac.unified_action is real per the 15th-instance memo
        assert _check_287b_module_is_importable("tac.unified_action") is True

    def test_phantom_returns_false(self):
        assert _check_287b_module_is_importable("tac.definitely_not_a_real_module") is False

    def test_phantom_from_15th_instance_memo(self):
        # tac.magic_codec is THE canonical phantom example from the 15th
        # instance memo. The real helper is tac.codec_magic_registry.
        assert _check_287b_module_is_importable("tac.magic_codec") is False

    def test_tac_atom_ledger_importable(self):
        # Function-inside-module case: tac.atom.ledger IS importable
        assert _check_287b_module_is_importable("tac.atom.ledger") is True


class TestStripCodeFencesAndHtmlComments:
    def test_strips_triple_backtick_fence(self):
        text = "Outside\n```\ntac.magic_codec\n```\nOutside again"
        masked = _check_287b_strip_code_fences_and_html_comments(text)
        # The phantom citation should be masked
        assert "tac.magic_codec" not in masked
        # Outside lines preserved
        assert "Outside" in masked
        assert "Outside again" in masked

    def test_strips_single_line_html_comment(self):
        text = "Before <!-- tac.magic_codec --> After"
        masked = _check_287b_strip_code_fences_and_html_comments(text)
        assert "tac.magic_codec" not in masked
        assert "Before" in masked
        assert "After" in masked

    def test_strips_multi_line_html_comment(self):
        text = "Before\n<!-- start\ntac.magic_codec\nend -->\nAfter"
        masked = _check_287b_strip_code_fences_and_html_comments(text)
        assert "tac.magic_codec" not in masked
        assert "Before" in masked
        assert "After" in masked

    def test_preserves_inline_code(self):
        # Inline code (single backtick) is NOT stripped — those are the
        # canonical "we have tac.X helper" claims the bug class targets
        text = "The canonical `tac.magic_codec` helper..."
        masked = _check_287b_strip_code_fences_and_html_comments(text)
        assert "tac.magic_codec" in masked


class TestFileWaived:
    def test_file_waived_with_rationale_in_first_30_lines(self):
        text = "# Title\n# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: design memo proposes new helpers\n\nBody"
        assert _check_287b_file_waived(text) is True

    def test_file_waived_outside_first_30_lines_not_accepted(self):
        head = "\n".join(["# Line " + str(i) for i in range(35)])
        text = head + "\n# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: too late"
        assert _check_287b_file_waived(text) is False

    def test_file_waived_with_rationale_placeholder_rejected(self):
        text = "# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: <rationale>"
        assert _check_287b_file_waived(text) is False

    def test_file_waived_with_reason_placeholder_rejected(self):
        text = "# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: <reason>"
        assert _check_287b_file_waived(text) is False

    def test_no_waiver_present(self):
        text = "# Plain memo title\n\nBody"
        assert _check_287b_file_waived(text) is False


class TestLineWaived:
    def test_phantom_intentional_ok_with_rationale_accepted(self):
        line = "tac.foo  # PHANTOM_NAME_INTENTIONAL_OK: documenting the bug"
        assert _check_287b_line_waived(line) is True

    def test_design_proposal_not_yet_implemented_with_rationale_accepted(self):
        line = "tac.future_helper  # DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED: queued for sister wave"
        assert _check_287b_line_waived(line) is True

    def test_phantom_intentional_ok_placeholder_rejected(self):
        line = "tac.foo  # PHANTOM_NAME_INTENTIONAL_OK: <rationale>"
        assert _check_287b_line_waived(line) is False

    def test_design_proposal_placeholder_rejected(self):
        line = "tac.foo  # DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED: <reason>"
        assert _check_287b_line_waived(line) is False

    def test_no_waiver_marker_returns_false(self):
        line = "tac.foo just a citation"
        assert _check_287b_line_waived(line) is False


# =============================================================================
# End-to-end gate behavior
# =============================================================================


@pytest.fixture
def synthetic_repo_with_phantom(tmp_path):
    """Create minimal repo root with .omx/research/ + src/tac/ for testing."""
    (tmp_path / ".omx" / "research").mkdir(parents=True)
    (tmp_path / "src" / "tac").mkdir(parents=True)
    return tmp_path


class TestSubScopeBEndToEnd:
    def test_phantom_citation_flagged(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "phantom_test_20260518.md"
        memo.write_text("# Test\n\nWe use `tac.definitely_not_a_real_module` helper.\n")
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert any("tac.definitely_not_a_real_module" in v for v in violations)

    def test_real_citation_accepted(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "real_test_20260518.md"
        memo.write_text("# Test\n\nWe use `tac.unified_action` helper.\n")
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.unified_action" in v for v in violations)

    def test_function_inside_module_accepted(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "func_inside_test_20260518.md"
        memo.write_text(
            "# Test\n\nCall `tac.atom.ledger.append_atom` per Catalog #245.\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        # tac.atom.ledger IS importable so the function-inside-module
        # citation is accepted
        assert not any("tac.atom.ledger.append_atom" in v for v in violations)

    def test_non_tac_citation_ignored(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "numpy_test_20260518.md"
        memo.write_text("# Test\n\nUse `numpy.array` and `torch.Tensor`.\n")
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("numpy" in v or "torch" in v for v in violations)

    def test_same_line_phantom_intentional_ok_waiver_accepted(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "waived_test_20260518.md"
        memo.write_text(
            "# Test\n\n"
            "Cite `tac.totally_fake_helper` for documentation purposes.  "
            "<!-- PHANTOM_NAME_INTENTIONAL_OK: documenting the bug class anchor -->\n"
        )
        # Note: HTML-comment waivers get stripped before scanning. Use
        # hash-form for inline waivers.
        # Re-write with hash-form waiver:
        memo.write_text(
            "# Test\n\n"
            "tac.totally_fake_helper  # PHANTOM_NAME_INTENTIONAL_OK: documenting the bug class anchor\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.totally_fake_helper" in v for v in violations)

    def test_same_line_phantom_intentional_ok_placeholder_rejected(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "placeholder_test_20260518.md"
        memo.write_text(
            "# Test\n\n"
            "tac.totally_fake_helper  # PHANTOM_NAME_INTENTIONAL_OK: <rationale>\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        # Placeholder rejected → violation still flagged
        assert any("tac.totally_fake_helper" in v for v in violations)

    def test_design_proposal_waiver_accepted(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "proposal_test_20260518.md"
        memo.write_text(
            "# Test\n\n"
            "Proposing `tac.new_proposed_helper`.  # DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED: queued for next sister wave\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.new_proposed_helper" in v for v in violations)

    def test_design_proposal_placeholder_rejected(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "proposal_placeholder_20260518.md"
        memo.write_text(
            "# Test\n\n"
            "Proposing `tac.new_proposed_helper`.  # DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED: <reason>\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert any("tac.new_proposed_helper" in v for v in violations)

    def test_file_level_waiver_accepts_whole_file(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "file_waiver_test_20260518.md"
        memo.write_text(
            "# Design memo\n"
            "# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: this memo proposes 5 new canonical helpers\n"
            "\n"
            "Proposed: `tac.foo_new`, `tac.bar_new`, `tac.baz_new`.\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.foo_new" in v or "tac.bar_new" in v for v in violations)

    def test_file_level_waiver_outside_first_30_lines_not_accepted(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "late_waiver_test_20260518.md"
        body_lines = [f"# Padding line {i}" for i in range(35)]
        memo.write_text(
            "\n".join(body_lines)
            + "\n# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: too late to count\n"
            "\nProposed: `tac.too_late_helper`.\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert any("tac.too_late_helper" in v for v in violations)

    def test_file_level_waiver_placeholder_rejected(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "file_placeholder_test_20260518.md"
        memo.write_text(
            "# Title\n# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: <rationale>\n\n"
            "`tac.unwaived_helper` cited.\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert any("tac.unwaived_helper" in v for v in violations)

    def test_code_fenced_blocks_excluded(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "code_fence_test_20260518.md"
        memo.write_text(
            "# Test\n\n"
            "```python\n"
            "from tac.future_helper_in_code_block import foo\n"
            "```\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.future_helper_in_code_block" in v for v in violations)

    def test_html_comment_lines_excluded(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "html_comment_test_20260518.md"
        memo.write_text(
            "# Test\n\n<!-- tac.commented_phantom -->\nVisible text\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.commented_phantom" in v for v in violations)

    def test_exempt_path_marker_archive_excluded(self, synthetic_repo_with_phantom):
        archive_dir = synthetic_repo_with_phantom / ".omx" / "research" / "_archive"
        archive_dir.mkdir(parents=True)
        memo = archive_dir / "archived_phantom_test_20260518.md"
        memo.write_text("`tac.archived_phantom`\n")
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
        )
        assert not any("tac.archived_phantom" in v for v in violations)

    def test_memory_files_off_by_default(self, synthetic_repo_with_phantom, monkeypatch):
        # By default, scan_memory_files=False — memory files should be ignored
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=False,
            scan_research_memos=True,
            scan_memory_files=False,
        )
        # No errors regardless of operator's actual memory dir state
        assert isinstance(violations, list)

    def test_strict_mode_raises_with_catalog_287_message(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "strict_raise_test_20260518.md"
        memo.write_text("`tac.phantom_for_strict_raise`\n")
        with pytest.raises(PreflightError) as excinfo:
            check_no_docstring_overstatement_without_evidence_tag(
                repo_root=synthetic_repo_with_phantom,
                strict=True,
                scan_research_memos=True,
            )
        assert "Catalog #287" in str(excinfo.value)

    def test_strict_silent_on_clean(self, synthetic_repo_with_phantom):
        memo = synthetic_repo_with_phantom / ".omx" / "research" / "clean_test_20260518.md"
        memo.write_text("Use `tac.unified_action`.\n")
        # Should NOT raise
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=synthetic_repo_with_phantom,
            strict=True,
            scan_research_memos=True,
        )
        assert violations == []

    def test_sub_scope_a_backwards_compat_still_works(self, tmp_path):
        # Sub-scope A: src/tac/**/*.py docstring overstatement still flagged
        (tmp_path / "src" / "tac").mkdir(parents=True)
        py_file = tmp_path / "src" / "tac" / "bad_docstring.py"
        py_file.write_text(
            '"""Docstring with overstatement.\n\nsaves 49% bytes\n"""\n'
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=tmp_path,
            strict=False,
            scan_research_memos=False,
        )
        assert any("saves 49%" in v for v in violations)


class TestLiveRepoRegressionGuard:
    def test_live_repo_subscope_a_existing_baseline_preserved(self):
        # Run sub-scope A only (no research memo scan) — baseline behavior
        violations = check_no_docstring_overstatement_without_evidence_tag(
            strict=False,
            scan_research_memos=False,
        )
        # The existing baseline at landing — sub-scope A alone has historical
        # warn-only state. Just verify it's a list (any count acceptable).
        assert isinstance(violations, list)

    def test_live_repo_subscope_b_bounded(self):
        # Sub-scope B is the new scope. Initial WARN-ONLY because backfill
        # is required; bound at a reasonable ceiling so this test surfaces
        # if the count explodes unexpectedly.
        violations = check_no_docstring_overstatement_without_evidence_tag(
            strict=False,
            scan_research_memos=True,
            scan_memory_files=False,
        )
        # Live count expected initially in the hundreds; cap at 5000 as a
        # sanity ceiling so a 10x explosion would surface.
        assert len(violations) <= 5000, (
            f"Sub-scope B live count = {len(violations)} exceeds ceiling 5000; "
            "either backfill sister wave has not landed yet OR a regression "
            "is producing many false positives."
        )

    def test_self_exempt_files_not_flagged_for_their_own_samples(self):
        # The test file itself contains 'tac.definitely_not_a_real_module'
        # etc. for fixture purposes; self-exempt mechanism should prevent
        # them flagging the live repo run.
        violations = check_no_docstring_overstatement_without_evidence_tag(
            strict=False,
            scan_research_memos=True,
            scan_memory_files=False,
        )
        # Ensure this test file's verbatim phantom samples are NOT in violations
        # (paired by file suffix match; verbatim phantom samples in self-
        # exempt files do not pollute the live repo count).
        for v in violations:
            assert "test_check_287_phantom_api_research_md_extension" not in v
