"""Tests for Catalog #176 — check_strict_preflight_callsites_have_claude_md_catalog_row.

FIX-WAVE-2 R2-1 (2026-05-13). META-meta gate: every strict=True callsite
in preflight_all() MUST have a matching numbered row in the CLAUDE.md
"Meta-bug class catalog" table. RECURSIVE-REVIEW-R1 missed entries for
Catalog #174 + #175 in the catalog table; this gate prevents recurrence.

Memory: feedback_fix_wave_2_r2_findings_LANDED_20260513.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_strict_preflight_callsites_have_claude_md_catalog_row,
)


def _write_repo(
    root: Path,
    *,
    preflight_body: str,
    claude_md_body: str,
) -> None:
    (root / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (root / "src" / "tac" / "preflight.py").write_text(preflight_body)
    (root / "CLAUDE.md").write_text(claude_md_body)


def test_check_176_live_count_zero():
    """The check MUST have 0 live violations at landing (strict-flip atom)."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert violations == [], (
        f"Live violations should be 0, got: {violations}"
    )


def test_check_176_returns_empty_when_files_missing(tmp_path):
    # No CLAUDE.md, no preflight.py
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_176_detects_uncataloged_strict_callsite(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_foo_undocumented(\n"
        "        strict=True, verbose=verbose,\n"
        "    )\n"
        "\n"
        "def check_foo_undocumented(*, strict=False, verbose=False):\n"
        "    return []\n"
    )
    claude_md_body = "# CLAUDE.md\n\nNo catalog table entries here.\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("check_foo_undocumented" in v for v in violations)


def test_check_176_accepts_cataloged_strict_callsite(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_foo_cataloged(\n"
        "        strict=True, verbose=verbose,\n"
        "    )\n"
    )
    claude_md_body = (
        "# CLAUDE.md\n\n"
        "123. `check_foo_cataloged` — entry text here.\n"
    )
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_176_ignores_warn_only_callsites(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_warn_only_check(\n"
        "        strict=False, verbose=verbose,\n"
        "    )\n"
    )
    claude_md_body = "# CLAUDE.md\n\nNo catalog table entries here.\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # strict=False callsite has no requirement
    assert violations == []


def test_check_176_strict_raises_preflight_error(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_uncataloged(\n"
        "        strict=True, verbose=verbose,\n"
        "    )\n"
    )
    claude_md_body = "# CLAUDE.md\n\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    with pytest.raises(PreflightError):
        check_strict_preflight_callsites_have_claude_md_catalog_row(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_176_respects_same_line_waiver(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_intentionally_uncataloged(strict=True)  "
        "# CLAUDE_MD_ENTRY_OK: META-internal helper\n"
    )
    claude_md_body = "# CLAUDE.md\n\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_176_ignores_calls_outside_preflight_all(tmp_path):
    preflight_body = (
        "def some_other_function():\n"
        "    check_outside_orchestrator(\n"
        "        strict=True, verbose=False,\n"
        "    )\n"
        "\n"
        "def preflight_all(verbose=False):\n"
        "    pass\n"
    )
    claude_md_body = "# CLAUDE.md\n\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # Outside preflight_all - not in scope
    assert violations == []


def test_check_176_handles_multiple_callsites_one_uncataloged(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_alpha(strict=True)\n"
        "    check_beta(strict=True)\n"
        "    check_gamma(strict=False)\n"
    )
    claude_md_body = (
        "# CLAUDE.md\n\n"
        "100. `check_alpha` — entry.\n"
        # check_beta missing!
        "102. `check_gamma` — entry.\n"
    )
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "check_beta" in violations[0]


def test_check_176_handles_strict_true_in_multiline_call(tmp_path):
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    check_multiline(\n"
        "        strict=True,\n"
        "        verbose=verbose,\n"
        "    )\n"
    )
    claude_md_body = "# CLAUDE.md\n\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("check_multiline" in v for v in violations)


def test_check_176_extract_callsites_helper_smoke():
    """Verify the helper can extract callsites from the real preflight.py."""
    from tac.preflight import _check_176_collect_strict_callsites
    repo_root = Path(__file__).resolve().parents[3]
    preflight_text = (repo_root / "src" / "tac" / "preflight.py").read_text(
        encoding="utf-8"
    )
    callsites = _check_176_collect_strict_callsites(preflight_text)
    # At minimum #174 + #175 + #176 + #177 should appear
    names = {name for _, name in callsites}
    assert "check_subagent_commit_serializer_always_uses_expected_content_sha256" in names
    assert "check_cost_band_anchor_writers_declare_outcome" in names
    assert "check_strict_preflight_callsites_have_claude_md_catalog_row" in names
    assert "check_cost_band_posterior_rows_have_outcome_field" in names


def test_check_176_legacy_allowlist_accepts_pre_existing(tmp_path):
    """Legacy pre-FIX-WAVE-2 strict callsites with no CLAUDE.md row are
    accepted via the snapshot allowlist."""
    from tac.preflight import _CHECK_176_LEGACY_ALLOWLIST
    # Pick the first allowlist entry
    legacy_name = sorted(_CHECK_176_LEGACY_ALLOWLIST)[0]
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        f"    {legacy_name}(strict=True)\n"
    )
    claude_md_body = "# CLAUDE.md\n\n"  # NO catalog entry for legacy_name
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []  # accepted via legacy allowlist


def test_check_176_legacy_allowlist_is_a_frozenset_of_strings():
    from tac.preflight import _CHECK_176_LEGACY_ALLOWLIST
    assert isinstance(_CHECK_176_LEGACY_ALLOWLIST, frozenset)
    assert all(isinstance(s, str) for s in _CHECK_176_LEGACY_ALLOWLIST)
    assert len(_CHECK_176_LEGACY_ALLOWLIST) > 0


def test_check_176_skips_def_lines(tmp_path):
    # The `def check_name(...)` line itself should NOT be treated as a callsite
    preflight_body = (
        "def preflight_all(verbose=False):\n"
        "    pass\n"
        "\n"
        "def check_some_definition(*, strict=True, verbose=False):\n"
        "    return []\n"
    )
    claude_md_body = "# CLAUDE.md\n\n"
    _write_repo(tmp_path, preflight_body=preflight_body, claude_md_body=claude_md_body)
    violations = check_strict_preflight_callsites_have_claude_md_catalog_row(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # The `def check_some_definition(*, strict=True, ...)` should not count
    # because it is not inside preflight_all() body
    assert violations == []
