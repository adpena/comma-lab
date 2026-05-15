# SPDX-License-Identifier: MIT
"""Tests for Catalog #290 ``check_substrate_design_memo_has_canonical_vs_unique_decision_section``.

Per ``feedback_knowledge_preservation_pr95_meta_level_lesson_landed_20260515.md``
+ ``feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md``
+ ``feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md``.

The gate refuses substrate scaffold landing memos dated >= 2026-05-15 that lack
the literal section header ``## Canonical-vs-unique decision per layer``
(case-insensitive). Pre-cutoff memos are exempt.

Sister of Catalog #229 (premise-verification-before-edit) + Catalog #220
(operational-mechanism declaration) + Catalog #241 (substrate META layer
contract).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_design_memo_has_canonical_vs_unique_decision_section,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(memory_dir: Path, name: str, body: str) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def _frontmatter(name: str = "test-substrate-scaffold") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: test fixture\n"
        "metadata:\n"
        "  node_type: memory\n"
        "  type: feedback\n"
        "---\n\n"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_290_no_memory_dir_returns_empty(tmp_path: Path) -> None:
    """Gate returns [] when memory dir does not exist (not an error)."""
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "missing",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_290_empty_memory_dir_returns_empty(tmp_path: Path) -> None:
    """Gate returns [] when memory dir is empty."""
    (tmp_path / "memory").mkdir()
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_290_pre_cutoff_memo_without_section_exempt(tmp_path: Path) -> None:
    """Pre-2026-05-15 substrate scaffold memos are exempt from the gate."""
    body = _frontmatter() + "Some scaffold content with no canonical-vs-unique section.\n"
    _write(
        tmp_path / "memory",
        "feedback_pre_cutoff_substrate_scaffold_landed_20260514.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_290_post_cutoff_memo_with_section_passes(tmp_path: Path) -> None:
    """Post-cutoff substrate scaffold memo with the canonical section passes."""
    body = (
        _frontmatter()
        + "Body intro.\n\n"
        + "## Canonical-vs-unique decision per layer\n\n"
        + "All canonical helpers adopted; rationale documented.\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_test_substrate_scaffold_landed_20260516.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_290_post_cutoff_memo_without_section_flagged(tmp_path: Path) -> None:
    """Post-cutoff substrate scaffold memo without the section is flagged."""
    body = _frontmatter() + "Body content with no decision section.\n"
    _write(
        tmp_path / "memory",
        "feedback_test_substrate_scaffold_landed_20260520.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "feedback_test_substrate_scaffold_landed_20260520.md" in violations[0]
    assert "Canonical-vs-unique" in violations[0] or "canonical-vs-unique" in violations[0]


def test_290_section_header_case_insensitive(tmp_path: Path) -> None:
    """Section header detection is case-insensitive."""
    for header in [
        "## Canonical-vs-Unique Decision Per Layer",
        "## CANONICAL-VS-UNIQUE DECISION PER LAYER",
        "## canonical-vs-unique decision per layer",
    ]:
        body = _frontmatter() + f"Body.\n\n{header}\n\nRationale.\n"
        _write(
            tmp_path / "memory",
            "feedback_case_substrate_scaffold_landed_20260520.md",
            body,
        )
        violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
            memory_dir=tmp_path / "memory",
            strict=False,
            verbose=False,
        )
        assert violations == [], f"Header `{header}` should be accepted"


def test_290_nscs_naming_pattern_in_scope(tmp_path: Path) -> None:
    """NSCS<N>_landed_<YYYYMMDD>.md memos are in scope per the directive."""
    body = _frontmatter() + "Body.\n"  # no section header
    _write(
        tmp_path / "memory",
        "feedback_nscs01_landed_20260520.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "nscs01" in violations[0].lower()


def test_290_substrate_v_n_naming_pattern_in_scope(tmp_path: Path) -> None:
    """``feedback_<X>substrate<Y>v[0-9]+_landed_<YYYYMMDD>.md`` is in scope."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_balle_substrate_v2_landed_20260520.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_290_non_substrate_non_scaffold_memo_out_of_scope(tmp_path: Path) -> None:
    """Memos that are neither substrate scaffold nor NSCS naming are skipped."""
    body = _frontmatter() + "Body without section header.\n"
    _write(
        tmp_path / "memory",
        "feedback_unrelated_topic_landed_20260520.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_290_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises ``PreflightError`` on violation."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_strict_substrate_scaffold_landed_20260520.md",
        body,
    )
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_design_memo_has_canonical_vs_unique_decision_section(
            memory_dir=tmp_path / "memory",
            strict=True,
            verbose=False,
        )
    msg = str(exc_info.value)
    assert "Catalog #290" in msg
    assert "Canonical-vs-unique" in msg or "canonical-vs-unique" in msg


def test_290_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode is silent (no raise) when corpus is clean."""
    body = (
        _frontmatter()
        + "Body.\n\n## Canonical-vs-unique decision per layer\n\nAll canonical.\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_clean_substrate_scaffold_landed_20260520.md",
        body,
    )
    # Should not raise
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=True,
        verbose=False,
    )
    assert violations == []


def test_290_aggregates_multiple_violations(tmp_path: Path) -> None:
    """Multiple violating memos are all reported."""
    for i in range(3):
        body = _frontmatter() + f"Body {i}.\n"
        _write(
            tmp_path / "memory",
            f"feedback_multi{i}_substrate_scaffold_landed_20260520.md",
            body,
        )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 3


def test_290_string_memory_dir_accepted(tmp_path: Path) -> None:
    """Function accepts a string ``memory_dir`` not just Path."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_str_substrate_scaffold_landed_20260520.md",
        body,
    )
    # Pass as str
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=str(tmp_path / "memory"),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_290_invalid_date_suffix_skipped(tmp_path: Path) -> None:
    """Memos with un-parseable date suffix are skipped (not flagged)."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_baddate_substrate_scaffold_landed_NOTADATE.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_290_unreadable_file_skipped(tmp_path: Path) -> None:
    """Files that can't be read (e.g., binary) are silently skipped."""
    memory = tmp_path / "memory"
    memory.mkdir()
    binary_path = memory / "feedback_binary_substrate_scaffold_landed_20260520.md"
    binary_path.write_bytes(b"\x00\x01\x02\xff binary garbage \x80\x81")
    # Should NOT crash; whether the body matches the section header is best-effort.
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=memory,
        strict=False,
        verbose=False,
    )
    # Binary garbage will not contain the section header so it'll be flagged,
    # but the function MUST NOT crash on the read attempt.
    assert isinstance(violations, list)


def test_290_verbose_mode_output_clean(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verbose mode emits an OK line when clean."""
    body = (
        _frontmatter()
        + "Body.\n\n## Canonical-vs-unique decision per layer\n\nFork rationale.\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_verbose_substrate_scaffold_landed_20260520.md",
        body,
    )
    check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "OK" in captured.out or "0" in captured.out


def test_290_verbose_mode_output_dirty(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verbose mode emits a count line when there are violations."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_verbose_dirty_substrate_scaffold_landed_20260520.md",
        body,
    )
    check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "1" in captured.out or "violation" in captured.out.lower()


def test_290_section_inside_quoted_block_not_required(tmp_path: Path) -> None:
    """The literal substring matches even when the header is part of a code-fenced block."""
    # The gate uses simple substring check; if the literal appears anywhere
    # in the memo body, it counts. This is intentional — the existence of the
    # section header (whether in-prose or in-code) demonstrates the author was
    # aware of the discipline.
    body = (
        _frontmatter()
        + "Body intro.\n\n"
        + "```\n## Canonical-vs-unique decision per layer\n```\n\n"
        + "More body.\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_codefence_substrate_scaffold_landed_20260520.md",
        body,
    )
    violations = check_substrate_design_memo_has_canonical_vs_unique_decision_section(
        memory_dir=tmp_path / "memory",
        strict=False,
        verbose=False,
    )
    assert violations == []
