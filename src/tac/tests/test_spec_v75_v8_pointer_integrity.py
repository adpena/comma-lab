"""Tests for check_spec_v75_v8_pointer_integrity (v7.5/v8 SPEC anti-rot gate, #362).

The gate verifies three invariants (all WARN-ONLY at the preflight_all wire-in):
  1. both SPEC files exist;
  2. each SPEC retains all of its load-bearing durable section anchors;
  3. CLAUDE.md names each SPEC by path (the pointer must not orphan the SPEC).

A `# SPEC_POINTER_INTEGRITY_OK:<rationale>` token in CLAUDE.md (non-placeholder
rationale) waives the gate (deliberate SPEC teardown path).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _SPEC_ANTI_ROT_TARGETS,
    _SPEC_V75_REL,
    _SPEC_V8_REL,
    check_spec_v75_v8_pointer_integrity,
)


def _make_repo(
    tmp_path: Path,
    *,
    v75: bool = True,
    v8: bool = True,
    drop_v75_anchor: bool = False,
    drop_v8_anchor: bool = False,
    pointer_v75: bool = True,
    pointer_v8: bool = True,
    waiver: str | None = None,
    claude: bool = True,
) -> Path:
    """Build a minimal fake repo satisfying (or violating) the invariants."""
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    anchors_by_rel = {rel: anchors for rel, anchors in _SPEC_ANTI_ROT_TARGETS}

    def _write_spec(rel: str, present: bool, drop_anchor: bool) -> None:
        if not present:
            return
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        anchors = list(anchors_by_rel[rel])
        if drop_anchor:
            anchors = anchors[:-1]  # drop the last load-bearing header
        body = ["# SPEC", ""]
        for a in anchors:
            body += [a, "", "content", ""]
        path.write_text("\n".join(body), encoding="utf-8")

    _write_spec(_SPEC_V75_REL, v75, drop_v75_anchor)
    _write_spec(_SPEC_V8_REL, v8, drop_v8_anchor)

    if claude:
        lines = ["# AGENTS", "", "## THE v7.5 / v8 VEHICLE LINE", ""]
        if pointer_v75:
            lines.append(f"- `{_SPEC_V75_REL}` — sealed constants")
        if pointer_v8:
            lines.append(f"- `{_SPEC_V8_REL}` — edge-centric carriers")
        if waiver is not None:
            lines.append(f"pointer body # SPEC_POINTER_INTEGRITY_OK:{waiver}")
        (root / "CLAUDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_live_repo_passes() -> None:
    """The real repo must be clean (both SPECs + both anchors + both pointers)."""
    violations = check_spec_v75_v8_pointer_integrity(strict=False, verbose=False)
    assert violations == []


def test_complete_fake_repo_passes(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert check_spec_v75_v8_pointer_integrity(repo_root=root) == []


def test_missing_v75_spec_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, v75=False)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("MISSING" in v and _SPEC_V75_REL in v for v in violations)


def test_missing_v8_spec_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, v8=False)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("MISSING" in v and _SPEC_V8_REL in v for v in violations)


def test_dropped_v75_anchor_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, drop_v75_anchor=True)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("section anchor" in v and _SPEC_V75_REL in v for v in violations)


def test_dropped_v8_anchor_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, drop_v8_anchor=True)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("section anchor" in v and _SPEC_V8_REL in v for v in violations)


def test_missing_v75_pointer_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, pointer_v75=False)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("pointer to" in v and _SPEC_V75_REL in v for v in violations)


def test_missing_v8_pointer_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, pointer_v8=False)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("pointer to" in v and _SPEC_V8_REL in v for v in violations)


def test_missing_claude_md_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, claude=False)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("CLAUDE.md" in v and "MISSING" in v for v in violations)


def test_waiver_suppresses_violations(tmp_path: Path) -> None:
    # both SPECs missing + both pointers absent, but a real waiver suppresses all.
    root = _make_repo(
        tmp_path, v75=False, v8=False, pointer_v75=False, pointer_v8=False,
        waiver="deliberate v7.5/v8 retirement per operator 2026-07-09")
    assert check_spec_v75_v8_pointer_integrity(repo_root=root) == []


def test_placeholder_waiver_rejected(tmp_path: Path) -> None:
    # a placeholder rationale must NOT waive (Catalog #287 sister discipline).
    root = _make_repo(tmp_path, v75=False, waiver="<rationale>")
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root)
    assert any("MISSING" in v and _SPEC_V75_REL in v for v in violations)


def test_strict_raises_on_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, v75=False, v8=False)
    with pytest.raises(PreflightError):
        check_spec_v75_v8_pointer_integrity(repo_root=root, strict=True)


def test_non_strict_returns_all_violations(tmp_path: Path) -> None:
    # v75 missing + v8 missing + both pointers absent = 4 independent violations.
    root = _make_repo(
        tmp_path, v75=False, v8=False, pointer_v75=False, pointer_v8=False)
    violations = check_spec_v75_v8_pointer_integrity(repo_root=root, strict=False)
    assert len(violations) == 4
