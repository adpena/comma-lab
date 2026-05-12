"""Catalog #159 — `check_claude_md_catalog_text_matches_preflight_strict_value` tests.

The check refuses CLAUDE.md catalog entries whose strictness text claim
contradicts the orchestrator callsite `strict=` value. Sister of
Catalog #118 — together they keep CLAUDE.md catalog table the canonical
strictness ledger. Per CLAUDE.md "Bugs must be permanently fixed AND
self-protected against" non-negotiable.

UUU's audit 2026-05-12 identified 23 drift entries; FFFF Bug 3 +
FFFF Bug 4 (Path A + Path B) drove drift to 0 + landed self-protection.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_claude_md_catalog_text_matches_preflight_strict_value,
    _check_159_extract_catalog_entries,
    _check_159_extract_callsite_strict,
)


def _write_files(tmp_path: Path, claude_md: str, preflight_py: str) -> Path:
    """Write CLAUDE.md + src/tac/preflight.py to tmp_path."""
    (tmp_path / "CLAUDE.md").write_text(claude_md)
    pf_dir = tmp_path / "src" / "tac"
    pf_dir.mkdir(parents=True, exist_ok=True)
    (pf_dir / "preflight.py").write_text(preflight_py)
    return tmp_path


def test_claim_strict_code_strict_passes(tmp_path):
    claude_md = """
# foo

200. `check_example` — Some thing. STRICT-FLIPPED 2026-05-12. Live count: 0.
"""
    preflight_py = """
def check_example():
    pass

if True:
    check_example(strict=True, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_claim_warn_code_strict_violates(tmp_path):
    claude_md = """
# foo

200. `check_example` — Held warn-only initially; flip to STRICT after baseline.
"""
    preflight_py = """
def check_example():
    pass

if True:
    check_example(strict=True, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "200" in v[0]
    assert "warn-only" in v[0]


def test_claim_strict_code_warn_violates(tmp_path):
    claude_md = """
# foo

200. `check_example` — STRICT-FLIPPED. Live count: 0.
"""
    preflight_py = """
def check_example():
    pass

if True:
    check_example(strict=False, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "200" in v[0]
    assert "strict" in v[0].lower()


def test_ambiguous_no_strictness_claim_accepted(tmp_path):
    claude_md = """
# foo

200. `check_example` — Some thing. Live count: 0. Memory: foo.md.
"""
    preflight_py = """
def check_example():
    pass

if True:
    check_example(strict=True, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_warn_initially_then_strict_flipped_passes(tmp_path):
    claude_md = """
# foo

200. `check_example` — Held warn-only initially; STRICT-FLIPPED 2026-05-12 per backfill.
"""
    preflight_py = """
def check_example():
    pass

if True:
    check_example(strict=True, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_not_invoked_check_skipped(tmp_path):
    claude_md = """
# foo

200. `check_orphan` — Some text claiming STRICT @ 0.
"""
    preflight_py = """
def check_orphan():
    pass

# function exists but not called from orchestrator
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_strict_mode_raises(tmp_path):
    claude_md = """
# foo

200. `check_example` — Held warn-only initially.
"""
    preflight_py = """
def check_example():
    pass

if True:
    check_example(strict=True, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    with pytest.raises(PreflightError):
        check_claude_md_catalog_text_matches_preflight_strict_value(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_extract_catalog_entries_parses_numbered_entries():
    text = """
# foo

1. `check_a` — entry A body line 1
   continuation of A.

2. `check_b` — entry B.

**Section header:**

3. `check_c` — entry C.
"""
    entries = _check_159_extract_catalog_entries(text)
    nums = [n for n, _, _ in entries]
    assert 1 in nums
    assert 2 in nums
    assert 3 in nums


def test_extract_callsite_strict_finds_strict_true():
    pf = """
def check_x():
    pass

def main():
    check_x(strict=True, verbose=False)
"""
    assert _check_159_extract_callsite_strict(pf, "check_x") is True


def test_extract_callsite_strict_finds_strict_false():
    pf = """
def check_x():
    pass

def main():
    check_x(strict=False, verbose=False)
"""
    assert _check_159_extract_callsite_strict(pf, "check_x") is False


def test_extract_callsite_strict_no_invocation_returns_none():
    pf = """
def check_x():
    pass
"""
    assert _check_159_extract_callsite_strict(pf, "check_x") is None


def test_multiline_invocation_resolves_strict():
    pf = """
def check_x():
    pass

def main():
    check_x(
        strict=True,
        verbose=False,
    )
"""
    assert _check_159_extract_callsite_strict(pf, "check_x") is True


def test_live_repo_clean():
    """Live repo must be clean — drift extincted by FFFF Bug 3."""
    repo_root = Path(__file__).resolve().parents[3]
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert v == [], f"live repo has catalog drift: {v}"


def test_missing_claude_md_returns_empty(tmp_path):
    """No CLAUDE.md → returns []."""
    pf_dir = tmp_path / "src" / "tac"
    pf_dir.mkdir(parents=True, exist_ok=True)
    (pf_dir / "preflight.py").write_text("def check_x():\n    pass\n")
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_multiple_drift_entries_all_flagged(tmp_path):
    claude_md = """
# foo

100. `check_a` — Held warn-only initially.
101. `check_b` — Currently warn-only.
102. `check_c` — Held warn-only initially.
"""
    preflight_py = """
def check_a():
    pass

def check_b():
    pass

def check_c():
    pass

def main():
    check_a(strict=True, verbose=False)
    check_b(strict=True, verbose=False)
    check_c(strict=True, verbose=False)
"""
    _write_files(tmp_path, claude_md, preflight_py)
    v = check_claude_md_catalog_text_matches_preflight_strict_value(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 3
    assert any("100" in x for x in v)
    assert any("101" in x for x in v)
    assert any("102" in x for x in v)
