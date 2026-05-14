# SPDX-License-Identifier: MIT
"""Catalog #185 — `check_strict_flipped_catalog_entries_have_live_count_zero`
tests.

The check refuses CLAUDE.md catalog entries that simultaneously claim
"Live count: 0" + a STRICT-flipped phrase when the underlying gate
function actually returns a non-empty violation list. Sister of
Catalog #159 (text-vs-strict-callsite) and #118 (duplicate-catalog-#).
Together they keep the CLAUDE.md catalog table the canonical strictness
ledger.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
against" non-negotiable. R5-4 anchor 2026-05-13: Catalog #124 description
claimed "Live count: 0 → STRICT" but `preflight --scope all` returned
4 violations. This gate refuses re-introduction of that exact bug class
at every catalog entry that claims STRICT @ 0.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    _CHECK_185_LIVE_COUNT_ZERO_PHRASES,
    _CHECK_185_SKIP_FUNCTIONS,
    PreflightError,
    _check_185_extract_strict_zero_entries,
    check_strict_flipped_catalog_entries_have_live_count_zero,
)


def _write_claude_md(tmp_path: Path, text: str) -> Path:
    (tmp_path / "CLAUDE.md").write_text(text)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: extraction-helper tests (text-shape parsing)
# ─────────────────────────────────────────────────────────────────────────


def test_extract_entry_with_strict_and_zero_count():
    text = """
# Catalog

200. `check_example` — Some description. STRICT-FLIPPED 2026-05-12.
Live count: 0. Memory: feedback_foo.md.
"""
    matched = _check_185_extract_strict_zero_entries(text)
    assert len(matched) == 1
    cat_num, name, body = matched[0]
    assert cat_num == 200
    assert name == "check_example"
    assert "live count: 0" in body.lower()


def test_extract_entry_with_only_zero_count_no_strict_phrase_skipped():
    """Entry claims Live count: 0 but never says STRICT — skipped."""
    text = """
200. `check_example` — Live count: 0. Held warn-only initially.
"""
    matched = _check_185_extract_strict_zero_entries(text)
    assert matched == []


def test_extract_entry_with_only_strict_no_zero_count_skipped():
    """Entry claims STRICT but never says Live count: 0 — skipped."""
    text = """
200. `check_example` — STRICT-FLIPPED 2026-05-12. Memory: foo.md.
"""
    matched = _check_185_extract_strict_zero_entries(text)
    assert matched == []


def test_extract_entry_with_arrow_zero_to_strict():
    """`0 -> STRICT` phrasing counts as both claims."""
    text = """
200. `check_example` — Live count: 0 -> STRICT. Memory: foo.md.
"""
    matched = _check_185_extract_strict_zero_entries(text)
    assert len(matched) == 1


def test_extract_multiple_entries():
    text = """
200. `check_example_one` — STRICT-FLIPPED. Live count: 0.

201. `check_example_two` — Held warn-only initially.

202. `check_example_three` — STRICT @ 0. Live count: 0.
"""
    matched = _check_185_extract_strict_zero_entries(text)
    names = [m[1] for m in matched]
    assert "check_example_one" in names
    assert "check_example_three" in names
    assert "check_example_two" not in names


def test_zero_count_phrases_all_present():
    """All declared phrases must be present in the constant."""
    assert "live count: 0" in _CHECK_185_LIVE_COUNT_ZERO_PHRASES
    assert "0 -> strict" in _CHECK_185_LIVE_COUNT_ZERO_PHRASES


def test_skip_functions_include_self_reference():
    """Self-reference must be in the skip list to prevent recursion."""
    assert (
        "check_strict_flipped_catalog_entries_have_live_count_zero"
        in _CHECK_185_SKIP_FUNCTIONS
    )


def test_skip_functions_include_sister_gate():
    """Sister #159 must be skipped — it's wired strict separately."""
    assert (
        "check_claude_md_catalog_text_matches_preflight_strict_value"
        in _CHECK_185_SKIP_FUNCTIONS
    )


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: missing-file / edge-case behavior
# ─────────────────────────────────────────────────────────────────────────


def test_missing_claude_md_returns_empty(tmp_path):
    """No CLAUDE.md → no violations (defensive)."""
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_empty_claude_md_returns_empty(tmp_path):
    _write_claude_md(tmp_path, "")
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_claude_md_without_catalog_entries_returns_empty(tmp_path):
    _write_claude_md(tmp_path, "# CLAUDE.md\n\nSome free-form text.")
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: clean-pass behavior on the LIVE repo
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_strict_zero_entries_have_zero_violations():
    """Live repo invariant: every STRICT-0 catalog entry must have a
    gate function whose live count is genuinely 0. This is the R5-4
    regression guard.
    """
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Live count drift detected on {len(v)} catalog entry "
        f"(catalog table claims Live count: 0 but gate returned >0):\n"
        + "\n".join(f"  - {x[:240]}" for x in v[:10])
    )


def test_live_repo_strict_mode_passes():
    """Strict-mode call on the LIVE repo must not raise — proves Catalog
    #183's strict-flip is genuine."""
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        strict=True, verbose=False
    )
    assert v == []


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: behavior on synthetic CLAUDE.md content
# ─────────────────────────────────────────────────────────────────────────


def test_strict_zero_claim_with_returning_gate_violates(tmp_path, monkeypatch):
    """If a catalog entry claims STRICT @ 0 + Live count: 0 but the gate
    returns a non-empty list, refuse with a clear message.
    """
    text = """
# Catalog

200. `check_synthetic_drift_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    # Inject a fake gate function into preflight's globals that returns
    # a stale-drift violation.
    import tac.preflight as preflight_mod

    def fake_gate(*, strict=False, verbose=False):
        if strict:
            raise PreflightError("synthetic strict raise")
        return ["synthetic drift violation A", "synthetic drift violation B"]

    monkeypatch.setitem(
        preflight_mod.__dict__, "check_synthetic_drift_example", fake_gate
    )
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "200" in v[0]
    assert "check_synthetic_drift_example" in v[0]
    assert "Live count: 0" in v[0]
    assert "2 violation" in v[0]


def test_strict_zero_claim_with_zero_returning_gate_passes(
    tmp_path, monkeypatch
):
    text = """
# Catalog

200. `check_synthetic_clean_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    import tac.preflight as preflight_mod

    def fake_clean_gate(*, strict=False, verbose=False):
        return []

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_synthetic_clean_example",
        fake_clean_gate,
    )
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_strict_mode_raises_on_drift(tmp_path, monkeypatch):
    text = """
# Catalog

200. `check_synthetic_strict_raise_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    import tac.preflight as preflight_mod

    def fake_drift_gate(*, strict=False, verbose=False):
        return ["drifted"]

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_synthetic_strict_raise_example",
        fake_drift_gate,
    )
    with pytest.raises(PreflightError, match="Live count: 0"):
        check_strict_flipped_catalog_entries_have_live_count_zero(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_self_reference_skipped_no_recursion(tmp_path):
    """Catalog #185 entry itself must be skipped — recursion guard."""
    text = """
# Catalog

185. `check_strict_flipped_catalog_entries_have_live_count_zero` —
STRICT-FLIPPED 2026-05-13. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    # Should NOT raise / recurse infinitely.
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_gate_not_found_in_globals_violates(tmp_path):
    """If the catalog entry names a gate that doesn't exist in
    preflight's globals, fail closed rather than certifying a phantom
    strict gate as Live count: 0."""
    text = """
# Catalog

300. `check_nonexistent_gate_does_not_exist` — STRICT @ 0. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "check_nonexistent_gate_does_not_exist" in v[0]
    assert "no callable gate" in v[0]


def test_skip_list_function_not_invoked(tmp_path, monkeypatch):
    """Functions in _CHECK_185_SKIP_FUNCTIONS must NOT be invoked even
    when their catalog entry claims STRICT @ 0."""
    text = """
# Catalog

159. `check_claude_md_catalog_text_matches_preflight_strict_value` —
STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    # Even if the real #159 returned violations (which it doesn't),
    # #183 must skip it via the skip list.
    call_count = {"n": 0}
    import tac.preflight as preflight_mod

    def counting_proxy(*, strict=False, verbose=False):
        call_count["n"] += 1
        return ["should be skipped"]

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_claude_md_catalog_text_matches_preflight_strict_value",
        counting_proxy,
    )
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []
    assert call_count["n"] == 0, (
        "Skip-list function was invoked despite being in skip list"
    )


def test_strict_mode_passes_with_zero_drift(tmp_path, monkeypatch):
    text = """
# Catalog

200. `check_synthetic_strict_pass_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    import tac.preflight as preflight_mod

    def fake_clean_gate(*, strict=False, verbose=False):
        return []

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_synthetic_strict_pass_example",
        fake_clean_gate,
    )
    # strict=True must not raise when gate returns empty.
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert v == []


def test_gate_raising_typeerror_is_skipped(tmp_path, monkeypatch):
    """If a gate function's signature doesn't accept (strict, verbose)
    kwargs, skip rather than fail."""
    text = """
# Catalog

200. `check_synthetic_typeerror_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    import tac.preflight as preflight_mod

    def fake_signature_mismatch(*, other_kwarg=None):
        return ["should not appear"]

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_synthetic_typeerror_example",
        fake_signature_mismatch,
    )
    v = check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_verbose_output_when_violations(tmp_path, monkeypatch, capsys):
    """Verbose mode prints a violation summary."""
    text = """
# Catalog

200. `check_synthetic_verbose_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    import tac.preflight as preflight_mod

    def fake_drift(*, strict=False, verbose=False):
        return ["drift A"]

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_synthetic_verbose_example",
        fake_drift,
    )
    check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "drift entries" in captured.out
    assert "200" in captured.out


def test_verbose_output_when_clean(tmp_path, monkeypatch, capsys):
    text = """
# Catalog

200. `check_synthetic_verbose_clean_example` — STRICT-FLIPPED. Live count: 0.
"""
    _write_claude_md(tmp_path, text)
    import tac.preflight as preflight_mod

    def fake_clean(*, strict=False, verbose=False):
        return []

    monkeypatch.setitem(
        preflight_mod.__dict__,
        "check_synthetic_verbose_clean_example",
        fake_clean,
    )
    check_strict_flipped_catalog_entries_have_live_count_zero(
        repo_root=tmp_path, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "verified" in captured.out
