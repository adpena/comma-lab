# SPDX-License-Identifier: MIT
"""Tests for Catalog #235 - check_no_sha_prefix_length_mismatch_comparisons.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
META-4 R1 self-protection (Catalog #117 short_sha[:7] in seen_hashes(9-char) bug).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac import preflight as pf
from tac.preflight import (
    _CHECK_235_FORBIDDEN_PATTERNS,
    REPO_ROOT,
    PreflightError,
    check_no_sha_prefix_length_mismatch_comparisons,
)


def test_check_235_returns_list() -> None:
    """Gate must always return a list."""
    result = check_no_sha_prefix_length_mismatch_comparisons(
        strict=False, verbose=False
    )
    assert isinstance(result, list)


def test_check_235_forbidden_patterns_constant() -> None:
    """The forbidden patterns tuple must contain the canonical bug patterns."""
    pattern_strings = [p[0] for p in _CHECK_235_FORBIDDEN_PATTERNS]
    assert "[:7] in seen_hashes" in pattern_strings
    assert "[:7] in seen_shas" in pattern_strings
    assert "[:7] in serializer_shas" in pattern_strings


def test_check_235_live_repo_clean() -> None:
    """The live repo must currently pass Catalog #235 strict-mode.

    Per CLAUDE.md "Strict-flip atomicity rule" — the gate lands STRICT @ 0
    because the canonical bug is already fixed in sister commit 4d483ac39
    via _check_117_serializer_sha_matches_commit. The 3 self-references in
    the gate's pattern tuple definition carry the SHA_PREFIX_LENGTH_MISMATCH_OK
    waiver.
    """
    result = check_no_sha_prefix_length_mismatch_comparisons(
        strict=False, verbose=False
    )
    assert result == [], (
        f"Catalog #235 STRICT must remain at 0 live count. "
        f"Found {len(result)} violation(s); fix or waive: {result}"
    )


def test_check_235_strict_mode_silent_on_clean() -> None:
    """When strict=True and live count = 0, the gate must NOT raise."""
    # The live repo IS clean (no forbidden patterns outside waivers).
    check_no_sha_prefix_length_mismatch_comparisons(strict=True, verbose=False)


def test_check_235_strict_mode_raises_on_violation(tmp_path) -> None:
    """When strict=True and there are violations, the gate must raise."""
    target = tmp_path / "src" / "tac"
    target.mkdir(parents=True)
    (target / "preflight.py").write_text(
        "def bad(short_sha, seen_hashes):\n"
        "    return short_sha[:7] in seen_hashes\n",
        encoding="utf-8",
    )

    with pytest.raises(PreflightError):
        check_no_sha_prefix_length_mismatch_comparisons(
            strict=True,
            verbose=False,
            repo_root=tmp_path,
        )


def test_check_235_waiver_token_honored() -> None:
    """The SHA_PREFIX_LENGTH_MISMATCH_OK waiver must exempt a line from
    flagging.

    The 3 self-references in `_CHECK_235_FORBIDDEN_PATTERNS` themselves
    carry this waiver — proving the waiver mechanism works on the live
    repo state.
    """
    target = Path(REPO_ROOT) / "src" / "tac" / "preflight.py"
    text = target.read_text(encoding="utf-8")

    # The 3 self-reference pattern lines must contain the waiver token.
    waiver_count = text.count("SHA_PREFIX_LENGTH_MISMATCH_OK")
    # 1 in the docstring (self-reference) + 1 in the gate body (waiver detection)
    # + 3 in the pattern tuple (the actual waivers being tested).
    assert waiver_count >= 4, (
        f"Expected >= 4 SHA_PREFIX_LENGTH_MISMATCH_OK markers in preflight.py, "
        f"found {waiver_count}"
    )


def test_check_235_orchestrator_callsite_strict_true() -> None:
    """Per CLAUDE.md 'Strict-flip atomicity rule' Catalog #235 lands STRICT
    @ 0 because the canonical bug is structurally extincted by sister
    Catalog #117 fix.
    """
    src = inspect.getsource(pf.preflight_all)
    saw_call = False
    for line in src.split("\n"):
        if (
            "check_no_sha_prefix_length_mismatch_comparisons(" in line
            and "def " not in line
        ):
            saw_call = True
            idx = src.find(line)
            window = src[idx : idx + 200]
            assert "strict=True" in window, (
                "Catalog #235 wire-in must be strict=True at landing per "
                "CLAUDE.md 'Strict-flip atomicity rule' (live count = 0)"
            )
    assert saw_call, (
        "Catalog #235 gate must be wired into preflight_all() per "
        "CLAUDE.md 'Operator gates must be wired and used' non-negotiable."
    )
