"""Tests for ``check_no_inflate_time_multipass`` (Check 92).

The check forbids any reference to Lane 8's ``MultiPassCompressor`` /
``compress_with_multipass`` from inflate-side files
(submissions/robust_current/inflate_renderer.py + inflate.sh +
submissions/exact_current/inflate.py + inflate.sh).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_no_inflate_time_multipass,
)


# ── current-codebase invariant: 0 violations ─────────────────────────────────


def test_check_no_inflate_time_multipass_clean_on_real_codebase() -> None:
    """The shipped Lane 8 implementation lives only in compress-time files
    (`src/tac/multipass_compressor.py`, `experiments/pipeline.py`). No
    inflate-side file references the symbol.
    """
    violations = check_no_inflate_time_multipass(strict=True, verbose=False)
    assert violations == [], (
        f"Lane 8 MultiPassCompressor leaked into an inflate-time file: "
        f"{violations}. Multi-pass is COMPRESS-time only — strict-scorer-rule "
        f"per CLAUDE.md."
    )


# ── synthetic violation: forbidden token in a fake inflate file ──────────────


def _seed_synthetic_repo(td: Path, inflate_renderer_text: str) -> None:
    """Create a minimal directory tree that mimics the real repo layout
    expected by the check (paths relative to repo root).
    """
    (td / "submissions" / "robust_current").mkdir(parents=True, exist_ok=True)
    (td / "submissions" / "robust_current" / "inflate_renderer.py").write_text(
        inflate_renderer_text
    )


def test_synthetic_violation_fires_on_strict() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _seed_synthetic_repo(
            td_path,
            "from tac.multipass_compressor import MultiPassCompressor\n"
            "MultiPassCompressor(target_score=0.5, max_passes=3, eps=1e-3)\n",
        )
        violations = check_no_inflate_time_multipass(
            repo_root=td_path, strict=False, verbose=False,
        )
        assert len(violations) >= 1
        with pytest.raises(MetaBugViolation, match="strict-scorer-rule"):
            check_no_inflate_time_multipass(
                repo_root=td_path, strict=True, verbose=False,
            )


def test_synthetic_comment_reference_does_not_fire() -> None:
    """A documentation reference in a comment is exempt."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _seed_synthetic_repo(
            td_path,
            "# DO NOT IMPORT MultiPassCompressor at inflate time.\n"
            "x = 1\n",
        )
        violations = check_no_inflate_time_multipass(
            repo_root=td_path, strict=True, verbose=False,
        )
        assert violations == []


def test_synthetic_same_line_waiver_exempts() -> None:
    """A same-line ``STRICT_PREFLIGHT_WAIVED:`` annotation exempts the line.

    Used for explicit operator-approved overrides (e.g., a documented
    compliance ruling). The waiver MUST be on the same source line.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _seed_synthetic_repo(
            td_path,
            "x = MultiPassCompressor  # STRICT_PREFLIGHT_WAIVED: doc-ref only\n",
        )
        violations = check_no_inflate_time_multipass(
            repo_root=td_path, strict=True, verbose=False,
        )
        assert violations == []


def test_synthetic_compress_with_multipass_helper_token_fires() -> None:
    """The functional wrapper ``compress_with_multipass`` is also forbidden."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _seed_synthetic_repo(
            td_path,
            "result = compress_with_multipass(state, encoder, scorer)\n",
        )
        with pytest.raises(MetaBugViolation):
            check_no_inflate_time_multipass(
                repo_root=td_path, strict=True, verbose=False,
            )


def test_check_returns_violations_list_on_strict_false() -> None:
    """Non-strict mode returns the violations list without raising."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _seed_synthetic_repo(
            td_path,
            "from tac.multipass_compressor import MultiPassCompressor\n",
        )
        v = check_no_inflate_time_multipass(
            repo_root=td_path, strict=False, verbose=False,
        )
        assert isinstance(v, list)
        assert len(v) >= 1
        # Either the MultiPassCompressor symbol OR the import path token
        # should be flagged on the line — at least one violation message
        # should reference one of the forbidden tokens.
        all_msgs = " ".join(v)
        assert (
            "MultiPassCompressor" in all_msgs
            or "from tac.multipass_compressor" in all_msgs
        )


def test_check_handles_missing_inflate_files() -> None:
    """If the inflate paths don't exist, the check returns 0 violations
    rather than crashing.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Empty repo — no inflate files.
        v = check_no_inflate_time_multipass(
            repo_root=td_path, strict=True, verbose=False,
        )
        assert v == []
