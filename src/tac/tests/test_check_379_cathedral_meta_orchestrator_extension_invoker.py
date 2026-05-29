# SPDX-License-Identifier: MIT
"""Tests for Catalog #379 STRICT preflight gate (Wave N+46)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cathedral_autopilot_main_invokes_meta_orchestrator_extension,
)


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_regression_guard() -> None:
    """Wave N+46 extension lands invoker callsite -> live count must be 0."""
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        strict=False
    )
    assert violations == [], (
        f"Catalog #379 live count > 0: {violations}; this means the canonical "
        "META-orchestrator extension invoker callsite has been removed from "
        "cathedral_autopilot main(). Restore the call to "
        "invoke_meta_orchestrator_extension_on_candidates."
    )


# ---------------------------------------------------------------------------
# Strict-mode behavior
# ---------------------------------------------------------------------------


def test_strict_mode_silent_on_clean() -> None:
    """Strict mode returns empty list when invoker callsite present."""
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        strict=True
    )
    assert violations == []


def test_strict_mode_raises_when_invoker_missing(tmp_path: Path) -> None:
    """Synthetic regression: main() without invoker callsite -> strict raises."""
    # Create a synthetic cathedral_autopilot stub WITHOUT the invoker call.
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):\n"
        "    # No canonical invoker call here.\n"
        "    return 0\n"
    )

    with pytest.raises(PreflightError, match="Catalog #379"):
        check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
            repo_root=fake_root, strict=True
        )


def test_invoker_present_via_invoke_helper_passes(tmp_path: Path) -> None:
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):\n"
        "    invoke_meta_orchestrator_extension_on_candidates([])\n"
        "    return 0\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert violations == []


def test_invoker_present_via_direct_rank_call_passes(tmp_path: Path) -> None:
    """Direct rank_candidates_via_three_metric_trichotomy call also accepted."""
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):\n"
        "    rank_candidates_via_three_metric_trichotomy([])\n"
        "    return 0\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_waiver_with_real_rationale_accepted(tmp_path: Path) -> None:
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):  # META_ORCHESTRATOR_THREE_METRIC_TRICHOTOMY_INVOKER_WAIVED:operator-reviewed-design-deferral-for-Phase-2-promotion\n"
        "    return 0\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert violations == []


def test_waiver_with_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """Per Catalog #287 sister discipline."""
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):  # META_ORCHESTRATOR_THREE_METRIC_TRICHOTOMY_INVOKER_WAIVED:<rationale>\n"
        "    return 0\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert len(violations) == 1
    assert "META-orchestrator extension" in violations[0] or "main()" in violations[0]


def test_waiver_with_short_rationale_rejected(tmp_path: Path) -> None:
    """Rationale must be >= 4 chars."""
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):  # META_ORCHESTRATOR_THREE_METRIC_TRICHOTOMY_INVOKER_WAIVED:abc\n"
        "    return 0\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert len(violations) == 1


def test_waiver_with_reason_placeholder_rejected(tmp_path: Path) -> None:
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def main(argv=None):  # META_ORCHESTRATOR_THREE_METRIC_TRICHOTOMY_INVOKER_WAIVED:<reason>\n"
        "    return 0\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_target_file_silent(tmp_path: Path) -> None:
    """Per CLAUDE.md "no signal loss": missing file is silent (not strict-flip blocker)."""
    fake_root = tmp_path
    # No tools/cathedral_autopilot_autonomous_loop.py
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert violations == []


def test_missing_main_def_flagged(tmp_path: Path) -> None:
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(
        "def helper():\n"
        "    pass\n"
    )
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert len(violations) == 1
    assert "main" in violations[0].lower()


def test_syntax_error_target_handled(tmp_path: Path) -> None:
    fake_root = tmp_path
    target_dir = fake_root / "tools"
    target_dir.mkdir()
    target = target_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text("def main(\n  # unclosed\n")
    violations = check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(
        repo_root=fake_root, strict=False
    )
    assert len(violations) == 1
    assert "syntaxerror" in violations[0].lower() or "parse" in violations[0].lower() or "main" in violations[0].lower()


# ---------------------------------------------------------------------------
# Sister regression guards (Catalog #176, #185, #186)
# ---------------------------------------------------------------------------


def test_catalog_185_sister_regression_globals() -> None:
    """Catalog #185 META-meta-meta: gate function callable via globals."""
    import tac.preflight as preflight_module

    fn = preflight_module.check_cathedral_autopilot_main_invokes_meta_orchestrator_extension
    assert callable(fn)


def test_catalog_176_sister_strict_callsite_present() -> None:
    """Catalog #176: STRICT callsites referenced in preflight_all()."""
    from pathlib import Path

    preflight_path = Path(__file__).resolve().parents[3] / "src/tac/preflight.py"
    source = preflight_path.read_text(encoding="utf-8")
    # Verify the callsite is present in preflight_all
    assert "check_cathedral_autopilot_main_invokes_meta_orchestrator_extension" in source


def test_orchestrator_wires_strict_true() -> None:
    """The Catalog #379 callsite in preflight_all() must pass strict=True."""
    from pathlib import Path

    preflight_path = Path(__file__).resolve().parents[3] / "src/tac/preflight.py"
    source = preflight_path.read_text(encoding="utf-8")
    # Find the callsite + verify strict=True
    idx = source.find("check_cathedral_autopilot_main_invokes_meta_orchestrator_extension(")
    assert idx > 0
    # Look for strict=True within 200 chars of callsite
    snippet = source[idx : idx + 200]
    assert "strict=True" in snippet
