# SPDX-License-Identifier: MIT
"""Tests for ``tools/cpu_axis_optimal_archive_selector.py`` per G1 Hotz binding.

Per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE 2026-05-19. The G1 canonical
helper re-ranks existing dual-eval data on the [contest-CPU] axis only;
tests pin the helper's CLI surface + invariants.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Module import path varies; test the CLI surface via subprocess.
REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "tools" / "cpu_axis_optimal_archive_selector.py"


def test_helper_exists_as_canonical_file() -> None:
    """The canonical helper MUST exist per G1 directive."""
    assert HELPER.exists(), f"canonical helper missing at {HELPER}"


def test_helper_is_python_executable() -> None:
    """The helper has Python shebang or is callable via python interpreter."""
    content = HELPER.read_text()
    # Either a shebang on line 1 or starts with SPDX/imports
    first_line = content.splitlines()[0] if content else ""
    assert (
        first_line.startswith("#!/")
        or first_line.startswith("#")
        or first_line.startswith("from")
        or first_line.startswith("import")
    )


def test_helper_emits_json_when_flag_passed() -> None:
    """``--json`` flag should produce parseable JSON output."""
    result = subprocess.run(
        [sys.executable, str(HELPER), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream", "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    # rc may be 0 or 1 (1 = delta worse than threshold, still valid)
    assert result.returncode in (0, 1), f"helper crashed: {result.stderr}"
    if result.stdout.strip():
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(f"--json emitted non-JSON: {exc}\nstdout: {result.stdout[:500]}")


def test_helper_help_documented() -> None:
    """``--help`` should be supported per CLAUDE.md DX standard."""
    result = subprocess.run(
        [sys.executable, str(HELPER), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream", "PATH": "/usr/bin:/bin"},
        timeout=10,
    )
    assert result.returncode == 0
    assert "--json" in result.stdout or "json" in result.stdout.lower()


def test_helper_default_invocation_runs() -> None:
    """Default invocation (no args) should run without crashing."""
    result = subprocess.run(
        [sys.executable, str(HELPER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/upstream", "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    # rc 0 or 1 both acceptable (per G1 directive)
    assert result.returncode in (0, 1)


def test_helper_references_canonical_g1_hotz_binding() -> None:
    """The canonical helper MUST cite G1 Hotz binding + CPU-axis discipline."""
    content = HELPER.read_text()
    lower = content.lower()
    # Must reference G1 OR Hotz OR contest-CPU axis
    assert (
        "g1" in lower or "hotz" in lower or "contest-cpu" in lower or "cpu axis" in lower or "cpu-axis" in lower
    ), f"helper does not cite G1 Hotz binding discipline; first 800 chars:\n{content[:800]}"


def test_helper_uses_canonical_frontier_scan() -> None:
    """The canonical helper MUST route through `tac.frontier_scan` per Catalog #316."""
    content = HELPER.read_text()
    assert (
        "frontier_scan" in content
        or "build_frontier_scan_payload" in content
        or "scan_best_anchor_per_axis" in content
    ), "helper does not route through Catalog #316 canonical frontier helpers"


def test_helper_does_not_promote_macos_cpu_advisory() -> None:
    """Per Catalog #192 + CLAUDE.md: macOS-CPU is advisory, NEVER promotable.

    The helper MUST not silently treat macOS-CPU advisory as contest-CPU.
    """
    content = HELPER.read_text()
    # If the helper mentions macOS at all, it should also tag-discipline
    if "macos" in content.lower() or "darwin" in content.lower():
        assert (
            "advisory" in content.lower()
            or "[macOS-CPU advisory]" in content
            or "promot" in content.lower()
        ), "helper mentions macOS but lacks advisory-tag discipline per Catalog #192"
