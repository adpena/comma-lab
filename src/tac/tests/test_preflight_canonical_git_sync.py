# SPDX-License-Identifier: MIT
"""Tests for Check 57: lane scripts must use canonical git sync.

Memory: feedback_canonical_git_sync_pattern_20260428
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_lane_scripts_use_canonical_git_sync,
)


def _make_repo(tmp_path: Path, scripts: dict[str, str]) -> Path:
    """Build a fake repo with scripts/<name>.sh files. Returns repo root."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name, body in scripts.items():
        (scripts_dir / name).write_text(textwrap.dedent(body).lstrip("\n"))
    return tmp_path


def test_check_passes_when_canonical_pattern_present(tmp_path: Path) -> None:
    """A lane script using `git fetch origin main && git reset --hard origin/main`
    passes Check 57 with zero violations."""
    repo = _make_repo(tmp_path, {
        "remote_lane_alpha.sh": """
            #!/bin/bash
            set -euo pipefail
            cd /workspace/pact
            git fetch origin main && git reset --hard origin/main
            python3 -u -m pip install -e .
        """,
    })
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"expected 0 violations, got: {violations}"


def test_check_fails_when_only_git_pull_present(tmp_path: Path) -> None:
    """A lane script using bare `git pull --ff-only` (no canonical fallback)
    fails Check 57 LOUDLY in strict mode."""
    repo = _make_repo(tmp_path, {
        "remote_lane_beta.sh": """
            #!/bin/bash
            set -euo pipefail
            cd /workspace/pact
            git pull --ff-only
            python3 -u -m pip install -e .
        """,
    })
    # Non-strict: returns violations.
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) >= 1, f"expected >=1 violation, got: {violations}"
    assert any("git pull --ff-only" in v for v in violations)

    # Strict: raises.
    with pytest.raises(MetaBugViolation, match="CANONICAL GIT SYNC VIOLATIONS"):
        check_lane_scripts_use_canonical_git_sync(
            repo_root=repo, strict=True, verbose=False,
        )


def test_check_passes_when_waiver_present(tmp_path: Path) -> None:
    """A bare `git pull --ff-only` line WITH a same-line
    `# GIT_SYNC_OPT_OUT:<reason>` waiver is allowed."""
    repo = _make_repo(tmp_path, {
        "remote_lane_gamma.sh": """
            #!/bin/bash
            set -euo pipefail
            cd /workspace/pact
            git pull --ff-only  # GIT_SYNC_OPT_OUT: legacy lane preserves prior behavior
            python3 -u -m pip install -e .
        """,
    })
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"expected 0 violations with waiver, got: {violations}"


def test_check_skips_non_lane_scripts(tmp_path: Path) -> None:
    """Scripts that don't match `remote_lane_*.sh` are ignored even if they
    use the forbidden pattern (e.g., remote_setup_full.sh has different
    responsibilities)."""
    repo = _make_repo(tmp_path, {
        "remote_setup_full.sh": """
            #!/bin/bash
            git pull --ff-only
        """,
        "build_lane_thing.sh": """
            #!/bin/bash
            git pull --ff-only
        """,
        "remote_lane_delta.sh": """
            #!/bin/bash
            git fetch origin main && git reset --hard origin/main
        """,
    })
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], (
        f"non-lane scripts should be ignored, got: {violations}"
    )


def test_check_wired_into_preflight_all() -> None:
    """Check 57 must be wired into preflight_all() at STRICT level so it
    fails commit/PR-time when violated. Verified by source inspection."""
    import tac.preflight as preflight_mod
    src = Path(preflight_mod.__file__).read_text()
    # The wiring must invoke check_lane_scripts_use_canonical_git_sync with
    # strict=True somewhere in the preflight_all body.
    assert "check_lane_scripts_use_canonical_git_sync(" in src, (
        "Check 57 not wired into preflight.py at all"
    )
    # Find the function definition end + the wiring call.
    # We check that there's a call site OUTSIDE the function definition.
    call_count = src.count("check_lane_scripts_use_canonical_git_sync(")
    # Definition + wiring call = at least 2 occurrences.
    assert call_count >= 2, (
        f"Expected >=2 occurrences (def + wiring), got {call_count}"
    )
    # And the wiring must use strict=True (search the wiring block).
    # Find the wiring call (not the def line).
    def_idx = src.find("def check_lane_scripts_use_canonical_git_sync(")
    pre_def = src[:def_idx]
    assert "check_lane_scripts_use_canonical_git_sync(" in pre_def, (
        "wiring call must appear BEFORE the function definition (in "
        "preflight_all body)"
    )
    # Find the wiring call in pre_def and check strict=True nearby.
    wiring_idx = pre_def.find("check_lane_scripts_use_canonical_git_sync(")
    snippet = pre_def[wiring_idx:wiring_idx + 200]
    assert "strict=True" in snippet, (
        f"wiring must use strict=True; got snippet: {snippet!r}"
    )


def test_check_handles_git_C_workspace_form(tmp_path: Path) -> None:
    """The `git -C "$WORKSPACE" fetch origin main` form must be accepted
    (j_imp_iterative_magnitude_pruning.sh uses this form)."""
    repo = _make_repo(tmp_path, {
        "remote_lane_jimp.sh": """
            #!/bin/bash
            WORKSPACE=/workspace/pact
            git -C "$WORKSPACE" fetch origin main && git -C "$WORKSPACE" reset --hard origin/main
        """,
    })
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], (
        f"git -C form should be accepted, got: {violations}"
    )


def test_check_passes_when_no_git_sync_at_all(tmp_path: Path) -> None:
    """Lane scripts that do no git sync at all (trust parent launcher)
    are exempt."""
    repo = _make_repo(tmp_path, {
        "remote_lane_zeta.sh": """
            #!/bin/bash
            set -euo pipefail
            cd /workspace/pact
            python3 -u experiments/run_lane_zeta.py
        """,
    })
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], (
        f"no-git-sync lane should be exempt, got: {violations}"
    )


def test_check_skips_pure_comment_git_pull_lines(tmp_path: Path) -> None:
    """A line containing `git pull --ff-only` only as documentation in a
    pure-comment line should NOT trigger the violation."""
    repo = _make_repo(tmp_path, {
        "remote_lane_eta.sh": """
            #!/bin/bash
            # Stage 1: replaces fragile `git pull --ff-only` with canonical sync
            git fetch origin main && git reset --hard origin/main
        """,
    })
    violations = check_lane_scripts_use_canonical_git_sync(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], (
        f"pure-comment doc lines should not violate, got: {violations}"
    )
