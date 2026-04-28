"""Regression tests for Check 41: remote_lane_*.sh heartbeat loop.

Catches the silent-non-start failure mode (W/K/OS-V2 on 2026-04-28).
Reference: feedback_vastai_launch_returns_success_before_lane_starts.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_remote_lane_scripts_have_heartbeat,
)


def _setup_fake_repo(root: Path) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)


def test_strict_passes_on_real_codebase() -> None:
    """Live repo must pass — every remote_lane_*.sh has heartbeat."""
    violations = check_remote_lane_scripts_have_heartbeat(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"Check 41 found {len(violations)} violation(s). Add canonical "
        f"heartbeat loop or exempt sweep orchestrators. Violations:\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


def test_detects_lane_script_missing_heartbeat(tmp_path: Path) -> None:
    """Lane script with no heartbeat reference → flag."""
    _setup_fake_repo(tmp_path)
    bad = tmp_path / "scripts" / "remote_lane_xyz.sh"
    bad.write_text(textwrap.dedent('''\
        #!/bin/bash
        set -euo pipefail
        WORKSPACE=/workspace/pact
        cd $WORKSPACE
        python experiments/run_thing.py
    '''))
    violations = check_remote_lane_scripts_have_heartbeat(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "remote_lane_xyz.sh" in violations[0]
    assert "heartbeat" in violations[0]


def test_passes_with_canonical_heartbeat_pattern(tmp_path: Path) -> None:
    """Canonical pattern from remote_lane_lm_zero_cost_poses.sh template → pass."""
    _setup_fake_repo(tmp_path)
    good = tmp_path / "scripts" / "remote_lane_xyz.sh"
    good.write_text(textwrap.dedent('''\
        #!/bin/bash
        set -euo pipefail
        WORKSPACE=/workspace/pact
        LOG_DIR="$WORKSPACE/lane_xyz_results"
        mkdir -p "$LOG_DIR"
        HEARTBEAT="$LOG_DIR/heartbeat.log"
        ( while true; do
            echo "[$(date)] lane=XYZ tick" >> "$HEARTBEAT"
            sleep 60
          done ) &
        cd $WORKSPACE
    '''))
    assert check_remote_lane_scripts_have_heartbeat(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_passes_with_quoted_log_dir_heartbeat(tmp_path: Path) -> None:
    """Quoted `>> "$LOG_DIR/heartbeat.log"` write → pass (canonical pattern)."""
    _setup_fake_repo(tmp_path)
    good = tmp_path / "scripts" / "remote_lane_xyz.sh"
    good.write_text(textwrap.dedent('''\
        #!/bin/bash
        LOG_DIR=/tmp/lane_xyz
        echo "tick" >> "$LOG_DIR/heartbeat.log" &
    '''))
    assert check_remote_lane_scripts_have_heartbeat(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_sweep_orchestrator_exempt(tmp_path: Path) -> None:
    """*_sweep.sh files are exempt (delegate to per-trial scripts)."""
    _setup_fake_repo(tmp_path)
    sweep = tmp_path / "scripts" / "remote_lane_xyz_sweep.sh"
    sweep.write_text("#!/bin/bash\npython experiments/sweep_xyz.py\n")
    assert check_remote_lane_scripts_have_heartbeat(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_mention_without_write_pattern_flagged(tmp_path: Path) -> None:
    """Comment mentions heartbeat but no actual write → still flagged."""
    _setup_fake_repo(tmp_path)
    bad = tmp_path / "scripts" / "remote_lane_xyz.sh"
    bad.write_text(textwrap.dedent('''\
        #!/bin/bash
        # TODO: add heartbeat loop here
        python experiments/run_thing.py
    '''))
    violations = check_remote_lane_scripts_have_heartbeat(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "no actual heartbeat-write pattern" in violations[0]


def test_strict_raises_metabugviolation(tmp_path: Path) -> None:
    _setup_fake_repo(tmp_path)
    bad = tmp_path / "scripts" / "remote_lane_xyz.sh"
    bad.write_text("#!/bin/bash\necho stub\n")
    with pytest.raises(MetaBugViolation, match="REMOTE LANE SCRIPTS"):
        check_remote_lane_scripts_have_heartbeat(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_only_remote_lane_files_scanned(tmp_path: Path) -> None:
    """Other scripts (setup_full.sh, probe_nvdec.sh, etc.) NOT flagged."""
    _setup_fake_repo(tmp_path)
    # This file is NOT remote_lane_*, should be ignored
    other = tmp_path / "scripts" / "setup_full.sh"
    other.write_text("#!/bin/bash\necho setup\n")
    assert check_remote_lane_scripts_have_heartbeat(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []
