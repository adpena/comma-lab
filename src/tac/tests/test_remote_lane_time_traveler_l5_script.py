# SPDX-License-Identifier: MIT
"""Regression tests for the Time-Traveler L5 remote driver custody path."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/remote_lane_substrate_time_traveler_l5_autonomy.sh"


def test_tt5l_remote_driver_bash_syntax_clean() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_tt5l_remote_driver_verifies_active_claim_and_terminalizes() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary' in text
    assert 'payload.get("active", [])' in text
    assert 'row.get("lane_id") == lane_id' in text
    assert 'row.get("instance_job_id") == job_id' in text
    assert "stage_0_dispatch_claim_verified" in text
    assert "CLAIM_VERIFIED=1" in text
    assert "append_terminal_claim()" in text
    assert "completed_tt5l_remote_driver" in text
    assert "failed_tt5l_claim_verification_rc_${rc}" in text
    assert "failed_tt5l_remote_driver_rc_${rc}" in text
    assert "--force" in text
    assert "trap cleanup EXIT" in text
    assert "trap 'if [ -n \"$HEARTBEAT_PID\" ]" not in text
