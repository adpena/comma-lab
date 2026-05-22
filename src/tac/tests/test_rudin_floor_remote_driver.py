# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REMOTE_DRIVER = REPO / "scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh"


def test_rudin_remote_driver_requires_active_claim_before_startup() -> None:
    """The Rudin remote wrapper must not rely on post-hoc claim notes."""

    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    assert 'DISPATCH_INSTANCE_JOB_ID="' in text
    assert "RUDIN_FLOOR_DISPATCH_INSTANCE_JOB_ID" in text
    assert "is required for active lane-claim verification" in text
    assert "exit 21" in text

    verify_idx = text.index("verify_active_dispatch_claim()")
    summary_idx = text.index('"$WORKSPACE/tools/claim_lane_dispatch.py" summary')
    live_idx = text.index("--live-only", summary_idx)
    json_idx = text.index("--format json", summary_idx)
    compare_idx = text.index('row.get("lane_id") == lane_id', summary_idx)
    job_idx = text.index('row.get("instance_job_id") == job_id', summary_idx)
    verified_idx = text.index("CLAIM_VERIFIED=1", summary_idx)
    startup_idx = text.index("verify_active_dispatch_claim", text.index("trap cleanup EXIT"))

    assert verify_idx < summary_idx < live_idx < json_idx < compare_idx < job_idx
    assert job_idx < verified_idx < startup_idx
    assert "no active dispatch claim for lane=$LANE_ID" in text
    assert "exit 27" in text


def test_rudin_remote_driver_terminalizes_claim_on_all_exit_paths() -> None:
    """The remote wrapper must close successful and failed claim lifecycles."""

    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    append_idx = text.index("append_terminal_claim()")
    claim_idx = text.index('"$WORKSPACE/tools/claim_lane_dispatch.py" claim', append_idx)
    cleanup_idx = text.index("cleanup()")
    trap_idx = text.index("trap cleanup EXIT")

    assert append_idx < cleanup_idx < trap_idx
    assert "append_terminal_claim \"$rc\"" in text

    append_block = text[append_idx:claim_idx]
    for required in (
        "completed_rudin_floor_remote_driver",
        "failed_rudin_floor_claim_verification_rc_${rc}",
        "failed_rudin_floor_remote_driver_rc_${rc}",
    ):
        assert required in append_block

    claim_block = text[claim_idx:]
    for required in (
        "--force",
        '--lane-id "$LANE_ID"',
        '--platform "$DISPATCH_PLATFORM"',
        '--instance-job-id "$DISPATCH_INSTANCE_JOB_ID"',
        'remote_driver_terminal rc=$rc',
    ):
        assert required in claim_block


def test_rudin_full_dispatch_defaults_to_cuda_device() -> None:
    """Full Modal dispatch must not late-fail through the CPU device guard."""

    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    assert 'RUDIN_FLOOR_DEVICE="${RUDIN_FLOOR_DEVICE:-cuda}"' in text
    assert "--device \"$RUDIN_FLOOR_DEVICE\"" in text
