from __future__ import annotations

import subprocess

from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionJointDescentTypedConfigV1,
)
from tools.launch_ddm_joint_descent import _expected_full_run_baseline_dseg
from tools.reseal_ddm_j7_366_ticket import (
    REPO,
    _source_commit,
    ws1_launchable_archive,
)


def test_source_commit_is_worktree_head() -> None:
    assert _source_commit() == subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        text=True,
    ).strip()


def test_ws1_receiver_closed_archives_are_launchable_starts() -> None:
    w_seg = ws1_launchable_archive("W_seg")
    w_joint = ws1_launchable_archive("W_joint")

    assert w_seg["kind"] == w_joint["kind"] == "receiver_closed_ws1_archive"
    assert w_seg["bytes"] == 138_031
    assert w_joint["bytes"] == 138_801
    assert w_seg["sha256"] != w_joint["sha256"]
    assert w_seg["optimizer_state_loadable"] is False
    assert w_joint["optimizer_state_loadable"] is False


def test_ws1_tickets_use_their_sealed_exact_baseline() -> None:
    w_seg = DirectDescriptionJointDescentTypedConfigV1.from_ticket(
        REPO / ".omx/research/configs/ddm_ws2_j7_366_w_seg_20260724.json"
    )
    w_joint = DirectDescriptionJointDescentTypedConfigV1.from_ticket(
        REPO / ".omx/research/configs/ddm_ws2_j7_366_w_joint_20260724.json"
    )

    assert _expected_full_run_baseline_dseg(w_seg) == 0.024124510023328993
    assert _expected_full_run_baseline_dseg(w_joint) == 0.07051923116048177
