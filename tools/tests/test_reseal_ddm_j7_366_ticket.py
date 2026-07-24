from __future__ import annotations

import subprocess

from tools.reseal_ddm_j7_366_ticket import REPO, _source_commit, ws1_launchable_archive


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
