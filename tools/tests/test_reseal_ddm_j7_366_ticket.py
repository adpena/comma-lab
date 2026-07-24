from __future__ import annotations

import subprocess

import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.reseal_ddm_j7_366_ticket import REPO, _source_commit, ws1_launchable_archive


def test_source_commit_is_worktree_head() -> None:
    assert _source_commit() == subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        text=True,
    ).strip()


@pytest.mark.parametrize("candidate_id", ["W_seg", "W_joint"])
def test_ws1_endpoint_rows_cannot_be_promoted_to_launchable_starts(
    candidate_id: str,
) -> None:
    with pytest.raises(
        DirectDescriptionError,
        match="WS1_START_NOT_LAUNCHABLE_ENDPOINT_ONLY",
    ):
        ws1_launchable_archive(candidate_id)
