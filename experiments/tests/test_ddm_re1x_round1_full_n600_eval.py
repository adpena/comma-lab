from __future__ import annotations

from pathlib import Path

import pytest

from experiments import ddm_re1x_round1_full_n600_eval as re1x
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_score_delta_is_sum_of_complete_candidate_terms() -> None:
    base = re1x.score(0.001, 1e-5, 186_252)
    candidate = re1x.score(0.0009, 2e-5, 186_252)
    delta = re1x.score_delta(base, candidate)
    assert delta["total"] == pytest.approx(delta["seg"] + delta["pose"] + delta["rate"])
    assert delta["rate"] == 0.0


@pytest.mark.parametrize(
    ("delta", "base_pose", "candidate_pose", "earned"),
    [
        (-1e-4, 1e-5, 1e-5, True),
        (0.0, 1e-5, 1e-5, False),
        (-1e-4, 1e-5, 1.1e-5, False),
    ],
)
def test_t4_fire_requires_negative_complete_delta_and_pose_held(
    delta: float, base_pose: float, candidate_pose: float, earned: bool
) -> None:
    result = re1x.adjudicate(delta, base_pose, candidate_pose)
    assert result["t4_confirmation_earned"] is earned
    assert result["disposition"] == ("QUEUED-WITH-A-FIRE-ORDER" if earned else "FOLDED")


def test_charter_candidate_and_runtime_tree_are_pinned() -> None:
    archive = re1x.require_file(
        re1x.CANDIDATE_ARCHIVE,
        size=re1x.CANDIDATE_ARCHIVE_BYTES,
        digest=re1x.CANDIDATE_ARCHIVE_SHA256,
    )
    runtime = re1x.require_runtime_tree()
    assert archive["bytes"] == 186_252
    assert runtime["file_count"] == 25
    assert runtime["tree_sha256"] == re1x.CANDIDATE_RUNTIME_TREE_SHA256


def test_runner_does_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=re1x.REPO,
        strict=False,
        roots=["experiments/ddm_re1x_round1_full_n600_eval.py"],
    )
    assert findings == []


def test_blocker_refuses_to_relabel_an_unclassified_receiver_failure(
    tmp_path: Path,
) -> None:
    log = tmp_path / "receivers/re1_round_01_candidate/inflate.log"
    log.parent.mkdir(parents=True)
    log.write_text("an unrelated receiver error\n")
    with pytest.raises(re1x.RE1XEvalError, match="not the known hash-pinned CUDA"):
        re1x.record_public_front_door_blocker(tmp_path, "unrelated")
