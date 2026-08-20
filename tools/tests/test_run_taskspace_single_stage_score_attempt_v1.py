from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.witness_control.taskspace_single_stage_score_attempt_v1 import (
    SingleStageScoreAttemptResultV1,
)
from tools import run_taskspace_single_stage_score_attempt_v1 as runner


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--run-id",
        "rank-zero",
        "--resume-from",
        str(tmp_path / "run"),
        "--g121-stage-ledger",
        str(tmp_path / "ledger.jsonl"),
        "--expected-g121-stage-ledger-sha256",
        _sha("ledger"),
        "--g121-attempt-identity-sha256",
        _sha("attempt"),
        "--expected-runtime-tree-sha256",
        _sha("runtime"),
        "--video-names-file",
        str(tmp_path / "names.txt"),
    ]


def test_cli_runs_rank_zero_without_pose_refit_claim(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report = tmp_path / "report.txt"
    receipt = tmp_path / "final.json"
    submission = tmp_path / "submission"
    result = SingleStageScoreAttemptResultV1(
        final_receipt_path=receipt,
        final_receipt_sha256=_sha("final"),
        submission_dir=submission,
        archive_sha256=_sha("archive"),
        archive_bytes=1234,
        report_path=report,
        d_pose=0.01,
        d_seg=0.001,
        recomputed_score=0.15,
        authority_axis="macOS-CPU advisory",
    )
    seen = {}

    def run(*, config, command):
        seen["config"] = config
        seen["command"] = command
        return result

    monkeypatch.setattr(runner, "run_rank_zero_score_attempt", run)
    assert runner.main(_argv(tmp_path)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["archive_sha256"] == result.archive_sha256
    assert output["pose_refit_run"] is False
    assert output["exhaustive_stage_coverage_claim"] is False
    assert output["pareto_claim"] is False
    assert seen["config"].g121_attempt_identity_sha256 == _sha("attempt")
    assert "--resume-from" in seen["command"]


def test_cli_emits_fail_closed_blocker(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    blocker = tmp_path / "blocker.json"

    def fail(**_kwargs):
        raise runner.SingleStageScoreAttemptError("custody drift")

    monkeypatch.setattr(runner, "run_rank_zero_score_attempt", fail)
    monkeypatch.setattr(
        runner,
        "write_blocker_receipt",
        lambda **_kwargs: blocker,
    )
    assert runner.main(_argv(tmp_path)) == 2
    error = capsys.readouterr().err
    assert "REFUSE: custody drift" in error
    assert str(blocker) in error
