# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.local_exact_auth_gate import (
    LOCAL_EXACT_AUTH_GATE_SCHEMA,
    LocalExactAuthGateConfig,
    build_local_exact_auth_gate_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "gate_local_candidate_for_exact_auth.py"


def _local_summary(*, score: float = 0.18) -> dict[str, object]:
    return {
        "schema": "local_submission_replay.v1",
        "evaluation_passed": True,
        "axis_tag": "[macOS-CPU advisory]",
        "local_score_estimate": score,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _mlx_summary(*, action: float = 0.17) -> dict[str, object]:
    return {
        "schema": "z8_full_video_mlx_replay.v1",
        "local_axis": "[macOS-MLX research-signal]",
        "full_video_local_replay_executed": True,
        "full_video_local_replay_scope": "full_video",
        "replay_ok": True,
        "contest_action_proxy": action,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_gate_recommends_exact_cpu_only_after_local_cpu_win() -> None:
    report = build_local_exact_auth_gate_report(
        local_replay_summary=_local_summary(score=0.18),
        mlx_prefilter_summary=_mlx_summary(action=0.17),
        config=LocalExactAuthGateConfig(
            auth_target_score=0.19,
            require_mlx_prefilter=True,
            mlx_target_action=0.18,
        ),
    )

    assert report.schema == LOCAL_EXACT_AUTH_GATE_SCHEMA
    assert report.exact_auth_dispatch_recommended is True
    assert report.exact_cpu_dispatch_recommended is True
    assert report.ready_for_exact_eval_dispatch is False
    assert report.score_claim is False
    assert report.next_required_action == "claim_lane_and_run_exact_cpu_auth_eval"
    assert report.blockers == []


def test_gate_blocks_local_advisory_score_that_does_not_clear_target() -> None:
    report = build_local_exact_auth_gate_report(
        local_replay_summary=_local_summary(score=0.20),
        config=LocalExactAuthGateConfig(auth_target_score=0.19),
    )

    assert report.exact_auth_dispatch_recommended is False
    assert report.exact_cpu_dispatch_recommended is False
    assert report.ready_for_exact_eval_dispatch is False
    assert "local_score_not_below_auth_target" in report.blockers
    assert report.next_required_action == "do_not_dispatch_exact_auth"


def test_gate_blocks_truthy_authority_leak_in_local_summary() -> None:
    payload = _local_summary(score=0.18)
    payload["ready_for_exact_eval_dispatch"] = True

    report = build_local_exact_auth_gate_report(
        local_replay_summary=payload,
        config=LocalExactAuthGateConfig(auth_target_score=0.19),
    )

    assert report.exact_auth_dispatch_recommended is False
    assert any(
        blocker.startswith("local_replay_forbidden_authority:ready_for_exact_eval_dispatch")
        for blocker in report.blockers
    )


def test_gate_blocks_when_mlx_prefilter_required_but_missing() -> None:
    report = build_local_exact_auth_gate_report(
        local_replay_summary=_local_summary(score=0.18),
        config=LocalExactAuthGateConfig(
            auth_target_score=0.19,
            require_mlx_prefilter=True,
        ),
    )

    assert report.exact_auth_dispatch_recommended is False
    assert "mlx_prefilter_required_but_missing" in report.blockers


def test_cli_writes_fail_closed_report_for_non_winner(tmp_path: Path) -> None:
    local_path = tmp_path / "local.json"
    out_path = tmp_path / "gate.json"
    local_path.write_text(json.dumps(_local_summary(score=22.5)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--local-replay-summary-json",
            str(local_path),
            "--auth-frontier-score",
            "0.1919853363",
            "--out-json",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["exact_auth_dispatch_recommended"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "local_score_not_below_auth_target" in report["blockers"]
