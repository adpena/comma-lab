# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.local_exact_auth_gate import (
    LocalExactAuthGateConfig,
    build_local_exact_auth_gate_report,
)
from comma_lab.local_exact_auth_gate_learning import (
    LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA,
    append_local_exact_auth_gate_posterior_signal,
    build_local_exact_auth_gate_learning_signal,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "record_local_exact_auth_gate_learning.py"


def _mlx_summary(action: float = 17.4) -> dict[str, object]:
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


def test_gate_learning_signal_is_stable_false_authority_planner_input(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    gate_path = tmp_path / "gate.json"
    replay = _mlx_summary(action=17.4)
    report = build_local_exact_auth_gate_report(
        mlx_prefilter_summary=replay,
        mlx_prefilter_summary_path=replay_path,
        config=LocalExactAuthGateConfig(
            auth_target_score=0.1919853363,
            mlx_target_action=0.1919853363,
        ),
    )
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    gate_path.write_text(report.to_json(), encoding="utf-8")

    signal_a = build_local_exact_auth_gate_learning_signal(
        gate_report=json.loads(report.to_json()),
        gate_report_path=gate_path,
        replay_summary=replay,
        replay_summary_path=replay_path,
        repo_root=REPO_ROOT,
        candidate_id="rd_waterfill",
        lane_id="lane_z8",
        family_id="z8_hierarchical_predictive_coding",
    )
    signal_b = build_local_exact_auth_gate_learning_signal(
        gate_report=json.loads(report.to_json()),
        gate_report_path=gate_path,
        replay_summary=replay,
        replay_summary_path=replay_path,
        repo_root=REPO_ROOT,
        candidate_id="rd_waterfill",
        lane_id="lane_z8",
        family_id="z8_hierarchical_predictive_coding",
    )

    assert signal_a["schema"] == LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA
    assert signal_a["signal_id"] == signal_b["signal_id"]
    assert signal_a["recommended_acquisition_policy"] == (
        "demote_candidate_for_archive_until_allocator_or_actuator_changes"
    )
    assert signal_a["priority_delta"] == "decrease"
    assert signal_a["ready_for_exact_eval_dispatch"] is False
    assert signal_a["reproducibility_provenance"]["row_deduplication_excludes_wall_clock_time"] is True


def test_gate_learning_posterior_append_deduplicates_by_stable_row_id(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    gate_path = tmp_path / "gate.json"
    signal_path = tmp_path / "signal.json"
    posterior = tmp_path / "posterior.jsonl"
    replay = _mlx_summary(action=0.17)
    report = build_local_exact_auth_gate_report(
        mlx_prefilter_summary=replay,
        mlx_prefilter_summary_path=replay_path,
        config=LocalExactAuthGateConfig(
            auth_target_score=0.1919853363,
            mlx_target_action=0.18,
        ),
    )
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    gate_path.write_text(report.to_json(), encoding="utf-8")
    signal = build_local_exact_auth_gate_learning_signal(
        gate_report=json.loads(report.to_json()),
        gate_report_path=gate_path,
        replay_summary=replay,
        replay_summary_path=replay_path,
        repo_root=REPO_ROOT,
        candidate_id="rd_waterfill",
        lane_id="lane_z8",
        family_id="z8_hierarchical_predictive_coding",
    )
    signal_path.write_text(json.dumps(signal), encoding="utf-8")

    first = append_local_exact_auth_gate_posterior_signal(
        learning_signal=signal,
        learning_signal_path=signal_path,
        repo_root=REPO_ROOT,
        posterior_path=posterior,
        lock_path=tmp_path / "posterior.lock",
    )
    second = append_local_exact_auth_gate_posterior_signal(
        learning_signal=signal,
        learning_signal_path=signal_path,
        repo_root=REPO_ROOT,
        posterior_path=posterior,
        lock_path=tmp_path / "posterior.lock",
    )

    assert first["appended"] is True
    assert second["appended"] is False
    assert second["skipped_duplicate"] is True
    assert len(posterior.read_text(encoding="utf-8").splitlines()) == 1


def test_gate_learning_cli_writes_signal_and_posterior(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    gate_path = tmp_path / "gate.json"
    signal_path = tmp_path / "signal.json"
    posterior = tmp_path / "posterior.jsonl"
    replay = _mlx_summary(action=17.4)
    report = build_local_exact_auth_gate_report(
        mlx_prefilter_summary=replay,
        mlx_prefilter_summary_path=replay_path,
        config=LocalExactAuthGateConfig(
            auth_target_score=0.1919853363,
            mlx_target_action=0.1919853363,
        ),
    )
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    gate_path.write_text(report.to_json(), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--gate-report-json",
            str(gate_path),
            "--replay-summary-json",
            str(replay_path),
            "--candidate-id",
            "rd_waterfill",
            "--lane-id",
            "lane_z8",
            "--family-id",
            "z8_hierarchical_predictive_coding",
            "--out-json",
            str(signal_path),
            "--posterior-path",
            str(posterior),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert signal["schema"] == LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA
    assert posterior.is_file()
