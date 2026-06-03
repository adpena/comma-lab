# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.nerv_active_campaign_feedback_audit import (
    SCHEMA,
    build_nerv_active_campaign_feedback_audit,
)


def test_active_campaign_feedback_audit_blocks_uningested_live_telemetry(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    claims = _claims(
        repo,
        status="active_local_mlx_training",
        output_dir=_output_with_telemetry(repo, latest_epoch=1000),
    )

    report = build_nerv_active_campaign_feedback_audit(
        claims_path=claims,
        repo_root=repo,
        stale_epoch_tolerance=16,
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["active_claim_count"] == 1
    assert report["artifact_count"] == 1
    assert any("active_campaign_feedback_not_ingested" in b for b in report["blockers"])


def test_active_campaign_feedback_audit_accepts_fresh_feedback_ingestion(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output = _output_with_telemetry(repo, latest_epoch=1000)
    telemetry = output / "hi_nerv_mlx_training" / "telemetry.jsonl"
    claims = _claims(repo, status="training", output_dir=output)
    _write_feedback(repo, telemetry_path=telemetry, last_epoch=995)

    report = build_nerv_active_campaign_feedback_audit(
        claims_path=claims,
        repo_root=repo,
        stale_epoch_tolerance=16,
    )

    assert report["blockers"] == []
    artifact = report["active_claim_rows"][0]["artifacts"][0]
    assert artifact["ingestion"]["ingested"] is True
    assert artifact["ingestion"]["stale"] is False
    assert artifact["ingestion"]["max_ingested_epoch"] == 995


def test_active_campaign_feedback_audit_follows_queue_output_dir(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output = _output_with_telemetry(repo, latest_epoch=1000)
    telemetry = output / "hi_nerv_mlx_training" / "telemetry.jsonl"
    queue = _queue_with_output_dir(repo, output)
    claims = _claims(
        repo,
        status="active_local_mlx_training",
        output_dir=queue,
        notes_suffix="pid=123/child=456 false-authority no replay/exact",
    )
    _write_feedback(repo, telemetry_path=telemetry, last_epoch=1000)

    report = build_nerv_active_campaign_feedback_audit(
        claims_path=claims,
        repo_root=repo,
        stale_epoch_tolerance=16,
    )

    assert report["blockers"] == []
    row = report["active_claim_rows"][0]
    assert any(str(output) == root for root in row["artifact_roots"])
    assert not any("/child=" in root or root.endswith("/replay/exact") for root in row["artifact_roots"])
    assert row["artifacts"][0]["path"] == telemetry.as_posix()


def test_active_campaign_feedback_audit_blocks_stale_feedback_ingestion(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output = _output_with_telemetry(repo, latest_epoch=1000)
    telemetry = output / "hi_nerv_mlx_training" / "telemetry.jsonl"
    claims = _claims(repo, status="running_local_mlx", output_dir=output)
    _write_feedback(repo, telemetry_path=telemetry, last_epoch=128)

    report = build_nerv_active_campaign_feedback_audit(
        claims_path=claims,
        repo_root=repo,
        stale_epoch_tolerance=16,
    )

    assert any("active_campaign_feedback_ingestion_stale" in b for b in report["blockers"])


def test_active_campaign_feedback_audit_ignores_newer_terminal_claim(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    output = _output_with_telemetry(repo, latest_epoch=1000)
    claims = repo / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                f"| 2026-06-03T00:01:00Z | codex | lane_nerv_test | local_mlx | job1 |  | completed_local_smoke | output dir {output} |",
                f"| 2026-06-03T00:00:00Z | codex | lane_nerv_test | local_mlx | job1 |  | training | output dir {output} |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_nerv_active_campaign_feedback_audit(
        claims_path=claims,
        repo_root=repo,
    )

    assert report["active_claim_count"] == 0
    assert report["blockers"] == []


def _claims(
    repo: Path,
    *,
    status: str,
    output_dir: Path,
    notes_suffix: str = "",
) -> Path:
    claims = repo / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    notes = f"output dir {output_dir}"
    if notes_suffix:
        notes += f" {notes_suffix}"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                f"| 2026-06-03T00:00:00Z | codex | lane_nerv_hinerv_test | local_mlx | job1 |  | {status} | {notes} |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return claims


def _output_with_telemetry(repo: Path, *, latest_epoch: int) -> Path:
    output = repo / "ssd/nerv_long_training_campaigns/live_hinerv"
    telemetry = output / "hi_nerv_mlx_training" / "telemetry.jsonl"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text(
        json.dumps({"epoch": 0, "per_axis_decomposition": {"seg": 6.0, "pose": 10.0}})
        + "\n"
        + json.dumps(
            {
                "epoch": latest_epoch,
                "per_axis_decomposition": {"seg": 5.9, "pose": 2.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _queue_with_output_dir(repo: Path, output: Path) -> Path:
    queue = repo / ".omx/research/nerv_long_training_campaign_queue_test.json"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "id": "hi_nerv_test",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "tools/run_compact_renderer_mlx_spine_runner.py",
                            "--execute-family",
                            "hi_nerv",
                            "--output-dir",
                            output.as_posix(),
                        ],
                    }
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return queue.relative_to(repo)


def _write_feedback(repo: Path, *, telemetry_path: Path, last_epoch: int) -> None:
    research = repo / ".omx/research"
    research.mkdir(parents=True, exist_ok=True)
    (research / "nerv_training_telemetry_feedback_test.json").write_text(
        json.dumps(
            {
                "schema": "nerv_training_telemetry_feedback.v1",
                "row": {
                    "source_report_path": telemetry_path.as_posix(),
                    "training_telemetry": {
                        "schema": "nerv_training_telemetry_feedback.v1",
                        "last_epoch": last_epoch,
                    },
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
