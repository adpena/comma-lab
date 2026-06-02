# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import normalize_queue_definition
from tac.analysis.nerv_long_training_campaign_admission import (
    ADMISSION_SCHEMA,
    build_nerv_long_training_campaign_execution_admission,
)
from tac.analysis.nerv_long_training_campaign_plan import (
    build_nerv_long_training_campaign_plan,
)
from tac.cathedral_consumers.nerv_long_training_campaign_consumer import consume_candidate

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nerv_long_training_campaign_admission_builds_storage_gated_queue(
    tmp_path: Path,
) -> None:
    verdict = dict(consume_candidate(_campaign_plan(tmp_path / "ssd")))
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["schema"] == ADMISSION_SCHEMA
    assert admission["experiment_queue_ready"] is True
    assert admission["local_mlx_execution_ready"] is True
    assert admission["admitted_experiment_count"] == 1
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False
    assert admission["blockers"] == []
    queue = normalize_queue_definition(admission["experiment_queue"])
    assert queue["queue_id"] == "nerv_manifest_pinned_long_training_local_mlx_admission.v1"
    assert queue["experiments"][0]["id"] == "nerv_campaign_storage_preflight"
    selected = queue["experiments"][1]
    assert selected["steps"][0]["requires"] == [
        "nerv_campaign_storage_preflight.proactive_cleanup"
    ]
    assert selected["steps"][0]["resources"]["kind"] == "local_mlx"
    assert selected["metadata"]["human_visual_fidelity_relevance"] == (
        "irrelevant_unless_scorer_causal"
    )


def test_nerv_long_training_campaign_admission_blocks_without_active_claim(
    tmp_path: Path,
) -> None:
    verdict = dict(consume_candidate(_campaign_plan(tmp_path / "ssd")))
    claims = tmp_path / "claims.md"
    claims.write_text("# empty\n", encoding="utf-8")

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["experiment_queue"] is None
    assert admission["admitted_experiment_count"] == 0
    assert "active_lane_claim_missing_or_terminal" in admission["blockers"]
    assert admission["score_claim"] is False


def test_nerv_long_training_campaign_admission_blocks_non_ssd_output(
    tmp_path: Path,
) -> None:
    verdict = dict(consume_candidate(_campaign_plan(tmp_path / "local_disk")))
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert "selected_row_output_dir_not_on_allowed_ssd_tier" in admission["blockers"]


def test_nerv_long_training_campaign_admission_cli_writes_artifacts(
    tmp_path: Path,
) -> None:
    verdict_path = tmp_path / "verdict.json"
    out_json = tmp_path / "admission.json"
    out_md = tmp_path / "admission.md"
    out_queue = tmp_path / "queue.json"
    verdict_path.write_text(
        json.dumps(dict(consume_candidate(_campaign_plan(tmp_path / "ssd")))),
        encoding="utf-8",
    )
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/build_nerv_long_training_campaign_execution_admission.py"),
            "--consumer-verdict",
            str(verdict_path),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--lane-id",
            "lane_nerv_local_mlx",
            "--instance-job-id",
            "job_first",
            "--active-claims-path",
            str(claims),
            "--storage-expected-bytes-per-row",
            "1024",
            "--storage-reserve-free-gb",
            "0",
            "--allowed-output-root",
            str(tmp_path / "ssd"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["experiment_queue_ready"] is True
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == ADMISSION_SCHEMA
    assert payload["score_claim"] is False
    assert out_md.read_text(encoding="utf-8").startswith(
        "# NeRV Long-Training Campaign Execution Admission"
    )
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue["schema"] == "experiment_queue.v1"


def _campaign_plan(output_root: Path) -> dict:
    return build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget={
            "schema": "nerv_modelsize_budget.v1",
            "selected_candidates": [
                {
                    "schema": "hinerv_modelsize_candidate.v1",
                    "family": "hi_nerv",
                    "candidate_id": "hinerv_tiny",
                    "num_pairs": 600,
                    "hard_byte_ceiling": 178_000,
                    "decoder_codec": "int4_mixed",
                    "nominal_total_payload_bytes": 120_000,
                    "nominal_under_ceiling": True,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        snerv_modelsize_budget={
            "schema": "snerv_modelsize_budget.v1",
            "selected_candidates": [
                {
                    "schema": "snerv_modelsize_candidate.v1",
                    "family": "snerv",
                    "candidate_id": "snerv_tiny",
                    "num_pairs": 600,
                    "hard_byte_ceiling": 178_000,
                    "decoder_payload_codec": "int4_symmetric",
                    "nominal_total_payload_bytes": 160_000,
                    "nominal_under_ceiling": True,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        optimizer_kinds=("adamw",),
        epochs=16,
        batch_pairs=4,
        learning_rate=3.0e-4,
        output_root=output_root,
        max_candidates_per_family=1,
    )


def _claims_file(tmp_path: Path, *, lane_id: str, instance_job_id: str) -> Path:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                (
                    f"| 2026-06-02T18:34:58Z | codex:gpt-5 | {lane_id} | "
                    f"local_mlx | {instance_job_id} | 2026-06-03T00:34:58Z | "
                    "active_local_mlx_queue_first_row | test claim |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return claims
