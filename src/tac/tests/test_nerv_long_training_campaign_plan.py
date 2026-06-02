# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import load_queue_definition
from tac.analysis.nerv_long_training_campaign_plan import (
    NervLongTrainingCampaignPlanError,
    build_nerv_long_training_campaign_plan,
    render_nerv_long_training_campaign_plan_markdown,
)
from tools import build_nerv_long_training_campaign_plan as cli


def test_long_training_campaign_plan_builds_optimizer_matrix() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion", "adafactor"),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    assert report["schema"] == "nerv_long_training_campaign_plan.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["campaign_row_count"] == 3
    assert report["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert report["experiment_queue_experiment_count"] == 3
    assert report["launchable_local_row_count"] == 2
    assert report["family_counts"] == {"hi_nerv": 2, "snerv": 1}

    hi_rows = [row for row in report["campaign_rows"] if row["family"] == "hi_nerv"]
    assert {row["optimizer_kind"] for row in hi_rows} == {"lion", "adafactor"}
    assert all("--optimizer-kind" in row["command_argv"] for row in hi_rows)
    assert all("--coder-aware-qat" in row["command_argv"] for row in hi_rows)
    assert all(row["local_mlx_launch_command_ready"] is True for row in hi_rows)
    assert all(row["local_mlx_executable"] is True for row in hi_rows)
    assert all(row["cpu_replay_ready"] is False for row in hi_rows)
    assert all(row["exact_gate_ready"] is False for row in hi_rows)
    assert all(
        row["score_lowering_gate"]["schema"]
        == "nerv_long_training_score_lowering_gate.v1"
        for row in hi_rows
    )
    assert all(
        {
            "archive_in_loop_byte_oracle",
            "byte_closed_archive_export",
            "receiver_proof",
            "full_video_local_prefilter",
            "local_cpu_replay_gate",
        }.issubset(
            set(row["score_lowering_gate"]["post_run_missing_requirement_ids"])
        )
        for row in hi_rows
    )
    assert all("hi_nerv_receiver_proof_missing" in row["blockers"] for row in hi_rows)
    assert all(
        "hi_nerv_byte_closed_archive_export_missing" in row["promotion_blockers"]
        for row in hi_rows
    )
    assert all(row["experiment_queue_entry"]["status"] == "queued" for row in hi_rows)
    assert all(
        row["experiment_queue_entry"]["cpu_replay_ready"] is False
        for row in hi_rows
    )
    assert all(
        row["experiment_queue_entry"]["exact_gate_ready"] is False
        for row in hi_rows
    )
    hi_step = hi_rows[0]["experiment_queue_entry"]["steps"][0]
    assert hi_step["command"] == hi_rows[0]["command_argv"]
    assert hi_step["resources"]["kind"] == "local_mlx"
    assert {
        (condition["key"], condition.get("equals"))
        for condition in hi_step["postconditions"]
        if condition["type"] == "json_equals"
    } >= {
        ("schema", "compact_renderer_mlx_spine_runner.v1"),
        ("execute_family", "hi_nerv"),
        ("training_executed", True),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("ready_for_exact_eval_dispatch", False),
    }

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["score_lowering_gate"]["local_mlx_executable"] is False
    assert snerv_row["cpu_replay_ready"] is False
    assert snerv_row["exact_gate_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"
    assert snerv_row["experiment_queue_entry"]["blocked"] is True
    assert "snerv_shared_mlx_scoreaware_long_training_harness_not_bound" in snerv_row[
        "blockers"
    ]
    assert "--snerv-scorer-loop-qat" in snerv_row["command_argv"]
    snerv_step = snerv_row["experiment_queue_entry"]["steps"][0]
    assert {
        condition["type"] for condition in snerv_step["postconditions"]
    } >= {"json_equals", "json_path_contains"}

    markdown = render_nerv_long_training_campaign_plan_markdown(report)
    assert "NeRV Long-Training Campaign Plan" in markdown
    assert "hi_nerv::hinerv_tiny::lion" in markdown


def test_long_training_campaign_plan_rejects_unknown_optimizer() -> None:
    with pytest.raises(NervLongTrainingCampaignPlanError, match="unsupported"):
        build_nerv_long_training_campaign_plan(
            hinerv_modelsize_budget=_hinerv_budget(),
            snerv_modelsize_budget=_snerv_budget(),
            optimizer_kinds=("muon",),
        )


def test_build_long_training_campaign_plan_cli_writes_outputs(tmp_path: Path) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    out_md = tmp_path / "campaign.md"
    out_queue = tmp_path / "campaign_queue.json"
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            str(hinerv),
            "--snerv-modelsize-budget",
            str(snerv),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["campaign_row_count"] == 2
    assert payload["experiment_queue"]["schema"] == "experiment_queue.v1"
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue == payload["experiment_queue"]
    assert queue["experiments"][0]["steps"][0]["postconditions"]
    loaded_queue = load_queue_definition(out_queue)
    assert loaded_queue["schema"] == "experiment_queue.v1"
    assert loaded_queue["experiments"][0]["status"] == "queued"
    assert loaded_queue["experiments"][0]["steps"][0]["resources"]["kind"] == "local_mlx"
    assert out_md.read_text(encoding="utf-8").startswith(
        "# NeRV Long-Training Campaign Plan"
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            str(hinerv),
            "--snerv-modelsize-budget",
            str(snerv),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--expected-output-json-sha256",
            _sha256(out_json),
            "--expected-output-md-sha256",
            _sha256(out_md),
            "--expected-output-queue-sha256",
            _sha256(out_queue),
        ]
    )

    assert rc == 0


def _hinerv_budget() -> dict:
    return {
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
    }


def _snerv_budget() -> dict:
    return {
        "schema": "snerv_modelsize_budget.v1",
        "selected_candidates": [
            {
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv_tiny",
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "decoder_payload_codec": "int4_symmetric",
                "nominal_total_payload_bytes": 190_000,
                "nominal_under_ceiling": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
