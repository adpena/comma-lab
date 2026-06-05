# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.snerv_lf_hf_replacement_queue import (
    SCHEMA,
    build_snerv_lf_hf_replacement_queue,
)
from tools.build_snerv_lf_hf_replacement_queue import main as cli_main


def test_lf_hf_replacement_queue_emits_bounded_smoke_when_unblocked(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["lf_payload_evidence_row_count"] == 1
    assert report["current_state"]["freshest_queue_has_no_lf_over_ceiling_rows"] is False
    assert report["local_executable_command_row_count"] == 1
    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    assert row["blocked"] is False
    assert row["status"] == "local_bounded_smoke_ready_no_authority"
    assert row["command_argv"]
    assert row["command_argv"][row["command_argv"].index("--num-pairs") + 1] == "16"
    assert (
        row["command_argv"][
            row["command_argv"].index("--snerv-score-aware-long-training-epochs") + 1
        ]
        == "128"
    )
    assert str(tmp_path) in row["command_argv"][row["command_argv"].index("--output-dir") + 1]
    assert row["dispatch_allowed"] is False
    assert row["local_mlx_long_training_allowed"] is False


def test_lf_hf_replacement_queue_blocks_current_snar2_no_lf_overrun_state(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=0)],
        campaign_plans=[
            _campaign_plan(
                blockers=(
                    "snerv_official_mfu_hfr_tub_export_not_bound",
                    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
                    "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
                    "snerv_scorer_input_distribution_guard_missing",
                )
            )
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["current_state"]["freshest_queue_has_no_lf_over_ceiling_rows"] is True
    assert "snerv_lf_hf_current_snar2_queue_has_no_lf_over_ceiling_rows" in report["blockers"]
    assert report["local_executable_command_row_count"] == 0
    assert all(row["blocked"] is True for row in report["queue_rows"])
    official = next(
        row
        for row in report["queue_rows"]
        if row["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    assert "snerv_official_mfu_hfr_tub_export_not_bound" in official["blockers"]
    assert "snerv_scorer_input_distribution_guard_missing" in official["blockers"]
    assert official["command_argv"] == []


def test_lf_hf_replacement_queue_cli_writes_ssd_handoff_artifacts(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.json"
    reroute_path = tmp_path / "reroute.json"
    campaign_path = tmp_path / "campaign.json"
    output_root = tmp_path / "out"
    output_json = output_root / "queue.json"
    output_md = output_root / "queue.md"
    lf_path.write_text(json.dumps(_lf_sweep_report()), encoding="utf-8")
    reroute_path.write_text(json.dumps(_reroute_queue(row_count=0)), encoding="utf-8")
    campaign_path.write_text(
        json.dumps(_campaign_plan(blockers=("snerv_official_mfu_hfr_tub_export_not_bound",))),
        encoding="utf-8",
    )

    rc = cli_main(
        [
            "--lf-payload-report",
            lf_path.as_posix(),
            "--reroute-queue",
            reroute_path.as_posix(),
            "--campaign-plan",
            campaign_path.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--output-json",
            output_json.as_posix(),
            "--output-md",
            output_md.as_posix(),
            "--allow-local-output",
            "--min-free-bytes",
            "0",
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert rc == 0
    assert payload["schema"] == SCHEMA
    assert payload["selected_lf_payload_evidence"]["source_path"] == lf_path.as_posix()
    assert len(payload["selected_lf_payload_evidence"]["source_sha256"]) == 64
    assert "SNeRV LF/HF Replacement Queue" in markdown
    assert payload["score_claim"] is False


def _lf_sweep_report() -> dict[str, object]:
    return {
        "schema": "snerv_lf_payload_codec_sweep.v1",
        "authority": "false_authority_lf_payload_codec_rate_only",
        "axis_tag": "[planning/control]",
        "family": "snerv",
        "plane_count": 96,
        "plane_shapes": [[110, 146]],
        "raw_i64_bytes": 12_334_080,
        "baseline_payload_bytes": 666_556,
        "selected_rate_only_row": {
            "mode": "int64_lzma",
            "payload_bytes": 666_556,
        },
        "blockers": [
            "snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _reroute_queue(*, row_count: int) -> dict[str, object]:
    return {
        "schema": "snerv_lf_over_ceiling_reroute_queue.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "queue_row_count": row_count,
        "snar_header_minimization_report_count": 2,
        "queue_rows": [{} for _ in range(row_count)],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _campaign_plan(*, blockers: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema": "nerv_long_training_campaign_plan.v1",
        "campaign_rows": [
            {
                "row_id": "snerv::auto_bytecap::native_rate_aware_training",
                "candidate_id": "snerv_np600_haar_lv1_lfb1_int8_ceil178000",
                "family": "snerv",
                "priority": 12,
                "implementation_status": "native_rate_aware_long_training_queue_ready",
                "hard_byte_ceiling": 178_000,
                "candidate_nominal_under_ceiling": True,
                "hard_byte_ceiling_satisfied_for_long_training": True,
                "local_mlx_launch_command_ready": True,
                "blockers": list(blockers),
                "command_argv": [
                    "uv",
                    "run",
                    "--extra",
                    "dev",
                    "--extra",
                    "runtime",
                    "--extra",
                    "mlx",
                    "python",
                    "tools/run_compact_renderer_mlx_spine_runner.py",
                    "--execute-family",
                    "snerv",
                    "--planner-row-id",
                    "snerv::auto_bytecap::native_rate_aware_training",
                    "--num-pairs",
                    "600",
                    "--epochs",
                    "29650",
                    "--snerv-score-aware-long-training-epochs",
                    "29650",
                    "--snerv-score-aware-long-training-batch-pairs",
                    "8",
                    "--output-dir",
                    "/Volumes/VertigoDataTier/pact/old",
                    "--planner-row-queue-artifact",
                    "/Volumes/VertigoDataTier/pact/old/queue.json",
                ],
            }
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
