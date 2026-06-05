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
        source_forward_artifacts=[
            _source_forward_artifact(
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
            )
        ],
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
    assert report["current_state"]["lf_dominance_launch_signal_active"] is False
    assert report["current_state"]["lf_dominance_signal_demoted"] is True
    assert "snerv_lf_hf_current_snar2_queue_has_no_lf_over_ceiling_rows" in report[
        "current_state"
    ]["demoted_blockers"]
    assert "snerv_lf_hf_current_snar2_queue_has_no_lf_over_ceiling_rows" not in report[
        "blockers"
    ]
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


def test_lf_hf_queue_consumes_partial_source_forward_frame_replay(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[
            _campaign_plan(
                blockers=(
                    "snerv_official_mfu_hfr_tub_export_not_bound",
                    "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                    "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
                    "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
                )
            )
        ],
        source_forward_artifacts=[_source_forward_artifact()],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    official = next(
        row
        for row in report["queue_rows"]
        if row["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    blockers = set(official["blockers"])
    assert "snerv_official_mfu_hfr_tub_receiver_payload_not_bound" not in blockers
    assert "snerv_official_mfu_hfr_tub_frame_producing_export_missing" not in blockers
    assert "snerv_official_mfu_hfr_tub_export_not_bound" in blockers
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in blockers
    assert "snerv_official_tub_output2_receiver_frame_decode_not_bound" in blockers
    assert official["source_forward_evidence"]["receiver_payload_frame_replay_proven"] is True
    assert official["source_forward_evidence"]["closed_campaign_blockers"] == [
        "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
        "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
    ]
    assert official["score_claim"] is False
    assert official["ready_for_exact_eval_dispatch"] is False


def test_lf_hf_queue_consumes_scorer_domain_tether_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[
            _campaign_plan(
                blockers=(
                    "snerv_official_mfu_hfr_tub_export_not_bound",
                    "snerv_scorer_input_distribution_guard_missing",
                )
            )
        ],
        source_forward_artifacts=[
            _source_forward_artifact(
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
            )
        ],
        candidate_feedback_rows=[_candidate_feedback_row()],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    official = next(
        row
        for row in report["queue_rows"]
        if row["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    blockers = set(official["blockers"])
    assert "snerv_scorer_input_distribution_guard_missing" not in blockers
    assert "snerv_official_mfu_hfr_tub_export_not_bound" in blockers
    assert official["scorer_domain_evidence"]["scorer_domain_tether_proof_passed"] is True
    assert official["scorer_domain_evidence"]["closed_campaign_blockers"] == [
        "snerv_scorer_input_distribution_guard_missing"
    ]
    assert official["scorer_domain_evidence"]["missing_metrics"] == []
    assert official["scorer_domain_evidence"]["lambda_inactive_metrics"] == []
    assert official["score_claim"] is False


def test_lf_hf_replacement_queue_cli_writes_ssd_handoff_artifacts(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.json"
    reroute_path = tmp_path / "reroute.json"
    campaign_path = tmp_path / "campaign.json"
    source_forward_path = tmp_path / "source_forward.json"
    feedback_path = tmp_path / "candidate_feedback.json"
    output_root = tmp_path / "out"
    output_json = output_root / "queue.json"
    output_md = output_root / "queue.md"
    lf_path.write_text(json.dumps(_lf_sweep_report()), encoding="utf-8")
    reroute_path.write_text(json.dumps(_reroute_queue(row_count=0)), encoding="utf-8")
    campaign_path.write_text(
        json.dumps(_campaign_plan(blockers=("snerv_official_mfu_hfr_tub_export_not_bound",))),
        encoding="utf-8",
    )
    source_forward_path.write_text(json.dumps(_source_forward_artifact()), encoding="utf-8")
    feedback_path.write_text(json.dumps(_candidate_feedback_row()), encoding="utf-8")

    rc = cli_main(
        [
            "--lf-payload-report",
            lf_path.as_posix(),
            "--reroute-queue",
            reroute_path.as_posix(),
            "--campaign-plan",
            campaign_path.as_posix(),
            "--source-forward-artifact",
            source_forward_path.as_posix(),
            "--candidate-feedback-row",
            feedback_path.as_posix(),
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
    assert payload["source_forward_evidence"]["source_path"] == source_forward_path.as_posix()
    assert payload["source_forward_evidence"]["receiver_payload_frame_replay_proven"] is True
    assert payload["scorer_domain_evidence"]["source_path"] == feedback_path.as_posix()
    assert payload["scorer_domain_evidence"]["scorer_domain_tether_proof_passed"] is True
    assert len(payload["selected_lf_payload_evidence"]["source_sha256"]) == 64
    assert "SNeRV LF/HF Replacement Queue" in markdown
    assert "receiver payload frame replay proven" in markdown
    assert "scorer domain tether proof passed" in markdown
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


def _source_forward_artifact(
    *,
    receiver_consumes_output2: bool = False,
    source_authority: bool = False,
    full_tub_parity: bool = False,
) -> dict[str, object]:
    blockers = []
    if not receiver_consumes_output2:
        blockers.append("snerv_official_tub_output2_receiver_frame_decode_not_bound")
    if not full_tub_parity:
        blockers.append("snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing")
    return {
        "schema": "snerv_official_mfu_hfr_tub_forward_parity.v1",
        "generated_utc": "20260605T000000Z",
        "_source_path": "/ssd/snerv_official_mfu_hfr_tub_forward_parity.json",
        "_source_sha256": "a" * 64,
        "full_tub_source_forward_parity_proven": full_tub_parity,
        "receiver_payload_frame_replay": {
            "schema": "snerv_official_mfu_hfr_tub_receiver_payload_frame_replay.v1",
            "receiver_runtime_decode_proven": True,
            "frame_producing_official_payload_replay_proven": True,
            "receiver_frame_decode_consumes_output2": receiver_consumes_output2,
            "source_forward_replay_authority": source_authority,
            "decoded_frames_shape": [2, 3, 16, 24],
            "decoded_frames_sha256": "b" * 64,
            "payload_bytes": 13052,
            "payload_sha256": "c" * 64,
        },
        "blockers": blockers,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _candidate_feedback_row(
    *,
    posenet_observed: bool = True,
    posenet_lambda_active: bool = True,
    segnet_observed: bool = True,
    segnet_lambda_active: bool = True,
) -> dict[str, object]:
    metric_health = {
        "snerv_posenet_yuv6_pair_distill": {
            "metric_observed": posenet_observed,
            "lambda_active_observed": posenet_lambda_active,
            "missing_metric_value": 0.0 if posenet_observed else 1.0,
            "lambda_value": 0.5 if posenet_lambda_active else 0.0,
        },
        "snerv_segnet_last_frame_distill": {
            "metric_observed": segnet_observed,
            "lambda_active_observed": segnet_lambda_active,
            "missing_metric_value": 0.0 if segnet_observed else 1.0,
            "lambda_value": 0.75 if segnet_lambda_active else 0.0,
        },
    }
    passed = all(
        (
            posenet_observed,
            posenet_lambda_active,
            segnet_observed,
            segnet_lambda_active,
        )
    )
    return {
        "schema": "nerv_candidate_feedback_row.v1",
        "created_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/nerv_candidate_byte_feedback_row.json",
        "_source_sha256": "d" * 64,
        "family": "snerv",
        "snerv_scorer_domain_tether_passed": passed,
        "snerv_scorer_domain_tether_blockers": (
            [] if passed else ["snerv_scorer_domain_tether_missing_telemetry"]
        ),
        "snerv_scorer_domain_tether_health": {
            "schema": "snerv_scorer_domain_tether_smoke_health.v1",
            "passed": passed,
            "metric_health": metric_health,
            "missing_metrics": [],
            "lambda_inactive_metrics": [],
            "blockers": (
                [] if passed else ["snerv_scorer_domain_tether_missing_telemetry"]
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
