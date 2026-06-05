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
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
            )
        ],
        official_replacement_authority_gates=[
            _official_replacement_authority_gate(ready=True)
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
    assert "--snerv-scorer-loop-qat" in row["command_argv"]
    assert row["command_argv"][
        row["command_argv"].index("--snerv-scorer-loop-max-trials") + 1
    ] == "1"
    assert row["command_argv"][
        row["command_argv"].index(
            "--snerv-scorer-loop-pair-guard-min-score-improved-fraction"
        )
        + 1
    ] == "0"
    assert row["command_argv"][
        row["command_argv"].index("--snerv-scorer-loop-max-archive-byte-growth") + 1
    ] == "0"
    assert str(tmp_path) in row["command_argv"][row["command_argv"].index("--output-dir") + 1]
    assert row["dispatch_allowed"] is False
    assert row["local_mlx_long_training_allowed"] is False


def test_lf_hf_replacement_queue_embeds_rebuild_command_with_source_paths(
    tmp_path: Path,
) -> None:
    lf_report = {**_lf_sweep_report(), "_source_path": "/ssd/lf_payload.json"}
    reroute = {**_reroute_queue(row_count=1), "_source_path": "/ssd/reroute.json"}
    campaign = {**_campaign_plan(blockers=()), "_source_path": "/ssd/campaign.json"}
    source = {
        **_source_forward_artifact(
            official_export_bound=True,
            receiver_consumes_output2=True,
        ),
        "_source_path": "/ssd/source_forward.json",
    }
    gate = {
        **_official_replacement_authority_gate(ready=False),
        "_source_path": "/ssd/gate.json",
        "next_unblock_command_argv": ["uv", "run", "python", "next.py"],
    }
    feedback = {
        **_candidate_feedback_row(guard_proof_passed=True),
        "_source_path": "/ssd/feedback.json",
    }
    xray = {**_value_domain_xray(noncollapse=False), "_source_path": "/ssd/xray.json"}

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[lf_report],
        reroute_queues=[reroute],
        campaign_plans=[campaign],
        source_forward_artifacts=[source],
        official_replacement_authority_gates=[gate],
        candidate_feedback_rows=[feedback],
        value_domain_xray_reports=[xray],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["input_source_paths"]["lf_payload_reports"] == ["/ssd/lf_payload.json"]
    assert report["input_source_paths"]["campaign_plans"] == ["/ssd/campaign.json"]
    assert report["input_source_paths"]["reroute_queues"] == ["/ssd/reroute.json"]
    assert report["input_source_paths"]["official_replacement_authority_gates"] == [
        "/ssd/gate.json"
    ]
    command = report["runnable_rebuild_command_argv"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_hf_replacement_queue.py",
    ]
    assert command[command.index("--lf-payload-report") + 1] == "/ssd/lf_payload.json"
    assert command[command.index("--campaign-plan") + 1] == "/ssd/campaign.json"
    assert command[command.index("--reroute-queue") + 1] == "/ssd/reroute.json"
    assert command[command.index("--source-forward-artifact") + 1] == (
        "/ssd/source_forward.json"
    )
    assert command[command.index("--official-replacement-authority-gate") + 1] == (
        "/ssd/gate.json"
    )
    assert command[command.index("--candidate-feedback-row") + 1] == (
        "/ssd/feedback.json"
    )
    assert command[command.index("--value-domain-xray") + 1] == "/ssd/xray.json"
    assert command[command.index("--output-root") + 1] == (tmp_path / "queue").as_posix()
    assert report["next_unblock_command_argv"] == ["uv", "run", "python", "next.py"]


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
    assert official["source_forward_evidence"]["receiver_bound_export_proven"] is False
    assert official["source_forward_evidence"]["closed_campaign_blockers"] == [
        "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
        "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
    ]
    assert official["score_claim"] is False
    assert official["ready_for_exact_eval_dispatch"] is False


def test_lf_hf_queue_closes_receiver_bound_export_but_blocks_materializer_until_source_authority(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[
            _campaign_plan(
                blockers=(
                    "snerv_official_mfu_hfr_tub_export_not_bound",
                    "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
                )
            )
        ],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=False,
                full_tub_parity=False,
            )
        ],
        official_replacement_authority_gates=[
            _official_replacement_authority_gate(ready=False)
        ],
        candidate_feedback_rows=[_candidate_feedback_row(guard_proof_passed=True)],
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
    assert official["source_forward_evidence"]["receiver_bound_export_proven"] is True
    assert official["source_forward_evidence"]["source_forward_replay_authority"] is False
    assert "snerv_official_mfu_hfr_tub_export_not_bound" not in blockers
    assert "snerv_scorer_input_distribution_guard_missing" not in blockers
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in blockers
    assert official["blocked"] is True
    assert official["command_argv"] == []
    assert report["local_executable_command_row_count"] == 0
    assert report["ready_for_exact_eval_dispatch"] is False


def test_lf_hf_queue_preserves_native_hfr_mapping_split_blockers(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                native_hfr_mapping=True,
            )
        ],
        candidate_feedback_rows=[_candidate_feedback_row(guard_proof_passed=True)],
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
    evidence = official["source_forward_evidence"]
    assert evidence["official_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert evidence["official_mfu_receiver_activation_payload_bound"] is True
    assert evidence["official_native_receiver_state_mapping_proven"] is True
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" not in blockers
    assert "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete" not in blockers
    assert (
        "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
        in blockers
    )
    assert "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded" in blockers
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert official["command_argv"] == []


def test_lf_hf_queue_consumes_value_domain_noncollapse_for_lf_conditioned_hf(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[
            _campaign_plan(
                blockers=(
                    "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
                    "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
                    "snerv_renderer_nondegenerate_target_value_domain_not_passed",
                )
            )
        ],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "lf_conditioned_hf_residual_generator"
    )
    blockers = set(row["blockers"])
    assert row["value_domain_evidence"]["value_domain_noncollapse_proof_passed"] is True
    assert (
        "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse"
        not in blockers
    )
    assert (
        "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed"
        not in blockers
    )
    assert (
        "snerv_renderer_nondegenerate_target_value_domain_not_passed" not in blockers
    )
    assert (
        "snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing"
        not in blockers
    )
    assert "snerv_hf_residual_generator_receiver_payload_not_implemented" in blockers
    assert row["blocked"] is True
    assert row["command_argv"] == []


def test_lf_hf_queue_keeps_distribution_guard_blocker_for_tether_only_proof(
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
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
            )
        ],
        candidate_feedback_rows=[_candidate_feedback_row(guard_proof_passed=False)],
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
    assert "snerv_scorer_input_distribution_guard_missing" in blockers
    assert official["scorer_domain_evidence"]["scorer_domain_tether_proof_passed"] is True
    assert (
        official["scorer_domain_evidence"][
            "scorer_input_distribution_guard_proof_passed"
        ]
        is False
    )
    assert official["scorer_domain_evidence"]["closed_campaign_blockers"] == []
    assert official["score_claim"] is False


def test_lf_hf_queue_consumes_scorer_input_distribution_guard_proof(
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
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
            )
        ],
        candidate_feedback_rows=[_candidate_feedback_row(guard_proof_passed=True)],
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
    assert "snerv_official_mfu_hfr_tub_export_not_bound" not in blockers
    assert official["scorer_domain_evidence"]["scorer_domain_tether_proof_passed"] is True
    assert official["source_forward_evidence"]["receiver_bound_export_proven"] is True
    assert (
        official["scorer_domain_evidence"][
            "scorer_input_distribution_guard_proof_passed"
        ]
        is True
    )
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
    official_export_bound: bool = False,
    receiver_consumes_output2: bool = False,
    source_authority: bool = False,
    full_tub_parity: bool = False,
    native_hfr_mapping: bool = False,
) -> dict[str, object]:
    blockers = []
    if not receiver_consumes_output2:
        blockers.append("snerv_official_tub_output2_receiver_frame_decode_not_bound")
    if not full_tub_parity:
        blockers.append("snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing")
    if native_hfr_mapping:
        blockers.extend(
            [
                "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping",
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_trained_checkpoint_source_forward_replay_missing",
                "snerv_official_trained_checkpoint_state_dict_mapping_missing",
            ]
        )
    return {
        "schema": "snerv_official_mfu_hfr_tub_forward_parity.v1",
        "generated_utc": "20260605T000000Z",
        "_source_path": "/ssd/snerv_official_mfu_hfr_tub_forward_parity.json",
        "_source_sha256": "a" * 64,
        "official_export_bound": official_export_bound,
        "official_checkpoint_export_binding_evidence": {
            "schema": "snerv_official_checkpoint_export_binding_evidence.v1",
            "official_export_bound": official_export_bound,
            "official_receiver_payload_bound": official_export_bound,
            "official_receiver_tensor_map_verified": official_export_bound,
            "native_checkpoint_export_bound_to_official_payload": official_export_bound,
            "closed_campaign_blockers": (
                ["snerv_official_mfu_hfr_tub_export_not_bound"]
                if official_export_bound
                else []
            ),
            "blockers": (
                []
                if official_export_bound
                else ["snerv_official_mfu_hfr_tub_export_not_bound"]
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "official_trained_checkpoint_loaded": native_hfr_mapping,
        "official_hfr_trained_checkpoint_weight_mapping_proven": native_hfr_mapping,
        "official_mfu_trained_checkpoint_weight_mapping_proven": False,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
        "official_mfu_receiver_activation_payload_bound": native_hfr_mapping,
        "official_tub_receiver_activation_payload_bound": False,
        "official_native_receiver_state_mapping_proven": native_hfr_mapping,
        "official_tub_temporal_encoder_weight_mapping_proven": False,
        "official_trained_checkpoint_mapping_manifest": (
            {
                "schema": "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1",
                "official_trained_checkpoint_loaded": True,
                "official_hfr_trained_checkpoint_weight_mapping_proven": True,
                "official_mfu_trained_checkpoint_weight_mapping_proven": False,
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
                "official_mfu_receiver_activation_payload_bound": True,
                "official_native_receiver_state_mapping_proven": True,
                "official_tub_temporal_encoder_weight_mapping_proven": False,
                "closed_campaign_blockers": [
                    "snerv_official_trained_checkpoint_state_dict_not_loaded",
                    "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete",
                ],
                "blockers": [
                    "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping",
                    "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                    "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                    "snerv_official_trained_checkpoint_source_forward_replay_missing",
                    "snerv_official_trained_checkpoint_state_dict_mapping_missing",
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
            if native_hfr_mapping
            else None
        ),
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


def _official_replacement_authority_gate(*, ready: bool) -> dict[str, object]:
    blockers = (
        []
        if ready
        else [
            "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
            "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
            "snerv_official_trained_checkpoint_state_dict_mapping_missing",
        ]
    )
    return {
        "schema": "snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/official_replacement_authority_gate.json",
        "_source_sha256": "f" * 64,
        "official_tub_lf_hf_decoder_replacement_ready": ready,
        "official_checkpoint_export_binding_ready": True,
        "receiver_output2_frame_replay_ready": True,
        "trained_checkpoint_state_dict_mapping_ready": ready,
        "tub_temporal_output2_weight_mapping_ready": ready,
        "full_tub_source_forward_replay_ready": ready,
        "closed_campaign_blockers": (
            [
                "snerv_official_mfu_hfr_tub_export_not_bound",
                "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
                "snerv_official_tub_output2_receiver_frame_decode_not_bound",
                "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
                "snerv_official_trained_checkpoint_state_dict_mapping_missing",
            ]
            if ready
            else [
                "snerv_official_mfu_hfr_tub_export_not_bound",
                "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
                "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
                "snerv_official_tub_output2_receiver_frame_decode_not_bound",
            ]
        ),
        "queue_blockers": blockers,
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
    guard_proof_passed: bool = True,
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
        "snerv_scorer_input_distribution_guard_proof_passed": guard_proof_passed,
        "snerv_scorer_input_distribution_guard_proof": {
            "schema": "snerv_scorer_input_distribution_guard_proof.v1",
            "required": True,
            "bound": guard_proof_passed,
            "telemetry_contract_passed": guard_proof_passed,
            "metric_observed": guard_proof_passed,
            "dual_metric_observed": guard_proof_passed,
            "passed": guard_proof_passed,
            "blockers": (
                []
                if guard_proof_passed
                else ["snerv_scorer_input_distribution_guard_metric_missing"]
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _value_domain_xray(*, noncollapse: bool) -> dict[str, object]:
    return {
        "schema": "snerv_receiver_value_domain_xray.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_receiver_value_domain_xray.json",
        "_source_sha256": "e" * 64,
        "packet_path": "/ssd/candidate.snar",
        "packet_bytes": 1234,
        "packet_sha256": "f" * 64,
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "value_domain_sample_status": "selected_pair_decode_completed",
        "receiver_payload_decode_sample_proven": True,
        "value_domain_noncollapse_proof_passed": noncollapse,
        "verdict": (
            "receiver_value_domain_sample_within_limits"
            if noncollapse
            else "RECEIVER_VALUE_DOMAIN_OUT_OF_RANGE"
        ),
        "closed_campaign_blockers": (
            [
                "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
                "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
                "snerv_renderer_nondegenerate_target_value_domain_not_passed",
            ]
            if noncollapse
            else []
        ),
        "blockers": (
            ["snerv_receiver_value_domain_xray_false_authority"]
            if noncollapse
            else [
                "snerv_receiver_value_domain_xray_false_authority",
                "snerv_receiver_decode_clipped_output_saturated",
            ]
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
