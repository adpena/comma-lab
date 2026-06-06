# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tac.analysis.snerv_lf_hf_replacement_queue import (
    SCHEMA,
    build_snerv_lf_hf_replacement_queue,
)
from tools.build_snerv_lf_hf_replacement_queue import main as cli_main


def test_lf_hf_replacement_queue_emits_bounded_smoke_when_unblocked(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    feedback_path = tmp_path / "nerv_candidate_byte_feedback_row.json"
    feedback_path.write_text(json.dumps({"schema": "placeholder"}), encoding="utf-8")
    feedback = _candidate_feedback_row()
    feedback.pop("_source_path", None)
    feedback["_candidate_feedback_source_path"] = feedback_path.as_posix()
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        candidate_feedback_rows=[feedback],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
                state_dict_path=state_dict_path,
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
    assert row["family"] == "snerv"
    assert row["row_id"] == row["queue_row_id"]
    assert row["launch_authority_contract"] == {
        "schema": "nerv_long_training_queue_launch_authority_contract.v1",
        "queue_status_is_local_mlx_plan": True,
        "queue_status_is_runnable_plan": True,
        "queue_launch_step_count": 1,
        "queue_steps_retained_as_post_unblock_handoff": False,
        "queue_launch_blockers": [],
        "queue_status_is_receiver_proof": False,
        "queue_status_is_cpu_replay_proof": False,
        "queue_status_is_exact_eval_authority": False,
        "source_queue_schema": SCHEMA,
        "source_row_schema": "snerv_lf_hf_replacement_candidate_row.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    assert row["bounded_training_binding_contract"]["schema"] == (
        "snerv_lf_hf_bounded_training_binding_contract.v1"
    )
    assert row["bounded_training_binding_contract"]["runner_actuator_bound"] is True
    assert row["bounded_training_binding_contract"]["runner_actuator"][
        "runner"
    ] == "tools/run_compact_renderer_mlx_spine_runner.py"
    assert row["command_argv"]
    assert row["command_argv"][row["command_argv"].index("--num-pairs") + 1] == "16"
    assert (
        row["command_argv"][
            row["command_argv"].index("--snerv-score-aware-long-training-epochs") + 1
        ]
        == "128"
    )
    assert "--snerv-scorer-loop-qat" not in row["command_argv"]
    assert "--snerv-scorer-loop-max-trials" not in row["command_argv"]
    assert "--snerv-scorer-loop-search-mode" not in row["command_argv"]
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-min-post-segnet-occupied-class-fraction"
        )
        + 1
    ] == "0.400001"
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-min-post-segnet-target-class-coverage-fraction"
        )
        + 1
    ] == "0.8"
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-min-post-segnet-target-class-min-ratio"
        )
        + 1
    ] == "0.2"
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-max-post-segnet-target-class-ratio-drop"
        )
        + 1
    ] == "0.05"
    assert row["command_argv"][
        row["command_argv"].index("--segnet-direct-live-rare-class-logit-weight") + 1
    ] == "4"
    assert row["command_argv"][
        row["command_argv"].index(
            "--segnet-direct-live-target-mass-floor-weight"
        )
        + 1
    ] == "0.5"
    assert row["command_argv"][
        row["command_argv"].index(
            "--segnet-direct-live-target-min-ratio-floor-weight"
        )
        + 1
    ] == "0.5"
    assert row["command_argv"][
        row["command_argv"].index("--segnet-direct-live-escape-warmup-epochs")
        + 1
    ] == "32"
    assert row["command_argv"][
        row["command_argv"].index("--segnet-direct-live-escape-class-multiplier")
        + 1
    ] == "16"
    assert row["command_argv"][
        row["command_argv"].index("--posenet-yuv6-geometry-tether-weight") + 1
    ] == "0.5"
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-max-post-segnet-distribution-mae"
        )
        + 1
    ] == "0.31"
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-max-post-posenet-yuv6-distribution-mae"
        )
        + 1
    ] == "0.22"
    assert row["command_argv"][
        row["command_argv"].index(
            "--scorer-space-step-guard-max-post-posenet-yuv6-contrast-ratio"
        )
        + 1
    ] == "3.75"
    qat_evidence = row["scorer_domain_evidence"][
        "scorer_loop_qat_nondegenerate_evidence"
    ]
    assert qat_evidence["passed"] is False
    assert "snerv_renderer_nondegenerate_measured_qat_evidence_missing" in (
        qat_evidence["blockers"]
    )
    assert row["command_argv"][
        row["command_argv"].index("--modelsize-candidate-id") + 1
    ] == row["candidate_id"]
    assert feedback_path.as_posix() in [
        row["command_argv"][idx + 1]
        for idx, token in enumerate(row["command_argv"][:-1])
        if token == "--modelsize-byte-cap-feedback-json"
    ]
    assert row["scorer_domain_evidence"]["source_path"] == feedback_path.as_posix()
    assert row["command_argv"][
        row["command_argv"].index(
            "--snerv-official-trained-checkpoint-state-dict-path"
        )
        + 1
    ] == state_dict_path.as_posix()
    assert str(tmp_path) in row["command_argv"][row["command_argv"].index("--output-dir") + 1]
    assert row["dispatch_allowed"] is False
    assert row["local_mlx_long_training_allowed"] is False


def test_official_replacement_emits_bounded_smoke_from_blocked_parent_template(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    queue_artifact_path = tmp_path / "queue" / "snerv_lf_hf_replacement_queue.json"
    campaign = _campaign_plan(
        blockers=(
            "snerv_pre_long_run_full_video_mlx_prefilter_missing",
            "snerv_receiver_proof_missing",
        )
    )
    campaign["campaign_rows"][0]["local_mlx_launch_command_ready"] = False
    campaign["campaign_rows"][0]["implementation_status"] = (
        "native_rate_aware_long_training_evidence_gate_blocked"
    )

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[campaign],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
                state_dict_path=state_dict_path,
            )
        ],
        official_replacement_authority_gates=[
            _official_replacement_authority_gate(ready=True)
        ],
        output_root=tmp_path / "queue",
        queue_artifact_path=queue_artifact_path,
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    assert row["blocked"] is False
    assert row["command_argv"]
    assert row["source_campaign_status"]["local_mlx_launch_command_ready"] is False
    assert row["command_argv"][row["command_argv"].index("--num-pairs") + 1] == "16"
    assert row["command_argv"][
        row["command_argv"].index("--planner-row-queue-artifact") + 1
    ] == queue_artifact_path.as_posix()
    assert row["command_argv"][
        row["command_argv"].index(
            "--snerv-official-trained-checkpoint-state-dict-path"
        )
        + 1
    ] == state_dict_path.as_posix()
    assert (
        row["command_argv"][
            row["command_argv"].index("--snerv-score-aware-long-training-epochs") + 1
        ]
        == "128"
    )
    assert row["local_mlx_long_training_allowed"] is False
    assert row["score_claim"] is False
    assert report["local_executable_command_row_count"] == 1


def test_lf_hf_queue_enables_scorer_loop_qat_only_after_measured_nondegenerate_feedback(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        candidate_feedback_rows=[_candidate_feedback_row(renderer_qat_ready=True)],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
                state_dict_path=state_dict_path,
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

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    assert row["blocked"] is False
    assert "--snerv-scorer-loop-qat" in row["command_argv"]
    assert row["command_argv"][
        row["command_argv"].index("--snerv-scorer-loop-max-trials") + 1
    ] == "1"
    assert row["command_argv"][
        row["command_argv"].index("--snerv-scorer-loop-max-archive-byte-growth") + 1
    ] == "0"
    qat_evidence = row["scorer_domain_evidence"][
        "scorer_loop_qat_nondegenerate_evidence"
    ]
    assert qat_evidence["passed"] is True
    assert qat_evidence["measured_num_pairs"] == 16
    assert qat_evidence["segnet_candidate_occupied_class_fraction"] == 0.6
    assert qat_evidence["segnet_candidate_target_class_coverage_fraction"] == 0.8


def test_lf_hf_queue_blocks_same_candidate_after_terminal_renderer_collapse_feedback(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    collapsed = _candidate_feedback_row()
    collapsed["created_utc"] = "2026-06-05T01:00:00+00:00"
    collapsed["candidate_id"] = "snerv_np600_haar_lv1_lfb1_int8_ceil178000"
    collapsed["snerv_scorer_domain_tether_passed"] = False
    collapsed["snerv_scorer_input_distribution_guard_proof_passed"] = False
    collapsed["snerv_renderer_nondegenerate_proof_passed"] = False
    collapsed["blockers"] = [
        "snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed"
    ]
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        candidate_feedback_rows=[_candidate_feedback_row(), collapsed],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
                state_dict_path=state_dict_path,
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

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    assert row["blocked"] is True
    assert row["command_argv"] == []
    command = row["unblock_command_argv"]
    assert command[:7] == [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "runtime",
        "--extra",
    ]
    assert command[command.index("--planner-row-id") + 1] == row["queue_row_id"]
    assert command[command.index("--num-pairs") + 1] == "16"
    assert (
        command[command.index("--snerv-score-aware-long-training-epochs") + 1]
        == "128"
    )
    assert command[
        command.index("--segnet-direct-live-target-mass-floor-weight") + 1
    ] == "0.5"
    assert command[
        command.index("--segnet-direct-live-target-min-ratio-floor-weight") + 1
    ] == "0.5"
    assert command[
        command.index("--segnet-direct-live-escape-warmup-epochs") + 1
    ] == "32"
    assert command[
        command.index("--segnet-direct-live-escape-class-multiplier") + 1
    ] == "16"
    assert report["next_unblock_command_argv"] == command
    unblock_contract = row["unblock_launch_authority_contract"]
    assert unblock_contract["schema"] == "snerv_lf_hf_queue_unblock_launch_contract.v1"
    assert unblock_contract["queue_unblock_kind"] == (
        "snerv_renderer_nondegenerate_smoke"
    )
    assert unblock_contract["queue_unblock_status_is_runnable_plan"] is True
    assert unblock_contract["queue_unblock_blockers"] == []
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed"
        in row["blockers"]
    )
    assert (
        row["scorer_domain_evidence"][
            "terminal_renderer_feedback_blockers_by_candidate"
        ][collapsed["candidate_id"]]
        == ["snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed"]
    )


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


def test_lf_hf_queue_ignores_ready_official_gate_unblock_command(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    stale_gate_command = ["uv", "run", "python", "stale_source_audit.py"]
    gate = {
        **_official_replacement_authority_gate(ready=True),
        "next_unblock_command_argv": stale_gate_command,
    }

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
                state_dict_path=state_dict_path,
            )
        ],
        official_replacement_authority_gates=[gate],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_replacement_authority_evidence"][
        "official_tub_lf_hf_decoder_replacement_ready"
    ] is True
    assert report["official_replacement_authority_evidence"]["queue_blockers"] == []
    assert report["next_unblock_command_argv"] != stale_gate_command
    assert "stale_source_audit.py" not in report["next_unblock_command_argv"]


def test_lf_hf_queue_prefers_ready_official_gate_over_newer_blocked_gate(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    ready_gate = {
        **_official_replacement_authority_gate(ready=True),
        "generated_utc": "2026-06-05T22:45:53+00:00",
        "_source_path": "/ssd/v5_ready_gate.json",
        "_source_sha256": "5" * 64,
    }
    newer_blocked_gate = {
        **_official_replacement_authority_gate(ready=False),
        "generated_utc": "2026-06-06T00:00:00+00:00",
        "_source_path": "/ssd/newer_blocked_gate.json",
        "_source_sha256": "6" * 64,
        "queue_blockers": ["stale_lower_readiness_blocker"],
    }

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
                state_dict_path=state_dict_path,
            )
        ],
        official_replacement_authority_gates=[newer_blocked_gate, ready_gate],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    official = report["official_replacement_authority_evidence"]
    assert official["selection_policy"] == "prefer_ready_artifacts_then_newest_generated_utc"
    assert official["source_path"] == "/ssd/v5_ready_gate.json"
    assert official["source_sha256"] == "5" * 64
    assert official["ready_artifact_count"] == 1
    assert official["queue_blockers"] == []
    assert official["ignored_lower_readiness_artifacts"] == [
        {
            "source_path": "/ssd/newer_blocked_gate.json",
            "source_sha256": "6" * 64,
            "generated_utc": "2026-06-06T00:00:00+00:00",
            "official_tub_lf_hf_decoder_replacement_ready": False,
            "queue_blockers": ["stale_lower_readiness_blocker"],
        }
    ]
    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    assert row["official_gate_selected_path"] == "/ssd/v5_ready_gate.json"
    assert row["official_gate_selected_sha256"] == "5" * 64
    assert row["official_gate_selection_policy"] == (
        "prefer_ready_artifacts_then_newest_generated_utc"
    )
    assert "stale_lower_readiness_blocker" not in row["blockers"]


def test_lf_hf_queue_ready_gate_cannot_mask_missing_source_forward_proof(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=False,
                full_tub_parity=True,
                state_dict_path=state_dict_path,
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

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    blockers = set(row["blockers"])
    contradiction = (
        "snerv_ready_official_tub_lf_hf_gate_contradicts_source_forward_queue_blockers"
    )
    assert row["official_replacement_authority_evidence"][
        "official_tub_lf_hf_decoder_replacement_ready"
    ] is True
    assert contradiction in blockers
    assert contradiction in row["official_gate_source_forward_contradiction_blockers"]
    assert "snerv_official_mfu_hfr_tub_numerical_source_forward_proof_missing" in (
        blockers
    )
    assert row["blocked"] is True
    assert row["command_argv"] == []


def test_lf_hf_rebuild_command_uses_candidate_feedback_source_alias(
    tmp_path: Path,
) -> None:
    feedback = {
        **_candidate_feedback_row(guard_proof_passed=True),
        "_candidate_feedback_source_path": "/ssd/feedback_from_plan_wrapper.json",
    }
    feedback.pop("_source_path", None)

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        candidate_feedback_rows=[feedback],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    command = report["runnable_rebuild_command_argv"]
    assert command[command.index("--candidate-feedback-row") + 1] == (
        "/ssd/feedback_from_plan_wrapper.json"
    )
    assert report["input_source_paths"]["candidate_feedback_rows"] == [
        "/ssd/feedback_from_plan_wrapper.json"
    ]


def test_lf_hf_queue_accepts_checkpoint_export_as_lf_payload_evidence(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_checkpoint_export_lf_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["lf_payload_evidence_row_count"] == 1
    row = report["lf_payload_evidence_rows"][0]
    assert row["evidence_kind"] == "checkpoint_export_lf_payload_section"
    assert row["lf_payload_bytes"] == 388
    assert row["raw_lf_bytes"] == 4096
    assert row["packet_bytes"] == 87344
    assert report["selected_lf_payload_evidence"]["lf_payload_bytes"] == 388
    assert "snerv_lf_hf_measured_lf_payload_report_missing" not in report["blockers"]


def test_lf_hf_queue_prefers_passing_scorer_guard_feedback(
    tmp_path: Path,
) -> None:
    stale_pass = {
        **_candidate_feedback_row(guard_proof_passed=True),
        "created_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/passing_guard.json",
    }
    newer_fail = {
        **_candidate_feedback_row(guard_proof_passed=False),
        "created_utc": "2026-06-05T01:00:00+00:00",
        "_source_path": "/ssd/newer_failing_guard.json",
    }

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        candidate_feedback_rows=[newer_fail, stale_pass],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    scorer = report["scorer_domain_evidence"]
    assert scorer["artifact_count"] == 2
    assert scorer["source_path"] == "/ssd/passing_guard.json"
    assert scorer["scorer_domain_tether_proof_passed"] is True
    assert scorer["scorer_input_distribution_guard_proof_passed"] is True
    assert scorer["queue_blockers"] == []
    assert "snerv_scorer_input_distribution_guard_missing" not in report["blockers"]


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


def test_lf_hf_queue_consumes_tub_source_fixture_without_full_authority(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=False,
        full_tub_parity=False,
        tub_source_fixture=True,
    )
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[
            _campaign_plan(
                blockers=(
                    "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity",
                    "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                    "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
                )
            )
        ],
        source_forward_artifacts=[source],
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
    evidence = official["source_forward_evidence"]
    blockers = set(official["blockers"])

    assert evidence["official_tub_source_fixture_forward_parity_proven"] is True
    assert "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity" in (
        evidence["tub_source_fixture_closed_blockers"]
    )
    assert (
        "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity"
        in evidence["closed_campaign_blockers"]
    )
    assert (
        "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity"
        not in blockers
    )
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in blockers
    assert "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing" in blockers
    assert official["blocked"] is True
    assert official["command_argv"] == []
    assert official["score_claim"] is False


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


def test_lf_hf_queue_consumes_nested_state_mapping_manifest_without_source_authority(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
    )
    source["official_trained_checkpoint_mapping_manifest"] = {
        "schema": "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1",
        "official_trained_checkpoint_loaded": True,
        "official_trained_checkpoint_state_dict_mapping_verified": True,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": True,
        "official_tub_temporal_encoder_weight_mapping_proven": False,
        "blockers": [
            "snerv_official_trained_checkpoint_state_dict_mapping_missing",
            "snerv_official_mfu_hfr_tub_weight_mapping_missing",
            "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    source["blockers"] = [
        "snerv_official_trained_checkpoint_state_dict_mapping_missing",
        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
        "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
        "snerv_official_trained_checkpoint_source_forward_replay_missing",
    ]

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        source_forward_artifacts=[source],
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

    assert evidence["official_trained_checkpoint_loaded"] is True
    assert evidence["official_trained_checkpoint_state_dict_mapping_verified"] is True
    assert evidence["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" not in blockers
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" not in blockers
    assert "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded" in blockers
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in blockers
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert evidence["source_forward_replay_authority"] is False
    assert official["score_claim"] is False
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


def test_lf_hf_queue_preserves_scalar_skip_high_xray_details(
    tmp_path: Path,
) -> None:
    xray = {
        **_value_domain_xray(noncollapse=False),
        "official_skip_high_value_domain": {
            "schema": "snerv_official_skip_high_value_domain_summary.v1",
            "scalar_mean_storage": True,
            "max_unclipped_outside_0_255_fraction_by_channel": 0.5,
            "blockers": [
                "snerv_official_skip_high_scalar_mean_receiver_range_unfit"
            ],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "official_scalar_skip_high_value_domain_scan": {
            "schema": "snerv_official_scalar_skip_high_value_domain_scan.v1",
            "scan_executed": True,
            "range_safe_scalar_value_count": 0,
            "safe_scalar_value_count": 0,
            "blockers": [
                "snerv_official_scalar_skip_high_no_range_safe_scalar_found"
            ],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "recommended_next_actions": [
            "rerun_bounded_snerv_smoke_with_non_scalar_skip_high_storage"
        ],
        "blockers": [
            "snerv_receiver_value_domain_xray_false_authority",
            "snerv_official_skip_high_scalar_mean_receiver_range_unfit",
            "snerv_official_scalar_skip_high_no_range_safe_scalar_found",
        ],
    }
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[xray],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    evidence = report["value_domain_evidence"]
    assert evidence["official_skip_high_value_domain"]["scalar_mean_storage"] is True
    assert evidence["official_scalar_skip_high_value_domain_scan"][
        "safe_scalar_value_count"
    ] == 0
    assert evidence["recommended_next_actions"] == [
        "rerun_bounded_snerv_smoke_with_non_scalar_skip_high_storage"
    ]
    assert "snerv_official_skip_high_scalar_mean_receiver_range_unfit" in evidence[
        "queue_blockers"
    ]


def test_lf_hf_queue_consumes_hf_residual_receiver_payload_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        hf_residual_receiver_payload_proofs=[_hf_residual_receiver_payload_proof()],
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
    evidence = row["hf_residual_payload_evidence"]
    assert evidence["receiver_payload_implemented"] is True
    assert evidence["receiver_decode_proven"] is True
    assert evidence["section_native_byte_telemetry_present"] is True
    assert "snerv_hf_residual_generator_receiver_payload_not_implemented" not in (
        blockers
    )
    assert report["hf_residual_payload_evidence"]["closed_campaign_blockers"] == [
        "snerv_hf_residual_generator_receiver_payload_not_implemented"
    ]
    assert "--hf-residual-receiver-payload-proof" in report[
        "runnable_rebuild_command_argv"
    ]
    assert (
        "snerv_lf_conditioned_hf_residual_receiver_runtime_binding_missing"
        in blockers
    )
    assert "snerv_lf_conditioned_hf_bounded_training_binding_missing" not in blockers
    assert row["blocked"] is True
    assert row["status"] == "blocked_until_prerequisite_evidence"
    assert "snerv_lf_hf_source_forward_artifact_missing" in row[
        "inherited_queue_authority_blockers"
    ]
    assert (
        "snerv_official_tub_lf_hf_decoder_replacement_authority_gate_missing"
        in row["inherited_queue_authority_blockers"]
    )
    assert row["command_argv"] == []
    assert row["unblock_command_argv"]
    assert "tools/build_snerv_lf_hf_runtime_binding_proof.py" in row[
        "unblock_command_argv"
    ]
    assert "--hf-residual-receiver-payload-proof" in row["unblock_command_argv"]
    assert row["launch_authority_contract"]["queue_status_is_runnable_plan"] is False
    assert (
        "snerv_lf_conditioned_hf_residual_receiver_runtime_binding_missing"
        in row["launch_authority_contract"]["queue_launch_blockers"]
    )
    contract = row["bounded_training_binding_contract"]
    assert contract["schema"] == "snerv_lf_hf_bounded_training_binding_contract.v1"
    assert contract["runner_actuator_required"] is True
    assert contract["runner_actuator_bound"] is False
    assert contract["runner_actuator"] is None
    assert contract["family_bounded_training_blocker"] == (
        "snerv_lf_conditioned_hf_bounded_training_binding_missing"
    )
    assert "snerv_lf_conditioned_hf_bounded_training_binding_missing" in contract[
        "blockers"
    ]


def test_lf_hf_queue_consumes_joint_codebook_receiver_payload_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        joint_codebook_receiver_payload_proofs=[
            _joint_codebook_receiver_payload_proof()
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "joint_lf_hf_factorized_codebook"
    )
    blockers = set(row["blockers"])
    evidence = row["joint_codebook_evidence"]
    assert evidence["receiver_payload_implemented"] is True
    assert evidence["receiver_decode_proven"] is True
    assert evidence["numpy_receiver_decode"] is True
    assert evidence["section_native_byte_telemetry_present"] is True
    assert "snerv_joint_lf_hf_factorized_codebook_not_implemented" not in blockers
    assert "snerv_joint_lf_hf_codebook_numpy_receiver_missing" not in blockers
    assert "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing" not in blockers
    assert report["joint_codebook_evidence"]["closed_campaign_blockers"] == [
        "snerv_joint_lf_hf_factorized_codebook_not_implemented",
        "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
        "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
    ]
    assert "--joint-codebook-receiver-payload-proof" in report[
        "runnable_rebuild_command_argv"
    ]
    assert (
        "snerv_joint_lf_hf_factorized_codebook_receiver_runtime_binding_missing"
        in blockers
    )
    assert "snerv_joint_lf_hf_bounded_training_binding_missing" not in blockers
    assert row["blocked"] is True
    assert row["status"] == "blocked_until_prerequisite_evidence"
    assert row["command_argv"] == []
    assert row["unblock_command_argv"]
    assert "tools/build_snerv_lf_hf_runtime_binding_proof.py" in row[
        "unblock_command_argv"
    ]
    assert "--joint-codebook-receiver-payload-proof" in row["unblock_command_argv"]
    assert row["launch_authority_contract"]["queue_status_is_runnable_plan"] is False
    assert (
        "snerv_joint_lf_hf_factorized_codebook_receiver_runtime_binding_missing"
        in row["launch_authority_contract"]["queue_launch_blockers"]
    )
    contract = row["bounded_training_binding_contract"]
    assert contract["schema"] == "snerv_lf_hf_bounded_training_binding_contract.v1"
    assert contract["runner_actuator_required"] is True
    assert contract["runner_actuator_bound"] is False
    assert contract["runner_actuator"] is None
    assert contract["family_bounded_training_blocker"] == (
        "snerv_joint_lf_hf_bounded_training_binding_missing"
    )
    assert "snerv_joint_lf_hf_bounded_training_binding_missing" in contract[
        "blockers"
    ]


def test_lf_conditioned_hf_runtime_binding_emits_bounded_smoke(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        hf_residual_receiver_payload_proofs=[_hf_residual_receiver_payload_proof()],
        lf_hf_runtime_binding_proofs=[
            _lf_hf_runtime_binding_proof("lf_conditioned_hf_residual_generator")
        ],
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
    assert row["blocked"] is False
    assert row["status"] == "local_bounded_smoke_ready_no_authority"
    assert row["command_argv"]
    assert row["command_argv"][
        row["command_argv"].index("--snerv-lf-hf-solution-family") + 1
    ] == "lf_conditioned_hf_residual_generator"
    assert row["launch_authority_contract"]["queue_status_is_runnable_plan"] is True
    contract = row["bounded_training_binding_contract"]
    assert contract["runner_actuator_bound"] is True
    assert contract["runner_actuator"]["consumes_solution_family"] == (
        "lf_conditioned_hf_residual_generator"
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_lf_conditioned_hf_runtime_binding_emits_renderer_unblock_smoke_when_blocked(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[
            _campaign_plan(blockers=("snerv_renderer_nondegenerate_smoke_missing",))
        ],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        hf_residual_receiver_payload_proofs=[_hf_residual_receiver_payload_proof()],
        lf_hf_runtime_binding_proofs=[
            _lf_hf_runtime_binding_proof("lf_conditioned_hf_residual_generator")
        ],
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
    assert row["blocked"] is True
    assert row["status"] == "blocked_until_prerequisite_evidence"
    assert row["command_argv"] == []
    assert row["unblock_command_argv"]
    assert row["unblock_command_argv"][
        row["unblock_command_argv"].index("--snerv-lf-hf-solution-family") + 1
    ] == "lf_conditioned_hf_residual_generator"
    assert row["launch_authority_contract"]["queue_status_is_runnable_plan"] is False
    assert (
        row["unblock_launch_authority_contract"][
            "queue_unblock_status_is_runnable_plan"
        ]
        is True
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_joint_lf_hf_runtime_binding_emits_bounded_smoke(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        joint_codebook_receiver_payload_proofs=[
            _joint_codebook_receiver_payload_proof()
        ],
        lf_hf_runtime_binding_proofs=[
            _lf_hf_runtime_binding_proof("joint_lf_hf_factorized_codebook")
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "joint_lf_hf_factorized_codebook"
    )
    assert row["blocked"] is False
    assert row["status"] == "local_bounded_smoke_ready_no_authority"
    assert row["command_argv"]
    assert row["command_argv"][
        row["command_argv"].index("--snerv-lf-hf-solution-family") + 1
    ] == "joint_lf_hf_factorized_codebook"
    assert row["launch_authority_contract"]["queue_status_is_runnable_plan"] is True
    contract = row["bounded_training_binding_contract"]
    assert contract["runner_actuator_bound"] is True
    assert contract["runner_actuator"]["consumes_solution_family"] == (
        "joint_lf_hf_factorized_codebook"
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_lf_hf_queue_emits_temporal_lf_predictor_payload_proof_unblock_command(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "temporal_lf_predictor_gate"
    )
    command = row["unblock_command_argv"]

    assert row["blocked"] is True
    assert row["command_argv"] == []
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_temporal_lf_predictor_payload_proof.py",
    ]
    assert command[command.index("--packet") + 1] == "/ssd/candidate.snar"
    assert command[command.index("--pair-indices") + 1] == "0,1"
    assert command[command.index("--output-json") + 1].startswith(
        (tmp_path / "queue" / row["queue_row_id"]).as_posix()
    )
    assert command[command.index("--output-payload") + 1].endswith(
        "snerv_temporal_lf_predictor.stlp"
    )
    assert report["next_unblock_command_argv"][3] == (
        "tools/build_snerv_lf_conditioned_hf_residual_payload_proof.py"
    )
    assert "snerv_lf_hf_source_forward_artifact_missing" not in row["blockers"]
    assert "snerv_temporal_lf_predictor_gate_not_implemented" in row["blockers"]
    assert (
        "snerv_temporal_lf_predictor_correction_stream_not_byte_charged"
        in row["blockers"]
    )


def test_lf_hf_queue_consumes_temporal_lf_predictor_receiver_payload_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        temporal_lf_predictor_receiver_payload_proofs=[
            _temporal_lf_predictor_receiver_payload_proof()
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "temporal_lf_predictor_gate"
    )
    blockers = set(row["blockers"])
    evidence = row["temporal_lf_predictor_evidence"]
    assert evidence["receiver_payload_implemented"] is True
    assert evidence["receiver_decode_proven"] is True
    assert evidence["numpy_receiver_decode"] is True
    assert evidence["correction_stream_byte_charged"] is True
    assert evidence["section_native_byte_telemetry_present"] is True
    assert "snerv_temporal_lf_predictor_gate_not_implemented" not in blockers
    assert (
        "snerv_temporal_lf_predictor_correction_stream_not_byte_charged"
        not in blockers
    )
    assert report["temporal_lf_predictor_evidence"]["closed_campaign_blockers"] == [
        "snerv_temporal_lf_predictor_gate_not_implemented",
        "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
    ]
    assert "--temporal-lf-predictor-receiver-payload-proof" in report[
        "runnable_rebuild_command_argv"
    ]
    assert "snerv_lf_hf_source_forward_artifact_missing" not in blockers
    assert "snerv_temporal_lf_predictor_receiver_runtime_binding_missing" in blockers
    assert row["blocked"] is True
    assert row["command_argv"] == []
    command = row["unblock_command_argv"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_hf_runtime_binding_proof.py",
    ]
    assert "--temporal-lf-predictor-receiver-payload-proof" in command
    assert (
        command[command.index("--temporal-lf-predictor-receiver-payload-proof") + 1]
        == "/ssd/snerv_temporal_lf_predictor_receiver_proof.json"
    )


def test_lf_hf_queue_records_missing_lf_super_resolution_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "lf_super_resolution_from_tiny_anchor"
    )
    blockers = set(row["blockers"])

    assert report["lf_super_resolution_evidence"]["artifact_count"] == 0
    assert (
        report["current_state"]["lf_super_resolution_evidence"]["schema"]
        == "snerv_lf_super_resolution_tiny_anchor_evidence.v1"
    )
    assert "snerv_lf_super_resolution_receiver_payload_not_implemented" in blockers
    assert "snerv_lf_downsampled_anchor_component_deltas_missing" in blockers
    assert row["blocked"] is True
    assert row["command_argv"] == []
    command = row["unblock_command_argv"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_super_resolution_tiny_anchor_payload_proof.py",
    ]
    assert command[command.index("--packet") + 1] == "/ssd/candidate.snar"
    assert command[command.index("--pair-indices") + 1] == "0,1"
    assert command[command.index("--output-payload") + 1].endswith(
        "snerv_lf_super_resolution_tiny_anchor.slsr"
    )
    assert report["next_unblock_command_argv"][3] == (
        "tools/build_snerv_lf_conditioned_hf_residual_payload_proof.py"
    )


def test_lf_hf_queue_consumes_lf_super_resolution_receiver_payload_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        lf_super_resolution_receiver_payload_proofs=[
            _lf_super_resolution_receiver_payload_proof()
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "lf_super_resolution_from_tiny_anchor"
    )
    blockers = set(row["blockers"])
    evidence = row["lf_super_resolution_evidence"]
    assert evidence["receiver_payload_implemented"] is True
    assert evidence["receiver_decode_proven"] is True
    assert evidence["numpy_receiver_decode"] is True
    assert evidence["tiny_anchor_component_deltas_present"] is True
    assert evidence["component_delta_scope"] == (
        "receiver_pixel_domain_not_scorer_component"
    )
    assert evidence["section_native_byte_telemetry_present"] is True
    assert "snerv_lf_super_resolution_receiver_payload_not_implemented" not in blockers
    assert "snerv_lf_downsampled_anchor_component_deltas_missing" not in blockers
    assert report["lf_super_resolution_evidence"]["closed_campaign_blockers"] == [
        "snerv_lf_super_resolution_receiver_payload_not_implemented",
        "snerv_lf_downsampled_anchor_component_deltas_missing",
    ]
    assert "--lf-super-resolution-receiver-payload-proof" in report[
        "runnable_rebuild_command_argv"
    ]
    assert "snerv_lf_super_resolution_receiver_runtime_binding_missing" in blockers
    assert row["blocked"] is True
    assert row["command_argv"] == []
    command = row["unblock_command_argv"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_hf_runtime_binding_proof.py",
    ]
    assert "--lf-super-resolution-receiver-payload-proof" in command


def test_lf_hf_queue_emits_spectral_band_allocator_payload_proof_unblock_command(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "score_tethered_spectral_band_allocator"
    )
    blockers = set(row["blockers"])
    command = row["unblock_command_argv"]

    assert "snerv_score_tethered_lf_hf_band_allocator_not_implemented" in blockers
    assert "snerv_mfu_hfr_section_native_byte_telemetry_missing" in blockers
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_spectral_band_allocator_payload_proof.py",
    ]
    assert command[command.index("--packet") + 1] == "/ssd/candidate.snar"
    assert command[command.index("--pair-indices") + 1] == "0,1"
    assert command[command.index("--output-payload") + 1].endswith(
        "snerv_score_tethered_spectral_band_allocator.ssba"
    )
    assert row["blocked"] is True
    assert row["command_argv"] == []


def test_lf_hf_queue_consumes_spectral_band_allocator_receiver_payload_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        spectral_band_allocator_receiver_payload_proofs=[
            _spectral_band_allocator_receiver_payload_proof()
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "score_tethered_spectral_band_allocator"
    )
    blockers = set(row["blockers"])
    evidence = row["spectral_band_allocator_evidence"]
    assert evidence["receiver_payload_implemented"] is True
    assert evidence["receiver_decode_proven"] is True
    assert evidence["numpy_receiver_decode"] is True
    assert evidence["score_tethered_allocation_implemented"] is True
    assert evidence["section_native_byte_telemetry_present"] is True
    assert evidence["human_readable_payload_labels"] is False
    assert (
        "snerv_score_tethered_lf_hf_band_allocator_not_implemented"
        not in blockers
    )
    assert "snerv_mfu_hfr_section_native_byte_telemetry_missing" not in blockers
    assert report["spectral_band_allocator_evidence"][
        "closed_campaign_blockers"
    ] == [
        "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
        "snerv_mfu_hfr_section_native_byte_telemetry_missing",
    ]
    assert "--spectral-band-allocator-receiver-payload-proof" in report[
        "runnable_rebuild_command_argv"
    ]
    assert (
        "snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing"
        in blockers
    )
    assert row["blocked"] is True
    assert row["command_argv"] == []
    command = row["unblock_command_argv"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_hf_runtime_binding_proof.py",
    ]
    assert "--spectral-band-allocator-receiver-payload-proof" in command


def test_lf_hf_queue_emits_lf_latent_hyperprior_payload_proof_unblock_command(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        value_domain_xray_reports=[_value_domain_xray(noncollapse=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "entropy_modeled_lf_latent_hyperprior"
    )
    blockers = set(row["blockers"])
    command = row["unblock_command_argv"]

    assert "snerv_lf_latent_hyperprior_not_implemented" in blockers
    assert "snerv_lf_latent_hyperprior_numpy_decoder_missing" in blockers
    assert "snerv_lf_latent_hyperprior_receiver_replay_missing" in blockers
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_latent_hyperprior_payload_proof.py",
    ]
    assert command[command.index("--packet") + 1] == "/ssd/candidate.snar"
    assert command[command.index("--pair-indices") + 1] == "0,1"
    assert command[command.index("--output-payload") + 1].endswith(
        "snerv_lf_latent_hyperprior.slhp"
    )
    assert row["blocked"] is True
    assert row["command_argv"] == []


def test_lf_hf_queue_consumes_lf_latent_hyperprior_receiver_payload_proof(
    tmp_path: Path,
) -> None:
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        lf_latent_hyperprior_receiver_payload_proofs=[
            _lf_latent_hyperprior_receiver_payload_proof()
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "entropy_modeled_lf_latent_hyperprior"
    )
    blockers = set(row["blockers"])
    evidence = row["lf_latent_hyperprior_evidence"]
    assert evidence["receiver_payload_implemented"] is True
    assert evidence["receiver_decode_proven"] is True
    assert evidence["numpy_receiver_decode"] is True
    assert evidence["entropy_model_implemented"] is True
    assert evidence["hyperprior_scale_present"] is True
    assert evidence["receiver_replay_proven"] is True
    assert evidence["section_native_byte_telemetry_present"] is True
    assert evidence["human_readable_payload_labels"] is False
    assert "snerv_lf_latent_hyperprior_not_implemented" not in blockers
    assert "snerv_lf_latent_hyperprior_numpy_decoder_missing" not in blockers
    assert "snerv_lf_latent_hyperprior_receiver_replay_missing" not in blockers
    assert report["lf_latent_hyperprior_evidence"][
        "closed_campaign_blockers"
    ] == [
        "snerv_lf_latent_hyperprior_not_implemented",
        "snerv_lf_latent_hyperprior_numpy_decoder_missing",
        "snerv_lf_latent_hyperprior_receiver_replay_missing",
    ]
    assert "--lf-latent-hyperprior-receiver-payload-proof" in report[
        "runnable_rebuild_command_argv"
    ]
    assert "snerv_lf_latent_hyperprior_runtime_binding_missing" in blockers
    assert row["blocked"] is True
    assert row["command_argv"] == []
    command = row["unblock_command_argv"]
    assert command[:4] == [
        "uv",
        "run",
        "python",
        "tools/build_snerv_lf_hf_runtime_binding_proof.py",
    ]
    assert "--lf-latent-hyperprior-receiver-payload-proof" in command


def test_lf_hf_queue_consumes_runtime_binding_proof_then_blocks_training_binding(
    tmp_path: Path,
) -> None:
    runtime_proof_path = tmp_path / "runtime_binding.json"
    runtime_proof = _lf_hf_runtime_binding_proof(
        "temporal_lf_predictor_gate",
        "lf_super_resolution_from_tiny_anchor",
        "score_tethered_spectral_band_allocator",
        "entropy_modeled_lf_latent_hyperprior",
        source_path=runtime_proof_path.as_posix(),
    )
    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        temporal_lf_predictor_receiver_payload_proofs=[
            _temporal_lf_predictor_receiver_payload_proof()
        ],
        lf_super_resolution_receiver_payload_proofs=[
            _lf_super_resolution_receiver_payload_proof()
        ],
        spectral_band_allocator_receiver_payload_proofs=[
            _spectral_band_allocator_receiver_payload_proof()
        ],
        lf_latent_hyperprior_receiver_payload_proofs=[
            _lf_latent_hyperprior_receiver_payload_proof()
        ],
        lf_hf_runtime_binding_proofs=[runtime_proof],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["runtime_binding_evidence"]["artifact_count"] == 1
    assert report["runtime_binding_evidence"]["source_path"] == (
        runtime_proof_path.as_posix()
    )
    assert "--lf-hf-runtime-binding-proof" in report["runnable_rebuild_command_argv"]
    row_by_family = {row["solution_family"]: row for row in report["queue_rows"]}
    temporal = row_by_family["temporal_lf_predictor_gate"]
    assert "snerv_temporal_lf_predictor_receiver_runtime_binding_missing" not in (
        temporal["blockers"]
    )
    assert "snerv_temporal_lf_predictor_bounded_training_binding_missing" in (
        temporal["blockers"]
    )
    sr = row_by_family["lf_super_resolution_from_tiny_anchor"]
    assert "snerv_lf_super_resolution_receiver_runtime_binding_missing" not in (
        sr["blockers"]
    )
    assert "snerv_lf_super_resolution_bounded_training_binding_missing" in (
        sr["blockers"]
    )
    spectral = row_by_family["score_tethered_spectral_band_allocator"]
    assert (
        "snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing"
        not in spectral["blockers"]
    )
    assert (
        "snerv_score_tethered_lf_hf_band_allocator_bounded_training_binding_missing"
        in spectral["blockers"]
    )
    hyperprior = row_by_family["entropy_modeled_lf_latent_hyperprior"]
    assert "snerv_lf_latent_hyperprior_runtime_binding_missing" not in (
        hyperprior["blockers"]
    )
    assert "snerv_lf_latent_hyperprior_bounded_training_binding_missing" in (
        hyperprior["blockers"]
    )
    for row in (temporal, sr, spectral, hyperprior):
        assert row["blocked"] is True
        assert row["command_argv"] == []
        assert row["unblock_command_argv"] == []
        assert row["runtime_binding_evidence"]["artifact_count"] == 1
        contract = row["bounded_training_binding_contract"]
        assert contract["schema"] == "snerv_lf_hf_bounded_training_binding_contract.v1"
        assert contract["runner_actuator_required"] is True
        assert contract["runner_actuator_bound"] is False
        assert contract["runner_actuator"] is None
        assert contract["family_bounded_training_blocker"] in contract["blockers"]


def test_lf_hf_queue_keeps_distribution_guard_blocker_for_tether_only_proof(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
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
                state_dict_path=state_dict_path,
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
    state_dict_path = _write_fake_state_dict(tmp_path)
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
                state_dict_path=state_dict_path,
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


def test_lf_hf_queue_blocks_source_authority_without_value_state_artifact(
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

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    evidence = row["source_forward_evidence"]

    assert row["blocked"] is True
    assert row["command_argv"] == []
    assert evidence["source_forward_replay_authority"] is False
    assert (
        evidence["official_trained_checkpoint_state_dict_value_artifact_ready"]
        is False
    )
    assert (
        "snerv_official_trained_checkpoint_state_dict_value_artifact_missing"
        in row["blockers"]
    )


def test_lf_hf_queue_preserves_raw_source_replay_blocker_despite_ready_flags(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
        state_dict_path=state_dict_path,
    )
    source["blockers"] = [
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing"
    ]

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        source_forward_artifacts=[source],
        official_replacement_authority_gates=[
            _official_replacement_authority_gate(ready=True)
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    evidence = row["source_forward_evidence"]
    blockers = set(row["blockers"])

    assert row["blocked"] is True
    assert row["command_argv"] == []
    assert evidence["source_forward_replay_authority"] is False
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in (
        blockers
    )
    assert (
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing"
        in blockers
    )
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        not in evidence["closed_campaign_blockers"]
    )


def test_lf_hf_queue_rejects_metadata_only_non_npz_state_artifact(
    tmp_path: Path,
) -> None:
    state_dict_path = tmp_path / "metadata_only_state_dict.npz"
    state_dict_path.write_bytes(b"not a zip-backed npz")

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
                state_dict_path=state_dict_path,
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

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    evidence = row["source_forward_evidence"]

    assert row["blocked"] is True
    assert row["command_argv"] == []
    assert evidence["source_forward_replay_authority"] is False
    assert (
        evidence["official_trained_checkpoint_state_dict_value_artifact_ready"]
        is False
    )
    assert (
        evidence["official_trained_checkpoint_state_dict_slice_npz_opened"]
        is False
    )
    assert (
        "snerv_official_trained_checkpoint_state_dict_value_artifact_npz_invalid"
        in row["blockers"]
    )


def test_lf_hf_queue_rejects_state_artifact_byte_mismatch(
    tmp_path: Path,
) -> None:
    state_dict_path = _write_fake_state_dict(tmp_path)
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
        state_dict_path=state_dict_path,
    )
    source["official_trained_checkpoint_state_dict_slice_bytes"] = (
        state_dict_path.stat().st_size + 1
    )

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=[_lf_sweep_report()],
        reroute_queues=[_reroute_queue(row_count=1)],
        campaign_plans=[_campaign_plan(blockers=())],
        source_forward_artifacts=[source],
        official_replacement_authority_gates=[
            _official_replacement_authority_gate(ready=True)
        ],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    row = next(
        item
        for item in report["queue_rows"]
        if item["solution_family"] == "official_tub_lf_hf_decoder_replacement"
    )
    evidence = row["source_forward_evidence"]

    assert row["blocked"] is True
    assert row["command_argv"] == []
    assert evidence["source_forward_replay_authority"] is False
    assert (
        evidence["official_trained_checkpoint_state_dict_slice_npz_opened"]
        is True
    )
    assert (
        evidence["official_trained_checkpoint_state_dict_slice_bytes_match_report"]
        is False
    )
    assert (
        "snerv_official_trained_checkpoint_state_dict_value_artifact_bytes_mismatch"
        in row["blockers"]
    )


def test_lf_hf_replacement_queue_cli_writes_ssd_handoff_artifacts(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.json"
    reroute_path = tmp_path / "reroute.json"
    campaign_path = tmp_path / "campaign.json"
    source_forward_path = tmp_path / "source_forward.json"
    feedback_path = tmp_path / "candidate_feedback.json"
    residual_payload_path = tmp_path / "hf_residual_payload.json"
    joint_codebook_path = tmp_path / "joint_codebook_payload.json"
    spectral_allocator_path = tmp_path / "spectral_allocator_payload.json"
    hyperprior_path = tmp_path / "lf_latent_hyperprior_payload.json"
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
    residual_payload_path.write_text(
        json.dumps(_hf_residual_receiver_payload_proof()),
        encoding="utf-8",
    )
    joint_codebook_path.write_text(
        json.dumps(_joint_codebook_receiver_payload_proof()),
        encoding="utf-8",
    )
    spectral_allocator_path.write_text(
        json.dumps(_spectral_band_allocator_receiver_payload_proof()),
        encoding="utf-8",
    )
    hyperprior_path.write_text(
        json.dumps(_lf_latent_hyperprior_receiver_payload_proof()),
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
            "--source-forward-artifact",
            source_forward_path.as_posix(),
            "--candidate-feedback-row",
            feedback_path.as_posix(),
            "--hf-residual-receiver-payload-proof",
            residual_payload_path.as_posix(),
            "--joint-codebook-receiver-payload-proof",
            joint_codebook_path.as_posix(),
            "--spectral-band-allocator-receiver-payload-proof",
            spectral_allocator_path.as_posix(),
            "--lf-latent-hyperprior-receiver-payload-proof",
            hyperprior_path.as_posix(),
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
    assert payload["hf_residual_payload_evidence"]["source_path"] == (
        residual_payload_path.as_posix()
    )
    assert payload["hf_residual_payload_evidence"]["receiver_decode_proven"] is True
    assert payload["joint_codebook_evidence"]["source_path"] == (
        joint_codebook_path.as_posix()
    )
    assert payload["joint_codebook_evidence"]["numpy_receiver_decode"] is True
    assert payload["spectral_band_allocator_evidence"]["source_path"] == (
        spectral_allocator_path.as_posix()
    )
    assert payload["spectral_band_allocator_evidence"][
        "score_tethered_allocation_implemented"
    ] is True
    assert payload["lf_latent_hyperprior_evidence"]["source_path"] == (
        hyperprior_path.as_posix()
    )
    assert payload["lf_latent_hyperprior_evidence"][
        "entropy_model_implemented"
    ] is True
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


def _checkpoint_export_lf_report() -> dict[str, object]:
    return {
        "schema": "snerv_checkpoint_archive_export.v1",
        "report_path": "/ssd/snerv_checkpoint_archive_export.json",
        "_source_path": "/ssd/snerv_checkpoint_archive_export.json",
        "_source_sha256": "0" * 64,
        "packet_path": "/ssd/snerv_checkpoint_packet.bin",
        "packet_bytes": 87344,
        "packet_sha256": "1" * 64,
        "lf_payload_codec": "spatial_delta_zigzag_leb128_lzma",
        "lf_payload_codec_selected": "spatial_delta_zigzag_leb128_lzma",
        "lf_payload_report_status": "receiver_visible_lf_payload_accounting_verified",
        "lf_payload_section_bytes": 388,
        "lf_payload_codec_selection_report": {
            "schema": "snerv_lf_payload_codec_selection.v1",
            "raw_bytes": 4096,
            "canonical_int64_raw_bytes": 4096,
            "payload_bytes": 384,
            "section_bytes": 388,
        },
        "packet_section_bytes": {
            "metadata_payload": 4,
            "lf_payload": 388,
            "decoder_payload": 76093,
            "step_map_packet": 499,
        },
        "receiver_contract_satisfied": False,
        "blockers": ["receiver_proof_not_requested"],
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


def _write_fake_state_dict(tmp_path: Path) -> Path:
    path = tmp_path / "official_snerv_t_state_dict.npz"
    np.savez(
        path,
        **{
            "encoder.1.weight": np.zeros((1,), dtype=np.float32),
            "encoder.2.weight": np.zeros((1,), dtype=np.float32),
            "decoder.4.weight": np.zeros((1,), dtype=np.float32),
        },
    )
    return path


def _source_forward_artifact(
    *,
    official_export_bound: bool = False,
    receiver_consumes_output2: bool = False,
    source_authority: bool = False,
    full_tub_parity: bool = False,
    native_hfr_mapping: bool = False,
    tub_source_fixture: bool = False,
    state_dict_path: Path | None = None,
) -> dict[str, object]:
    blockers = []
    if not receiver_consumes_output2:
        blockers.append("snerv_official_tub_output2_receiver_frame_decode_not_bound")
    if tub_source_fixture:
        blockers.extend(
            [
                "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            ]
        )
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
    trained_mapping = bool(source_authority and full_tub_parity)
    state_dict_sha256 = (
        hashlib.sha256(state_dict_path.read_bytes()).hexdigest()
        if state_dict_path is not None and state_dict_path.is_file()
        else None
    )
    state_dict_section = {
        "official_trained_checkpoint_state_dict_path": (
            None if state_dict_path is None else state_dict_path.as_posix()
        ),
        "official_trained_checkpoint_state_dict_slice_path": (
            None if state_dict_path is None else state_dict_path.as_posix()
        ),
        "official_trained_checkpoint_state_dict_slice_present": state_dict_path
        is not None,
        "official_trained_checkpoint_state_dict_slice_file_present": (
            state_dict_path is not None and state_dict_path.is_file()
        ),
        "official_trained_checkpoint_state_dict_slice_bytes": (
            None
            if state_dict_path is None or not state_dict_path.is_file()
            else state_dict_path.stat().st_size
        ),
        "official_trained_checkpoint_state_dict_slice_sha256": state_dict_sha256,
        "official_trained_checkpoint_state_dict_slice_member_count": (
            None if state_dict_path is None else 3
        ),
        "official_trained_checkpoint_state_dict_slice_member_names": (
            []
            if state_dict_path is None
            else [
                "encoder.1.weight.npy",
                "encoder.2.weight.npy",
                "decoder.4.weight.npy",
            ]
        ),
        "official_trained_checkpoint_state_dict_slice_runner_arg": (
            None
            if state_dict_path is None
            else "--snerv-official-trained-checkpoint-state-dict-path"
        ),
    }
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
        "official_trained_checkpoint_loaded": native_hfr_mapping or trained_mapping,
        "official_hfr_trained_checkpoint_weight_mapping_proven": native_hfr_mapping
        or trained_mapping,
        "official_mfu_trained_checkpoint_weight_mapping_proven": trained_mapping,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": trained_mapping,
        "official_mfu_receiver_activation_payload_bound": native_hfr_mapping
        or trained_mapping,
        "official_tub_receiver_activation_payload_bound": trained_mapping,
        "official_native_receiver_state_mapping_proven": native_hfr_mapping
        or trained_mapping,
        "official_tub_temporal_encoder_weight_mapping_proven": trained_mapping,
        "official_tub_output2_decoder_weight_mapping_proven": trained_mapping,
        **state_dict_section,
        "official_tub_source_fixture_forward_parity_proven": tub_source_fixture,
        "tub_source_fixture_closed_blockers": (
            [
                "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
                "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
                "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing",
                "snerv_official_tub_frame_reconstruction_source_forward_replay_missing",
            ]
            if tub_source_fixture
            else []
        ),
        "official_tub_source_forward_replay": (
            {
                "schema": "snerv_official_tub_source_forward_replay.v1",
                "official_tub_temporal_encoder_output2_source_fixture_replay_passed": True,
                "closed_blockers": [
                    "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
                    "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
                ],
                "temporal_path": {
                    "closed_blockers": [
                        "snerv_official_tub_frame_reconstruction_source_forward_replay_missing"
                    ],
                },
                "portable_output2_fusion": {
                    "closed_blockers": [
                        "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing"
                    ],
                },
            }
            if tub_source_fixture
            else None
        ),
        "official_trained_checkpoint_mapping_manifest": (
            {
                "schema": "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1",
                "official_trained_checkpoint_loaded": True,
                "official_hfr_trained_checkpoint_weight_mapping_proven": True,
                "official_mfu_trained_checkpoint_weight_mapping_proven": trained_mapping,
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": trained_mapping,
                "official_mfu_receiver_activation_payload_bound": True,
                "official_native_receiver_state_mapping_proven": True,
                "official_tub_temporal_encoder_weight_mapping_proven": trained_mapping,
                "official_tub_output2_decoder_weight_mapping_proven": trained_mapping,
                "official_trained_checkpoint_state_dict_mapping_verified": (
                    trained_mapping
                ),
                **state_dict_section,
                "closed_campaign_blockers": [
                    "snerv_official_trained_checkpoint_state_dict_not_loaded",
                    "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete",
                    *(
                        [
                            "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping",
                            "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                            "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                            "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                            "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
                            "snerv_official_trained_checkpoint_state_dict_mapping_missing",
                        ]
                        if trained_mapping
                        else []
                    ),
                ],
                "blockers": (
                    []
                    if trained_mapping
                    else [
                        "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping",
                        "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                        "snerv_official_trained_checkpoint_source_forward_replay_missing",
                        "snerv_official_trained_checkpoint_state_dict_mapping_missing",
                    ]
                ),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
            if native_hfr_mapping or trained_mapping
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
    renderer_qat_ready: bool = False,
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
    row: dict[str, object] = {
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
    if renderer_qat_ready:
        row.update(
            {
                "measured_num_pairs": 16,
                "snerv_renderer_nondegenerate_proof_passed": True,
                "snerv_renderer_nondegenerate_blockers": [],
                "snerv_renderer_nondegenerate_proof": {
                    "schema": "snerv_renderer_nondegenerate_proof.v1",
                    "measured_num_pairs": 16,
                    "segnet_direct_live_max_candidate_occupied_class_fraction": 0.6,
                    "segnet_direct_live_max_candidate_target_class_coverage_fraction": 0.8,
                    "passed": True,
                    "blockers": [],
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                },
            }
        )
    return row


def _value_domain_xray(*, noncollapse: bool) -> dict[str, object]:
    return {
        "schema": "snerv_receiver_value_domain_xray.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_receiver_value_domain_xray.json",
        "_source_sha256": "e" * 64,
        "packet_path": "/ssd/candidate.snar",
        "packet_bytes": 1234,
        "packet_sha256": "f" * 64,
        "pair_indices": [0, 1],
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


def _hf_residual_receiver_payload_proof() -> dict[str, object]:
    return {
        "schema": "snerv_lf_conditioned_hf_residual_receiver_proof.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_lf_conditioned_hf_residual_receiver_proof.json",
        "_source_sha256": "0" * 64,
        "packet_path": "/ssd/candidate.snar",
        "source_packet_sha256": "1" * 64,
        "payload_path": "/ssd/lf_conditioned_hf_residual.slhr",
        "payload_bytes": 456,
        "payload_sha256": "2" * 64,
        "pair_indices": [0, 1],
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "section_native_byte_telemetry_present": True,
        "lf_anchor_bytes": 144,
        "hf_residual_bytes": 2304,
        "compressed_payload_bytes": 300,
        "closed_campaign_blockers": [
            "snerv_hf_residual_generator_receiver_payload_not_implemented"
        ],
        "blockers": ["snerv_lf_conditioned_hf_residual_payload_false_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _joint_codebook_receiver_payload_proof() -> dict[str, object]:
    return {
        "schema": "snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_joint_lf_hf_factorized_codebook_receiver_proof.json",
        "_source_sha256": "3" * 64,
        "packet_path": "/ssd/candidate.snar",
        "source_packet_sha256": "4" * 64,
        "payload_path": "/ssd/joint_lf_hf_factorized_codebook.sjlc",
        "payload_bytes": 512,
        "payload_sha256": "5" * 64,
        "pair_indices": [0, 1],
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "section_native_byte_telemetry_present": True,
        "codebook_raw_bytes": 384,
        "index_raw_bytes": 768,
        "compressed_payload_bytes": 320,
        "codebook_entry_count": 16,
        "block_count": 192,
        "closed_campaign_blockers": [
            "snerv_joint_lf_hf_factorized_codebook_not_implemented",
            "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
            "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
        ],
        "blockers": ["snerv_joint_lf_hf_factorized_codebook_false_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _temporal_lf_predictor_receiver_payload_proof() -> dict[str, object]:
    return {
        "schema": "snerv_temporal_lf_predictor_receiver_proof.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_temporal_lf_predictor_receiver_proof.json",
        "_source_sha256": "6" * 64,
        "packet_path": "/ssd/candidate.snar",
        "source_packet_sha256": "7" * 64,
        "payload_path": "/ssd/temporal_lf_predictor.stlp",
        "payload_bytes": 384,
        "payload_sha256": "8" * 64,
        "pair_indices": [0, 1],
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "lf_shape_b2chw": [1, 2, 3, 4, 6],
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "correction_stream_byte_charged": True,
        "section_native_byte_telemetry_present": True,
        "first_lf_anchor_bytes": 288,
        "correction_stream_raw_bytes": 144,
        "compressed_payload_bytes": 192,
        "closed_campaign_blockers": [
            "snerv_temporal_lf_predictor_gate_not_implemented",
            "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
        ],
        "blockers": ["snerv_temporal_lf_predictor_payload_false_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _lf_super_resolution_receiver_payload_proof() -> dict[str, object]:
    return {
        "schema": "snerv_lf_super_resolution_tiny_anchor_receiver_proof.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_lf_super_resolution_tiny_anchor_receiver_proof.json",
        "_source_sha256": "9" * 64,
        "packet_path": "/ssd/candidate.snar",
        "source_packet_sha256": "a" * 64,
        "payload_path": "/ssd/lf_super_resolution_tiny_anchor.slsr",
        "payload_bytes": 320,
        "payload_sha256": "b" * 64,
        "pair_indices": [0, 1],
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "anchor_shape_b2chw": [1, 2, 3, 2, 3],
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "tiny_anchor_component_deltas_present": True,
        "component_delta_scope": "receiver_pixel_domain_not_scorer_component",
        "section_native_byte_telemetry_present": True,
        "anchor_raw_bytes": 72,
        "compressed_payload_bytes": 128,
        "receiver_component_delta_stats": {
            "all_frames": {"count": 2304, "mean_abs": 4.0, "max_abs": 16.0},
            "scope": "receiver_pixel_domain_not_scorer_component",
        },
        "closed_campaign_blockers": [
            "snerv_lf_super_resolution_receiver_payload_not_implemented",
            "snerv_lf_downsampled_anchor_component_deltas_missing",
        ],
        "blockers": ["snerv_lf_super_resolution_tiny_anchor_payload_false_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _spectral_band_allocator_receiver_payload_proof() -> dict[str, object]:
    return {
        "schema": "snerv_score_tethered_spectral_band_allocator_receiver_proof.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_score_tethered_spectral_band_allocator_receiver_proof.json",
        "_source_sha256": "c" * 64,
        "packet_path": "/ssd/candidate.snar",
        "source_packet_sha256": "d" * 64,
        "payload_path": "/ssd/spectral_band_allocator.ssba",
        "payload_bytes": 128,
        "payload_sha256": "e" * 64,
        "pair_indices": [0, 1],
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "score_tethered_allocation_implemented": True,
        "section_native_byte_telemetry_present": True,
        "human_readable_payload_labels": False,
        "allocation_table_raw_bytes": 24,
        "allocation_band_count": 4,
        "allocation_budget_units": 255,
        "closed_campaign_blockers": [
            "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
            "snerv_mfu_hfr_section_native_byte_telemetry_missing",
        ],
        "blockers": ["snerv_score_tethered_spectral_band_allocator_false_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _lf_latent_hyperprior_receiver_payload_proof() -> dict[str, object]:
    return {
        "schema": "snerv_lf_latent_hyperprior_receiver_proof.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/ssd/snerv_lf_latent_hyperprior_receiver_proof.json",
        "_source_sha256": "f" * 64,
        "packet_path": "/ssd/candidate.snar",
        "source_packet_sha256": "0" * 64,
        "payload_path": "/ssd/lf_latent_hyperprior.slhp",
        "payload_bytes": 160,
        "payload_sha256": "1" * 64,
        "pair_indices": [0, 1],
        "sample_shape_b2chw": [1, 2, 3, 16, 24],
        "lf_shape_b2chw": [1, 2, 3, 4, 6],
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "entropy_model_implemented": True,
        "hyperprior_scale_present": True,
        "receiver_replay_proven": True,
        "section_native_byte_telemetry_present": True,
        "human_readable_payload_labels": False,
        "mean_raw_bytes": 12,
        "scale_raw_bytes": 12,
        "latent_symbol_raw_bytes": 288,
        "compressed_payload_bytes": 96,
        "estimated_entropy_bits": 128.0,
        "closed_campaign_blockers": [
            "snerv_lf_latent_hyperprior_not_implemented",
            "snerv_lf_latent_hyperprior_numpy_decoder_missing",
            "snerv_lf_latent_hyperprior_receiver_replay_missing",
        ],
        "blockers": ["snerv_lf_latent_hyperprior_payload_false_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _lf_hf_runtime_binding_proof(
    *families: str,
    source_path: str = "/ssd/snerv_lf_hf_runtime_binding_proof.json",
) -> dict[str, object]:
    blocker_by_family = {
        "lf_conditioned_hf_residual_generator": (
            "snerv_lf_conditioned_hf_residual_receiver_runtime_binding_missing"
        ),
        "joint_lf_hf_factorized_codebook": (
            "snerv_joint_lf_hf_factorized_codebook_receiver_runtime_binding_missing"
        ),
        "temporal_lf_predictor_gate": (
            "snerv_temporal_lf_predictor_receiver_runtime_binding_missing"
        ),
        "lf_super_resolution_from_tiny_anchor": (
            "snerv_lf_super_resolution_receiver_runtime_binding_missing"
        ),
        "score_tethered_spectral_band_allocator": (
            "snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing"
        ),
        "entropy_modeled_lf_latent_hyperprior": (
            "snerv_lf_latent_hyperprior_runtime_binding_missing"
        ),
    }
    selected = list(families) or list(blocker_by_family)
    return {
        "schema": "snerv_lf_hf_runtime_binding_proof.v1",
        "generated_utc": "2026-06-06T00:00:00+00:00",
        "_source_path": source_path,
        "_source_sha256": "2" * 64,
        "runtime_binding_row_count": len(selected),
        "runtime_bound_solution_families": selected,
        "closed_campaign_blockers": [
            blocker_by_family[family]
            for family in selected
            if family in blocker_by_family
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
