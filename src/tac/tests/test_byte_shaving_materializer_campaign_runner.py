# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
    load_queue_definition,
    normalize_queue_definition,
    ready_steps,
    run_ready_step,
    set_control_mode,
)
from tac.optimization.byte_shaving_campaign import (
    SIGNAL_SURFACE_SCHEMA,
    build_byte_shaving_campaign_plan,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_MANIFEST_NAME as INVERSE_CELL_CHAIN_MANIFEST_NAME,
)
from tac.optimization.inverse_steganalysis_acquisition import (
    observations_from_queue_performance_summary,
)
from tools import run_byte_shaving_materializer_campaign as runner


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _inverse_cell_candidate_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "inverse_cell_candidate_runner_fixture",
        "candidate_id": "fixture_seed",
        "lane_id": "lane_inverse_cell_candidate_runner_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "inverse_action_pair_0007",
                "unit_kind": "scorer_inverse_surface_cell",
                "candidate_saved_bytes": 0,
                "predicted_quality_score_delta": -0.0001,
                "confidence": 0.6,
                "operations": [
                    {
                        "operation_id": "materialize_inverse_surface_pair_0007",
                        "operation_family": "materialize_inverse_scorer_cell_candidate",
                        "target_kind": runner.INVERSE_SCORER_CELL_TARGET_KIND,
                    }
                ],
                "blockers": [
                    "inverse_action_unit_is_planning_only",
                    "requires_inverse_scorer_cell_materializer",
                    "requires_exact_auth_eval_before_score_claim",
                ],
            }
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=1)


def _packet_member_recompress_plan() -> dict[str, object]:
    surface = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "packet_member_recompress_runner_fixture",
        "candidate_id": "packet_fixture_seed",
        "lane_id": "lane_packet_member_recompress_runner_fixture",
        "combo_beam_width": 4,
        "max_combo_count": 4,
        "units": [
            {
                "unit_id": "packet_payload_member",
                "unit_kind": "packet_member",
                "candidate_saved_bytes": 1,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.8,
                "operations": [
                    {
                        "operation_id": "recompress_payload_member",
                        "operation_family": "member_recompress",
                        "target_kind": runner.PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
                    }
                ],
            }
        ],
        **_false_authority(),
    }
    return build_byte_shaving_campaign_plan(surface, max_k=1)


def _write_constant_inflate_runtime(path: Path) -> None:
    path.mkdir(parents=True)
    inflate = path / "inflate.sh"
    inflate.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'data_dir="$1"',
                'out_dir="$2"',
                'file_list="$3"',
                'test -f "$data_dir/x"',
                'test -s "$file_list"',
                'mkdir -p "$out_dir/frames"',
                'printf frame-bytes > "$out_dir/frames/000000.raw"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    inflate.chmod(0o755)


def _write_template_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("x", b"base-payload")


def _write_packet_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.bin", b"payload-bytes" * 16)


def _write_inverse_action_functional(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "water_bucket": {
                    "schema": "inverse_steganalysis_water_bucket_plan.v1",
                    "selected_count": 1,
                    "selected_cells": [
                        {
                            "atom_id": "inverse_surface_pair0007",
                            "candidate_id": "candidate_pair0007",
                            "scope_axis": "pairs",
                            "component": "posenet",
                            "water_fill_cost_bytes": 32,
                            "expected_score_gain": 0.0001,
                            "euler_lagrange_residual": 0.00009,
                        }
                    ],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_result_review_packet(
    path: Path,
    *,
    axis: str,
    technique: str = "auto_calibration_candidate",
    archive_sha256: str = "a" * 64,
    archive_bytes: int = 12345,
    n_samples: int = 600,
    runtime_content_tree_sha256: str = "b" * 64,
) -> None:
    is_cpu = axis == "contest_cpu"
    path.write_text(
        json.dumps(
            {
                "schema": runner.RESULT_REVIEW_PACKET_SCHEMA,
                "technique": technique,
                "score_axis": axis,
                "exact_cpu_evidence": is_cpu,
                "exact_cuda_evidence": not is_cpu,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "family_falsified": False,
                "method_family_retired": False,
                "baseline_score": 0.2,
                "canonical_score": 0.199,
                "measured_config_status": (
                    "contest_cpu_result_reviewed"
                    if is_cpu
                    else "exact_cuda_result_reviewed"
                ),
                "custody": {
                    "archive_sha256": archive_sha256,
                    "archive_bytes": archive_bytes,
                    "n_samples": n_samples,
                },
                "runtime_custody": {
                    "runtime_content_tree_sha256": runtime_content_tree_sha256,
                    "inflated_output_aggregate_sha256": ("c" if is_cpu else "d") * 64,
                },
                "score_recomputation": {
                    "available": True,
                    "matches_reported": True,
                },
                "engineering_forensic_audit": {
                    "engineering_or_config_bug_found": False,
                    "score_formula_reviewed": True,
                    "archive_runtime_closure_reviewed": True,
                },
                "dispatch_claim_state": {
                    "terminal_status_recorded": True,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_materializer_campaign_runner_builds_queue_owned_followup_command(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--queue-id",
            "materializer_campaign_fixture",
            "--materializer-resource-concurrency",
            "local_mlx=2",
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(tmp_path / "campaign" / "work"),
            "--storage-workload-subdir",
            "work",
            "--proactive-cleanup-root",
            "experiments/results",
            "--proactive-cleanup-cold-store-root",
            str(tmp_path / "cold_store"),
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert command[:2] == [
        runner.sys.executable,
        "tools/build_byte_shaving_campaign_queue.py",
    ]
    assert "--include-materializer-exact-readiness-followup" in command
    assert "--materializer-execution-queue-out" in command
    assert "--materializer-resource-concurrency" in command
    assert "local_mlx=2" in command
    assert "--include-materializer-scheduler-preflight" in command
    assert "--materializer-scheduler-proactive-cleanup-execute" in command
    assert "--materializer-scheduler-proactive-cleanup-cold-store-root" in command
    assert str(tmp_path / "cold_store") in command
    assert "--dispatch-mode" not in command
    assert "--allow-paid-dispatch-queue" not in command


def test_materializer_campaign_runner_can_auto_generate_contexts_from_artifact_map(
    tmp_path: Path,
) -> None:
    artifact_map = tmp_path / "artifact_map.json"
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-artifact-map",
            str(artifact_map),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--materializer-contexts-fail-if-blocked",
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert "--materializer-artifact-map" in command
    assert str(artifact_map) in command
    assert "--materializer-contexts-out" in command
    assert str(tmp_path / "campaign" / "materializer_contexts.json") in command
    assert "--materializer-context-default-output-root" in command
    assert str(tmp_path / "campaign" / "materializer_outputs") in command
    assert "--materializer-contexts-fail-if-blocked" in command
    assert "--materializer-contexts" not in command


def test_materializer_campaign_runner_places_generated_context_outputs_under_workload_root(
    tmp_path: Path,
) -> None:
    artifact_map = tmp_path / "artifact_map.json"
    run_dir = tmp_path / "campaign"
    expected_workload_root = run_dir / "work"
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-artifact-map",
            str(artifact_map),
            "--run-dir",
            str(run_dir),
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(expected_workload_root),
            "--storage-workload-subdir",
            "work",
        ]
    )

    command = runner._build_queue_command(args, run_dir=run_dir)

    assert "--materializer-context-default-output-root" in command
    index = command.index("--materializer-context-default-output-root")
    assert command[index + 1] == str(expected_workload_root / "materializer_outputs")


def test_materializer_campaign_runner_keeps_context_outputs_per_run_inside_workload_root(
    tmp_path: Path,
) -> None:
    artifact_map = tmp_path / "artifact_map.json"
    expected_workload_root = tmp_path / "work"
    run_dir = expected_workload_root / "campaign"
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-artifact-map",
            str(artifact_map),
            "--run-dir",
            str(run_dir),
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(expected_workload_root),
            "--storage-workload-subdir",
            "work",
        ]
    )

    command = runner._build_queue_command(args, run_dir=run_dir)

    index = command.index("--materializer-context-default-output-root")
    assert command[index + 1] == str(run_dir / "materializer_outputs")


def test_materializer_campaign_runner_builds_stateful_experiment_queue_command(
    tmp_path: Path,
) -> None:
    command = runner._experiment_queue_command(
        execution_queue=tmp_path / "queue.json",
        state_path=tmp_path / "queue.sqlite",
        subcommand=["validate"],
    )

    assert command == [
        runner.sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        str(tmp_path / "queue.json"),
        "--state",
        str(tmp_path / "queue.sqlite"),
        "validate",
    ]


def test_materializer_campaign_runner_performance_fallback_is_fail_closed() -> None:
    payload = runner._queue_performance_summary_payload(
        runner.CommandResult(
            command=["experiment_queue", "performance"],
            returncode=2,
            stdout="not json",
            stderr="boom",
            elapsed_seconds=0.25,
        ),
        queue={"queue_id": "fallback_queue"},
    )

    assert payload["schema"] == runner.UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert payload["queue_id"] == "fallback_queue"
    assert payload["performance_command_failed"] is True
    assert "queue_performance_command_failed" in payload["blockers"]
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_materializer_campaign_runner_builds_runtime_policy_command(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--apply-runtime-policy",
            "--runtime-policy-cpu-count",
            "8",
            "--runtime-policy-timeout-multiplier",
            "2",
            "--runtime-policy-min-timeout-seconds",
            "10",
            "--runtime-policy-max-timeout-seconds",
            "120",
        ]
    )
    run_dir = tmp_path / "campaign"

    policy_output, applied_output, command = runner._runtime_policy_command(
        args,
        execution_queue=run_dir / "materializer_execution_queue.json",
        state_path=run_dir / "queue.sqlite",
        run_dir=run_dir,
    )

    assert policy_output == run_dir / "scheduler_runtime_policy.json"
    assert applied_output == run_dir / "materializer_execution_queue.runtime_policy.json"
    assert command[:7] == [
        runner.sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        str(run_dir / "materializer_execution_queue.json"),
        "--state",
        str(run_dir / "queue.sqlite"),
        "runtime-policy",
    ]
    assert "--cpu-count" in command
    assert "8" in command
    assert "--timeout-multiplier" in command
    assert "2.0" in command
    assert "--policy-output" in command
    assert str(policy_output) in command
    assert "--applied-queue-output" in command
    assert str(applied_output) in command
    assert "--no-apply-timeouts" in command


def test_materializer_campaign_runner_can_generate_inverse_scorer_artifact_map(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    source_inflate = tmp_path / "source_inflate"
    candidate_inflate = tmp_path / "candidate_inflate"
    action.write_text("{}", encoding="utf-8")
    template.write_bytes(b"zip fixture")
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--inverse-scorer-action-functional",
            str(action),
            "--inverse-scorer-candidate-archive-template",
            str(template),
            "--inverse-scorer-raw-contest-video-digest",
            "f" * 64,
            "--inverse-scorer-atom-id",
            "inverse_surface_pair0007",
            "--inverse-scorer-selected-limit",
            "2",
            "--inverse-scorer-chain-output-dir",
            str(run_dir / "inverse_cell_chain"),
            "--inverse-scorer-source-inflate-output-dir",
            str(source_inflate),
            "--inverse-scorer-candidate-inflate-output-dir",
            str(candidate_inflate),
            "--inverse-scorer-fail-if-inflate-parity-blocked",
            "--run-dir",
            str(run_dir),
        ]
    )

    generated = runner._write_generated_materializer_artifact_map(
        args,
        run_dir=run_dir,
        generated_action_functional_path=None,
    )

    assert generated == run_dir / "materializer_artifact_map.json"
    payload = json.loads(generated.read_text(encoding="utf-8"))
    context = payload["artifacts"][runner.INVERSE_SCORER_CELL_TARGET_KIND]
    assert context["candidate_archive_template"] == str(template)
    assert context["inverse_action_functional"] == str(action)
    assert context["raw_contest_video_digest"] == "f" * 64
    assert context["atom_ids"] == ["inverse_surface_pair0007"]
    assert context["selected_limit"] == 2
    assert context["chain_output_dir"] == str(run_dir / "inverse_cell_chain")
    assert context["source_inflate_output_dir"] == str(source_inflate)
    assert context["candidate_inflate_output_dir"] == str(candidate_inflate)
    assert context["fail_if_inflate_parity_blocked"] is True
    assert context["score_claim"] is False

    command = runner._build_queue_command(
        args,
        run_dir=run_dir,
        plan_path=tmp_path / "plan.json",
        generated_materializer_artifact_map=generated,
    )

    assert "--materializer-artifact-map" in command
    assert str(generated) in command
    assert "--materializer-contexts-out" in command
    assert str(run_dir / "materializer_contexts.json") in command


def test_materializer_campaign_runner_generated_artifact_map_uses_generated_action(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    template = tmp_path / "template.zip"
    generated_action = run_dir / "inverse_steganalysis_action_functional.json"
    template.write_bytes(b"zip fixture")
    generated_action.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--scorer-response",
            str(tmp_path / "scorer_response.json"),
            "--inverse-scorer-candidate-archive-template",
            str(template),
            "--inverse-scorer-raw-contest-video-digest",
            "a" * 64,
            "--run-dir",
            str(run_dir),
        ]
    )

    generated = runner._write_generated_materializer_artifact_map(
        args,
        run_dir=run_dir,
        generated_action_functional_path=generated_action,
    )

    payload = json.loads(generated.read_text(encoding="utf-8"))
    context = payload["artifacts"][runner.INVERSE_SCORER_CELL_TARGET_KIND]
    assert context["inverse_action_functional"] == str(generated_action)
    assert context["candidate_archive_template"] == str(template)
    assert context["raw_contest_video_digest"] == "a" * 64


def test_materializer_campaign_runner_can_generate_family_artifact_map(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--archive-section-archive-path",
            str(tmp_path / "archive_source.zip"),
            "--archive-section-manifest",
            str(tmp_path / "sections.json"),
            "--archive-section-name",
            "decoder_packed_brotli",
            "--archive-section-brotli-quality",
            "11",
            "--archive-section-runtime-consumption-proof",
            str(tmp_path / "archive_runtime_proof.json"),
            "--archive-section-min-free-bytes",
            "1024",
            "--archive-section-allow-size-regression",
            "--packet-member-archive-path",
            str(tmp_path / "packet_source.zip"),
            "--packet-member-manifest",
            str(tmp_path / "members.json"),
            "--packet-member-name",
            "payload.bin",
            "--packet-member-zip-compression-method",
            "deflated",
            "--packet-member-zip-compresslevel",
            "9",
            "--packet-member-runtime-consumption-proof",
            str(tmp_path / "packet_runtime_proof.json"),
            "--packet-member-min-free-bytes",
            "2048",
            "--tensor-factorize-archive-path",
            str(tmp_path / "tensor_source.zip"),
            "--tensor-factorize-manifest",
            str(tmp_path / "tensor_manifest.json"),
            "--tensor-factorize-contract",
            str(tmp_path / "factor_contract.json"),
            "--tensor-factorize-rank",
            "2",
            "--tensor-factorize-runtime-consumption-proof",
            str(tmp_path / "tensor_runtime_proof.json"),
            "--tensor-factorize-allow-size-regression",
            "--run-dir",
            str(run_dir),
        ]
    )

    generated = runner._write_generated_materializer_artifact_map(
        args,
        run_dir=run_dir,
        generated_action_functional_path=None,
    )

    assert generated == run_dir / "materializer_artifact_map.json"
    payload = json.loads(generated.read_text(encoding="utf-8"))
    assert set(payload["artifacts"]) == {
        runner.ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        runner.PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        runner.TENSOR_FACTORIZE_TARGET_KIND,
    }
    archive = payload["artifacts"][runner.ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]
    packet = payload["artifacts"][runner.PACKET_MEMBER_RECOMPRESS_TARGET_KIND]
    tensor = payload["artifacts"][runner.TENSOR_FACTORIZE_TARGET_KIND]
    assert archive["archive_path"] == str(tmp_path / "archive_source.zip")
    assert archive["section_manifest"] == str(tmp_path / "sections.json")
    assert archive["target_sections"] == ["decoder_packed_brotli"]
    assert archive["brotli_quality"] == ["11"]
    assert archive["runtime_consumption_proof"] == str(tmp_path / "archive_runtime_proof.json")
    assert archive["min_free_bytes"] == 1024
    assert archive["allow_size_regression"] is True
    assert packet["archive_path"] == str(tmp_path / "packet_source.zip")
    assert packet["packet_member_manifest"] == str(tmp_path / "members.json")
    assert packet["member_name"] == "payload.bin"
    assert packet["zip_compression_method"] == ["deflated"]
    assert packet["zip_compresslevel"] == ["9"]
    assert packet["min_free_bytes"] == 2048
    assert tensor["archive_path"] == str(tmp_path / "tensor_source.zip")
    assert tensor["tensor_manifest"] == str(tmp_path / "tensor_manifest.json")
    assert tensor["factorization_contract"] == str(tmp_path / "factor_contract.json")
    assert tensor["rank"] == 2
    assert tensor["allow_size_regression"] is True
    assert payload["score_claim"] is False

    command = runner._build_queue_command(
        args,
        run_dir=run_dir,
        plan_path=tmp_path / "plan.json",
        generated_materializer_artifact_map=generated,
    )
    assert "--materializer-artifact-map" in command
    assert str(generated) in command
    assert "--materializer-contexts-out" in command


def test_materializer_campaign_runner_rejects_auto_artifact_map_with_contexts(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="auto artifact-map flags"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--materializer-contexts",
                str(tmp_path / "contexts.json"),
                "--inverse-scorer-candidate-archive-template",
                str(tmp_path / "template.zip"),
                "--inverse-scorer-raw-contest-video-digest",
                "f" * 64,
            ]
        )


def test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    state_path = tmp_path / "materializer_execution_queue.sqlite"
    plan = tmp_path / "plan.json"
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    inflate_runtime = tmp_path / "inflate_runtime"
    chain_output_dir = run_dir / "inverse_cell_chain"
    inflate_work_dir = tmp_path / "inflate_work"
    feedback_followup_state = tmp_path / "feedback_followup_execution.sqlite"

    plan.write_text(json.dumps(_inverse_cell_candidate_plan()), encoding="utf-8")
    _write_inverse_action_functional(action)
    _write_template_archive(template)
    _write_constant_inflate_runtime(inflate_runtime)

    result = runner.main(
        [
            "--plan",
            str(plan),
            "--inverse-scorer-action-functional",
            str(action),
            "--inverse-scorer-candidate-archive-template",
            str(template),
            "--inverse-scorer-raw-contest-video-digest",
            "f" * 64,
            "--inverse-scorer-atom-id",
            "inverse_surface_pair0007",
            "--inverse-scorer-selected-limit",
            "1",
            "--inverse-scorer-chain-output-dir",
            str(chain_output_dir),
            "--inverse-scorer-inflate-runtime-dir",
            str(inflate_runtime),
            "--inverse-scorer-source-archive-for-parity",
            str(template),
            "--inverse-scorer-inflate-timeout-seconds",
            "30",
            "--inverse-scorer-inflate-work-dir",
            str(inflate_work_dir),
            "--materializer-contexts-fail-if-blocked",
            "--run-dir",
            str(run_dir),
            "--queue-state",
            str(state_path),
            "--queue-state-rationale",
            "isolated no-paid runner e2e state for inverse scorer campaign smoke",
            "--queue-id",
            "inverse_scorer_runner_e2e_fixture",
            "--apply-runtime-policy",
            "--runtime-policy-cpu-count",
            "8",
            "--max-steps",
            "6",
            "--max-parallel",
            "2",
            "--idle-sleep-seconds",
            "0",
            "--max-idle-cycles",
            "1",
            "--execute",
            "--queue-feedback-replan-followup-policy-local-autopilot",
            "--queue-feedback-replan-followup-state",
            str(feedback_followup_state),
            "--queue-feedback-replan-followup-state-rationale",
            "isolated local feedback child queue state for materializer e2e",
            "--queue-feedback-replan-followup-max-steps",
            "1",
            "--queue-feedback-replan-followup-max-parallel",
            "1",
        ]
    )

    assert result == 0
    summary = json.loads((run_dir / "materializer_campaign_run.json").read_text(encoding="utf-8"))
    assert summary["schema"] == runner.RUN_SCHEMA
    assert summary["execute"] is True
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["state_path"] == str(state_path)
    assert summary["runtime_policy_path"] == str(run_dir / "scheduler_runtime_policy.json")
    assert summary["runtime_policy_applied_queue_path"] == str(
        run_dir / "materializer_execution_queue.runtime_policy.json"
    )
    assert summary["queue_path"] == str(run_dir / "materializer_execution_queue.runtime_policy.json")
    assert summary["queue_performance_summary_path"] == str(
        run_dir / "queue_performance_summary.json"
    )
    assert summary["queue_feedback_replan_request_path"] == str(
        run_dir / "queue_feedback_replan_request.json"
    )
    assert summary["queue_feedback_replan_followup_queue_path"] == str(
        run_dir / "queue_feedback_replan_followup_queue.json"
    )
    assert summary["queue_feedback_replan_followup_queue_emitted"] is True
    assert summary["queue_feedback_replan_followup_queue_blockers"] == []
    assert summary["queue_feedback_replan_followup_execution_policy"] == (
        "local_autopilot_policy"
    )
    assert summary["queue_feedback_replan_followup_policy_enabled"] is True
    assert summary["queue_feedback_replan_followup_policy_blockers"] == []
    assert summary["queue_feedback_replan_followup_execution_requested"] is True
    assert summary["queue_feedback_replan_followup_executed"] is True
    assert summary["queue_feedback_replan_followup_execution_success"] is True
    assert summary["queue_feedback_replan_followup_state_path"] == str(
        feedback_followup_state
    )
    assert summary["queue_feedback_replan_followup_action_functional_path"] == str(
        run_dir / "inverse_steganalysis_action_functional.feedback.json"
    )
    assert summary["queue_feedback_replan_policy_path"] == str(
        run_dir / "queue_feedback_replan_policy.json"
    )
    feedback_policy = json.loads(
        (run_dir / "queue_feedback_replan_policy.json").read_text(encoding="utf-8")
    )
    assert feedback_policy["schema"] == "queue_feedback_replan_policy.v1"
    assert feedback_policy["decision"] == "run_next_materializer_campaign_iteration"
    assert feedback_policy["should_continue_feedback_loop"] is True
    assert feedback_policy["feedback_followup_queue_validation"]["valid"] is True
    assert feedback_policy["feedback_followup_queue_validation"]["queue_sha256"]
    assert feedback_policy["score_claim"] is False
    assert feedback_policy["ready_for_exact_eval_dispatch"] is False
    assert summary["queue_feedback_replan_continuation_queue_path"] == str(
        run_dir / "queue_feedback_replan_continuation_queue.json"
    )
    assert summary["queue_feedback_replan_continuation_queue_emitted"] is True
    assert summary["queue_feedback_replan_continuation_queue_blockers"] == []
    continuation_queue = json.loads(
        (run_dir / "queue_feedback_replan_continuation_queue.json").read_text(
            encoding="utf-8"
        )
    )
    assert continuation_queue["schema"] == "experiment_queue.v1"
    assert continuation_queue["controls"]["mode"] == "paused"
    assert continuation_queue["controls"]["max_concurrency"] == {"local_cpu": 1}
    assert continuation_queue["experiments"][0]["lane_id"] == (
        "lane_inverse_cell_candidate_runner_fixture"
    )
    continuation_step = continuation_queue["experiments"][0]["steps"][0]
    assert continuation_step["resources"] == {"kind": "local_cpu"}
    assert continuation_step["command"] == feedback_policy["next_iteration_command_template"]
    continuation_metadata = continuation_queue["experiments"][0]["metadata"]
    assert continuation_metadata["schema"] == "queue_feedback_replan_continuation_metadata.v1"
    assert continuation_metadata["score_claim"] is False
    assert continuation_metadata["ready_for_exact_eval_dispatch"] is False
    assert summary["queue_feedback_replan_continuation_staircase_artifacts"][
        "dependent_queue_ref_count"
    ] == 1
    feedback_execution = summary["queue_feedback_replan_followup_execution"]
    assert feedback_execution["schema"] == (
        runner.QUEUE_FEEDBACK_REPLAN_FOLLOWUP_EXECUTION_SCHEMA
    )
    assert feedback_execution["success"] is True
    assert feedback_execution["blockers"] == []
    assert feedback_execution["activation_policy"] == "local_autopilot_policy"
    assert feedback_execution["score_claim"] is False
    assert feedback_execution["ready_for_exact_eval_dispatch"] is False
    assert feedback_execution["worker"]["schema"] == "experiment_queue_worker_result.v1"
    assert feedback_execution["worker"]["success_count"] == 1
    assert feedback_execution["worker"]["failure_count"] == 0
    feedback_staircase = summary["queue_feedback_replan_staircase_artifacts"]
    assert feedback_staircase["dependent_queue_ref_count"] == 1
    assert feedback_staircase["child_control_mode"] == "paused"
    assert feedback_staircase["child_selected_count"] == 0
    assert feedback_staircase["child_blocked_count"] == 1
    assert feedback_staircase["score_claim"] is False
    assert feedback_staircase["ready_for_exact_eval_dispatch"] is False
    assert summary["queue_performance_runtime_identity_path"] == str(
        run_dir / "queue_performance_runtime_identity.json"
    )
    assert summary["queue_performance_cache_identity_path"] == str(
        run_dir / "queue_performance_cache_identity.json"
    )
    assert summary["response_update_placeholder_path"] == str(
        run_dir / "canonical_response_update_placeholder.json"
    )
    assert summary["response_update_applied"] is False
    assert summary["replan_required"] is True
    assert summary["queue_feedback_replan_ready"] is True
    assert summary["queue_feedback_replan_blockers"] == []
    assert summary["runtime_policy"]["schema"] == "scheduler_runtime_policy.v1"
    assert summary["runtime_policy"]["score_claim"] is False
    assert summary["performance"]["schema"] == runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert summary["performance"]["score_claim"] is False
    assert summary["performance"]["producer"] == "tools/run_byte_shaving_materializer_campaign.py"
    assert summary["performance"]["source_run_path"] == str(run_dir / "materializer_campaign_run.json")
    assert summary["performance"]["worker_returncode"] == 0
    assert summary["worker"]["schema"] == "experiment_queue_worker_result.v1"
    assert summary["worker"]["success_count"] == 3
    assert summary["worker"]["failure_count"] == 0

    performance = json.loads(
        (run_dir / "queue_performance_summary.json").read_text(encoding="utf-8")
    )
    placeholder = json.loads(
        (run_dir / "canonical_response_update_placeholder.json").read_text(
            encoding="utf-8"
        )
    )
    replan_request = json.loads(
        (run_dir / "queue_feedback_replan_request.json").read_text(encoding="utf-8")
    )
    feedback_queue = json.loads(
        (run_dir / "queue_feedback_replan_followup_queue.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_identity = json.loads(
        (run_dir / "queue_performance_runtime_identity.json").read_text(
            encoding="utf-8"
        )
    )
    cache_identity = json.loads(
        (run_dir / "queue_performance_cache_identity.json").read_text(encoding="utf-8")
    )
    assert performance == summary["performance"]
    assert performance["schema"] == runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert performance["queue_id"] == "inverse_scorer_runner_e2e_fixture"
    assert performance["score_claim"] is False
    assert replan_request["schema"] == runner.QUEUE_FEEDBACK_REPLAN_REQUEST_SCHEMA
    assert replan_request["queue_performance_summary_path"] == str(
        run_dir / "queue_performance_summary.json"
    )
    assert replan_request["queue_performance_runtime_identity_path"] == str(
        run_dir / "queue_performance_runtime_identity.json"
    )
    assert replan_request["queue_performance_cache_identity_path"] == str(
        run_dir / "queue_performance_cache_identity.json"
    )
    assert replan_request["queue_performance_runtime_identity_generated"] is True
    assert replan_request["queue_performance_cache_identity_generated"] is True
    assert replan_request["ready_for_action_functional_feedback"] is True
    assert replan_request["score_claim"] is False
    assert replan_request["blockers"] == []
    assert replan_request["queue_owned_followup_queue_emitted"] is True
    assert replan_request["queue_owned_followup_queue_path"] == str(
        run_dir / "queue_feedback_replan_followup_queue.json"
    )
    assert replan_request["queue_owned_followup_queue_blockers"] == []
    assert replan_request["suggested_action_functional_command"] == replan_request[
        "command_template"
    ]
    assert replan_request["command_template"][1] == "tools/build_inverse_steganalysis_action_functional.py"
    assert "--queue-performance-summary" in replan_request["command_template"]
    assert "--queue-performance-runtime-identity" in replan_request["command_template"]
    assert "--queue-performance-cache-identity" in replan_request["command_template"]
    assert feedback_queue["schema"] == "experiment_queue.v1"
    assert feedback_queue["controls"]["mode"] == "paused"
    assert feedback_queue["controls"]["max_concurrency"] == {"local_cpu": 1}
    feedback_child_dag = json.loads(
        (run_dir / "queue_feedback_replan_staircase_dag.json").read_text(
            encoding="utf-8"
        )
    )
    feedback_child_plan = json.loads(
        (run_dir / "queue_feedback_replan_staircase_dispatch_plan.json").read_text(
            encoding="utf-8"
        )
    )
    feedback_dependent_refs = json.loads(
        (run_dir / "queue_feedback_replan_dependent_queue_refs.json").read_text(
            encoding="utf-8"
        )
    )
    assert feedback_child_dag["controls"]["mode"] == "paused"
    assert feedback_child_plan["selected_count"] == 0
    assert feedback_child_plan["blocked_nodes"][0]["reason"] == "queue_control_not_running"
    dependent_ref = feedback_dependent_refs["refs"][0]
    assert dependent_ref["schema"] == "staircase_dependent_queue_ref.v1"
    assert dependent_ref["relationship"] == "feedback_replan_child"
    assert dependent_ref["child_queue_id"] == feedback_queue["queue_id"]
    assert dependent_ref["child_queue_path"] == str(
        run_dir / "queue_feedback_replan_followup_queue.json"
    )
    assert dependent_ref["child_controls"]["mode"] == "paused"
    assert dependent_ref["child_controls"]["max_concurrency"] == {"local_cpu": 1}
    assert dependent_ref["score_claim"] is False
    assert dependent_ref["ready_for_exact_eval_dispatch"] is False
    feedback_experiment = feedback_queue["experiments"][0]
    assert feedback_experiment["metadata"]["schema"] == (
        runner.QUEUE_FEEDBACK_REPLAN_EXPERIMENT_METADATA_SCHEMA
    )
    assert feedback_experiment["metadata"]["score_claim"] is False
    assert feedback_experiment["metadata"]["promotion_eligible"] is False
    assert feedback_experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    feedback_step = feedback_experiment["steps"][0]
    assert feedback_step["id"] == "build_feedback_action_functional"
    assert feedback_step["command"] == replan_request["command_template"]
    assert feedback_step["resources"]["kind"] == "local_cpu"
    assert feedback_step["postconditions"][1]["type"] == "json_completion_contract"
    assert "exact_cuda_auth_eval" in feedback_step["postconditions"][1]["false_or_missing"]
    assert "contest_cuda_auth_eval" in feedback_step["postconditions"][1]["false_or_missing"]
    assert "gpu_launched" in feedback_step["postconditions"][1]["false_or_missing"]
    feedback_action = json.loads(
        (
            run_dir / "inverse_steganalysis_action_functional.feedback.json"
        ).read_text(encoding="utf-8")
    )
    assert feedback_action["schema"] == "inverse_steganalysis_discrete_action_functional.v1"
    assert feedback_action["score_claim"] is False
    assert feedback_action["ready_for_exact_eval_dispatch"] is False
    assert load_queue_definition(
        run_dir / "queue_feedback_replan_followup_queue.json"
    ) == feedback_queue
    with connect_state(tmp_path / "feedback_followup_manual.sqlite") as conn:
        initialize_queue_state(conn, feedback_queue)
        assert ready_steps(conn, feedback_queue) == []
        set_control_mode(conn, feedback_queue["queue_id"], "running", reason="unit resume")
        ready = ready_steps(conn, feedback_queue)
    assert [step.step_id for step in ready] == ["build_feedback_action_functional"]
    assert runtime_identity["schema"] == (
        "byte_shaving_materializer_campaign_queue_runtime_identity.v1"
    )
    assert runtime_identity["score_claim"] is False
    assert runtime_identity["queue_id"] == "inverse_scorer_runner_e2e_fixture"
    assert cache_identity["schema"] == (
        "byte_shaving_materializer_campaign_queue_cache_identity.v1"
    )
    assert cache_identity["score_claim"] is False
    assert cache_identity["queue_performance_summary_path"] == str(
        run_dir / "queue_performance_summary.json"
    )
    assert placeholder["schema"] == runner.RESPONSE_UPDATE_PLACEHOLDER_SCHEMA
    assert placeholder["queue_performance_summary_path"] == str(
        run_dir / "queue_performance_summary.json"
    )
    assert placeholder["queue_feedback_replan_request_path"] == str(
        run_dir / "queue_feedback_replan_request.json"
    )
    assert placeholder["queue_feedback_replan_followup_queue_path"] == str(
        run_dir / "queue_feedback_replan_followup_queue.json"
    )
    assert placeholder["queue_feedback_replan_followup_queue_emitted"] is True
    assert placeholder["queue_feedback_replan_followup_queue_blockers"] == []
    assert placeholder["next_run_hint"] == [
        "--queue-performance-summary",
        str(run_dir / "queue_performance_summary.json"),
    ]
    assert placeholder["response_update_applied"] is False
    assert placeholder["replan_required"] is True
    assert "placeholder_not_scorer_response_dataset" in placeholder["blockers"]
    assert placeholder["score_claim"] is False
    observations = observations_from_queue_performance_summary(
        performance,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path=summary["queue_performance_summary_path"],
    )
    assert observations
    assert all(observation["score_claim"] is False for observation in observations)

    contexts = json.loads((run_dir / "materializer_contexts.json").read_text(encoding="utf-8"))
    assert summary["build"]["materializer_contexts_blocked_count"] == 0
    assert contexts["blocked_context_count"] == 0
    context = contexts["rows"][0]["context"]
    assert context["chain_output_dir"] == str(chain_output_dir)
    assert context["inflate_runtime_dir"] == str(inflate_runtime)
    assert context["source_archive_for_parity"] == str(template)

    chain = json.loads((chain_output_dir / INVERSE_CELL_CHAIN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert chain["schema"] == "inverse_scorer_cell_candidate_chain_v1"
    assert chain["byte_closed_candidate_emitted"] is True
    assert chain["receiver_contract_satisfied"] is True
    assert chain["inflate_parity_satisfied"] is True
    assert chain["score_claim"] is False
    assert chain["ready_for_exact_eval_dispatch"] is False
    assert "exact_auth_eval_required_before_score_claim" in chain["readiness_blockers"]

    handoff_dir = chain_output_dir / "exact_eval_handoff"
    harvest = json.loads((handoff_dir / "harvest_report.json").read_text(encoding="utf-8"))
    bridge = json.loads((handoff_dir / "exact_readiness_bridge_report.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((handoff_dir / "dispatch_plan.json").read_text(encoding="utf-8"))
    handoff_paths = placeholder["exact_readiness_handoff_paths"]
    assert placeholder["exact_readiness_handoff_count"] == 1
    assert handoff_paths[0]["handoff_dir"] == str(handoff_dir)
    assert handoff_paths[0]["harvest_report_exists"] is True
    assert handoff_paths[0]["exact_readiness_bridge_report_exists"] is True
    assert handoff_paths[0]["dispatch_plan_exists"] is True
    assert handoff_paths[0]["readiness_dir_path"] == str(
        handoff_dir / "exact_readiness"
    )
    assert handoff_paths[0]["readiness_dir_exists"] is True
    assert harvest["accepted_manifest_count"] == 1
    assert harvest["source_queue_dispatch_ready_count"] == 0
    assert bridge["candidate_count"] == 1
    assert bridge["ready_candidate_count"] == 0
    assert bridge["score_claim"] is False
    assert dispatch_plan["authorized_candidate_count"] == 0
    assert dispatch_plan["score_claim"] is False
    assert not inflate_work_dir.exists()


def test_materializer_campaign_feedback_replan_preserves_calibration_inputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    plan = tmp_path / "plan.json"
    queue = tmp_path / "materializer_execution_queue.json"
    state = tmp_path / "materializer_execution_queue.sqlite"
    performance = run_dir / "queue_performance_summary.json"
    runtime_identity = run_dir / "queue_performance_runtime_identity.json"
    cache_identity = run_dir / "queue_performance_cache_identity.json"
    observation = tmp_path / "local_observation.json"
    cpu_packet = tmp_path / "cpu_review_packet.json"
    cuda_packet = tmp_path / "cuda_review_packet.json"
    request_path = run_dir / "queue_feedback_replan_request.json"

    for path in (
        plan,
        queue,
        state,
        performance,
        runtime_identity,
        cache_identity,
        observation,
    ):
        path.write_text("{}", encoding="utf-8")
    _write_result_review_packet(cpu_packet, axis="contest_cpu")
    _write_result_review_packet(cuda_packet, axis="contest_cuda")

    args = runner.parse_args(
        [
            "--plan",
            str(plan),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--observation",
            str(observation),
            "--exact-auth-calibration-packet",
            str(cpu_packet),
            "--exact-auth-calibration-packet",
            str(cuda_packet),
            "--exact-auth-calibration-candidate-id",
            "materialized_candidate_0001",
        ]
    )
    performance_payload = {
        "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
        "event_count": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    request = runner._queue_feedback_replan_request_payload(
        args,
        summary_path=run_dir / "materializer_campaign_run.json",
        plan_path=plan,
        queue_performance_summary_path=performance,
        queue_performance_summary=performance_payload,
        runtime_identity_path=runtime_identity,
        cache_identity_path=cache_identity,
        generated_runtime_identity=False,
        generated_cache_identity=False,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
    )

    command = request["command_template"]
    assert request["ready_for_action_functional_feedback"] is True
    assert request["feedback_observation_paths"] == [str(observation)]
    assert request["exact_auth_calibration_packet_paths"] == [
        str(cpu_packet),
        str(cuda_packet),
    ]
    assert request["exact_auth_calibration_candidate_id"] == (
        "materialized_candidate_0001"
    )
    assert request["exact_auth_calibration_discovery_pair"] == {
        "archive_sha256": "a" * 64,
        "archive_bytes": 12345,
        "n_samples": 600,
        "runtime_content_tree_sha256": "b" * 64,
        "contest_cpu_packet_path": str(cpu_packet),
        "contest_cuda_packet_path": str(cuda_packet),
    }
    assert [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--observation"
    ] == [str(observation)]
    assert [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--exact-auth-calibration-packet"
    ] == [str(cpu_packet), str(cuda_packet)]
    assert "--exact-auth-calibration-candidate-id" in command
    assert "materialized_candidate_0001" in command

    child_queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request=request,
        queue_feedback_replan_request_path=request_path,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
        source_queue={"queue_id": "materializer_unit_queue"},
    )

    assert blockers == []
    assert child_queue is not None
    step = child_queue["experiments"][0]["steps"][0]
    assert step["command"] == command
    input_paths = step["telemetry"]["input_artifact_paths"]
    assert str(observation) in input_paths
    assert str(cpu_packet) in input_paths
    assert str(cuda_packet) in input_paths
    metadata = child_queue["experiments"][0]["metadata"]
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False


def test_materializer_campaign_feedback_replan_discovers_calibration_packet_pair(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    packet_root = tmp_path / "review_packets"
    packet_root.mkdir()
    plan = tmp_path / "plan.json"
    queue = tmp_path / "materializer_execution_queue.json"
    state = tmp_path / "materializer_execution_queue.sqlite"
    performance = run_dir / "queue_performance_summary.json"
    runtime_identity = run_dir / "queue_performance_runtime_identity.json"
    cache_identity = run_dir / "queue_performance_cache_identity.json"
    cpu_packet = packet_root / "candidate_contest_cpu_result_review.json"
    cuda_packet = packet_root / "candidate_contest_cuda_result_review.json"

    for path in (plan, queue, state, performance, runtime_identity, cache_identity):
        path.write_text("{}", encoding="utf-8")
    _write_result_review_packet(cpu_packet, axis="contest_cpu")
    _write_result_review_packet(cuda_packet, axis="contest_cuda")

    args = runner.parse_args(
        [
            "--plan",
            str(plan),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--exact-auth-calibration-packet-root",
            str(packet_root),
        ]
    )
    performance_payload = {
        "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
        "event_count": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    request = runner._queue_feedback_replan_request_payload(
        args,
        summary_path=run_dir / "materializer_campaign_run.json",
        plan_path=plan,
        queue_performance_summary_path=performance,
        queue_performance_summary=performance_payload,
        runtime_identity_path=runtime_identity,
        cache_identity_path=cache_identity,
        generated_runtime_identity=False,
        generated_cache_identity=False,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
    )

    command = request["command_template"]
    assert request["ready_for_action_functional_feedback"] is True
    assert request["blockers"] == []
    assert request["exact_auth_calibration_packet_paths"] == [
        str(cpu_packet),
        str(cuda_packet),
    ]
    assert request["exact_auth_calibration_candidate_id"] == "auto_calibration_candidate"
    assert request["exact_auth_calibration_packet_source"] == "auto_discovery"
    assert request["exact_auth_calibration_discovery_roots"] == [str(packet_root)]
    assert request["exact_auth_calibration_discovery_pair"] == {
        "archive_sha256": "a" * 64,
        "archive_bytes": 12345,
        "n_samples": 600,
        "runtime_content_tree_sha256": "b" * 64,
        "contest_cpu_packet_path": str(cpu_packet),
        "contest_cuda_packet_path": str(cuda_packet),
    }
    assert [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--exact-auth-calibration-packet"
    ] == [str(cpu_packet), str(cuda_packet)]
    assert "--exact-auth-calibration-candidate-id" in command
    assert "auto_calibration_candidate" in command

    child_queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request=request,
        queue_feedback_replan_request_path=run_dir / "queue_feedback_replan_request.json",
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
        source_queue={"queue_id": "materializer_unit_queue"},
    )

    assert blockers == []
    assert child_queue is not None
    input_paths = child_queue["experiments"][0]["steps"][0]["telemetry"][
        "input_artifact_paths"
    ]
    assert str(cpu_packet) in input_paths
    assert str(cuda_packet) in input_paths


def test_materializer_campaign_feedback_replan_discovers_run_derived_packet_pair(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    packet_root = run_dir / "exact_eval_handoff" / "result_reviews"
    packet_root.mkdir(parents=True)
    plan = tmp_path / "plan.json"
    queue = tmp_path / "materializer_execution_queue.json"
    state = tmp_path / "materializer_execution_queue.sqlite"
    performance = run_dir / "queue_performance_summary.json"
    runtime_identity = run_dir / "queue_performance_runtime_identity.json"
    cache_identity = run_dir / "queue_performance_cache_identity.json"
    cpu_packet = packet_root / "candidate_contest_cpu_result_review.json"
    cuda_packet = packet_root / "candidate_contest_cuda_result_review.json"

    for path in (plan, queue, state, performance, runtime_identity, cache_identity):
        path.write_text("{}", encoding="utf-8")
    _write_result_review_packet(cpu_packet, axis="contest_cpu")
    _write_result_review_packet(cuda_packet, axis="contest_cuda")

    args = runner.parse_args(
        [
            "--plan",
            str(plan),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
        ]
    )
    performance_payload = {
        "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
        "event_count": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    request = runner._queue_feedback_replan_request_payload(
        args,
        summary_path=run_dir / "materializer_campaign_run.json",
        plan_path=plan,
        queue_performance_summary_path=performance,
        queue_performance_summary=performance_payload,
        runtime_identity_path=runtime_identity,
        cache_identity_path=cache_identity,
        generated_runtime_identity=False,
        generated_cache_identity=False,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
    )

    command = request["command_template"]
    assert request["ready_for_action_functional_feedback"] is True
    assert request["blockers"] == []
    assert request["exact_auth_calibration_packet_paths"] == [
        str(cpu_packet),
        str(cuda_packet),
    ]
    assert request["exact_auth_calibration_packet_source"] == "run_derived_discovery"
    assert str(run_dir) in request["exact_auth_calibration_discovery_roots"]
    assert str(run_dir / "exact_eval_handoff") in request[
        "exact_auth_calibration_discovery_roots"
    ]
    assert [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--exact-auth-calibration-packet"
    ] == [str(cpu_packet), str(cuda_packet)]


def test_materializer_campaign_feedback_replan_treats_empty_run_root_as_optional(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    plan = tmp_path / "plan.json"
    queue = tmp_path / "materializer_execution_queue.json"
    state = tmp_path / "materializer_execution_queue.sqlite"
    performance = run_dir / "queue_performance_summary.json"
    runtime_identity = run_dir / "queue_performance_runtime_identity.json"
    cache_identity = run_dir / "queue_performance_cache_identity.json"

    for path in (plan, queue, state, performance, runtime_identity, cache_identity):
        path.write_text("{}", encoding="utf-8")

    args = runner.parse_args(
        [
            "--plan",
            str(plan),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
        ]
    )
    performance_payload = {
        "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
        "event_count": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    request = runner._queue_feedback_replan_request_payload(
        args,
        summary_path=run_dir / "materializer_campaign_run.json",
        plan_path=plan,
        queue_performance_summary_path=performance,
        queue_performance_summary=performance_payload,
        runtime_identity_path=runtime_identity,
        cache_identity_path=cache_identity,
        generated_runtime_identity=False,
        generated_cache_identity=False,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
    )

    assert request["ready_for_action_functional_feedback"] is True
    assert request["exact_auth_calibration_packet_paths"] == []
    assert request["exact_auth_calibration_packet_source"] == "run_derived_discovery"
    assert request["exact_auth_calibration_discovery_roots"] == [str(run_dir)]
    assert request["blockers"] == []


def test_materializer_campaign_feedback_replan_blocks_on_missing_calibration_pair(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    packet_root = tmp_path / "empty_review_packets"
    packet_root.mkdir()
    plan = tmp_path / "plan.json"
    queue = tmp_path / "materializer_execution_queue.json"
    state = tmp_path / "materializer_execution_queue.sqlite"
    performance = run_dir / "queue_performance_summary.json"
    runtime_identity = run_dir / "queue_performance_runtime_identity.json"
    cache_identity = run_dir / "queue_performance_cache_identity.json"

    for path in (plan, queue, state, performance, runtime_identity, cache_identity):
        path.write_text("{}", encoding="utf-8")

    args = runner.parse_args(
        [
            "--plan",
            str(plan),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--exact-auth-calibration-packet-root",
            str(packet_root),
        ]
    )
    performance_payload = {
        "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
        "event_count": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    request = runner._queue_feedback_replan_request_payload(
        args,
        summary_path=run_dir / "materializer_campaign_run.json",
        plan_path=plan,
        queue_performance_summary_path=performance,
        queue_performance_summary=performance_payload,
        runtime_identity_path=runtime_identity,
        cache_identity_path=cache_identity,
        generated_runtime_identity=False,
        generated_cache_identity=False,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
    )

    assert request["ready_for_action_functional_feedback"] is False
    assert request["exact_auth_calibration_packet_paths"] == []
    assert request["exact_auth_calibration_packet_source"] == "auto_discovery"
    assert request["exact_auth_calibration_discovery_roots"] == [str(packet_root)]
    assert request["suggested_action_functional_command"] is None
    assert "exact_auth_calibration_packet_pair_not_found" in request["blockers"]


def test_materializer_campaign_feedback_replan_blocks_on_invalid_calibration_pair(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    packet_root = tmp_path / "invalid_review_packets"
    packet_root.mkdir()
    plan = tmp_path / "plan.json"
    queue = tmp_path / "materializer_execution_queue.json"
    state = tmp_path / "materializer_execution_queue.sqlite"
    performance = run_dir / "queue_performance_summary.json"
    runtime_identity = run_dir / "queue_performance_runtime_identity.json"
    cache_identity = run_dir / "queue_performance_cache_identity.json"
    cpu_packet = packet_root / "candidate_contest_cpu_result_review.json"
    cuda_packet = packet_root / "candidate_contest_cuda_result_review.json"

    for path in (plan, queue, state, performance, runtime_identity, cache_identity):
        path.write_text("{}", encoding="utf-8")
    _write_result_review_packet(cpu_packet, axis="contest_cpu")
    _write_result_review_packet(cuda_packet, axis="contest_cuda")
    invalid_cpu_payload = json.loads(cpu_packet.read_text(encoding="utf-8"))
    invalid_cpu_payload.pop("baseline_score")
    cpu_packet.write_text(json.dumps(invalid_cpu_payload), encoding="utf-8")

    args = runner.parse_args(
        [
            "--plan",
            str(plan),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--exact-auth-calibration-packet-root",
            str(packet_root),
        ]
    )
    performance_payload = {
        "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
        "event_count": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    request = runner._queue_feedback_replan_request_payload(
        args,
        summary_path=run_dir / "materializer_campaign_run.json",
        plan_path=plan,
        queue_performance_summary_path=performance,
        queue_performance_summary=performance_payload,
        runtime_identity_path=runtime_identity,
        cache_identity_path=cache_identity,
        generated_runtime_identity=False,
        generated_cache_identity=False,
        run_dir=run_dir,
        execution_queue=queue,
        state_path=state,
    )

    assert request["ready_for_action_functional_feedback"] is False
    assert request["exact_auth_calibration_packet_paths"] == []
    assert request["suggested_action_functional_command"] is None
    assert "exact_auth_calibration_packet_pair_not_found" in request["blockers"]
    assert any(
        blocker.startswith("exact_auth_calibration_packet_pair_invalid:")
        and "baseline_score is required" in blocker
        for blocker in request["blockers"]
    )


def test_materializer_campaign_runner_poison_summary_when_performance_stdout_invalid() -> None:
    result = runner.CommandResult(
        command=["experiment_queue", "performance"],
        returncode=0,
        stdout="not-json",
        stderr="",
        elapsed_seconds=0.1,
    )

    payload = runner._queue_performance_summary_payload(
        result,
        queue={"queue_id": "queue_a"},
    )

    assert payload["schema"] == runner.UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert payload["queue_id"] == "queue_a"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["blockers"] == ["queue_performance_stdout_not_json_object"]


def test_materializer_campaign_runner_poison_summary_when_performance_command_fails() -> None:
    result = runner.CommandResult(
        command=["experiment_queue", "performance"],
        returncode=7,
        stdout="",
        stderr="boom",
        elapsed_seconds=0.2,
    )

    payload = runner._queue_performance_summary_payload(
        result,
        queue={"queue_id": "queue_a"},
    )

    assert payload["schema"] == runner.UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert payload["performance_command_failed"] is True
    assert payload["blockers"] == ["queue_performance_command_failed"]
    assert payload["score_claim"] is False


def test_materializer_campaign_runner_poison_summary_with_nested_authority() -> None:
    result = runner.CommandResult(
        command=["experiment_queue", "performance"],
        returncode=0,
        stdout=json.dumps(
            {
                "schema": runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
                "queue_id": "queue_a",
                "telemetry_only": True,
                "event_count": 1,
                "candidate_id_by_experiment": {"exp": ["candidate"]},
                "by_step": {
                    "exp.step": {
                        "run_count": 1,
                        "success_count": 1,
                        "score_claim": True,
                    }
                },
            }
        ),
        stderr="",
        elapsed_seconds=0.2,
    )

    payload = runner._queue_performance_summary_payload(
        result,
        queue={"queue_id": "queue_a"},
    )

    assert payload["schema"] == runner.UNAVAILABLE_QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert payload["blockers"] == [
        "queue_performance_summary_truthy_authority_fields"
    ]
    assert payload["authority_violations"] == ["by_step.exp.step.score_claim=truthy"]
    assert payload["score_claim"] is False


def test_materializer_campaign_runner_executes_no_paid_packet_member_handoff(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    state_path = tmp_path / "materializer_execution_queue.sqlite"
    plan = tmp_path / "plan.json"
    source_archive = tmp_path / "packet_source.zip"

    plan.write_text(json.dumps(_packet_member_recompress_plan()), encoding="utf-8")
    _write_packet_archive(source_archive)

    result = runner.main(
        [
            "--plan",
            str(plan),
            "--packet-member-archive-path",
            str(source_archive),
            "--packet-member-name",
            "payload.bin",
            "--packet-member-zip-compression-method",
            "stored",
            "--packet-member-allow-size-regression",
            "--materializer-contexts-fail-if-blocked",
            "--run-dir",
            str(run_dir),
            "--queue-state",
            str(state_path),
            "--queue-state-rationale",
            "isolated no-paid runner e2e state for packet member materializer smoke",
            "--queue-id",
            "packet_member_runner_e2e_fixture",
            "--apply-runtime-policy",
            "--runtime-policy-cpu-count",
            "8",
            "--max-steps",
            "6",
            "--max-parallel",
            "2",
            "--idle-sleep-seconds",
            "0",
            "--max-idle-cycles",
            "1",
            "--execute",
        ]
    )

    assert result == 0
    summary = json.loads((run_dir / "materializer_campaign_run.json").read_text(encoding="utf-8"))
    assert summary["schema"] == runner.RUN_SCHEMA
    assert summary["execute"] is True
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert (run_dir / "queue_performance_summary.json").exists()
    assert (run_dir / "queue_feedback_replan_request.json").exists()
    assert (run_dir / "queue_performance_runtime_identity.json").exists()
    assert (run_dir / "queue_performance_cache_identity.json").exists()
    assert (run_dir / "canonical_response_update_placeholder.json").exists()
    assert summary["next_run_hint"] == [
        "--queue-performance-summary",
        str(run_dir / "queue_performance_summary.json"),
    ]
    assert summary["build"]["materializer_work_queue_executable_row_count"] == 1
    assert summary["worker"]["success_count"] == 3
    assert summary["worker"]["failure_count"] == 0

    execution_queue = json.loads(
        (run_dir / "materializer_execution_queue.runtime_policy.json").read_text(encoding="utf-8")
    )
    materializer_experiment = next(
        experiment for experiment in execution_queue["experiments"] if experiment["id"].startswith("materializer_work_")
    )
    assert [step["id"] for step in materializer_experiment["steps"]] == [
        "materialize_local_proof_chain",
        "harvest_materializer_chains",
        "build_exact_eval_dispatch_plan",
    ]
    assert materializer_experiment["metadata"]["target_kind"] == (runner.PACKET_MEMBER_RECOMPRESS_TARGET_KIND)
    assert materializer_experiment["metadata"]["exact_readiness_followup_enabled"] is True

    materializer_step = materializer_experiment["steps"][0]
    manifest_path = Path(materializer_step["telemetry"]["artifact_paths"][1])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof_path = Path(materializer_step["telemetry"]["artifact_paths"][2])
    assert manifest["schema"] == "packet_member_recompress_candidate.v1"
    assert manifest["byte_closed_candidate_emitted"] is True
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["candidate_member"]["sha256"] == manifest["source_member"]["sha256"]
    assert manifest["runtime_consumption_proof_path"] == str(proof_path)
    proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof_payload["candidate_archive"]["sha256"] == manifest["candidate_archive"]["sha256"]
    assert proof_payload["candidate_member_payload_identical_to_source"] is True
    assert "runtime_consumption_proof_missing" not in manifest["readiness_blockers"]
    assert manifest["score_claim"] is False

    handoff_dir = manifest_path.parent / "exact_eval_handoff"
    harvest = json.loads((handoff_dir / "harvest_report.json").read_text(encoding="utf-8"))
    bridge = json.loads((handoff_dir / "exact_readiness_bridge_report.json").read_text(encoding="utf-8"))
    dispatch_plan = json.loads((handoff_dir / "dispatch_plan.json").read_text(encoding="utf-8"))
    assert harvest["accepted_manifest_count"] == 1
    assert harvest["source_queue_candidate_count"] == 1
    assert harvest["source_queue_dispatch_ready_count"] == 0
    assert bridge["candidate_count"] == 1
    assert bridge["ready_candidate_count"] == 0
    assert bridge["score_claim"] is False
    assert dispatch_plan["authorized_candidate_count"] == 0
    assert dispatch_plan["score_claim"] is False


def test_materializer_campaign_feedback_followup_queue_fails_truthy_dispatch_alias(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "feedback.json"
    output_path.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "exact_cuda_auth_eval": True,
            }
        ),
        encoding="utf-8",
    )
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
        ]
    )
    request = {
        "ready_for_action_functional_feedback": True,
        "blockers": [],
        "plan_path": str(tmp_path / "plan.json"),
        "queue_performance_summary_path": str(tmp_path / "performance.json"),
        "queue_performance_runtime_identity_path": str(tmp_path / "runtime.json"),
        "queue_performance_cache_identity_path": str(tmp_path / "cache.json"),
        "command_template": [
            runner.sys.executable,
            "tools/build_inverse_steganalysis_action_functional.py",
            "--output",
            str(output_path),
        ],
    }
    queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request=request,
        queue_feedback_replan_request_path=tmp_path / "request.json",
        run_dir=tmp_path,
        execution_queue=tmp_path / "source_queue.json",
        state_path=tmp_path / "source_queue.sqlite",
        source_queue={"queue_id": "unit_queue"},
    )

    assert blockers == []
    assert queue is not None
    with connect_state(tmp_path / "followup.sqlite") as conn:
        initialize_queue_state(conn, queue)
        set_control_mode(conn, queue["queue_id"], "running", reason="unit resume")
        ready = ready_steps(conn, queue)
        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )

    assert result["succeeded"] is False
    assert result["failed_postconditions"][0]["type"] == "json_completion_contract"


def test_materializer_campaign_feedback_followup_execution_requires_execution_intent(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(["--plan", str(tmp_path / "plan.json")])
    assert args.execute_queue_feedback_replan_followup is False
    assert args.queue_feedback_replan_followup_policy_local_autopilot is False

    with pytest.raises(SystemExit, match="queue-feedback-replan-followup-state"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--execute-queue-feedback-replan-followup",
                "--queue-feedback-replan-followup-state",
                str(tmp_path / "feedback.sqlite"),
            ]
        )

    with pytest.raises(SystemExit, match="queue-feedback-replan-followup-state"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--queue-feedback-replan-followup-policy-local-autopilot",
                "--queue-feedback-replan-followup-state",
                str(tmp_path / "feedback.sqlite"),
            ]
        )


def test_materializer_campaign_feedback_followup_queue_blocks_non_action_command(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
        ]
    )
    queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request={
            "ready_for_action_functional_feedback": True,
            "blockers": [],
            "command_template": [
                runner.sys.executable,
                "tools/parallel_dispatch_top_k.py",
                "--output",
                str(tmp_path / "feedback.json"),
                "--dispatch-mode",
                "cuda_auth",
            ],
        },
        queue_feedback_replan_request_path=tmp_path / "request.json",
        run_dir=tmp_path,
        execution_queue=tmp_path / "source_queue.json",
        state_path=tmp_path / "source_queue.sqlite",
        source_queue={"queue_id": "unit_queue"},
    )

    assert queue is None
    assert "queue_feedback_replan_command_not_action_functional_tool" in blockers
    assert "queue_feedback_replan_command_forbidden_flag:--dispatch-mode" in blockers

    queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request={
            "ready_for_action_functional_feedback": True,
            "blockers": [],
            "command_template": [
                runner.sys.executable,
                "tools/build_inverse_steganalysis_action_functional.py",
                "--output",
                str(tmp_path / "feedback.json"),
                "--provider=modal",
            ],
        },
        queue_feedback_replan_request_path=tmp_path / "request.json",
        run_dir=tmp_path,
        execution_queue=tmp_path / "source_queue.json",
        state_path=tmp_path / "source_queue.sqlite",
        source_queue={"queue_id": "unit_queue"},
    )

    assert queue is None
    assert "queue_feedback_replan_command_forbidden_flag:--provider=modal" in blockers

    queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request={
            "ready_for_action_functional_feedback": True,
            "blockers": [],
            "command_template": [
                runner.sys.executable,
                "tools/build_inverse_steganalysis_action_functional.py",
                "--output",
                str(tmp_path.parent / "feedback.json"),
            ],
        },
        queue_feedback_replan_request_path=tmp_path / "request.json",
        run_dir=tmp_path,
        execution_queue=tmp_path / "source_queue.json",
        state_path=tmp_path / "source_queue.sqlite",
        source_queue={"queue_id": "unit_queue"},
    )

    assert queue is None
    assert "queue_feedback_replan_output_path_outside_run_dir" in blockers


def test_materializer_campaign_feedback_followup_local_autopolicy_guard(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
        ]
    )
    queue, blockers = runner._queue_feedback_replan_followup_queue_payload(
        args,
        queue_feedback_replan_request={
            "ready_for_action_functional_feedback": True,
            "blockers": [],
            "plan_path": str(tmp_path / "plan.json"),
            "queue_performance_summary_path": str(tmp_path / "performance.json"),
            "queue_performance_runtime_identity_path": str(tmp_path / "runtime.json"),
            "queue_performance_cache_identity_path": str(tmp_path / "cache.json"),
            "command_template": [
                runner.sys.executable,
                "tools/build_inverse_steganalysis_action_functional.py",
                "--output",
                str(tmp_path / "feedback.json"),
            ],
        },
        queue_feedback_replan_request_path=tmp_path / "request.json",
        run_dir=tmp_path,
        execution_queue=tmp_path / "source_queue.json",
        state_path=tmp_path / "source_queue.sqlite",
        source_queue={"queue_id": "unit_queue"},
    )

    assert blockers == []
    assert queue is not None
    assert (
        runner._queue_feedback_replan_followup_local_autopolicy_blockers(
            queue,
            run_dir=tmp_path,
        )
        == []
    )

    unsafe_queue = json.loads(json.dumps(queue))
    unsafe_queue["controls"]["mode"] = "running"
    unsafe_queue["controls"]["max_concurrency"]["cuda"] = 1
    unsafe_queue["experiments"][0]["metadata"]["score_claim"] = True
    unsafe_queue["experiments"][0]["metadata"]["dispatch_ready"] = True
    unsafe_queue["experiments"][0]["metadata"]["exact_eval_dispatch_ready"] = True
    unsafe_queue["experiments"][0]["metadata"]["score_affecting_payload_changed"] = True
    unsafe_queue["experiments"][0]["metadata"]["charged_bits_changed"] = True
    unsafe_queue["experiments"][0]["metadata"]["dispatch_packet_ready"] = True
    unsafe_step = unsafe_queue["experiments"][0]["steps"][0]
    unsafe_step["resources"]["kind"] = "cuda"
    unsafe_step["command"].extend(["--provider", "modal"])

    policy_blockers = runner._queue_feedback_replan_followup_local_autopolicy_blockers(
        unsafe_queue,
        run_dir=tmp_path,
    )

    assert "queue_feedback_replan_followup_control_mode_not_paused" in policy_blockers
    assert "queue_feedback_replan_followup_non_local_concurrency:cuda" in policy_blockers
    assert (
        "queue_feedback_replan_followup_truthy_authority_field:"
        "experiments[0].metadata.score_claim=truthy"
    ) in policy_blockers
    assert (
        "queue_feedback_replan_followup_truthy_authority_field:"
        "experiments[0].metadata.dispatch_ready=truthy"
    ) in policy_blockers
    assert (
        "queue_feedback_replan_followup_truthy_authority_field:"
        "experiments[0].metadata.exact_eval_dispatch_ready=truthy"
    ) in policy_blockers
    assert (
        "queue_feedback_replan_followup_truthy_authority_field:"
        "experiments[0].metadata.score_affecting_payload_changed=truthy"
    ) in policy_blockers
    assert (
        "queue_feedback_replan_followup_truthy_authority_field:"
        "experiments[0].metadata.charged_bits_changed=truthy"
    ) in policy_blockers
    assert (
        "queue_feedback_replan_followup_truthy_authority_field:"
        "experiments[0].metadata.dispatch_packet_ready=truthy"
    ) in policy_blockers
    assert "queue_feedback_replan_followup_step_not_local_cpu:0:0" in policy_blockers
    assert (
        "queue_feedback_replan_followup_step_command_forbidden_flag:0:0:--provider"
    ) in policy_blockers


def test_materializer_campaign_runner_preserves_feedback_artifacts_when_worker_fails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    state_path = tmp_path / "materializer_execution_queue.sqlite"
    plan = tmp_path / "plan.json"
    source_archive = tmp_path / "not_a_zip.zip"

    plan.write_text(json.dumps(_packet_member_recompress_plan()), encoding="utf-8")
    source_archive.write_bytes(b"not a zip archive")

    result = runner.main(
        [
            "--plan",
            str(plan),
            "--packet-member-archive-path",
            str(source_archive),
            "--packet-member-name",
            "payload.bin",
            "--packet-member-zip-compression-method",
            "stored",
            "--materializer-contexts-fail-if-blocked",
            "--run-dir",
            str(run_dir),
            "--queue-state",
            str(state_path),
            "--queue-state-rationale",
            "isolated worker failure feedback preservation smoke",
            "--queue-id",
            "packet_member_failure_runner_fixture",
            "--max-steps",
            "1",
            "--max-parallel",
            "1",
            "--idle-sleep-seconds",
            "0",
            "--max-idle-cycles",
            "1",
            "--execute",
        ]
    )

    assert result == 2
    summary = json.loads((run_dir / "materializer_campaign_run.json").read_text(encoding="utf-8"))
    performance = json.loads((run_dir / "queue_performance_summary.json").read_text(encoding="utf-8"))
    replan = json.loads((run_dir / "queue_feedback_replan_request.json").read_text(encoding="utf-8"))
    placeholder = json.loads(
        (run_dir / "canonical_response_update_placeholder.json").read_text(encoding="utf-8")
    )

    assert summary["worker"]["schema"] == "experiment_queue_worker_result.v1"
    assert summary["worker"]["failure_count"] == 1
    assert summary["performance"]["worker_returncode"] == 2
    assert performance["schema"] == runner.QUEUE_PERFORMANCE_SUMMARY_SCHEMA
    assert performance["event_count"] == 1
    assert performance["score_claim"] is False
    assert replan["score_claim"] is False
    assert placeholder["score_claim"] is False
    assert placeholder["replan_required"] is True


def test_materializer_campaign_runner_uses_policy_cold_store_default_for_move_preflight(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(tmp_path / "campaign" / "work"),
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert "--include-materializer-scheduler-preflight" in command
    assert "--materializer-scheduler-proactive-cleanup-execute" in command
    assert "--materializer-scheduler-proactive-cleanup-cold-store-root" not in command


def test_materializer_campaign_runner_emits_staircase_artifacts(tmp_path: Path) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "campaign_staircase_fixture",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
            "experiments": [
                {
                    "id": "candidate",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "materialize",
                            "command": ["python", "-c", "print('ok')"],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [{"type": "path_exists", "path": str(tmp_path / "done.json")}],
                        }
                    ],
                }
            ],
        }
    )
    queue_path = tmp_path / "queue.json"
    state_path = tmp_path / "queue.sqlite"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--emit-staircase-plan",
            "--staircase-resource-pool",
            "sshbox:local_cpu=1,memory_gb=8,disk_gb=8,"
            "executor=ssh_experiment_queue,ssh_target=user@sshbox,"
            "remote_repo_root=/remote/pact",
        ]
    )
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()

    result = runner._build_staircase_artifacts(
        args,
        run_dir=run_dir,
        execution_queue=queue_path,
        state_path=state_path,
        queue=queue,
    )

    assert result["selected_count"] == 1
    plan = json.loads((run_dir / "staircase_dispatch_plan.json").read_text(encoding="utf-8"))
    task = plan["dask_task_specs"][0]
    assert task["machine"]["executor"] == "ssh_experiment_queue"
    assert task["machine"]["remote_repo_root"] == "/remote/pact"
    assert task["queue_state_writeback"]["required"] is True
    assert plan["score_claim"] is False


def test_materializer_campaign_runner_builds_ssh_dry_run_command(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--staircase-ssh-dry-run",
            "--staircase-ssh-machine-id",
            "sshbox",
            "--staircase-ssh-remote-repo-root",
            "sshbox=/remote/pact",
        ]
    )

    command = runner._ssh_executor_dry_run_command(
        args,
        execution_queue=tmp_path / "queue.json",
        state_path=tmp_path / "queue.sqlite",
        staircase_plan_path=tmp_path / "plan.staircase.json",
        run_dir=tmp_path / "campaign",
    )

    assert command[:2] == [
        runner.sys.executable,
        "tools/run_staircase_ssh_executor.py",
    ]
    assert "--execute" not in command
    assert "--machine-id" in command
    assert "sshbox" in command
    assert "--remote-repo-root" in command
    assert "sshbox=/remote/pact" in command


def test_materializer_campaign_runner_builds_ssh_execute_command_with_artifact_pullback(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--staircase-ssh-execute",
            "--staircase-ssh-max-steps",
            "3",
            "--staircase-ssh-machine-id",
            "sshbox",
            "--staircase-ssh-remote-repo-root",
            "sshbox=/remote/pact",
            "--staircase-ssh-artifact-path-map",
            f"{tmp_path / 'campaign'}=/remote/campaign",
            "--staircase-ssh-rsync-binary",
            "rsync-fixture",
            "--staircase-ssh-artifact-pull-timeout-seconds",
            "42",
        ]
    )

    command = runner._ssh_executor_command(
        args,
        execution_queue=tmp_path / "queue.json",
        state_path=tmp_path / "queue.sqlite",
        staircase_plan_path=tmp_path / "plan.staircase.json",
        run_dir=tmp_path / "campaign",
        execute=True,
    )

    assert "--execute" in command
    assert "--max-steps" in command
    assert "3" in command
    assert "--require-artifact-mobility" in command
    assert "--artifact-path-map" in command
    assert f"{tmp_path / 'campaign'}=/remote/campaign" in command
    assert "--rsync-binary" in command
    assert "rsync-fixture" in command
    assert "--artifact-pull-timeout-seconds" in command
    assert "42" in command
    assert any("staircase_ssh_executor_execute.json" in part for part in command)


def test_materializer_campaign_runner_generates_plan_from_high_level_sources(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer_response.json"
    mlx_batch = tmp_path / "mlx_acquisition_batch.json"
    action_path = tmp_path / "campaign" / "inverse_steganalysis_action_functional.json"
    plan_path = tmp_path / "campaign" / "byte_shaving_campaign_plan.json"
    scorer.write_text("{}", encoding="utf-8")
    mlx_batch.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--scorer-response",
            str(scorer),
            "--mlx-acquisition-batch",
            str(mlx_batch),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "high_level_fixture",
            "--candidate-id",
            "candidate_a",
            "--total-byte-budget",
            "64",
            "--campaign-plan-max-k",
            "3",
            "--queue-id",
            "high_level_materializer_queue",
        ]
    )

    action_command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )
    plan_command = runner._build_campaign_plan_command(
        args,
        action_functional_path=action_path,
        run_dir=tmp_path / "campaign",
    )
    queue_command = runner._build_queue_command(
        args,
        run_dir=tmp_path / "campaign",
        plan_path=plan_path,
    )

    assert action_command[:2] == [
        runner.sys.executable,
        "tools/build_inverse_steganalysis_action_functional.py",
    ]
    assert "--scorer-response" in action_command
    assert str(scorer) in action_command
    assert "--mlx-acquisition-batch" in action_command
    assert str(mlx_batch) in action_command
    assert "--total-byte-budget" in action_command
    assert "64" in action_command
    assert "--candidate-id" in action_command
    assert "candidate_a" in action_command
    assert plan_command[:2] == [
        runner.sys.executable,
        "tools/plan_byte_shaving_campaign.py",
    ]
    assert "--from-inverse-action-functional" in plan_command
    assert "--campaign-id" in plan_command
    assert "high_level_fixture" in plan_command
    assert "--max-k" in plan_command
    assert "3" in plan_command
    assert "--plan" in queue_command
    assert str(plan_path) in queue_command


def test_materializer_campaign_runner_treats_mlx_acquisition_batch_as_action_source(
    tmp_path: Path,
) -> None:
    mlx_batch = tmp_path / "mlx_acquisition_batch.json"
    mlx_batch.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--mlx-acquisition-batch",
            str(mlx_batch),
            "--run-dir",
            str(tmp_path / "campaign"),
        ]
    )

    assert runner._action_source_count(args) == 1
    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )

    assert "--mlx-acquisition-batch" in command
    assert str(mlx_batch) in command


def test_materializer_campaign_runner_forwards_family_byte_shaving_sources(
    tmp_path: Path,
) -> None:
    surface = tmp_path / "family_surface.json"
    campaign_plan = tmp_path / "family_campaign_plan.json"
    surface.write_text("{}", encoding="utf-8")
    campaign_plan.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--byte-shaving-signal-surface",
            str(surface),
            "--byte-shaving-campaign-plan",
            str(campaign_plan),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "family_mix_runner",
        ]
    )

    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )

    assert runner._action_source_count(args) == 2
    assert "--byte-shaving-signal-surface" in command
    assert str(surface) in command
    assert "--byte-shaving-campaign-plan" in command
    assert str(campaign_plan) in command
    assert "--candidate-id" not in command


def test_materializer_campaign_runner_builds_mlx_batch_from_selection_inline(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "mlx_selection.json"
    selection.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "campaign"
    args = runner.parse_args(
        [
            "--mlx-effective-spend-triage-selection",
            str(selection),
            "--run-dir",
            str(run_dir),
            "--mlx-acquisition-set-size",
            "4",
            "--mlx-acquisition-limit",
            "8",
            "--overwrite-output",
        ]
    )

    batch_commands = runner._build_mlx_acquisition_batch_commands(
        args,
        run_dir=run_dir,
    )
    assert len(batch_commands) == 1
    batch_path, batch_command = batch_commands[0]
    assert batch_path == run_dir / "mlx_acquisition_batch_0000.json"
    assert batch_command[:2] == [
        runner.sys.executable,
        "tools/build_mlx_acquisition_batch.py",
    ]
    assert "--mlx-effective-spend-triage-selection" in batch_command
    assert str(selection) in batch_command
    assert "--set-size" in batch_command
    assert "4" in batch_command
    assert "--limit" in batch_command
    assert "8" in batch_command
    assert "--allow-overwrite" in batch_command

    action_command = runner._build_action_functional_command(
        args,
        run_dir=run_dir,
        generated_mlx_acquisition_batches=[batch_path],
    )
    assert "--mlx-acquisition-batch" in action_command
    assert str(batch_path) in action_command
    assert "--mlx-effective-spend-triage-selection" not in action_command


def test_materializer_campaign_runner_can_preserve_direct_mlx_selection_mode(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "mlx_selection.json"
    selection.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--mlx-effective-spend-triage-selection",
            str(selection),
            "--mlx-effective-spend-triage-selection-mode",
            "direct",
            "--run-dir",
            str(tmp_path / "campaign"),
        ]
    )

    assert (
        runner._build_mlx_acquisition_batch_commands(
            args,
            run_dir=tmp_path / "campaign",
        )
        == []
    )
    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )
    assert "--mlx-effective-spend-triage-selection" in command
    assert str(selection) in command


def test_materializer_campaign_runner_loads_file_driven_run_config(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer_response.json"
    selection = tmp_path / "mlx_selection.json"
    config_path = tmp_path / "rate_attack_config.json"
    scorer.write_text("{}", encoding="utf-8")
    selection.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "schema": runner.RUN_CONFIG_SCHEMA,
                "args": {
                    "scorer_response": [str(scorer)],
                    "mlx_effective_spend_triage_selection": [str(selection)],
                    "run_dir": str(tmp_path / "campaign"),
                    "campaign_id": "configured_final_rate_attack",
                    "candidate_id": "candidate_from_config",
                    "total_byte_budget": 96,
                    "campaign_plan_max_k": 4,
                    "queue_id": "configured_materializer_queue",
                    "mlx_acquisition_set_size": 3,
                    "materializer_resource_concurrency": ["local_cpu=2"],
                    "emit_staircase_plan": True,
                },
            }
        ),
        encoding="utf-8",
    )

    args = runner.parse_args(["--run-config", str(config_path)])

    assert args.run_config == config_path
    assert args.scorer_response == [str(scorer)]
    assert args.mlx_effective_spend_triage_selection == [str(selection)]
    assert args.run_dir == tmp_path / "campaign"
    assert args.campaign_id == "configured_final_rate_attack"
    assert args.candidate_id == "candidate_from_config"
    assert args.total_byte_budget == 96
    assert args.campaign_plan_max_k == 4
    assert args.queue_id == "configured_materializer_queue"
    assert args.mlx_acquisition_set_size == 3
    assert args.materializer_resource_concurrency == ["local_cpu=2"]
    assert args.emit_staircase_plan is True

    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )
    assert "--scorer-response" in command
    assert str(scorer) in command
    assert "--candidate-id" in command
    assert "candidate_from_config" in command


def test_materializer_campaign_runner_cli_scalar_overrides_run_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "rate_attack_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": runner.RUN_CONFIG_SCHEMA,
                "args": {
                    "atom": [str(tmp_path / "atom.json")],
                    "run_dir": str(tmp_path / "campaign_from_config"),
                    "campaign_id": "configured_campaign",
                    "candidate_id": "configured_candidate",
                    "total_byte_budget": 96,
                },
            }
        ),
        encoding="utf-8",
    )

    args = runner.parse_args(
        [
            "--run-config",
            str(config_path),
            "--candidate-id",
            "cli_candidate",
            "--total-byte-budget",
            "32",
        ]
    )

    assert args.atom == [str(tmp_path / "atom.json")]
    assert args.campaign_id == "configured_campaign"
    assert args.candidate_id == "cli_candidate"
    assert args.total_byte_budget == 32


def test_materializer_campaign_runner_rejects_unknown_run_config_key(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "rate_attack_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": runner.RUN_CONFIG_SCHEMA,
                "args": {"definitely_not_a_runner_field": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unknown run config key"):
        runner.parse_args(["--run-config", str(config_path)])


def test_materializer_campaign_runner_rejects_missing_plan_and_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="provide --plan or high-level action sources"):
        runner.main(["--run-dir", str(tmp_path / "campaign")])


def test_materializer_campaign_runner_rejects_mixed_plan_and_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--scorer-response",
                str(tmp_path / "scorer_response.json"),
                "--run-dir",
                str(tmp_path / "campaign"),
            ]
        )


def test_materializer_campaign_runner_rejects_mixed_local_and_ssh_execute(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="cannot target the same queue run"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--run-dir",
                str(tmp_path / "campaign"),
                "--execute",
                "--staircase-ssh-execute",
                "--staircase-ssh-artifact-path-map",
                f"{tmp_path / 'campaign'}=/remote/campaign",
            ]
        )


def test_materializer_campaign_runner_rejects_ssh_execute_without_artifact_mobility(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="requires --staircase-ssh-artifact-path-map"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--run-dir",
                str(tmp_path / "campaign"),
                "--staircase-ssh-execute",
            ]
        )


def test_materializer_campaign_runner_rejects_bad_resource_concurrency() -> None:
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=0"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=two"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu"])


def test_materializer_campaign_runner_rejects_bad_ssh_remote_root_mapping() -> None:
    with pytest.raises(SystemExit):
        runner._parse_remote_repo_roots(["sshbox"])


def test_materializer_campaign_runner_rejects_bad_artifact_path_mapping() -> None:
    with pytest.raises(SystemExit):
        runner._parse_artifact_path_maps(["/local-only"])


def test_materializer_campaign_runner_requires_json_from_ssh_dry_run() -> None:
    with pytest.raises(SystemExit, match="did not emit a JSON object"):
        runner._require_json_stdout(
            runner.CommandResult(
                command=["ssh-dry-run"],
                returncode=0,
                stdout="not-json",
                stderr="",
                elapsed_seconds=0.0,
            ),
            label="staircase SSH executor dry-run",
        )
    with pytest.raises(SystemExit, match="failed"):
        runner._require_json_stdout(
            runner.CommandResult(
                command=["ssh-dry-run"],
                returncode=2,
                stdout="",
                stderr="bad plan",
                elapsed_seconds=0.0,
            ),
            label="staircase SSH executor dry-run",
        )
