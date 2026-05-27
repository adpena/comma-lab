# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.repair_campaign_stackability_queue import (
    REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA,
    build_repair_campaign_stackability_queue,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_ROW_SCHEMA,
    append_repair_campaign_stackability_posterior_signal,
    load_repair_campaign_stackability_posterior_rows,
)
from tac.optimization.repair_campaign_replay_bundle import (
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA,
    build_repair_campaign_stackability_replay_bundle,
    diff_repair_campaign_stackability_replay_bundles,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
    build_repair_campaign_stackability_probe,
    score_repair_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _work_order(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "segnet_mlx_response.json"
    ref = tmp_path / "segnet_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 40,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 40,
            "rows": [
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "segnet_region_ready",
                    "candidate_id": "segnet_class_region_waterfill",
                    "acquisition_id": "segnet_region_acq",
                    "correction_family": "segnet_class_region_waterfill",
                    "targeted_dimensions": ["segnet", "region"],
                    "operation_levels": ["frame", "region"],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    "interaction_scope": {
                        "pair_indices": [7, 9],
                        "region_ids": ["road_boundary"],
                        **_false_authority(),
                    },
                    "stacking_interaction_terms": {
                        "must_remeasure_with_parent_and_sibling_repairs": True,
                        **_false_authority(),
                    },
                    **_false_authority(),
                },
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "selector_missing",
                    "candidate_id": "per_region_selector_codec",
                    "acquisition_id": "selector_acq",
                    "correction_family": "per_region_selector_codec",
                    "targeted_dimensions": ["selector_stream", "region"],
                    "operation_levels": ["entropy_coder"],
                    "entropy_position_label": "selector_codec_entropy",
                    "requested_repair_bytes": 8,
                    "objective_delta_score_units": -0.0008,
                    **_false_authority(),
                },
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def test_stackability_probe_preserves_optimizer_signal_and_authority(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)

    probe = build_repair_campaign_stackability_probe(
        score_report=report,
        typed_response_id="segnet_region_ready",
        repo_root=tmp_path,
    )

    assert probe["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert probe["stackability_ready"] is True
    assert probe["component_response_axis"] == "[macOS-MLX research-signal]"
    assert probe["allocated_repair_bytes"] == 32
    assert probe["entropy_position_class"] == "pre_entropy_distribution_shaping"
    assert probe["optimizer_allocation"]["targeted_dimensions"] == [
        "segnet",
        "region",
    ]
    assert probe["source_score_row"]["interaction_scope"]["pair_indices"] == [7, 9]
    assert probe["budget_spend_allowed"] is False
    assert probe["ready_for_exact_eval_dispatch"] is False
    assert "local_mlx_probe_is_not_score_authority" in probe["blockers"]


def test_repair_campaign_learning_signal_is_public_optimization_export() -> None:
    from tac.optimization import (
        REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA as exported_schema,
    )
    from tac.optimization import (
        append_repair_campaign_stackability_posterior_signal as exported_append,
    )
    from tac.optimization import (
        build_repair_campaign_learning_signal as exported_builder,
    )

    assert exported_schema == REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA
    assert callable(exported_builder)
    assert exported_append is append_repair_campaign_stackability_posterior_signal


def test_stackability_queue_emits_executable_local_probe(tmp_path: Path) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    report_path = tmp_path / "repair_campaign_score_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_repair_campaign_stackability_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        results_root=tmp_path / "results",
        queue_id="stackability_test",
        posterior_path=tmp_path / "repair_campaign_stackability_posterior.jsonl",
        posterior_lock_path=tmp_path / ".repair_campaign_stackability_posterior.lock",
    )

    assert queue["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA
    )
    assert queue["metadata"]["ready_experiment_count"] == 1
    assert queue["metadata"]["blocked_experiment_count"] == 0
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["replay_bundle_path"].endswith(
        "repair_campaign_stackability_replay_bundle.json"
    )
    assert experiment["metadata"]["learning_signal_path"].endswith(
        "repair_campaign_learning_signal.json"
    )
    command = [
        sys.executable if item == ".venv/bin/python" else str(item)
        for item in experiment["steps"][0]["command"]
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(
        Path(experiment["metadata"]["probe_output_path"]).read_text(encoding="utf-8")
    )
    assert artifact["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert artifact["stackability_ready"] is True
    assert artifact["budget_spend_allowed"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False
    bundle_command = [
        sys.executable if item == ".venv/bin/python" else str(item)
        for item in experiment["steps"][1]["command"]
    ]
    bundle_result = subprocess.run(
        bundle_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bundle_result.returncode == 0, bundle_result.stderr
    bundle = json.loads(
        Path(experiment["metadata"]["replay_bundle_path"]).read_text(encoding="utf-8")
    )
    assert bundle["schema"] == REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA
    assert bundle["replay_argv"] == experiment["steps"][0]["command"]
    assert bundle["hash_manifest_sha256"]
    assert bundle["source_records_sha256"]
    assert bundle["replay_argv_sha256"]
    assert bundle["execution_context_sha256"]
    assert bundle["environment"]["schema"] == "safe_replay_environment_capture.v1"
    assert bundle["budget_spend_allowed"] is False
    assert bundle["ready_for_exact_eval_dispatch"] is False
    rerun_command = [
        sys.executable if item == ".venv/bin/python" else str(item)
        for item in experiment["steps"][2]["command"]
    ]
    rerun_result = subprocess.run(
        rerun_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rerun_result.returncode == 0, rerun_result.stderr
    rerun_summary = json.loads(
        Path(experiment["metadata"]["replay_rerun_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert rerun_summary["schema"] == REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA
    assert rerun_summary["matched"] is True
    assert rerun_summary["probe_payload_matched"] is True
    assert rerun_summary["local_mlx_custody_hashes_matched"] is True
    assert rerun_summary["budget_spend_allowed"] is False
    assert rerun_summary["ready_for_exact_eval_dispatch"] is False
    learning_command = [
        sys.executable if item == ".venv/bin/python" else str(item)
        for item in experiment["steps"][3]["command"]
    ]
    learning_result = subprocess.run(
        learning_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert learning_result.returncode == 0, learning_result.stderr
    learning_signal = json.loads(
        Path(experiment["metadata"]["learning_signal_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert learning_signal["schema"] == REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA
    assert learning_signal["typed_response_id"] == "segnet_region_ready"
    assert learning_signal["replay_identity"]["hash_manifest_sha256"] == (
        bundle["hash_manifest_sha256"]
    )
    assert learning_signal["local_planning_update"][
        "local_planning_update_ready"
    ] is True
    assert learning_signal["local_planning_update"]["planner_feature_vector"][
        "improvement_per_allocated_byte"
    ] > 0.0
    assert learning_signal["budget_spend_allowed"] is False
    assert learning_signal["ready_for_exact_eval_dispatch"] is False
    posterior_command = [
        sys.executable if item == ".venv/bin/python" else str(item)
        for item in experiment["steps"][4]["command"]
    ]
    posterior_result = subprocess.run(
        posterior_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert posterior_result.returncode == 0, posterior_result.stderr
    posterior_report = json.loads(
        Path(experiment["metadata"]["posterior_append_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(
        tmp_path / "repair_campaign_stackability_posterior.jsonl"
    )
    assert posterior_report["schema"] == (
        REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
    )
    assert posterior_report["appended"] is True
    assert posterior_report["ready_for_exact_eval_dispatch"] is False
    assert len(posterior_rows) == 1
    assert posterior_rows[0]["schema"] == REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_ROW_SCHEMA
    assert posterior_rows[0]["row_id"] == posterior_report["row_id"]
    assert posterior_rows[0]["replay_identity"]["hash_manifest_sha256"] == (
        bundle["hash_manifest_sha256"]
    )
    duplicate_result = subprocess.run(
        posterior_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert duplicate_result.returncode == 0, duplicate_result.stderr
    duplicate_report = json.loads(
        Path(experiment["metadata"]["posterior_append_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    duplicate_rows = load_repair_campaign_stackability_posterior_rows(
        tmp_path / "repair_campaign_stackability_posterior.jsonl"
    )
    assert duplicate_report["appended"] is False
    assert duplicate_report["skipped_duplicate"] is True
    assert len(duplicate_rows) == 1


def test_stackability_replay_bundle_diff_detects_environment_drift(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    report_path = tmp_path / "repair_campaign_score_report.json"
    probe_path = tmp_path / "repair_campaign_stackability_probe.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    probe = build_repair_campaign_stackability_probe(
        score_report=report,
        typed_response_id="segnet_region_ready",
        repo_root=tmp_path,
    )
    probe_path.write_text(
        json.dumps(probe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replay_argv = [
        ".venv/bin/python",
        "tools/run_repair_campaign_stackability_probe.py",
        "--score-report",
        str(report_path),
        "--typed-response-id",
        "segnet_region_ready",
        "--output",
        str(probe_path),
        "--overwrite",
    ]

    left = build_repair_campaign_stackability_replay_bundle(
        score_report_path=report_path,
        probe_path=probe_path,
        score_report=report,
        probe=probe,
        replay_argv=replay_argv,
        invocation_argv=["python", "bundle"],
        repo_root=tmp_path,
        environment={"MLX_VISIBLE_DEVICES": "all", "API_TOKEN": "secret"},
    )
    right = build_repair_campaign_stackability_replay_bundle(
        score_report_path=report_path,
        probe_path=probe_path,
        score_report=report,
        probe=probe,
        replay_argv=replay_argv,
        invocation_argv=["python", "bundle"],
        repo_root=tmp_path,
        environment={"MLX_VISIBLE_DEVICES": "none", "API_TOKEN": "changed"},
    )

    assert left["environment"]["env"]["API_TOKEN"] == "[REDACTED]"
    diff = diff_repair_campaign_stackability_replay_bundles(left, right)
    assert diff["schema"] == REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA
    assert diff["stable_replay_identity_matched"] is True
    assert diff["execution_context_matched"] is False
    assert diff["matched"] is False
    assert diff["changed_environment_keys"] == ["MLX_VISIBLE_DEVICES"]
    assert diff["budget_spend_allowed"] is False
    assert diff["ready_for_exact_eval_dispatch"] is False

    left_path = tmp_path / "left_replay_bundle.json"
    right_path = tmp_path / "right_replay_bundle.json"
    diff_path = tmp_path / "replay_bundle_diff.json"
    left_path.write_text(json.dumps(left, indent=2, sort_keys=True) + "\n")
    right_path.write_text(json.dumps(right, indent=2, sort_keys=True) + "\n")
    command = [
        sys.executable,
        "tools/diff_repair_campaign_stackability_replay_bundle.py",
        "--left",
        str(left_path),
        "--right",
        str(right_path),
        "--diff-out",
        str(diff_path),
        "--overwrite",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    cli_diff = json.loads(diff_path.read_text(encoding="utf-8"))
    assert cli_result["matched"] is False
    assert cli_diff["schema"] == REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA
    assert cli_diff["changed_environment_keys"] == ["MLX_VISIBLE_DEVICES"]


def test_repair_stackability_rerun_command_rewrites_probe_output(
    tmp_path: Path,
) -> None:
    tool_path = REPO_ROOT / "tools" / "rerun_repair_campaign_stackability_replay_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "repair_stackability_rerun_tool_under_test",
        tool_path,
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    bundle = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
        "replay_target_tool": "tools/run_repair_campaign_stackability_probe.py",
        "typed_response_id": "segnet_region_ready",
        "replay_argv": [
            ".venv/bin/python",
            "tools/run_repair_campaign_stackability_probe.py",
            "--score-report",
            "score.json",
            "--typed-response-id",
            "segnet_region_ready",
            "--output=original_probe.json",
            "--overwrite",
        ],
        "calibration_gate": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        **_false_authority(),
    }

    record = tool.build_rerun_command(
        bundle,
        bundle_path=tmp_path / "source_replay_bundle.json",
        output_dir=tmp_path / "reruns",
        python_executable="/usr/bin/python3",
        run_id="stable_repair_replay",
    )
    command = record["command"]

    assert command[:2] == [
        "/usr/bin/python3",
        str(REPO_ROOT / "tools" / "run_repair_campaign_stackability_probe.py"),
    ]
    assert "--score-report" in command
    assert "--typed-response-id" in command
    assert not any(str(part).endswith("original_probe.json") for part in command)
    assert sum(str(part).startswith("--output") for part in command) == 1
    assert record["run_dir"] == str(tmp_path / "reruns" / "stable_repair_replay")
    assert record["side_effect_policy"]["original_probe_output_rewritten"] is True
    assert record["budget_spend_allowed"] is False
    assert record["ready_for_exact_eval_dispatch"] is False


def test_repair_stackability_rerun_rejects_truthy_authority(
    tmp_path: Path,
) -> None:
    tool_path = REPO_ROOT / "tools" / "rerun_repair_campaign_stackability_replay_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "repair_stackability_rerun_tool_bad_authority",
        tool_path,
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    bundle = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
        "replay_target_tool": "tools/run_repair_campaign_stackability_probe.py",
        "replay_argv": [
            ".venv/bin/python",
            "tools/run_repair_campaign_stackability_probe.py",
            "--score-report",
            "score.json",
            "--typed-response-id",
            "segnet_region_ready",
            "--output",
            "probe.json",
        ],
        "calibration_gate": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": True,
        },
        **_false_authority(),
    }

    with pytest.raises(
        tool.RepairCampaignReplayBundleError,
        match="ready_for_exact_eval_dispatch",
    ):
        tool.build_rerun_command(
            bundle,
            bundle_path=tmp_path / "bad_replay_bundle.json",
            output_dir=tmp_path / "reruns",
            python_executable="/usr/bin/python3",
        )


def test_stackability_queue_freezes_bad_selected_allocation_with_exact_missing_names(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    selected = report["optimizer_decision"]["selected_allocation_rows"][0]
    selected["typed_response_id"] = "selector_missing"

    queue = build_repair_campaign_stackability_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=tmp_path / "repair_campaign_score_report.json",
        results_root=tmp_path / "results",
        queue_id="stackability_blocked_test",
    )

    experiment = queue["experiments"][0]
    blockers = experiment["metadata"]["queue_actuation_blockers"]
    assert experiment["status"] == "frozen"
    assert "local_mlx_response_path:missing_or_unverified" in blockers
    assert "reference_local_mlx_response_path:missing_or_unverified" in blockers
    assert "runtime_consumption_proof_path" in blockers
    assert experiment["metadata"]["budget_spend_allowed"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False


def test_stackability_queue_freezes_empty_optimizer_selection(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    work_order["receiver_closed_rate_credit"]["receiver_closed_saved_bytes_total"] = 0
    work_order["typed_response_ledger"]["available_receiver_closed_rate_credit_bytes"] = 0
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    queue = build_repair_campaign_stackability_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=tmp_path / "repair_campaign_score_report.json",
        results_root=tmp_path / "results",
        queue_id="stackability_empty_test",
    )

    assert queue["metadata"]["selected_allocation_count"] == 0
    assert queue["metadata"]["ready_experiment_count"] == 0
    assert queue["metadata"]["blocked_experiment_count"] == 1
    assert "optimizer_selected_allocation_rows_empty" in queue["metadata"][
        "queue_actuation_blockers"
    ]
    assert queue["experiments"][0]["status"] == "frozen"
    assert queue["experiments"][0]["metadata"]["budget_spend_allowed"] is False
