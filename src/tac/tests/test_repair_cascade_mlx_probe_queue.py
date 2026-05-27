# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA
from comma_lab.scheduler.repair_cascade_mlx_probe_queue import (
    REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA,
    REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
    REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
    REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA,
    build_repair_cascade_mlx_learning_signal,
    build_repair_cascade_mlx_probe_queue,
    build_repair_cascade_mlx_probe_result,
    build_repair_cascade_mlx_probe_spec,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import score_repair_campaign

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


def _work_order() -> dict[str, object]:
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 24,
            **_false_authority(),
        },
        "repair_cascade_opportunity_rows": [
            {
                "schema": "frontier_rate_attack_repair_cascade_opportunity_row.v1",
                "cascade_id": "cascade_c_posenet_null_segnet_region_selector_codec",
                "label": "Cascade C",
                "source_relation": "PR110-OPT-5+7+10+12_UNTOUCHED",
                "pipeline_position": "scorer_entropy_repair_before_selector_codec",
                "targeted_positions": [
                    {"position_id": "P19", "entropy_surface": "scorer_entropy"},
                    {"position_id": "P18", "entropy_surface": "scorer_entropy"},
                    {"position_id": "P11", "entropy_surface": "selector_codec_entropy"},
                ],
                "required_probe_measurements": [
                    "posenet_null_bottom_decile_pair_ids",
                    "segnet_class_region_mask_ids",
                    "selector_payload_bits_per_region",
                ],
                "next_queue_action": (
                    "build_cascade_c_mlx_local_probe_queue_and_emit_component_"
                    "response_rows"
                ),
                "blockers": [
                    "cascade_c_empirical_component_response_missing",
                    "per_region_selector_codec_materializer_missing",
                ],
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_repair_cascade_mlx_probe_spec_names_exact_missing_artifacts(
    tmp_path: Path,
) -> None:
    work_order = _work_order()

    spec = build_repair_cascade_mlx_probe_spec(
        source_payload=work_order,
        source_payload_path=tmp_path / "repair_budget_waterfill_work_order.json",
        cascade_id="cascade_c_posenet_null_segnet_region_selector_codec",
        repo_root=tmp_path,
    )

    assert spec["schema"] == REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA
    assert spec["cascade_id"] == "cascade_c_posenet_null_segnet_region_selector_codec"
    assert spec["local_mlx_probe_execution_ready"] is False
    assert spec["ready_for_exact_eval_dispatch"] is False
    assert "local_mlx_response_path:missing_or_unverified" in (
        spec["missing_local_mlx_artifacts"]
    )
    assert "segnet_class_region_mask_ids" in spec["required_probe_measurements"]
    assert all(row["score_claim"] is False for row in spec["probe_measurement_plan"])
    assert spec["repair_family_campaign_schema"] == (
        REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA
    )
    assert spec["repair_family_campaign_count"] == 5
    assert {
        row["family_id"] for row in spec["repair_family_campaign_rows"]
    } == {
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
        "frame0_k16_palette_asymmetry",
        "entropy_boundary_probe",
    }
    assert all(
        row["campaign_execution_mode"] == "local_mlx_advisory_only"
        and row["ready_for_exact_eval_dispatch"] is False
        for row in spec["repair_family_campaign_rows"]
    )


def test_repair_cascade_mlx_probe_result_records_missing_mlx_inputs(
    tmp_path: Path,
) -> None:
    work_order = _work_order()
    spec = build_repair_cascade_mlx_probe_spec(
        source_payload=work_order,
        source_payload_path=tmp_path / "repair_budget_waterfill_work_order.json",
        cascade_id="cascade_c_posenet_null_segnet_region_selector_codec",
        repo_root=tmp_path,
    )

    result = build_repair_cascade_mlx_probe_result(
        probe_spec=spec,
        probe_spec_path=tmp_path / "repair_cascade_mlx_probe_spec.json",
        repo_root=tmp_path,
    )

    assert result["schema"] == REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA
    assert result["component_response_axis"] == "[macOS-MLX research-signal]"
    assert result["local_mlx_probe_execution_ready"] is False
    assert result["component_response_row_emitted"] is False
    assert result["learning_signal_kind"] == "blocked_repair_cascade_mlx_probe"
    assert "local_mlx_probe_artifacts_missing" in result["blockers"]
    assert "local_mlx_response_path:missing_or_unverified" in (
        result["missing_local_mlx_artifacts"]
    )
    assert result["repair_family_campaign_count"] == 5
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_repair_cascade_mlx_learning_signal_records_multiscale_blockers(
    tmp_path: Path,
) -> None:
    work_order = _work_order()
    spec = build_repair_cascade_mlx_probe_spec(
        source_payload=work_order,
        source_payload_path=tmp_path / "repair_budget_waterfill_work_order.json",
        cascade_id="cascade_c_posenet_null_segnet_region_selector_codec",
        repo_root=tmp_path,
    )
    result = build_repair_cascade_mlx_probe_result(
        probe_spec=spec,
        probe_spec_path=tmp_path / "repair_cascade_mlx_probe_spec.json",
        repo_root=tmp_path,
    )
    result_path = _write_json(tmp_path / "repair_cascade_mlx_probe_result.json", result)

    signal = build_repair_cascade_mlx_learning_signal(
        probe_result=result,
        probe_result_path=result_path,
        repo_root=tmp_path,
    )

    assert signal["schema"] == REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA
    assert signal["family_id"] == "entropy_position_cascade"
    assert signal["component_response_axis"] == "[macOS-MLX research-signal]"
    assert signal["ready_for_exact_eval_dispatch"] is False
    assert signal["local_planning_update"]["recommended_acquisition_policy"] == (
        "materialize_missing_local_mlx_custody_before_stackability"
    )
    features = signal["local_planning_update"]["planner_feature_vector"]
    assert features["targeted_position_count"] == 3
    assert features["repair_family_campaign_count"] == 5
    assert "frame0_k16_palette_asymmetry" in features["repair_family_ids"]
    assert "scorer_entropy" in features["entropy_surfaces"]
    assert "selector_codec" in features["operation_levels"]
    assert "local_mlx_probe_artifacts_missing" in signal["blockers"]


def test_repair_cascade_mlx_probe_queue_from_score_report_runs_spec_step(
    tmp_path: Path,
) -> None:
    work_order = _work_order()
    score_report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    score_report_path = _write_json(tmp_path / "repair_campaign_score_report.json", score_report)

    queue = build_repair_cascade_mlx_probe_queue(
        repo_root=REPO_ROOT,
        source_payload=score_report,
        source_payload_path=score_report_path,
        results_root=tmp_path / "results",
        queue_id="cascade_c_probe_unit",
        posterior_path=tmp_path / "repair_campaign_stackability_posterior.jsonl",
        posterior_lock_path=tmp_path / ".repair_campaign_stackability_posterior.lock",
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["metadata"]["schema"] == REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA
    assert queue["metadata"]["structural_repair_cascade_count"] == 1
    assert queue["metadata"]["repair_family_campaign_count_per_cascade"] == 5
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["component_response_axis"] == (
        "[macOS-MLX research-signal]"
    )
    assert experiment["metadata"]["probe_result_schema"] == (
        REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA
    )
    assert experiment["metadata"]["learning_signal_schema"] == (
        REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA
    )
    assert experiment["metadata"]["posterior_append_report_schema"] == (
        REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
    )
    assert experiment["steps"][0]["command"][1] == (
        "tools/build_repair_cascade_mlx_probe_spec.py"
    )
    result_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "record_repair_cascade_mlx_probe_result"
    )
    assert result_step["command"][1] == "tools/run_repair_cascade_mlx_probe.py"
    learning_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "build_repair_cascade_mlx_learning_signal"
    )
    assert learning_step["command"][1] == (
        "tools/build_repair_cascade_mlx_learning_signal.py"
    )

    queue_path = _write_json(tmp_path / "repair_cascade_mlx_probe_queue.json", queue)
    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr

    worker_out = tmp_path / "worker_result.json"
    worker = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "run-worker",
            "--execute",
            "--max-steps",
            "5",
            "--max-experiments",
            "1",
            "--output",
            str(worker_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert worker.returncode == 0, worker.stderr
    result = json.loads(worker_out.read_text(encoding="utf-8"))
    assert result["failure_count"] == 0
    spec_path = REPO_ROOT / experiment["metadata"]["probe_spec_path"]
    assert spec_path.is_file()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["schema"] == REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA
    assert spec["repair_family_campaign_count"] == 5
    assert "local_mlx_response_path:missing_or_unverified" in (
        spec["missing_local_mlx_artifacts"]
    )
    result_path = REPO_ROOT / experiment["metadata"]["probe_result_path"]
    assert result_path.is_file()
    probe_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert probe_result["schema"] == REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA
    assert probe_result["learning_signal_kind"] == "blocked_repair_cascade_mlx_probe"
    signal_path = REPO_ROOT / experiment["metadata"]["learning_signal_path"]
    assert signal_path.is_file()
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert signal["schema"] == REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA
    posterior_path = Path(experiment["metadata"]["posterior_path"])
    assert posterior_path.is_file()
    posterior_lines = posterior_path.read_text(encoding="utf-8").splitlines()
    assert len(posterior_lines) == 1
    posterior_row = json.loads(posterior_lines[0])
    assert posterior_row["typed_response_id"] == signal["typed_response_id"]


def test_repair_cascade_mlx_probe_queue_cli_writes_queue(tmp_path: Path) -> None:
    work_order_path = _write_json(
        tmp_path / "repair_budget_waterfill_work_order.json",
        _work_order(),
    )
    queue_path = tmp_path / "repair_cascade_mlx_probe_queue.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_repair_cascade_mlx_probe_queue.py",
            "--source-payload",
            str(work_order_path),
            "--probe-queue-out",
            str(queue_path),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "cascade_c_probe_cli_unit",
            "--posterior-path",
            str(tmp_path / "repair_campaign_stackability_posterior.jsonl"),
            "--posterior-lock-path",
            str(tmp_path / ".repair_campaign_stackability_posterior.lock"),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert queue["metadata"]["structural_repair_cascade_count"] == 1


def test_repair_cascade_mlx_probe_result_cli_writes_result(tmp_path: Path) -> None:
    work_order = _work_order()
    spec = build_repair_cascade_mlx_probe_spec(
        source_payload=work_order,
        source_payload_path=tmp_path / "repair_budget_waterfill_work_order.json",
        cascade_id="cascade_c_posenet_null_segnet_region_selector_codec",
        repo_root=tmp_path,
    )
    spec_path = _write_json(tmp_path / "repair_cascade_mlx_probe_spec.json", spec)
    result_path = tmp_path / "repair_cascade_mlx_probe_result.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_repair_cascade_mlx_probe.py",
            "--probe-spec",
            str(spec_path),
            "--output",
            str(result_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA
    assert payload["local_mlx_probe_execution_ready"] is False


def test_repair_cascade_mlx_learning_signal_cli_writes_signal(tmp_path: Path) -> None:
    work_order = _work_order()
    spec = build_repair_cascade_mlx_probe_spec(
        source_payload=work_order,
        source_payload_path=tmp_path / "repair_budget_waterfill_work_order.json",
        cascade_id="cascade_c_posenet_null_segnet_region_selector_codec",
        repo_root=tmp_path,
    )
    result_payload = build_repair_cascade_mlx_probe_result(
        probe_spec=spec,
        probe_spec_path=tmp_path / "repair_cascade_mlx_probe_spec.json",
        repo_root=tmp_path,
    )
    result_path = _write_json(
        tmp_path / "repair_cascade_mlx_probe_result.json",
        result_payload,
    )
    signal_path = tmp_path / "repair_cascade_mlx_learning_signal.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_repair_cascade_mlx_learning_signal.py",
            "--probe-result",
            str(result_path),
            "--learning-signal-out",
            str(signal_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert signal["schema"] == REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA
    assert signal["family_id"] == "entropy_position_cascade"
