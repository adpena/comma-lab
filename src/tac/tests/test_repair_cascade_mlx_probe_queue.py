# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA
from comma_lab.scheduler.repair_cascade_mlx_probe_queue import (
    REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA,
    REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
    build_repair_cascade_mlx_probe_queue,
    build_repair_cascade_mlx_probe_spec,
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
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["metadata"]["schema"] == REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA
    assert queue["metadata"]["structural_repair_cascade_count"] == 1
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["component_response_axis"] == (
        "[macOS-MLX research-signal]"
    )
    assert experiment["steps"][0]["command"][1] == (
        "tools/build_repair_cascade_mlx_probe_spec.py"
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
            "2",
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
    assert "local_mlx_response_path:missing_or_unverified" in (
        spec["missing_local_mlx_artifacts"]
    )


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
