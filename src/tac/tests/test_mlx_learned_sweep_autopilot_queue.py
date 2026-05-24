# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import (
    MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA,
    MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA,
    MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA,
)
from comma_lab.scheduler import (
    build_mlx_learned_sweep_autopilot_batch_queue as package_build_batch_queue,
)
from comma_lab.scheduler import (
    build_mlx_learned_sweep_autopilot_queue as package_build_queue,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.mlx_learned_sweep_autopilot_queue import (
    build_mlx_learned_sweep_autopilot_batch_queue,
    build_mlx_learned_sweep_autopilot_queue,
)
from tac.optimization.mlx_dynamic_learned_sweep import (
    build_mlx_dynamic_learned_sweep_plan,
)
from tac.optimization.mlx_learned_sweep_batch_roots import (
    MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA,
    MLXLearnedSweepBatchRootError,
    build_mlx_learned_sweep_autopilot_batch_root_plan,
)
from tac.tests.test_mlx_dynamic_learned_sweep_local_actuator import (
    INCUMBENT_SCORE,
    _candidate_payload,
    _plan,
    _selection,
)

REPO = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    selection = _selection(tmp_path)
    candidate_payload = _candidate_payload(selection)
    plan = _plan(selection)
    return {
        "plan": _write_json(tmp_path / "plan.json", plan),
        "selection": _write_json(tmp_path / "selection.json", selection),
        "candidate_payload": _write_json(tmp_path / "candidate_payload.json", candidate_payload),
        "observations": tmp_path / "observations.jsonl",
        "output_root": tmp_path / "autopilot_runs",
    }


def _multi_pass_plan(selection: dict) -> dict:
    return build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[_candidate_payload(selection)],
        execution_configs=[
            {
                "config_id": "mlx_local_response",
                "substrate": "[macOS-MLX research-signal]",
                "execution_layer": "local_mlx",
                "cost_units": 1.0,
                "signal_quality": 0.45,
                "parallelizable": True,
                "exact_eval_candidate": False,
            }
        ],
        optimization_passes=[
            {
                "pass_id": "micro",
                "scale": "micro",
                "recursive_stage": 1,
                "sample_budget": 8,
                "cost_multiplier": 1.0,
                "expected_improvement_weight": 0.75,
                "exploration_weight": 1.25,
                "freeze_candidate": False,
            },
            {
                "pass_id": "intermediate",
                "scale": "intermediate",
                "recursive_stage": 2,
                "sample_budget": 24,
                "cost_multiplier": 2.0,
                "expected_improvement_weight": 1.25,
                "exploration_weight": 1.0,
                "freeze_candidate": False,
            },
            {
                "pass_id": "macro",
                "scale": "macro",
                "recursive_stage": 3,
                "sample_budget": 48,
                "cost_multiplier": 4.0,
                "expected_improvement_weight": 2.0,
                "exploration_weight": 0.75,
                "freeze_candidate": True,
            },
        ],
        top_k=3,
    )


def _ready_mlx_rows(plan: dict) -> list[dict]:
    return [
        row
        for row in plan["ranked_sweep_rows"]
        if row.get("schema") == "mlx_dynamic_learned_sweep_row.v1"
        and row.get("ready_for_local_sweep") is True
        and row.get("sweep_config_id") == "mlx_local_response"
    ]


def _local_mlx_runtime_summary(
    queue_candidate_ids: list[str],
    *,
    elapsed_seconds: float,
    **overrides: object,
) -> dict:
    payload = {
        "schema": "mlx_dynamic_learned_sweep_local_autopilot.v1",
        "elapsed_seconds": elapsed_seconds,
        "executed_filter_match": True,
        "executed_filter_violation_count": 0,
        "executed_queue_candidate_id_count": len(queue_candidate_ids),
        "executed_row_count": len(queue_candidate_ids),
        "executed_queue_candidate_id_set": sorted(queue_candidate_ids),
        "score_claim": False,
        "score_claim_valid": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    payload.update(overrides)
    return payload


def _worker_runtime_result(
    queue_candidate_ids: list[str],
    *,
    elapsed_seconds: float,
) -> dict:
    command = ["tools/run_mlx_dynamic_learned_sweep_autopilot.py"]
    for queue_candidate_id in queue_candidate_ids:
        command.extend(["--queue-candidate-id", queue_candidate_id])
    return {
        "schema": "experiment_queue_worker_result.v1",
        "step_results": [
            {
                "succeeded": True,
                "timed_out": False,
                "failed_postconditions": [],
                "postcondition_errors": [],
                "elapsed_seconds": elapsed_seconds,
                "command": command,
            }
        ],
    }


def _write_runtime_queue_state(
    path: Path,
    queue_candidate_ids: list[str],
    *,
    elapsed_seconds: float,
    queue_id: str = "prior_queue",
    step_id: str = "run_mlx_learned_sweep_autopilot",
    resource_kind: str = "local_mlx",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["tools/run_mlx_dynamic_learned_sweep_autopilot.py"]
    for queue_candidate_id in queue_candidate_ids:
        command.extend(["--queue-candidate-id", queue_candidate_id])
    payload = {
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "succeeded": True,
        "timed_out": False,
        "failed_postconditions": [],
        "postcondition_errors": [],
        "resource_kind": resource_kind,
        "telemetry": {
            "schema": "experiment_queue_step_telemetry.v1",
            "artifact_record_count": 1,
            "artifact_records": [
                {
                    "path": "local_mlx_autopilot_summary.json",
                    "exists": True,
                    "is_file": True,
                    "bytes": 128,
                }
            ],
            "recursive_truncated": False,
        },
    }
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE queue_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_utc TEXT NOT NULL,
              queue_id TEXT NOT NULL,
              experiment_id TEXT,
              step_id TEXT,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE step_state (
              queue_id TEXT NOT NULL,
              experiment_id TEXT NOT NULL,
              step_id TEXT NOT NULL,
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              updated_at_utc TEXT NOT NULL,
              last_event_json TEXT,
              resource_kind TEXT,
              PRIMARY KEY (queue_id, experiment_id, step_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO step_state(
              queue_id,
              experiment_id,
              step_id,
              status,
              attempts,
              updated_at_utc,
              last_event_json,
              resource_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                "prior_experiment",
                step_id,
                "succeeded",
                1,
                "2026-05-24T12:00:00Z",
                json.dumps(payload),
                resource_kind,
            ),
        )
        conn.execute(
            """
            INSERT INTO queue_events(
              ts_utc,
              queue_id,
              experiment_id,
              step_id,
              event_type,
              payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-05-24T12:00:00Z",
                queue_id,
                "prior_experiment",
                step_id,
                "step_succeeded",
                json.dumps(payload),
            ),
        )
    return path


def test_mlx_learned_sweep_autopilot_queue_is_exported() -> None:
    assert (
        MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA
        == "mlx_dynamic_learned_sweep_autopilot_batch_queue_plan.v1"
    )
    assert (
        MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA
        == "mlx_dynamic_learned_sweep_autopilot_queue_plan.v1"
    )
    assert package_build_queue is build_mlx_learned_sweep_autopilot_queue
    assert package_build_batch_queue is build_mlx_learned_sweep_autopilot_batch_queue


def test_builds_auto_batch_root_plan_from_pass_utility(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)

    root_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "auto_roots.jsonl",
        root_count=2,
        rows_per_root=1,
        max_new_observations=1,
        rows_per_replan=1,
        chain_steps=1,
        device="gpu",
        allow_gpu_research_signal=True,
        source_artifact_root=tmp_path,
        batch_pairs=1,
    )

    assert root_plan["schema"] == MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA
    assert root_plan["score_claim"] is False
    assert root_plan["candidate_specific_filter_supported"] is True
    assert root_plan["queue_candidate_filter_supported"] is True
    assert root_plan["queue_candidate_disjoint_guaranteed"] is True
    assert root_plan["selected_root_count"] == 2
    queue_candidate_ids = [
        root["queue_candidate_ids"][0] for root in root_plan["selected_roots"]
    ]
    assert len(set(queue_candidate_ids)) == 2
    utilities = [root["root_utility"] for root in root_plan["selected_roots"]]
    assert utilities == sorted(utilities, reverse=True)
    assert {
        spec["queue_candidate_ids"][0] for spec in root_plan["run_specs"]
    } == set(queue_candidate_ids)
    assert len(
        {spec["observation_jsonl"] for spec in root_plan["run_specs"]}
    ) == 2
    for root in root_plan["selected_roots"]:
        assert root["root_granularity"] == "queue_candidate_set"
        assert root["row_refs"][0]["score_claim"] is False


def test_auto_batch_root_plan_groups_multiple_exact_rows_without_pass_filter(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)

    root_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "auto_roots.jsonl",
        root_count=1,
        rows_per_root=2,
        max_new_observations=2,
        rows_per_replan=2,
        chain_steps=1,
        device="gpu",
        allow_gpu_research_signal=True,
        source_artifact_root=tmp_path,
        batch_pairs=1,
    )

    root = root_plan["selected_roots"][0]
    run_spec = root_plan["run_specs"][0]
    assert root["selected_row_count"] == 2
    assert len(root["queue_candidate_ids"]) == 2
    assert run_spec["queue_candidate_ids"] == root["queue_candidate_ids"]
    if len(root["optimization_pass_ids"]) > 1:
        assert root["optimization_pass_id"] is None
        assert root["representative_optimization_pass_id"]
        assert "optimization_pass_id" not in run_spec


def test_auto_batch_root_plan_adaptive_rows_balances_positive_utility(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)

    root_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "auto_roots.jsonl",
        root_count=2,
        rows_per_root=None,
        adaptive_rows_per_root=True,
        max_new_observations=3,
        rows_per_replan=3,
        chain_steps=1,
        device="gpu",
        allow_gpu_research_signal=True,
        source_artifact_root=tmp_path,
        batch_pairs=1,
    )

    assert root_plan["adaptive_rows_per_root"] is True
    assert root_plan["rows_per_root"] is None
    assert root_plan["max_rows_per_root"] == 3
    assert root_plan["rows_per_root_policy"] == (
        "adaptive_positive_utility_cost_balanced_waterfill"
    )
    assert root_plan["row_grouping"]["strategy"] == (
        "positive_utility_cost_balanced_waterfill"
    )
    root_size_counts = root_plan["row_grouping"]["root_size_counts"]
    assert sum(root_size_counts) == root_plan["selected_total_row_count"]
    assert root_plan["selected_total_row_count"] == min(
        root_plan["row_grouping"]["eligible_positive_utility_row_count"],
        2 * root_plan["max_rows_per_root"],
    )
    assert max(root_size_counts) <= 3
    assert max(root_size_counts) - min(root_size_counts) <= 1
    for run_spec, root in zip(
        root_plan["run_specs"],
        root_plan["selected_roots"],
        strict=True,
    ):
        assert len(run_spec["queue_candidate_ids"]) == root["selected_row_count"]
        assert len(set(run_spec["queue_candidate_ids"])) == len(
            run_spec["queue_candidate_ids"]
        )
        if len(root["optimization_pass_ids"]) > 1:
            assert root["optimization_pass_id"] is None
            assert root["representative_optimization_pass_id"]
            assert "optimization_pass_id" not in run_spec


def test_auto_batch_root_plan_adaptive_uses_runtime_telemetry(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    ready_rows = _ready_mlx_rows(plan)
    for row in ready_rows:
        row["cost_units"] = 1.0
    telemetry = [
        _local_mlx_runtime_summary(
            [ready_rows[0]["queue_candidate_id"]],
            elapsed_seconds=100.0,
        ),
        _local_mlx_runtime_summary(
            [ready_rows[1]["queue_candidate_id"], ready_rows[2]["queue_candidate_id"]],
            elapsed_seconds=2.0,
        ),
    ]

    static_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "static_roots.jsonl",
        root_count=2,
        rows_per_root=None,
        adaptive_rows_per_root=True,
        max_new_observations=3,
        rows_per_replan=3,
    )
    runtime_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "runtime_roots.jsonl",
        root_count=2,
        rows_per_root=None,
        adaptive_rows_per_root=True,
        max_new_observations=3,
        rows_per_replan=3,
        runtime_telemetry_payloads=telemetry,
    )

    static_groups = [
        root["queue_candidate_ids"] for root in static_plan["selected_roots"]
    ]
    runtime_groups = [
        root["queue_candidate_ids"] for root in runtime_plan["selected_roots"]
    ]
    assert runtime_groups != static_groups
    assert runtime_plan["runtime_telemetry_used"] is True
    assert runtime_plan["runtime_telemetry_key_count"] == 3
    assert runtime_plan["runtime_telemetry_observation_count"] == 3
    assert runtime_plan["runtime_telemetry"]["runtime_telemetry_assignment_policy"] == (
        "elapsed_seconds_even_split_by_queue_candidate"
    )
    assert runtime_plan["runtime_telemetry"][
        "runtime_telemetry_even_split_observation_count"
    ] == 1
    assert runtime_plan["row_grouping"]["strategy"] == (
        "positive_utility_runtime_balanced_waterfill"
    )
    assert runtime_plan["selection_policy"]["group_by"] == (
        "positive_utility_runtime_balanced_buckets"
    )
    first_ref_by_queue_id = {
        row_ref["queue_candidate_id"]: row_ref
        for root in runtime_plan["selected_roots"]
        for row_ref in root["row_refs"]
    }
    assert first_ref_by_queue_id[ready_rows[0]["queue_candidate_id"]][
        "runtime_cost_estimate"
    ] == 100.0
    assert first_ref_by_queue_id[ready_rows[1]["queue_candidate_id"]][
        "runtime_cost_estimate"
    ] == 1.0
    assert first_ref_by_queue_id[ready_rows[0]["queue_candidate_id"]][
        "runtime_cost_source"
    ] == "telemetry_seconds_per_queue_candidate"
    for key in (
        "score_claim",
        "score_claim_valid",
        "rank_or_kill_eligible",
        "promotable",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
    ):
        assert runtime_plan[key] is False
        assert runtime_plan["runtime_telemetry"][key] is False


def test_auto_batch_root_plan_runtime_telemetry_averages_duplicate_samples(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    queue_candidate_id = _ready_mlx_rows(plan)[0]["queue_candidate_id"]

    root_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "runtime_roots.jsonl",
        root_count=1,
        rows_per_root=1,
        adaptive_rows_per_root=True,
        max_new_observations=1,
        rows_per_replan=1,
        runtime_telemetry_payloads=[
            _local_mlx_runtime_summary([queue_candidate_id], elapsed_seconds=10.0),
            _local_mlx_runtime_summary([queue_candidate_id], elapsed_seconds=30.0),
        ],
    )

    assert root_plan["runtime_telemetry"]["runtime_telemetry_key_count"] == 1
    assert root_plan["runtime_telemetry"]["runtime_source_count_by_queue_candidate_id"][
        queue_candidate_id
    ] == 2
    assert root_plan["runtime_telemetry"]["runtime_seconds_by_queue_candidate_id"][
        queue_candidate_id
    ] == 20.0


def test_auto_batch_root_plan_worker_runtime_telemetry_even_splits_elapsed_time(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    queue_candidate_ids = [
        row["queue_candidate_id"] for row in _ready_mlx_rows(plan)[:2]
    ]

    root_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
        plan,
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=tmp_path / "runtime_roots.jsonl",
        root_count=2,
        rows_per_root=1,
        adaptive_rows_per_root=True,
        max_new_observations=1,
        rows_per_replan=1,
        runtime_telemetry_payloads=[
            _worker_runtime_result(queue_candidate_ids, elapsed_seconds=40.0)
        ],
    )

    assert root_plan["runtime_telemetry"]["runtime_telemetry_schema_counts"] == {
        "experiment_queue_worker_result.v1": 1
    }
    assert root_plan["runtime_telemetry"][
        "runtime_telemetry_even_split_observation_count"
    ] == 1
    for queue_candidate_id in queue_candidate_ids:
        assert root_plan["runtime_telemetry"][
            "runtime_seconds_by_queue_candidate_id"
        ][queue_candidate_id] == 20.0


@pytest.mark.parametrize(
    ("telemetry", "message"),
    [
        ({"schema": "unknown"}, "unsupported"),
        (
            _local_mlx_runtime_summary(["qc"], elapsed_seconds=0.0),
            "elapsed_seconds must be > 0",
        ),
        (
            _local_mlx_runtime_summary(
                ["qc"],
                elapsed_seconds=1.0,
                executed_filter_match=False,
            ),
            "executed_filter_match",
        ),
        (
            _local_mlx_runtime_summary(
                ["qc"],
                elapsed_seconds=1.0,
                executed_filter_violation_count=1,
            ),
            "executed_filter_violation_count",
        ),
        (
            _local_mlx_runtime_summary(["qc"], elapsed_seconds=1.0, score_claim=True),
            "score_claim",
        ),
        (
            _local_mlx_runtime_summary(
                ["qc"],
                elapsed_seconds=1.0,
                ready_for_exact_eval_dispatch=True,
            ),
            "ready_for_exact",
        ),
    ],
)
def test_auto_batch_root_plan_runtime_telemetry_fails_closed(
    tmp_path: Path,
    telemetry: dict,
    message: str,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)

    with pytest.raises(MLXLearnedSweepBatchRootError, match=message):
        build_mlx_learned_sweep_autopilot_batch_root_plan(
            plan,
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=tmp_path / "runtime_roots.jsonl",
            root_count=1,
            rows_per_root=1,
            adaptive_rows_per_root=True,
            max_new_observations=1,
            rows_per_replan=1,
            runtime_telemetry_payloads=[telemetry],
        )


def test_auto_batch_root_plan_refuses_duplicate_queue_candidate_ids(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    duplicate_source = next(
        row
        for row in plan["ranked_sweep_rows"]
        if row.get("schema") == "mlx_dynamic_learned_sweep_row.v1"
        and row.get("ready_for_local_sweep") is True
        and row.get("sweep_config_id") == "mlx_local_response"
    )
    plan["ranked_sweep_rows"].insert(0, dict(duplicate_source))

    with pytest.raises(MLXLearnedSweepBatchRootError, match="queue_candidate_ids"):
        build_mlx_learned_sweep_autopilot_batch_root_plan(
            plan,
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=tmp_path / "auto_roots.jsonl",
            root_count=1,
            rows_per_root=2,
            max_new_observations=2,
            rows_per_replan=2,
        )


def test_auto_batch_root_plan_adaptive_refuses_no_positive_utility(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    for row in plan["ranked_sweep_rows"]:
        if (
            row.get("schema") == "mlx_dynamic_learned_sweep_row.v1"
            and row.get("ready_for_local_sweep") is True
            and row.get("sweep_config_id") == "mlx_local_response"
        ):
            row["acquisition_value"] = 0.0
            row["learning_value_per_cost"] = 0.0

    with pytest.raises(MLXLearnedSweepBatchRootError, match="positive-utility"):
        build_mlx_learned_sweep_autopilot_batch_root_plan(
            plan,
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=tmp_path / "auto_roots.jsonl",
            root_count=2,
            rows_per_root=None,
            adaptive_rows_per_root=True,
            max_new_observations=2,
            rows_per_replan=2,
        )


def test_auto_batch_root_plan_refuses_truthy_plan_authority(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    plan["ready_for_exact_eval_dispatch"] = True

    with pytest.raises(MLXLearnedSweepBatchRootError, match="ready_for_exact"):
        build_mlx_learned_sweep_autopilot_batch_root_plan(
            plan,
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=tmp_path / "auto_roots.jsonl",
            root_count=2,
        )


def test_auto_batch_root_plan_requires_explicit_false_row_authority(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    del plan["ranked_sweep_rows"][0]["score_claim"]

    with pytest.raises(MLXLearnedSweepBatchRootError, match="score_claim"):
        build_mlx_learned_sweep_autopilot_batch_root_plan(
            plan,
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=tmp_path / "auto_roots.jsonl",
            root_count=2,
        )


def test_builds_queue_owned_autopilot_step(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    queue = build_mlx_learned_sweep_autopilot_queue(
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=paths["observations"],
        queue_id="mlx_autopilot_fixture",
        repo_root=tmp_path,
        local_cpu_concurrency=4,
        local_mlx_concurrency=2,
        timeout_seconds=900,
        max_iterations=3,
        max_new_observations=5,
        rows_per_replan=2,
        optimization_pass_id="micro",
        source_artifact_root=tmp_path,
        device="gpu",
        allow_gpu_research_signal=True,
        replan_top_k=64,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"] == {"local_cpu": 4, "local_mlx": 2}
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["schema"] == MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    step = experiment["steps"][0]
    assert step["id"] == "run_mlx_learned_sweep_autopilot"
    assert step["resources"]["kind"] == "local_mlx"
    assert step["timeout_seconds"] == 900
    command = step["command"]
    assert command[:2] == [
        ".venv/bin/python",
        "tools/run_mlx_dynamic_learned_sweep_autopilot.py",
    ]
    assert "--allow-gpu-research-signal" in command
    assert command[command.index("--max-iterations") + 1] == "3"
    assert command[command.index("--max-new-observations") + 1] == "5"
    assert command[command.index("--rows-per-replan") + 1] == "2"
    assert command[command.index("--optimization-pass-id") + 1] == "micro"
    assert command[command.index("--replan-top-k") + 1] == "64"
    assert step["telemetry"]["recursive"] is True
    assert any(
        condition["type"] == "json_false_authority"
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "jsonl_false_authority"
        and condition["schema_equals"] == "mlx_dynamic_sweep_observation.v1"
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_completion_contract"
        and "new_observation_row_count" in condition["required_positive_int"]
        for condition in step["postconditions"]
    )


def test_autopilot_queue_threads_exact_queue_candidate_filter(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    queue_candidate_id = "mlx_scorer_response:window:10:11::mlx_local_response::micro"

    queue = build_mlx_learned_sweep_autopilot_queue(
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=paths["observations"],
        queue_id="mlx_autopilot_filter_fixture",
        repo_root=tmp_path,
        max_iterations=1,
        max_new_observations=1,
        rows_per_replan=1,
        candidate_ids=["mlx_scorer_response:window:10:11"],
        queue_candidate_ids=[queue_candidate_id],
        source_artifact_root=tmp_path,
        device="gpu",
        allow_gpu_research_signal=True,
    )

    experiment = queue["experiments"][0]
    assert experiment["metadata"]["candidate_id_filters"] == [
        "mlx_scorer_response:window:10:11"
    ]
    assert experiment["metadata"]["queue_candidate_id_filters"] == [queue_candidate_id]
    command = experiment["steps"][0]["command"]
    assert command[command.index("--candidate-id") + 1] == (
        "mlx_scorer_response:window:10:11"
    )
    assert command[command.index("--queue-candidate-id") + 1] == queue_candidate_id
    contracts = [
        condition
        for condition in experiment["steps"][0]["postconditions"]
        if condition["type"] == "json_completion_contract"
    ]
    assert contracts[0]["required_equals"]["executed_filter_match"] is True
    assert contracts[0]["required_equals"]["executed_unique_queue_candidate_id"] == (
        queue_candidate_id
    )


def test_autopilot_queue_threads_multi_row_queue_candidate_filters(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)
    queue_candidate_ids = [
        row["queue_candidate_id"]
        for row in plan["ranked_sweep_rows"]
        if row["sweep_config_id"] == "mlx_local_response"
    ][:2]

    queue = build_mlx_learned_sweep_autopilot_queue(
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=paths["observations"],
        queue_id="mlx_autopilot_multi_filter_fixture",
        repo_root=tmp_path,
        max_iterations=1,
        max_new_observations=2,
        rows_per_replan=2,
        queue_candidate_ids=queue_candidate_ids,
        source_artifact_root=tmp_path,
        device="gpu",
        allow_gpu_research_signal=True,
    )

    command = queue["experiments"][0]["steps"][0]["command"]
    assert command.count("--queue-candidate-id") == 2
    contracts = [
        condition
        for condition in queue["experiments"][0]["steps"][0]["postconditions"]
        if condition["type"] == "json_completion_contract"
    ]
    required_equals = contracts[0]["required_equals"]
    assert required_equals["executed_queue_candidate_id_count"] == 2
    assert required_equals["executed_queue_candidate_id_set"] == sorted(
        queue_candidate_ids
    )
    assert "executed_unique_queue_candidate_id" not in required_equals


def test_autopilot_queue_builds_macos_cpu_advisory_step(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    queue_candidate_id = (
        "mlx_scorer_response:window:10:11::macos_cpu_advisory::micro"
    )

    queue = build_mlx_learned_sweep_autopilot_queue(
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=paths["observations"],
        queue_id="cpu_advisory_autopilot_fixture",
        repo_root=tmp_path,
        local_cpu_concurrency=3,
        local_mlx_concurrency=1,
        max_iterations=1,
        max_new_observations=1,
        rows_per_replan=1,
        sweep_config_id="macos_cpu_advisory",
        queue_candidate_ids=[queue_candidate_id],
        source_artifact_root=tmp_path,
        device="cpu",
        allow_gpu_research_signal=False,
    )

    assert queue["controls"]["max_concurrency"] == {"local_cpu": 3, "local_mlx": 1}
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["sweep_config_id"] == "macos_cpu_advisory"
    step = experiment["steps"][0]
    assert step["resources"]["kind"] == "local_cpu"
    command = step["command"]
    assert command[command.index("--sweep-config-id") + 1] == "macos_cpu_advisory"
    assert command[command.index("--device") + 1] == "cpu"
    assert "--allow-gpu-research-signal" not in command
    contracts = [
        condition
        for condition in step["postconditions"]
        if condition["type"] == "json_completion_contract"
    ]
    assert contracts[0]["required_equals"]["sweep_config_id"] == "macos_cpu_advisory"
    assert contracts[0]["required_equals"]["executed_unique_queue_candidate_id"] == (
        queue_candidate_id
    )


def test_autopilot_queue_rejects_macos_cpu_advisory_gpu_queue(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)

    with pytest.raises(ExperimentQueueError, match="device=cpu"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="cpu_advisory_autopilot_fixture",
            repo_root=tmp_path,
            sweep_config_id="macos_cpu_advisory",
            device="gpu",
            allow_gpu_research_signal=True,
        )


def test_autopilot_queue_rejects_macos_cpu_advisory_without_selection_paths(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    for key in (
        "local_cpu_advisory_source_path",
        "window_baseline_local_cpu_advisory_source_path",
    ):
        selection["selected_rows"][0].pop(key)
    _write_json(paths["selection"], selection)

    with pytest.raises(ExperimentQueueError, match="advisory path is required"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="cpu_advisory_missing_selection_paths_fixture",
            repo_root=tmp_path,
            sweep_config_id="macos_cpu_advisory",
            device="cpu",
            allow_gpu_research_signal=False,
        )


def test_autopilot_queue_refuses_duplicate_queue_candidate_rows_before_execution(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    duplicate_source = next(
        row
        for row in plan["ranked_sweep_rows"]
        if row.get("schema") == "mlx_dynamic_learned_sweep_row.v1"
        and row.get("ready_for_local_sweep") is True
        and row.get("sweep_config_id") == "mlx_local_response"
    )
    plan["ranked_sweep_rows"].insert(0, dict(duplicate_source))
    _write_json(paths["plan"], plan)

    with pytest.raises(ExperimentQueueError, match="duplicate ready rows"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="mlx_autopilot_duplicate_filter_fixture",
            repo_root=tmp_path,
            max_iterations=1,
            max_new_observations=1,
            rows_per_replan=1,
            queue_candidate_ids=[str(duplicate_source["queue_candidate_id"])],
            source_artifact_root=tmp_path,
            device="gpu",
            allow_gpu_research_signal=True,
        )


def test_autopilot_queue_rejects_chained_exact_queue_candidate_filter(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)

    with pytest.raises(ExperimentQueueError, match="queue_candidate_ids require"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="mlx_autopilot_filter_fixture",
            repo_root=tmp_path,
            chain_steps=2,
            max_iterations=1,
            max_new_observations=1,
            rows_per_replan=1,
            queue_candidate_ids=[
                "mlx_scorer_response:window:10:11::mlx_local_response::micro"
            ],
            device="gpu",
            allow_gpu_research_signal=True,
        )


def test_autopilot_queue_rejects_truthy_plan_authority(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan["score_claim"] = True
    _write_json(paths["plan"], plan)

    with pytest.raises(ExperimentQueueError, match="score_claim"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="mlx_autopilot_fixture",
            repo_root=tmp_path,
            allow_gpu_research_signal=True,
        )


def test_autopilot_queue_requires_gpu_research_flag_for_gpu(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    with pytest.raises(ExperimentQueueError, match="allow-gpu-research-signal"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="mlx_autopilot_fixture",
            repo_root=tmp_path,
            device="gpu",
            allow_gpu_research_signal=False,
        )


def test_autopilot_queue_rejects_plan_without_ready_mlx_rows(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    for row in plan["ranked_sweep_rows"]:
        if row["sweep_config_id"] == "mlx_local_response":
            row["ready_for_local_sweep"] = False
    _write_json(paths["plan"], plan)

    with pytest.raises(ExperimentQueueError, match="no ready mlx_local_response"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="mlx_autopilot_fixture",
            repo_root=tmp_path,
            device="gpu",
            allow_gpu_research_signal=True,
        )


def test_builds_chained_queue_owned_autopilot_steps(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    queue = build_mlx_learned_sweep_autopilot_queue(
        plan_path=paths["plan"],
        selection_path=paths["selection"],
        candidate_payload_paths=[paths["candidate_payload"]],
        incumbent_score=INCUMBENT_SCORE,
        output_root=paths["output_root"],
        observation_jsonl=paths["observations"],
        queue_id="mlx_autopilot_chain_fixture",
        repo_root=tmp_path,
        device="gpu",
        allow_gpu_research_signal=True,
        max_iterations=1,
        max_new_observations=1,
        chain_steps=3,
        replan_top_k=64,
    )

    experiment = queue["experiments"][0]
    assert experiment["metadata"]["chain_steps"] == 3
    assert [step["id"] for step in experiment["steps"]] == [
        "run_mlx_learned_sweep_autopilot_0001",
        "run_mlx_learned_sweep_autopilot_0002",
        "run_mlx_learned_sweep_autopilot_0003",
    ]
    assert experiment["steps"][0]["requires"] == []
    assert experiment["steps"][1]["requires"] == [
        "run_mlx_learned_sweep_autopilot_0001"
    ]
    assert experiment["steps"][2]["requires"] == [
        "run_mlx_learned_sweep_autopilot_0002"
    ]
    first_command = experiment["steps"][0]["command"]
    second_command = experiment["steps"][1]["command"]
    assert first_command[first_command.index("--plan") + 1] == "plan.json"
    assert second_command[second_command.index("--plan") + 1].endswith(
        "step_0001/cycle_0001/learned_sweep_plan.after_cycle.json"
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["key"] == "schema"
        and condition["equals"] == "mlx_dynamic_learned_sweep_plan.v1"
        for condition in experiment["steps"][0]["postconditions"]
    )


def test_chained_autopilot_queue_requires_single_cycle_steps(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    with pytest.raises(ExperimentQueueError, match="max_iterations=1"):
        build_mlx_learned_sweep_autopilot_queue(
            plan_path=paths["plan"],
            selection_path=paths["selection"],
            candidate_payload_paths=[paths["candidate_payload"]],
            incumbent_score=INCUMBENT_SCORE,
            output_root=paths["output_root"],
            observation_jsonl=paths["observations"],
            queue_id="mlx_autopilot_chain_fixture",
            repo_root=tmp_path,
            device="gpu",
            allow_gpu_research_signal=True,
            max_iterations=2,
            chain_steps=2,
        )


def test_builds_multi_root_autopilot_batch_queue(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    batch = build_mlx_learned_sweep_autopilot_batch_queue(
        [
            {
                "run_id": "root_a",
                "plan_path": str(paths["plan"]),
                "selection_path": str(paths["selection"]),
                "candidate_payload_paths": [str(paths["candidate_payload"])],
                "incumbent_score": INCUMBENT_SCORE,
                "output_root": str(paths["output_root"] / "root_a"),
                "observation_jsonl": str(tmp_path / "root_a.jsonl"),
                "optimization_pass_id": "micro",
                "chain_steps": 2,
            },
            {
                "run_id": "root_b",
                "plan_path": str(paths["plan"]),
                "selection_path": str(paths["selection"]),
                "candidate_payload_paths": [str(paths["candidate_payload"])],
                "incumbent_score": INCUMBENT_SCORE,
                "output_root": str(paths["output_root"] / "root_b"),
                "observation_jsonl": str(tmp_path / "root_b.jsonl"),
                "optimization_pass_id": "micro",
                "chain_steps": 2,
            },
        ],
        queue_id="mlx_batch_fixture",
        repo_root=tmp_path,
        local_mlx_concurrency=2,
        max_iterations=1,
        max_new_observations=1,
        allow_gpu_research_signal=True,
    )

    assert batch["schema"] == "experiment_queue.v1"
    assert batch["controls"]["max_concurrency"]["local_mlx"] == 2
    assert len(batch["experiments"]) == 2
    assert {experiment["metadata"]["batch_run_id"] for experiment in batch["experiments"]} == {
        "root_a",
        "root_b",
    }
    for experiment in batch["experiments"]:
        assert experiment["metadata"]["batch_schema"] == (
            MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA
        )
        assert experiment["metadata"]["score_claim"] is False
        assert len(experiment["steps"]) == 2
        assert experiment["steps"][0]["resources"]["kind"] == "local_mlx"
        assert experiment["steps"][1]["requires"] == [
            "run_mlx_learned_sweep_autopilot_0001"
        ]


def test_batch_queue_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    spec = {
        "run_id": "same",
        "plan_path": str(paths["plan"]),
        "selection_path": str(paths["selection"]),
        "candidate_payload_paths": [str(paths["candidate_payload"])],
        "incumbent_score": INCUMBENT_SCORE,
        "output_root": str(paths["output_root"]),
        "observation_jsonl": str(tmp_path / "obs.jsonl"),
        "optimization_pass_id": "micro",
    }

    with pytest.raises(ExperimentQueueError, match="duplicate autopilot run_id"):
        build_mlx_learned_sweep_autopilot_batch_queue(
            [spec, dict(spec)],
            queue_id="mlx_batch_fixture",
            repo_root=tmp_path,
            allow_gpu_research_signal=True,
        )


def test_build_mlx_learned_sweep_autopilot_queue_cli(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    queue_path = tmp_path / "queue.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(paths["observations"]),
            "--source-artifact-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--max-new-observations",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue_id"] == "mlx_autopilot_cli_fixture"
    assert payload["experiments"][0]["metadata"]["score_claim"] is False
    summary = json.loads(result.stdout)
    assert summary["ready_for_exact_eval_dispatch"] is False


def test_build_mlx_learned_sweep_autopilot_batch_queue_cli(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    queue_path = tmp_path / "batch_queue.json"
    batch_spec = tmp_path / "batch_spec.json"
    _write_json(
        batch_spec,
        {
            "runs": [
                {
                    "run_id": "root_a",
                    "plan_path": str(paths["plan"]),
                    "selection_path": str(paths["selection"]),
                    "candidate_payload_paths": [str(paths["candidate_payload"])],
                    "incumbent_score": INCUMBENT_SCORE,
                    "output_root": str(paths["output_root"] / "root_a"),
                    "observation_jsonl": str(tmp_path / "root_a.jsonl"),
                },
                {
                    "run_id": "root_b",
                    "plan_path": str(paths["plan"]),
                    "selection_path": str(paths["selection"]),
                    "candidate_payload_paths": [str(paths["candidate_payload"])],
                    "incumbent_score": INCUMBENT_SCORE,
                    "output_root": str(paths["output_root"] / "root_b"),
                    "observation_jsonl": str(tmp_path / "root_b.jsonl"),
                },
            ]
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--batch-spec",
            str(batch_spec),
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_batch_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--optimization-pass-id",
            "micro",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(payload["experiments"]) == 2
    assert payload["controls"]["max_concurrency"]["local_mlx"] == 2
    for experiment in payload["experiments"]:
        command = experiment["steps"][0]["command"]
        assert command[command.index("--optimization-pass-id") + 1] == "micro"
    summary = json.loads(result.stdout)
    assert summary["experiment_count"] == 2


def test_build_mlx_learned_sweep_auto_batch_queue_cli(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)
    queue_path = tmp_path / "auto_batch_queue.json"
    root_plan_path = tmp_path / "auto_batch_roots.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "auto_batch_observations.jsonl"),
            "--auto-batch-root-count",
            "2",
            "--auto-batch-root-plan-output",
            str(root_plan_path),
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_auto_batch_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root_plan = json.loads(root_plan_path.read_text(encoding="utf-8"))
    assert root_plan["schema"] == MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA
    assert root_plan["selected_root_count"] == 2
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(payload["experiments"]) == 2
    selected_passes = {
        root["optimization_pass_id"] for root in root_plan["selected_roots"]
    }
    selected_queue_candidate_ids = {
        root["queue_candidate_ids"][0] for root in root_plan["selected_roots"]
    }
    command_passes = set()
    command_queue_candidate_ids = set()
    for experiment in payload["experiments"]:
        assert experiment["metadata"]["score_claim"] is False
        command = experiment["steps"][0]["command"]
        command_passes.add(command[command.index("--optimization-pass-id") + 1])
        command_queue_candidate_ids.add(
            command[command.index("--queue-candidate-id") + 1]
        )
    assert command_passes == selected_passes
    assert command_queue_candidate_ids == selected_queue_candidate_ids
    summary = json.loads(result.stdout)
    assert summary["auto_batch_selected_root_count"] == 2
    assert summary["ready_for_exact_eval_dispatch"] is False
    for key in (
        "score_claim",
        "score_claim_valid",
        "rank_or_kill_eligible",
        "promotable",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
    ):
        assert summary[key] is False


def test_build_mlx_learned_sweep_adaptive_auto_batch_queue_cli(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)
    queue_path = tmp_path / "adaptive_auto_batch_queue.json"
    root_plan_path = tmp_path / "adaptive_auto_batch_roots.json"
    telemetry_path = _write_json(
        tmp_path / "runtime_telemetry.json",
        _local_mlx_runtime_summary(
            [_ready_mlx_rows(plan)[0]["queue_candidate_id"]],
            elapsed_seconds=12.5,
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "adaptive_auto_batch_observations.jsonl"),
            "--auto-batch-root-count",
            "2",
            "--auto-batch-root-plan-output",
            str(root_plan_path),
            "--auto-batch-runtime-telemetry",
            str(telemetry_path),
            "--max-new-observations",
            "2",
            "--rows-per-replan",
            "2",
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_adaptive_auto_batch_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root_plan = json.loads(root_plan_path.read_text(encoding="utf-8"))
    assert root_plan["adaptive_rows_per_root"] is True
    assert root_plan["max_rows_per_root"] == 2
    assert root_plan["runtime_telemetry_used"] is True
    assert root_plan["runtime_telemetry_key_count"] == 1
    assert root_plan["runtime_cost_policy"] == (
        "telemetry_seconds_per_queue_candidate_with_planned_fallback"
    )
    root_size_counts = root_plan["row_grouping"]["root_size_counts"]
    assert sum(root_size_counts) == root_plan["selected_total_row_count"]
    assert max(root_size_counts) <= 2
    assert max(root_size_counts) - min(root_size_counts) <= 1
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(payload["experiments"]) == 2
    expected_counts = {
        root["run_id"]: root["selected_row_count"]
        for root in root_plan["selected_roots"]
    }
    for experiment in payload["experiments"]:
        command = experiment["steps"][0]["command"]
        assert command.count("--queue-candidate-id") == expected_counts[
            experiment["metadata"]["batch_run_id"]
        ]
        assert "--optimization-pass-id" not in command or command[
            command.index("--optimization-pass-id") + 1
        ] in {"micro", "intermediate", "macro"}
    summary = json.loads(result.stdout)
    assert summary["auto_batch_selected_root_count"] == 2
    assert summary["auto_batch_runtime_telemetry_used"] is True
    assert summary["auto_batch_runtime_telemetry_key_count"] == 1
    assert summary["auto_batch_runtime_cost_policy"] == (
        "telemetry_seconds_per_queue_candidate_with_planned_fallback"
    )
    assert summary["ready_for_exact_eval_dispatch"] is False
    for key in (
        "score_claim",
        "score_claim_valid",
        "rank_or_kill_eligible",
        "promotable",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
    ):
        assert summary[key] is False


def test_build_mlx_learned_sweep_adaptive_auto_batch_queue_cli_from_state_telemetry(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    ready_rows = _ready_mlx_rows(plan)
    _write_json(paths["plan"], plan)
    queue_path = tmp_path / "adaptive_auto_batch_queue_from_state.json"
    root_plan_path = tmp_path / "adaptive_auto_batch_roots_from_state.json"
    state_path = _write_runtime_queue_state(
        tmp_path / "prior_queue.sqlite",
        [ready_rows[0]["queue_candidate_id"]],
        elapsed_seconds=25.0,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "adaptive_state_observations.jsonl"),
            "--auto-batch-root-count",
            "2",
            "--auto-batch-root-plan-output",
            str(root_plan_path),
            "--auto-batch-runtime-telemetry-state",
            str(state_path),
            "--max-new-observations",
            "2",
            "--rows-per-replan",
            "2",
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_adaptive_auto_batch_state_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root_plan = json.loads(root_plan_path.read_text(encoding="utf-8"))
    assert root_plan["runtime_telemetry_used"] is True
    assert root_plan["runtime_telemetry_key_count"] == 1
    assert root_plan["runtime_telemetry"]["runtime_telemetry_schema_counts"] == {
        "experiment_queue_worker_result.v1": 1
    }
    assert root_plan["runtime_telemetry"]["runtime_telemetry_source_state_paths"] == [
        str(state_path)
    ]
    assert root_plan["runtime_telemetry"]["runtime_telemetry_source_queue_ids"] == [
        "prior_queue"
    ]
    assert root_plan["runtime_telemetry"]["runtime_seconds_by_queue_candidate_id"][
        ready_rows[0]["queue_candidate_id"]
    ] == 25.0
    summary = json.loads(result.stdout)
    assert summary["auto_batch_runtime_telemetry_state_count"] == 1
    assert summary["auto_batch_runtime_telemetry_used"] is True
    assert summary["auto_batch_runtime_telemetry_key_count"] == 1
    for key in (
        "score_claim",
        "score_claim_valid",
        "rank_or_kill_eligible",
        "promotable",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
    ):
        assert summary[key] is False


def test_build_mlx_learned_sweep_adaptive_auto_batch_queue_cli_discovers_state_telemetry(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    ready_rows = _ready_mlx_rows(plan)
    _write_json(paths["plan"], plan)
    queue_path = tmp_path / "adaptive_auto_batch_queue_discovered_state.json"
    root_plan_path = tmp_path / "adaptive_auto_batch_roots_discovered_state.json"
    state_dir = tmp_path / ".omx" / "state"
    incompatible_state = _write_runtime_queue_state(
        state_dir / "experiment_queue_mlx_learned_sweep_incompatible.sqlite",
        ["not-a-ready-queue-candidate"],
        elapsed_seconds=9.0,
        queue_id="mlx_learned_sweep_incompatible_queue",
    )
    wrong_lineage_state = _write_runtime_queue_state(
        state_dir / "experiment_queue_mlx_learned_sweep_wrong_lineage.sqlite",
        [ready_rows[0]["queue_candidate_id"]],
        elapsed_seconds=17.0,
        queue_id="ad_hoc_queue",
    )
    wrong_resource_state = _write_runtime_queue_state(
        state_dir / "experiment_queue_mlx_learned_sweep_wrong_resource.sqlite",
        [ready_rows[2]["queue_candidate_id"]],
        elapsed_seconds=19.0,
        queue_id="mlx_learned_sweep_wrong_resource_queue",
        resource_kind="local_cpu",
    )
    compatible_state = _write_runtime_queue_state(
        state_dir / "experiment_queue_mlx_learned_sweep_compatible.sqlite",
        [ready_rows[1]["queue_candidate_id"]],
        elapsed_seconds=31.0,
        queue_id="mlx_learned_sweep_compatible_queue",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--auto-batch-discover-runtime-telemetry-states",
            "--auto-batch-runtime-telemetry-state-dir",
            str(state_dir),
            "--auto-batch-runtime-telemetry-state-pattern",
            "*.sqlite",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "adaptive_discovered_state_observations.jsonl"),
            "--auto-batch-root-count",
            "2",
            "--auto-batch-root-plan-output",
            str(root_plan_path),
            "--max-new-observations",
            "2",
            "--rows-per-replan",
            "2",
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_adaptive_auto_batch_discovered_state_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root_plan = json.loads(root_plan_path.read_text(encoding="utf-8"))
    assert root_plan["runtime_telemetry_used"] is True
    assert root_plan["runtime_telemetry_key_count"] == 1
    assert root_plan["runtime_telemetry_state_policy"]["schema"] == (
        MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA
    )
    assert root_plan["runtime_telemetry_state_policy"]["mode"] == (
        "auto_discover_compatible_states"
    )
    assert root_plan["runtime_telemetry_state_policy"]["selected_state_paths"] == [
        str(compatible_state)
    ]
    assert root_plan["runtime_telemetry_state_policy"]["discovered_state_paths"] == [
        str(compatible_state)
    ]
    assert root_plan["runtime_telemetry"]["runtime_telemetry_source_state_paths"] == [
        str(compatible_state)
    ]
    assert str(incompatible_state) not in root_plan["runtime_telemetry"][
        "runtime_telemetry_source_state_paths"
    ]
    assert str(wrong_lineage_state) not in root_plan["runtime_telemetry"][
        "runtime_telemetry_source_state_paths"
    ]
    assert str(wrong_resource_state) not in root_plan["runtime_telemetry"][
        "runtime_telemetry_source_state_paths"
    ]
    assert root_plan["runtime_telemetry"]["runtime_telemetry_source_queue_ids"] == [
        "mlx_learned_sweep_compatible_queue"
    ]
    assert root_plan["runtime_telemetry"]["runtime_seconds_by_queue_candidate_id"][
        ready_rows[1]["queue_candidate_id"]
    ] == 31.0
    summary = json.loads(result.stdout)
    assert summary["auto_batch_runtime_telemetry_state_discovery_enabled"] is True
    assert summary["auto_batch_runtime_telemetry_state_discovered_count"] == 1
    assert summary["auto_batch_runtime_telemetry_state_discovered_paths"] == [
        str(compatible_state)
    ]
    assert summary["auto_batch_runtime_telemetry_state_count"] == 1
    assert summary["auto_batch_runtime_telemetry_used"] is True
    assert summary["auto_batch_runtime_telemetry_policy_schema"] == (
        MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA
    )
    assert (
        summary["auto_batch_runtime_telemetry_policy_id"]
        == "mlx_learned_sweep_runtime_telemetry_state_discovery"
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    policy = queue["experiments"][0]["metadata"]["runtime_telemetry_policy"]
    assert policy["schema"] == MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA
    assert policy["mode"] == "auto_discover_compatible_states"
    assert policy["selected_state_paths"] == [str(compatible_state)]
    assert policy["score_claim"] is False


def test_build_mlx_learned_sweep_discovery_policy_survives_empty_match(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)
    queue_path = tmp_path / "adaptive_auto_batch_queue_empty_state.json"
    root_plan_path = tmp_path / "adaptive_auto_batch_roots_empty_state.json"
    state_dir = tmp_path / ".omx" / "state"
    _write_runtime_queue_state(
        state_dir / "experiment_queue_mlx_learned_sweep_incompatible.sqlite",
        ["not-a-ready-queue-candidate"],
        elapsed_seconds=9.0,
        queue_id="mlx_learned_sweep_incompatible_queue",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--auto-batch-discover-runtime-telemetry-states",
            "--auto-batch-runtime-telemetry-state-dir",
            str(state_dir),
            "--auto-batch-runtime-telemetry-state-pattern",
            "*.sqlite",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "adaptive_empty_state_observations.jsonl"),
            "--auto-batch-root-count",
            "2",
            "--auto-batch-root-plan-output",
            str(root_plan_path),
            "--max-new-observations",
            "2",
            "--rows-per-replan",
            "2",
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_adaptive_auto_batch_empty_state_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root_plan = json.loads(root_plan_path.read_text(encoding="utf-8"))
    assert root_plan["runtime_telemetry_used"] is False
    assert root_plan["runtime_telemetry_state_policy"]["schema"] == (
        MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA
    )
    assert root_plan["runtime_telemetry_state_policy"]["mode"] == (
        "auto_discover_compatible_states"
    )
    assert root_plan["runtime_telemetry_state_policy"]["selected_state_paths"] == []
    assert root_plan["runtime_telemetry_state_policy"]["discovered_state_paths"] == []
    summary = json.loads(result.stdout)
    assert summary["auto_batch_runtime_telemetry_used"] is False
    assert summary["auto_batch_runtime_telemetry_policy_schema"] == (
        MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA
    )
    assert (
        summary["auto_batch_runtime_telemetry_policy_id"]
        == "mlx_learned_sweep_runtime_telemetry_state_discovery"
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    policy = queue["experiments"][0]["metadata"]["runtime_telemetry_policy"]
    assert policy["schema"] == MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA
    assert policy["selected_state_paths"] == []
    assert policy["score_claim"] is False


def test_build_mlx_learned_sweep_auto_batch_cli_reports_exhausted_local_mlx_rows(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    for row in plan["ranked_sweep_rows"]:
        if isinstance(row, dict) and row.get("sweep_config_id") == "mlx_local_response":
            row["ready_for_local_sweep"] = False
            row["suppression_reason"] = "observed_candidate_suppressed"
    _write_json(paths["plan"], plan)
    queue_path = tmp_path / "exhausted_mlx_queue.json"
    root_plan_path = tmp_path / "exhausted_mlx_roots.json"
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--auto-batch-discover-runtime-telemetry-states",
            "--auto-batch-runtime-telemetry-state-dir",
            str(state_dir),
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "exhausted_mlx_observations.jsonl"),
            "--auto-batch-root-count",
            "2",
            "--auto-batch-root-plan-output",
            str(root_plan_path),
            "--max-new-observations",
            "2",
            "--rows-per-replan",
            "2",
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_autopilot_exhausted_local_mlx_rows_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--local-mlx-concurrency",
            "2",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "plan has no ready mlx_local_response rows" in result.stderr
    assert "cannot discover runtime telemetry states" not in result.stderr
    assert not queue_path.exists()
    assert not root_plan_path.exists()


def test_build_mlx_learned_sweep_auto_batch_cli_rejects_missing_discovery_state_dir(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--auto-batch-discover-runtime-telemetry-states",
            "--auto-batch-runtime-telemetry-state-dir",
            str(tmp_path / "missing-state-dir"),
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "missing_discovery_state_observations.jsonl"),
            "--output",
            str(tmp_path / "missing_discovery_state_queue.json"),
            "--queue-id",
            "mlx_autopilot_missing_discovery_state_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "FATAL:" in result.stderr
    assert "--auto-batch-runtime-telemetry-state-dir not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_build_mlx_learned_sweep_auto_batch_cli_rejects_missing_state_telemetry(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "missing_state_observations.jsonl"),
            "--auto-batch-runtime-telemetry-state",
            str(tmp_path / "missing.sqlite"),
            "--output",
            str(tmp_path / "missing_state_queue.json"),
            "--queue-id",
            "mlx_autopilot_missing_state_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "FATAL:" in result.stderr
    assert "--auto-batch-runtime-telemetry-state not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_build_mlx_learned_sweep_auto_batch_cli_rejects_zero_rows_per_root(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    _write_json(paths["plan"], plan)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "zero_rows_observations.jsonl"),
            "--auto-batch-rows-per-root",
            "0",
            "--output",
            str(tmp_path / "zero_rows_queue.json"),
            "--queue-id",
            "mlx_autopilot_zero_rows_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "FATAL:" in result.stderr
    assert "rows_per_root must be >= 1" in result.stderr
    assert "Traceback" not in result.stderr


def test_build_mlx_learned_sweep_adaptive_cli_reports_nonpositive_cleanly(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan = _multi_pass_plan(selection)
    for row in plan["ranked_sweep_rows"]:
        if (
            row.get("schema") == "mlx_dynamic_learned_sweep_row.v1"
            and row.get("ready_for_local_sweep") is True
            and row.get("sweep_config_id") == "mlx_local_response"
        ):
            row["acquisition_value"] = 0.0
            row["learning_value_per_cost"] = 0.0
    _write_json(paths["plan"], plan)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_learned_sweep_autopilot_queue.py"),
            "--auto-batch-from-plan",
            "--auto-batch-adaptive-rows-per-root",
            "--plan",
            str(paths["plan"]),
            "--selection",
            str(paths["selection"]),
            "--candidate-payload",
            str(paths["candidate_payload"]),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--output-root",
            str(paths["output_root"]),
            "--observation-jsonl",
            str(tmp_path / "nonpositive_observations.jsonl"),
            "--output",
            str(tmp_path / "nonpositive_queue.json"),
            "--queue-id",
            "mlx_autopilot_nonpositive_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--source-artifact-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "FATAL:" in result.stderr
    assert "positive-utility ready row" in result.stderr
    assert "Traceback" not in result.stderr
