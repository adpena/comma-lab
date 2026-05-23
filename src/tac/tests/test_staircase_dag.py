# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    _step_hashes,
    connect_state,
    initialize_queue_state,
)
from comma_lab.scheduler.staircase_dag import (
    build_staircase_dag_from_experiment_queue,
    build_storage_plan_payload,
    experiment_queue_status_map,
    parse_resource_pool_spec,
    plan_staircase_dispatch,
    write_staircase_dag,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "plan_staircase_dag.py"


def _queue() -> dict[str, object]:
    return {
        "schema": "experiment_queue.v1",
        "queue_id": "staircase_fixture",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 2, "local_mlx": 1}},
        "experiments": [
            {
                "id": "cand_a",
                "status": "queued",
                "priority": 1,
                "lane_id": "lane_family_a",
                "metadata": {
                    "schema": "fixture_metadata.v1",
                    "source_archive_sha256": "a" * 64,
                },
                "steps": [
                    {
                        "id": "plan",
                        "kind": "command",
                        "command": ["python", "-c", "print('plan')"],
                        "resources": {"kind": "local_cpu", "MEM_GB": 1},
                    },
                    {
                        "id": "score",
                        "kind": "command",
                        "command": ["python", "-c", "print('score')"],
                        "requires": ["plan"],
                        "resources": {"kind": "local_mlx", "MLX": 1},
                    },
                ],
            },
            {
                "id": "cand_b",
                "status": "queued",
                "priority": 2,
                "lane_id": "lane_family_b",
                "steps": [
                    {
                        "id": "plan",
                        "kind": "command",
                        "command": ["python", "-c", "print('other')"],
                        "resources": {"kind": "local_cpu", "MEM_GB": 1},
                    }
                ],
            },
        ],
    }


def test_build_queue_dag_plans_executor_specs_without_authority(tmp_path: Path) -> None:
    dag = build_staircase_dag_from_experiment_queue(
        _queue(),
        dag_id="fixture_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 2, "local_mlx": 1}, "memory_gb": 128, "disk_gb": 80}
        ],
    )
    plan = plan_staircase_dispatch(dag, max_nodes=4)

    selected = {node["node_id"]: node for node in plan["selected_nodes"]}
    assert dag["schema"] == "staircase_dag.v1"
    assert dag["score_claim"] is False
    assert plan["schema"] == "staircase_dispatch_plan.v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert set(selected) == {"cand_a.plan", "cand_b.plan"}
    assert all(task["pure"] is False for task in plan["dask_task_specs"])
    assert plan["dask_task_specs"][0]["resources"] == {"local_cpu": 1}
    assert plan["dask_task_specs"][0]["queue_id"] == "staircase_fixture"
    assert plan["dask_task_specs"][0]["experiment_id"] in {"cand_a", "cand_b"}
    assert plan["dask_task_specs"][0]["step_id"] == "plan"
    assert plan["dask_task_specs"][0]["step_hashes"]["definition_hash"]
    if plan["dask_task_specs"][0]["experiment_id"] == "cand_a":
        expected_hashes = _step_hashes(
            _queue()["experiments"][0]["steps"][0],
            experiment_metadata=_queue()["experiments"][0]["metadata"],
        )
        assert plan["dask_task_specs"][0]["step_hashes"] == expected_hashes
    assert plan["dask_task_specs"][0]["queue_state_writeback"]["required"] is True
    assert (
        plan["dask_task_specs"][0]["queue_state_writeback"]["step_hashes"]
        == plan["dask_task_specs"][0]["step_hashes"]
    )
    assert (
        plan["dask_task_specs"][0]["executor_boundary"]
        == "planning_only_task_must_write_back_to_experiment_queue_state"
    )


def test_plan_uses_queue_state_and_unblocks_successors(tmp_path: Path) -> None:
    queue = _queue()
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded'
            WHERE queue_id = 'staircase_fixture'
              AND experiment_id = 'cand_a'
              AND step_id = 'plan'
            """
        )
        conn.commit()
    status_map = experiment_queue_status_map(
        queue_path=queue_path,
        repo_root=tmp_path,
        state_path=state,
    )

    assert status_map == {
        "cand_a.plan": "succeeded",
        "cand_a.score": "queued",
        "cand_b.plan": "queued",
    }
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="fixture_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 1, "local_mlx": 1}, "memory_gb": 128, "disk_gb": 80}
        ],
    )
    plan = plan_staircase_dispatch(dag, status_map=status_map)

    assert [node["node_id"] for node in plan["selected_nodes"]] == ["cand_a.score", "cand_b.plan"]
    assert plan["dask_task_specs"][0]["resources"] == {"local_mlx": 1}


def test_staircase_dag_preserves_cross_experiment_dependencies() -> None:
    queue = _queue()
    queue["experiments"].insert(
        0,
        {
            "id": "preflight",
            "priority": 0,
            "steps": [
                {
                    "id": "cleanup",
                    "command": ["python", "-c", "print('cleanup')"],
                    "resources": {"kind": "local_cpu"},
                }
            ],
        },
    )
    queue["experiments"][1]["steps"][0]["requires"] = ["preflight.cleanup"]
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="fixture_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 2, "local_mlx": 1}, "memory_gb": 128, "disk_gb": 80}
        ],
    )
    by_id = {node["node_id"]: node for node in dag["nodes"]}

    assert by_id["cand_a.plan"]["dependencies"] == ["preflight.cleanup"]
    plan = plan_staircase_dispatch(dag, max_nodes=4)
    assert [node["node_id"] for node in plan["selected_nodes"]] == [
        "preflight.cleanup",
        "cand_b.plan",
    ]


def test_staircase_dag_skips_nonqueued_experiments() -> None:
    queue = _queue()
    queue["experiments"][0]["status"] = "paused"
    queue["experiments"][1]["status"] = "disabled"

    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="fixture_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 2, "local_mlx": 1}, "memory_gb": 128, "disk_gb": 80}
        ],
    )

    assert dag["nodes"] == []
    assert plan_staircase_dispatch(dag)["selected_count"] == 0


def test_staircase_dag_respects_queue_max_concurrency_over_pool_slots() -> None:
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "staircase_concurrency_fixture",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": f"cand_{index}",
                "status": "queued",
                "priority": index,
                "steps": [
                    {
                        "id": "materialize",
                        "kind": "command",
                        "command": ["python", "-c", f"print({index})"],
                        "resources": {"kind": "local_cpu"},
                    }
                ],
            }
            for index in range(5)
        ],
    }
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="fixture_concurrency_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 8}, "memory_gb": 128, "disk_gb": 80}
        ],
    )

    plan = plan_staircase_dispatch(dag, max_nodes=5)

    assert dag["controls"]["max_concurrency"] == {"local_cpu": 1}
    assert plan["selected_count"] == 1


def test_staircase_dag_respects_queue_control_pause() -> None:
    queue = _queue()
    queue["controls"]["mode"] = "paused"
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="fixture_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 2, "local_mlx": 1}, "memory_gb": 128, "disk_gb": 80}
        ],
    )

    plan = plan_staircase_dispatch(dag, max_nodes=4)

    assert plan["selected_count"] == 0
    assert {row["reason"] for row in plan["blocked_nodes"]} == {"queue_control_not_running"}
    assert {row["mode"] for row in plan["blocked_nodes"]} == {"paused"}


def test_staircase_storage_plan_blocks_when_no_tier_selected(tmp_path: Path) -> None:
    storage = build_storage_plan_payload(
        repo_root=tmp_path,
        storage_tiers=["missing=/definitely/not/a/real/storage/tier"],
        workload_subdir="experiments/results",
    )
    dag = build_staircase_dag_from_experiment_queue(
        _queue(),
        dag_id="fixture_dag",
        resource_pools=[
            {"id": "m5", "slots": {"local_cpu": 2, "local_mlx": 1}, "memory_gb": 128, "disk_gb": 80}
        ],
        storage_plan=storage,
    )
    plan = plan_staircase_dispatch(dag)

    assert plan["selected_count"] == 0
    assert {row["reason"] for row in plan["blocked_nodes"]} == {"no_eligible_storage_tier"}


def test_rejects_truthy_authority_in_dag_node() -> None:
    queue = _queue()
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="fixture_dag",
        resource_pools=[{"id": "m5", "slots": {"local_cpu": 1}, "memory_gb": 128, "disk_gb": 80}],
    )
    dag["nodes"][0]["score_claim"] = True

    with pytest.raises(ExperimentQueueError, match="score_claim"):
        plan_staircase_dispatch(dag)


def test_parse_resource_pool_spec() -> None:
    pool = parse_resource_pool_spec("bat00:local_cpu=8,local_cuda=1,memory_gb=32,disk_gb=64,tags=windows+cuda")

    assert pool["id"] == "bat00"
    assert pool["slots"] == {"local_cpu": 8, "local_cuda": 1}
    assert pool["tags"] == ["windows", "cuda"]


def test_parse_resource_pool_spec_rejects_unknown_resource_kind() -> None:
    with pytest.raises(ExperimentQueueError, match="unsupported resource kind"):
        parse_resource_pool_spec("bat00:local_cpu=8,cuda_gpu=1,memory_gb=32,disk_gb=64")


def test_write_staircase_dag_refuses_silent_overwrite(tmp_path: Path) -> None:
    dag = build_staircase_dag_from_experiment_queue(_queue(), dag_id="fixture_dag")
    path = tmp_path / "dag.json"

    write_staircase_dag(path, dag)

    with pytest.raises(ExperimentQueueError, match="refusing to overwrite"):
        write_staircase_dag(path, dag)


def test_cli_from_queue_and_plan(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(_queue()), encoding="utf-8")
    dag_path = tmp_path / "dag.json"
    plan_path = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "from-queue",
            "--queue",
            str(queue_path),
            "--output",
            str(dag_path),
            "--resource-pool",
            "m5:local_cpu=2,local_mlx=1,memory_gb=128,disk_gb=80",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "plan",
            "--dag",
            str(dag_path),
            "--output",
            str(plan_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["selected_count"] == 2
    assert plan["ready_for_exact_eval_dispatch"] is False
