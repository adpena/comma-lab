from __future__ import annotations

from comma_lab.scheduler.experiment_queue_rewriters import (
    batch_mlx_scorer_response_steps,
    normalize_mlx_response_singleton_batches,
    optimize_mlx_first_receiver_preinflated_cache_handoff,
)


def test_normalize_mlx_response_singleton_batches_preserves_cache_batch() -> None:
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "q",
        "metadata": {},
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {
                        "id": "build_cache",
                        "command": [
                            ".venv/bin/python",
                            "tools/materialize_mlx_scorer_cache_from_submission.py",
                            "--batch-pairs",
                            "8",
                        ],
                    },
                    {
                        "id": "mlx_response",
                        "command": [
                            ".venv/bin/python",
                            "tools/run_mlx_scorer_response_from_cache.py",
                            "--batch-pairs",
                            "8",
                        ],
                    },
                    {
                        "id": "compile_followup",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_scorer_region_selector_cascade_queue_from_policy.py",
                            "--mlx-cache-batch-pairs",
                            "8",
                            "--mlx-batch-pairs",
                            "8",
                        ],
                    },
                    {
                        "id": "cpu_gate",
                        "command": [
                            ".venv/bin/python",
                            "tools/gate_mlx_scorer_response_for_cpu_spend.py",
                            "--mlx-response",
                            "mlx.json",
                        ],
                    },
                ],
            }
        ],
    }

    updated = normalize_mlx_response_singleton_batches(queue, reason="test")
    steps = {step["id"]: step["command"] for step in updated["experiments"][0]["steps"]}

    assert steps["build_cache"][steps["build_cache"].index("--batch-pairs") + 1] == "8"
    assert steps["mlx_response"][steps["mlx_response"].index("--batch-pairs") + 1] == "1"
    assert (
        steps["compile_followup"][steps["compile_followup"].index("--mlx-cache-batch-pairs") + 1]
        == "8"
    )
    assert (
        steps["compile_followup"][steps["compile_followup"].index("--mlx-batch-pairs") + 1]
        == "1"
    )
    assert updated["experiments"][0]["steps"][3]["on_postcondition_failure"] == "skipped"
    migration = updated["metadata"]["queue_migrations"][-1]
    assert migration["changed_command_count"] == 3
    assert migration["ready_for_exact_eval_dispatch"] is False


def test_optimize_mlx_receiver_cache_handoff_reuses_output_change_raw() -> None:
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "q",
        "metadata": {"output_root": "/tmp/campaign"},
        "experiments": [
            {
                "id": "candidate",
                "metadata": {},
                "steps": [
                    {
                        "id": "prove_receiver_patch_full_frame_output_change",
                        "command": [
                            ".venv/bin/python",
                            "tools/prove_shell_inflate_output_change.py",
                            "--output-dir",
                            "/tmp/campaign/v/frame1_region_waterfill_runtime_patch/full_frame_output_change_proof",
                            "--file-list-entry",
                            "0.raw",
                        ],
                    },
                    {
                        "id": "build_mlx_component_cache",
                        "command": [
                            ".venv/bin/python",
                            "tools/materialize_mlx_scorer_cache_from_submission.py",
                            "--work-dir",
                            "/tmp/campaign/v/frame1_region_waterfill_runtime_patch/local_component_spot_check/mlx_first_inflate_only_work",
                            "--report-output",
                            "/tmp/campaign/v/report.json",
                            "--inflate-timeout",
                            "1800",
                            "--local-acquisition-max-pairs",
                            "12",
                        ],
                    },
                ],
            }
        ],
    }

    updated = optimize_mlx_first_receiver_preinflated_cache_handoff(
        queue,
        reason="test",
    )
    steps = {step["id"]: step["command"] for step in updated["experiments"][0]["steps"]}
    proof = steps["prove_receiver_patch_full_frame_output_change"]
    cache = steps["build_mlx_component_cache"]

    assert proof[proof.index("--right-cache-dir") + 1] == (
        "/tmp/campaign/_shell_inflate_right_cache"
    )
    assert "--keep-scratch" in proof
    assert cache[cache.index("--preinflated-output-dir") + 1] == (
        "/tmp/campaign/v/frame1_region_waterfill_runtime_patch/"
        "full_frame_output_change_proof/scratch/right_out"
    )
    assert "--local-acquisition-max-pairs" not in cache
    assert updated["experiments"][0]["metadata"][
        "receiver_patch_output_change_right_cache_dir"
    ] == "/tmp/campaign/_shell_inflate_right_cache"
    assert updated["experiments"][0]["metadata"][
        "mlx_first_preinflated_receiver_output_dir"
    ].endswith("full_frame_output_change_proof/scratch/right_out")
    migration = updated["metadata"]["queue_migrations"][-1]
    assert migration["schema"] == "experiment_queue_mlx_receiver_cache_handoff_migration.v1"
    assert migration["changed_command_count"] == 6
    assert migration["score_claim"] is False


def test_batch_mlx_scorer_response_steps_reuses_hot_scorer_process() -> None:
    def experiment(candidate: str) -> dict:
        root = f"/tmp/campaign/{candidate}/frame1_region_waterfill_runtime_patch"
        return {
            "id": candidate,
            "priority": 2,
            "status": "queued",
            "metadata": {},
            "steps": [
                {
                    "id": "build_mlx_component_cache",
                    "command": [".venv/bin/python", "tools/build_cache.py"],
                    "postconditions": [{"type": "exists", "path": f"{root}/cache.done"}],
                },
                {
                    "id": "local_mlx_component_response",
                    "requires": ["build_mlx_component_cache"],
                    "resources": {"kind": "local_mlx"},
                    "command": [
                        ".venv/bin/python",
                        "tools/run_mlx_scorer_response_from_cache.py",
                        "--reference-cache-dir",
                        "ref_cache",
                        "--candidate-cache-dir",
                        f"{root}/local_component_spot_check/mlx_scorer_input_cache",
                        "--archive",
                        f"{root}/submission_dir/archive.zip",
                        "--output",
                        f"{root}/local_component_spot_check/mlx_scorer_response.json",
                        "--repo-root",
                        ".",
                        "--batch-pairs",
                        "1",
                        "--device",
                        "gpu",
                        "--allow-unaudited-candidate-cache-debug",
                        "--allow-gpu-research-signal",
                        "--components-dir",
                        f"{root}/local_component_spot_check/mlx_components",
                        "--response-family",
                        "scorer_region_frame1_waterfill_patch",
                    ],
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": f"{root}/local_component_spot_check/mlx_scorer_response.json",
                            "key": "schema_version",
                            "equals": "mlx_scorer_response.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": f"{root}/local_component_spot_check/mlx_scorer_response.json",
                        },
                    ],
                },
            ],
        }

    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "q",
        "controls": {"max_concurrency": {"local_cpu": 2, "local_mlx": 1}},
        "metadata": {},
        "experiments": [experiment("candidate_a"), experiment("candidate_b")],
    }

    updated = batch_mlx_scorer_response_steps(
        queue,
        max_jobs_per_batch=4,
        reason="test",
    )

    batch = updated["experiments"][0]
    assert batch["id"] == "mlx_scorer_response_batch_0000"
    batch_step = batch["steps"][0]
    assert batch_step["command"][1] == "tools/run_mlx_scorer_response_from_cache_batch.py"
    assert batch_step["requires"] == [
        "candidate_a.build_mlx_component_cache",
        "candidate_b.build_mlx_component_cache",
    ]
    assert batch_step["command"].count("--job-json") == 2
    assert batch_step["resources"]["kind"] == "local_mlx"
    for experiment in updated["experiments"][1:]:
        response = experiment["steps"][1]
        assert response["command"][1] == "tools/validate_json_artifact_contract.py"
        assert response["resources"]["kind"] == "local_cpu"
        assert "mlx_scorer_response_batch_0000.run_mlx_scorer_response_batch" in response[
            "requires"
        ]
    migration = updated["metadata"]["queue_migrations"][-1]
    assert migration["schema"] == "experiment_queue_mlx_response_batching_migration.v1"
    assert migration["batch_experiment_count"] == 1
    assert migration["changed_command_count"] == 4
    assert migration["ready_for_exact_eval_dispatch"] is False
