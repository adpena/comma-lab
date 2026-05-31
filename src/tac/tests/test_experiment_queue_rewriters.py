from __future__ import annotations

from comma_lab.scheduler.experiment_queue_rewriters import (
    normalize_mlx_response_singleton_batches,
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
