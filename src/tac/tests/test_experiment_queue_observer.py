# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
)
from comma_lab.scheduler.experiment_queue_observer import (
    observe_experiment_queue,
    render_observation_markdown,
)


def _queue(artifact_path: Path) -> dict[str, object]:
    return {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "exp0",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "smoke",
                        "kind": "command",
                        "command": ["python", "-c", "print('hello queue')"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": artifact_path.as_posix(),
                                "key": "schema",
                                "equals": "artifact.v1",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_observer_surfaces_running_step_log_tail_and_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"schema": "artifact.v1", "canonical_score": 0.5}),
        encoding="utf-8",
    )
    log_path = tmp_path / "logs" / "smoke.log"
    log_path.parent.mkdir()
    log_path.write_text("first\nsecond\n", encoding="utf-8")
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (
                json.dumps(
                    {
                        "command": ["python", "-c", "print('hello queue')"],
                        "log_path": str(log_path),
                    }
                ),
            ),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    running = observation["running_steps"][0]

    assert observation["schema"] == "experiment_queue_observation.v1"
    assert running["log_tail"] == ["second"]
    assert running["expected_artifacts"][0]["exists"] is True
    assert running["expected_artifacts"][0]["json_schema"] == "artifact.v1"
    assert running["expected_artifacts"][0]["postcondition_passed"] is True
    markdown = render_observation_markdown(observation)
    assert "observer_test" in markdown
    assert "smoke" in markdown


def test_observer_marks_existing_artifact_failed_when_postcondition_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"schema": "wrong.v1", "canonical_score": 0.5}),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    artifact_record = observation["running_steps"][0]["expected_artifacts"][0]

    assert artifact_record["exists"] is True
    assert artifact_record["json_schema"] == "wrong.v1"
    assert artifact_record["postcondition_passed"] is False
    markdown = render_observation_markdown(observation)
    assert "0/1" in markdown
