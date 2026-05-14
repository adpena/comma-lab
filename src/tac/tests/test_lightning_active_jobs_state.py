# SPDX-License-Identifier: MIT
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from tac.deploy.lightning.active_jobs_state import (
    load_active_jobs,
    mark_job_terminal,
    register_job,
    upsert_job,
)


def _register_worker(path_s: str, lock_s: str, index: int) -> None:
    from pathlib import Path

    from tac.deploy.lightning.active_jobs_state import register_job

    register_job(
        {"job_name": f"job-{index}", "lane_id": "lane_test"},
        path=Path(path_s),
        lock_path=Path(lock_s),
    )


def test_load_active_jobs_missing_or_invalid_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "lightning_active_jobs.json"
    assert load_active_jobs(path) == []

    path.write_text("{not-json", encoding="utf-8")
    assert load_active_jobs(path) == []

    path.write_text('{"not": "a list"}', encoding="utf-8")
    assert load_active_jobs(path) == []


def test_register_upsert_and_mark_terminal_use_injected_paths(tmp_path: Path) -> None:
    path = tmp_path / "lightning_active_jobs.json"
    lock = tmp_path / "lightning_active_jobs.json.lock"

    rows = register_job(
        {"job_name": "alpha", "lane_id": "lane_a"},  # FAKE_LANE_OK: test fixture row
        path=path,
        lock_path=lock,
    )
    assert [r["job_name"] for r in rows] == ["alpha"]

    rows = upsert_job(
        {"job_name": "alpha", "lane_id": "lane_b"},  # FAKE_LANE_OK: test fixture row
        path=path,
        lock_path=lock,
    )
    assert rows == [{"job_name": "alpha", "lane_id": "lane_b"}]  # FAKE_LANE_OK: test fixture row

    rows = mark_job_terminal(
        "alpha",
        terminal_status="completed",
        extra_fields={"score": 0.1928},
        path=path,
        lock_path=lock,
    )
    assert rows == [
        {
            "job_name": "alpha",
            "lane_id": "lane_b",  # FAKE_LANE_OK: test fixture row
            "terminal_status": "completed",
            "score": 0.1928,
        }
    ]
    assert load_active_jobs(path) == rows


def test_parallel_register_job_preserves_distinct_rows(tmp_path: Path) -> None:
    path = tmp_path / "lightning_active_jobs.json"
    lock = tmp_path / "lightning_active_jobs.json.lock"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_register_worker, args=(str(path), str(lock), i))
        for i in range(8)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=10)
    assert all(proc.exitcode == 0 for proc in procs)

    rows = load_active_jobs(path)
    assert sorted(row["job_name"] for row in rows) == [f"job-{i}" for i in range(8)]
