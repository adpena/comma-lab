from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.optimizer.exact_ready_audit import (
    audit_exact_ready_queues,
    discover_exact_ready_queues,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_claims(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def _ready_queue(path: Path, *, lane_id: str, archive_sha: str, ready: bool = True) -> Path:
    return _write_json(
        path,
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "candidate",
                    "lane_id": lane_id,
                    "ready_for_exact_eval_dispatch": ready,
                    "candidate_archive_sha256": archive_sha,
                    "archive_sha256": archive_sha,
                    "archive_bytes": 123,
                    "score_claim": False,
                    "dispatch_blockers": [],
                }
            ],
        },
    )


def test_audit_flags_ready_queue_after_terminal_negative_same_archive(
    tmp_path: Path,
) -> None:
    archive_sha = "a" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_pr103",  # FAKE_LANE_OK: synthetic terminal-evidence fixture.
        archive_sha=archive_sha,
    )
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:00:00Z | test | lane_pr103 | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |"
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is False
    assert payload["stale_ready_row_count"] == 1
    blockers = payload["queues"][0]["stale_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith("same_lane_terminal_negative_for_same_archive")
        for blocker in blockers
    )


def test_audit_flags_ready_queue_after_cuda_score_not_below_floor(
    tmp_path: Path,
) -> None:
    archive_sha = "b" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_pr101",  # FAKE_LANE_OK: synthetic terminal-evidence fixture.
        archive_sha=archive_sha,
    )
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:00:00Z | test | lane_pr101 | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score_recomputed=0.22650343150032118 |"
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
        active_floor_score=0.2089810755823297,
    )

    assert payload["passed"] is False
    blockers = payload["queues"][0]["stale_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith(
            "same_lane_terminal_score_not_below_active_floor_for_same_archive"
        )
        for blocker in blockers
    )


def test_audit_flags_promotable_terminal_success_as_already_evaluated(
    tmp_path: Path,
) -> None:
    archive_sha = "1" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_pr101",  # FAKE_LANE_OK: synthetic terminal-evidence fixture.
        archive_sha=archive_sha,
    )
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:00:00Z | test | lane_pr101 | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score_recomputed=1.95e-1 |"
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
        active_floor_score=0.2089810755823297,
    )

    assert payload["passed"] is False
    blockers = payload["queues"][0]["stale_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith(
            "same_lane_terminal_score_already_below_active_floor_for_same_archive"
        )
        for blocker in blockers
    )


def test_audit_allows_runtime_changed_row_after_different_runtime_terminal(
    tmp_path: Path,
) -> None:
    archive_sha = "2" * 64
    old_runtime = "3" * 64
    new_runtime = "4" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_runtime_patch",  # FAKE_LANE_OK: synthetic runtime fixture.
        archive_sha=archive_sha,
    )
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["dispatch_ready"][0]
    row["runtime_tree_sha256"] = new_runtime
    row["score_affecting_runtime_changed"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:01:00Z | test | lane_runtime_patch | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |",
            f"| 2026-05-10T00:00:00Z | test | lane_runtime_patch | modal | job1 |  | active_dispatching | archive_sha={archive_sha}; runtime_tree_sha={old_runtime}; score_claim=false_until_modal_validation |",
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is True
    assert payload["stale_ready_row_count"] == 0


def test_audit_blocks_runtime_changed_row_after_same_runtime_terminal(
    tmp_path: Path,
) -> None:
    archive_sha = "5" * 64
    runtime_sha = "6" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_runtime_patch",  # FAKE_LANE_OK: synthetic runtime fixture.
        archive_sha=archive_sha,
    )
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["dispatch_ready"][0]
    row["runtime_tree_sha256"] = runtime_sha
    row["score_affecting_runtime_changed"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:01:00Z | test | lane_runtime_patch | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |",
            f"| 2026-05-10T00:00:00Z | test | lane_runtime_patch | modal | job1 |  | active_dispatching | archive_sha={archive_sha}; runtime_tree_sha={runtime_sha}; score_claim=false_until_modal_validation |",
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is False
    blockers = payload["queues"][0]["stale_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith("same_lane_terminal_negative_for_same_archive")
        for blocker in blockers
    )


def _add_live_runtime_fields(
    queue: Path,
    *,
    repo_root: Path,
    runtime_sha_override: str | None = None,
) -> str:
    submission = repo_root / "packet"
    submission.mkdir(parents=True)
    archive_bytes = b"archive-bytes"
    (submission / "archive.zip").write_bytes(archive_bytes)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    (submission / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n",
        encoding="utf-8",
    )
    actual_runtime_sha = runtime_dependency_manifest(submission, repo_root)[
        "runtime_tree_sha256"
    ]
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["dispatch_ready"][0]
    row["archive_path"] = "packet/archive.zip"
    row["candidate_archive_sha256"] = archive_sha
    row["archive_sha256"] = archive_sha
    row["submission_dir"] = "packet"
    row["runtime_tree_sha256"] = runtime_sha_override or actual_runtime_sha
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return actual_runtime_sha


def test_audit_flags_ready_row_with_stale_runtime_tree_sha(tmp_path: Path) -> None:
    archive_sha = "7" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_runtime_patch",  # FAKE_LANE_OK: synthetic runtime fixture.
        archive_sha=archive_sha,
    )
    actual_runtime_sha = _add_live_runtime_fields(
        queue,
        repo_root=tmp_path,
        runtime_sha_override="8" * 64,
    )
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is False
    row = payload["queues"][0]["stale_ready_rows"][0]
    assert row["live_custody"]["actual_runtime_tree_sha256"] == actual_runtime_sha
    assert any(
        blocker.startswith("ready_row_runtime_tree_sha_mismatch")
        for blocker in row["blockers"]
    )


def test_audit_allows_ready_row_with_current_live_runtime_tree_sha(
    tmp_path: Path,
) -> None:
    archive_sha = "9" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_runtime_patch",  # FAKE_LANE_OK: synthetic runtime fixture.
        archive_sha=archive_sha,
    )
    _add_live_runtime_fields(queue, repo_root=tmp_path)
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is True
    assert payload["stale_ready_row_count"] == 0


def test_audit_allows_ready_queue_after_infra_failure_same_archive(
    tmp_path: Path,
) -> None:
    archive_sha = "c" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_pr103",  # FAKE_LANE_OK: synthetic terminal-evidence fixture.
        archive_sha=archive_sha,
    )
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:00:00Z | test | lane_pr103 | modal | job1 |  | failed_runtime_dependency_missing_constriction | archive_sha={archive_sha}; no score |"
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is True
    assert payload["stale_ready_row_count"] == 0


def test_audit_ignores_non_ready_rows_with_terminal_negative(tmp_path: Path) -> None:
    archive_sha = "d" * 64
    queue = _ready_queue(
        tmp_path / "experiments/results/fixture/exact_ready_queue.json",
        lane_id="lane_pr103",  # FAKE_LANE_OK: synthetic terminal-evidence fixture.
        archive_sha=archive_sha,
        ready=False,
    )
    claims = _write_claims(
        tmp_path / ".omx/state/active_lane_dispatch_claims.md",
        [
            f"| 2026-05-10T00:00:00Z | test | lane_pr103 | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.3495 |"
        ],
    )

    payload = audit_exact_ready_queues(
        [queue],
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert payload["passed"] is True
    assert payload["queues"][0]["row_count"] == 1


def test_discover_exact_ready_queues_deduplicates_patterns(tmp_path: Path) -> None:
    first = _ready_queue(
        tmp_path / "experiments/results/a/exact_ready_queue.json",
        lane_id="lane_a",  # FAKE_LANE_OK: synthetic discovery fixture.
        archive_sha="e" * 64,
    )
    second = _ready_queue(
        tmp_path / "experiments/results/b/exact_ready_queue.json",
        lane_id="lane_b",  # FAKE_LANE_OK: synthetic discovery fixture.
        archive_sha="f" * 64,
    )

    found = discover_exact_ready_queues(
        repo_root=tmp_path,
        scan_root=Path("experiments/results"),
        patterns=["**/exact_ready_queue.json", "**/exact_ready_queue.json"],
    )

    assert found == [first, second]


def test_discover_exact_ready_queues_includes_prefixed_filenames(tmp_path: Path) -> None:
    exact_name = _ready_queue(
        tmp_path / "experiments/results/a/exact_ready_queue.json",
        lane_id="lane_a",  # FAKE_LANE_OK: synthetic discovery fixture.
        archive_sha="e" * 64,
    )
    prefixed = _ready_queue(
        tmp_path / "experiments/results/b/pr106_q10_exact_ready_queue.json",
        lane_id="lane_b",  # FAKE_LANE_OK: synthetic discovery fixture.
        archive_sha="f" * 64,
    )

    found = discover_exact_ready_queues(
        repo_root=tmp_path,
        scan_root=Path("experiments/results"),
    )

    assert found == [exact_name, prefixed]
