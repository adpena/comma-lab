# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import materializer_exact_eval_consumer as consumer
from tac.optimization.proxy_candidate_contract import truthy_authority_field_violations
from tac.optimizer.exact_dispatch_authority import ExactDispatchAuthorityVerdict

ARCHIVE_SHA = "a" * 64
RUNTIME_TREE_SHA = "b" * 64
RUNTIME_CONTENT_SHA = "c" * 64
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _exact_ready_queue(
    repo: Path,
    name: str,
    *,
    candidate_id: str = "candidate_a",
    archive_sha: str = ARCHIVE_SHA,
    runtime_tree_sha: str = RUNTIME_TREE_SHA,
    runtime_content_sha: str = RUNTIME_CONTENT_SHA,
) -> Path:
    row = {
        "candidate_id": candidate_id,
        "lane_id": "lane_exact_ready_consumer_fixture",
        "target_modes": ["contest_exact_eval"],
        "ready_for_exact_eval_dispatch": True,
        "dispatch_packet_ready": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "archive_sha256": archive_sha,
        "candidate_archive_sha256": archive_sha,
        "archive_bytes": 123,
        "candidate_archive_bytes": 123,
        "archive_path": "submission/archive.zip",
        "submission_dir": "submission",
        "runtime_tree_sha256": runtime_tree_sha,
        "runtime_content_tree_sha256": runtime_content_sha,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
    }
    return _write_json(
        repo / name,
        {
            "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
            "dispatch_ready_count": 1,
            "dispatch_ready": [row],
            "top_k": [row],
        },
    )


def _bridge(repo: Path, paths: list[Path], *, score_claim: bool = False) -> Path:
    rows = [
        {
            "candidate_id": f"candidate_{index}",
            "exact_ready_queue_written": True,
            "exact_ready_queue_path": path.relative_to(repo).as_posix(),
            "blockers": [],
        }
        for index, path in enumerate(paths)
    ]
    return _write_json(
        repo / "bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            "ready_candidate_count": len(rows),
            "blocked_candidate_count": 0,
            "rows": rows,
            "score_claim": score_claim,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )


def _patch_authority(monkeypatch: pytest.MonkeyPatch, *, authorized: bool = True) -> None:
    monkeypatch.setattr(
        consumer,
        "audit_exact_ready_queue",
        lambda *args, **kwargs: {"stale_ready_rows": []},
    )

    def fake_authority(*args: object, **kwargs: object) -> ExactDispatchAuthorityVerdict:
        return ExactDispatchAuthorityVerdict(
            source="fixture_authority",
            authorized=authorized,
            blockers=() if authorized else ("fixture_not_ready",),
            ready_for_exact_eval_dispatch=authorized,
            contest_exact_eval_target=True,
            facts={},
        )

    monkeypatch.setattr(consumer, "exact_dispatch_authority", fake_authority)


def test_consumer_builds_paused_dry_run_queue_from_bridge_and_dedupes_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    first = _exact_ready_queue(repo, "first.exact_ready_queue.json", candidate_id="first")
    duplicate = _exact_ready_queue(
        repo,
        "duplicate.exact_ready_queue.json",
        candidate_id="renamed_duplicate",
    )
    bridge = _bridge(repo, [first, duplicate])
    _patch_authority(monkeypatch)

    result = consumer.build_materializer_exact_eval_consumer_queue(
        repo_root=repo,
        bridge_report_paths=[bridge],
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )

    report = result["report"]
    queue = result["experiment_queue"]
    assert report["schema"] == consumer.CONSUMER_SCHEMA
    assert report["authorized_candidate_count"] == 1
    assert report["duplicate_candidate_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(report) == []
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["mode"] == "paused"
    assert len(queue["experiments"]) == 1
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["promotion_eligible"] is False
    claim_step, dispatch_step = experiment["steps"]
    assert claim_step["id"] == "claim_lane_dispatch"
    assert "--dry-run" in claim_step["command"]
    assert dispatch_step["id"] == "dispatch_exact_eval_dry_run"
    assert "--dry-run" in dispatch_step["command"]
    blocked = [row for row in report["rows"] if row["blockers"]]
    assert blocked[0]["blockers"][0].startswith("duplicate_stable_identity:")


def test_consumer_dedupes_same_runtime_content_with_different_runtime_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    first = _exact_ready_queue(repo, "first.exact_ready_queue.json", candidate_id="first")
    duplicate = _exact_ready_queue(
        repo,
        "same_consumed_runtime.exact_ready_queue.json",
        candidate_id="same_consumed_runtime",
        runtime_tree_sha="d" * 64,
    )
    _patch_authority(monkeypatch)

    result = consumer.build_materializer_exact_eval_consumer_queue(
        repo_root=repo,
        exact_ready_queue_paths=[first, duplicate],
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )

    report = result["report"]
    assert report["authorized_candidate_count"] == 1
    assert report["duplicate_candidate_count"] == 1
    blocked = [row for row in report["rows"] if row["blockers"]]
    assert blocked[0]["blockers"][0].startswith("duplicate_stable_identity:")
    assert ":runtime_tree=" not in blocked[0]["stable_identity"]


def test_consumer_blocks_archive_sha_alias_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(repo, "stale_alias.exact_ready_queue.json")
    payload = json.loads(ready_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        for row in payload[key]:
            row["archive_sha256"] = "d" * 64
    _write_json(ready_queue, payload)
    _patch_authority(monkeypatch)

    result = consumer.build_materializer_exact_eval_consumer_queue(
        repo_root=repo,
        exact_ready_queue_paths=[ready_queue],
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )

    row = result["report"]["rows"][0]
    assert result["report"]["authorized_candidate_count"] == 0
    assert any(
        blocker.startswith("archive_sha_alias_mismatch:")
        for blocker in row["blockers"]
    )
    assert row["archive_sha256"] == ARCHIVE_SHA


def test_consumer_blocks_exact_ready_queue_counter_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(repo, "bad_count.exact_ready_queue.json")
    payload = json.loads(ready_queue.read_text(encoding="utf-8"))
    payload["dispatch_ready_count"] = 2
    _write_json(ready_queue, payload)

    result = consumer.build_materializer_exact_eval_consumer_queue(
        repo_root=repo,
        exact_ready_queue_paths=[ready_queue],
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )

    row = result["report"]["rows"][0]
    assert result["report"]["authorized_candidate_count"] == 0
    assert row["blockers"][0].startswith(
        "exact_ready_queue_dispatch_ready_count_mismatch:"
    )


def test_consumer_fails_closed_when_exact_dispatch_authority_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(repo, "ready.exact_ready_queue.json")
    _patch_authority(monkeypatch, authorized=False)

    result = consumer.build_materializer_exact_eval_consumer_queue(
        repo_root=repo,
        exact_ready_queue_paths=[ready_queue],
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )

    report = result["report"]
    queue = result["experiment_queue"]
    assert report["authorized_candidate_count"] == 0
    assert report["blocked_candidate_count"] == 1
    assert report["rows"][0]["blockers"] == [
        "exact_dispatch_authority:fixture_not_ready",
        "exact_dispatch_authority:not_authorized",
    ]
    assert queue["controls"]["mode"] == "paused"
    assert queue["experiments"][0]["id"] == "no_authorized_materializer_exact_eval_consumer_rows"
    assert queue["experiments"][0]["metadata"]["ready_for_exact_eval_dispatch"] is False


def test_consumer_refuses_truthy_authority_in_bridge_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(repo, "ready.exact_ready_queue.json")
    bridge = _bridge(repo, [ready_queue], score_claim=True)

    with pytest.raises(ValueError, match="forbidden truthy authority fields"):
        consumer.build_materializer_exact_eval_consumer_queue(
            repo_root=repo,
            bridge_report_paths=[bridge],
        )


def test_consumer_blocks_rows_without_stable_archive_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(
        repo,
        "missing_identity.exact_ready_queue.json",
        runtime_content_sha="",
    )
    _patch_authority(monkeypatch)

    result = consumer.build_materializer_exact_eval_consumer_queue(
        repo_root=repo,
        exact_ready_queue_paths=[ready_queue],
    )

    row = result["report"]["rows"][0]
    assert result["report"]["authorized_candidate_count"] == 0
    assert row["blockers"] == ["stable_identity_runtime_content_tree_sha256_missing"]
    assert row["ready_for_exact_eval_dispatch"] is False


def test_write_json_refuses_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "consumer_report.json"
    consumer.write_json(path, {"value": 1})

    with pytest.raises(Exception, match="refusing_to_overwrite_json"):
        consumer.write_json(path, {"value": 2})

    consumer.write_json(path, {"value": 3}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 3


def test_consumer_cli_writes_false_authority_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(repo, "ready.exact_ready_queue.json")
    report_out = tmp_path / "consumer_report.json"
    queue_out = tmp_path / "consumer_queue.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_materializer_exact_eval_consumer.py"),
            "--exact-ready-queue",
            str(ready_queue),
            "--consumer-report-out",
            str(report_out),
            "--experiment-queue-out",
            str(queue_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    report = json.loads(report_out.read_text(encoding="utf-8"))
    queue = json.loads(queue_out.read_text(encoding="utf-8"))
    assert summary["score_claim"] is False
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["mode"] == "paused"


def test_consumer_cli_disabling_active_floors_requires_override(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_materializer_exact_eval_consumer.py"),
            "--disable-active-floor-score",
            "--consumer-report-out",
            str(tmp_path / "consumer_report.json"),
            "--experiment-queue-out",
            str(tmp_path / "consumer_queue.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "disabling active floors requires --operator-override-reason" in proc.stderr


def test_consumer_cli_require_authorized_exits_nonzero_on_no_authorized_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ready_queue = _exact_ready_queue(repo, "ready.exact_ready_queue.json")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_materializer_exact_eval_consumer.py"),
            "--exact-ready-queue",
            str(ready_queue),
            "--consumer-report-out",
            str(tmp_path / "consumer_report.json"),
            "--experiment-queue-out",
            str(tmp_path / "consumer_queue.json"),
            "--require-authorized",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
