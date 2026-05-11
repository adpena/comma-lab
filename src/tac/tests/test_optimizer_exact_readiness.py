from __future__ import annotations

import datetime as dt
import json
import stat
import zipfile
import importlib.util
import sys
from pathlib import Path

from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
    ACTIVE_RATE_ONLY_FLOOR_SCORE,
    ACTIVE_SCORE_FRONTIER_SCORE,
    promote_candidate_for_exact_eval,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_active_floor_score_tracks_score_frontier_not_rate_only_anchor() -> None:
    assert ACTIVE_FLOOR_ARCHIVE_BYTES == 185_578
    assert ACTIVE_RATE_ONLY_FLOOR_SCORE == 0.2089810755823297
    assert ACTIVE_SCORE_FRONTIER_SCORE == 0.2066181354574151
    assert ACTIVE_FLOOR_SCORE == ACTIVE_SCORE_FRONTIER_SCORE


def _load_parallel_dispatch_tool():
    path = REPO_ROOT / "tools" / "parallel_dispatch_top_k.py"
    spec = importlib.util.spec_from_file_location("parallel_dispatch_top_k_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path, member: str = "0.bin", payload: bytes = b"payload") -> tuple[int, str]:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _make_submission(repo: Path) -> tuple[Path, int, str]:
    submission = repo / "experiments/results/exact_ready_fixture"
    archive = submission / "archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    inflate = submission / "inflate.sh"
    inflate.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n",
        encoding="utf-8",
    )
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text(
        f"archive.zip sha256={archive_sha} bytes={archive_bytes}\n",
        encoding="utf-8",
    )
    _write_json(
        submission / "archive_manifest.json",
        {
            "score_claim": False,
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
        },
    )
    (repo / "upstream").mkdir(parents=True, exist_ok=True)
    (repo / "upstream/evaluate.py").write_text("# fixture\n", encoding="utf-8")
    return submission, archive_bytes, archive_sha


def _make_queue(repo: Path, submission: Path, archive_bytes: int, archive_sha: str) -> Path:
    return _write_json(
        repo / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "fixture_candidate",
                    "lane_id": "fixture_lane",
                    "archive_path": (submission / "archive.zip").relative_to(repo).as_posix(),
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "ready_for_exact_eval_dispatch": False,
                    "score_claim": False,
                    "predicted_contest_cpu_gha": 0.1,
                    "score_affecting_payload_changed": True,
                    "charged_bits_changed": True,
                    "dispatch_blockers": [
                        "optimizer_candidate_queue_is_planning_only",
                        "requires_exact_eval_readiness_gate",
                        "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
                    ],
                }
            ],
            "dispatch_ready": [],
        },
    )


def _write_pr101_runtime_proof(
    submission: Path,
    archive_sha: str,
    *,
    proven: bool = True,
) -> Path:
    return _write_json(
        submission / "runtime_consumption_proof.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "inflate_sh_routes_to_packet_inflate_py": True,
            "runtime_consumption_proven_for_supported_bias_params": proven,
            "archive_unchanged_proof": {"archive_sha256": archive_sha},
        },
    )


def _add_required_runtime_proof_fields(
    queue: Path,
    submission: Path,
    repo: Path,
    *,
    status: str,
) -> None:
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["runtime_consumption_proof_required"] = True
    row["runtime_consumption_proof_status"] = status
    row["runtime_consumption_proof_path"] = (
        submission / "runtime_consumption_proof.json"
    ).relative_to(repo).as_posix()
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_promotes_byte_closed_candidate_without_score_claim(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    promoted = result["promoted_queue"]
    assert promoted["dispatch_ready_count"] == 1
    row = promoted["dispatch_ready"][0]
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["target_modes"] == ["contest_exact_eval"]
    assert row["archive_sha256"] == archive_sha
    assert row["archive_bytes"] == archive_bytes
    assert "predicted_contest_cpu_gha" not in row
    assert row["dispatch_blockers"] == []
    assert row["runtime_tree_sha256"]
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["cpu_or_proxy_score_not_cuda_evidence"] is True
    assert row["cuda_gap_review_required_before_promotion"] is True
    assert promoted["evidence_boundary"]["cpu_or_proxy_score_not_cuda_evidence"] is True


def test_refuses_pr101_runtime_packet_without_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="missing")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "runtime_consumption_proof_missing" in result["report"]["blockers"]
    assert "runtime_consumption_proof_file_missing" in result["report"]["blockers"]


def test_refuses_pr101_runtime_packet_when_runtime_consumption_not_proven(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_pr101_runtime_proof(submission, archive_sha, proven=False)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "runtime_consumption_proof_not_proven" in result["report"]["blockers"]


def test_promotes_pr101_runtime_packet_with_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    proof_path = _write_pr101_runtime_proof(submission, archive_sha)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["runtime_consumption_proof_path"] == proof_path.relative_to(tmp_path).as_posix()
    assert row["runtime_consumption_proof_sha256"]
    assert row["runtime_consumption_proof_schema"] == "pr101_kaggle_proxy_runtime_consumption_proof_v1"


def test_promoted_queue_passes_existing_dispatcher_readiness_loader(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    promoted_path = tmp_path / "promoted.json"
    promoted_path.write_text(
        json.dumps(result["promoted_queue"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tool = _load_parallel_dispatch_tool()
    rows = tool._load_top_k(promoted_path, 1, active_floor_archive_bytes=None)

    assert [row["candidate_id"] for row in rows] == ["fixture_candidate"]


def test_refuses_kaggle_proxy_row_without_archive(tmp_path: Path) -> None:
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "proxy_only",
                    "lane_id": "kaggle_proxy_sweep",
                    "proxy_only": True,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_blockers": [
                        "optimizer_candidate_queue_is_planning_only",
                        "kaggle_proxy_output_requires_archive_builder_promotion",
                        "no_archive_zip_emitted",
                    ],
                }
            ],
            "dispatch_ready": [],
        },
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "proxy_only",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "source_row_proxy_only" in result["report"]["blockers"]
    assert "archive_path_missing" in result["report"]["blockers"]
    assert "score_affecting_change_proof_missing" in result["report"]["blockers"]


def test_exact_readiness_cli_import_does_not_require_repo_root_on_sys_path(
    tmp_path: Path, monkeypatch
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    original_path = list(sys.path)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in original_path if entry not in {"", str(REPO_ROOT)}],
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True


def test_refuses_archive_sha_mismatch(tmp_path: Path) -> None:
    submission, archive_bytes, _archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, "a" * 64)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "archive_sha256_mismatch" in result["report"]["blockers"]


def test_refuses_non_executable_inflate(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    inflate = submission / "inflate.sh"
    inflate.chmod(inflate.stat().st_mode & ~(
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ))
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "inflate_sh_not_executable" in result["report"]["blockers"]


def test_refuses_above_active_floor_without_override(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=1,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("above_active_floor_archive_bytes_without_operator_override")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_cosmetic_candidate_without_change_proof(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0].pop("score_affecting_payload_changed")
    payload["top_k"][0].pop("charged_bits_changed")
    queue.write_text(json.dumps(payload), encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "score_affecting_change_proof_missing" in result["report"]["blockers"]


def test_refuses_same_lane_active_claim(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    timestamp = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| {timestamp} | test | fixture_lane | modal | job1 | 2026-05-10T01:00:00Z | active_dispatching | cost=$0.50 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("same_lane_active_dispatch_claim")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_negative_for_same_archive(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.35 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("same_lane_terminal_negative_for_same_archive")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_cuda_score_not_below_floor_for_same_archive(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score_recomputed=0.22650343150032118 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        active_floor_score=0.2089810755823297,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith(
            "same_lane_terminal_score_not_below_active_floor_for_same_archive"
        )
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_cuda_score_already_below_floor_for_same_archive(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score_recomputed=1.95e-1 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        active_floor_score=0.2089810755823297,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith(
            "same_lane_terminal_score_already_below_active_floor_for_same_archive"
        )
        for blocker in result["report"]["blockers"]
    )


def test_allows_runtime_changed_candidate_after_different_runtime_terminal(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    initial = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    runtime_sha = initial["promoted_queue"]["dispatch_ready"][0]["runtime_tree_sha256"]
    old_runtime_sha = "0" * 64 if runtime_sha != "0" * 64 else "1" * 64
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["score_affecting_payload_changed"] = False
    row["charged_bits_changed"] = False
    row["score_affecting_runtime_changed"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:01:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.35 |\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | active_dispatching | archive_sha={archive_sha}; runtime_tree_sha={old_runtime_sha}; score_claim=false_until_modal_validation |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"] is not None


def test_allows_same_lane_terminal_infra_failure_for_same_archive(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | failed_runtime_dependency_missing_constriction | archive_sha={archive_sha}; no score result |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"] is not None
