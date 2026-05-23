# SPDX-License-Identifier: MIT
from __future__ import annotations

import datetime as dt
import hashlib
import json
import stat
import zipfile
from pathlib import Path

from tac.optimizer.exact_dispatch_authority import exact_dispatch_authority
from tac.optimizer.exact_readiness import promote_candidate_for_exact_eval


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _ready_row(repo: Path) -> dict[str, object]:
    submission = repo / "experiments/results/exact_dispatch_authority_fixture"
    archive = submission / "archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
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
    queue = _write_json(
        repo / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "fixture_candidate",
                    "lane_id": "fixture_lane",
                    "archive_path": archive.relative_to(repo).as_posix(),
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "ready_for_exact_eval_dispatch": False,
                    "score_claim": False,
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
    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=repo,
        active_floor_archive_bytes=None,
    )
    return result["promoted_queue"]["dispatch_ready"][0]


def _write_claims(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for timestamp, platform, job_id, status in rows:
        lines.append(
            "| "
            f"{timestamp} | codex | fixture_lane | {platform} | {job_id} | "
            f"{timestamp} | {status} | active claim policy test |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recent_claim_timestamp() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def test_exact_dispatch_authority_preclaim_policy_treats_active_claim_as_conflict(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(claims, [(_recent_claim_timestamp(), "lightning", "job-1", "running")])

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert verdict.authorized is False
    assert any(
        blocker.startswith("same_lane_active_dispatch_claim:fixture_lane:job-1")
        for blocker in verdict.blockers
    )


def test_exact_dispatch_authority_require_active_claim_blocks_missing_claim(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
        claim_policy="require_active_claim",
        required_claim_platform="lightning",
        required_claim_instance_job_ids=["job-1"],
    )

    assert verdict.authorized is False
    assert (
        "active_dispatch_claim_required_not_found:platform=lightning:job_id=job-1"
        in verdict.blockers
    )


def test_exact_dispatch_authority_require_active_claim_accepts_matching_claim(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(claims, [(_recent_claim_timestamp(), "lightning", "job-1", "running")])

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
        claim_policy="require_active_claim",
        required_claim_platform="lightning",
        required_claim_instance_job_ids=["job-1"],
    )

    assert verdict.authorized is True
    assert verdict.blockers == ()
    assert verdict.facts["claim_policy"] == "require_active_claim"


def test_exact_dispatch_authority_require_active_claim_respects_terminal_closeout(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(
        claims,
        [
            ("2026-05-17T12:10:00Z", "lightning", "job-1", "completed_contest_cuda"),
            ("2026-05-17T12:00:00Z", "lightning", "job-1", "running"),
        ],
    )

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
        claim_policy="require_active_claim",
        required_claim_platform="lightning",
        required_claim_instance_job_ids=["job-1"],
    )

    assert verdict.authorized is False
    assert (
        "active_dispatch_claim_required_not_found:platform=lightning:job_id=job-1"
        in verdict.blockers
    )


def test_exact_dispatch_authority_requires_contest_target_metadata(tmp_path: Path) -> None:
    row = _ready_row(tmp_path)
    row["target_modes"] = []

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert "contest_exact_eval_target_mode_missing" in verdict.blockers
