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


def _write_runtime_bound_pr101_proof(
    submission: Path,
    *,
    archive_sha: str,
) -> Path:
    inflate_sh_sha = hashlib.sha256((submission / "inflate.sh").read_bytes()).hexdigest()
    inflate_py_sha = hashlib.sha256((submission / "inflate.py").read_bytes()).hexdigest()
    manifest_path = _write_json(
        submission / "runtime_packet_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_packet_v1",
            "packet_dir": str(submission),
            "runtime_custody": {
                "runtime_files": [
                    {"relpath": "inflate.sh", "sha256": inflate_sh_sha},
                    {"relpath": "inflate.py", "sha256": inflate_py_sha},
                ],
            },
        },
    )
    return _write_json(
        submission / "runtime_consumption_proof.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
            "proof_kind": "fixture_runtime_bound_pr101_proof",
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "packet_dir": str(submission),
            "runtime_consumption_proven_for_supported_bias_params": True,
            "inflate_sh_routes_to_packet_inflate_py": True,
            "archive_unchanged_proof": {
                "archive_sha256": archive_sha,
            },
            "inflate_wrapper_route_proof": {
                "wrapper_invoked_packet_inflate_py": True,
                "inflate_sh_sha256": inflate_sh_sha,
                "packet_inflate_py_sha256": inflate_py_sha,
            },
            "inflate_static_bias_patch_proof": {
                "inflate_sha256": inflate_py_sha,
            },
            "inflate_runtime_bias_logic_proof": {
                "packet_inflate_function_executed": True,
                "inflate_py_sha256": inflate_py_sha,
            },
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
    )


def _ready_row(repo: Path) -> dict[str, object]:
    submission = repo / "experiments/results/exact_dispatch_authority_fixture"
    archive = submission / "archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    inflate = submission / "inflate.sh"
    inflate_py = submission / "inflate.py"
    inflate_py.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(b'')\n",
        encoding="utf-8",
    )
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "python \"$SCRIPT_DIR/inflate.py\" \"$1\" \"$2\"\n",
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
    runtime_proof = _write_runtime_bound_pr101_proof(
        submission,
        archive_sha=archive_sha,
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
                    "runtime_consumption_proof_required": True,
                    "runtime_consumption_proof_status": "present",
                    "runtime_consumption_proof_path": runtime_proof.relative_to(
                        repo
                    ).as_posix(),
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
    row = result["promoted_queue"]["dispatch_ready"][0]
    row["score_axis"] = "contest_cuda"
    row["target_score_axis"] = "contest_cuda"
    return row


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


def _stale_claim_timestamp() -> str:
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        - dt.timedelta(hours=25)
    ).isoformat().replace("+00:00", "Z")


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


def test_exact_dispatch_authority_require_active_claim_rejects_stale_claim(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(claims, [(_stale_claim_timestamp(), "lightning", "job-1", "running")])

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


def test_exact_dispatch_authority_require_active_claim_blocks_stale_claim(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(
        claims,
        [("2026-05-17T12:00:00Z", "lightning", "job-1", "running")],
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
    assert any(
        blocker.startswith(
            "same_lane_stale_nonterminal_dispatch_claim:fixture_lane:job-1"
        )
        for blocker in verdict.blockers
    )
    assert (
        "active_dispatch_claim_required_not_found:platform=lightning:job_id=job-1"
        in verdict.blockers
    )


def test_exact_dispatch_authority_require_active_claim_blocks_other_active_claims(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(
        claims,
        [
            (_recent_claim_timestamp(), "lightning", "job-1", "running"),
            (_recent_claim_timestamp(), "lightning", "job-2", "running"),
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
    assert any(
        blocker.startswith("same_lane_active_dispatch_claim:fixture_lane:job-2")
        for blocker in verdict.blockers
    )


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


def test_exact_dispatch_authority_does_not_treat_contest_mode_as_exact_eval(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row.pop("target_modes", None)
    row["contest_mode"] = True

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert "contest_exact_eval_target_mode_missing" in verdict.blockers


def test_exact_dispatch_authority_does_not_treat_deployment_target_as_exact_eval(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row.pop("target_modes", None)
    row["deployment_target"] = "contest_exact_eval"

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert "contest_exact_eval_target_mode_missing" in verdict.blockers


def test_exact_dispatch_authority_blocks_truthy_pre_dispatch_authority_fields(
    tmp_path: Path,
) -> None:
    base = _ready_row(tmp_path)
    truthy_fields = (
        "rank_or_kill_eligible",
        "score_claim_valid",
        "exact_cuda_auth_eval",
        "contest_cuda_auth_eval",
        "dispatch_attempted",
        "promotable",
    )

    for field in truthy_fields:
        row = dict(base)
        row[field] = True
        verdict = exact_dispatch_authority(
            row,
            repo_root=tmp_path,
            source="test",
            active_floor_archive_bytes=None,
        )

        assert verdict.authorized is False
        assert f"truthy_authority_field:{field}=truthy" in verdict.blockers


def test_exact_dispatch_authority_blocks_pre_dispatch_score_fields(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row["proxy_score"] = 0.1
    row["macos_cpu_score"] = 0.2

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert (
        "pre_dispatch_score_field_present:macos_cpu_score,proxy_score"
        in verdict.blockers
    )


def test_exact_dispatch_authority_requires_some_explicit_score_axis(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row.pop("score_axis", None)
    row.pop("target_score_axis", None)

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert "score_axis_missing:required=explicit_contest_axis" in verdict.blockers


def test_exact_dispatch_authority_requires_contest_score_axis_for_exact_target(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row["score_axis"] = "[macOS-MLX research-signal]"
    row["target_score_axis"] = "[macOS-MLX research-signal]"

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert (
        "score_axis_non_contest_for_exact_dispatch:"
        "score_axis=[macos_mlx_research_signal],"
        "target_score_axis=[macos_mlx_research_signal]"
    ) in verdict.blockers


def test_exact_dispatch_authority_blocks_raw_mlx_advisory_ready_bit(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row.update(
        {
            "source_schema": "mlx_scorer_response.v1",
            "source_evidence_grade": "macOS-MLX-research-signal",
            "source_evidence_tag": "[macOS-MLX research-signal]",
            "axis_tag": "[macOS-MLX research-signal]",
            "ready_for_exact_eval_dispatch": True,
        }
    )

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
    )

    assert verdict.authorized is False
    assert (
        "archive_bound_candidate_contract_requires_readiness_promotion"
        in verdict.blockers
    )
    assert (
        "archive_bound_candidate_contract_required_for_source_row"
        in verdict.blockers
    )


def test_exact_dispatch_authority_requires_declared_score_axis_when_requested(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row.pop("score_axis", None)
    row.pop("target_score_axis", None)

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        required_score_axis="contest_cuda",
    )

    assert verdict.authorized is False
    assert "score_axis_missing:required=contest_cuda" in verdict.blockers


def test_exact_dispatch_authority_blocks_wrong_score_axis_when_requested(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row["score_axis"] = "contest_cpu"
    row["target_score_axis"] = "contest_cpu"

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        required_score_axis="contest_cuda",
    )

    assert verdict.authorized is False
    assert (
        "score_axis_required:contest_cuda:declared=score_axis=contest_cpu,"
        "target_score_axis=contest_cpu"
    ) in verdict.blockers


def test_exact_dispatch_authority_blocks_conflicting_score_axis_aliases(
    tmp_path: Path,
) -> None:
    row = _ready_row(tmp_path)
    row["target_auth_axis"] = "contest_cpu"
    row["contest_axis"] = "contest_cuda"

    verdict = exact_dispatch_authority(
        row,
        repo_root=tmp_path,
        source="test",
        active_floor_archive_bytes=None,
        required_score_axis="contest_cuda",
    )

    assert verdict.authorized is False
    assert any(
        blocker.startswith("score_axis_field_mismatch:")
        for blocker in verdict.blockers
    )
