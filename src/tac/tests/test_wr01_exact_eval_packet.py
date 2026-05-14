# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APPLY_TRANSFORM_TOOL = "tac.hnerv_wavelet_apply_transform.build_wavelet_apply_transform_candidate"
RUNTIME_DECODE_VALIDATION_SCHEMA = "hnerv_wavelet_runtime_decode_validation.v1"
RUNTIME_DECODE_REVIEW_SCHEMA = "hnerv_wavelet_compress_time_runtime_decode_review.v1"
RUNTIME_APPLY_SCHEMA = "hnerv_wavelet_runtime_apply.v1"
RUNTIME_DISPATCH_BLOCKERS = [
    "requires_archive_manifest_preflight",
    "requires_component_response_or_exact_cuda_eval",
    "requires_lane_dispatch_claim",
    "requires_exact_cuda_auth_eval",
]


def test_wr01_exact_eval_packet_reports_missing_env_without_dispatch(tmp_path: Path) -> None:
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env.pop(key, None)
    out = tmp_path / "packet.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--claims-path",
            str(tmp_path / "claims.md"),
            "--rate-only-priority-packet",
            str(tmp_path / "missing_rate_only_priority.json"),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["schema"] == "wr01_exact_eval_operator_packet_v1"
    assert payload["packet_kind"] == "wr01_exact_eval_operator_packet"
    assert payload["candidate_id"] == "wr01_apply_pr106x_half"
    assert payload["family_group"] == "hnerv_wavelet_wr01_apply"
    assert payload["evidence_grade"] == "empirical_archive_candidate_until_exact_cuda"
    assert payload["byte_delta"] == -9
    assert math.isclose(payload["expected_total_score_delta"], -9 * 25 / 37_545_489)
    assert payload["interaction_assumptions"]
    assert payload["proxy_row"] is False
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["dispatch_gate"] == "blocked_operator_lane_gates_until_env_claim_approval"
    assert payload["dispatch_unlocked"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_exact_eval_dispatch_claim"] is False
    assert payload["byte_custody_exact_eval_candidate_ready"] is True
    assert payload["candidate_static_preflight_ready"] is True
    assert payload["static_packet_ready"] is True
    assert payload["ready_for_submit"] is False
    assert payload["static_blockers"] == []
    assert payload["runtime_decode_gate_ready"] is True
    assert payload["runtime_decode_gate"]["ready"] is True
    assert payload["runtime_decode_gate"]["blockers"] == []
    assert "missing_lightning_environment" in payload["blockers"]
    assert "missing_lightning_environment" in payload["operator_lane_blockers"]
    assert payload["source_archive_sha256"] == "d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e"
    assert payload["source_archive_bytes"] == 186231
    assert "pre_submission_compliance_failed" not in payload["blockers"]
    assert "pre_submission_compliance_failed" not in payload["static_blockers"]
    assert "dry_run_queue_payload_section_diff_mismatch" not in payload["blockers"]
    assert "lightning_dry_run_not_ready" not in payload["blockers"]
    assert "missing_active_lane_dispatch_claim" in payload["blockers"]
    assert "missing_operator_exact_cuda_approval" in payload["blockers"]
    assert payload["preflight_ready"] is True
    assert payload["compliance_ok"] is True
    assert payload["compliance_failure_summary"]["failed_count"] == 0
    assert payload["payload_diff_ready"] is True
    assert payload["dry_run_ready"] is True
    assert payload["dry_run_submit_readiness"]["remote_preflight_only"] is True
    assert payload["dry_run_queue_metadata"]["mismatches"] == []
    assert payload["operator_approved_exact_cuda"] is False
    assert payload["adversarial_priority_review"]["ready"] is True
    assert payload["adversarial_priority_review"]["blockers"] == []
    assert payload["artifact_flag_violations"] == []
    assert "--stage-workspace" in payload["commands"]["submit"]
    assert "'$LIGHTNING_SSH_TARGET'" not in payload["commands"]["submit"]
    assert "--remote $LIGHTNING_SSH_TARGET" in payload["commands"]["submit"]
    assert "tools/claim_lane_dispatch.py" in payload["commands"]["claim"]
    assert "--predicted-eta-utc 2026-05-06T07:30Z" not in payload["commands"]["claim"]
    next_steps = payload["operator_next_steps"]
    assert next_steps["schema"] == "wr01_operator_next_steps_v1"
    assert next_steps["copy_safe"] is True
    assert next_steps["must_run_in_order"] is True
    assert next_steps["first_remote_gpu_step"] == "submit_exact_cuda"
    assert next_steps["current_blockers"] == payload["blockers"]
    assert [step["id"] for step in next_steps["steps"]] == [
        "verify_lightning_env",
        "refresh_static_packet_no_dispatch",
        "review_adversarial_priority",
        "claim_lane_no_dispatch",
        "refresh_packet_with_operator_exact_cuda_approval",
        "assert_packet_ready_for_submit",
        "submit_exact_cuda",
        "harvest_after_completion",
    ]
    assert next_steps["steps"][0]["dispatches_remote_gpu"] is False
    assert "LIGHTNING_SSH_TARGET" in next_steps["steps"][0]["copy_safe_command"]
    assert "adversarial priority gate" in next_steps["steps"][2]["copy_safe_command"]
    assert "--operator-approved-exact-cuda" in next_steps["steps"][4]["copy_safe_command"]
    assert "ready_for_submit=true" in next_steps["steps"][5]["copy_safe_command"]
    assert next_steps["steps"][6]["dispatches_remote_gpu"] is True
    assert "--stage-workspace --submit" in next_steps["steps"][6]["copy_safe_command"]


def test_wr01_exact_eval_packet_builds_release_surface_and_refreshes_static_compliance(
    tmp_path: Path,
) -> None:
    fixture = _write_matching_packet_artifacts(
        tmp_path,
        archive_member_payload=b"candidate payload",
    )
    result_dir = fixture["result_dir"]
    inflate = result_dir / "source_inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(0o755)
    claims_path = tmp_path / "claims.md"
    out = tmp_path / "packet.json"
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env.pop(key, None)

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--job-name",
            "job",
            "--lane-id",
            "lane",
            "--archive",
            str(fixture["archive"]),
            "--baseline-json",
            str(fixture["baseline"]),
            "--inflate-sh",
            str(inflate),
            "--result-dir",
            str(result_dir),
            "--archive-sha256",
            fixture["archive_sha256"],
            "--archive-bytes",
            str(fixture["archive_bytes"]),
            "--claims-path",
            str(claims_path),
            "--rate-only-priority-packet",
            str(tmp_path / "missing_rate_only_priority.json"),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--build-release-surface",
            "--refresh-static-compliance",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    release_surface = result_dir / "release_surface"
    assert _sha256((release_surface / "archive.zip").read_bytes()) == fixture["archive_sha256"]
    assert os.access(release_surface / "inflate.sh", os.X_OK)
    report = (release_surface / "report.txt").read_text(encoding="utf-8")
    assert f"archive_sha256: {fixture['archive_sha256']}" in report
    assert "score_claim: false" in report
    surface_manifest = json.loads((release_surface / "archive_manifest.json").read_text(encoding="utf-8"))
    assert surface_manifest["schema"] == "wr01_release_surface_manifest_v1"
    assert surface_manifest["score_claim"] is False
    assert surface_manifest["dispatch_attempted"] is False
    assert surface_manifest["candidate_archive_sha256"] == fixture["archive_sha256"]
    assert surface_manifest["manifest_links"]["candidate_manifest"]["exists"] is True

    compliance = json.loads((result_dir / "pre_submission_compliance.json").read_text(encoding="utf-8"))
    assert compliance["passed"] is True
    assert compliance["submission_dir"]["path"].endswith("release_surface")
    assert compliance["archive_manifest"]["path"].endswith("release_surface/archive_manifest.json")
    assert all(check["passed"] for check in compliance["checks"] if check["severity"] == "error")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["compliance_ok"] is True
    assert "pre_submission_compliance_failed" not in payload["static_blockers"]
    assert payload["static_packet_ready"] is True
    assert payload["ready_for_submit"] is False
    assert payload["dispatch_gate"] == "blocked_operator_lane_gates_until_env_claim_approval"
    assert payload["dispatch_unlocked"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["runtime_decode_gate_ready"] is True
    assert payload["byte_custody_exact_eval_candidate_ready"] is True
    assert payload["adversarial_priority_review"]["ready"] is True
    assert "missing_lightning_environment" in payload["blockers"]
    assert "missing_active_lane_dispatch_claim" in payload["blockers"]
    assert "missing_operator_exact_cuda_approval" in payload["blockers"]
    assert payload["release_surface_generation"]["files"]["archive.zip"]["sha256"] == fixture["archive_sha256"]
    assert payload["static_compliance_refresh"]["returncode"] == 0
    assert payload["release_surface_manifest_consistency"]["ready"] is True
    assert payload["release_surface_manifest_consistency"]["blockers"] == []


def test_wr01_exact_eval_packet_accepts_matching_custody_artifacts(tmp_path: Path) -> None:
    result_dir = tmp_path / "wr01"
    result_dir.mkdir()
    archive = result_dir / "candidate.zip"
    archive.write_bytes(b"candidate archive bytes")
    archive_sha256 = _sha256(archive.read_bytes())
    archive_bytes = archive.stat().st_size
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")

    source_archive_sha256 = "a" * 64
    source_payload_sha256 = "b" * 64
    candidate_payload_sha256 = "c" * 64
    changed_source_sha256 = "d" * 64
    changed_section_sha256 = "e" * 64
    manifest = result_dir / "manifest.json"
    preflight = result_dir / "public_replay_preflight.json"
    compliance = result_dir / "pre_submission_compliance.json"
    payload_diff = result_dir / "payload_section_diff_vs_pr106x.json"
    dry_run = result_dir / "lightning_exact_eval_dry_run.json"
    _write_json(
        manifest,
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "source_archive_custody_mode": "operator_supplied_source_archive_identity",
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": 123,
            "source_payload_sha256": source_payload_sha256,
            "candidate_payload_sha256": candidate_payload_sha256,
            "changed_section": {
                "name": "latents_and_sidecar_brotli",
                "source_sha256": changed_source_sha256,
                "candidate_sha256": changed_section_sha256,
            },
        },
    )
    _write_json(
        payload_diff,
        {
            "ready_for_archive_preflight": True,
            "changed_section_count": 1,
            "blockers": [],
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": 123,
            "source_payload_sha256": source_payload_sha256,
            "sections": [
                {
                    "name": "latents_and_sidecar_brotli",
                    "changed": True,
                    "source_sha256": changed_source_sha256,
                    "candidate_sha256": changed_section_sha256,
                }
            ],
        },
    )
    _write_json(
        preflight,
        {
            "ready_for_exact_eval_dispatch": True,
            "blockers": [],
            "archive": {
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                "members": [
                    {
                        "decode_smoke": {
                            "sha256": candidate_payload_sha256,
                            "format": {
                                "latents_and_sidecar_brotli": {
                                    "sha256": changed_section_sha256,
                                }
                            },
                        }
                    }
                ],
            },
        },
    )
    _write_json(
        compliance,
        {
            "passed": True,
            "checks": [],
            "archive": {
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                "members": [{"sha256": candidate_payload_sha256}],
            },
            "archive_manifest": {"path": manifest.as_posix()},
        },
    )
    _write_json(
        dry_run,
        {
            "status": "DRY_RUN",
            "submit_readiness": {"ok": True, "blockers": []},
            "spec": {
                "expected_archive_sha256": archive_sha256,
                "expected_archive_size_bytes": archive_bytes,
                "job_name": "job",
                "queue_metadata": {
                    "lane": "lane",
                    "archive_manifest": manifest.as_posix(),
                    "public_preflight": preflight.as_posix(),
                    "payload_section_diff": payload_diff.as_posix(),
                },
            },
        },
    )
    _write_matching_runtime_decode_gate_artifacts(
        result_dir=result_dir,
        manifest=manifest,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        source_archive_sha256=source_archive_sha256,
        source_archive_bytes=123,
        source_payload_sha256=source_payload_sha256,
        candidate_payload_sha256=candidate_payload_sha256,
        changed_section_name="latents_and_sidecar_brotli",
        changed_source_sha256=changed_source_sha256,
        changed_section_sha256=changed_section_sha256,
    )
    out = tmp_path / "packet.json"
    claims_path = tmp_path / "claims.md"
    _write_claims(claims_path, lane_id="lane", job_name="job")
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env[key] = "fixture"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--job-name",
            "job",
            "--lane-id",
            "lane",
            "--archive",
            str(archive),
            "--baseline-json",
            str(baseline),
            "--result-dir",
            str(result_dir),
            "--archive-sha256",
            archive_sha256,
            "--archive-bytes",
            str(archive_bytes),
            "--claims-path",
            str(claims_path),
            "--rate-only-priority-packet",
            str(tmp_path / "missing_rate_only_priority.json"),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--operator-approved-exact-cuda",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["ready_for_submit"] is True
    assert payload["ready_for_exact_eval_dispatch"] is True
    assert payload["byte_custody_exact_eval_candidate_ready"] is True
    assert payload["adversarial_priority_review"]["ready"] is True
    assert payload["runtime_decode_gate_ready"] is True
    assert payload["runtime_decode_gate_blockers"] == []
    assert payload["runtime_decode_gate"]["ready"] is True
    assert payload["runtime_decode_gate"]["blockers"] == []
    assert payload["candidate_id"] == "lane"
    assert payload["family"] == "hnerv_wavelet_wr01_apply_transform"
    assert payload["pareto_scope"] == "hnerv_wavelet_apply_transform_exact_archive"
    assert "rate-only single-member archive transform" not in payload["interaction_assumptions"]
    assert payload["byte_delta"] == archive_bytes - 123
    assert math.isclose(
        payload["expected_total_score_delta"],
        (archive_bytes - 123) * 25 / 37_545_489,
    )
    assert payload["expected_seg_dist_delta"] == 0.0
    assert payload["expected_pose_dist_delta"] == 0.0
    assert payload["conflicts_with_families"] == []
    assert payload["conflicts_with_atoms"] == []
    assert payload["blockers"] == []
    assert payload["artifact_consistency_ok"] is True
    assert payload["compliance_failure_summary"]["failed_count"] == 0
    assert payload["dry_run_queue_metadata"]["mismatches"] == []
    assert payload["static_packet_ready"] is True
    assert payload["candidate_static_preflight_ready"] is True
    assert payload["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
    assert payload["dispatch_unlocked"] is True
    assert payload["ready_for_exact_eval_dispatch_claim"] is True
    assert payload["operator_lane_blockers"] == []
    assert payload["operator_approved_exact_cuda"] is True
    assert payload["lane_claim_preflight"]["active_claim_present"] is True
    assert payload["source_archive_sha256"] == source_archive_sha256
    assert payload["source_archive_custody_mode"] == "operator_supplied_source_archive_identity"
    assert payload["source_payload_sha256"] == source_payload_sha256
    assert payload["changed_section_sha256"] == changed_section_sha256
    assert payload["artifact_flag_violations"] == []
    next_steps = payload["operator_next_steps"]
    assert next_steps["current_blockers"] == []
    assert next_steps["steps"][5]["id"] == "assert_packet_ready_for_submit"


def test_wr01_exact_eval_packet_refuses_truthy_score_or_dispatch_flags(tmp_path: Path) -> None:
    result_dir = tmp_path / "wr01"
    result_dir.mkdir()
    archive = result_dir / "candidate.zip"
    archive.write_bytes(b"candidate archive bytes")
    archive_sha256 = _sha256(archive.read_bytes())
    archive_bytes = archive.stat().st_size
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")

    source_archive_sha256 = "a" * 64
    source_payload_sha256 = "b" * 64
    candidate_payload_sha256 = "c" * 64
    changed_source_sha256 = "d" * 64
    changed_section_sha256 = "e" * 64
    manifest = result_dir / "manifest.json"
    preflight = result_dir / "public_replay_preflight.json"
    compliance = result_dir / "pre_submission_compliance.json"
    payload_diff = result_dir / "payload_section_diff_vs_pr106x.json"
    dry_run = result_dir / "lightning_exact_eval_dry_run.json"
    _write_json(
        manifest,
        {
            "score_claim": True,
            "dispatch_attempted": False,
            "source_archive_custody_mode": "operator_supplied_source_archive_identity",
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": 123,
            "source_payload_sha256": source_payload_sha256,
            "candidate_payload_sha256": candidate_payload_sha256,
            "changed_section": {
                "name": "latents_and_sidecar_brotli",
                "source_sha256": changed_source_sha256,
                "candidate_sha256": changed_section_sha256,
            },
        },
    )
    _write_json(
        payload_diff,
        {
            "score_claim": False,
            "dispatch_attempted": True,
            "ready_for_archive_preflight": True,
            "changed_section_count": 1,
            "blockers": [],
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": 123,
            "source_payload_sha256": source_payload_sha256,
            "sections": [
                {
                    "name": "latents_and_sidecar_brotli",
                    "changed": True,
                    "source_sha256": changed_source_sha256,
                    "candidate_sha256": changed_section_sha256,
                }
            ],
        },
    )
    _write_json(
        preflight,
        {
            "ready_for_exact_eval_dispatch": True,
            "blockers": [],
            "archive": {
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                "members": [
                    {
                        "decode_smoke": {
                            "sha256": candidate_payload_sha256,
                            "format": {
                                "latents_and_sidecar_brotli": {
                                    "sha256": changed_section_sha256,
                                }
                            },
                        }
                    }
                ],
            },
        },
    )
    _write_json(
        compliance,
        {
            "passed": True,
            "checks": [],
            "archive": {
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                "members": [{"sha256": candidate_payload_sha256}],
            },
            "archive_manifest": {"path": manifest.as_posix()},
        },
    )
    _write_json(
        dry_run,
        {
            "status": "DRY_RUN",
            "submit_readiness": {"ok": True, "blockers": []},
            "spec": {
                "expected_archive_sha256": archive_sha256,
                "expected_archive_size_bytes": archive_bytes,
                "job_name": "job",
                "queue_metadata": {
                    "lane": "lane",
                    "archive_manifest": manifest.as_posix(),
                    "public_preflight": preflight.as_posix(),
                    "payload_section_diff": payload_diff.as_posix(),
                },
            },
        },
    )
    out = tmp_path / "packet.json"
    claims_path = tmp_path / "claims.md"
    _write_claims(claims_path, lane_id="lane", job_name="job")
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env[key] = "fixture"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--job-name",
            "job",
            "--lane-id",
            "lane",
            "--archive",
            str(archive),
            "--baseline-json",
            str(baseline),
            "--result-dir",
            str(result_dir),
            "--archive-sha256",
            archive_sha256,
            "--archive-bytes",
            str(archive_bytes),
            "--claims-path",
            str(claims_path),
            "--rate-only-priority-packet",
            str(tmp_path / "missing_rate_only_priority.json"),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--operator-approved-exact-cuda",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["ready_for_submit"] is False
    assert payload["static_packet_ready"] is False
    assert payload["candidate_static_preflight_ready"] is False
    assert payload["dispatch_unlocked"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "artifact_score_or_dispatch_flag_violation" in payload["blockers"]
    assert payload["artifact_flag_violations"] == [
        {
            "artifact": "manifest",
            "flag": "score_claim",
            "actual": True,
            "expected": False,
        },
        {
            "artifact": "payload_section_diff",
            "flag": "dispatch_attempted",
            "actual": True,
            "expected": False,
        },
    ]


def test_wr01_exact_eval_packet_fails_closed_on_stale_release_surface_manifest(
    tmp_path: Path,
) -> None:
    fixture = _write_matching_packet_artifacts(tmp_path)
    result_dir = fixture["result_dir"]
    stale_sha = "f" * 64
    release_surface = result_dir / "release_surface"
    release_surface.mkdir()
    _write_json(
        release_surface / "archive_manifest.json",
        {
            "schema": "wr01_release_surface_manifest_v1",
            "score_claim": False,
            "dispatch_attempted": False,
            "remote_gpu_run": False,
            "candidate_archive_sha256": stale_sha,
            "candidate_archive_bytes": fixture["archive_bytes"],
            "archive": {
                "sha256": stale_sha,
                "bytes": fixture["archive_bytes"],
            },
            "manifest_links": {
                "candidate_manifest": {
                    "exists": True,
                    "repo_path": (result_dir / "manifest.json").as_posix(),
                },
                "public_replay_preflight": {
                    "exists": True,
                    "repo_path": (result_dir / "public_replay_preflight.json").as_posix(),
                },
                "payload_section_diff": {
                    "exists": True,
                    "repo_path": (result_dir / "payload_section_diff_vs_pr106x.json").as_posix(),
                },
                "runtime_decode_validation": {
                    "exists": True,
                    "repo_path": (
                        result_dir / "hnerv_wavelet_runtime_decode_validation.json"
                    ).as_posix(),
                },
                "runtime_decode_review": {
                    "exists": True,
                    "repo_path": (
                        result_dir
                        / "hnerv_wavelet_compress_time_runtime_decode_review.json"
                    ).as_posix(),
                },
            },
        },
    )

    payload = _run_packet_builder_for_fixture(tmp_path, fixture)

    assert payload["ready_for_submit"] is False
    assert payload["static_packet_ready"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "release_surface_manifest_not_ready" in payload["static_blockers"]
    assert (
        "release_surface_manifest_candidate_archive_sha256_mismatch"
        in payload["release_surface_manifest_consistency"]["blockers"]
    )
    assert (
        "release_surface_manifest_archive_sha256_mismatch"
        in payload["release_surface_manifest_consistency"]["blockers"]
    )


def test_wr01_exact_eval_packet_fails_closed_without_runtime_decode_review(
    tmp_path: Path,
) -> None:
    fixture = _write_matching_packet_artifacts(tmp_path)
    result_dir = fixture["result_dir"]
    (result_dir / "hnerv_wavelet_runtime_decode_validation.json").unlink()
    (result_dir / "hnerv_wavelet_compress_time_runtime_decode_review.json").unlink()

    payload = _run_packet_builder_for_fixture(tmp_path, fixture)

    assert payload["ready_for_submit"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["static_packet_ready"] is False
    assert payload["operator_lane_blockers"] == []
    assert "missing_artifacts" in payload["static_blockers"]
    assert "runtime_decode_gate_not_ready" in payload["static_blockers"]
    assert payload["runtime_decode_gate_ready"] is False
    assert "runtime_decode_validation_file_missing" in payload["runtime_decode_gate_blockers"]
    assert "runtime_decode_review_file_missing" in payload["runtime_decode_gate_blockers"]


def test_wr01_exact_eval_packet_requires_sha_linked_scoreless_runtime_review(
    tmp_path: Path,
) -> None:
    fixture = _write_matching_packet_artifacts(tmp_path)
    review_path = (
        fixture["result_dir"] / "hnerv_wavelet_compress_time_runtime_decode_review.json"
    )
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["score_claim"] = True
    review_payload["dispatch_attempted"] = True
    review_payload["runtime_decode_validation_manifest_sha256"] = "f" * 64
    review_payload["dispatch_blockers"] = [
        blocker
        for blocker in review_payload["dispatch_blockers"]
        if blocker != "requires_lane_dispatch_claim"
    ]
    _write_json_with_self_hash(review_path, review_payload)

    payload = _run_packet_builder_for_fixture(tmp_path, fixture)

    assert payload["ready_for_submit"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["static_packet_ready"] is False
    assert payload["operator_lane_blockers"] == []
    assert "runtime_decode_gate_not_ready" in payload["static_blockers"]
    assert "artifact_score_or_dispatch_flag_violation" in payload["static_blockers"]
    gate_blockers = payload["runtime_decode_gate_blockers"]
    assert "runtime_decode_review_score_claim_not_false" in gate_blockers
    assert "runtime_decode_review_dispatch_attempted_not_false" in gate_blockers
    assert (
        "runtime_decode_review_runtime_decode_validation_sha256_mismatch"
        in gate_blockers
    )
    assert (
        "runtime_decode_review_missing_dispatch_blocker:requires_lane_dispatch_claim"
        in gate_blockers
    )
    assert {
        "artifact": "runtime_decode_review",
        "flag": "score_claim",
        "actual": True,
        "expected": False,
    } in payload["artifact_flag_violations"]


def test_wr01_exact_eval_packet_adversarial_review_defers_to_rate_only_candidate(
    tmp_path: Path,
) -> None:
    fixture = _write_matching_packet_artifacts(tmp_path)
    rate_only_packet = tmp_path / "q10_rate_only_packet.json"
    _write_json(
        rate_only_packet,
        {
            "schema": "hnerv_lowlevel_exact_eval_operator_packet_v1",
            "candidate_id": "pr106_q10_151byte_brotli",
            "family": "hnerv_lowlevel_brotli_repack",
            "pareto_scope": "hnerv_rate_only_exact_archive",
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "score_claim": False,
            "dispatch_attempted": False,
            "static_packet_ready": True,
            "static_blockers": [],
            "ready_for_submit": False,
            "byte_delta": -151,
            "archive_sha256": "a" * 64,
            "archive_bytes": 186088,
            "source_archive_sha256": "b" * 64,
            "source_archive_bytes": 186239,
        },
    )
    claims_path = tmp_path / "claims.md"
    _write_claims(claims_path, lane_id="lane", job_name="job")
    out = tmp_path / "packet.json"
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env[key] = "fixture"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--job-name",
            "job",
            "--lane-id",
            "lane",
            "--archive",
            str(fixture["archive"]),
            "--baseline-json",
            str(fixture["baseline"]),
            "--result-dir",
            str(fixture["result_dir"]),
            "--archive-sha256",
            str(fixture["archive_sha256"]),
            "--archive-bytes",
            str(fixture["archive_bytes"]),
            "--claims-path",
            str(claims_path),
            "--rate-only-priority-packet",
            str(rate_only_packet),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--operator-approved-exact-cuda",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["static_packet_ready"] is True
    assert payload["byte_custody_exact_eval_candidate_ready"] is True
    assert payload["ready_for_submit"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["operator_lane_blockers"] == [
        "adversarial_priority_review_prioritizes_rate_only_candidate"
    ]
    review = payload["adversarial_priority_review"]
    assert review["ready"] is False
    assert review["priority_decision"] == "defer_wr01_behind_hnerv_rate_only_reference"
    assert review["rate_only_reference"]["candidate_id"] == "pr106_q10_151byte_brotli"
    assert review["rate_only_reference"]["byte_delta"] == -151
    assert review["wr01_candidate"]["byte_delta"] == payload["byte_delta"]


def test_wr01_exact_eval_packet_blocks_noop_hashes_and_missing_source_custody(tmp_path: Path) -> None:
    fixture = _write_matching_packet_artifacts(
        tmp_path,
        source_archive_custody_mode="missing_source_archive_identity_fail_closed",
        source_payload_sha256="b" * 64,
        candidate_payload_sha256="b" * 64,
        changed_source_sha256="d" * 64,
        changed_section_sha256="d" * 64,
    )
    claims_path = tmp_path / "claims.md"
    _write_claims(claims_path, lane_id="lane", job_name="job")
    out = tmp_path / "packet.json"
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env[key] = "fixture"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--job-name",
            "job",
            "--lane-id",
            "lane",
            "--archive",
            str(fixture["archive"]),
            "--baseline-json",
            str(fixture["baseline"]),
            "--result-dir",
            str(fixture["result_dir"]),
            "--archive-sha256",
            fixture["archive_sha256"],
            "--archive-bytes",
            str(fixture["archive_bytes"]),
            "--claims-path",
            str(claims_path),
            "--rate-only-priority-packet",
            str(tmp_path / "missing_rate_only_priority.json"),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--operator-approved-exact-cuda",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["ready_for_submit"] is False
    assert payload["static_packet_ready"] is False
    assert payload["candidate_static_preflight_ready"] is False
    assert payload["dispatch_gate"] == "blocked_static_packet_ready_until_static_blockers_clear"
    assert "manifest_source_archive_custody_mode_invalid" in payload["blockers"]
    assert "manifest_candidate_payload_sha256_equals_source_payload_sha256_noop" in payload["blockers"]
    assert "manifest_changed_section_candidate_sha256_equals_source_sha256_noop" in payload["blockers"]
    assert "dry_run_job_name_mismatch" not in payload["blockers"]


def _run_packet_builder_for_fixture(tmp_path: Path, fixture: dict[str, object]) -> dict:
    out = tmp_path / "packet.json"
    claims_path = tmp_path / "claims.md"
    _write_claims(claims_path, lane_id="lane", job_name="job")
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env[key] = "fixture"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--job-name",
            "job",
            "--lane-id",
            "lane",
            "--archive",
            str(fixture["archive"]),
            "--baseline-json",
            str(fixture["baseline"]),
            "--result-dir",
            str(fixture["result_dir"]),
            "--archive-sha256",
            str(fixture["archive_sha256"]),
            "--archive-bytes",
            str(fixture["archive_bytes"]),
            "--claims-path",
            str(claims_path),
            "--rate-only-priority-packet",
            str(tmp_path / "missing_rate_only_priority.json"),
            "--now-utc",
            "2026-05-06T10:00:00Z",
            "--operator-approved-exact-cuda",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )
    return json.loads(out.read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _manifest_sha256_excluding_self(payload: dict) -> str:
    stripped = {
        key: value for key, value in payload.items() if key != "manifest_sha256_excluding_self"
    }
    return _sha256(
        (
            json.dumps(stripped, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    )


def _write_json_with_self_hash(path: Path, payload: dict) -> dict:
    hashed = dict(payload)
    hashed["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(hashed)
    _write_json(path, hashed)
    return hashed


def _write_single_member_zip(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        zf.writestr(info, payload)


def _write_claims(path: Path, *, lane_id: str, job_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims - test fixture",
                "",
                "## Claims (newest first)",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                (
                    f"| 2026-05-06T09:30:00Z | codex:test | {lane_id} | lightning | "
                    f"{job_name} | 2026-05-06T11:00:00Z | active_exact_eval | fixture |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_matching_packet_artifacts(
    root: Path,
    *,
    source_archive_custody_mode: str = "operator_supplied_source_archive_identity",
    source_archive_sha256: str = "a" * 64,
    source_payload_sha256: str = "b" * 64,
    candidate_payload_sha256: str = "c" * 64,
    changed_source_sha256: str = "d" * 64,
    changed_section_sha256: str = "e" * 64,
    archive_member_payload: bytes | None = None,
) -> dict[str, object]:
    result_dir = root / "wr01_fixture"
    result_dir.mkdir()
    archive = result_dir / "candidate.zip"
    if archive_member_payload is None:
        archive.write_bytes(b"candidate archive bytes")
    else:
        _write_single_member_zip(archive, archive_member_payload)
        candidate_payload_sha256 = _sha256(archive_member_payload)
    archive_sha256 = _sha256(archive.read_bytes())
    archive_bytes = archive.stat().st_size
    baseline = root / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    source_archive_bytes = 123
    manifest = result_dir / "manifest.json"
    preflight = result_dir / "public_replay_preflight.json"
    compliance = result_dir / "pre_submission_compliance.json"
    payload_diff = result_dir / "payload_section_diff_vs_pr106x.json"
    dry_run = result_dir / "lightning_exact_eval_dry_run.json"
    _write_json(
        manifest,
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "source_archive_custody_mode": source_archive_custody_mode,
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": source_archive_bytes,
            "source_payload_sha256": source_payload_sha256,
            "candidate_payload_sha256": candidate_payload_sha256,
            "changed_section": {
                "name": "latents_and_sidecar_brotli",
                "source_sha256": changed_source_sha256,
                "candidate_sha256": changed_section_sha256,
            },
        },
    )
    _write_json(
        payload_diff,
        {
            "ready_for_archive_preflight": True,
            "changed_section_count": 1,
            "blockers": [],
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": source_archive_bytes,
            "source_payload_sha256": source_payload_sha256,
            "sections": [
                {
                    "name": "latents_and_sidecar_brotli",
                    "changed": True,
                    "source_sha256": changed_source_sha256,
                    "candidate_sha256": changed_section_sha256,
                }
            ],
        },
    )
    _write_json(
        preflight,
        {
            "ready_for_exact_eval_dispatch": True,
            "blockers": [],
            "archive": {
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                "members": [
                    {
                        "decode_smoke": {
                            "sha256": candidate_payload_sha256,
                            "format": {
                                "latents_and_sidecar_brotli": {
                                    "sha256": changed_section_sha256,
                                }
                            },
                        }
                    }
                ],
            },
        },
    )
    _write_json(
        compliance,
        {
            "passed": True,
            "checks": [],
            "archive": {
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                "members": [{"sha256": candidate_payload_sha256}],
            },
            "archive_manifest": {"path": manifest.as_posix()},
        },
    )
    _write_json(
        dry_run,
        {
            "status": "DRY_RUN",
            "submit_readiness": {"ok": True, "blockers": []},
            "spec": {
                "expected_archive_sha256": archive_sha256,
                "expected_archive_size_bytes": archive_bytes,
                "name": "job",
                "queue_metadata": {
                    "lane": "lane",
                    "archive_manifest": manifest.as_posix(),
                    "public_preflight": preflight.as_posix(),
                    "payload_section_diff": payload_diff.as_posix(),
                },
            },
        },
    )
    _write_matching_runtime_decode_gate_artifacts(
        result_dir=result_dir,
        manifest=manifest,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        source_archive_sha256=source_archive_sha256,
        source_archive_bytes=source_archive_bytes,
        source_payload_sha256=source_payload_sha256,
        candidate_payload_sha256=candidate_payload_sha256,
        changed_section_name="latents_and_sidecar_brotli",
        changed_source_sha256=changed_source_sha256,
        changed_section_sha256=changed_section_sha256,
    )
    return {
        "archive": archive,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "baseline": baseline,
        "result_dir": result_dir,
    }


def _write_matching_runtime_decode_gate_artifacts(
    *,
    result_dir: Path,
    manifest: Path,
    archive_sha256: str,
    archive_bytes: int,
    source_archive_sha256: str,
    source_archive_bytes: int,
    source_payload_sha256: str,
    candidate_payload_sha256: str,
    changed_section_name: str,
    changed_source_sha256: str,
    changed_section_sha256: str,
) -> None:
    dispatch_blockers = list(RUNTIME_DISPATCH_BLOCKERS)
    validation = result_dir / "hnerv_wavelet_runtime_decode_validation.json"
    review = result_dir / "hnerv_wavelet_compress_time_runtime_decode_review.json"
    validation_payload = _write_json_with_self_hash(
        validation,
        {
            "schema": RUNTIME_DECODE_VALIDATION_SCHEMA,
            "manifest_path": validation.as_posix(),
            "validation_mode": "local_wr01_runtime_decode_validation_not_score",
            "ready_for_runtime_decode_review": True,
            "ready_for_archive_preflight": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "exact_cuda_auth_eval": False,
            "blockers": [],
            "dispatch_blockers": dispatch_blockers,
            "source_archive_sha256": source_archive_sha256,
            "source_archive_bytes": source_archive_bytes,
            "candidate_archive_sha256": archive_sha256,
            "candidate_archive_bytes": archive_bytes,
            "source_payload_sha256": source_payload_sha256,
            "candidate_payload_sha256": candidate_payload_sha256,
            "section_name": changed_section_name,
            "changed_section_only": True,
        },
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.setdefault("source_raw_sha256", source_payload_sha256)
    manifest_payload.setdefault("candidate_raw_sha256", candidate_payload_sha256)
    runtime_apply = {
        "schema": RUNTIME_APPLY_SCHEMA,
        "status": "applied",
        "ready_for_runtime_apply_review": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "section_name": changed_section_name,
        "source_section_sha256": changed_source_sha256,
        "candidate_section_sha256": changed_section_sha256,
        "source_raw_sha256": manifest_payload["source_raw_sha256"],
        "candidate_raw_sha256": manifest_payload["candidate_raw_sha256"],
        "applied_atom_ids": ["wr01_fixture_atom"],
        "applied_atom_count": 1,
    }
    manifest_payload.update(
        {
            "tool": APPLY_TRANSFORM_TOOL,
            "manifest_path": manifest.as_posix(),
            "ready_for_archive_preflight": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": dispatch_blockers,
            "runtime_apply": runtime_apply,
            "runtime_decode_validation_manifest_path": validation.as_posix(),
            "runtime_decode_validation_manifest_sha256": validation_payload[
                "manifest_sha256_excluding_self"
            ],
            "runtime_decode_validation": validation_payload,
        }
    )
    manifest_payload = _write_json_with_self_hash(manifest, manifest_payload)
    _write_json_with_self_hash(
        review,
        {
            "schema": RUNTIME_DECODE_REVIEW_SCHEMA,
            "manifest_path": review.as_posix(),
            "status": "ready",
            "ready_for_runtime_apply_review": True,
            "ready_for_decode_validation_review": True,
            "ready_for_archive_preflight": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "blockers": [],
            "runtime_apply_blockers": [],
            "decode_validation_blockers": [],
            "dispatch_blockers": dispatch_blockers,
            "runtime_apply_manifest_path": manifest.as_posix(),
            "runtime_apply_manifest_sha256": manifest_payload["manifest_sha256_excluding_self"],
            "runtime_decode_validation_schema": RUNTIME_DECODE_VALIDATION_SCHEMA,
            "runtime_decode_validation_manifest_path": validation.as_posix(),
            "runtime_decode_validation_manifest_sha256": validation_payload[
                "manifest_sha256_excluding_self"
            ],
            "runtime_decode_validation_ready": True,
            "runtime_apply_candidate_archive_sha256": archive_sha256,
            "runtime_apply_candidate_archive_bytes": archive_bytes,
            "runtime_apply_changed_section_name": changed_section_name,
        },
    )
