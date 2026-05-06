from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


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
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_submit"] is False
    assert "missing_lightning_environment" in payload["blockers"]
    assert payload["source_archive_sha256"] == "d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e"
    assert payload["source_archive_bytes"] == 186231
    assert "pre_submission_compliance_failed" in payload["blockers"]
    assert "dry_run_queue_payload_section_diff_mismatch" in payload["blockers"]
    assert payload["preflight_ready"] is True
    assert payload["compliance_ok"] is False
    assert payload["payload_diff_ready"] is True
    assert "--stage-workspace" in payload["commands"]["submit"]
    assert "'$LIGHTNING_SSH_TARGET'" not in payload["commands"]["submit"]
    assert "--remote $LIGHTNING_SSH_TARGET" in payload["commands"]["submit"]
    assert "tools/claim_lane_dispatch.py" in payload["commands"]["claim"]


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
            str(archive),
            "--baseline-json",
            str(baseline),
            "--result-dir",
            str(result_dir),
            "--archive-sha256",
            archive_sha256,
            "--archive-bytes",
            str(archive_bytes),
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["ready_for_submit"] is True
    assert payload["blockers"] == []
    assert payload["artifact_consistency_ok"] is True
    assert payload["source_archive_sha256"] == source_archive_sha256
    assert payload["source_payload_sha256"] == source_payload_sha256
    assert payload["changed_section_sha256"] == changed_section_sha256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
