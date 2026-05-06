from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tools.build_field_meta_dispatch_selection import build_selection_report

REPO = Path(__file__).resolve().parents[3]


def test_field_meta_selector_reports_static_ready_but_not_dispatch_ready_without_claim(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="closed_candidate",
        lane_id="lane_closed_candidate",
        job_name="job_closed_candidate",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_static_preflight_ready"] is True
    assert report["candidate_static_preflight_ready_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_candidate_count"] == 0
    assert row["candidate_static_preflight_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["strict_candidate_preflight_ready"] is True
    assert row["strict_candidate_static_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["dispatch_identity_proof"]["status"] == "passed"
    assert row["dispatch_claim_proof"]["checked"] is False
    assert "missing_active_lane_dispatch_claim" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "matching_active_level2_lane_claim_for_manifest_lane_and_job",
    ]


def test_field_meta_selector_requires_lane_and_job_identity_for_static_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="identity_missing_candidate",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_local_preflight_ready"] is True
    assert report["candidate_local_preflight_ready_count"] == 1
    assert report["candidate_static_preflight_ready"] is False
    assert report["candidate_static_preflight_ready_count"] == 0
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_candidate_count"] == 0
    assert row["strict_candidate_preflight_ready"] is True
    assert row["candidate_local_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["dispatch_identity_proof"]["status"] == "blocked"
    assert row["dispatch_identity_proof"]["blockers"] == [
        "dispatch_lane_id_missing",
        "dispatch_instance_job_id_missing",
    ]
    assert "missing_dispatch_identity_for_lane_claim" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "manifest_lane_id_and_instance_job_id_for_level2_claim",
    ]


def test_field_meta_selector_requires_matching_active_claim_for_dispatch_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="claimed_candidate",
        lane_id="lane_claimed_candidate",
        job_name="job_claimed_candidate",
    )
    claims = _claims_file(tmp_path, lane_id="lane_claimed_candidate", job_name="job_claimed_candidate")

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        claims_path=claims,
        now_utc="2026-05-06T12:00:00Z",
    )

    row = report["rows"][0]
    assert report["candidate_static_preflight_ready_count"] == 1
    assert report["ready_candidate_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is True
    assert row["candidate_static_preflight_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["dispatch_claim_proof"]["checked"] is True
    assert row["dispatch_claim_proof"]["active_lane_claim"] is True
    assert row["dispatch_claim_proof"]["active_claim"]["instance_job_id"] == "job_claimed_candidate"
    assert row["next_required_proof"] == [
        "exact_cuda_auth_eval_on_selected_archive_bytes",
    ]


def test_field_meta_selector_rejects_non_zip_archive_even_with_matching_sha(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="not_a_zip",
        lane_id="lane_not_a_zip",
        job_name="job_not_a_zip",
        valid_zip=False,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["candidate_static_preflight_ready"] is False
    assert row["archive_proof"]["byte_closed"] is False
    assert row["archive_proof"]["zip_custody"]["status"] == "blocked"
    assert "archive:zip:archive_zip_unreadable" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "local_archive_file_with_matching_sha256_and_bytes"
    ]


def test_field_meta_selector_rejects_packet_without_runtime_tree_even_when_strict_preflight_passes(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="missing_runtime_tree",
        runtime_tree_sha256=None,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["strict_candidate_preflight_ready"] is True
    assert row["candidate_static_preflight_ready"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "missing_byte_closed_runtime_proof" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract"
    ]


def test_field_meta_selector_never_overrides_strict_candidate_preflight(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="strict_blocked_candidate",
        dispatch_gate="planning_only/no_remote_dispatch",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["strict_candidate_preflight_ready"] is False
    assert row["candidate_static_preflight_ready"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "strict_candidate_preflight_not_ready" in row["candidate_blockers"]
    assert any(
        blocker.startswith("strict:dispatch_gate_blocked")
        for blocker in row["candidate_blockers"]
    )


def test_field_meta_selector_is_deterministic_and_orders_static_ready_packets_first(
    tmp_path: Path,
) -> None:
    blocked = _packet_manifest(
        tmp_path / "blocked",
        candidate_id="blocked",
        runtime_tree_sha256=None,
    )
    ready = _packet_manifest(
        tmp_path / "ready",
        candidate_id="ready",
        expected_score_delta=-0.0002,
        lane_id="lane_ready",
        job_name="job_ready",
    )

    first = build_selection_report(repo_root=REPO, manifest_paths=[blocked, ready])
    second = build_selection_report(repo_root=REPO, manifest_paths=[ready, blocked])

    assert first == second
    assert first["selected_candidate"]["candidate_id"] == "ready"
    assert [row["candidate_id"] for row in first["rows"]] == ["ready", "blocked"]
    assert first["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert first["selected_candidate"]["ready_for_exact_eval_dispatch"] is False


def test_build_field_meta_dispatch_selection_cli_writes_json(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="cli_candidate",
        lane_id="lane_cli_candidate",
        job_name="job_cli_candidate",
    )
    out = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_field_meta_dispatch_selection.py"),
            "--manifest",
            str(manifest),
            "--json-out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    assert payload["selected_candidate"]["candidate_id"] == "cli_candidate"
    assert payload["candidate_static_preflight_ready_count"] == 1
    assert payload["ready_candidate_count"] == 0
    assert payload["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert payload["selected_candidate"]["ready_for_exact_eval_dispatch"] is False


def _packet_manifest(
    root: Path,
    *,
    candidate_id: str,
    runtime_tree_sha256: str | None = "b" * 64,
    dispatch_gate: str = "eligible_for_cuda_auth_eval_after_lane_claim",
    expected_score_delta: float = -0.0001,
    lane_id: str | None = None,
    job_name: str | None = None,
    valid_zip: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
    if valid_zip:
        info = zipfile.ZipInfo("x")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 0o644 << 16
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(info, f"{candidate_id} archive bytes".encode())
    else:
        archive.write_bytes(f"{candidate_id} archive bytes".encode())
    fixed_runtime = {
        "ready_for_fixed_runtime_exact_eval": True,
        "remaining_blockers": [],
    }
    if runtime_tree_sha256 is not None:
        fixed_runtime["runtime_tree_sha256"] = runtime_tree_sha256
    payload = {
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_gate": dispatch_gate,
        "dispatch_unlocked": True,
        "ready_for_exact_eval_dispatch_claim": True,
        "archive": {
            "path": archive.as_posix(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bytes": archive.stat().st_size,
        },
        "fixed_runtime_preflight": fixed_runtime,
        "byte_delta": -9,
        "expected_total_score_delta": expected_score_delta,
        "expected_information_gain_nats": 0.1,
    }
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if job_name is not None:
        payload["job_name"] = job_name
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _claims_file(root: Path, *, lane_id: str, job_name: str) -> Path:
    claims = root / "active_lane_dispatch_claims.md"
    claims.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims - test fixture",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                f"| 2026-05-06T11:55:00Z | codex:test | {lane_id} | lightning | {job_name} | 2026-05-06T12:30:00Z | active_exact_eval | test claim |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return claims
