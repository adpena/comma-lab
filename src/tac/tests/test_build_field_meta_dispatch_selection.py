from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tools.build_field_meta_dispatch_selection import build_selection_report

REPO = Path(__file__).resolve().parents[3]


def test_field_meta_selector_accepts_only_strict_preflighted_byte_runtime_closed_packet(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(tmp_path, candidate_id="closed_candidate")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["ready_for_exact_eval_dispatch"] is True
    assert report["ready_candidate_count"] == 1
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["strict_candidate_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["next_required_proof"] == [
        "active_lane_dispatch_claim_before_remote_gpu_submit",
        "exact_cuda_auth_eval_on_selected_archive_bytes",
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
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "strict_candidate_preflight_not_ready" in row["candidate_blockers"]
    assert any(
        blocker.startswith("strict:dispatch_gate_blocked")
        for blocker in row["candidate_blockers"]
    )


def test_field_meta_selector_is_deterministic_and_orders_ready_packets_first(
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
    )

    first = build_selection_report(repo_root=REPO, manifest_paths=[blocked, ready])
    second = build_selection_report(repo_root=REPO, manifest_paths=[ready, blocked])

    assert first == second
    assert first["selected_candidate"]["candidate_id"] == "ready"
    assert [row["candidate_id"] for row in first["rows"]] == ["ready", "blocked"]


def test_build_field_meta_dispatch_selection_cli_writes_json(tmp_path: Path) -> None:
    manifest = _packet_manifest(tmp_path, candidate_id="cli_candidate")
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
    assert payload["selected_candidate"]["ready_for_exact_eval_dispatch"] is True


def _packet_manifest(
    root: Path,
    *,
    candidate_id: str,
    runtime_tree_sha256: str | None = "b" * 64,
    dispatch_gate: str = "eligible_for_cuda_auth_eval_after_lane_claim",
    expected_score_delta: float = -0.0001,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
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
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
