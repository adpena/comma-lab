from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.hnerv_entropy_frontier_selector import (
    ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    RATE_ONLY_FLOOR_BLOCKER_PREFIX,
    build_hnerv_entropy_frontier_selection,
    build_rate_only_floor_proof,
    render_markdown,
)
from tac.repo_io import json_text, sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_active_pr103_byte_floor_blocks_hdm3_and_records_smaller_blocked_pr101(
    tmp_path: Path,
) -> None:
    active = _write_manifest(
        tmp_path,
        "active_pr103.json",
        {
            "tool": "tools.prove_pr103_pr106_final_runtime_packet",
            "score_claim": False,
            "passed": True,
            "candidate_archive": _archive(
                tmp_path,
                "pr103_active.zip",
                b"a" * ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
            ),
            "exact_cuda_remaining_blockers": [
                "claim dispatch lane before exact CUDA auth eval",
                "run archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
            ],
        },
    )
    pr101 = _write_manifest(
        tmp_path,
        "pr101_schema.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            **_flat_candidate(tmp_path, "pr101.zip", b"pr101"),
            "source_archive_sha256": "a" * 64,
            "source_archive_bytes": 20,
            "source_payload_sha256": "b" * 64,
            "candidate_payload_sha256": "c" * 64,
            "dispatch_blockers": [
                "pr101_schema_runtime_tree_parity_manifest_missing",
                "strict_pre_submission_compliance_json_missing",
                "lane_dispatch_claim_missing",
                "exact_cuda_auth_eval_missing",
            ],
        },
    )
    hdm3 = _write_manifest(
        tmp_path,
        "hdm3.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            **_flat_candidate(
                tmp_path,
                "hdm3.zip",
                b"h" * (ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES + 1),
            ),
            "source_archive_sha256": "d" * 64,
            "source_archive_bytes": 30,
            "source_payload_sha256": "e" * 64,
            "candidate_payload_sha256": "f" * 64,
            "exact_eval_packet_readiness": {
                "static_packet_ready": True,
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
            "dispatch_blockers": [
                "lane_dispatch_claim_missing",
                "exact_cuda_auth_eval_missing",
            ],
        },
    )
    q10 = _write_manifest(
        tmp_path,
        "q10.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            **_flat_candidate(
                tmp_path,
                "q10.zip",
                b"q" * (ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES + 2),
            ),
            "source_archive_sha256": "0" * 64,
            "source_archive_bytes": 40,
            "source_payload_sha256": "1" * 64,
            "candidate_payload_sha256": "2" * 64,
            "dispatch_blockers": [
                "requires_archive_manifest_preflight",
                "requires_lane_dispatch_claim",
                "requires_exact_cuda_auth_eval",
            ],
        },
    )

    manifest = build_hnerv_entropy_frontier_selection(
        [
            ("pr101_schema", pr101),
            ("hdm3", hdm3),
            ("q10", q10),
        ],
        active_candidates=[("active_pr103", active)],
        repo_root=tmp_path,
    )

    assert manifest["selected_next_candidate"] is None
    assert manifest["active_candidate_byte_floor"] == ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES
    assert manifest["active_rate_only_floor_archive_bytes"] == ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES
    assert manifest["active_candidate"]["label"] == "active_pr103"
    assert manifest["active_candidate"]["active_excluded"] is True
    assert manifest["blocked_smaller_than_selected"] == []
    hdm3_row = next(row for row in manifest["ranked_candidates"] if row["label"] == "hdm3")
    assert hdm3_row["exact_evaluable_after_lane_claim"] is False
    assert hdm3_row["byte_delta_vs_active_rate_only_floor"] == 1
    assert f"{RATE_ONLY_FLOOR_BLOCKER_PREFIX}:{ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES}" in hdm3_row[
        "hard_blockers"
    ]
    pr101_row = next(row for row in manifest["ranked_candidates"] if row["label"] == "pr101_schema")
    assert "pr101_schema_runtime_tree_parity_manifest_missing" in manifest[
        "ranked_candidates"
    ][0]["hard_blockers"]
    assert pr101_row["exact_evaluable_after_lane_claim"] is False
    q10_row = next(row for row in manifest["ranked_candidates"] if row["label"] == "q10")
    assert "static_exact_eval_packet_not_ready" in q10_row["blocking_reasons"]
    markdown = render_markdown(manifest)
    assert "active_candidate_byte_floor" in markdown
    assert "Ranked Candidates" in markdown
    assert RATE_ONLY_FLOOR_BLOCKER_PREFIX in markdown


def test_selects_static_ready_candidate_when_it_beats_active_byte_floor(tmp_path: Path) -> None:
    active = _write_manifest(
        tmp_path,
        "active_large.json",
        {
            "tool": "tools.prove_pr103_pr106_final_runtime_packet",
            "score_claim": False,
            "passed": True,
            "candidate_archive": _archive(
                tmp_path,
                "active_large.zip",
                b"x" * ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
            ),
            "exact_cuda_remaining_blockers": [
                "claim dispatch lane before exact CUDA auth eval",
                "run archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
            ],
        },
    )
    hdm3 = _write_manifest(
        tmp_path,
        "hdm3.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            **_flat_candidate(
                tmp_path,
                "hdm3.zip",
                b"h" * (ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES - 1),
            ),
            "source_archive_sha256": "d" * 64,
            "source_archive_bytes": 30,
            "source_payload_sha256": "e" * 64,
            "candidate_payload_sha256": "f" * 64,
            "exact_eval_packet_readiness": {
                "static_packet_ready": True,
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
        },
    )

    manifest = build_hnerv_entropy_frontier_selection(
        [("hdm3", hdm3)],
        active_candidates=[("active_large", active)],
        repo_root=tmp_path,
    )

    selected = manifest["selected_next_candidate"]
    assert manifest["active_candidate_byte_floor"] == ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES
    assert selected["label"] == "hdm3"
    assert selected["exact_evaluable_after_lane_claim"] is True


def test_blocks_rate_only_candidate_above_floor_with_arbitrary_scorer_changing_stack_path_string(
    tmp_path: Path,
) -> None:
    candidate = _write_manifest(
        tmp_path,
        "string_bypass.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            "scorer_changing_stack_path": "wr01_after_pr103_pr106_floor",
            **_flat_candidate(
                tmp_path,
                "stacked.zip",
                b"s" * (ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES + 100),
            ),
            "source_archive_sha256": "d" * 64,
            "source_archive_bytes": ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
            "source_payload_sha256": "e" * 64,
            "candidate_payload_sha256": "f" * 64,
            "exact_eval_packet_readiness": {
                "static_packet_ready": True,
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
        },
    )

    manifest = build_hnerv_entropy_frontier_selection(
        [("string_bypass", candidate)],
        repo_root=tmp_path,
    )

    row = manifest["ranked_candidates"][0]
    assert manifest["selected_next_candidate"] is None
    assert row["label"] == "string_bypass"
    assert row["candidate_archive_bytes"] == ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES + 100
    assert row["scorer_changing_stack_path_declared"] is True
    assert row["exact_evaluable_after_lane_claim"] is False
    assert f"{RATE_ONLY_FLOOR_BLOCKER_PREFIX}:{ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES}" in row[
        "hard_blockers"
    ]
    proof = row["rate_only_floor_proof"]
    assert proof["scorer_changing_stack_path_payload"] == {
        "scorer_changing_stack_path": "wr01_after_pr103_pr106_floor"
    }
    assert proof["scorer_changing_stack_path_structured"] is False
    assert proof["scorer_changing_stack_path_meaningful"] is False
    assert "scorer_changing_stack_path_proof_unstructured" in proof["blockers"]


def test_preserves_structured_scorer_changing_stack_path_payload_in_floor_proof() -> None:
    stack_path = {
        "id": "wr01_pose_residual_stack",
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": -0.0003,
        "charged_scorer_changing_bytes": 96,
        "scorer_changing_members": ["renderer.bin"],
        "runtime_consumed_scorer_changing_bytes": True,
    }
    proof = build_rate_only_floor_proof(
        {
            "scorer_changing_stack_path": stack_path,
            "exact_eval_packet_readiness": {
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
        },
        candidate_archive_bytes=ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES + 100,
        declared_rate_only=True,
    )

    assert proof["beats_active_floor"] is False
    assert proof["exact_eval_spend_allowed_by_policy"] is True
    assert proof["scorer_changing_stack_path_payload"] == {
        "scorer_changing_stack_path": stack_path
    }
    assert proof["scorer_changing_stack_path_structured"] is True
    assert proof["scorer_changing_stack_path_meaningful"] is True
    assert proof["scorer_changing_stack_path_proof"]["component_delta_proof"] is True
    assert proof["scorer_changing_stack_path_proof"]["charged_scorer_changing_bytes"] is True
    assert proof["scorer_changing_stack_path_proof"][
        "runtime_consumed_scorer_changing_bytes"
    ] is True
    assert proof["blockers"] == []


def test_missing_candidate_archive_blocks_exact_evaluable_row(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "missing.zip", b"selected-hdm3")
    missing_archive_path = tmp_path / "missing.zip"
    missing_archive_path.unlink()
    manifest_path = _write_manifest(
        tmp_path,
        "missing_archive.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            "candidate_archive_path": archive["path"],
            "candidate_archive_sha256": archive["archive_sha256"],
            "candidate_archive_bytes": archive["archive_bytes"],
            "source_archive_sha256": "d" * 64,
            "source_archive_bytes": 30,
            "source_payload_sha256": "e" * 64,
            "candidate_payload_sha256": "f" * 64,
            "exact_eval_packet_readiness": {
                "static_packet_ready": True,
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
        },
    )

    manifest = build_hnerv_entropy_frontier_selection(
        [("missing_archive", manifest_path)],
        repo_root=tmp_path,
    )

    row = manifest["ranked_candidates"][0]
    assert manifest["selected_next_candidate"] is None
    assert row["candidate_archive_exists"] is False
    assert row["exact_evaluable_after_lane_claim"] is False
    assert "candidate_archive_missing" in row["blocking_reasons"]


def test_rejects_static_packet_when_archive_or_payload_is_not_changed(tmp_path: Path) -> None:
    candidate_archive = _archive(tmp_path, "unchanged.zip", b"same")
    manifest_path = _write_manifest(
        tmp_path,
        "unchanged.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            "candidate_archive_path": candidate_archive["path"],
            "candidate_archive_sha256": candidate_archive["archive_sha256"],
            "candidate_archive_bytes": candidate_archive["archive_bytes"],
            "candidate_payload_sha256": "a" * 64,
            "source_archive_sha256": candidate_archive["archive_sha256"],
            "source_archive_bytes": candidate_archive["archive_bytes"],
            "source_payload_sha256": "a" * 64,
            "exact_eval_packet_readiness": {
                "static_packet_ready": True,
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
        },
    )

    manifest = build_hnerv_entropy_frontier_selection(
        [("unchanged", manifest_path)],
        repo_root=tmp_path,
    )

    row = manifest["ranked_candidates"][0]
    assert manifest["selected_next_candidate"] is None
    assert row["exact_evaluable_after_lane_claim"] is False
    assert row["identity_blockers"] == [
        "candidate_archive_sha256_not_byte_different",
        "candidate_payload_sha256_not_changed",
    ]


def test_selector_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    hdm3 = _write_manifest(
        tmp_path,
        "hdm3.json",
        {
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": True,
            **_flat_candidate(tmp_path, "hdm3.zip", b"selected-hdm3"),
            "source_archive_sha256": "d" * 64,
            "source_archive_bytes": 30,
            "source_payload_sha256": "e" * 64,
            "candidate_payload_sha256": "f" * 64,
            "exact_eval_packet_readiness": {
                "static_packet_ready": True,
                "remaining_dispatch_blockers": [
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            },
        },
    )
    json_out = tmp_path / "selection.json"
    md_out = tmp_path / "selection.md"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "select_hnerv_entropy_frontier_candidate.py"),
            "--candidate",
            f"hdm3={hdm3}",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--fail-if-none",
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["selected_next_candidate"]["label"] == "hdm3"
    assert payload["tool_run_manifest"]["tool"] == "tools/select_hnerv_entropy_frontier_candidate.py"
    assert "Selected Next Candidate" in md_out.read_text(encoding="utf-8")


def _archive(root: Path, name: str, data: bytes) -> dict[str, object]:
    path = root / name
    path.write_bytes(data)
    return {
        "path": name,
        "archive_bytes": len(data),
        "archive_sha256": sha256_file(path),
    }


def _flat_candidate(root: Path, name: str, data: bytes) -> dict[str, object]:
    archive = _archive(root, name, data)
    return {
        "candidate_archive_path": archive["path"],
        "candidate_archive_sha256": archive["archive_sha256"],
        "candidate_archive_bytes": archive["archive_bytes"],
    }


def _write_manifest(root: Path, name: str, payload: dict[str, object]) -> Path:
    path = root / name
    path.write_text(json_text(payload), encoding="utf-8")
    return path
