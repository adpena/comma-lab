from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.witness_dsl.taskspace_frontier_closure_audit import (
    ClosureEvidencePaths,
    TaskspaceFrontierClosureError,
    build_taskspace_frontier_closure_audit,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path, *, upstream_score: float = 0.172) -> ClosureEvidencePaths:
    pointer = {
        "schema_version": "canonical_frontier_pointer_v1_20260519",
        "our_local_frontier_contest_cpu": {
            "score": 0.188,
            "axis": "contest_cpu",
            "archive_sha256": "a" * 64,
            "lane_id": "local",
            "hardware_substrate": "linux_x86_64_cpu",
            "measured_at_utc": "2026-07-01T00:00:00Z",
            "evidence_grade": "[contest-CPU]",
        },
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {"best_entry": {"score": upstream_score, "rank": 1, "name": "upstream"}},
        "upstream_leaderboard_snapshot_at_utc": "2026-07-26T00:00:00Z",
        "last_refreshed_utc": "2026-07-26T00:00:00Z",
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "refresh",
        "refresh_provenance": {},
        "effective_frontier": {"score": upstream_score},
    }
    ms1 = {
        "authority": {"score_claim": False},
        "realized_frozen_scorer_diagnostic": {
            "realized_d_seg": 0.00015,
            "realized_d_pose": 0.0001,
        },
        "diagnostic_conditioning_coder": {"previous_frame_exception_bytes": 700_000_000},
    }
    ms2r = {
        "authority": {"score_claim": False},
        "headline": {
            "stored_problem": {"bytes": 290_000_000, "sha256": "b" * 64},
            "diagnostic_distortions": {"realized_d_seg": 0.0011, "realized_d_pose": 0.016},
        },
    }
    g20 = {
        "selected": {"archive_bytes": 81_000, "archive_sha256": "c" * 64},
        "proof": {"full_quantized_state_equal": True},
        "truth": {
            "full_n600_receiver_replay_owed": True,
            "contest_cpu_cuda_same_bytes_owed": True,
            "candidate_claim": False,
            "score_claim": False,
        },
    }
    return ClosureEvidencePaths(
        pointer=_write(tmp_path / "pointer.json", pointer),
        ms1=_write(tmp_path / "ms1.json", ms1),
        ms2r=_write(tmp_path / "ms2r.json", ms2r),
        g20=_write(tmp_path / "g20.json", g20),
    )


def test_audit_keeps_rate_and_distortion_anchors_separate(tmp_path: Path) -> None:
    audit = build_taskspace_frontier_closure_audit(_fixtures(tmp_path))
    assert audit["competitive_target"]["score"] == 0.172
    assert audit["anchors"]["low_rate"]["archive_bytes"] == 81_000
    assert audit["anchors"]["low_rate"]["d_seg"] is None
    assert audit["anchors"]["low_distortion"]["conditional_frontier_max_archive_bytes"] is not None
    assert audit["anchors"]["receiver_closed_dense_control"]["feasible_distortions_at_zero_rate"] is False
    assert audit["measured_bridge"] == {
        "exists": False,
        "complete_own_lineage_candidate_triples": 0,
        "scalar_distance_to_frontier": None,
        "reason": "low-rate and low-distortion evidence belong to different objects; no exact finite transition joins them",
    }


def test_target_is_recomputed_dynamically(tmp_path: Path) -> None:
    first = build_taskspace_frontier_closure_audit(_fixtures(tmp_path / "a", upstream_score=0.172))
    second = build_taskspace_frontier_closure_audit(_fixtures(tmp_path / "b", upstream_score=0.165))
    assert first["competitive_target"]["score"] == 0.172
    assert second["competitive_target"]["score"] == 0.165
    assert (
        second["anchors"]["low_distortion"]["conditional_frontier_max_archive_bytes"]
        < first["anchors"]["low_distortion"]["conditional_frontier_max_archive_bytes"]
    )


def test_cached_pointer_disagreement_fails_closed(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    pointer = json.loads(paths.pointer.read_text())
    pointer["effective_frontier"]["score"] = 0.19
    _write(paths.pointer, pointer)
    with pytest.raises(TaskspaceFrontierClosureError, match="disagrees"):
        build_taskspace_frontier_closure_audit(paths)


def test_duplicate_evidence_key_fails_closed(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    paths.g20.write_text('{"selected":{},"selected":{}}', encoding="utf-8")
    with pytest.raises(TaskspaceFrontierClosureError, match="duplicate JSON key"):
        build_taskspace_frontier_closure_audit(paths)
