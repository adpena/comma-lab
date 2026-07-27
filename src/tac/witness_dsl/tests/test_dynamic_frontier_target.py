from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tac import score_geometry
from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION
from tac.witness_dsl import dynamic_frontier_target as target_module
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    load_dynamic_frontier_target,
    score_sublevel_against_dynamic_frontier,
    score_transition_against_dynamic_frontier,
    verify_dynamic_frontier_target_snapshot,
)


def _local_anchor(score: float, axis: str) -> dict[str, object]:
    return {
        "score": score,
        "axis": axis,
        "archive_sha256": ("a" if axis == "contest_cpu" else "b") * 64,
        "lane_id": f"synthetic_{axis}",
        "hardware_substrate": f"synthetic_{axis}_hardware",
        "measured_at_utc": "2029-12-01T00:00:00+00:00",
        "evidence_grade": f"[synthetic-{axis}]",
        "source_path": "synthetic-only.json",
        "extra": {},
    }


def _pointer_payload(
    *,
    upstream_score: float | None,
    cpu_score: float | None,
    cuda_score: float | None = None,
    refreshed_at: str | None = None,
    source_snapshot_at: str | None = None,
    cached_score: float | None = None,
) -> dict[str, object]:
    now = refreshed_at or datetime.now(UTC).isoformat()
    source_now = source_snapshot_at or now
    if upstream_score is None:
        upstream = None
    else:
        entry = {
            "score": upstream_score,
            "rank": 1,
            "name": "synthetic-public-row",
            "pr_number": 9001,
            "pr_url": "https://invalid.example/synthetic",
        }
        upstream = {"best_entry": dict(entry), "entries": [dict(entry)]}
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "our_local_frontier_contest_cpu": (_local_anchor(cpu_score, "contest_cpu") if cpu_score is not None else None),
        "our_local_frontier_contest_cuda": (
            _local_anchor(cuda_score, "contest_cuda") if cuda_score is not None else None
        ),
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": upstream,
        "upstream_leaderboard_snapshot_at_utc": source_now if upstream is not None else None,
        "last_refreshed_utc": now,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "synthetic-fixture-do-not-run",
        "refresh_provenance": {"fixture": True},
        "effective_frontier": (
            {
                "score": cached_score,
                "source": "forged_cached_display_only",
                "axis": "forged",
            }
            if cached_score is not None
            else None
        ),
    }


def _write_pointer(repo: Path, payload: dict[str, object]) -> Path:
    path = repo / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_upstream_constituent_beats_weaker_local_exact_row(tmp_path: Path) -> None:
    path = _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.367, cpu_score=0.413, cuda_score=0.449),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)

    assert snapshot.target_score == 0.367
    assert snapshot.selected_source == "upstream_official_leaderboard"
    assert snapshot.selected_axis == "official_leaderboard"
    assert snapshot.selected_score_precision == "official_display"
    assert snapshot.selected_custody == "external target only; no local archive authority implied"
    assert snapshot.pointer_bytes == path.stat().st_size
    assert snapshot.pointer_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert snapshot.pointer_inode == path.stat().st_ino
    assert snapshot.research_only is True
    assert snapshot.derived_planning_only is True
    assert snapshot.score_claim is False
    assert snapshot.evaluation_claim is False
    assert snapshot.promotion_eligible is False


def test_local_exact_constituent_beats_weaker_upstream_row(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.391, cpu_score=0.329, cuda_score=0.407),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)

    assert snapshot.target_score == 0.329
    assert snapshot.selected_source == "our_local_frontier_contest_cpu"
    assert snapshot.selected_axis == "contest_cpu"
    assert snapshot.selected_archive_sha256 == "a" * 64
    assert snapshot.selected_lane_id == "synthetic_contest_cpu"
    assert snapshot.selected_hardware_substrate == "synthetic_contest_cpu_hardware"
    assert snapshot.selected_score_precision == "unspecified"


def test_cached_effective_frontier_tampering_cannot_override_constituents(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.353, cpu_score=0.379, cached_score=0.011),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)

    assert snapshot.target_score == 0.353
    assert snapshot.selected_source == "upstream_official_leaderboard"


def test_missing_pointer_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DynamicFrontierTargetError, match="missing"):
        load_dynamic_frontier_target(repo_root=tmp_path)


def test_corrupt_pointer_is_refused(tmp_path: Path) -> None:
    path = tmp_path / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(DynamicFrontierTargetError, match="corrupt"):
        load_dynamic_frontier_target(repo_root=tmp_path)


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, 0.0, -0.25])
def test_nonfinite_or_nonpositive_only_target_is_refused(tmp_path: Path, bad_score: float) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=bad_score, cpu_score=None),
    )
    with pytest.raises(DynamicFrontierTargetError, match="finite positive competitive target"):
        load_dynamic_frontier_target(repo_root=tmp_path)


def test_stale_pointer_is_refused(tmp_path: Path) -> None:
    stale = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=None, cpu_score=0.337, refreshed_at=stale),
    )
    with pytest.raises(DynamicFrontierTargetError, match="24-hour"):
        load_dynamic_frontier_target(repo_root=tmp_path)


def test_fresh_wrapper_cannot_hide_stale_selected_official_snapshot(tmp_path: Path) -> None:
    stale = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    _write_pointer(
        tmp_path,
        _pointer_payload(
            upstream_score=0.347,
            cpu_score=0.419,
            source_snapshot_at=stale,
        ),
    )
    with pytest.raises(DynamicFrontierTargetError, match="upstream_leaderboard_snapshot_at_utc"):
        load_dynamic_frontier_target(repo_root=tmp_path)


def test_pointer_mutation_between_snapshot_and_audit_is_refused(tmp_path: Path) -> None:
    path = _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.361, cpu_score=0.421),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)
    path.write_text(
        json.dumps(_pointer_payload(upstream_score=0.359, cpu_score=0.421)),
        encoding="utf-8",
    )

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        score_sublevel_against_dynamic_frontier(
            snapshot,
            d_seg=0.0011,
            d_pose=0.0007,
            archive_bytes=731,
        )


def test_forged_snapshot_target_cannot_bypass_pointer_reopen(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.363, cpu_score=0.419),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)
    forged = replace(snapshot, target_score=0.001)

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        score_sublevel_against_dynamic_frontier(
            forged,
            d_seg=0.0,
            d_pose=0.0,
            archive_bytes=0,
        )


def test_sublevel_helper_exactly_delegates_same_object_three_axis_geometry(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.433, cpu_score=0.461),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)
    wrapped = score_sublevel_against_dynamic_frontier(
        snapshot,
        d_seg=0.0017,
        d_pose=0.0023,
        archive_bytes=12_347,
    )
    direct = score_geometry.score_sublevel_audit(
        target_score=snapshot.target_score,
        d_seg=0.0017,
        d_pose=0.0023,
        archive_bytes=12_347,
    )

    assert wrapped == direct
    assert wrapped.target_score == snapshot.target_score


def test_transition_helper_exactly_delegates_all_before_and_after_axes(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.427, cpu_score=0.463),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)
    kwargs = {
        "before_d_seg": 0.0021,
        "before_d_pose": 0.0031,
        "before_archive_bytes": 14_111,
        "after_d_seg": 0.0018,
        "after_d_pose": 0.0034,
        "after_archive_bytes": 14_509,
    }
    wrapped = score_transition_against_dynamic_frontier(snapshot, **kwargs)
    direct = score_geometry.score_transition_audit(
        target_score=snapshot.target_score,
        **kwargs,
    )

    assert wrapped == direct
    assert wrapped.changed_axes == ("seg", "pose", "bytes")


def test_target_equality_is_not_inside_strict_sublevel(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.5, cpu_score=0.75),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)
    audit = score_sublevel_against_dynamic_frontier(
        snapshot,
        d_seg=snapshot.target_score / 100.0,
        d_pose=0.0,
        archive_bytes=0,
    )

    assert audit.score == snapshot.target_score
    assert audit.inside_strict_sublevel is False
    assert audit.on_target_boundary is True


def test_audit_api_refuses_numeric_target_substitution(tmp_path: Path) -> None:
    _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=None, cpu_score=0.317),
    )
    with pytest.raises(TypeError, match="DynamicFrontierTargetSnapshot"):
        score_sublevel_against_dynamic_frontier(  # type: ignore[arg-type]
            0.123,
            d_seg=0.0,
            d_pose=0.0,
            archive_bytes=0,
        )


def test_public_non_score_snapshot_guard_reopens_exact_pointer(tmp_path: Path) -> None:
    pointer = _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=0.301, cpu_score=0.317),
    )
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)

    assert verify_dynamic_frontier_target_snapshot(snapshot) is snapshot

    pointer.write_text(
        json.dumps(_pointer_payload(upstream_score=0.299, cpu_score=0.317)),
        encoding="utf-8",
    )
    with pytest.raises(DynamicFrontierTargetError, match="identity or derived target changed"):
        verify_dynamic_frontier_target_snapshot(snapshot)


def test_pointer_symlink_is_refused(tmp_path: Path) -> None:
    real_repo = tmp_path / "real"
    link_repo = tmp_path / "linked"
    real = _write_pointer(
        real_repo,
        _pointer_payload(upstream_score=None, cpu_score=0.309),
    )
    link = link_repo / ".omx/state/canonical_frontier_pointer.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)

    with pytest.raises(DynamicFrontierTargetError, match="without following links"):
        load_dynamic_frontier_target(repo_root=link_repo)


def test_pointer_directory_is_refused_as_non_regular(tmp_path: Path) -> None:
    pointer_path = tmp_path / ".omx/state/canonical_frontier_pointer.json"
    pointer_path.mkdir(parents=True)

    with pytest.raises(DynamicFrontierTargetError, match="regular file"):
        load_dynamic_frontier_target(repo_root=tmp_path)


def test_pointer_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    pointer_path = tmp_path / ".omx/state/canonical_frontier_pointer.json"
    pointer_path.parent.mkdir(parents=True)
    os.mkfifo(pointer_path)

    with pytest.raises(DynamicFrontierTargetError, match="regular file"):
        load_dynamic_frontier_target(repo_root=tmp_path)


def test_pointer_path_is_lexical_absolute_and_never_resolved_through_parent_symlink(
    tmp_path: Path,
) -> None:
    real_repo = tmp_path / "real-repo"
    _write_pointer(
        real_repo,
        _pointer_payload(upstream_score=None, cpu_score=0.301),
    )
    alias_repo = tmp_path / "repo-alias"
    alias_repo.symlink_to(real_repo, target_is_directory=True)

    snapshot = load_dynamic_frontier_target(repo_root=alias_repo)
    lexical = os.path.abspath(os.fspath(alias_repo / ".omx/state/canonical_frontier_pointer.json"))

    assert snapshot.pointer_path == lexical
    assert snapshot.pointer_path != os.path.abspath(os.fspath(real_repo / ".omx/state/canonical_frontier_pointer.json"))


def test_atomic_replacement_during_parse_is_refused_before_snapshot_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path = _write_pointer(
        tmp_path,
        _pointer_payload(upstream_score=None, cpu_score=0.303),
    )
    original_parser = target_module._pointer_from_bytes

    def replace_after_descriptor_read(payload: bytes):
        parsed = original_parser(payload)
        replacement = pointer_path.with_suffix(".replacement")
        replacement.write_text(
            json.dumps(_pointer_payload(upstream_score=None, cpu_score=0.299)),
            encoding="utf-8",
        )
        os.replace(replacement, pointer_path)
        return parsed

    monkeypatch.setattr(target_module, "_pointer_from_bytes", replace_after_descriptor_read)

    with pytest.raises(DynamicFrontierTargetError, match="path identity changed"):
        load_dynamic_frontier_target(repo_root=tmp_path)
