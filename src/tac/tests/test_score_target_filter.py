# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tac import score_target_filter as score_filter
from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION
from tac.score_target_filter import decide_score_target_routing, parse_predicted_band
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    load_dynamic_frontier_target,
)


def _write_pointer(repo: Path, *, score: float, cached_score: float = 0.001) -> tuple[Path, str]:
    now = datetime.now(UTC).isoformat()
    entry = {
        "score": score,
        "rank": 1,
        "name": "synthetic-public-row",
        "pr_number": 9001,
        "pr_url": "https://invalid.example/synthetic",
    }
    payload = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "our_local_frontier_contest_cpu": None,
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {
            "best_entry": dict(entry),
            "entries": [dict(entry)],
        },
        "upstream_leaderboard_snapshot_at_utc": now,
        "last_refreshed_utc": now,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "synthetic-fixture-do-not-run",
        "refresh_provenance": {"fixture": True},
        "effective_frontier": {
            "score": cached_score,
            "source": "forged-cache-must-not-steer",
            "axis": "forged",
        },
    }
    path = repo / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, now


def _canonical_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    score: float = 0.25,
):
    _, now = _write_pointer(tmp_path, score=score)
    monkeypatch.setattr(score_filter, "_DYNAMIC_TARGET_REPO_ROOT", tmp_path)
    return load_dynamic_frontier_target(repo_root=tmp_path, now_utc_iso=now), now


def test_parse_predicted_band_accepts_sequence_and_string() -> None:
    assert parse_predicted_band((0.205, 0.208)) == (0.205, 0.208)
    assert parse_predicted_band("[0.208, 0.205]") == (0.205, 0.208)


def test_parse_predicted_band_rejects_malformed_string() -> None:
    with pytest.raises(ValueError):
        parse_predicted_band("0.205")


def test_decision_uses_recomputed_dynamic_target_not_forged_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, now = _canonical_snapshot(tmp_path, monkeypatch, score=0.25)
    above = decide_score_target_routing(
        (0.26, 0.27), target_snapshot=snapshot, now_utc_iso=now
    )
    plausible = decide_score_target_routing(
        (0.24, 0.26), target_snapshot=snapshot, now_utc_iso=now
    )

    assert snapshot.target_score == 0.25
    assert above.target_score == 0.25
    assert above.target_score != 0.001
    assert above.active is False
    assert above.status == "above_target"
    assert plausible.active is True
    assert plausible.status == "target_plausible"
    assert plausible.target_pointer_sha256 == snapshot.pointer_sha256


def test_decision_fails_closed_on_unknown_band_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, now = _canonical_snapshot(tmp_path, monkeypatch)
    decision = decide_score_target_routing(
        None,
        target_snapshot=snapshot,
        keep_unknown=False,
        now_utc_iso=now,
    )
    assert decision.active is False
    assert decision.status == "unknown_band"


def test_numeric_target_override_is_not_an_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _canonical_snapshot(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        vars(score_filter)["decide_score_target_routing"]((0.0, 1.0), target_score=0.001)


def test_forged_stale_and_path_swapped_snapshots_cannot_steer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_repo = tmp_path / "canonical"
    snapshot, now = _canonical_snapshot(canonical_repo, monkeypatch)

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        decide_score_target_routing(
            (0.0, 1.0),
            target_snapshot=replace(snapshot, target_score=0.001),
            now_utc_iso=now,
        )

    stale_time = (datetime.fromisoformat(now) - timedelta(hours=25)).isoformat()
    with pytest.raises(DynamicFrontierTargetError, match="24-hour"):
        decide_score_target_routing(
            (0.0, 1.0),
            target_snapshot=replace(snapshot, last_refreshed_utc=stale_time),
            now_utc_iso=now,
        )

    swapped_repo = tmp_path / "swapped"
    _, swapped_now = _write_pointer(swapped_repo, score=0.24)
    swapped = load_dynamic_frontier_target(repo_root=swapped_repo, now_utc_iso=swapped_now)
    with pytest.raises(DynamicFrontierTargetError, match="noncanonical pointer path"):
        decide_score_target_routing(
            (0.0, 1.0), target_snapshot=swapped, now_utc_iso=now
        )
